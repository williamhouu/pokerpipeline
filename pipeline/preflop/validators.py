"""Layer 7 audit validators -- preflop edition.

Postflop's Layer 7 lives in ``pipeline.validators`` and binds to
``DecisionData``. Preflop spots don't carry DecisionData -- they carry
:class:`pipeline.preflop.fact_extractor.PreflopFacts` -- so this module
hosts the preflop-shape validators.

Per the brief, Layer 7 is the difference between unusable and
shippable output: roughly 30-50% of first-pass LLM generations have
some defect; one corrective retry against a targeted validator
message brings that under 15%, which is the QA budget the team wants.
This module provides the validator API + the v1 stack of hard
validators.

v1 hard validators (run in series; first failure short-circuits and
becomes the retry message):

  1. :func:`validate_option_set` -- every option's primary action verb
     appears in the spot's actual Pio strategy. Catches inventions
     like "Raise" appearing on a call/fold spot.
  2. :func:`validate_no_standalone_sometimes` -- bans the ambiguous
     "Sometimes X" / "Rarely X" leading-word labels per Ryan's
     Apr 2026 V6 review.
  3. :func:`validate_composite_label_frequencies` -- when an option
     uses the "Mostly X, sometimes Y" composite shape, X must
     actually be the dominant action and both X + Y must each be
     at least 5% of the canonical strategy.
  4. :func:`validate_banned_phrases` -- em dashes, semicolons, and
     the team's literal-phrase blocklist (e.g. "leverage the dead
     money", "dynamic spot") are banned. The LLM is told these in
     the system prompt; this validator hard-enforces.
  5. :func:`validate_card_suit_consistency` -- specific cards named in
     prose must be hero's own two (June 2026 round-2 audit: "your K❤️"
     while holding K♠5♠).
  6. :func:`validate_blocker_claims` -- "your hand blocks X" only when
     X is in the blockers fact; no blocker talk at all when the fact
     is empty (round-2 audit: 4/20 rows claimed impossible blocks).
  7. :func:`validate_terminology` -- cold-call/squeeze/open-fold/N-bet
     claims the action history disproves (round-2 audit).

(A ``validate_no_postflop_talk`` keyword check was removed -- it
hard-banned terms like "runout"/"draws" and false-positived on
legitimate preflop playability talk; the system prompt already steers
the model to preflop-only reasoning.)

Strategy / failure-pattern validators are stubbed in
:func:`run_preflop_audit_validators` but currently always pass --
will be tuned against the first batches of real review feedback.

Soft (warning-only) validators (June 2026): run via
:func:`run_preflop_soft_validators` after generation succeeds; they
never fail a row. The batch driver marks warned rows
``validation_status='flagged'`` and stores the warnings in the meta
sidecar. They are tuned for PRECISION over recall -- a flag must mean
"a human should actually look", because false positives are pure review
noise. Four soft checks:

  1. :func:`soft_validate_position_words` -- hero-bound position words vs
     the hero_position fact.
  2. :func:`soft_validate_number_vs_data` -- a cited equity / pot-odds
     percentage that contradicts the data block (heads-up spots only, so
     field-vs-per-opponent numbers don't false-fire).
  3. :func:`soft_validate_verdict_vs_answer` -- the opening verdict's
     action (fold / passive / aggressive) vs the spot's dominant action.
  4. :func:`soft_validate_named_combo_in_range` -- a "villain has X" claim
     naming a hand class absent from villain's actual range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.explanation_generator import (
    BANNED_LITERAL_PHRASES,
    GeneratedExplanation,
)
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.grammars.types import PreflopActionType
from pipeline.preflop.options import canonicalize_strategy

# Minimum conditional frequency for an option to count as a valid
# "primary" or "secondary" verb in a composite label. Tuned to drop
# tiny mix-ins (e.g. 0.5% raises in BvB call/fold spots) without
# losing legitimate 5-10% mixes.
_COMPOSITE_LABEL_MIN_FREQ = 0.05

# Frequency prefix tokens the LLM is allowed to use as the LEADING
# word of an option. Standalone "Sometimes" / "Rarely" are explicitly
# excluded (validator 2) -- they're only valid as the SECONDARY verb
# of a composite label like "Mostly call, sometimes raise".
_FREQUENCY_PREFIXES_ALL = frozenset({
    "always", "mostly", "sometimes", "rarely",
})
_FREQUENCY_PREFIXES_BANNED_AS_LEADING = frozenset({"sometimes", "rarely"})

# The set of action verbs the preflop strategy can contain. Mirrors
# the canonical labels produced by canonicalize_strategy().
_KNOWN_PREFLOP_VERBS = frozenset({
    "fold", "call", "raise", "3-bet", "4-bet", "5-bet", "all-in", "check",
})

# Raise-family verbs that share a single frequency entry in the
# strategy lookup. canonicalize_strategy collapses multi-size raises
# at the same node into one canonical-label entry, and the validator
# stores them all under "raise". To keep storage and lookup in sync,
# both directions normalise raise-family verbs through this helper.
_RAISE_FAMILY = frozenset({"raise", "3-bet", "4-bet", "5-bet"})


def _normalize_verb(verb: str) -> str:
    """Map raise-family verbs to a single 'raise' bucket for lookup.

    Pio's canonical strategy can label hero's raise as `"3-bet"`,
    `"4-bet"`, or `"5-bet"` depending on the action history's raise
    level (set by ``pipeline.preflop.options.canonicalize_action_label``).
    The validators store all of these under a single ``"raise"`` key
    so multi-size raise spots merge cleanly; this helper applies the
    same normalisation on the lookup side so a LLM-side option verb
    like ``"4-bet"`` finds the strategy frequency stored under
    ``"raise"``.

    Without this, a poker-correct LLM choosing "Mostly 4-bet" at a
    BB-vs-3bet spot would be rejected because the lookup returned 0%
    -- the actual 4-bet frequency is stored under "raise", not
    "4-bet". See git history for the bug-fix commit message.
    """
    return "raise" if verb in _RAISE_FAMILY else verb

# Em dashes (Unicode and the LLM's tendency to use them) + semicolons
# are banned by the team's voice rules. The LLM is told this in the
# system prompt; this validator enforces.
_BANNED_PUNCTUATION_PATTERNS = tuple(re.compile(p) for p in [
    r"—",      # em dash (U+2014)
    r";",      # semicolons (the voice rules ban these too)
    r"–",      # en dash (U+2013) -- often a stand-in for em dash
])


@dataclass(frozen=True)
class PreflopValidationResult:
    """Outcome of one validator. Composable.

    Mirrors :class:`pipeline.validators.ValidationResult` so the
    retry-loop wiring in
    :mod:`pipeline.preflop.explanation_generator` reads the same on
    both paths.
    """

    is_valid: bool
    error_message: str = ""

    @classmethod
    def ok(cls) -> PreflopValidationResult:
        return cls(is_valid=True)

    @classmethod
    def fail(cls, message: str) -> PreflopValidationResult:
        return cls(is_valid=False, error_message=message)


# --- helpers ----------------------------------------------------------------
def _option_strings(generated: GeneratedExplanation) -> list[tuple[str, str]]:
    """The non-empty options as (slot_name, text) pairs."""
    slots = (
        ("option_1", generated.option_1),
        ("option_2", generated.option_2),
        ("option_3", generated.option_3),
        ("option_4", generated.option_4),
    )
    return [(name, text) for name, text in slots if text]


def _leading_word(text: str) -> str:
    """Lowercase first token, stripped of punctuation. '' on empty."""
    words = text.lower().split()
    if not words:
        return ""
    return words[0].strip(".,;:!?\"'()[]")


def _primary_verb(option_text: str) -> str | None:
    """The action verb of an option, ignoring any leading frequency word.

    "Mostly fold"            -> "fold"
    "Mostly call, sometimes raise" -> "call"  (primary action, NOT secondary)
    "Raise 308%"             -> "raise"
    "All-in"                 -> "all-in"
    "Fold"                   -> "fold"
    None when nothing recognisable (e.g. pure sizing label).
    """
    if not option_text:
        return None
    words = option_text.lower().split()
    if not words:
        return None
    # Skip a leading frequency prefix.
    if words[0].strip(".,") in _FREQUENCY_PREFIXES_ALL:
        words = words[1:]
    for word in words:
        cleaned = word.strip(".,;:!?\"'()[]")
        if cleaned in _KNOWN_PREFLOP_VERBS:
            return cleaned
    return None


def _composite_secondary_verb(option_text: str) -> str | None:
    """For "Mostly X, sometimes Y" -> 'y'. None if not a composite.

    Recognises the pattern: ``<primary> ... , sometimes <secondary>``.
    Same for "rarely" as the secondary qualifier.
    """
    if "," not in option_text:
        return None
    after_comma = option_text.split(",", 1)[1].lower().strip()
    after_words = after_comma.split()
    if not after_words:
        return None
    first = after_words[0].strip(".,")
    if first not in ("sometimes", "rarely"):
        return None
    remaining = after_words[1:]
    for word in remaining:
        cleaned = word.strip(".,;:!?\"'()[]")
        if cleaned in _KNOWN_PREFLOP_VERBS:
            return cleaned
    return None


# --- the validators ---------------------------------------------------------
def validate_option_set(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> PreflopValidationResult:
    """Every option's primary verb appears in the canonical strategy.

    Catches the case where the LLM invents an option Pio never offers
    (e.g. "Raise" on a pure call/fold spot). Skips options whose verb
    is not recognisable (e.g. raw sizing labels) -- those would be
    structurally invalid anyway and caught upstream.
    """
    strategy = canonicalize_strategy(facts)
    if not strategy:
        return PreflopValidationResult.ok()  # no signal to check against

    # Map canonical-strategy labels to their primary verb tokens so
    # we can match against the option-side verbs.
    pio_verbs: set[str] = set()
    for label in strategy:
        first = label.lower().split()[0] if label else ""
        first = first.strip(".,")
        if first in _KNOWN_PREFLOP_VERBS:
            pio_verbs.add(first)
        elif first == "3-bet" or first == "4-bet" or first == "5-bet":
            # Composite verbs -- map to "raise" since that's how
            # canonicalize_strategy renders them in the option-side
            # too. Defensive in case label normalisation differs.
            pio_verbs.add(first)
            pio_verbs.add("raise")

    invalid: list[str] = []
    for slot, text in _option_strings(generated):
        verb = _primary_verb(text)
        if verb is None:
            continue  # sizing label or otherwise non-verb; skip
        if verb not in pio_verbs:
            invalid.append(f"{slot}={text!r} (verb={verb!r})")

    if invalid:
        pio_list = sorted(pio_verbs)
        return PreflopValidationResult.fail(
            "one or more options reference an action Pio doesn't offer "
            f"at this node. Pio's strategy uses {pio_list}; offending "
            "options: " + "; ".join(invalid)
        )
    return PreflopValidationResult.ok()


def validate_no_standalone_sometimes(
    generated: GeneratedExplanation,
    facts: PreflopFacts,  # noqa: ARG001 -- signature uniformity
) -> PreflopValidationResult:
    """No option may start with "Sometimes" or "Rarely".

    Per Ryan's Apr 2026 V6 review (Fix 2b (a)): standalone Sometimes/
    Rarely options are ambiguous to players ("Sometimes call" and
    "Sometimes fold" are not mutually exclusive readings of the
    strategy). Composite "Mostly X, sometimes Y" labels are fine
    (the LEADING word is "Mostly", not "Sometimes") -- this validator
    only fires on bare leading Sometimes/Rarely.
    """
    offending: list[str] = []
    for slot, text in _option_strings(generated):
        leading = _leading_word(text)
        if leading in _FREQUENCY_PREFIXES_BANNED_AS_LEADING:
            offending.append(f"{slot}={text!r}")

    if offending:
        return PreflopValidationResult.fail(
            "standalone 'Sometimes X' / 'Rarely X' option labels are "
            "banned per Apr 2026 review. Use 'Mostly X' for 2-action "
            "spots or composite 'Mostly X, sometimes Y' labels for 3+ "
            "action spots. Offending options: " + "; ".join(offending)
        )
    return PreflopValidationResult.ok()


def validate_composite_label_frequencies(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> PreflopValidationResult:
    """Composite "Mostly X, sometimes Y" labels must reflect Pio frequencies.

    Both X (primary) and Y (secondary) must appear in the canonical
    strategy with frequency >= 5%, and X must be the dominant action
    (highest frequency among the recognised verbs). Catches cases
    where the LLM uses "Mostly call, sometimes raise" on a spot where
    raise is actually only 1% (cosmetic noise) or where the dominant
    action is actually fold.
    """
    strategy = canonicalize_strategy(facts)
    if not strategy:
        return PreflopValidationResult.ok()

    # Build a verb->freq lookup keyed by first-word verb. Raise-family
    # verbs (3-bet / 4-bet / 5-bet) collapse into a single "raise"
    # entry via _normalize_verb -- the SAME normalisation is applied
    # when looking up the option-side verb below, so a poker-correct
    # "Mostly 4-bet" at a BB-vs-3bet spot resolves to the raise
    # frequency stored under "raise".
    verb_to_freq: dict[str, float] = {}
    for label, freq in strategy.items():
        first = label.lower().split()[0].strip(".,") if label else ""
        if first not in _KNOWN_PREFLOP_VERBS:
            continue
        key = _normalize_verb(first)
        verb_to_freq[key] = verb_to_freq.get(key, 0.0) + freq

    if not verb_to_freq:
        return PreflopValidationResult.ok()

    failures: list[str] = []
    for slot, text in _option_strings(generated):
        primary = _primary_verb(text)
        secondary = _composite_secondary_verb(text)
        if secondary is None:
            continue  # not a composite label; the option_set validator covers basic cases

        if primary is None:
            failures.append(
                f"{slot}={text!r}: composite label has no recognisable primary verb"
            )
            continue

        # Apply the same raise-family normalisation that the storage
        # side used so "4-bet" finds the raise frequency.
        p_freq = verb_to_freq.get(_normalize_verb(primary), 0.0)
        s_freq = verb_to_freq.get(_normalize_verb(secondary), 0.0)

        if p_freq < _COMPOSITE_LABEL_MIN_FREQ:
            failures.append(
                f"{slot}={text!r}: primary verb {primary!r} has Pio freq "
                f"{p_freq:.0%} (< {_COMPOSITE_LABEL_MIN_FREQ:.0%}); the "
                f"strategy is {dict(sorted(verb_to_freq.items(), key=lambda kv: -kv[1]))}"
            )
            continue

        if s_freq < _COMPOSITE_LABEL_MIN_FREQ:
            failures.append(
                f"{slot}={text!r}: secondary verb {secondary!r} has Pio freq "
                f"{s_freq:.0%} (< {_COMPOSITE_LABEL_MIN_FREQ:.0%}); don't "
                "promote a tiny mix-in to a 'sometimes' label"
            )
            continue

        if p_freq < s_freq:
            failures.append(
                f"{slot}={text!r}: primary verb {primary!r} ({p_freq:.0%}) "
                f"is less frequent than secondary {secondary!r} ({s_freq:.0%}); "
                "swap them so the label reads 'Mostly <higher-freq>, "
                "sometimes <lower-freq>'"
            )

    if failures:
        return PreflopValidationResult.fail(
            "composite label frequencies don't match Pio: "
            + "; ".join(failures)
        )
    return PreflopValidationResult.ok()


def validate_banned_phrases(
    generated: GeneratedExplanation,
    facts: PreflopFacts,  # noqa: ARG001
) -> PreflopValidationResult:
    """Bans em dashes, semicolons, and the team's literal-phrase blocklist.

    The LLM is told this in the system prompt's banned-phrases section;
    this validator hard-enforces. ``BANNED_LITERAL_PHRASES`` is the
    shared list from :mod:`pipeline.explanation_generator`.
    """
    text = generated.answer_explanation or ""
    if not text:
        return PreflopValidationResult.ok()

    hits: list[str] = []

    # Banned punctuation (em dash, semicolon, en dash).
    for pattern in _BANNED_PUNCTUATION_PATTERNS:
        if pattern.search(text):
            hits.append(f"banned punctuation {pattern.pattern!r}")

    # Banned literal phrases (case-insensitive contains).
    lower = text.lower()
    for phrase in BANNED_LITERAL_PHRASES:
        if phrase.lower() in lower:
            hits.append(f"banned phrase {phrase!r}")

    if hits:
        return PreflopValidationResult.fail(
            "explanation uses banned punctuation or phrases: "
            + "; ".join(hits)
            + ". Rewrite using the team's voice (no em dashes, no "
            "semicolons, no template/corporate phrasing)."
        )
    return PreflopValidationResult.ok()


# Concepts that are impossible once the money is all-in preflop -- the hand
# goes straight to showdown with no further betting. Deliberately NARROW
# (not a general "no postflop words" ban -- that was removed for
# false-positiving): mentioning that flush/straight outs add to your SHOWDOWN
# equity on the runout is fine; "implied odds" / "postflop play" is not.
_BANNED_ON_ALLIN = (
    "implied odds",
    "implied-odds",
    "postflop",
    "post-flop",
    "later streets",
    "future streets",
)

# A negation right before a banned phrase flips it into CORRECT all-in framing
# -- "there's NO postflop play", "there are NO future streets", "NO implied
# odds" -- which is exactly the pot-odds-vs-equity frame this validator wants,
# not a violation. We scan a short window before each hit for a negation cue,
# stopping at sentence punctuation so a negation in a prior clause can't
# license a real violation. Erring toward ALLOW is deliberate: a false
# rejection burns a retry and routes a good question to human review (the bug
# this guard fixes), whereas a genuine "you have implied odds / play postflop"
# framing carries no negation and still fails. The `n't` arm (no leading word
# boundary) catches contractions like "isn't"/"aren't"/"won't".
_ALLIN_NEGATION = re.compile(
    r"(\b(no|not|never|without|zero|nor|cannot)\b|n't)[^.;:!?]{0,24}$",
    re.IGNORECASE,
)


def _allin_phrase_is_negated(text: str, phrase_start: int) -> bool:
    """True if the banned phrase at ``phrase_start`` sits in a negating
    context -- correct showdown framing ("no postflop play"), not a violation."""
    return bool(_ALLIN_NEGATION.search(text[:phrase_start]))


def validate_no_postflop_on_allin(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> PreflopValidationResult:
    """On an all-in spot, ban implied-odds / postflop framing.

    When the money is all-in preflop there are no future streets: the hand
    is a pure pot-odds-vs-equity decision. The screenshot bug was an all-in
    call whose explanation talked about implied odds, chasing draws to stack
    callers postflop -- impossible. Runs ONLY on all-in spots (so it can't
    false-positive on normal preflop playability talk), and the message
    steers the model back to a pot-odds frame.
    """
    node = facts.spot.node
    is_allin = (
        facts.archetype == "call_allin"
        or any(
            a.action_type is PreflopActionType.ALL_IN
            for a in node.history_before
        )
        or "allin" in (facts.spot.dominant_action or "").lower().replace(" ", "")
    )
    if not is_allin:
        return PreflopValidationResult.ok()

    text = (generated.answer_explanation or "").lower()
    # Flag a phrase only when it appears in a NON-negated spot. "There are no
    # future streets" / "no postflop play to save you" are the correct frame,
    # not the banned one, so they must not trip this.
    hits: list[str] = []
    for phrase in _BANNED_ON_ALLIN:
        start = text.find(phrase)
        while start != -1:
            if not _allin_phrase_is_negated(text, start):
                hits.append(phrase)
                break  # one un-negated use is enough to flag this phrase
            start = text.find(phrase, start + 1)
    if hits:
        return PreflopValidationResult.fail(
            "this is an ALL-IN spot -- the hand is decided at showdown with no "
            "further betting, so framing it around "
            + ", ".join(repr(h) for h in hits)
            + " is wrong. Reframe purely around pot odds vs raw equity: the "
            "price you're getting and whether your equity against the all-in "
            "range clears it. (It's fine to note that flush/straight outs add "
            "to your SHOWDOWN equity on the runout -- just not implied odds or "
            "postflop play.)"
        )
    return PreflopValidationResult.ok()


# --- June 2026 round-2 audit validators --------------------------------------
# The round-2 prose audit (LATEST AUDIT READY_20260612_122054) found three
# mechanical failure modes after the multiway fixes killed the round-1 ones:
# blocker claims hero's cards can't make (4/20 rows said "blocks AA" with no
# ace in hand), terminology the action history disproves (the opener
# "cold-calling", a "squeeze" with no caller), and a wrong suit emoji on
# hero's own card. All three are verifiable from PreflopFacts alone, so they
# run as HARD validators: a failure is a real defect, never a judgment call.
# Each check is deliberately conservative (negation guards, both-named
# escapes) -- a hard validator must never cry wolf, because its failure
# message becomes the corrective retry prompt.

# "block/blocks/blocker(s)/blocking" but NOT "unblock*" -- un-block claims
# assert the ABSENCE of a blocker and are checked by no rule here.
_BLOCK_WORD = re.compile(r"\b(?<!un)(?<!Un)[Bb]lock\w*")
# A 169-grid hand-class token: two ranks + optional s/o suffix. (?!%) keeps
# bare percentages like "33%" from reading as pocket threes.
_HAND_CLASS_TOKEN = re.compile(r"\b([AKQJT2-9]{2})(s|o)?\b(?!%)")
# Sentences whose blocker claims we skip: negated or hedged claims ("you
# don't block", "blocks almost nothing", "blocks zero combos of AA") can be
# TRUE precisely because a hand is absent from the blockers fact, so only
# positive claims are policed. "zero" reads as a quantity but means absence --
# "blocks zero combos of X" is the same assertion as "blocks none of X".
_NEGATION_MARKERS = ("not ", "n't", " no ", " nothing", " never", " zero", "unblock")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Specific-card token: rank immediately followed by a suit emoji (voice
# rule 9's format). Both heart emojis appear in the wild (rule 9 itself
# uses U+2764, gold prose uses U+2665) -- normalise both to 'h'.
_CARD_EMOJI = re.compile(r"([AKQJT2-9])([♠♥♦♣❤])")
_SUIT_LETTER = {
    "♠": "s",  # ♠
    "♥": "h",  # ♥
    "❤": "h",  # ❤
    "♦": "d",  # ♦
    "♣": "c",  # ♣
}

_COLD_CALL = re.compile(r"\bcold[\s-]?call\w*", re.I)
# "you'd be cold-calling" / "you can cold-call": the cold-call verb bound to
# HERO. Requires you + (optional single auxiliary) + cold-call adjacency so
# an incidental "...behind you and can cold-call..." doesn't bind to hero.
_HERO_COLD_CALL = re.compile(
    r"\byou(?:'d|'re|'ll)?"
    r"(?:\s+(?:be|are|were|would|could|can|will|should|might|just))?"
    r"\s+cold[\s-]?call",
    re.I,
)
_SQUEEZE = re.compile(r"\bsqueez\w*", re.I)
_OPEN_FOLD = re.compile(r"\bopen[\s-]fold\w*", re.I)
_N_BET_TOKEN = re.compile(r"\b([3-9])-bet", re.I)
_CAPPED = re.compile(r"\bcapped\b", re.I)
# Longest-first so "UTG" can't match inside "UTG+1".
_POSITION_SHORT = re.compile(r"\b(UTG\+[12]|UTG|LJ|HJ|CO|BTN|SB|BB)\b")
_POSITION_LONG = {
    "under the gun": "UTG",
    "lojack": "LJ",
    "hijack": "HJ",
    "cutoff": "CO",
    "button": "BTN",
    "small blind": "SB",
    "big blind": "BB",
}


def _positions_named(sentence: str) -> set[str]:
    """Every seat the sentence names, short or long form, as canon tokens."""
    named = set(_POSITION_SHORT.findall(sentence))
    low = sentence.lower()
    for phrase, pos in _POSITION_LONG.items():
        if phrase in low:
            named.add(pos)
    return named


def _raisers(facts: PreflopFacts) -> set[str]:
    """Seats that have voluntarily raised (or jammed) before hero's decision.

    These players can never cold-call: they already have aggression money
    in. Limp/call-only players stay "cold" on purpose -- a caller of an
    open WAS cold-calling, and policing their later calls would cry wolf.
    """
    return {
        a.position
        for a in facts.spot.node.history_before
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    }


def _pending_cold_seats(facts: PreflopFacts) -> list[str] | None:
    """Seats that still act after hero AND could legitimately cold-call.

    None when the spot's pack isn't registered (bare unit-test fixtures) --
    callers treat None as "unknown, don't judge". Mirrors the graceful
    degradation of the Layer-6 solver-data builder.
    """
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        compute_action_pending,
    )
    from pipeline.preflop.pack import get_pack  # noqa: PLC0415

    node = facts.spot.node
    try:
        pack = get_pack(node.pack_id)
    except KeyError:
        return None
    _others, pending, _closes = compute_action_pending(
        node.history_before, node.actor, pack.table_size
    )
    raisers = _raisers(facts)
    return [p for p in pending if p not in raisers]


def _hand_blocked_in_fact(
    core: str, suffix: str, blockers: dict[str, int]
) -> bool:
    """Is the prose's hand token licensed by the blockers fact?

    Pairs match exactly; bare non-pair tokens like "AK" are satisfied by
    either the suited or offsuit entry (prose often drops the suffix).
    """
    if suffix:
        return f"{core}{suffix}" in blockers
    if core[0] == core[1]:
        return core in blockers
    return (
        core in blockers
        or f"{core}s" in blockers
        or f"{core}o" in blockers
    )


def validate_blocker_claims(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> PreflopValidationResult:
    """Every "your hand blocks X" claim must trace to the blockers fact.

    June 2026 round-2 audit: 4 of 20 rows asserted physically impossible
    blocks ("your queens block KK and AA combos" with no ace or king in
    hero's hand) -- the flagship reversed-blocker-logic disease. The
    blockers fact lists exactly which villain hand classes hero's cards
    remove combos from, so the check is mechanical:

      * a sentence containing a positive block-word may only name hand
        classes present in ``facts.blockers``;
      * when the blockers fact is EMPTY (open spots: no villain), any
        positive blocker sentence at all is a defect -- there is no fact
        to license blocker talk (round-1 finding #6).

    Negated/hedged sentences ("you block almost nothing", "don't block")
    and un-block claims are skipped: those assert absence, which is often
    true BECAUSE the hand is missing from the fact.
    """
    text = generated.answer_explanation or ""
    if not text:
        return PreflopValidationResult.ok()

    for sentence in _SENTENCE_SPLIT.split(text):
        if not _BLOCK_WORD.search(sentence):
            continue
        low = sentence.lower()
        if any(marker in low for marker in _NEGATION_MARKERS):
            continue
        if not facts.blockers:
            return PreflopValidationResult.fail(
                "the explanation makes a blocker claim but this spot has NO "
                "blockers fact (no villain range -- e.g. an open spot). "
                "Remove all blocker talk: nothing licenses it. Offending "
                f"sentence: {sentence.strip()!r}"
            )
        unlicensed = sorted({
            f"{core}{suffix or ''}"
            for core, suffix in _HAND_CLASS_TOKEN.findall(sentence)
            if not _hand_blocked_in_fact(core, suffix, facts.blockers)
        })
        if unlicensed:
            return PreflopValidationResult.fail(
                "the explanation claims your hand blocks "
                + ", ".join(unlicensed)
                + " but the blockers fact licenses only "
                + ", ".join(sorted(facts.blockers))
                + ". A hand may be named as blocked ONLY if it appears in "
                "the blockers fact (your cards physically cannot block the "
                f"others). Offending sentence: {sentence.strip()!r}"
            )
    return PreflopValidationResult.ok()


def validate_terminology(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> PreflopValidationResult:
    """Preflop terms the action history disproves (round-2 audit, June 2026).

    Four checks, all grounded in ``history_before``:

      * **cold-call by a raiser** -- a player who already raised can never
        cold-call (3 of 20 rows called the opener's hypothetical call a
        cold-call). Fails when every seat a cold-call sentence names has
        already raised, or -- for an unnamed "a cold-caller behind" -- when
        no pending seat could still cold-call. Sentences that also name a
        genuinely cold seat pass (no wolf-crying on mixed sentences).
      * **squeeze with no caller** -- describing the villain's recorded
        raise as a squeeze when nobody called before it. Prospective
        squeeze talk ("if you call, the blinds can squeeze") stays legal:
        hero's own call would create the open-plus-caller precondition.
      * **open-fold while facing action** -- open-folding means folding
        when the action folds TO you; facing a raise you simply fold.
      * **raise-ladder overflow** -- an N-bet the line cannot reach. Max
        nameable level = raises so far + hero's next raise + one re-raise
        over it. (Subject-bound miscounts inside that bound, like "CO
        folds to a 5-bet" where only CO could make the 5-bet, are left to
        the prose audit -- parsing who makes which raise is not mechanical.)
    """
    text = generated.answer_explanation or ""
    if not text:
        return PreflopValidationResult.ok()

    history = facts.spot.node.history_before
    hero = facts.spot.node.actor
    raisers = _raisers(facts)
    any_call_in_history = any(
        a.action_type is PreflopActionType.CALL for a in history
    )
    n_raises = sum(
        1
        for a in history
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )
    max_nameable_bet = n_raises + 3  # current top + hero's raise + one re-raise
    villain_pos = facts.villain_stats.position if facts.villain_stats else None

    for sentence in _SENTENCE_SPLIT.split(text):
        low = sentence.lower()

        if _COLD_CALL.search(sentence):
            named = _positions_named(sentence)
            if _HERO_COLD_CALL.search(sentence):
                named.add(hero)
            if named:
                if all(seat in raisers for seat in named):
                    return PreflopValidationResult.fail(
                        "the explanation says "
                        + ", ".join(sorted(named))
                        + " could cold-call, but a player who already "
                        "raised can NEVER cold-call (cold-calling means "
                        "calling with no voluntary money in yet). Say "
                        "'call' for the opener/raiser, or name a seat that "
                        f"is actually cold. Offending sentence: {sentence.strip()!r}"
                    )
            else:
                cold_pending = _pending_cold_seats(facts)
                if cold_pending is not None and not cold_pending:
                    return PreflopValidationResult.fail(
                        "the explanation mentions a cold-caller but no seat "
                        "still to act could cold-call here (every pending "
                        "player already raised). Use their recorded role "
                        f"instead. Offending sentence: {sentence.strip()!r}"
                    )

        if (
            _SQUEEZE.search(sentence)
            and not any_call_in_history
            and villain_pos is not None
            and villain_pos in _positions_named(sentence)
        ):
            return PreflopValidationResult.fail(
                f"the explanation calls {villain_pos}'s raise a squeeze, "
                "but nobody called before it -- a squeeze is a 3-bet over "
                "an open PLUS at least one caller. Call it a 3-bet. "
                f"Offending sentence: {sentence.strip()!r}"
            )

        if _OPEN_FOLD.search(sentence) and any(
            a.action_type
            in (
                PreflopActionType.RAISE,
                PreflopActionType.ALL_IN,
                PreflopActionType.CALL,
            )
            for a in history
        ):
            return PreflopValidationResult.fail(
                "the explanation says 'open-fold' but you are FACING "
                "action -- open-folding means folding when the action "
                "folds to you unopened. Just say fold. Offending "
                f"sentence: {sentence.strip()!r}"
            )

        for match in _N_BET_TOKEN.finditer(sentence):
            level = int(match.group(1))
            if level > max_nameable_bet:
                return PreflopValidationResult.fail(
                    f"the explanation mentions a {level}-bet but this line "
                    f"can only reach a {max_nameable_bet}-bet (raises so "
                    f"far: {n_raises}, plus your raise and one re-raise). "
                    "Count the ladder: open, 3-bet, 4-bet, 5-bet. "
                    f"Offending sentence: {sentence.strip()!r}"
                )

        if (
            _CAPPED.search(sentence)
            and facts.villain_stats is not None
            and (villain_pos in _positions_named(sentence) or "villain" in low)
            and any(
                hand_class in ("AA", "KK") and weight >= 0.5
                for hand_class, weight in facts.villain_stats.top_combos
            )
        ):
            return PreflopValidationResult.fail(
                "the explanation calls villain's range capped, but that "
                "range contains AA/KK at high weight -- capped means the "
                "range CANNOT contain the top hands. Describe it as strong "
                f"or condensed instead. Offending sentence: {sentence.strip()!r}"
            )

    return PreflopValidationResult.ok()


def validate_card_suit_consistency(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> PreflopValidationResult:
    """Every specific card named in prose must be one of hero's two cards.

    Preflop there is no board and villain's cards are unknown, so the only
    specific cards the explanation may name are hero's own. Round-2 audit:
    one row wrote "your K❤️" twice while hero held K♠5♠ -- strategy-neutral
    preflop, but trust-destroying and trivially checkable.
    """
    text = generated.answer_explanation or ""
    if not text:
        return PreflopValidationResult.ok()

    combo = facts.spot.hero_card_combo or ""
    hero_cards = {
        (combo[i].upper(), combo[i + 1].lower())
        for i in range(0, len(combo) - 1, 2)
    }
    if not hero_cards:
        return PreflopValidationResult.ok()

    offending = sorted({
        f"{rank}{_SUIT_LETTER[suit]}"
        for rank, suit in _CARD_EMOJI.findall(text)
        if (rank, _SUIT_LETTER[suit]) not in hero_cards
    })
    if offending:
        return PreflopValidationResult.fail(
            "the explanation names specific cards that are not yours: "
            + ", ".join(offending)
            + f". Your cards are {combo}. Preflop the only specific cards "
            "you may name are your own two (hand-class labels like AKo "
            "stay plain text)."
        )
    return PreflopValidationResult.ok()


# --- soft validators (flag, never fail) --------------------------------------
# Position phrases + subject cues for the soft position check. A phrase only
# contradicts hero_position when it describes HERO. "the small blind 3-bets
# out of position" is about the villain and must NOT flag -- that false
# positive showed up on the first batch run through the new in-depth prompt
# (June 2026). We read the SUBJECT slot: the sentence text BEFORE the phrase.
# Hero-bound when a hero pronoun precedes it; other-bound when only a seat
# name / villain word does.
_POS_IN = re.compile(r"\bin position\b", re.I)
_POS_OUT = re.compile(r"\bout of position\b", re.I)
_HERO_SUBJECT = re.compile(r"\b(?:you|your|yourself)\b", re.I)
_OTHER_SUBJECT = re.compile(
    r"\b(?:villain|opponent|the raiser|the opener|the limper|small blind|"
    r"big blind|under the gun|lojack|hijack|cutoff|button|UTG\+?[12]?|LJ|HJ|"
    r"CO|BTN|SB|BB)\b",
    re.I,
)
# A negation right before the phrase ("not in position", "isn't in position")
# AGREES with an OOP hero, so it must not flag.
_POS_NEGATED = re.compile(r"(?:\bnot|n't)\s+$", re.I)


def _position_claim_is_hero_bound(before: str) -> bool:
    """Whether a position phrase describes hero, judged from its subject slot.

    ``before`` is the sentence text preceding the phrase. Hero-bound when a
    hero pronoun appears in it; other-bound (False) when only a seat name or
    villain word does. Subjectless text reads as hero-bound -- the verdict
    sentences are about hero by default.
    """
    if _HERO_SUBJECT.search(before):
        return True
    return not _OTHER_SUBJECT.search(before)


def soft_validate_position_words(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> list[str]:
    """Warn when a HERO-bound position word contradicts the hero_position fact.

    SOFT on purpose: a contradiction is *usually* the round-2 garble (#15's
    "Sorry, you're in the HJ"), but can be a legitimate multiway-realization
    note ("you'd be out of position against the callers behind"), so it flags
    for review instead of failing the generation. Only fires when the
    phrase's SUBJECT is hero -- describing the VILLAIN as out of position
    (extremely common and correct) no longer false-positives.
    """
    from pipeline.preflop.position import (  # noqa: PLC0415
        hero_relative_position,
    )

    text = generated.answer_explanation or ""
    if not text:
        return []

    relative = hero_relative_position(facts)
    if relative == "In Position":
        phrase_re, claim = _POS_OUT, "out of position"
    elif relative == "Out of Position":
        phrase_re, claim = _POS_IN, "in position"
    else:
        return []

    for sentence in _SENTENCE_SPLIT.split(text):
        for m in phrase_re.finditer(sentence):
            before = sentence[: m.start()]
            if _POS_NEGATED.search(before):
                continue
            if _position_claim_is_hero_bound(before):
                return [
                    f"prose says '{claim}' about hero, but hero is "
                    f"{relative} vs the villain. Review (may be a legit note "
                    "about players still to act behind)."
                ]
    return []


# --- soft #2: number-vs-data ------------------------------------------------
# A percentage figure in the prose, e.g. "55%" / "41.5 %".
_PCT_FIGURE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
# How far a cited number may sit from the matching fact before we flag.
# Generous on purpose: equity is Monte-Carlo'd (~1-2% noise) and prose rounds,
# so only a CLEAR miss (the LLM inventing a number) trips this.
_NUMBER_TOLERANCE_PCT = 10.0
# Context cues, scanned in a small window around each figure. Pot-odds /
# break-even cues win over equity cues when both are present ("you need 41%
# equity to call" is a break-even threshold, not hero's actual equity).
_EQUITY_CUES = ("equity", "favorite", "favourite")
_POTODDS_CUES = (
    "to call", "break-even", "break even", "breakeven", "pot odds",
    "the price", "priced",
)
# "your range realizes 53%" is RANGE equity (a different fact); skip those so
# we never compare a range number to hero's HAND equity.
_HERO_RANGE_CUES = ("your range", "our range")


def soft_validate_number_vs_data(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> list[str]:
    """Warn when a cited equity / pot-odds % contradicts the data block.

    Guards the core invariant -- the LLM must use the solver's numbers, never
    invent them. HEADS-UP ONLY (single villain, not a multi-way pot) so a
    field-vs-per-opponent equity number can't false-fire. For each ``NN%`` in
    the prose we read a small window of surrounding text:

      * a break-even / pot-odds figure ("need 41% to call") is compared to
        ``break_even_equity``;
      * an equity figure ("55% equity / favorite") is compared to
        ``hero_equity_vs_villain`` (and, defensively, the range-equity fact),
        unless it reads as a "your range" (range-equity) claim.

    Only a mismatch larger than :data:`_NUMBER_TOLERANCE_PCT` flags, and a
    figure we can't confidently map to a fact is left alone. SOFT: a real
    miss is the LLM inventing a number, but odd phrasings exist, so it queues
    for review rather than failing the row.
    """
    # Gate: single villain, heads-up. showdown_opponents has length >= 2 only
    # on multi-way pots; it's empty heads-up.
    if facts.villain_stats is None or len(facts.showdown_opponents) >= 2:
        return []

    text = generated.answer_explanation or ""
    if not text:
        return []

    for m in _PCT_FIGURE.finditer(text):
        value = float(m.group(1))
        if value <= 0 or value >= 100:
            continue  # 0% / 100% are not equity or pot-odds claims
        window = text[max(0, m.start() - 55) : m.end() + 30].lower()
        is_potodds = any(cue in window for cue in _POTODDS_CUES)
        is_equity = any(cue in window for cue in _EQUITY_CUES)

        if is_potodds and facts.break_even_equity is not None:
            target = facts.break_even_equity * 100.0
            if abs(value - target) > _NUMBER_TOLERANCE_PCT:
                return [
                    f"prose cites {value:g}% as the price to call, but the "
                    f"data block's break-even equity is {target:.0f}%. Review "
                    "the number (the LLM may have invented it)."
                ]
        elif (
            is_equity
            and not is_potodds
            and facts.hero_equity_vs_villain is not None
            and not any(cue in window for cue in _HERO_RANGE_CUES)
        ):
            candidates = [facts.hero_equity_vs_villain * 100.0]
            if facts.hero_range_equity_vs_villain is not None:
                candidates.append(facts.hero_range_equity_vs_villain * 100.0)
            if min(abs(value - c) for c in candidates) > _NUMBER_TOLERANCE_PCT:
                return [
                    f"prose cites {value:g}% equity, but the data block has "
                    f"hero equity {candidates[0]:.0f}% vs the villain. Review "
                    "the number (the LLM may have invented it)."
                ]
    return []


# --- soft #3: verdict-vs-answer ---------------------------------------------
# Coarse action buckets for the opening-verdict check. We compare INTENT
# (fold / passive / aggressive), not exact sizing -- "raise" vs "shove" or
# "call" vs "check" are not conflicts worth a human's time, but a verdict that
# says "fold" when the answer is "raise" is. Order: most specific first so a
# raise-family token isn't shadowed.
_VERDICT_BUCKETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfold(?:s|ed|ing)?\b", re.I), "fold"),
    (re.compile(
        r"\b(?:3-?bet|4-?bet|5-?bet|three-?bet|four-?bet|five-?bet)"
        r"(?:s|ting|ted)?\b",
        re.I,
    ), "aggressive"),
    (re.compile(r"\braise(?:s|d)?\b|\braising\b", re.I), "aggressive"),
    (re.compile(
        r"\b(?:all-?in|shove[sd]?|shoving|jam(?:s|med|ming)?)\b", re.I,
    ), "aggressive"),
    (re.compile(r"\bcall(?:s|ed|ing)?\b", re.I), "passive"),
    (re.compile(r"\bcheck(?:s|ed|ing)?\b", re.I), "passive"),
)


def _dominant_action_bucket(dominant_action: str) -> str | None:
    """Map a dominant-action label to a coarse fold/passive/aggressive bucket.

    ``"Fold"`` -> fold, ``"Call"`` / ``"Check"`` -> passive, ``"Raise 60%"`` /
    ``"AllIn"`` -> aggressive. None for anything unrecognised (don't flag).
    A BB check is filed under "Call" by the pack, so "Call" maps to passive,
    keeping check/call compatible.
    """
    low = dominant_action.lower()
    if "fold" in low:
        return "fold"
    if "raise" in low or "allin" in low or "all-in" in low:
        return "aggressive"
    if "call" in low or "check" in low:
        return "passive"
    return None


def soft_validate_verdict_vs_answer(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> list[str]:
    """Warn when the opening verdict's action conflicts with the answer.

    The first sentence is the verdict. We collect the action buckets it names
    (fold / passive / aggressive). If the spot's dominant-action bucket is
    among them, the verdict references the right action -- pass (this lets
    "folding looks tempting but calling is correct" through unflagged when the
    answer is call). Only when the dominant bucket is ENTIRELY ABSENT from the
    first sentence and a conflicting action is named do we flag -- e.g. a
    "this is a clear fold" verdict on a spot whose answer is raise. SOFT:
    almost always a real garble, but occasionally a hedge, so it's a flag.
    """
    text = generated.answer_explanation or ""
    if not text:
        return []

    dominant_bucket = _dominant_action_bucket(facts.spot.dominant_action or "")
    if dominant_bucket is None:
        return []

    first_sentence = _SENTENCE_SPLIT.split(text)[0]
    present: set[str] = set()
    earliest_pos = len(first_sentence) + 1
    earliest_word = ""
    for pattern, bucket in _VERDICT_BUCKETS:
        m = pattern.search(first_sentence)
        if m is None:
            continue
        present.add(bucket)
        if m.start() < earliest_pos:
            earliest_pos = m.start()
            earliest_word = m.group(0)

    if not present or dominant_bucket in present:
        return []

    return [
        f"the opening verdict reads as '{earliest_word}' but the spot's "
        f"answer is {facts.spot.dominant_action!r} (a {dominant_bucket} "
        "action). Review whether the first sentence states the right action."
    ]


# --- soft #4: named-combo-in-range ------------------------------------------
# A sentence asserts villain HOLDS something. We require a villain referent
# AND a possession verb in the same sentence, so a bare "AK is strong for us"
# (hero's hand) isn't policed. Two shapes cover the common prose.
_VILLAIN_HAS = re.compile(
    r"\b(?:villains?|opponents?|the\s+raiser|the\s+opener|the\s+\d-?bettor|"
    r"the\s+caller|they|their\s+range|its\s+range)\b"
    r"[^.!?]*?\b(?:has|have|holds?|shows?\s+up\s+with|include[sd]?|"
    r"contain[sd]?|turns?\s+up\s+with|shows?\s+down)\b",
    re.I,
)
_IN_VILLAIN_RANGE = re.compile(
    r"\bin\s+(?:villain'?s?|their|its|the\s+(?:raiser|opener|caller)'?s?)"
    r"\s+range\b",
    re.I,
)
# Negation/hedge cues: a sentence saying villain does NOT have a class is a
# TRUE statement about an absent hand, so it must never be policed. Extends
# the blocker-validator markers with "rarely"/"unlikely"/"seldom"/"without".
_RANGE_NEGATION = _NEGATION_MARKERS + (
    " rarely", " unlikely", " seldom", " without",
)


def _class_in_range(core: str, suffix: str, played: frozenset[str]) -> bool:
    """Is the prose's hand-class token present in villain's played classes?

    Mirrors :func:`_hand_blocked_in_fact`: pairs match exactly; a bare
    non-pair like "AK" counts as present if EITHER the suited or offsuit class
    is played (prose often drops the s/o suffix).
    """
    if suffix:
        return f"{core}{suffix}" in played
    if core[0] == core[1]:  # pair
        return core in played
    return f"{core}s" in played or f"{core}o" in played


def soft_validate_named_combo_in_range(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> list[str]:
    """Warn when prose claims villain HAS a hand class absent from their range.

    Lowest-precision of the soft checks, so it is scoped tightly: HEADS-UP
    only (so "villain has X" can't mean the wrong villain on a multi-way pot),
    skips negated/hedged sentences ("villain doesn't have AA"), and fires only
    on sentences that pair a villain referent with a possession verb -- so a
    bare "AK is strong for us" (hero's own hand) is never policed. A named
    class is flagged only when NONE of its representable classes appear in
    ``villain_played_classes`` (the full weight>0 range). SOFT: a genuine "has
    a hand they fold" claim is a real defect, but the parse is fuzzy, so it
    flags for review.
    """
    played = facts.villain_played_classes
    if not played or len(facts.showdown_opponents) >= 2:
        return []  # no villain range to check, or multi-way (ambiguous referent)

    text = generated.answer_explanation or ""
    if not text:
        return []

    for sentence in _SENTENCE_SPLIT.split(text):
        if not (_VILLAIN_HAS.search(sentence) or _IN_VILLAIN_RANGE.search(sentence)):
            continue
        low = sentence.lower()
        if any(marker in low for marker in _RANGE_NEGATION):
            continue
        absent = sorted({
            f"{core}{suffix or ''}"
            for core, suffix in _HAND_CLASS_TOKEN.findall(sentence)
            if not _class_in_range(core, suffix, played)
        })
        if absent:
            return [
                "prose says villain's range has "
                + ", ".join(absent)
                + ", but that hand class is at 0% in villain's actual range. "
                f"Review the claim. Offending sentence: {sentence.strip()!r}"
            ]
    return []


def run_preflop_soft_validators(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> list[str]:
    """Run every soft validator; return all warnings (empty = clean).

    Soft warnings never fail a generation and never trigger a retry. The
    batch driver marks warned rows ``validation_status='flagged'`` and
    records the warnings in the meta sidecar so the Review page surfaces
    them for a human.
    """
    warnings: list[str] = []
    for check in (
        soft_validate_position_words,
        soft_validate_number_vs_data,
        soft_validate_verdict_vs_answer,
        soft_validate_named_combo_in_range,
    ):
        warnings.extend(check(generated, facts))
    return warnings


# --- runner -----------------------------------------------------------------
def run_preflop_audit_validators(
    generated: GeneratedExplanation,
    facts: PreflopFacts,
) -> PreflopValidationResult:
    """Run every hard validator in series; return the first failure or ok.

    Mirrors ``pipeline.validators.run_audit_validators`` for the
    preflop path. The retry loop in
    :func:`pipeline.preflop.explanation_generator.generate_preflop_answer_explanation`
    calls this after the structural ``_validate`` check and uses the
    returned ``error_message`` as the corrective LLM feedback for the
    retry round.

    Order matters: cheaper / more-likely-to-fire checks first so a
    catastrophic failure (e.g. inventing an option) short-circuits
    before we re-scan the prose for banned punctuation.
    """
    for check in (
        validate_option_set,
        validate_no_standalone_sometimes,
        validate_composite_label_frequencies,
        validate_no_postflop_on_allin,
        validate_banned_phrases,
        validate_card_suit_consistency,
        validate_blocker_claims,
        validate_terminology,
    ):
        result = check(generated, facts)
        if not result.is_valid:
            return result
    return PreflopValidationResult.ok()


__all__ = [
    "PreflopValidationResult",
    "run_preflop_audit_validators",
    "run_preflop_soft_validators",
    "soft_validate_named_combo_in_range",
    "soft_validate_number_vs_data",
    "soft_validate_position_words",
    "soft_validate_verdict_vs_answer",
    "validate_banned_phrases",
    "validate_blocker_claims",
    "validate_card_suit_consistency",
    "validate_composite_label_frequencies",
    "validate_no_postflop_on_allin",
    "validate_no_standalone_sometimes",
    "validate_option_set",
    "validate_terminology",
]
