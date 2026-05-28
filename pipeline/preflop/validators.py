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
  4. :func:`validate_no_postflop_talk` -- preflop prose shouldn't
     reference the flop, turn, river, board, or community cards
     (recurring LLM failure mode).
  5. :func:`validate_banned_phrases` -- em dashes, semicolons, and
     the team's literal-phrase blocklist (e.g. "leverage the dead
     money", "dynamic spot") are banned. The LLM is told these in
     the system prompt; this validator hard-enforces.

Strategy / failure-pattern validators are stubbed in
:func:`run_preflop_audit_validators` but currently always pass --
will be tuned against the first batches of real review feedback.

Soft (warning-only) validators run after the hard stack and never
fail a generation. Reserved for v2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.explanation_generator import (
    BANNED_LITERAL_PHRASES,
    GeneratedExplanation,
)
from pipeline.preflop.fact_extractor import PreflopFacts
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

# Postflop terms that should never appear in preflop prose. Compiled
# as word-boundary regex so "flopped" matches but "preflop" doesn't.
# Case-insensitive.
_POSTFLOP_TERM_PATTERNS = tuple(re.compile(rf"\b{term}\b", re.IGNORECASE) for term in [
    "flop", "flops", "flopped",
    "turn card", "turn",
    "river", "rivered",
    "board", "boards",
    "community card", "community cards",
    "runout", "runouts", "run out",
    "draws", "drawing dead",
    "set mining", "set-mining",  # could be relevant preflop ("smallpair sets")
    # but more often a postflop concept; tune if we see false positives
])
# Phrases that LOOK postflop but are actually fine preflop.
# If the term appears INSIDE one of these phrases, don't flag it.
_POSTFLOP_TERM_EXEMPTIONS = (
    "preflop", "before the flop", "before any community cards",
    "set mine",  # "we want to set mine this small pair" is preflop talk
    "set-mine",
)

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


def validate_no_postflop_talk(
    generated: GeneratedExplanation,
    facts: PreflopFacts,  # noqa: ARG001
) -> PreflopValidationResult:
    """Preflop prose shouldn't reference flop/turn/river/board/etc.

    Recurring LLM failure mode: the model knows poker and reaches for
    postflop concepts (range vs. board interaction, drawing odds, etc.)
    when none of that has happened yet. Hard-flag any postflop term
    that isn't inside one of the exempt phrases (e.g. "preflop", which
    contains "flop").
    """
    text = generated.answer_explanation or ""
    if not text:
        return PreflopValidationResult.ok()

    # Mask out exempt phrases so their substring matches don't flag.
    masked = text.lower()
    for phrase in _POSTFLOP_TERM_EXEMPTIONS:
        masked = masked.replace(phrase, " " * len(phrase))

    hits: set[str] = set()
    for pattern in _POSTFLOP_TERM_PATTERNS:
        for match in pattern.finditer(masked):
            hits.add(match.group(0))

    if hits:
        return PreflopValidationResult.fail(
            "preflop explanation references postflop concepts -- "
            "remove mentions of: " + ", ".join(sorted(hits))
            + ". Preflop questions cover only the action before any "
            "community cards are dealt."
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
        validate_no_postflop_talk,
        validate_banned_phrases,
    ):
        result = check(generated, facts)
        if not result.is_valid:
            return result
    return PreflopValidationResult.ok()


__all__ = [
    "PreflopValidationResult",
    "run_preflop_audit_validators",
    "validate_banned_phrases",
    "validate_composite_label_frequencies",
    "validate_no_postflop_talk",
    "validate_no_standalone_sometimes",
    "validate_option_set",
]
