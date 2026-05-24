"""Layer 7 (initial): post-LLM strategic-content validators.

These run AFTER Layer 6's existing structural validation (correct_answer must
match one of the option strings) and BEFORE the explanation is accepted into
a CSV row. They enforce the trust-chain invariants the May-2026 audit
(test_output/audit_report.md) caught the LLM violating:

  * validate_option_set                -- every option's primary verb is a
                                          verb Pio actually offers at the
                                          decision node.
  * validate_correct_answer_verb       -- correct_answer's prefix is the one
                                          Python computed; correct_answer's
                                          verb equals decision.correct_action.
  * validate_option_set_completeness   -- on frequency-style spots, the
                                          option set covers Pio's two
                                          most-played actions (catches the
                                          Row-1 defect: LLM dropped fold
                                          because it preferred a call-vs-raise
                                          template).

Each validator is a pure function `(GeneratedExplanation, DecisionData) ->
ValidationResult`; they don't call the LLM, don't hit Pio, and don't mutate
state. The retry orchestration lives in `explanation_generator.generate_
explanation`.

This file is deliberately small. Per the brief: "spend a full day reading raw
output yourself to calibrate the quality bar before building automated
checkers." We are only adding what the audit empirically proved necessary;
hypothetical failure modes are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.explanation_generator import (
    GeneratedExplanation, frequency_to_verb_prefix,
)
from pipeline.fact_extractor.spot_data import DecisionData

# Known action verbs that may appear as the primary verb in an option string.
# Sourced from the action-history renderer's `_format_action` + the verbs
# PioSolver labels (`bet`, `raise`, `call`, `fold`, `check`, plus `donk`,
# `shove`, `jam`, `all-in` which appear in some scenarios).
_KNOWN_ACTION_VERBS = frozenset({
    "call", "fold", "check", "bet", "raise",
    "donk", "shove", "jam", "all-in",
    "limp",                                   # preflop only, kept for symmetry
})

# Frequency-prefix words. Lowercased so the matcher is case-insensitive.
_FREQUENCY_PREFIXES = frozenset({"always", "mostly", "sometimes", "rarely"})

# An action played at or above this freq must appear in the option set
# (otherwise the LLM has silently dropped a non-trivial Pio action -- the
# Row-1 defect). Tuned to catch fold-at-34% (v3 Row 1) without firing on
# tiny mix-in actions like the b608 raise (freq 0.0008).
COMPLETENESS_MIN_FREQ = 0.10


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of one validator. Composable via `combine`."""

    is_valid: bool
    error_message: str = ""

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(is_valid=True)

    @classmethod
    def fail(cls, message: str) -> "ValidationResult":
        return cls(is_valid=False, error_message=message)


# --- helpers ----------------------------------------------------------------
def extract_action_verb(option: str) -> str | None:
    """The primary action verb of an option string, or None if not parseable.

    Handles the four template shapes the pipeline produces:

      * binary action   "Call", "Fold", "Check", "Bet", "Raise"
      * frequency       "Always call", "Mostly fold", "Sometimes check"
      * compound freq   "Mostly fold, sometimes call"  (rare; primary verb wins)
      * sizing label    "33% pot", "$0.75", "$1.25 bet"

    For sizing labels with no explicit verb (e.g. "33% pot") returns None --
    callers must handle the None case (sizing-style validators skip the verb
    extraction step). Returns the verb in lowercase.
    """
    if not option:
        return None
    words = option.lower().split()
    if not words:
        return None
    # Strip a leading frequency prefix.
    if words[0] in _FREQUENCY_PREFIXES:
        words = words[1:]
    # The first known action verb wins. (For compound options like "Mostly
    # fold, sometimes call", that's the primary verb -- the one the prefix
    # modifies.)
    for word in words:
        cleaned = word.strip(".,;:!?\"'()[]")
        if cleaned in _KNOWN_ACTION_VERBS:
            return cleaned
    return None


def extract_frequency_prefix(option: str) -> str | None:
    """The frequency-prefix word of an option string, or None if absent.

    Case-normalised to title case ("Always" / "Mostly" / etc.). Returns None
    for binary-action strings ("Call") and sizing labels ("33% pot").
    """
    if not option:
        return None
    first = option.split()[0].lower() if option.split() else ""
    if first in _FREQUENCY_PREFIXES:
        return first.capitalize()
    return None


# --- the three validators ---------------------------------------------------
def validate_option_set(generated: GeneratedExplanation,
                        decision: DecisionData) -> ValidationResult:
    """Every option's primary verb appears in Pio's offered actions.

    Catches the case where the LLM invents an action Pio never offers (e.g.
    proposing 'raise' as an option on a spot where only call/fold are in
    `range_aggregate_strategy`). Sizing-style options whose verb is implicit
    (no recognisable verb token) skip the check -- the option-set
    completeness validator covers the missing-verb case.
    """
    pio_verbs = set(decision.range_aggregate_strategy.keys())
    if not pio_verbs:
        return ValidationResult.ok()         # no strategy data -- nothing to check
    invalid = []
    for slot, option in zip(("option_1", "option_2", "option_3", "option_4"),
                            (generated.option_1, generated.option_2,
                             generated.option_3, generated.option_4)):
        if not option:
            continue
        verb = extract_action_verb(option)
        if verb is None:
            continue                         # sizing-only / unparseable -- skip
        if verb not in pio_verbs:
            invalid.append(
                f"{slot}={option!r} references action {verb!r}, which is "
                f"not in Pio's range_aggregate_strategy "
                f"(offered: {sorted(pio_verbs)})"
            )
    if invalid:
        return ValidationResult.fail("; ".join(invalid))
    return ValidationResult.ok()


def validate_correct_answer_verb(generated: GeneratedExplanation,
                                 decision: DecisionData) -> ValidationResult:
    """correct_answer's verb equals decision.correct_action, and its prefix
    (if any) equals the deterministic Python-computed prefix.

    Two distinct checks combined here because they share the verb-extraction
    step and a failure of either means the same retry message ("the LLM did
    not honour the Python-computed correct answer").

    Catches:
      * LLM picks "Mostly raise" when decision.correct_action is "call"
        (would slip past `_validate(explanation)` since "Mostly raise" can
        legitimately be one of the four options);
      * LLM picks "Always call" when Python supplied "Mostly" (prefix mismatch).
    """
    if not decision.correct_action:
        return ValidationResult.ok()         # no Pio-derived expected verb
    verb = extract_action_verb(generated.correct_answer)
    if verb is None:
        return ValidationResult.fail(
            f"correct_answer={generated.correct_answer!r} has no "
            f"recognisable action verb")
    if verb != decision.correct_action:
        return ValidationResult.fail(
            f"correct_answer verb {verb!r} (extracted from "
            f"{generated.correct_answer!r}) does not match Pio's "
            f"correct_action {decision.correct_action!r}")
    # Prefix check applies only when Python computed one (frequency-style).
    strategy = decision.range_aggregate_strategy
    if not strategy:
        return ValidationResult.ok()
    top_freq = max(strategy.values())
    # The Python helper for "what prefix does Pio's dominant action get" is
    # frequency_to_verb_prefix. For binary_action style (top_freq >= 0.80)
    # the correct_answer has no prefix and the check is vacuous.
    if top_freq >= 0.80:
        return ValidationResult.ok()
    expected_prefix = frequency_to_verb_prefix(top_freq)
    actual_prefix = extract_frequency_prefix(generated.correct_answer)
    if actual_prefix != expected_prefix:
        return ValidationResult.fail(
            f"correct_answer={generated.correct_answer!r} uses prefix "
            f"{actual_prefix!r} but Python's deterministic mapping requires "
            f"{expected_prefix!r} (Pio top-action freq = {top_freq:.4f})"
        )
    return ValidationResult.ok()


def validate_option_set_completeness(
        generated: GeneratedExplanation,
        decision: DecisionData) -> ValidationResult:
    """The option set covers Pio's top two actions by frequency.

    Defends against the Row-1 defect: Pio mixes call (66%) / fold (34%) /
    raise (0.08%) and the LLM emitted options ["Always call", "Mostly call",
    "Mostly raise", "Always raise"] -- fold absent despite being the
    second-most-frequent action.

    Specifically: every Pio verb whose frequency >= COMPLETENESS_MIN_FREQ
    must appear as the primary verb of some option. Tiny mix-in actions
    (freq below the threshold) are exempt -- the LLM legitimately can elide
    a verb that Pio almost never plays.
    """
    strategy = decision.range_aggregate_strategy
    if not strategy:
        return ValidationResult.ok()
    must_cover = {verb for verb, freq in strategy.items()
                  if freq >= COMPLETENESS_MIN_FREQ}
    if not must_cover:
        return ValidationResult.ok()
    option_verbs: set[str] = set()
    for option in (generated.option_1, generated.option_2,
                   generated.option_3, generated.option_4):
        verb = extract_action_verb(option)
        if verb is not None:
            option_verbs.add(verb)
    missing = sorted(must_cover - option_verbs)
    if missing:
        details = ", ".join(
            f"{verb!r} (Pio freq {strategy[verb]:.2%})" for verb in missing)
        return ValidationResult.fail(
            f"option set omits action(s) Pio plays at >= "
            f"{COMPLETENESS_MIN_FREQ:.0%}: {details}. Options were: "
            f"{[generated.option_1, generated.option_2, generated.option_3, generated.option_4]!r}"
        )
    return ValidationResult.ok()


# --- Ryan-feedback Fix 5 hard validator (May 2026) --------------------------
# Per-archetype ANTI-PATTERN phrases. If any of these substrings appears in
# answer_explanation when the spot's archetype is set, the framing is wrong
# and the explanation is rejected (one retry, then routed to human review by
# the standard Layer 6 retry orchestration).
#
# Lowercased for case-insensitive comparison; the validator lowercases the
# explanation before checking. Substrings are short and unambiguous to keep
# false positives near zero.
_ARCHETYPE_ANTI_PATTERNS = {
    "trap_check": (
        # V7.1's exact failure: matched "nut_advantage_villain" concept tag
        # literally and wrote "villain has the nut advantage" -- wrong frame,
        # hero in trap_check is the one with the strong hand inducing villain.
        "villain has the nut advantage",
        "villain has nut advantage",
        "btn has the nut advantage", "bb has the nut advantage",
        "co has the nut advantage", "hj has the nut advantage",
        "utg has the nut advantage", "sb has the nut advantage",
        "you have the weaker range",
        "your range is weaker",
        "you don't have many strong hands",
        # Pot-control framing is also wrong for trap_check -- hero WANTS the
        # pot big with this hand, so calling the check "pot control" misframes.
        "pot control", "controlling the pot size",
    ),
    "bluff_catch": (
        # Bluff catch is NOT a value call; framing it as one is the prior
        # failure mode the user wants caught.
        "for value", "you bet for value",
        "your hand is strong enough to win at showdown most of the time",
    ),
    "pot_control_check": (
        # Pot control is a strong/medium hand checking; "villain has nut
        # advantage" mis-frames the spot (hero has the equity).
        "villain has the nut advantage",
        "villain has nut advantage",
        "your hand is too weak",
    ),
    "value_bet": (
        # A value bet should not be framed as a bluff or as "fold equity".
        "fold equity is the primary",
        "this is a bluff",
        "fold out better hands",
    ),
    "bluff": (
        # A bluff should not be framed as a value bet.
        "for thin value",
        "extracting value from worse hands",
    ),
    "fold_to_polar": (
        # Folding should not be framed as "the price is right" (that's
        # bluff_catch framing).
        "the price is right",
        "you have enough equity",
    ),
    "call_drawing": (
        # Drawing call should not be framed as a bluff catch.
        "bluff combos outweigh value",
    ),
}


def validate_archetype_consistency(
        generated: "GeneratedExplanation",
        decision: "DecisionData") -> "ValidationResult":
    """Hard check: explanation prose does NOT contain anti-pattern phrasing
    for the spot's recommended_action_archetype.

    Per Ryan-feedback Fix 5 (May 2026), this catches the strategic-frame
    mismatches the V7.1 audit surfaced -- chiefly a trap_check spot framed
    with 'villain has the nut advantage' (concept-tag matching) instead of
    'hero checks to induce continued villain aggression' (the archetype
    frame). Failures route through the standard Layer 6 retry loop: one
    corrective retry, then ExplanationValidationError for Layer 7 / human
    review.

    Skips when no archetype is set (test fixtures, legacy SpotData) and
    when no anti-patterns are configured for the archetype.
    """
    archetype = getattr(decision, "recommended_action_archetype", "")
    if not archetype or archetype not in _ARCHETYPE_ANTI_PATTERNS:
        return ValidationResult.ok()
    text = generated.answer_explanation.lower()
    hits = [p for p in _ARCHETYPE_ANTI_PATTERNS[archetype] if p in text]
    if hits:
        return ValidationResult.fail(
            f"explanation frames a {archetype!r} spot with anti-pattern "
            f"phrasing: {hits!r}. Re-read the per-archetype guidance in "
            f"the system prompt (STRATEGIC ARCHETYPES section, entry for "
            f"{archetype!r}) and re-emit the explanation with the correct "
            f"strategic frame -- the action stays the same, only the prose "
            f"changes."
        )
    return ValidationResult.ok()


# --- Ryan-feedback Fix 4 soft validator (May 2026) --------------------------
# Detects when answer_explanation discusses villain's range but doesn't name a
# specific combo or hand class. "Specific" means at least one of: an emoji-card
# pair (e.g. "K♠️K♣️") OR a hand-class label substring from a known list.
import re as __re_for_fix4
_EMOJI_COMBO_RE = __re_for_fix4.compile(
    r"[2-9TJQKA][♠♣♦❤]")  # rank + spade/club/diamond/heart


def validate_villain_combo_citation(
        generated: "GeneratedExplanation",
        decision: "DecisionData") -> "ValidationResult":
    """Soft check: answer_explanation cites at least one specific villain combo
    or hand class when villain's range is being discussed.

    Per Ryan-feedback Fix 4 (May 2026), prior explanations described villain's
    range abstractly ("villain has value hands and bluffs"). Layer 6 now has
    `range_data.villain_top_value_combos` with example_combos in emoji form and
    voice rule 10 instructs the LLM to cite 2-3 of them. This validator catches
    regressions where the LLM ignores the data and reverts to abstract phrases.

    Heuristic:
      * if the explanation references villain at all (mentions "villain", "BB",
        "BTN", etc.) AND has no emoji-combo pattern AND has no hand-class label
        substring -> warn (soft, doesn't fail).

    Soft per Ryan's instruction. Will track in the v7.3 verification batch
    whether to harden to a rejection.
    """
    text = generated.answer_explanation
    if not text:
        return ValidationResult.ok()
    # If the text doesn't talk about villain at all, no requirement to cite.
    text_lc = text.lower()
    villain_mentioned = any(
        token in text_lc
        for token in ("villain", "villain's", "they ", "their ",
                      "bb ", "btn ", "co ", "hj ", "utg ", "sb ", "lj ")
    )
    if not villain_mentioned:
        return ValidationResult.ok()
    # Look for either an emoji-combo pattern or a known hand-class label.
    has_emoji_combo = bool(_EMOJI_COMBO_RE.search(text))
    # Pull known hand-class label substrings out of the data block so the
    # validator stays accurate as the hand_class taxonomy evolves.
    known_classes = set()
    rd = getattr(decision, "_range_data_for_validation", None)  # unused hook
    # Fallback: hardcoded common substrings that always indicate hand-class
    # citation. (The LLM rarely writes "top_pair_no_draws" verbatim; it writes
    # natural-language equivalents like "top pair" or "two pair".)
    hand_class_phrases = (
        "set ", "sets ", "set.", "sets.",
        "two pair", "top pair", "second pair", "third pair", "bottom pair",
        "overpair", "trips", "quads", "straight", "flush",
        "full house", "boat",
        "pocket pair", "ace-high", "ace high",
    )
    has_class_phrase = any(p in text_lc for p in hand_class_phrases)
    if not has_emoji_combo and not has_class_phrase:
        import sys
        print("  [soft-warn validate_villain_combo_citation] explanation "
              "discusses villain but cites no specific combo (no emoji card "
              "pair like 'K♠️K♣️') and no hand-class phrase. Voice rule 10 "
              "requires anchoring abstract villain-range talk to actual "
              "combos. Not retrying.", file=sys.stderr)
    return ValidationResult.ok()


# --- Ryan-feedback Fix 3 soft validator (May 2026) --------------------------
# Plain-text suit notation in an answer_explanation indicates the LLM bypassed
# voice rule 9 (use suit emojis). Regex matches a run of one-or-more rank+suit
# pairs ([2-9TJQKA] followed by [shdc]), bordered by non-alphanumeric chars or
# string edges. This catches both standalone "Kh" and concatenated "KsKh" /
# "AdKdQd" forms PioSolver dumps cards in. English words like "Adam" or "is"
# never match (the rank letter isn't followed by a suit letter at the right
# spot, and the boundary lookarounds reject letter-adjacent fragments).
import re as _re

_PLAIN_CARD_RE = _re.compile(
    r"(?<![A-Za-z0-9])"                  # left edge: non-alnum or start of string
    r"([2-9TJQKA][shdc])"                # one rank+suit pair, captured
    r"(?:[2-9TJQKA][shdc])*"             # optional further pairs in the same run
    r"(?![A-Za-z0-9])"                   # right edge: non-alnum or end of string
)
_PLAIN_CARD_TOKEN_RE = _re.compile(r"[2-9TJQKA][shdc]")


def validate_no_plain_card_notation(
        generated: GeneratedExplanation,
        decision: DecisionData) -> ValidationResult:
    """Soft check: answer_explanation must not contain plain-text card notation.

    Voice rule 9 says specific card / combo citations use suit emojis (♠️ ❤️
    ♦️ ♣️). Plain forms like "Kh" or "AdKd" are internal solver notation and
    don't belong in coaching prose. This is the lighter sibling of the other
    validators: warnings are SURFACED (printed to stderr) but the validator
    returns ok so the explanation isn't rejected outright. Track failures for
    a follow-up hard validator if they persist after the prompt change lands.
    """
    # Find every run of plain-card pattern, then expand each run into its
    # individual rank+suit tokens for the warning (so "KsKh" reports both
    # "Ks" and "Kh" rather than just the first match).
    runs = _PLAIN_CARD_RE.finditer(generated.answer_explanation)
    tokens: set[str] = set()
    for run in runs:
        for tok in _PLAIN_CARD_TOKEN_RE.findall(run.group(0)):
            tokens.add(tok)
    if tokens:
        unique = sorted(tokens)
        # Print to stderr so batch logs surface the warning without polluting
        # the validator's binary ok/fail contract.
        import sys
        print(f"  [soft-warn validate_no_plain_card_notation] "
              f"explanation contains plain card notation {unique} (voice "
              f"rule 9 requires suit emojis like K♠️). "
              f"Not retrying.", file=sys.stderr)
    return ValidationResult.ok()


# --- Ryan-feedback Fix 2b validators (Apr 2026) -----------------------------
# Threshold above which a Pio action is considered "in the mix" for option
# coverage purposes. Mirrors COMPLETENESS_MIN_FREQ above (also 0.05) but
# named separately so the two thresholds can drift if Ryan tunes them.
COMPOSITE_LABEL_MIN_FREQ = 0.05

_COMPOSITE_LABEL_RE = __import__("re").compile(
    r"^(?:Mostly|mostly)\s+(\w[\w-]*),\s*sometimes\s+(\w[\w-]*)\s*$"
)


def _meaningful_pio_verbs(decision: DecisionData,
                          min_freq: float = COMPOSITE_LABEL_MIN_FREQ) -> list[str]:
    """Pio verbs played at frequency >= min_freq, ordered by frequency desc."""
    strategy = decision.range_aggregate_strategy
    return [verb for verb, _ in
            sorted(strategy.items(), key=lambda kv: -kv[1])
            if strategy[verb] >= min_freq]


def validate_no_standalone_sometimes(generated: GeneratedExplanation,
                                     decision: DecisionData) -> ValidationResult:
    """No option may be a bare \"Sometimes X\" or \"Rarely X\" label.

    Per Ryan's Apr-2026 V6 review (Fix 2b (a)): standalone Sometimes/Rarely
    options are ambiguous to players ("Sometimes check" and "Sometimes bet"
    are not mutually exclusive readings of the strategy). The frequency
    prefix mapping has been collapsed to Always/Mostly; the LLM uses
    "sometimes" only as the secondary verb of a composite label like
    \"Mostly call, sometimes raise\".

    This validator fires regardless of how many Pio actions are in the mix
    -- a standalone Sometimes/Rarely label is wrong in any context.
    """
    offending: list[str] = []
    for slot, option in zip(("option_1", "option_2", "option_3", "option_4"),
                            (generated.option_1, generated.option_2,
                             generated.option_3, generated.option_4)):
        if not option:
            continue
        prefix = extract_frequency_prefix(option)
        if prefix in ("Sometimes", "Rarely"):
            # Allow composite "Mostly X, sometimes Y" to pass -- the prefix
            # we extracted is "Mostly", not "Sometimes". This branch only
            # fires when the LEADING word is Sometimes/Rarely.
            offending.append(f"{slot}={option!r}")
    if offending:
        return ValidationResult.fail(
            "standalone \"Sometimes X\" / \"Rarely X\" option labels are "
            "banned per Apr-2026 review. Use \"Mostly X\" for 2-action "
            "spots or composite \"Mostly X, sometimes Y\" labels for 3+ "
            "action spots. Offending options: " + "; ".join(offending)
        )
    return ValidationResult.ok()


def validate_composite_label_frequencies(generated: GeneratedExplanation,
                                         decision: DecisionData) -> ValidationResult:
    """Composite labels of form "Mostly X, sometimes Y" must reflect the
    actual Pio frequencies (X dominant over Y, both at >= 5%).

    Per Ryan's Apr-2026 V6 review (Fix 2b (b)): a composite label implies a
    specific strategic shape -- one verb is dominant, another is the
    secondary mix-in. If Pio's actual frequencies don't fit, the label
    misrepresents the data block.

    Specifically, for every option matching the literal "Mostly X,
    sometimes Y" pattern:

      * X must be a Pio-offered verb;
      * Y must be a Pio-offered verb;
      * Pio's freq(X) must be >= freq(Y) (otherwise X isn't dominant);
      * Pio's freq(Y) must be >= COMPOSITE_LABEL_MIN_FREQ (otherwise Y
        isn't "in the mix" at all and the LLM is fabricating a secondary).

    Options that don't match the composite pattern are skipped (they go
    through the other validators).
    """
    strategy = decision.range_aggregate_strategy
    if not strategy:
        return ValidationResult.ok()
    failures: list[str] = []
    for slot, option in zip(("option_1", "option_2", "option_3", "option_4"),
                            (generated.option_1, generated.option_2,
                             generated.option_3, generated.option_4)):
        if not option:
            continue
        match = _COMPOSITE_LABEL_RE.match(option)
        if not match:
            continue
        x_verb, y_verb = match.group(1).lower(), match.group(2).lower()
        x_freq = strategy.get(x_verb, 0.0)
        y_freq = strategy.get(y_verb, 0.0)
        if x_verb not in strategy:
            failures.append(
                f"{slot}={option!r}: composite-label dominant verb {x_verb!r} "
                f"is not in Pio's strategy {sorted(strategy)}")
            continue
        if y_verb not in strategy:
            failures.append(
                f"{slot}={option!r}: composite-label secondary verb {y_verb!r}"
                f" is not in Pio's strategy {sorted(strategy)}")
            continue
        if x_freq < y_freq:
            failures.append(
                f"{slot}={option!r}: composite label claims {x_verb!r} is "
                f"dominant over {y_verb!r}, but Pio plays {x_verb} at "
                f"{x_freq:.2%} < {y_verb} at {y_freq:.2%}")
            continue
        if y_freq < COMPOSITE_LABEL_MIN_FREQ:
            failures.append(
                f"{slot}={option!r}: secondary verb {y_verb!r} only at "
                f"{y_freq:.2%} (< {COMPOSITE_LABEL_MIN_FREQ:.0%}), which is "
                f"below the 'meaningfully mixed' threshold -- omit the "
                f"composite and use a plain \"Mostly {x_verb}\" or "
                f"\"Always {x_verb}\" instead")
    if failures:
        return ValidationResult.fail("; ".join(failures))
    return ValidationResult.ok()


# --- combined entry point used by Layer 6 retry loop ------------------------
def run_audit_validators(generated: GeneratedExplanation,
                         decision: DecisionData) -> ValidationResult:
    """Run all validators in order; return the first failure or ok.

    The retry loop in `explanation_generator.generate_explanation` calls this
    after the structural check (`_validate`) and uses the returned
    error_message as the corrective feedback for the LLM retry.
    """
    for check in (validate_option_set,
                  validate_correct_answer_verb,
                  validate_option_set_completeness,
                  validate_no_standalone_sometimes,
                  validate_composite_label_frequencies,
                  # Ryan-feedback Fix 5: hard validator; runs in the retry loop.
                  validate_archetype_consistency):
        result = check(generated, decision)
        if not result.is_valid:
            return result
    # Soft validators last: they may print warnings but never fail.
    validate_no_plain_card_notation(generated, decision)
    validate_villain_combo_citation(generated, decision)
    return ValidationResult.ok()


__all__ = [
    "COMPLETENESS_MIN_FREQ",
    "COMPOSITE_LABEL_MIN_FREQ",
    "ValidationResult",
    "extract_action_verb",
    "extract_frequency_prefix",
    "run_audit_validators",
    "validate_archetype_consistency",
    "validate_composite_label_frequencies",
    "validate_correct_answer_verb",
    "validate_no_plain_card_notation",
    "validate_no_standalone_sometimes",
    "validate_option_set",
    "validate_option_set_completeness",
    "validate_villain_combo_citation",
]
