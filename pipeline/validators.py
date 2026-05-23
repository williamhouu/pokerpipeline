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
                  validate_composite_label_frequencies):
        result = check(generated, decision)
        if not result.is_valid:
            return result
    return ValidationResult.ok()


__all__ = [
    "COMPLETENESS_MIN_FREQ",
    "COMPOSITE_LABEL_MIN_FREQ",
    "ValidationResult",
    "extract_action_verb",
    "extract_frequency_prefix",
    "run_audit_validators",
    "validate_composite_label_frequencies",
    "validate_correct_answer_verb",
    "validate_no_standalone_sometimes",
    "validate_option_set",
    "validate_option_set_completeness",
]
