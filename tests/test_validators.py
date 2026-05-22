"""Tests for pipeline.validators (Layer 7 initial set).

Run directly (`python tests/test_validators.py`) or under pytest. Pure unit
tests -- no PioSolver, no Anthropic API. Coverage:

  * extract_action_verb on every template shape;
  * extract_frequency_prefix;
  * each of the three validators with positive AND negative cases;
  * Row 1 and Row 18 canonical regressions (synthetic SpotData/Explanation
    fixtures that reproduce the audit's caught defects);
  * run_audit_validators short-circuits on the first failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import GeneratedExplanation               # noqa: E402
from pipeline.fact_extractor.spot_data import DecisionData                    # noqa: E402
from pipeline.validators import (                                              # noqa: E402
    COMPLETENESS_MIN_FREQ, ValidationResult, extract_action_verb,
    extract_frequency_prefix, run_audit_validators,
    validate_correct_answer_verb, validate_option_set,
    validate_option_set_completeness,
)


def _explanation(option_1="", option_2="", option_3="", option_4="",
                 correct_answer="", answer_explanation="ok.") -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1=option_1, option_2=option_2, option_3=option_3,
        option_4=option_4, correct_answer=correct_answer,
        answer_explanation=answer_explanation)


# --- helpers ----------------------------------------------------------------
def test_extract_action_verb_handles_all_template_shapes():
    # Binary action.
    assert extract_action_verb("Call") == "call"
    assert extract_action_verb("Fold") == "fold"
    assert extract_action_verb("Check") == "check"
    assert extract_action_verb("Bet") == "bet"
    assert extract_action_verb("Raise") == "raise"
    # Frequency-prefixed.
    assert extract_action_verb("Always call") == "call"
    assert extract_action_verb("Mostly fold") == "fold"
    assert extract_action_verb("Sometimes check") == "check"
    assert extract_action_verb("Rarely raise") == "raise"
    # Compound: primary verb (after the prefix) wins.
    assert extract_action_verb("Mostly fold, sometimes call") == "fold"
    # Sizing labels return None.
    assert extract_action_verb("33% pot") is None
    assert extract_action_verb("$1.25") is None
    # Empty / unparseable.
    assert extract_action_verb("") is None
    assert extract_action_verb("???") is None
    # Case-insensitive input handled.
    assert extract_action_verb("MOSTLY CALL") == "call"


def test_extract_frequency_prefix():
    assert extract_frequency_prefix("Always call") == "Always"
    assert extract_frequency_prefix("Mostly fold") == "Mostly"
    assert extract_frequency_prefix("Sometimes check") == "Sometimes"
    assert extract_frequency_prefix("Rarely raise") == "Rarely"
    # Binary action / sizing return None.
    assert extract_frequency_prefix("Call") is None
    assert extract_frequency_prefix("33% pot") is None
    assert extract_frequency_prefix("") is None


# --- validate_option_set ----------------------------------------------------
def test_validate_option_set_passes_when_verbs_are_in_pio_strategy():
    decision = DecisionData(
        range_aggregate_strategy={"call": 0.6557, "fold": 0.3435,
                                  "raise": 0.0008})
    # Even raise (freq 0.0008) is in Pio's strategy -- the set check accepts it.
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly raise", option_4="Always raise",
        correct_answer="Mostly call")
    assert validate_option_set(explanation, decision).is_valid


def test_validate_option_set_fails_on_invented_action():
    """If the LLM proposes an action Pio never offers, fail."""
    decision = DecisionData(
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly bet", option_4="Always bet",     # 'bet' invented
        correct_answer="Mostly call")
    result = validate_option_set(explanation, decision)
    assert not result.is_valid
    assert "bet" in result.error_message
    assert "['call', 'fold']" in result.error_message


def test_validate_option_set_skips_sizing_only_options():
    """Sizing labels like '33% pot' have no verb; skip them."""
    decision = DecisionData(
        range_aggregate_strategy={"bet": 0.7, "check": 0.3})
    explanation = _explanation(
        option_1="33% pot", option_2="75% pot",
        option_3="Check", option_4="",
        correct_answer="75% pot")
    # 'check' verb is in Pio; sizing options skipped.
    assert validate_option_set(explanation, decision).is_valid


# --- validate_correct_answer_verb -------------------------------------------
def test_validate_correct_answer_verb_matches_decision_action():
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly fold", option_4="Always fold",
        correct_answer="Mostly call")
    assert validate_correct_answer_verb(explanation, decision).is_valid


def test_validate_correct_answer_verb_fails_on_verb_mismatch():
    """LLM picked 'Mostly fold' when Python said correct_action='call'."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly fold", option_4="Always fold",
        correct_answer="Mostly fold")
    result = validate_correct_answer_verb(explanation, decision)
    assert not result.is_valid
    assert "'fold'" in result.error_message
    assert "'call'" in result.error_message


def test_validate_correct_answer_verb_fails_on_prefix_mismatch():
    """Pio freq 0.57 -> deterministic prefix 'Sometimes'. LLM emitted 'Mostly'."""
    decision = DecisionData(
        correct_action="check",
        range_aggregate_strategy={"check": 0.57, "bet": 0.43})
    explanation = _explanation(
        option_1="Always check", option_2="Mostly check",
        option_3="Mostly bet", option_4="Always bet",
        correct_answer="Mostly check")           # Mostly is wrong; should be Sometimes
    result = validate_correct_answer_verb(explanation, decision)
    assert not result.is_valid
    assert "Sometimes" in result.error_message
    assert "Mostly" in result.error_message


def test_validate_correct_answer_verb_skips_prefix_for_binary_style():
    """Top freq >= 0.80 means binary_action style: no prefix in correct_answer."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.94, "fold": 0.06})
    explanation = _explanation(
        option_1="Call", option_2="Fold",
        correct_answer="Call")
    assert validate_correct_answer_verb(explanation, decision).is_valid


def test_validate_correct_answer_verb_skips_when_no_correct_action():
    """No Pio-derived expected verb -> nothing to check."""
    decision = DecisionData()         # correct_action defaults to ""
    explanation = _explanation(option_1="Call", correct_answer="Call")
    assert validate_correct_answer_verb(explanation, decision).is_valid


# --- validate_option_set_completeness ---------------------------------------
def test_validate_option_set_completeness_passes_when_top_two_covered():
    decision = DecisionData(
        range_aggregate_strategy={"call": 0.66, "fold": 0.34, "raise": 0.0008})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly fold", option_4="Always fold",
        correct_answer="Mostly call")
    assert validate_option_set_completeness(explanation, decision).is_valid


def test_validate_option_set_completeness_ignores_tiny_mixin_actions():
    """A 0.08% action (freq < COMPLETENESS_MIN_FREQ) need not appear."""
    decision = DecisionData(
        range_aggregate_strategy={"call": 0.6557, "fold": 0.3435,
                                  "raise": 0.0008})
    # raise omitted -- ok, it's a tiny mix-in.
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly fold", option_4="Always fold",
        correct_answer="Mostly call")
    assert validate_option_set_completeness(explanation, decision).is_valid


def test_row1_regression_dropped_fold_is_caught():
    """v3 Row 1: Pio plays call(66%)/fold(34%)/raise(0.08%). LLM dropped fold
    from the options entirely and templated against raise instead. The
    completeness validator must catch this."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.6557, "fold": 0.3435,
                                  "raise": 0.0008})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly raise", option_4="Always raise",         # fold missing
        correct_answer="Mostly call")
    result = validate_option_set_completeness(explanation, decision)
    assert not result.is_valid
    assert "'fold'" in result.error_message
    assert "34.35%" in result.error_message       # Pio freq cited in message


def test_completeness_threshold_constant_is_documented():
    """The COMPLETENESS_MIN_FREQ knob is the audit-derived value; lock it
    so future tuning has to be deliberate."""
    assert COMPLETENESS_MIN_FREQ == 0.10


# --- Row 18 regression: field rename makes the prompt unambiguous -----------
def test_row18_regression_field_rename_propagated():
    """v3 Row 18: LLM cited range_mean EVs as 'Ks3s specifically' -- caused
    by the misleading old field name `hero_combo_evs`. Commit #1 renamed the
    field to range_mean_evs_per_action; this test locks the rename in (so the
    LLM's SOLVER DATA block contains the explicit name, not the old one)."""
    decision = DecisionData(
        range_mean_evs_per_action={"bet": 26.78, "check": 17.18})
    assert decision.range_mean_evs_per_action["bet"] == 26.78
    # Old field name must no longer be a writable attribute.
    assert not hasattr(decision, "hero_combo_evs")
    # The explanation_generator's prompt-construction step in commit #2 adds
    # an explicit AGGREGATE-vs-PER-COMBO note in the system prompt; verify
    # it's there so the Row-18 attribution defect can't recur silently.
    from pipeline.explanation_generator import build_system_prompt
    prompt = build_system_prompt()
    assert "range_mean_evs_per_action" in prompt
    assert "Never attribute these numbers to a specific combo" in prompt


# --- run_audit_validators short-circuits ------------------------------------
def test_run_audit_validators_short_circuits_on_first_failure():
    """When option_set fails, we don't compute the other two -- the LLM gets
    one clear error message at a time."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    # Both invented action ('bet') AND verb mismatch ('Mostly bet' for correct).
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly bet", option_4="Always bet",     # bet invented
        correct_answer="Mostly bet")                       # verb mismatch
    result = run_audit_validators(explanation, decision)
    assert not result.is_valid
    # The first failure (option_set) wins.
    assert "not in Pio" in result.error_message


def test_run_audit_validators_passes_when_all_valid():
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly fold", option_4="Always fold",
        correct_answer="Mostly call")
    assert run_audit_validators(explanation, decision).is_valid


if __name__ == "__main__":
    suite = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in suite:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
