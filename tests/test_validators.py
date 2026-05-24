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

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import GeneratedExplanation               # noqa: E402
from pipeline.fact_extractor.spot_data import DecisionData                    # noqa: E402
from pipeline.validators import (                                              # noqa: E402
    COMPLETENESS_MIN_FREQ, COMPOSITE_LABEL_MIN_FREQ, ValidationResult,
    extract_action_verb, extract_frequency_prefix, run_audit_validators,
    validate_archetype_consistency, validate_composite_label_frequencies,
    validate_correct_answer_verb, validate_no_plain_card_notation,
    validate_no_standalone_sometimes, validate_option_set,
    validate_option_set_completeness, validate_villain_combo_citation,
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
    """Pio freq 0.57 -> deterministic prefix 'Mostly' (post-Apr-2026
    bracket collapse). LLM emitted 'Always' -- prefix mismatch.

    Pre-Apr-2026 this case caught a 'Mostly' vs 'Sometimes' confusion;
    post-collapse the only remaining mismatch is Always-when-Mostly-required
    (or empty-when-Mostly-required for sub-5% actions, but those don't
    appear as correct_answer)."""
    decision = DecisionData(
        correct_action="check",
        range_aggregate_strategy={"check": 0.57, "bet": 0.43})
    explanation = _explanation(
        option_1="Always check", option_2="Mostly check",
        option_3="Mostly bet", option_4="Always bet",
        correct_answer="Always check")           # Always wrong; should be Mostly
    result = validate_correct_answer_verb(explanation, decision)
    assert not result.is_valid
    assert "Mostly" in result.error_message
    assert "Always" in result.error_message


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


# --- Ryan-feedback Fix 2b validators ----------------------------------------
def test_validate_no_standalone_sometimes_passes_on_all_mostly():
    """Pure Always/Mostly option set passes the standalone-Sometimes ban."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly fold", option_4="Always fold",
        correct_answer="Mostly call")
    assert validate_no_standalone_sometimes(explanation, decision).is_valid


def test_validate_no_standalone_sometimes_fails_on_sometimes_option():
    """A standalone 'Sometimes X' label is rejected regardless of how many
    Pio actions are in the mix -- Ryan-feedback Fix 2b (a)."""
    decision = DecisionData(
        correct_action="check",
        range_aggregate_strategy={"check": 0.55, "bet": 0.45})
    explanation = _explanation(
        option_1="Always check", option_2="Mostly check",
        option_3="Sometimes bet", option_4="Always bet",     # standalone Sometimes
        correct_answer="Mostly check")
    result = validate_no_standalone_sometimes(explanation, decision)
    assert not result.is_valid
    assert "Sometimes" in result.error_message
    assert "banned" in result.error_message.lower()


def test_validate_no_standalone_sometimes_allows_composite_label():
    """'Mostly X, sometimes Y' starts with 'Mostly' -- composite labels
    pass the standalone-Sometimes ban (the secondary 'sometimes' is
    embedded, not a prefix)."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.55, "fold": 0.25, "raise": 0.20})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call, sometimes fold",
        option_3="Mostly call, sometimes raise", option_4="Always raise",
        correct_answer="Mostly call, sometimes fold")
    assert validate_no_standalone_sometimes(explanation, decision).is_valid


def test_validate_no_standalone_sometimes_catches_rarely_too():
    """'Rarely X' is also banned -- same ambiguity argument as 'Sometimes'."""
    decision = DecisionData(
        correct_action="check",
        range_aggregate_strategy={"check": 0.85, "bet": 0.15})
    explanation = _explanation(
        option_1="Mostly check", option_2="Rarely bet",
        correct_answer="Mostly check")
    assert not validate_no_standalone_sometimes(explanation, decision).is_valid


def test_validate_composite_label_frequencies_passes_correct_pairing():
    """'Mostly call, sometimes raise' is valid when call > raise > 5%."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.55, "fold": 0.25, "raise": 0.20})
    explanation = _explanation(
        option_1="Always call",
        option_2="Mostly call, sometimes raise",
        option_3="Mostly call, sometimes fold",
        option_4="",
        correct_answer="Mostly call, sometimes raise")
    assert validate_composite_label_frequencies(explanation, decision).is_valid


def test_validate_composite_label_frequencies_fails_on_inverted_dominance():
    """'Mostly raise, sometimes call' is wrong when call > raise."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.60, "raise": 0.20, "fold": 0.20})
    explanation = _explanation(
        option_1="Always call",
        option_2="Mostly raise, sometimes call",          # inverted dominance
        option_3="Mostly call, sometimes fold",
        option_4="",
        correct_answer="Mostly call, sometimes fold")
    result = validate_composite_label_frequencies(explanation, decision)
    assert not result.is_valid
    assert "dominant" in result.error_message
    assert "20.00%" in result.error_message or "20%" in result.error_message


def test_validate_composite_label_frequencies_fails_on_phantom_secondary():
    """A composite citing a secondary verb Pio plays below 5% is wrong --
    the label fabricates a 'meaningfully mixed' shape that doesn't exist."""
    decision = DecisionData(
        correct_action="call",
        # raise plays 1% (below COMPOSITE_LABEL_MIN_FREQ = 5%)
        range_aggregate_strategy={"call": 0.70, "fold": 0.29, "raise": 0.01})
    explanation = _explanation(
        option_1="Always call",
        option_2="Mostly call, sometimes raise",          # raise too rare
        option_3="Mostly call, sometimes fold",
        option_4="",
        correct_answer="Mostly call, sometimes fold")
    result = validate_composite_label_frequencies(explanation, decision)
    assert not result.is_valid
    assert "1.00%" in result.error_message or "1%" in result.error_message
    assert "fabricat" in result.error_message.lower() \
        or "below" in result.error_message.lower()


def test_validate_composite_label_frequencies_fails_on_invented_verb():
    """A composite citing a verb Pio never offers is wrong."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    explanation = _explanation(
        option_1="Always call",
        option_2="Mostly call, sometimes raise",          # raise not offered
        option_3="",
        option_4="",
        correct_answer="Always call")
    result = validate_composite_label_frequencies(explanation, decision)
    assert not result.is_valid
    assert "'raise'" in result.error_message


def test_validate_composite_label_frequencies_ignores_non_composite_options():
    """Plain 'Always X' / 'Mostly X' labels are not composites and are
    untouched by this validator."""
    decision = DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.66, "fold": 0.34})
    explanation = _explanation(
        option_1="Always call", option_2="Mostly call",
        option_3="Mostly fold", option_4="Always fold",
        correct_answer="Mostly call")
    assert validate_composite_label_frequencies(explanation, decision).is_valid


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


# --- Ryan-feedback Fix 3 soft validator (May 2026) --------------------------
def test_no_plain_card_notation_passes_with_emoji_text(capsys):
    # Emoji-style citations are clean; no warning printed, returns ok.
    decision = DecisionData(correct_action="call",
                            range_aggregate_strategy={"call": 1.0})
    explanation = _explanation(
        option_1="Call", correct_answer="Call",
        answer_explanation="Villain shows up with K♠️K♣️ "
                            "and J♦️J♠️ for sets.")
    assert validate_no_plain_card_notation(explanation, decision).is_valid
    assert "soft-warn" not in capsys.readouterr().err


def test_no_plain_card_notation_warns_on_plain_form(caplog):
    # Plain "Kh"/"Ad"/"5c" notation triggers a logger.warning() but the
    # validator returns ok (soft validator, no rejection initially per Ryan's
    # instruction).
    decision = DecisionData(correct_action="call",
                            range_aggregate_strategy={"call": 1.0})
    explanation = _explanation(
        option_1="Call", correct_answer="Call",
        answer_explanation="BTN can show up with KsKh and JdJc here.")
    with caplog.at_level(logging.WARNING, logger="pipeline.validators"):
        result = validate_no_plain_card_notation(explanation, decision)
    assert result.is_valid                       # soft -- does not fail
    assert "soft-warn validate_no_plain_card_notation" in caplog.text
    # Each plain-card token appears in the warning.
    assert "Kh" in caplog.text and "Js" not in caplog.text


def test_no_plain_card_notation_ignores_non_card_text(capsys):
    # English words like "is", "as", "the", "ad" (as in "ad-hoc") should not
    # match. The regex requires a rank LETTER [2-9TJQKA] followed by a single
    # lowercase suit letter; "is"/"as"/"the" fail the rank prefix.
    decision = DecisionData(correct_action="call",
                            range_aggregate_strategy={"call": 1.0})
    explanation = _explanation(
        option_1="Call", correct_answer="Call",
        answer_explanation="It is clear that the call is the right play, as "
                            "your hand value is high and the pot is laid.")
    assert validate_no_plain_card_notation(explanation, decision).is_valid
    assert "soft-warn" not in capsys.readouterr().err


# --- Ryan-feedback Fix 4 soft validator (May 2026) --------------------------
def test_villain_combo_citation_passes_with_emoji_combo(capsys):
    # Explanation discusses villain AND cites at least one emoji combo -> ok,
    # no warning.
    decision = DecisionData(correct_action="check",
                            range_aggregate_strategy={"check": 0.7, "bet": 0.3})
    explanation = _explanation(
        option_1="Mostly check", correct_answer="Mostly check",
        answer_explanation="BTN shows up with K♠️K♣️ for sets here, plus "
                            "A♠️K♠️ for top two. Pot control is right.")
    assert validate_villain_combo_citation(explanation, decision).is_valid
    assert "soft-warn" not in capsys.readouterr().err


def test_villain_combo_citation_passes_with_hand_class_phrase(capsys):
    # Explanation discusses villain and uses a hand-class phrase ("two pair",
    # "set", "overpair") -> ok, no warning. Combo-emoji is not required when
    # a clear class is named.
    decision = DecisionData(correct_action="call",
                            range_aggregate_strategy={"call": 1.0})
    explanation = _explanation(
        option_1="Call", correct_answer="Call",
        answer_explanation="BTN's range is heavy on sets and top two pair "
                            "here. Calling collects value from them.")
    assert validate_villain_combo_citation(explanation, decision).is_valid
    assert "soft-warn" not in capsys.readouterr().err


def test_villain_combo_citation_warns_on_abstract_villain_talk(caplog):
    # Discusses villain but cites no specific combo OR hand-class -> soft warn.
    # Validator still returns ok (soft per Ryan, doesn't reject the explanation).
    decision = DecisionData(correct_action="call",
                            range_aggregate_strategy={"call": 1.0})
    explanation = _explanation(
        option_1="Call", correct_answer="Call",
        answer_explanation="BTN has value hands and bluffs in this spot. "
                            "Calling realises equity against their range.")
    with caplog.at_level(logging.WARNING, logger="pipeline.validators"):
        result = validate_villain_combo_citation(explanation, decision)
    assert result.is_valid                       # soft -- does not fail
    assert "soft-warn validate_villain_combo_citation" in caplog.text


def test_villain_combo_citation_silent_when_villain_not_discussed(capsys):
    # Explanation doesn't reference villain at all -> no requirement, no warn.
    decision = DecisionData(correct_action="check",
                            range_aggregate_strategy={"check": 1.0})
    explanation = _explanation(
        option_1="Check", correct_answer="Check",
        answer_explanation="Your hand has showdown value and the pot is small. "
                            "Check it down.")
    assert validate_villain_combo_citation(explanation, decision).is_valid
    assert "soft-warn" not in capsys.readouterr().err


# --- Ryan-feedback Fix 5 hard validator (May 2026) --------------------------
def test_archetype_consistency_passes_when_frame_matches():
    # trap_check spot framed correctly (inducing villain to bet) -> ok.
    decision = DecisionData(correct_action="check",
                            range_aggregate_strategy={"check": 0.7, "bet": 0.3})
    decision.recommended_action_archetype = "trap_check"
    explanation = _explanation(
        option_1="Mostly check", correct_answer="Mostly check",
        answer_explanation="Check here. Your set is too strong to fold villain "
                            "out by betting. Letting BB barrel river turns "
                            "their bluffs into chips.")
    assert validate_archetype_consistency(explanation, decision).is_valid


def test_archetype_consistency_rejects_v71_trap_check_failure():
    # The exact V7.1 failure: trap_check spot but explanation says "villain
    # has the nut advantage" -- wrong strategic frame. Must reject.
    decision = DecisionData(correct_action="check",
                            range_aggregate_strategy={"check": 0.7, "bet": 0.3})
    decision.recommended_action_archetype = "trap_check"
    explanation = _explanation(
        option_1="Mostly check", correct_answer="Mostly check",
        answer_explanation="Check here because villain has the nut advantage "
                            "on this board and your range is weaker.")
    result = validate_archetype_consistency(explanation, decision)
    assert not result.is_valid
    assert "trap_check" in result.error_message
    assert "anti-pattern" in result.error_message


def test_archetype_consistency_rejects_bluff_catch_framed_as_value():
    decision = DecisionData(correct_action="call",
                            range_aggregate_strategy={"call": 0.55, "fold": 0.45})
    decision.recommended_action_archetype = "bluff_catch"
    explanation = _explanation(
        option_1="Mostly call", correct_answer="Mostly call",
        answer_explanation="Call because you bet for value with second pair "
                            "and villain's range has air.")
    result = validate_archetype_consistency(explanation, decision)
    assert not result.is_valid
    assert "bluff_catch" in result.error_message


def test_archetype_consistency_skips_when_archetype_not_set():
    # Legacy / test SpotData without an archetype: no check, always ok.
    decision = DecisionData(correct_action="call",
                            range_aggregate_strategy={"call": 1.0})
    # recommended_action_archetype defaults to "".
    explanation = _explanation(
        option_1="Call", correct_answer="Call",
        answer_explanation="Villain has the nut advantage here.")
    # Would fail if archetype were set, but isn't -> ok.
    assert validate_archetype_consistency(explanation, decision).is_valid


def test_run_audit_validators_includes_archetype_check():
    """The Layer 6 retry loop must catch a trap_check anti-pattern -- so the
    full run_audit_validators returns the failure (not just individual call).
    Option set covers both verbs so the earlier completeness validator passes,
    leaving archetype_consistency as the failing check.
    """
    decision = DecisionData(
        correct_action="check",
        range_aggregate_strategy={"check": 0.7, "bet": 0.3})
    decision.recommended_action_archetype = "trap_check"
    explanation = _explanation(
        option_1="Always check", option_2="Mostly check",
        option_3="Mostly bet", option_4="Always bet",
        correct_answer="Mostly check",
        answer_explanation="Check because villain has the nut advantage.")
    result = run_audit_validators(explanation, decision)
    assert not result.is_valid
    assert "trap_check" in result.error_message


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
