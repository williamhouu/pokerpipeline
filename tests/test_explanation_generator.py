"""Tests for pipeline.explanation_generator (Layer 6).

These tests never hit the real Anthropic API -- the client is always a mock.
Coverage:

  * prompt assembly: system prompt embeds all 8 voice rules and the banned-
    phrase list; the user prompt carries gold examples, framing, and the
    SOLVER DATA block;
  * parser: extracts the six fields from a clean JSON response; tolerates
    accidental code fences and leading prose;
  * validation: catches a `correct_answer` that does not match any option;
  * retry: the first call returns a bad payload, the second a good one --
    the wrapper must return the good one;
  * option-style detection from real SpotData shapes (binary / frequency /
    sizing) and the SpotData -> CSV-override integration.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import (                             # noqa: E402
    BANNED_LITERAL_PHRASES, DEFAULT_MODEL, ExplanationValidationError,
    GeneratedExplanation, VOICE_RULES, _detect_option_style,
    _expected_correct_prefix, _extract_text, _top_two_verbs,
    build_system_prompt, build_user_prompt, explanation_to_row_overrides,
    frequency_to_verb_prefix, generate_explanation, parse_response,
)
from pipeline.fact_extractor.spot_data import (                          # noqa: E402
    BoardTexture, DecisionData, HandClass, SpotData, SpotMetadata,
)

# --- fixtures ---------------------------------------------------------------
_GOLD_STUB = [
    {"Hand Stage": "flop", "Question": "Flop spot stem.",
     "option 1": "Call", "option 2": "Fold", "option 3": "", "option 4": "",
     "Correct Answer": "Call", "Answer Explanation": "You should call.",
     "Preflop Pot Type": "", "Pot Participant": "", "Stack Depth": ""},
    {"Hand Stage": "turn", "Question": "Turn spot stem.",
     "option 1": "Always check", "option 2": "Mostly check",
     "option 3": "Mostly bet", "option 4": "Always bet",
     "Correct Answer": "Always check",
     "Answer Explanation": "AQ should always be checking in this spot.",
     "Preflop Pot Type": "", "Pot Participant": "", "Stack Depth": ""},
]


def _binary_spot() -> SpotData:
    """A flop spot with a clearly-dominant action -- binary-action style."""
    return SpotData(
        SpotMetadata("flop", hero_position="BB", villain_position="BTN",
                     position_dynamic="BB_vs_BTN", game_format="cash",
                     preflop_raise_count=1, hero_cards=("Ah", "Kh"),
                     board=["2c", "Js", "7s"]),
        decision_data=DecisionData(
            options=["call", "fold"], correct_action="call", ev_gap_bb=1.4,
            range_aggregate_strategy={"call": 0.88, "fold": 0.12},
            range_mean_evs_per_action={"call": 1.2, "fold": -0.2}),
        hand_class=HandClass("top_pair_top_kicker", strength_bucket="strong",
                             label="top_pair_top_kicker_no_draws"),
        board_texture=BoardTexture("two_tone", "unpaired", "disconnected",
                                   "middling", "semi_wet"),
        concept_tags=["bluffcatch_spot", "range_advantage_hero"],
    )


def _frequency_spot() -> SpotData:
    """A flop spot with a 60/40 mixed strategy -- frequency style."""
    spot = _binary_spot()
    spot.decision_data = DecisionData(
        options=["check", "bet 25"], correct_action="check", ev_gap_bb=0.6,
        range_aggregate_strategy={"check": 0.60, "bet": 0.40},
        option_pot_fractions={"bet": 0.33})
    return spot


def _sizing_spot() -> SpotData:
    """A flop spot with two bet sizes available -- sizing style."""
    spot = _binary_spot()
    spot.decision_data = DecisionData(
        options=["bet 10", "bet 30"], correct_action="bet", ev_gap_bb=0.7,
        range_aggregate_strategy={"bet": 0.95},
        option_pot_fractions={"bet 10": 0.33, "bet 30": 1.35})
    return spot


def _mock_client(responses):
    """A fake Anthropic client whose messages.create returns each response in turn.

    `responses` is a list of strings (the assistant's text); each call pops the
    next one. The mock records every call's (model, system, messages) so tests
    can assert on the prompt the wrapper actually sent.
    """
    calls = []
    queue = list(responses)

    def create(*, model, max_tokens, system, messages, temperature=None, **_extra):
        # temperature kwarg is conditionally omitted by call_messages_create
        # when the model is in MODELS_WITHOUT_TEMPERATURE (Opus 4.x); the
        # default of None lets the mock accept either call shape.
        # **_extra defensively swallows any future per-model kwargs.
        calls.append({"model": model, "system": system, "messages": messages,
                      "temperature": temperature})
        text = queue.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    client = SimpleNamespace(messages=SimpleNamespace(create=create), _calls=calls)
    return client


# --- (a) prompt assembly carries all 8 voice rules + banned phrases ---------
def test_system_prompt_includes_every_voice_rule():
    system = build_system_prompt()
    # Rule 9 (suit-emoji citation) added per Ryan-feedback Fix 3, May 2026;
    # Rule 10 (villain combo citation) added per Fix 4, same review.
    assert len(VOICE_RULES) == 10
    for rule in VOICE_RULES:
        # Rules are long; assert on the leading clause so a future word-tweak
        # doesn't break the test.
        assert rule.split(".")[0] in system, rule[:40]


def test_system_prompt_includes_banned_phrases_and_schema():
    system = build_system_prompt()
    for phrase in BANNED_LITERAL_PHRASES:
        assert phrase in system, phrase
    # Output schema -- the six keys must appear so the LLM knows what to emit.
    for key in ("option_1", "option_2", "option_3", "option_4",
                "correct_answer", "answer_explanation"):
        assert key in system, key
    # Sanity: no em dash leaks into the system prompt itself.
    assert "; " not in system or "semicolon" in system          # only meta-mention OK


def test_user_prompt_has_gold_examples_solver_data_and_framing():
    spot = _binary_spot()
    prompt = build_user_prompt(spot, _GOLD_STUB, style="binary_action")
    # Gold-example block.
    assert "GOLD EXAMPLE 1" in prompt
    assert "Flop spot stem." in prompt
    assert "AQ should always be checking" in prompt
    # Solver-data block: the JSON dump must carry the correct action and tags.
    assert "SOLVER DATA" in prompt
    assert "bluffcatch_spot" in prompt
    assert "\"correct_action\": \"call\"" in prompt
    # Framing.
    assert "BB" in prompt and "BTN" in prompt
    # Option-style instruction is included.
    assert "OPTION STYLE" in prompt


# --- (b) parser extracts the 6 fields ---------------------------------------
def test_parse_clean_json_response():
    response = (
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Call", '
        '"answer_explanation": "Top pair plays for value here."}'
    )
    explanation = parse_response(response)
    assert isinstance(explanation, GeneratedExplanation)
    assert explanation.option_1 == "Call"
    assert explanation.option_3 == "" and explanation.option_4 == ""
    assert explanation.correct_answer == "Call"
    assert "value" in explanation.answer_explanation


def test_parse_strips_code_fences_and_leading_prose():
    response = (
        "Sure, here's the JSON:\n"
        "```json\n"
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Call", '
        '"answer_explanation": "ok."}\n'
        "```"
    )
    explanation = parse_response(response)
    assert explanation.option_1 == "Call"
    assert explanation.correct_answer == "Call"


def test_parse_rejects_missing_keys():
    response = '{"option_1": "Call", "correct_answer": "Call"}'
    try:
        parse_response(response)
    except ExplanationValidationError as exc:
        assert "missing keys" in str(exc)
    else:
        raise AssertionError("expected ExplanationValidationError")


# --- (c) validation catches a mismatched correct_answer ---------------------
def test_validation_catches_mismatched_correct_answer():
    bad = (
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Raise", '
        '"answer_explanation": "x."}'
    )
    # parse_response itself does NOT enforce equality -- it's the caller's job
    # (so we can retry with a corrective message). Verify generate_explanation
    # surfaces the mismatch as an ExplanationValidationError when both attempts
    # fail.
    client = _mock_client([bad, bad])
    try:
        generate_explanation(_binary_spot(), client=client, gold_examples=_GOLD_STUB)
    except ExplanationValidationError as exc:
        assert "Raise" in str(exc)
        assert "Call" in str(exc) or "Fold" in str(exc)
    else:
        raise AssertionError("expected ExplanationValidationError after retry")
    assert len(client._calls) == 2          # one initial + one retry


# --- (d) retry logic: first call bad, second call good ----------------------
def test_retry_recovers_after_one_failure():
    bad = (
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Raise", '
        '"answer_explanation": "wrong action label."}'
    )
    good = (
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Call", '
        '"answer_explanation": "Top pair plays for value here."}'
    )
    client = _mock_client([bad, good])
    explanation = generate_explanation(_binary_spot(), client=client,
                                       gold_examples=_GOLD_STUB)
    assert explanation.correct_answer == "Call"
    assert len(client._calls) == 2
    # The retry call's message list must carry a corrective user turn.
    retry_messages = client._calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "failed validation" in retry_messages[-1]["content"]


def test_audit_validators_trigger_retry_on_row1_defect():
    """Layer 7 wiring: a Row-1-shape defect (LLM drops Pio's #2 action from
    the option set) is caught by validate_option_set_completeness on attempt
    1, the wrapper retries with corrective feedback, attempt 2 ships clean."""
    # A frequency-style spot where Pio plays call(66) / fold(34) / raise(0.001).
    # Same shape as v3 Row 1.
    spot = _binary_spot()
    spot.decision_data = DecisionData(
        options=["call", "fold", "raise 608"], correct_action="call",
        ev_gap_bb=2.83,
        range_aggregate_strategy={"call": 0.6557, "fold": 0.3435,
                                  "raise": 0.0008})

    # Attempt 1: Row-1 defect -- fold absent from options.
    bad = (
        '{"option_1": "Always call", "option_2": "Mostly call", '
        '"option_3": "Mostly raise", "option_4": "Always raise", '
        '"correct_answer": "Mostly call", '
        '"answer_explanation": "Trap the check-raise with the set."}'
    )
    # Attempt 2: corrected -- fold present.
    good = (
        '{"option_1": "Always call", "option_2": "Mostly call", '
        '"option_3": "Mostly fold", "option_4": "Always fold", '
        '"correct_answer": "Mostly call", '
        '"answer_explanation": "Trap the check-raise with the set."}'
    )
    client = _mock_client([bad, good])
    explanation = generate_explanation(spot, client=client,
                                       gold_examples=_GOLD_STUB)
    assert explanation.correct_answer == "Mostly call"
    assert explanation.option_3 == "Mostly fold"
    assert len(client._calls) == 2
    # The corrective retry message names the missing action.
    retry_msg = client._calls[1]["messages"][-1]["content"]
    assert "fold" in retry_msg.lower()


def test_audit_validators_raise_after_two_failures():
    """If both attempts emit the same Row-1 defect, ExplanationValidationError."""
    spot = _binary_spot()
    spot.decision_data = DecisionData(
        options=["call", "fold", "raise 608"], correct_action="call",
        ev_gap_bb=2.83,
        range_aggregate_strategy={"call": 0.6557, "fold": 0.3435,
                                  "raise": 0.0008})
    bad = (
        '{"option_1": "Always call", "option_2": "Mostly call", '
        '"option_3": "Mostly raise", "option_4": "Always raise", '
        '"correct_answer": "Mostly call", '
        '"answer_explanation": "ok."}'
    )
    client = _mock_client([bad, bad])
    try:
        generate_explanation(spot, client=client, gold_examples=_GOLD_STUB)
    except ExplanationValidationError as exc:
        assert "fold" in str(exc).lower()
        assert "human review" in str(exc)
        return
    raise AssertionError("expected ExplanationValidationError after both retries failed")


def test_no_retry_when_first_response_is_good():
    good = (
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Call", '
        '"answer_explanation": "ok."}'
    )
    client = _mock_client([good])
    explanation = generate_explanation(_binary_spot(), client=client,
                                       gold_examples=_GOLD_STUB)
    assert explanation.correct_answer == "Call"
    assert len(client._calls) == 1


# --- (e) integration with SpotData + option-style detection -----------------
def test_option_style_detection_binary():
    assert _detect_option_style(_binary_spot()) == "binary_action"


def test_option_style_detection_frequency():
    assert _detect_option_style(_frequency_spot()) == "frequency"


def test_option_style_detection_sizing():
    assert _detect_option_style(_sizing_spot()) == "sizing"


def test_style_instruction_appears_in_prompt():
    sizing_prompt = build_user_prompt(_sizing_spot(), _GOLD_STUB, "sizing")
    assert "sizing" in sizing_prompt.lower()
    assert "% pot" in sizing_prompt

    freq_prompt = build_user_prompt(_frequency_spot(), _GOLD_STUB, "frequency")
    assert "Always" in freq_prompt and "Mostly" in freq_prompt

    binary_prompt = build_user_prompt(_binary_spot(), _GOLD_STUB, "binary_action")
    assert "Call" in binary_prompt and "single verb" in binary_prompt


def test_default_model_passed_to_client():
    good = (
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Call", '
        '"answer_explanation": "ok."}'
    )
    client = _mock_client([good])
    generate_explanation(_binary_spot(), client=client,
                        gold_examples=_GOLD_STUB)
    assert client._calls[0]["model"] == DEFAULT_MODEL


def test_frequency_to_verb_prefix_brackets():
    """Brackets are inclusive at the lower bound. Pre-Apr-2026 there were
    four prefixes (Always/Mostly/Sometimes/Rarely); Ryan-feedback Fix 2
    collapsed the lower three into a single Mostly so standalone
    \"Sometimes X\" options never appear."""
    # Always: 0.95 and above.
    assert frequency_to_verb_prefix(1.0) == "Always"
    assert frequency_to_verb_prefix(0.95) == "Always"
    # Mostly: everything in [0.05, 0.95). Was Always/Mostly/Sometimes/Rarely
    # before; now collapsed.
    assert frequency_to_verb_prefix(0.9499) == "Mostly"
    assert frequency_to_verb_prefix(0.66) == "Mostly"
    assert frequency_to_verb_prefix(0.60) == "Mostly"
    assert frequency_to_verb_prefix(0.59) == "Mostly"
    assert frequency_to_verb_prefix(0.57) == "Mostly"        # v3 Row 8's freq
    assert frequency_to_verb_prefix(0.20) == "Mostly"
    assert frequency_to_verb_prefix(0.19) == "Mostly"
    assert frequency_to_verb_prefix(0.05) == "Mostly"
    # Below 0.05: empty (action essentially not played).
    assert frequency_to_verb_prefix(0.04) == ""
    assert frequency_to_verb_prefix(0.0008) == ""            # v3 Row 1's raise


def test_expected_correct_prefix_only_for_frequency_style():
    """The deterministic prefix applies only to frequency-style spots --
    binary_action and sizing styles use bare verbs / sizing labels."""
    # frequency style: top freq < 0.80, strategy is mixed.
    freq_spot = SpotData(
        SpotMetadata("flop"),
        decision_data=DecisionData(
            range_aggregate_strategy={"call": 0.66, "fold": 0.34}))
    assert _expected_correct_prefix(freq_spot) == "Mostly"
    # binary_action style: top freq >= 0.80 -- bare verbs, no prefix.
    binary_spot = SpotData(
        SpotMetadata("flop"),
        decision_data=DecisionData(
            range_aggregate_strategy={"call": 0.94, "fold": 0.06}))
    assert _expected_correct_prefix(binary_spot) is None


def test_top_two_verbs_orders_by_frequency():
    """_top_two_verbs returns (dominant, second) so the prompt can pin
    which two actions must appear in the option set."""
    spot = SpotData(
        SpotMetadata("flop"),
        decision_data=DecisionData(
            range_aggregate_strategy={"call": 0.66, "fold": 0.34, "raise": 0.001}))
    assert _top_two_verbs(spot) == ("call", "fold")
    # When fewer than two actions are recorded, return None.
    no_strategy = SpotData(SpotMetadata("flop"),
                           decision_data=DecisionData(
                               range_aggregate_strategy={"call": 1.0}))
    assert _top_two_verbs(no_strategy) is None


def test_frequency_style_instruction_pins_prefix_and_verbs():
    """The frequency-style instruction now hard-pins the prefix Python
    chose and names the top-two verbs the option set must cover."""
    from pipeline.explanation_generator import _option_style_instruction
    spot = SpotData(
        SpotMetadata("flop"),
        decision_data=DecisionData(
            range_aggregate_strategy={"call": 0.66, "fold": 0.34, "raise": 0.001}))
    instruction = _option_style_instruction("frequency", spot)
    assert "HARD CONSTRAINT" in instruction
    assert "'Mostly'" in instruction or "\"Mostly\"" in instruction
    # Both top verbs explicitly named.
    assert "'call'" in instruction
    assert "'fold'" in instruction
    # Raise (freq=0.001) is NOT named -- only top-two.
    assert "'raise'" not in instruction


def test_explanation_to_row_overrides_maps_six_csv_columns():
    explanation = GeneratedExplanation(
        option_1="Call", option_2="Fold", option_3="", option_4="",
        correct_answer="Call", answer_explanation="ok.")
    overrides = explanation_to_row_overrides(explanation)
    assert overrides == {
        "option 1": "Call", "option 2": "Fold",
        "option 3": "", "option 4": "",
        "Correct Answer": "Call", "Answer Explanation": "ok.",
    }


def test_extract_text_handles_sdk_content_blocks_and_strings():
    # Raw string fallback (handy for ad-hoc mocking).
    assert _extract_text("hello") == "hello"
    # SDK shape: a content list of objects with .text.
    obj = SimpleNamespace(content=[SimpleNamespace(text="hi")])
    assert _extract_text(obj) == "hi"
    # Dict shape: harmless extra coverage.
    obj = SimpleNamespace(content=[{"text": "yo"}])
    assert _extract_text(obj) == "yo"


def test_caching_marks_on_system_and_gold_block():
    """Both the system prompt and the gold-examples block carry cache_control,
    so prompt caching hits across thousands of generations."""
    good = (
        '{"option_1": "Call", "option_2": "Fold", "option_3": "", '
        '"option_4": "", "correct_answer": "Call", '
        '"answer_explanation": "ok."}'
    )
    client = _mock_client([good])
    generate_explanation(_binary_spot(), client=client,
                        gold_examples=_GOLD_STUB)
    call = client._calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    # First content block of the user message is the cacheable gold-example block.
    first_user_block = call["messages"][0]["content"][0]
    assert first_user_block["cache_control"] == {"type": "ephemeral"}
    assert "GOLD EXAMPLE 1" in first_user_block["text"]


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
        except Exception as exc:
            failed += 1
            print(f"  [ERROR] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
