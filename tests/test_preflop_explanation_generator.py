"""Tests for pipeline.preflop.explanation_generator (Layer 6, preflop).

Sibling of ``tests/test_explanation_generator.py``. These tests never hit
the real Anthropic API -- the client is always a mock. Coverage:

  * prompt assembly: system prompt embeds the 10 preflop voice rules,
    the preflop archetype catalog, and the banned-phrase list. The user
    prompt carries gold examples, framing, and the SOLVER DATA block.
  * option-style detection from real PreflopFacts shapes (binary /
    frequency).
  * parser + retry behaviour (delegates to the shared parser; the
    delegation surface is what's tested).
  * cache markers on system + gold-example blocks.

Notes on fixtures: PreflopFacts is built directly via the dataclass
constructors rather than running the full extract_facts pipeline, so
the tests don't need real range files on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import (  # noqa: E402
    BANNED_LITERAL_PHRASES,
    DEFAULT_MODEL,
    ExplanationValidationError,
    GeneratedExplanation,
)
from pipeline.preflop.explanation_generator import (  # noqa: E402
    _PROMPT_OVERRIDE_PATH,
    PREFLOP_ARCHETYPE_GUIDANCE,
    VOICE_RULES_PREFLOP,
    _detect_option_style_preflop,
    _expected_correct_prefix_preflop,
    _extract_text,
    _normalize_prose,
    _option_style_instruction_preflop,
    _question_framing_preflop,
    _trim_facts_for_prompt,
    _validate,
    build_preflop_system_prompt,
    build_preflop_user_prompt,
    generate_preflop_explanation,
    load_preflop_system_prompt,
)
from pipeline.preflop.fact_extractor import (  # noqa: E402
    PreflopFacts,
    VillainRangeStats,
)
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402

# --- fixtures ---------------------------------------------------------------
_GOLD_STUB: list[dict[str, Any]] = [
    {
        "Hand Stage": "preflop",
        "Question": "Preflop spot stem 1.",
        "option 1": "Fold",
        "option 2": "Call",
        "option 3": "Raise 308%",
        "option 4": "",
        "Correct Answer": "Raise 308%",
        "Answer Explanation": "The best play is to 3-bet.",
        "Preflop Pot Type": "SRP",
        "Pot Participant": "Heads-up",
        "Stack Depth": "100bb",
    },
    {
        "Hand Stage": "preflop",
        "Question": "Preflop spot stem 2.",
        "option 1": "Always Fold",
        "option 2": "Mostly Fold",
        "option 3": "Mostly Call",
        "option 4": "Always Call",
        "Correct Answer": "Mostly Fold",
        "Answer Explanation": "This is a mostly-fold spot.",
        "Preflop Pot Type": "3-bet",
        "Pot Participant": "Heads-up",
        "Stack Depth": "100bb",
    },
]


def _node(
    actor: str = "BTN",
    history: tuple[ParsedAction, ...] = (),
) -> PreflopDecisionNode:
    """Minimal node fixture. ``actions`` is left empty -- the Layer 6
    prompt trim only reads ``actor`` and ``history_before`` from the node."""
    return PreflopDecisionNode(
        pack_id="test_pack",
        actor=actor,
        history_before=history,
        actions=(),
    )


def _binary_facts() -> PreflopFacts:
    """A clearly-dominant action: BTN opens AKo at 100% frequency."""
    spot = PreflopSpot(
        node=_node(actor="BTN", history=()),
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies={"Fold": 0.0, "Raise 60%": 1.0},
        dominant_action="Raise 60%",
        dominant_frequency=1.0,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=None,
        hero_equity_vs_villain=None,
        archetype="open_for_value",
    )


def _frequency_facts() -> PreflopFacts:
    """A 60/40 mixed strategy: SB facing a BTN open, mixing call and 3-bet."""
    spot = PreflopSpot(
        node=_node(
            actor="SB",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.FOLD),
                ParsedAction("CO", PreflopActionType.FOLD),
                ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
            ),
        ),
        hero_hand_class="AQs",
        hero_card_combo="AsQs",
        action_frequencies={
            "Fold": 0.0,
            "Call": 0.60,
            "Raise 308%": 0.40,
        },
        dominant_action="Call",
        dominant_frequency=0.60,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="BTN",
            action_label="Raise 60%",
            weighted_combo_count=618.0,
            pct_of_dealt_hands=46.6,
            top_combos=(
                ("AA", 1.0),
                ("KK", 1.0),
                ("AKs", 1.0),
                ("AKo", 1.0),
                ("QQ", 1.0),
            ),
            # Combo-weighted order (what the data block surfaces): AKo's 12
            # combos lead the 6-combo pairs and the 4-combo suited AKs.
            most_common_combos=(
                ("AKo", 1.0),
                ("AA", 1.0),
                ("KK", 1.0),
                ("QQ", 1.0),
                ("AKs", 1.0),
            ),
        ),
        hero_equity_vs_villain=0.474,
        hero_equity_runouts_used=200,
        hero_range_equity_vs_villain=0.482,
        blockers={"AA": 1, "AKs": 1, "AKo": 4},
        archetype="3bet_as_bluff",
    )


def _open_facts() -> PreflopFacts:
    """A first-to-act open spot -- no villain. Used to exercise the
    no-villain branch of the framing + trim helpers."""
    spot = PreflopSpot(
        node=_node(actor="UTG", history=()),
        hero_hand_class="72o",
        hero_card_combo="7h2c",
        action_frequencies={"Fold": 1.0, "Raise 60%": 0.0},
        dominant_action="Fold",
        dominant_frequency=1.0,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=None,
        archetype="fold_outranged",
    )


def _mock_client(responses: list[str]) -> SimpleNamespace:
    """A fake Anthropic client whose ``messages.create`` returns each
    response in turn. Records every call so tests can assert on the
    prompt the wrapper sent."""
    calls: list[dict[str, Any]] = []
    queue = list(responses)

    def create(
        *,
        model: str,
        max_tokens: int,
        temperature: float | None = None,  # Opus 4.x dropped this kwarg
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        **_extra: Any,  # defensively swallow any future per-model kwargs
    ) -> SimpleNamespace:
        calls.append(
            {"model": model, "system": system, "messages": messages,
             "temperature": temperature}
        )
        text = queue.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
        _calls=calls,
    )
    return client


# --- system prompt carries voice rules + archetypes + banned phrases --------
def test_system_prompt_includes_every_preflop_voice_rule() -> None:
    system = build_preflop_system_prompt()
    # 19 rules since June 2026: rule 11 (one villain's range only +
    # still-to-act discipline) from the round-1 audit; rules 12-16
    # (blocker discipline, no invented reasons for alternative hands,
    # position wording from the fact, full-range-only equity talk,
    # cold-call/squeeze/open-fold/ladder terminology) from round 2;
    # rule 17 (range-vs-hand hedging); rule 18 (multi-way all-in: cite the
    # field equity, not the heads-up number); rule 19 (realization +
    # playability hedging -- "you often won't realize it OOP", not "won't").
    assert len(VOICE_RULES_PREFLOP) == 19
    for rule in VOICE_RULES_PREFLOP:
        # Rules are long; assert on the leading clause so a future
        # word-tweak doesn't break the test.
        assert rule.split(".")[0] in system, rule[:60]


def test_system_prompt_lists_all_preflop_archetypes() -> None:
    system = build_preflop_system_prompt()
    for archetype in PREFLOP_ARCHETYPE_GUIDANCE:
        assert archetype in system, archetype


def test_system_prompt_includes_banned_phrases_and_schema() -> None:
    system = build_preflop_system_prompt()
    for phrase in BANNED_LITERAL_PHRASES:
        assert phrase in system, phrase
    for key in (
        "option_1",
        "option_2",
        "option_3",
        "option_4",
        "correct_answer",
        "answer_explanation",
    ):
        assert key in system, key


def test_system_prompt_does_not_reference_board_streets() -> None:
    """Preflop voice rules explicitly forbid board talk. The system prompt
    must not contain 'flop', 'turn', or 'river' as standalone strategic
    terms (banning them in the voice rules itself is fine, but they
    shouldn't show up as descriptive references)."""
    system = build_preflop_system_prompt().lower()
    # The voice rule that BANS board references mentions these terms once
    # each, in quotes; allow that, but reject any other usage.
    for token in ("the flop", "the turn", "the river"):
        # Count occurrences: rule 5 mentions each once in quoted form.
        assert system.count(token) <= 1, f"unexpected board-street reference: {token!r}"


# --- option-style detection -------------------------------------------------
def test_option_style_detection_binary() -> None:
    assert _detect_option_style_preflop(_binary_facts()) == "binary_action"


def test_option_style_detection_frequency() -> None:
    assert _detect_option_style_preflop(_frequency_facts()) == "frequency"


def test_expected_correct_prefix_only_for_frequency_style() -> None:
    """The deterministic prefix applies only to frequency-style spots --
    binary_action style uses bare action labels."""
    assert _expected_correct_prefix_preflop(_frequency_facts()) == "Mostly"
    assert _expected_correct_prefix_preflop(_binary_facts()) is None


def test_option_style_instruction_binary_lists_actions() -> None:
    facts = _binary_facts()
    instruction = _option_style_instruction_preflop("binary_action", facts)
    assert "binary action" in instruction
    # Every action label Pio offers appears in the instruction.
    for label in facts.spot.action_frequencies:
        assert repr(label) in instruction, label


def test_option_style_instruction_frequency_pins_prefix_and_labels() -> None:
    facts = _frequency_facts()
    instruction = _option_style_instruction_preflop("frequency", facts)
    assert "HARD CONSTRAINT" in instruction
    assert "'Mostly'" in instruction
    # Both top two action labels appear in the instruction so the LLM
    # can't drop one in favour of an elegant template.
    assert "'Call'" in instruction
    assert "'Raise 308%'" in instruction


# --- user prompt carries gold examples + framing + solver data -------------
def test_user_prompt_has_gold_examples_solver_data_and_framing() -> None:
    facts = _frequency_facts()
    prompt = build_preflop_user_prompt(facts, _GOLD_STUB, style="frequency")
    # Gold-example block.
    assert "GOLD EXAMPLE 1" in prompt
    assert "Preflop spot stem 1." in prompt
    # Framing.
    assert "Stage: preflop" in prompt
    assert "AQs" in prompt
    assert "SB" in prompt
    assert "BTN" in prompt
    # Solver data block.
    assert "SOLVER DATA" in prompt
    assert '"archetype": "3bet_as_bluff"' in prompt
    assert "your_hand_equity_vs_villain_range" in prompt  # disambiguated key
    assert '"hero_position":' in prompt                   # IP/OOP fed, not inferred
    # Option-style instruction is included.
    assert "OPTION STYLE" in prompt


def test_framing_renders_prior_action_history_with_bet_levels() -> None:
    facts = _frequency_facts()
    framing = _question_framing_preflop(facts)
    # Prior action uses deterministic bet-level labels, NOT raw Pio tokens
    # (BTN's first raise is the open) -- so the LLM never recounts levels.
    assert "UTG folds" in framing
    assert "BTN opens" in framing
    assert "raises 60%" not in framing  # no raw % tokens
    # Hero's hand class + position appear.
    assert "AQs" in framing
    assert "SB" in framing


def test_framing_includes_full_labeled_strategy_with_zeros() -> None:
    """The framing carries the canonical bet-level strategy including
    0%-frequency actions, so the model knows what the solver never does."""
    facts = _frequency_facts()  # Call 60%, 3-bet 40%, Fold 0%
    framing = _question_framing_preflop(facts)
    assert "Full solver strategy at this node:" in framing
    assert "Call: 60%" in framing
    assert "3-bet: 40%" in framing
    assert "Fold: 0%" in framing  # zero action shown, not dropped


def test_framing_handles_no_villain_open_spot() -> None:
    facts = _open_facts()
    framing = _question_framing_preflop(facts)
    assert "no prior action" in framing
    assert "no specific villain" in framing
    # Hero's hand class still appears.
    assert "72o" in framing


def test_framing_includes_archetype_guidance_when_set() -> None:
    facts = _frequency_facts()
    framing = _question_framing_preflop(facts)
    assert "RECOMMENDED-ACTION ARCHETYPE" in framing
    assert "3bet_as_bluff" in framing
    # The actual guidance string for the archetype is included.
    assert PREFLOP_ARCHETYPE_GUIDANCE["3bet_as_bluff"] in framing


def test_trim_facts_canonical_strategy_keeps_zeros_and_keeps_villain_stats() -> None:
    facts = _frequency_facts()
    trimmed = _trim_facts_for_prompt(facts)
    # Hand class + dominant action (canonical bet-level label).
    assert trimmed["hand_class"] == "AQs"
    assert trimmed["dominant_action"] == "Call"
    # The strategy uses canonical labels (3-bet, not "Raise 308%") and now
    # KEEPS 0%-frequency actions so the LLM knows what the solver never does.
    assert trimmed["action_frequencies"]["Call"] == 0.60
    assert trimmed["action_frequencies"]["3-bet"] == 0.40
    assert trimmed["action_frequencies"]["Fold"] == 0.0  # kept, not dropped
    assert "Raise 308%" not in trimmed["action_frequencies"]  # raw token gone
    # Villain stats present, with a bet-level action label (not raw token).
    assert trimmed["villain_stats"]["position"] == "BTN"
    assert trimmed["villain_stats"]["action"] == "open"  # BTN's raise = the open
    # The block surfaces MOST-COMMON combos (combo-weighted), not the old
    # weight-sorted top_combos key. AKo (12 combos) leads the equal-weight set.
    mc = trimmed["villain_stats"]["most_common_combos"]
    assert "top_combos" not in trimmed["villain_stats"]
    assert mc[0]["hand_class"] == "AKo"
    assert any(combo["hand_class"] == "AA" for combo in mc)
    # Two distinct, unambiguously-named equity fields.
    assert trimmed["your_hand_equity_vs_villain_range"] == 0.474
    assert "hero_equity_vs_villain" not in trimmed  # old ambiguous key gone
    # IP/OOP + bet-level prior action are fed, not left to the LLM.
    assert trimmed["hero_position"] == "Out of Position"   # SB vs BTN open
    assert "BTN opens" in trimmed["prior_action"]          # bet-level label
    assert "raises 60%" not in trimmed["prior_action"]     # not raw token
    # Blockers.
    assert trimmed["blockers"] == {"AKo": 4, "AA": 1, "AKs": 1}


def test_trim_facts_no_villain_skips_villain_stats() -> None:
    trimmed = _trim_facts_for_prompt(_open_facts())
    assert "villain_stats" not in trimmed
    assert "your_hand_equity_vs_villain_range" not in trimmed
    assert "blockers" not in trimmed
    assert trimmed["archetype"] == "fold_outranged"


def test_trim_facts_multiway_emits_field_and_breakdown_not_single_villain() -> None:
    """Multi-way pot: the data block carries who's in, the field equity, and
    the per-opponent breakdown -- and DROPS the single-villain equity key so
    the LLM can't frame a multi-way pot as heads-up (voice rule 18)."""
    import dataclasses

    facts = dataclasses.replace(
        _frequency_facts(),
        hero_equity_vs_field=0.33,
        per_opponent_equity={"BTN": 0.42, "HJ": 0.55},
        showdown_opponents=("BTN", "HJ"),
    )
    trimmed = _trim_facts_for_prompt(facts)
    assert trimmed["players_in_pot"] == ["BTN", "HJ"]
    assert trimmed["your_hand_equity_vs_whole_field"] == 0.33
    assert trimmed["your_hand_equity_vs_each_opponent"] == {"BTN": 0.42, "HJ": 0.55}
    # The single-opponent key is gone so the model can't cite it as overall.
    assert "your_hand_equity_vs_villain_range" not in trimmed
    # The 3-bet fixture has no all-in in its history -> a squeeze-style pot,
    # so the field number is flagged as NOT the all-in decision.
    assert trimmed["multiway_pot_is_all_in"] is False


# --- validation ------------------------------------------------------------
def test_validate_accepts_well_formed_explanation() -> None:
    explanation = GeneratedExplanation(
        option_1="Fold",
        option_2="Call",
        option_3="Raise 308%",
        option_4="",
        correct_answer="Raise 308%",
        answer_explanation="The best play is to 3-bet.",
    )
    assert _validate(explanation) is None


def test_validate_catches_mismatched_correct_answer() -> None:
    explanation = GeneratedExplanation(
        option_1="Fold",
        option_2="Call",
        option_3="Raise 308%",
        option_4="",
        correct_answer="Raise 100%",  # not in options
        answer_explanation="ok.",
    )
    error = _validate(explanation)
    assert error is not None
    assert "Raise 100%" in error


def test_validate_catches_empty_explanation() -> None:
    explanation = GeneratedExplanation(
        option_1="Fold",
        option_2="Call",
        option_3="",
        option_4="",
        correct_answer="Fold",
        answer_explanation="   ",  # whitespace only
    )
    assert _validate(explanation) == "answer_explanation was empty"


def test_validate_catches_no_options() -> None:
    explanation = GeneratedExplanation(
        option_1="",
        option_2="",
        option_3="",
        option_4="",
        correct_answer="",
        answer_explanation="ok.",
    )
    assert _validate(explanation) == "the response had no non-empty options"


# --- generate_preflop_explanation: end-to-end via mock client --------------
def test_happy_path_returns_valid_explanation() -> None:
    good = (
        '{"option_1": "Fold", "option_2": "Call", "option_3": "Raise 308%", '
        '"option_4": "", "correct_answer": "Raise 308%", '
        '"answer_explanation": "The best play is to 3-bet."}'
    )
    client = _mock_client([good])
    explanation = generate_preflop_explanation(
        _frequency_facts(),
        client=client,
        gold_examples=_GOLD_STUB,
    )
    assert explanation.correct_answer == "Raise 308%"
    assert "3-bet" in explanation.answer_explanation
    assert len(client._calls) == 1
    # The default model is the Opus 4.7 production setting (shared with
    # the postflop entry point).
    assert client._calls[0]["model"] == DEFAULT_MODEL


def test_retry_recovers_after_one_validation_failure() -> None:
    bad = (
        '{"option_1": "Fold", "option_2": "Call", "option_3": "Raise 308%", '
        '"option_4": "", "correct_answer": "Raise 100%", '  # wrong label
        '"answer_explanation": "wrong action label."}'
    )
    good = (
        '{"option_1": "Fold", "option_2": "Call", "option_3": "Raise 308%", '
        '"option_4": "", "correct_answer": "Raise 308%", '
        '"answer_explanation": "The best play is to 3-bet."}'
    )
    client = _mock_client([bad, good])
    explanation = generate_preflop_explanation(
        _frequency_facts(),
        client=client,
        gold_examples=_GOLD_STUB,
    )
    assert explanation.correct_answer == "Raise 308%"
    assert len(client._calls) == 2
    # The retry call carries a corrective user turn.
    retry_messages = client._calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "failed validation" in retry_messages[-1]["content"]


def test_two_failures_raise_validation_error() -> None:
    bad = (
        '{"option_1": "Fold", "option_2": "Call", "option_3": "Raise 308%", '
        '"option_4": "", "correct_answer": "Limp", '  # not in options
        '"answer_explanation": "wrong."}'
    )
    client = _mock_client([bad, bad])
    with pytest.raises(ExplanationValidationError) as exc_info:
        generate_preflop_explanation(
            _frequency_facts(),
            client=client,
            gold_examples=_GOLD_STUB,
        )
    assert "Limp" in str(exc_info.value)
    assert "human review" in str(exc_info.value)
    assert len(client._calls) == 2


def test_caching_marks_on_system_and_gold_block() -> None:
    """Both the system prompt and the gold-examples block carry
    ``cache_control`` so prompt caching hits across many generations."""
    good = (
        '{"option_1": "Fold", "option_2": "Call", "option_3": "Raise 308%", '
        '"option_4": "", "correct_answer": "Raise 308%", '
        '"answer_explanation": "ok."}'
    )
    client = _mock_client([good])
    generate_preflop_explanation(
        _frequency_facts(),
        client=client,
        gold_examples=_GOLD_STUB,
    )
    call = client._calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    first_user_block = call["messages"][0]["content"][0]
    assert first_user_block["cache_control"] == {"type": "ephemeral"}
    assert "GOLD EXAMPLE 1" in first_user_block["text"]


def test_open_spot_no_villain_branch_still_generates() -> None:
    """A first-to-act spot (no villain, no equity data) should still
    produce a working prompt -- the open / fold-outranged archetypes
    don't need villain data."""
    good = (
        '{"option_1": "Fold", "option_2": "Raise 60%", "option_3": "", '
        '"option_4": "", "correct_answer": "Fold", '
        '"answer_explanation": "72o is a clear fold from UTG."}'
    )
    client = _mock_client([good])
    explanation = generate_preflop_explanation(
        _open_facts(),
        client=client,
        gold_examples=_GOLD_STUB,
    )
    assert explanation.correct_answer == "Fold"
    # The framing block in the user prompt mentions no-villain branch.
    user_prompt_blocks = client._calls[0]["messages"][0]["content"]
    live_block = user_prompt_blocks[1]["text"]
    assert "no specific villain" in live_block


# --- _extract_text mirrors the postflop helper -----------------------------
def test_extract_text_handles_sdk_content_blocks_and_strings() -> None:
    assert _extract_text("hello") == "hello"
    obj = SimpleNamespace(content=[SimpleNamespace(text="hi")])
    assert _extract_text(obj) == "hi"
    obj = SimpleNamespace(content=[{"text": "yo"}])
    assert _extract_text(obj) == "yo"


def test_extract_text_raises_on_missing_content() -> None:
    obj = SimpleNamespace(content=None)
    with pytest.raises(ExplanationValidationError):
        _extract_text(obj)


# --- load_preflop_system_prompt: override file mechanism --------------------
def test_load_prompt_falls_back_to_built_in_when_no_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the override file doesn't exist, the built-in default is used."""
    override = tmp_path / "preflop_system.txt"
    monkeypatch.setattr(
        "pipeline.preflop.explanation_generator._PROMPT_OVERRIDE_PATH",
        override,
    )
    assert not override.exists()
    result = load_preflop_system_prompt()
    # Same as the built-in default.
    assert result == build_preflop_system_prompt()


def test_load_prompt_reads_override_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the override file exists, its content is returned verbatim --
    no merging with the built-in default."""
    override = tmp_path / "preflop_system.txt"
    custom = "CUSTOM PROMPT for testing -- replaces the default entirely."
    override.write_text(custom, encoding="utf-8")
    monkeypatch.setattr(
        "pipeline.preflop.explanation_generator._PROMPT_OVERRIDE_PATH",
        override,
    )
    assert load_preflop_system_prompt() == custom
    # And it really is different from the default.
    assert load_preflop_system_prompt() != build_preflop_system_prompt()


def test_load_prompt_reads_file_each_call_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No caching -- edits to the override file take effect on the next
    call. Critical for the admin panel's edit-test-iterate workflow."""
    override = tmp_path / "preflop_system.txt"
    monkeypatch.setattr(
        "pipeline.preflop.explanation_generator._PROMPT_OVERRIDE_PATH",
        override,
    )
    override.write_text("version 1", encoding="utf-8")
    assert load_preflop_system_prompt() == "version 1"
    # Edit the file -- the next call should reflect the change.
    override.write_text("version 2", encoding="utf-8")
    assert load_preflop_system_prompt() == "version 2"
    # Delete the file -- back to default.
    override.unlink()
    assert load_preflop_system_prompt() == build_preflop_system_prompt()


def test_override_path_points_at_admin_panel_prompts_dir() -> None:
    """Sanity: the override path is the expected location under
    admin_panel/prompts/. Catches a future accidental move of the file."""
    assert _PROMPT_OVERRIDE_PATH.name == "preflop_system.txt"
    assert _PROMPT_OVERRIDE_PATH.parent.name == "prompts"
    assert _PROMPT_OVERRIDE_PATH.parent.parent.name == "admin_panel"


def test_generate_preflop_answer_explanation_uses_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: an override file changes the system prompt the LLM
    actually receives."""
    from pipeline.preflop.explanation_generator import (
        generate_preflop_answer_explanation,
    )

    override = tmp_path / "preflop_system.txt"
    custom = "OVERRIDDEN SYSTEM PROMPT FOR TEST"
    override.write_text(custom, encoding="utf-8")
    monkeypatch.setattr(
        "pipeline.preflop.explanation_generator._PROMPT_OVERRIDE_PATH",
        override,
    )

    good = '{"answer_explanation": "This is a clear fold."}'
    client = _mock_client([good])
    generate_preflop_answer_explanation(
        _frequency_facts(),
        options=["Fold", "Call"],
        correct_answer="Call",
        client=client,
        gold_examples=_GOLD_STUB,
    )
    # The system block sent to the API contains the override content.
    sent_system = client._calls[0]["system"]
    assert sent_system[0]["text"] == custom


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --- answer_explanation prose normalization (clean Sheets paste) -----------
def test_normalize_prose_strips_leading_paragraph_spaces() -> None:
    raw = "First para.\n\n Second para.\n\n  Third para."
    out = _normalize_prose(raw)
    assert out == "First para.\n\nSecond para.\n\nThird para."
    assert all(not line.startswith(" ") for line in out.split("\n"))


def test_normalize_prose_collapses_blank_runs_to_single() -> None:
    raw = "A.\n\n\n\nB."
    assert _normalize_prose(raw) == "A.\n\nB."


def test_normalize_prose_normalizes_crlf() -> None:
    raw = "A.\r\n\r\nB."
    assert _normalize_prose(raw) == "A.\n\nB."
    assert "\r" not in _normalize_prose(raw)


def test_normalize_prose_breaks_inline_bullets_onto_lines() -> None:
    """The run-on case: inline ' - ' bullets after a sentence/colon become
    newline bullets (the LLM sometimes emits the reason list on one line)."""
    raw = (
        "The best play is to fold. Here's why: - LJ opens tight. "
        "- Your equity is low. - You're out of position."
    )
    out = _normalize_prose(raw)
    assert out == (
        "The best play is to fold. Here's why:\n"
        "- LJ opens tight.\n"
        "- Your equity is low.\n"
        "- You're out of position."
    )


def test_normalize_prose_leaves_hyphens_and_ranges_alone() -> None:
    """Hyphenated tokens and numeric ranges (no sentence boundary before the
    dash) must not be split into bullets."""
    raw = "You should 3-bet ace-jacks. SB opens 44% - 50% here."
    assert _normalize_prose(raw) == "You should 3-bet ace-jacks. SB opens 44% - 50% here."


def test_normalize_prose_does_not_touch_already_formatted_bullets() -> None:
    raw = "Verdict.\n\nHere's why:\n- one\n- two\n\nClosing."
    assert _normalize_prose(raw) == raw


# --- prompt-workshop support: explicit system_prompt + inspection preview ---
def test_explicit_system_prompt_param_beats_override_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ``system_prompt`` param wins over the on-disk override file, so the
    workshop UI can test a named prompt without mutating the saved override."""
    from pipeline.preflop.explanation_generator import (
        generate_preflop_answer_explanation,
    )

    override = tmp_path / "preflop_system.txt"
    override.write_text("FILE OVERRIDE PROMPT", encoding="utf-8")
    monkeypatch.setattr(
        "pipeline.preflop.explanation_generator._PROMPT_OVERRIDE_PATH",
        override,
    )
    client = _mock_client(['{"answer_explanation": "This is a clear fold."}'])
    param_prompt = "PARAM PROMPT WINS"
    generate_preflop_answer_explanation(
        _frequency_facts(),
        options=["Fold", "Call"],
        correct_answer="Call",
        client=client,
        gold_examples=_GOLD_STUB,
        system_prompt=param_prompt,
    )
    # The system block sent to the API is the param, not the override file.
    assert client._calls[0]["system"][0]["text"] == param_prompt


def test_build_explanation_prompt_parts_matches_live_payload() -> None:
    """The inspection preview reuses the same block builders as the live API
    path, so what the workshop UI shows equals what the model receives."""
    from pipeline.preflop.explanation_generator import (
        _explanation_only_user_prompt,
        build_explanation_prompt_parts,
    )

    facts = _frequency_facts()
    options = ["Fold", "Call", "Raise 308%"]
    correct = "Raise 308%"
    custom = "WORKSHOP SYSTEM PROMPT"

    system, messages = _explanation_only_user_prompt(
        facts, _GOLD_STUB, options, correct, system_prompt=custom
    )
    parts = build_explanation_prompt_parts(
        facts, options, correct, system_prompt=custom, gold_examples=_GOLD_STUB
    )
    # System text matches the system block actually sent.
    assert parts["system_prompt"] == custom == system[0]["text"]
    # The two user content blocks (cached gold + live) match the preview.
    sent_blocks = messages[0]["content"]
    assert parts["gold_block"] == sent_blocks[0]["text"]
    assert parts["live_block"] == sent_blocks[1]["text"]


def test_build_explanation_prompt_parts_exposes_per_spot_inputs() -> None:
    """The preview surfaces the per-spot varying inputs (options, correct
    answer, solver data, framing) for read-only inspection."""
    from pipeline.preflop.explanation_generator import (
        build_explanation_prompt_parts,
    )

    facts = _frequency_facts()
    options = ["Fold", "Call", "Raise 308%"]
    correct = "Raise 308%"
    parts = build_explanation_prompt_parts(
        facts, options, correct, gold_examples=_GOLD_STUB
    )
    assert parts["options"] == options
    assert parts["correct_answer"] == correct
    # solver_data is the trimmed facts dict, with the disambiguated equity key.
    assert isinstance(parts["solver_data"], dict)
    assert "your_hand_equity_vs_villain_range" in parts["solver_data"]
    # framing names the hero hand.
    assert "AQs" in parts["framing"]
    # assembled string carries every labeled section.
    assert "SYSTEM PROMPT" in parts["assembled"]
    assert "GOLD EXAMPLE 1" in parts["assembled"]
    assert "SOLVER DATA" in parts["assembled"]


def test_solver_data_includes_action_pending_fields() -> None:
    """With the node's pack registered, the SOLVER DATA block carries the
    multiway-awareness facts (June 2026 audit findings #1/#3)."""
    from pathlib import Path

    from pipeline.preflop.explanation_generator import _trim_facts_for_prompt
    from pipeline.preflop.fact_extractor import PreflopFacts
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType
    from pipeline.preflop.node_enumerator import PreflopDecisionNode
    from pipeline.preflop.pack import PreflopPack, clear_registry, register_pack
    from pipeline.preflop.spot_sampler import PreflopSpot

    PT = PreflopActionType
    clear_registry()
    try:
        register_pack(PreflopPack(
            pack_id="pending_test_pack",
            root_path=Path("/tmp/fake"),
            grammar_name="monker_nlhe",
            table_size=9,
            stack_depth_bb=100,
            open_size_bb=4.0,
        ))
        history = (
            ParsedAction("UTG", PT.RAISE, 120.0),
            ParsedAction("UTG+1", PT.FOLD),
            ParsedAction("UTG+2", PT.CALL),
            ParsedAction("LJ", PT.FOLD), ParsedAction("HJ", PT.FOLD),
            ParsedAction("CO", PT.FOLD), ParsedAction("BTN", PT.FOLD),
            ParsedAction("SB", PT.FOLD),
        )
        node = PreflopDecisionNode(
            pack_id="pending_test_pack", actor="BB",
            history_before=history, actions=(),
        )
        spot = PreflopSpot(
            node=node, hero_hand_class="AJs", hero_card_combo="AsJs",
            action_frequencies={"Fold": 0.3, "Call": 0.7},
            dominant_action="Call", dominant_frequency=0.7,
        )
        out = _trim_facts_for_prompt(PreflopFacts(spot=spot))
        assert out["other_players_still_in_hand"] == ["UTG", "UTG+2"]
        # UTG raised and UTG+2 called: if the BB just calls, both have no
        # further decision -> hero's call/fold closes the preflop action.
        assert out["still_to_act_after_you"] == []
        assert out["your_call_or_fold_closes_the_action"] is True
    finally:
        clear_registry()


def test_solver_data_omits_pending_fields_without_registry() -> None:
    from pipeline.preflop.explanation_generator import _trim_facts_for_prompt
    from pipeline.preflop.fact_extractor import PreflopFacts
    from pipeline.preflop.node_enumerator import PreflopDecisionNode
    from pipeline.preflop.pack import clear_registry
    from pipeline.preflop.spot_sampler import PreflopSpot

    clear_registry()
    node = PreflopDecisionNode(
        pack_id="never_registered", actor="BB", history_before=(), actions=(),
    )
    spot = PreflopSpot(
        node=node, hero_hand_class="AA", hero_card_combo="AcAd",
        action_frequencies={"Fold": 0.1, "Call": 0.9},
        dominant_action="Call", dominant_frequency=0.9,
    )
    out = _trim_facts_for_prompt(PreflopFacts(spot=spot))
    assert "other_players_still_in_hand" not in out
    assert "your_call_or_fold_closes_the_action" not in out
