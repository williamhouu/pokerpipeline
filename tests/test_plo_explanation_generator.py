"""Tests for pipeline.plo.explanation_generator (Layer 6, mock client)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import ExplanationValidationError  # noqa: E402
from pipeline.plo.explanation_generator import (  # noqa: E402
    VOICE_RULES_PLO,
    build_plo_system_prompt,
    build_solver_data,
    generate_plo_answer_explanation,
)
from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.gold_examples import load_plo_gold_examples  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

CARDS = ("As", "Ks", "Ah", "Kh")


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [type("C", (), {"text": text})()]
        self.usage = None


class _Messages:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def create(self, **_kw: object) -> _Resp:
        return _Resp(self._texts.pop(0))


class _MockClient:
    def __init__(self, *texts: str) -> None:
        self.messages = _Messages(list(texts))


def _json(prose: str) -> str:
    return json.dumps({"answer_explanation": prose})


def _facts() -> PloFacts:
    node = PloDecisionNode(
        actor="HJ",
        history_before=(PloAction("LJ", PloActionType.RAISE, 100),),
        actions=(),
        history_stem="40100",
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=CARDS,
        action_frequencies={"Call": 0.7, "Raise 100%": 0.3},
        ev_by_action={"Call": 2.0, "Raise 100%": 1.0, "Fold": -7.0},
        presence=1.0,
    )
    return PloFacts(
        spot=spot,
        hand_class=classify_plo_hand(CARDS),
        archetype="3bet_for_value",
        villain_stats=PloVillainStats(seat="LJ", action_label="Raise 100%", weighted_combo_count=1.0, pct_of_dealt_hands=18.0),
        hero_equity_vs_villain=0.55,
        hero_range_equity_vs_villain=0.42,
    )


# --- prompt + data block --------------------------------------------------
def test_ships_without_examples_by_default():
    assert load_plo_gold_examples() == ()
    prompt = build_plo_system_prompt()
    assert "EXAMPLES" not in prompt


def test_system_prompt_has_rules_and_bans_em_dash():
    prompt = build_plo_system_prompt()
    assert "VOICE RULES" in prompt
    assert "Never use an em dash" in prompt
    assert "BANNED PHRASES" in prompt
    assert len(VOICE_RULES_PLO) == 10  # noqa: PLR2004


def test_examples_seam_injects_when_provided():
    prompt = build_plo_system_prompt(
        examples=({"question": "Q1", "answer_explanation": "A1"},)
    )
    assert "EXAMPLES" in prompt
    assert "A1" in prompt


def test_solver_data_has_the_facts():
    data = build_solver_data(_facts(), ["Fold", "Call", "3-bet"], "Call")
    assert data["correct_action"] == "Call"
    assert data["your_hand_equity_vs_villain_range_pct"] == 55  # noqa: PLR2004
    assert data["villain"]["seat"] == "LJ"
    assert "3bet_for_value" in data["strategic_frame"]
    assert data["your_hand"]  # emoji cards


# --- generation -----------------------------------------------------------
def test_generation_returns_prose():
    client = _MockClient(_json("You should 3-bet here. Strong double-suited aces in position."))
    result = generate_plo_answer_explanation(_facts(), ["Fold", "Call", "3-bet"], "3-bet", client=client, examples=())
    assert result.correct_answer == "3-bet"
    assert "3-bet" in result.answer_explanation
    assert (result.option_1, result.option_2, result.option_3) == ("Fold", "Call", "3-bet")


def test_em_dash_and_semicolon_are_guaranteed_out():
    # Even when the model emits them, the deterministic strip removes them.
    client = _MockClient(_json("Call here. It plays well in position — double-suited; take a flop."))
    result = generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "Call", client=client, examples=())
    assert "—" not in result.answer_explanation
    assert ";" not in result.answer_explanation
    assert "–" not in result.answer_explanation


def test_banned_phrase_triggers_a_retry():
    # First attempt uses a banned phrase; the clean retry is accepted.
    client = _MockClient(
        _json("Call. We should leverage our position."),  # 'leverage' is banned
        _json("Call. Your position lets you realize this hand's equity."),
    )
    result = generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "Call", client=client, examples=())
    assert "leverage" not in result.answer_explanation.lower()


def test_correct_answer_must_be_an_option():
    with pytest.raises(ValueError, match="not in options"):
        generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "3-bet", client=_MockClient(), examples=())


def test_unparseable_response_raises_after_retries():
    client = _MockClient("not json at all", "still not json")
    with pytest.raises(ExplanationValidationError):
        generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "Call", client=client, examples=(), max_retries=1)


# --- system_prompt override (the prompt-library / Compare seam) ------------
class _CapturingMessages:
    """Records the ``system`` kwarg of each create() call."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.systems: list[str] = []

    def create(self, **kw: object) -> _Resp:
        self.systems.append(str(kw.get("system", "")))
        return _Resp(self.text)


class _CapturingClient:
    def __init__(self, text: str) -> None:
        self.messages = _CapturingMessages(text)


def test_system_prompt_override_is_used_verbatim():
    client = _CapturingClient(_json("Call. It plays well in position."))
    custom = "CUSTOM PLO SYSTEM PROMPT -- edited in the admin panel."
    generate_plo_answer_explanation(
        _facts(), ["Fold", "Call"], "Call",
        client=client, examples=(), system_prompt=custom,
    )
    assert client.messages.systems == [custom]


def test_no_override_uses_built_in_system_prompt():
    client = _CapturingClient(_json("Call. It plays well in position."))
    generate_plo_answer_explanation(
        _facts(), ["Fold", "Call"], "Call", client=client, examples=(),
    )
    assert client.messages.systems == [build_plo_system_prompt(examples=())]
