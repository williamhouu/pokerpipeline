"""Tests for the shared per-question chatbot context (pipeline.chat_context)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.chat_context import (  # noqa: E402
    CHATBOT_GUARDRAILS,
    StrategyEntry,
    build_chat_context,
)


def _ctx(**over):
    base = dict(
        pipeline="postflop",
        situation="BTN c-bets the flop.",
        hero_hand="A♠ K♠",
        hand_summary="top pair (strong)",
        recommended_action="Bet 75%",
        also_acceptable=["", "Check"],
        full_strategy=[
            StrategyEntry("Bet 75%", 70.0, 1.2),
            StrategyEntry("Check", 30.0, 1.0),
        ],
        key_facts={"spr": 4.0, "empty": "", "none": None},
        villain={"seat": "BB"},
        strategic_frame="value_bet: bet for value",
        concept_tags=["c_bet_spot"],
        skills_tested=["C-Betting"],
        difficulty=1500,
        coaching_answer="Bet for value.",
    )
    base.update(over)
    return json.loads(build_chat_context(**base))


def test_chat_context_is_valid_json_with_all_sections() -> None:
    ctx = _ctx()
    for key in (
        "pipeline", "situation", "hero_hand", "hand_summary", "recommended_action",
        "also_acceptable", "full_strategy", "key_facts", "villain",
        "strategic_frame", "concept_tags", "skills_tested", "difficulty",
        "coaching_answer", "guardrails",
    ):
        assert key in ctx
    assert ctx["guardrails"] == CHATBOT_GUARDRAILS


def test_full_strategy_sorted_by_freq_and_carries_ev() -> None:
    ctx = _ctx()
    fs = ctx["full_strategy"]
    assert [e["action"] for e in fs] == ["Bet 75%", "Check"]  # descending freq
    assert fs[0]["ev_bb"] == 1.2  # noqa: PLR2004


def test_strategy_omits_ev_when_absent() -> None:
    ctx = _ctx(full_strategy=[StrategyEntry("Call", 100.0)])
    assert "ev_bb" not in ctx["full_strategy"][0]


def test_blank_acceptable_and_empty_facts_are_stripped() -> None:
    ctx = _ctx()
    assert ctx["also_acceptable"] == ["Check"]  # the "" is dropped
    assert "empty" not in ctx["key_facts"] and "none" not in ctx["key_facts"]


def test_deterministic() -> None:
    assert build_chat_context(
        pipeline="plo", situation="s", hero_hand="h", hand_summary="hs",
        recommended_action="Call", also_acceptable=[],
        full_strategy=[StrategyEntry("Call", 100.0)], key_facts={}, villain=None,
        strategic_frame="f", concept_tags=[], skills_tested=[], difficulty=1,
        coaching_answer="c",
    ) == build_chat_context(
        pipeline="plo", situation="s", hero_hand="h", hand_summary="hs",
        recommended_action="Call", also_acceptable=[],
        full_strategy=[StrategyEntry("Call", 100.0)], key_facts={}, villain=None,
        strategic_frame="f", concept_tags=[], skills_tested=[], difficulty=1,
        coaching_answer="c",
    )
