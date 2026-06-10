"""Tests for pipeline.plo.action_history (pot-limit calc + prose)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.action_history import (  # noqa: E402
    ResolvedAction,
    format_plo_action_history,
    format_plo_context,
    pot_bb,
    resolve_pot_limit,
)
from pipeline.plo.fact_extractor import PloFacts  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402


def R(seat: str) -> PloAction:
    return PloAction(seat, PloActionType.RAISE, 100)


def F(seat: str) -> PloAction:
    return PloAction(seat, PloActionType.FOLD)


def C(seat: str) -> PloAction:
    return PloAction(seat, PloActionType.CALL)


def J(seat: str) -> PloAction:
    return PloAction(seat, PloActionType.ALL_IN)


# --- resolve_pot_limit ----------------------------------------------------
def test_open_is_three_and_a_half_bb():
    acts, pot = resolve_pot_limit((R("LJ"),))
    assert acts[0] == ResolvedAction("LJ", "open", 3.5)
    assert pot == pytest.approx(5.0)  # SB .5 + BB 1 + open 3.5


def test_3bet_from_bb_is_smaller_than_from_open_seat():
    # The BB already has 1bb posted, so its pot-sized 3-bet is a touch smaller.
    bb = resolve_pot_limit((R("LJ"), F("HJ"), F("CO"), F("BU"), F("SB"), R("BB")))[0]
    hj = resolve_pot_limit((R("LJ"), R("HJ")))[0]
    assert bb[-1] == ResolvedAction("BB", "3-bet", 11.0)
    assert hj[-1] == ResolvedAction("HJ", "3-bet", 12.0)


def test_4bet_level_and_pot():
    acts, pot = resolve_pot_limit((R("LJ"), R("HJ"), R("CO")))
    assert [a.verb for a in acts] == ["open", "3-bet", "4-bet"]
    assert acts[-1].to_bb == pytest.approx(41.0)
    assert pot == pytest.approx(58.0)


def test_all_in_is_full_stack_and_counts_as_a_level():
    acts, _pot = resolve_pot_limit((R("LJ"), R("HJ"), J("CO")), stack_bb=100.0)
    assert acts[-1] == ResolvedAction("CO", "all-in", 100.0)


def test_call_does_not_raise_the_bet():
    acts, pot = resolve_pot_limit((R("LJ"), C("HJ")))
    assert acts[-1] == ResolvedAction("HJ", "call", None)
    assert pot == pytest.approx(0.5 + 1.0 + 3.5 + 3.5)  # SB+BB+open+HJ call


def test_pot_bb_helper():
    node = PloDecisionNode(actor="BB", history_before=(R("LJ"),), actions=(), history_stem="")
    assert pot_bb(node) == pytest.approx(5.0)


# --- prose ----------------------------------------------------------------
def _facts(actor: str, history: tuple[PloAction, ...], cards: tuple[str, str, str, str]) -> PloFacts:
    node = PloDecisionNode(actor=actor, history_before=history, actions=(), history_stem="")
    spot = PloSpot(node=node, hero_index=0, hero_label="x", hero_cards=cards, presence=1.0)
    return PloFacts(spot=spot, hand_class=classify_plo_hand(cards), archetype="")


def test_action_history_renders_hero_villain_and_drops_folds():
    facts = _facts(
        "LJ",
        (R("LJ"), F("HJ"), F("CO"), F("BU"), F("SB"), R("BB")),
        ("Ac", "Ad", "4h", "8h"),
    )
    s = format_plo_action_history(facts)
    assert s.startswith("You're UTG with A♣️ A♦️ 4❤️ 8❤️.")
    assert "You open to $3.50." in s  # hero's own prior action
    assert "The Big Blind 3-bets to $11." in s
    assert "fold" not in s  # preflop folds dropped


def test_action_history_open_node_has_no_actions():
    facts = _facts("LJ", (), ("Ac", "Ad", "Kc", "Kd"))
    assert format_plo_action_history(facts) == "You're UTG with A♣️ A♦️ K♣️ K♦️."


def test_action_history_bb_display():
    facts = _facts("HJ", (R("LJ"),), ("Ac", "Ad", "Kc", "Kd"))
    s = format_plo_action_history(facts, display_in_bb=True)
    assert "UTG opens to 3.5bb." in s


# --- context --------------------------------------------------------------
def test_context_cash_dollars():
    assert format_plo_context() == (
        "$0.5/$1 Online PLO cash. 6-handed. $100 effective stacks."
    )


def test_context_bb_and_tournament():
    assert "100bb effective stacks" in format_plo_context(display_in_bb=True)
    assert format_plo_context(game_format="tournament").startswith("PLO tournament.")
