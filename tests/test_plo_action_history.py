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


def test_action_history_include_folds_renders_them():
    # The LLM-facing variant: folds rendered so the model can tell a
    # heads-up-after-the-opener-folded spot from a multiway one (the player
    # sees the fold on the app's table render instead).
    facts = _facts(
        "SB",
        (F("LJ"), R("HJ"), F("CO"), F("BU"), C("SB"), R("BB"), F("HJ")),
        ("Qc", "Qd", "Th", "Kh"),
    )
    s = format_plo_action_history(facts, display_in_bb=True, include_folds=True)
    assert "The Hijack opens to 3.5bb." in s
    assert "You call." in s
    assert "The Hijack folds." in s  # the entrant's exit is visible
    assert "UTG folds." in s  # never-entered folds render too
    # The player-facing default still drops every fold.
    assert "fold" not in format_plo_action_history(facts)


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
        "$0.5/$1 Online PLO cash. $100 effective stacks."
    )


def test_context_bb_and_tournament():
    assert "100bb effective stacks" in format_plo_context(display_in_bb=True)
    assert format_plo_context(game_format="tournament").startswith("Tournament.")


def test_min_raise_resolves_to_the_min_raise_rule_not_a_pot_raise():
    """GROUND TRUTH (July 22 2026, user catch): a preflop first-in min-raise
    goes to exactly 2bb (BB post = the opening bet increment), and a pot
    3-bet over it goes to 2 + (2 + 3.5) = 7.5bb. The old resolver defaulted
    MIN_RAISE into the pot-raise arm ("raise_pct or 100"), rendering the
    2bb open as 3.5bb and the 3-bet as 12bb -- while the solver file's own
    fold EV (-2bb) proved the real size. Display, prices, SOLVER DATA,
    animation, and app tokens all read this walk, so all were wrong on
    min-raise lines."""
    from pipeline.plo.pack import PloAction, PloActionType

    history = (
        PloAction("LJ", PloActionType.FOLD, None),
        PloAction("HJ", PloActionType.MIN_RAISE, None),
        PloAction("CO", PloActionType.FOLD, None),
        PloAction("BU", PloActionType.RAISE, 100),
        PloAction("SB", PloActionType.FOLD, None),
        PloAction("BB", PloActionType.FOLD, None),
    )
    resolved, pot = resolve_pot_limit(history, stack_bb=20.0)
    sizes = {r.seat: r.to_bb for r in resolved if r.to_bb is not None}
    assert sizes["HJ"] == 2.0  # min-raise open = 2bb, never 3.5bb
    assert sizes["BU"] == 7.5  # pot 3-bet over a 2bb open
    assert pot == 11.0

    # Min-3-bet over a POT open: 3.5 + (3.5 - 1) = 6bb.
    history2 = (
        PloAction("HJ", PloActionType.RAISE, 100),
        PloAction("BU", PloActionType.MIN_RAISE, None),
    )
    resolved2, _pot2 = resolve_pot_limit(history2, stack_bb=100.0)
    sizes2 = {r.seat: r.to_bb for r in resolved2 if r.to_bb is not None}
    assert sizes2["HJ"] == 3.5
    assert sizes2["BU"] == 6.0

    # MTT bb-ante: the ante joins the pot but never the raise increment, so
    # a min-raise open is STILL 2bb.
    history3 = (PloAction("HJ", PloActionType.MIN_RAISE, None),)
    resolved3, pot3 = resolve_pot_limit(history3, stack_bb=20.0, ante_bb=1.0)
    assert resolved3[0].to_bb == 2.0
    assert pot3 == 1.5 + 1.0 + 2.0  # blinds + ante + the min-raise


def test_fold_ev_consistency_guard_catches_size_resolution_bugs():
    """The tripwire born from the min-raise bug (July 22 2026): the pack's
    fold EV always equals minus hero's invested chips, so any displayed-size
    arithmetic error trips it -- for any pack and any token type. Also pins
    the blind/ante conventions (SB fold = -0.5bb, BB fold = -1bb with the
    ante EXCLUDED -- the ante is dead before the EV baseline)."""
    from types import SimpleNamespace

    from pipeline.plo.action_history import (
        fold_ev_consistency_issue,
        resolved_commitment_bb,
    )
    from pipeline.plo.pack import PloAction, PloActionType

    open_then_3bet = (
        PloAction("HJ", PloActionType.MIN_RAISE, None),  # to 2bb
        PloAction("BU", PloActionType.RAISE, 100),       # pot: to 7.5bb
    )
    # Commitments follow the fixed walk.
    assert resolved_commitment_bb(open_then_3bet, "HJ", stack_bb=20.0) == 2.0
    assert resolved_commitment_bb(open_then_3bet, "BU", stack_bb=20.0) == 7.5
    # Blinds count; the MTT ante does not.
    assert resolved_commitment_bb((), "SB", ante_bb=1.0) == 0.5
    assert resolved_commitment_bb((), "BB", ante_bb=1.0) == 1.0

    def spot(fold_ev_sb, history, actor):
        return SimpleNamespace(
            ev_by_action={"Fold": fold_ev_sb, "Call": -1.0},
            node=SimpleNamespace(history_before=history, actor=actor),
        )

    # Consistent: HJ invested 2bb -> fold EV -4.0sb (-2bb). No issue.
    assert fold_ev_consistency_issue(
        spot(-4.0, open_then_3bet, "HJ"), stack_bb=20.0
    ) is None
    # The old bug's exact shape: sizes resolved as if the open were 3.5bb
    # would DISAGREE with the file's -2bb fold EV. Simulate by lying about
    # the fold EV instead (same arithmetic): -7.0sb (-3.5bb) vs invested 2bb.
    issue = fold_ev_consistency_issue(
        spot(-7.0, open_then_3bet, "HJ"), stack_bb=20.0
    )
    assert issue is not None and "size-resolution" in issue
    # No fold on the menu -> no check.
    no_fold = SimpleNamespace(
        ev_by_action={"Call": -1.0},
        node=SimpleNamespace(history_before=(), actor="BB"),
    )
    assert fold_ev_consistency_issue(no_fold) is None


def test_short_stack_packs_are_venue_neutral():
    """July 22 2026 (team ask): the 12bb/20bb cash packs carry NO Live/Online
    framing -- the Context is stacks-only in both display modes, and the
    100bb 6-max format is unchanged."""
    from pipeline.plo.action_history import format_plo_context
    from pipeline.plo.pack import PLO_PACK_6MAX_12BB, PLO_PACK_6MAX_20BB

    assert PLO_PACK_6MAX_12BB.venue_neutral is True
    assert PLO_PACK_6MAX_20BB.venue_neutral is True
    assert format_plo_context(
        stack_bb=20.0, display_in_bb=True, venue_neutral=True
    ) == "20bb effective stacks."
    assert format_plo_context(
        stack_bb=20.0, display_in_bb=False, venue_neutral=True
    ) == "$20 effective stacks."
    # The 100bb 6-max pack keeps its venue framing exactly as before.
    assert format_plo_context(stack_bb=100.0, display_in_bb=True).startswith(
        "Online PLO cash."
    )
