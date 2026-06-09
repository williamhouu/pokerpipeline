"""Tests for pipeline.plo.difficulty (3-axis PLO difficulty rating; EV dropped)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.difficulty import (  # noqa: E402
    _HARD_CEILING,
    _HARD_FLOOR,
    compute_plo_difficulty,
)
from pipeline.plo.fact_extractor import PloFacts  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

PREMIUM = ("As", "Ks", "Ah", "Kh")  # double-suited AAKK -> premium
MARGINAL = ("2c", "7d", "3h", "9s")  # -> marginal (hardest hand axis)


def _act(seat: str, atype: PloActionType, pct: int | None = None) -> PloAction:
    return PloAction(seat=seat, action=atype, raise_pct=pct)


def _facts(
    *,
    freqs: dict[str, float],
    ev_by_action: dict[str, float],
    archetype: str,
    hero_cards: tuple[str, str, str, str] = PREMIUM,
    history: tuple[PloAction, ...] = (),
) -> PloFacts:
    node = PloDecisionNode(
        actor="HJ", history_before=history, actions=(), history_stem=""
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=hero_cards,
        action_frequencies=freqs,
        ev_by_action=ev_by_action,
        presence=1.0,
    )
    return PloFacts(
        spot=spot, hand_class=classify_plo_hand(hero_cards), archetype=archetype
    )


def test_easy_open_spot_scores_low():
    facts = _facts(
        freqs={"Raise 100%": 1.0, "Fold": 0.0},
        ev_by_action={"Raise 100%": 5.0, "Fold": 0.0},
        archetype="open_for_value",
        hero_cards=PREMIUM,
    )
    result = compute_plo_difficulty(facts)
    assert result.easy_freq == pytest.approx(1.0)  # 100% pure
    assert result.easy_hand == pytest.approx(1.0)  # premium
    assert result.ev_available
    assert result.score < 1000  # noqa: PLR2004  # clearly an easy spot


def test_hard_4bet_bluff_spot_scores_high():
    facts = _facts(
        freqs={"Raise 100%": 0.6, "Call": 0.4},
        ev_by_action={"Raise 100%": 0.2, "Call": 0.0, "Fold": -7.0},
        archetype="4bet_as_bluff",
        hero_cards=MARGINAL,
    )
    result = compute_plo_difficulty(facts)
    assert result.easy_freq < 0.2  # noqa: PLR2004  # ~0.11, near the 55% floor
    assert result.easy_hand == pytest.approx(0.40)  # marginal
    assert result.score > 2000  # noqa: PLR2004  # clearly a hard spot


def test_ev_gap_is_free_from_the_pack():
    # ev_gap_bb = (best - 2nd-best ev) / 2; here (5.0 - 0.0)/2 = 2.5 bb.
    facts = _facts(
        freqs={"Raise 100%": 1.0, "Fold": 0.0},
        ev_by_action={"Raise 100%": 5.0, "Fold": 0.0},
        archetype="open_for_value",
    )
    assert facts.ev_gap_bb == pytest.approx(2.5)
    result = compute_plo_difficulty(facts)
    assert result.ev_available
    assert result.easy_ev == pytest.approx(2.5 / 3.0)


def test_single_action_node_has_no_ev_gap():
    # One action -> no EV gap -> ev_available False, easy_ev 0.0. EV is not in
    # the blend anyway, so this only affects the diagnostic fields.
    facts = _facts(
        freqs={"Fold": 1.0},
        ev_by_action={"Fold": -7.0},
        archetype="fold_dominated",
        hero_cards=MARGINAL,
    )
    assert facts.ev_gap_bb is None
    result = compute_plo_difficulty(facts)
    assert not result.ev_available
    assert result.easy_ev == 0.0
    # The 3-axis blend stays a valid weighted average.
    assert 0.0 <= result.easy_blend <= 1.0


def test_ev_gap_does_not_affect_score():
    # The EV axis was dropped from the PLO blend (June 2026): it is a diagnostic
    # only. Two spots identical but for the EV gap get the SAME difficulty score,
    # even though their easy_ev diagnostic differs.
    base = dict(
        freqs={"Raise 100%": 0.7, "Call": 0.3},
        archetype="3bet_for_value",
        hero_cards=PREMIUM,
    )
    small_gap = compute_plo_difficulty(
        _facts(**base, ev_by_action={"Raise 100%": 0.1, "Call": 0.0})
    )
    big_gap = compute_plo_difficulty(
        _facts(**base, ev_by_action={"Raise 100%": 10.0, "Call": 0.0})
    )
    assert small_gap.easy_ev != big_gap.easy_ev  # the diagnostic differs
    assert small_gap.score == big_gap.score  # but the score does not


def test_score_stays_within_hard_bounds():
    facts = _facts(
        freqs={"Raise 100%": 1.0},
        ev_by_action={"Raise 100%": 50.0, "Fold": 0.0},
        archetype="open_for_value",
        hero_cards=PREMIUM,
    )
    result = compute_plo_difficulty(facts)
    assert _HARD_FLOOR <= result.score <= _HARD_CEILING


def test_fold_spot_not_penalised_for_multiway():
    history = (_act("LJ", PloActionType.RAISE, 100), _act("HJ", PloActionType.CALL))
    base = dict(
        freqs={"Fold": 1.0},
        ev_by_action={"Fold": -7.0, "Call": -9.0, "Raise 100%": -12.0},
        archetype="fold_dominated",
        hero_cards=MARGINAL,
    )
    heads_up = compute_plo_difficulty(_facts(**base))
    multiway = compute_plo_difficulty(_facts(**base, history=history))
    # A trash fold is trivial whether multiway or not -> concept ease unchanged.
    assert multiway.easy_concept == pytest.approx(heads_up.easy_concept)
