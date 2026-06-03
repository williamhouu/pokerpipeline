"""Tests for pipeline.plo.spot_sampler (conditional strategy + per-action EV)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.hand_order import HAND_COUNT, monker_label  # noqa: E402
from pipeline.plo.node_enumerator import (  # noqa: E402
    PloDecisionNode,
    enumerate_plo_nodes,
)
from pipeline.plo.pack import PloPack  # noqa: E402
from pipeline.plo.spot_sampler import (  # noqa: E402
    PloSpot,
    enumerate_plo_spots_for_node,
    sample_plo_spot,
)

HERO = 100  # an arbitrary hand index used across the strategy tests


def _write_rng(path: Path, weights: dict[int, tuple[float, float]]) -> None:
    """weights: {index: (p, ev_sb)}; everything else (0.0, 0.0)."""
    out: list[str] = []
    for i in range(HAND_COUNT):
        out.append("????")  # pattern line (ignored, as by the viewer)
        p, ev = weights.get(i, (0.0, 0.0))
        out.append(f"{p};{ev * 1000}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _hj_facing_open_node(tmp_path: Path) -> PloDecisionNode:
    """HJ facing LJ's open: fold / call / 3-bet, with HERO mixing call-3bet.

    HERO reaches the node 40% of the time (presence 0.4): of that, calls 0.3
    and 3-bets 0.1 -> conditional Call 0.75 / Raise 0.25. EVs: fold -7 (forfeit
    the open), call 2.5 (best), 3-bet 2.0.
    """
    _write_rng(tmp_path / "40100.0.rng", {HERO: (0.0, -7.0)})  # HJ fold
    _write_rng(tmp_path / "40100.1.rng", {HERO: (0.3, 2.5)})  # HJ call
    _write_rng(tmp_path / "40100.40100.rng", {HERO: (0.1, 2.0)})  # HJ 3-bet
    pack = PloPack(root=tmp_path, label="test")
    return next(n for n in enumerate_plo_nodes(pack) if n.actor == "HJ")


def test_joint_weights_normalise_to_conditional_strategy(tmp_path):
    spot = sample_plo_spot(_hj_facing_open_node(tmp_path), HERO)
    assert spot.presence == pytest.approx(0.4)
    assert spot.action_frequencies["Fold"] == pytest.approx(0.0)
    assert spot.action_frequencies["Call"] == pytest.approx(0.75)
    assert spot.action_frequencies["Raise 100%"] == pytest.approx(0.25)
    assert sum(spot.action_frequencies.values()) == pytest.approx(1.0)


def test_dominant_action_and_frequency(tmp_path):
    spot = sample_plo_spot(_hj_facing_open_node(tmp_path), HERO)
    assert spot.dominant_action == "Call"
    assert spot.dominant_frequency == pytest.approx(0.75)


def test_per_action_ev_is_kept(tmp_path):
    spot = sample_plo_spot(_hj_facing_open_node(tmp_path), HERO)
    assert spot.ev_by_action["Fold"] == pytest.approx(-7.0)
    assert spot.ev_by_action["Call"] == pytest.approx(2.5)
    assert spot.ev_by_action["Raise 100%"] == pytest.approx(2.0)


def test_ev_gap_ignores_dominated_fold(tmp_path):
    # Best (call 2.5) minus second-best (3-bet 2.0); the -7 fold is dominated.
    spot = sample_plo_spot(_hj_facing_open_node(tmp_path), HERO)
    assert spot.ev_gap_sb == pytest.approx(0.5)


def test_hero_identity_fields(tmp_path):
    spot = sample_plo_spot(_hj_facing_open_node(tmp_path), HERO)
    assert spot.hero_index == HERO
    assert spot.hero_label == monker_label(HERO)
    assert len(spot.hero_cards) == 4  # noqa: PLR2004
    assert len(set(spot.hero_cards)) == 4  # distinct concrete cards  # noqa: PLR2004


def test_absent_hand_has_zero_presence(tmp_path):
    node = _hj_facing_open_node(tmp_path)
    spot = sample_plo_spot(node, hero_index=9999)  # not present in any file
    assert spot.presence == 0.0
    assert all(v == 0.0 for v in spot.action_frequencies.values())


def test_enumerate_filters_by_presence(tmp_path):
    node = _hj_facing_open_node(tmp_path)
    present = {s.hero_index for s in enumerate_plo_spots_for_node(node, min_presence=0.01)}
    assert present == {HERO}  # only the one hand with weight in the fixtures


def test_out_of_range_index_raises(tmp_path):
    node = _hj_facing_open_node(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        sample_plo_spot(node, HAND_COUNT)
    with pytest.raises(ValueError, match="out of range"):
        sample_plo_spot(node, -1)


def test_ev_gap_none_for_single_action_node():
    node = PloDecisionNode(
        actor="LJ", history_before=(), actions=(), history_stem=""
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="AAAA",
        hero_cards=("Ac", "Ad", "Ah", "As"),
        ev_by_action={"Fold": -7.0},
    )
    assert spot.ev_gap_sb is None


def test_empty_spot_has_neutral_dominant():
    node = PloDecisionNode(
        actor="LJ", history_before=(), actions=(), history_stem=""
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="AAAA",
        hero_cards=("Ac", "Ad", "Ah", "As"),
    )
    assert spot.dominant_action == ""
    assert spot.dominant_frequency == 0.0
    assert spot.ev_gap_sb is None
