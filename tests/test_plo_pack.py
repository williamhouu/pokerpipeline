"""Tests for pipeline.plo.pack (.rng reader + filename action decoder)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.hand_order import HAND_COUNT, monker_label  # noqa: E402
from pipeline.plo.pack import (  # noqa: E402
    PloActionType,
    discover_plo_pack,
    node_actor,
    parse_node_path,
    range_at,
    read_rng,
)

F = PloActionType.FOLD
C = PloActionType.CALL
R = PloActionType.RAISE
J = PloActionType.ALL_IN


def _seats_actions(stem):
    return [(a.seat, a.action) for a in parse_node_path(stem)]


# --- parse_node_path ------------------------------------------------------
def test_single_open_raise():
    actions = parse_node_path("40100")
    assert actions[0].seat == "LJ"
    assert actions[0].action is R
    assert actions[0].raise_pct == 100  # noqa: PLR2004
    assert node_actor(actions) == "LJ"


def test_open_then_fold():
    assert _seats_actions("40100.0") == [("LJ", R), ("HJ", F)]
    assert node_actor(parse_node_path("40100.0")) == "HJ"


def test_five_folds_to_sb():
    assert _seats_actions("0.0.0.0.0") == [
        ("LJ", F), ("HJ", F), ("CO", F), ("BU", F), ("SB", F),
    ]


def test_fold_removes_seat_from_action():
    # LJ raises, HJ folds, CO 3-bets -- the 3-bettor is CO, not HJ.
    assert _seats_actions("40100.0.40100") == [("LJ", R), ("HJ", F), ("CO", R)]
    assert node_actor(parse_node_path("40100.0.40100")) == "CO"


def test_raiser_rotates_and_acts_again():
    # LJ open, everyone calls, BB raises (squeeze) -- action wraps to BB.
    actions = parse_node_path("40100.1.1.1.1.40100")
    assert [a.seat for a in actions] == ["LJ", "HJ", "CO", "BU", "SB", "BB"]
    assert actions[-1].action is R
    assert node_actor(actions) == "BB"


def test_all_in_token():
    actions = parse_node_path("40100.3")
    assert actions[1].seat == "HJ"
    assert actions[1].action is J


def test_unknown_token_raises():
    with pytest.raises(ValueError, match="unrecognised action token"):
        parse_node_path("40100.zzz")


def test_too_many_actions_raises():
    # 7 actions but only 6 seats and no seat rotates back in (all folds).
    with pytest.raises(ValueError, match="no seat left"):
        parse_node_path("0.0.0.0.0.0.0")


# --- read_rng -------------------------------------------------------------
def _write_rng(path, weights):
    """weights: {index: (p, ev_sb)}; everything else p=0."""
    out = []
    for i in range(HAND_COUNT):
        out.append("????")  # pattern line (ignored by reader, as by the viewer)
        p, ev = weights.get(i, (0.0, 0.0))
        out.append(f"{p};{ev * 1000}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_read_rng_returns_present_hands(tmp_path):
    rng = tmp_path / "node.rng"
    _write_rng(rng, {0: (1.0, -1.0), 5: (0.5, 2.5)})
    entries = read_rng(rng)
    assert len(entries) == 2  # noqa: PLR2004
    by_index = {e.index: e for e in entries}
    assert by_index[0].label == monker_label(0) == "AAAA"
    assert by_index[0].p == 1.0
    assert by_index[0].ev == pytest.approx(-1.0)
    assert by_index[5].p == 0.5  # noqa: PLR2004
    assert by_index[5].ev == pytest.approx(2.5)


def test_read_rng_rejects_wrong_line_count(tmp_path):
    rng = tmp_path / "bad.rng"
    rng.write_text("????\n1.0;0.0\n", encoding="utf-8")  # only 1 entry
    with pytest.raises(ValueError, match="16,432-hand"):
        read_rng(rng)


# --- discovery ------------------------------------------------------------
def test_discover_and_range_at(tmp_path):
    nested = tmp_path / "ranges" / "Omaha" / "6-way" / "100bb"
    nested.mkdir(parents=True)
    _write_rng(nested / "40100.rng", {0: (1.0, 3.0)})
    pack = discover_plo_pack(tmp_path)
    assert pack.root == nested
    entries = range_at(pack, "40100")
    assert entries[0].index == 0
    assert entries[0].p == 1.0


def test_discover_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .rng files"):
        discover_plo_pack(tmp_path)


def test_cash_depth_packs_match_by_dir_signature(tmp_path):
    # The Aug-2026 cash depth packs (chart-export working data): each
    # stack-suffixed signature must win over the bare Omaha/6-way 100bb
    # fallback (registry ORDER MATTERS -- see KNOWN_PLO_PACKS).
    for depth in (30, 40, 50, 75, 150, 200):
        base = tmp_path / f"plo{depth}"
        nested = base / "ranges" / "Omaha" / "6-way" / f"{depth}bb[5p-1bb]"
        nested.mkdir(parents=True)
        _write_rng(nested / "40100.rng", {0: (1.0, 3.0)})
        pack = discover_plo_pack(base)
        assert pack.pack_id == f"plo_6max_{depth}bb"
        assert pack.spec.stack_bb == float(depth)
        assert pack.spec.table_size == 6
        assert pack.spec.game_format == "cash"
        assert pack.spec.ev_in_bb is False  # milli-SMALL-blind EV units
        assert pack.spec.venue_neutral is True
        assert pack.spec.default_base == f"plo{depth}_ranges"
