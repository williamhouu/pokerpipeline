"""Tests for the PLO 9-max pack integration (July 2026, multi-pack era).

Covers the pack registry + seat/token grammar, the table-size-aware seat
display and position logic, and an end-to-end batch from a synthetic 9-max
mini pack (deterministic, no API).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.action_history import display_seat  # noqa: E402
from pipeline.plo.hand_order import HAND_COUNT  # noqa: E402
from pipeline.plo.pack import (  # noqa: E402
    KNOWN_PLO_PACKS,
    SEATS_9MAX,
    PloActionType,
    PloPack,
    discover_plo_pack,
    parse_node_path,
)
from pipeline.plo.position import ip_oop_positions, position_bucket  # noqa: E402


# --- grammar -----------------------------------------------------------------
def test_token_2_is_a_pot_raise():
    actions = parse_node_path("2.0.1", seats=SEATS_9MAX)
    assert [(a.seat, a.action) for a in actions] == [
        ("UTG", PloActionType.RAISE),
        ("UTG+1", PloActionType.FOLD),
        ("UTG+2", PloActionType.CALL),
    ]
    assert actions[0].raise_pct == 100  # pot-limit's one raise = the pot


def test_nine_seat_queue_rotates_raiser_to_the_back():
    # UTG opens, everyone folds to the BB, BB 3-bets pot, UTG must act again.
    stem = "2." + ".".join(["0"] * 7) + ".2"
    actions = parse_node_path(stem, seats=SEATS_9MAX)
    assert actions[0].seat == "UTG"
    assert actions[-1].seat == "BB"
    follow = parse_node_path(stem + ".1", seats=SEATS_9MAX)
    assert follow[-1].seat == "UTG"  # the opener re-acts vs the 3-bet


def test_six_max_grammar_is_unchanged():
    actions = parse_node_path("40100.0")
    assert [(a.seat, a.action) for a in actions] == [
        ("LJ", PloActionType.RAISE),
        ("HJ", PloActionType.FOLD),
    ]
    assert actions[0].raise_pct == 100


# --- registry / discovery ------------------------------------------------------
def _touch_rng(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n0.0;0\n", encoding="utf-8")


def test_discovery_matches_specs_by_path_signature(tmp_path):
    _touch_rng(tmp_path / "nine/ranges/Omaha/9-way/100bb/2.rng")
    _touch_rng(tmp_path / "six/ranges/Omaha/6-way/100bb(5p-1bb)/40100.rng")
    _touch_rng(tmp_path / "odd/somewhere/else/40100.rng")

    nine = discover_plo_pack(tmp_path / "nine")
    assert nine.pack_id == "plo_9max_100bb"
    assert nine.table_size == 9
    assert nine.seats == SEATS_9MAX

    six = discover_plo_pack(tmp_path / "six")
    assert six.pack_id == "plo_6max_100bb"
    assert six.table_size == 6

    # Unknown layout -> the legacy 6-max spec (pre-registry behavior).
    odd = discover_plo_pack(tmp_path / "odd")
    assert odd.pack_id == "plo_6max_100bb"


def test_registry_ids_are_unique():
    ids = [s.pack_id for s in KNOWN_PLO_PACKS]
    assert len(ids) == len(set(ids))


def test_nine_max_60bb_signature_wins_over_bare_9way(tmp_path):
    # The Aug-2026 60bb 9-max cash depth: its stack-suffixed signature must
    # match BEFORE the bare "Omaha/9-way" 100bb spec (registry ORDER
    # MATTERS -- see the KNOWN_PLO_PACKS invariant comment).
    _touch_rng(tmp_path / "sixty/ranges/Omaha/9-way/60bb[5p-2bb]/2.rng")
    sixty = discover_plo_pack(tmp_path / "sixty")
    assert sixty.pack_id == "plo_9max_60bb"
    assert sixty.table_size == 9
    assert sixty.seats == SEATS_9MAX
    assert sixty.spec.stack_bb == 60.0  # noqa: PLR2004
    assert sixty.spec.game_format == "cash"
    assert sixty.spec.ev_in_bb is False  # milli-SMALL-blind EV units
    assert sixty.spec.default_base == "plo9_60_ranges"

    # A bare 9-way root (no stack folder) still resolves to the 100bb spec.
    _touch_rng(tmp_path / "nine/ranges/Omaha/9-way/100bb/2.rng")
    assert discover_plo_pack(tmp_path / "nine").pack_id == "plo_9max_100bb"


# --- seat display + positions ----------------------------------------------------
def test_display_seat_is_table_size_aware():
    # 6-max: Monker dialect remap (its LJ IS the UTG-equivalent seat).
    assert display_seat("LJ") == "UTG"
    assert display_seat("BU") == "BTN"
    # 9-max: identity -- its LJ is a REAL Lojack, and remapping it to UTG
    # would corrupt every seat reference (the collision this design avoids).
    assert display_seat("LJ", table_size=9) == "LJ"
    assert display_seat("UTG+1", table_size=9) == "UTG+1"
    assert display_seat("BTN", table_size=9) == "BTN"


def test_nine_max_postflop_position_order():
    assert ip_oop_positions("UTG", "BB") == ("UTG", "BB")   # blinds act first
    assert ip_oop_positions("UTG+1", "BTN") == ("BTN", "UTG+1")
    assert ip_oop_positions("SB", "BB") == ("BB", "SB")     # ring-table BvB rule


def test_position_buckets_differ_by_table_size():
    # The pivotal case: LJ is EARLY at 6-max (it's the first seat) but
    # MIDDLE at 9-max (three UTG seats act before it).
    assert position_bucket("LJ", table_size=6) == "early"
    assert position_bucket("LJ", table_size=9) == "middle"
    assert position_bucket("UTG+2", table_size=9) == "early"
    assert position_bucket("CO", table_size=6) == "late"
    assert position_bucket("BTN", table_size=9) == "late"
    assert position_bucket("SB", table_size=9) == "sb"
    with pytest.raises(ValueError, match="unknown seat"):
        position_bucket("BTN", table_size=6)  # BTN is not a 6-max pack code


# --- context line -------------------------------------------------------------
def test_nine_max_context_is_effective_stacks_only():
    """9-max pack questions: the Context says ONLY the effective stack size
    (team ask, July 2026). 6-max keeps the full stakes/venue framing."""
    from pipeline.plo.action_history import format_plo_context

    assert format_plo_context(table_size=9) == "$100 effective stacks."
    assert (
        format_plo_context(table_size=9, display_in_bb=True)
        == "100bb effective stacks."
    )
    assert (
        format_plo_context(table_size=9, stakes_bb_dollars=2.0, stack_bb=200.0)
        == "$400 effective stacks."
    )
    # 6-max (the default) is byte-identical to before.
    assert format_plo_context() == (
        "$0.5/$1 Online PLO cash. $100 effective stacks."
    )


# --- end-to-end: a synthetic 9-max mini pack ---------------------------------------
def _write_rng(path: Path, p: float) -> None:
    out = []
    for _ in range(HAND_COUNT):
        out.append("x")
        out.append(f"{p};0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _mini_9max_pack(tmp_path: Path) -> PloPack:
    """UTG opens pot; UTG+1's fold/call/raise decision is the one worthy node."""
    root = tmp_path / "ranges" / "Omaha" / "9-way" / "100bb"
    _write_rng(root / "2.rng", 1.0)      # UTG's open range (the villain file)
    _write_rng(root / "2.0.rng", 0.3)    # UTG+1 folds 30%
    _write_rng(root / "2.1.rng", 0.7)    # UTG+1 calls 70%
    _write_rng(root / "2.2.rng", 0.0)    # UTG+1 3-bets 0%
    return discover_plo_pack(tmp_path)


def test_nine_max_batch_end_to_end(tmp_path):
    from pipeline.plo.batch import generate_plo_batch

    pack = _mini_9max_pack(tmp_path)
    assert pack.table_size == 9
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False,
    )
    assert result.questions_written == 1

    with out.open(encoding="utf-8-sig") as handle:
        row = next(iter(csv.DictReader(handle)))
    # Seat rendering: hero is a REAL UTG+1, villain a REAL UTG; the open is
    # the pot-limit 3.5bb (default dollar display at $0.5/$1).
    assert row["Question"].startswith("You're UTG+1 with ")
    assert "UTG opens to $3.50." in row["Question"]
    assert row["Table Size"] == "9"
    # 9-max is labelled Live in the CSV (team ask, July 2026): full-ring PLO is
    # a live-casino format. (The Context prose still omits the venue.)
    assert row["Live or Online"] == "Live"
    assert row["Position Matchup"] == "UTG+1_vs_UTG"
    # The node reference lives in Notes' Node: field now (July 2026).
    from pipeline.provenance import node_reference_from_notes

    assert node_reference_from_notes(row["Notes"]).startswith("plo_9max_100bb/")
    assert "solver_reference" not in row or not row.get("solver_reference")
    # The app Seats tokens carry the 9-max seat names and the blind folds.
    assert "UTG-" in row["Seats"]
    assert "SB-" in row["Seats"] and "BB-" in row["Seats"]

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["pack_id"] == "plo_9max_100bb"
    assert meta["table_size"] == 9
    assert meta["pack_label"] == "plo_9max_100bb"
