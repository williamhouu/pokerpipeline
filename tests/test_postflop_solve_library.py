"""Tests for postflop solve self-description + spot selection.

Covers the pieces that make the admin solve-picker work: metadata-derived
scenario fields (table size / positions / cash-vs-tournament), the
``.db`` summary + directory discovery, and the hero-filter / diversify
spot selector. No real solve file is needed -- the metadata tests build a
tiny in-memory ``.db`` with only a ``metadata`` table.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.postflop.adapters.sqlite_db import (  # noqa: E402
    _first_position,
    derive_scenario,
    discover_db_solves,
    summarize_db,
)
from pipeline.postflop.spot_selection import (  # noqa: E402
    combo_class,
    diversify_spots,
    make_spot_selector,
    node_kind,
)

_V8_META = {
    "spot": "BTN_SRP_9max_noante_100bb",
    "game_format": "9max NLHE",
    "flop": "QsJd9s",
    "stack_bb": "100",
    "ante": "None",
    "rake": "10% cap 3bb (300 chips)",
    "ip_range": "BTN_open_81pot.txt (Monker)",
    "oop_range": "BB_call_vs_BTN_81pot.txt (Monker)",
    "customer_id": "Ryan_2026_06_18_v8",
    "solve_date": "Thu Jun 18 01:38:36 2026",
}


# --- scenario derivation -----------------------------------------------------
def test_derive_scenario_reads_table_size_positions_format() -> None:
    sc = derive_scenario(_V8_META)
    assert sc["table_size"] == 9
    assert sc["ip_position"] == "BTN"
    assert sc["oop_position"] == "BB"
    assert sc["game_format"] == "cash"


def test_derive_scenario_six_max_and_tournament() -> None:
    six = derive_scenario({"game_format": "6max NLHE", "spot": "CO_SRP_6max"})
    assert six["table_size"] == 6
    # An ante (not "None"/0) marks a tournament.
    tourney = derive_scenario({"game_format": "9max NLHE", "ante": "0.125"})
    assert tourney["game_format"] == "tournament"


def test_derive_scenario_degrades_gracefully() -> None:
    # Nothing parseable -> just the cash default, no crash, no bogus keys.
    sc = derive_scenario({})
    assert sc == {"game_format": "cash"}


def test_first_position_token_based() -> None:
    assert _first_position("BTN_open_81pot.txt") == "BTN"
    assert _first_position("BB_call_vs_BTN") == "BB"
    assert _first_position("UTG+1_open.rng") == "UTG+1"
    assert _first_position("CO_3bet_vs_BTN") == "CO"
    assert _first_position("garbage.txt") is None
    assert _first_position("") is None


# --- .db summary + discovery (tiny in-memory metadata-only db) ---------------
def _make_meta_db(path: Path, meta: dict[str, str]) -> None:
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.executemany("INSERT INTO metadata VALUES (?, ?)", list(meta.items()))
    con.commit()
    con.close()


def test_summarize_db_builds_a_label(tmp_path: Path) -> None:
    db = tmp_path / "v8.db"
    _make_meta_db(db, _V8_META)
    s = summarize_db(str(db))
    assert s.ok
    assert s.table_size == 9
    assert s.ip_position == "BTN" and s.oop_position == "BB"
    assert s.flop_pretty == "Qs Jd 9s"
    assert s.stack_bb == 100.0
    assert "BTN vs BB" in s.label and "9-max" in s.label and "Qs Jd 9s" in s.label


def test_summarize_db_flags_a_non_solve(tmp_path: Path) -> None:
    db = tmp_path / "notasolve.db"
    _make_meta_db(db, {"hello": "world"})  # no flop/spot
    s = summarize_db(str(db))
    assert not s.ok and s.error


def test_discover_db_solves_scans_recursively(tmp_path: Path) -> None:
    _make_meta_db(tmp_path / "a.db", _V8_META)
    sub = tmp_path / "nested"
    sub.mkdir()
    _make_meta_db(sub / "b.db", {**_V8_META, "spot": "BTN_SRP_6max", "game_format": "6max"})
    found = discover_db_solves(str(tmp_path))
    assert len(found) == 2
    assert all(s.ok for s in found)
    assert discover_db_solves(str(tmp_path / "does_not_exist")) == []


# --- spot selection ----------------------------------------------------------
@dataclass
class _Node:
    node_id: str
    actor: str
    street: str = "flop"
    is_facing_bet: bool = False


@dataclass
class _Spot:
    node: _Node
    hero_combo: str
    dominant_verb: str = "check"


def test_node_kind_and_combo_class() -> None:
    assert node_kind("r:0") == "bb_lead"
    assert node_kind("r:0:c") == "btn_cbet"
    assert node_kind("r:0:b216") == "btn_faces_donk"
    assert node_kind("r:0:c:b216") == "bb_faces_cbet"
    assert combo_class("AsKs") == "AKs"
    assert combo_class("3s3c") == "33"
    assert combo_class("AhKd") == "AKo"


def test_make_spot_selector_filters_by_hero() -> None:
    spots = [
        _Spot(_Node("r:0:c", "BTN"), "AsKs"),
        _Spot(_Node("r:0", "BB"), "7h7d"),
        _Spot(_Node("r:0:c:b216", "BB"), "QsJs"),
    ]
    bb_only = make_spot_selector(heroes=("BB",))(spots)
    assert {s.node.actor for s in bb_only} == {"BB"}
    assert len(bb_only) == 2
    both = make_spot_selector(heroes=None)(spots)
    assert len(both) == 3


def test_diversify_round_robins_decision_types() -> None:
    spots = [
        _Spot(_Node("r:0:c", "BTN"), "AsKs"),  # btn_cbet
        _Spot(_Node("r:0", "BB"), "7h7d"),  # bb_lead
        _Spot(_Node("r:0:c:b216", "BB"), "QsJs"),  # bb_faces_cbet
    ]
    out = diversify_spots(spots)
    assert len(out) == 3
    # First out is the highest-priority kind present (btn_cbet).
    assert out[0].node.node_id == "r:0:c"


def test_diversify_keeps_turn_river_and_drops_raise_wars() -> None:
    spots = [
        # a flop c-bet (kept), a turn bet, a turn facing-bet, a river bet.
        _Spot(_Node("r:0:c", "BTN", street="flop"), "AsKs", dominant_verb="bet"),
        _Spot(_Node("r:0:c:c:2c", "BB", street="turn"), "7h7d", dominant_verb="bet"),
        _Spot(_Node("r:0:c:c:2c:b216", "BTN", street="turn", is_facing_bet=True),
              "QsJs", dominant_verb="call"),
        _Spot(_Node("r:0:c:c:2c:c:c:7h", "BB", street="river"), "8h8d",
              dominant_verb="bet"),
        # a re-raise war on the turn -> dropped (3 bets on one street).
        _Spot(_Node("r:0:c:c:2c:b216:b440:b900", "BB", street="turn",
                    is_facing_bet=True), "AcAd", dominant_verb="call"),
        # an all-in line -> dropped.
        _Spot(_Node("r:0:c:b9697", "BB", street="flop", is_facing_bet=True),
              "KsKd", dominant_verb="call"),
    ]
    out = diversify_spots(spots)
    ids = {s.node.node_id for s in out}
    assert "r:0:c:c:2c:b216:b440:b900" not in ids  # raise war dropped
    assert "r:0:c:b9697" not in ids  # all-in dropped
    # the flop + the three turn/river spots survive, spread across streets.
    assert {s.node.street for s in out} == {"flop", "turn", "river"}
    assert len(out) == 4
