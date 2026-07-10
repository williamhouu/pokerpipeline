"""Tests for the per-solve street-by-street structure report.

Builds a tiny synthetic vendor ``.db`` (only the columns the report reads:
``gto_postflop.node``/``action`` + ``metadata``) shaped like the known v7
truncations, and checks the walk, the menu labels, the auto-derived
limitations, and the sidecar cache staleness rules -- all browserless.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.postflop.structure_report import (  # noqa: E402
    compute_and_cache_structure_report,
    compute_structure_report,
    load_structure_report,
)

_META = {
    "spot": "BTN_vs_BB_SRP_8max_noante_200bb",
    "game_format": "8max NLHE",
    "flop": "ThTd5c",
    "pot": "650",
    "pot_bb": "6.5",
    "eff_stack": "19700",
    "eff_stack_bb": "197",
    "stack_bb": "200",
    "ante": "0",
    "rake": "8% cap 2bb",
    "preflop_line": "BTN open 3bb, BB call",
    "ip_range": "BTN_open_raise3_200bb",
    "oop_range": "BB_call_vs_BTN_open_200bb",
}

# A miniature v7-shaped tree: flop with sizes + an all-in-only raise, a
# 67%-only turn, and a river where the IP player can never bet after a check.
_NODES = {
    "r:0": ["CHECK", "BET_214"],                       # BB first: check / bet 33%
    "r:0:c": ["CHECK", "BET_214"],                     # BTN after check
    "r:0:c:b214": ["FOLD", "CALL", "BET_19700"],       # BB facing bet: all-in raise only
    "r:0:c:c:2h": ["CHECK", "BET_436"],                # turn BB first: bet 67% only
    "r:0:c:c:2h:c": ["CHECK"],                         # turn BTN after check: no bet
    "r:0:c:c:2h:c:c:2d": ["CHECK"],                    # river BB first: no bet
    "r:0:c:c:2h:c:c:2d:c": ["CHECK"],                  # river BTN after check: CHECK ONLY
}


def _make_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.executemany("INSERT INTO metadata VALUES (?, ?)", list(_META.items()))
    con.execute(
        "CREATE TABLE gto_postflop (node TEXT, action TEXT, freq_blob BLOB,"
        " ev_blob_oop BLOB, ev_blob_ip BLOB)"
    )
    for node, actions in _NODES.items():
        for a in actions:
            con.execute(
                "INSERT INTO gto_postflop VALUES (?, ?, ?, ?, ?)",
                (node, a, b"", b"", b""),
            )
    con.commit()
    con.close()


def test_report_menus_and_geometry(tmp_path: Path) -> None:
    db = tmp_path / "mini.db"
    _make_db(db)
    rep = compute_structure_report(db)
    assert rep["oop"] == "BB" and rep["ip"] == "BTN"
    assert rep["start_pot_bb"] == 6.5 and rep["eff_bb"] == 197.0

    flop = {(r["actor"], r["context"]): r for r in rep["streets"]["flop"]}
    # BB first to act: passive first, bet labelled as % of the current pot.
    assert flop[("BB", "first to act")]["menus"][0]["options"] == ["Check", "Bet 33%"]
    # BB facing the 214 bet: the only raise is the all-in.
    facing = flop[("BB", "facing a bet")]["menus"][0]["options"]
    assert facing == ["Fold", "Call", "All-in raise"]
    # Turn (via the check-check flop line, so the pot is still 650 chips):
    # BB's single 436-chip barrel labels as 67% of the pot.
    turn = {(r["actor"], r["context"]): r for r in rep["streets"]["turn"]}
    assert turn[("BB", "first to act")]["menus"][0]["options"] == ["Check", "Bet 67%"]


def test_report_limitations(tmp_path: Path) -> None:
    db = tmp_path / "mini.db"
    _make_db(db)
    notes = compute_structure_report(db)["limitations"]
    text = "\n".join(notes)
    # The three v7-shaped truncations must all be called out.
    assert "After a river check, BTN can never bet" in text
    assert "Facing a flop bet, the only raise in the tree is all-in" in text
    assert "Turn bets come in one size only" in text
    # River first-in has no bet either -> the sparse-betting note fires.
    assert "river first-in/after-check spots offer no bet" in text


def test_sidecar_cache_and_staleness(tmp_path: Path) -> None:
    db = tmp_path / "mini.db"
    _make_db(db)
    assert load_structure_report(db) is None  # nothing cached yet
    rep = compute_and_cache_structure_report(db)
    assert load_structure_report(db) == rep  # fresh cache serves
    # Touching the file (size or mtime change) invalidates the sidecar.
    with open(db, "ab") as fh:
        fh.write(b"x")
    os.utime(db, (1, 1))
    assert load_structure_report(db) is None
