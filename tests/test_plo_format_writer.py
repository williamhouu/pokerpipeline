"""Tests for pipeline.plo.format_writer (Layer 8 CSV row + writer)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.difficulty import compute_plo_difficulty  # noqa: E402
from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.format_writer import (  # noqa: E402
    PLO_CSV_COLUMNS,
    build_plo_row,
    write_plo_csv,
)
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

AAKK = ("Ac", "Ad", "Kc", "Kd")


def R(seat: str) -> PloAction:
    return PloAction(seat, PloActionType.RAISE, 100)


def _facts(
    *,
    actor: str = "HJ",
    history: tuple[PloAction, ...] = (PloAction("LJ", PloActionType.RAISE, 100),),
    freqs: dict[str, float] | None = None,
    ev: dict[str, float] | None = None,
    archetype: str = "3bet_for_value",
    villain_seat: str | None = "LJ",
) -> PloFacts:
    node = PloDecisionNode(actor=actor, history_before=history, actions=(), history_stem="")
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=AAKK,
        action_frequencies=freqs or {"Call": 0.7, "Raise 100%": 0.3},
        ev_by_action=ev or {"Call": 2.0, "Raise 100%": 1.0, "Fold": -3.5},
        presence=1.0,
    )
    vstats = (
        PloVillainStats(seat=villain_seat, action_label="Raise 100%", weighted_combo_count=1.0, pct_of_dealt_hands=1.0)
        if villain_seat
        else None
    )
    return PloFacts(spot=spot, hand_class=classify_plo_hand(AAKK), archetype=archetype, villain_stats=vstats)


def test_schema_has_49_columns_and_no_ranges():
    assert "ranges" not in PLO_CSV_COLUMNS
    # 49-col shared template minus `ranges` (48) plus the PLO-only `hand_shape`.
    assert len(PLO_CSV_COLUMNS) == 49  # noqa: PLR2004
    # hand_shape sits right after archetype.
    i = PLO_CSV_COLUMNS.index("archetype")
    assert PLO_CSV_COLUMNS[i + 1] == "hand_shape"


def test_row_covers_full_schema_exactly():
    facts = _facts()
    row = build_plo_row(
        facts,
        difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call", "3-bet"],
        correct_answer="Call",
        number=7,
    )
    assert set(row) == set(PLO_CSV_COLUMNS)


def test_row_field_mapping():
    facts = _facts()
    row = build_plo_row(
        facts,
        difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call", "3-bet"],
        correct_answer="Call",
        number=7,
    )
    assert row["No"] == "7"
    assert (row["option 1"], row["option 2"], row["option 3"], row["option 4"]) == (
        "Fold", "Call", "3-bet", "",
    )
    assert row["Correct Answer"] == "Call"
    assert row["Answer Explanation"] == ""  # Layer 6 not built
    assert row["archetype"] == "3bet_for_value"
    assert row["Position Matchup"] == "HJ_vs_UTG"  # display codes (pack seat LJ)
    assert row["Preflop Pot Type"] == "Single raise pot"  # 1 prior raise, hero calls
    # EV gap = best(Call 2.0 sb) - 2nd(Raise 1.0 sb) = 1.0 sb = 0.5 bb.
    assert row["ev_gap_bb"] == "0.50"


def test_action_frequencies_sum_to_100():
    facts = _facts(freqs={"Call": 0.66, "Raise 100%": 0.34})
    row = build_plo_row(
        facts,
        difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call", "3-bet"],
        correct_answer="Call",
        number=1,
    )
    pcts = [int(p.rsplit(": ", 1)[1].rstrip("%")) for p in row["action_frequencies"].split(", ")]
    assert sum(pcts) == 100  # noqa: PLR2004


def test_write_csv_roundtrips(tmp_path):
    facts = _facts()
    row = build_plo_row(
        facts,
        difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call"],
        correct_answer="Call",
        number=1,
    )
    out = tmp_path / "plo.csv"
    assert write_plo_csv([row], out) == 1

    with out.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(PLO_CSV_COLUMNS)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["Correct Answer"] == "Call"
    assert "ranges" not in rows[0]
