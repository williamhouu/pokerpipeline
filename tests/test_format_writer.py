"""Tests for pipeline.format_writer (Layer 8).

Run directly (`python tests/test_format_writer.py`) or under pytest. Uses a
synthetic SpotData -- no PioSolver needed.
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.spot_data import (                       # noqa: E402
    BoardTexture, DecisionData, HandClass, SpotData, SpotMetadata,
)
from pipeline.format_writer import CSV_COLUMNS, build_row, write_csv  # noqa: E402


def _spot() -> SpotData:
    """A populated SpotData fixture for a flop BB-vs-BTN single-raised pot."""
    return SpotData(
        SpotMetadata("flop", hero_position="BB", villain_position="BTN",
                     position_dynamic="BB_vs_BTN", game_format="cash",
                     preflop_raise_count=1, active_players_on_flop=2,
                     stack_depth_bb=98.0, effective_stack_bb=98.0,
                     node_id="r:0:c:b36", hero_cards=("Ah", "Kh"),
                     board=["2c", "Js", "7s"]),
        decision_data=DecisionData(correct_action="bet", ev_gap_bb=1.37),
        hand_class=HandClass("top_pair_top_kicker", strength_bucket="strong",
                             label="top_pair_top_kicker_no_draws"),
        board_texture=BoardTexture("two_tone", "unpaired", "disconnected",
                                   "middling", "semi_wet"),
        concept_tags=["single_raised_pot", "range_advantage_hero"],
    )


def test_build_row_has_every_column():
    row = build_row(_spot(), 1500)
    assert set(row) == set(CSV_COLUMNS)
    assert len(row) == len(CSV_COLUMNS)


def test_new_pipeline_columns():
    row = build_row(_spot(), 1500)
    assert row["concept_tags"] == "single_raised_pot, range_advantage_hero"
    assert row["hand_class"] == "top_pair_top_kicker_no_draws"
    assert row["board_texture"] == "semi_wet"
    assert row["solver_reference"] == "r:0:c:b36"
    assert row["ev_gap_bb"] == "1.37"


def test_classification_columns_filled_from_spotdata():
    row = build_row(_spot(), 1500)
    assert row["Hand Stage"] == "Flop"
    assert row["Correct Answer"] == "bet"
    assert row["Difficulty Rating"] == "1500"
    assert row["Cash/Tourney"] == "Cash"
    assert row["Preflop Pot Type"] == "Single Raised Pot"
    assert row["Pot Participant"] == "Heads-Up"
    assert row["User Cards"] == "Ah Kh"
    assert row["Cards on Table"] == "2c Js 7s"


def test_placeholders_for_unbuilt_columns():
    row = build_row(_spot(), 1500)
    for column in ("Question", "Option 1", "Option 4", "Answer Explanation"):
        assert row[column] == "[TBD by Layer 6]", column
    for column in ("Table Size", "Seats", "POT"):
        assert row[column] == "[TBD]", column


def test_write_csv_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "questions.csv"        # parent auto-created
        count = write_csv(path, [(_spot(), 1500), (_spot(), 800)])
        assert count == 2
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            data_rows = list(reader)
        assert header == CSV_COLUMNS
        assert len(data_rows) == 2
        # The two rows differ only in difficulty.
        difficulty_index = CSV_COLUMNS.index("Difficulty Rating")
        assert [r[difficulty_index] for r in data_rows] == ["1500", "800"]


if __name__ == "__main__":
    suite = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in suite:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
