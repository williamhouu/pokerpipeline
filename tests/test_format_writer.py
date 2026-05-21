"""Tests for pipeline.format_writer (Layer 8).

Run directly (`python tests/test_format_writer.py`) or under pytest. Uses a
synthetic SpotData -- no PioSolver needed. Asserts the 35-column structure and
the field conventions from docs/output_format_examples.xlsx.
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


def _spot(effective_stack_bb: float = 98.0,
          rank_distribution: str = "middling") -> SpotData:
    """A populated SpotData fixture for a flop BB-vs-BTN single-raised pot."""
    return SpotData(
        SpotMetadata("flop", hero_position="BB", villain_position="BTN",
                     position_dynamic="BB_vs_BTN", game_format="cash",
                     preflop_raise_count=1, active_players_on_flop=2,
                     stack_depth_bb=100.0, effective_stack_bb=effective_stack_bb,
                     node_id="r:0:c:b36", hero_cards=("Ah", "Kh"),
                     board=["2c", "Js", "7s"]),
        decision_data=DecisionData(correct_action="bet", ev_gap_bb=1.37),
        hand_class=HandClass("top_pair_top_kicker", strength_bucket="strong",
                             label="top_pair_top_kicker_no_draws"),
        board_texture=BoardTexture("two_tone", "unpaired", "disconnected",
                                   rank_distribution, "semi_wet"),
        concept_tags=["single_raised_pot", "range_advantage_hero"],
    )


def test_thirty_five_column_structure():
    assert len(CSV_COLUMNS) == 35
    # The columns the .xlsx sample adds beyond the original 28.
    assert CSV_COLUMNS[0] == "No"
    assert CSV_COLUMNS[-1] == "validation_status"
    # Header casing fixes.
    assert ["option 1", "option 2", "option 3", "option 4"] == CSV_COLUMNS[12:16]
    assert "Live or Online" in CSV_COLUMNS and "Live/Online" not in CSV_COLUMNS
    row = build_row(_spot(), 1500, 1)
    assert set(row) == set(CSV_COLUMNS) and len(row) == 35


def test_no_and_validation_status():
    assert build_row(_spot(), 1500, 7)["No"] == "7"
    assert build_row(_spot(), 1500, 1)["validation_status"] == "auto_approved"


def test_new_pipeline_columns():
    row = build_row(_spot(), 1500, 1)
    assert row["concept_tags"] == "single_raised_pot, range_advantage_hero"
    assert row["hand_class"] == "top_pair_top_kicker_no_draws"
    # board_texture is the 3-axis string, not the composite word.
    assert row["board_texture"] == "two_tone_disconnected_middling"
    # solver_reference is a descriptive cache path, not a Pio node id.
    assert row["solver_reference"] == (
        "PioSolver_Cash_100bb/BB_vs_BTN/single_raised_pot/flop_2cJs7s/AhKh")
    assert row["ev_gap_bb"] == "1.37"


def test_board_texture_broadway_alias():
    # The board_texture module's 'broadway_heavy' maps to the sample's 'broadway'.
    row = build_row(_spot(rank_distribution="broadway_heavy"), 1500, 1)
    assert row["board_texture"] == "two_tone_disconnected_broadway"


def test_stack_depth_buckets():
    assert build_row(_spot(effective_stack_bb=20), 1500, 1)["Stack Depth"] == "Short stack"
    assert build_row(_spot(effective_stack_bb=98), 1500, 1)["Stack Depth"] == "Standard Stack"
    assert build_row(_spot(effective_stack_bb=250), 1500, 1)["Stack Depth"] == "Deep stack"


def test_classification_wording():
    row = build_row(_spot(), 1500, 1)
    assert row["Hand Stage"] == "Flop"
    assert row["Correct Answer"] == "bet"
    assert row["Difficulty Rating"] == "1500"
    assert row["Cash/Tourney"] == "Cash"
    assert row["Preflop Pot Type"] == "Single raise pot"     # not "3-Bet Pot" style
    assert row["Pot Participant"] == "Heads-Up"               # "Multi-Way" when 3+
    assert row["User Cards"] == "Ah Kh"
    assert row["Cards on Table"] == "2c Js 7s"


def test_placeholders_for_unbuilt_columns():
    row = build_row(_spot(), 1500, 1)
    for column in ("Question", "option 1", "option 4", "Answer Explanation"):
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
        # The No column auto-increments across the output.
        no_index = CSV_COLUMNS.index("No")
        assert [r[no_index] for r in data_rows] == ["1", "2"]
        status_index = CSV_COLUMNS.index("validation_status")
        assert all(r[status_index] == "auto_approved" for r in data_rows)


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
