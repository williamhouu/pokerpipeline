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
from pipeline.scenario_config import SCENARIOS                        # noqa: E402


def _spot(effective_stack_bb: float = 98.0,
          rank_distribution: str = "middling",
          action_sequence=None,
          pot_bb: float = 5.5) -> SpotData:
    """A populated SpotData fixture for a flop BB-vs-BTN single-raised pot."""
    return SpotData(
        SpotMetadata("flop", hero_position="BB", villain_position="BTN",
                     position_dynamic="BB_vs_BTN", game_format="cash",
                     preflop_raise_count=1, active_players_on_flop=2,
                     stack_depth_bb=100.0, effective_stack_bb=effective_stack_bb,
                     node_id="r:0:c:b36", hero_cards=("Ah", "Kh"),
                     board=["2c", "Js", "7s"],
                     action_sequence=action_sequence or [],
                     big_blind_chips=87.75, pot_bb=pot_bb),
        decision_data=DecisionData(correct_action="bet", ev_gap_bb=1.37),
        hand_class=HandClass("top_pair_top_kicker", strength_bucket="strong",
                             label="top_pair_top_kicker_no_draws"),
        board_texture=BoardTexture("two_tone", "unpaired", "disconnected",
                                   rank_distribution, "semi_wet"),
        concept_tags=["single_raised_pot", "range_advantage_hero"],
    )


def test_column_structure():
    """Column-count history:
      * 35 baseline (.xlsx Sheet 1)
      * 36 = +action_frequencies (Apr 2026 Ryan-feedback Fix 3)
      * 38 = +ip_range +oop_range (May 2026 Ryan ask: UI range-grid columns)
      * 39 = +skills (May 2026 Phase 3: user-facing skill labels)
      * 40 = +archetype (May 2026: surface preflop strategic frame for QA)
      * 45 = +easy_freq +easy_ev +easy_concept +easy_hand
             +difficulty_bumps (May 2026: 4-axis difficulty diagnostic
             columns -- the per-axis breakdown of the Difficulty Rating
             so reviewers can see WHY a spot got its score).
      * 43 = -tag_1 -tag_2 -tag_3 (dropped), +Position Matchup (the
             hero-vs-villain seat matchup, split out of Relative
             Position when that column was repurposed to IP/OOP), and
             a reorder: Notes + hand_class moved to the tail. (May 2026.)
      * 44 = +ranges (May 2026: position-labeled JSON of every active
             player's range -- multiway-capable, supersedes ip/oop).
      * 42 = -ip_range -oop_range (dropped May 2026: superseded by the
             `ranges` column, which is a strict superset).
      * 40 = -difficulty_bumps (always empty -- BUMP_RULES unpopulated)
             -hand_class (duplicated User Cards on preflop); Notes moved
             to sit right after concept_tags. (June 2026.)
      * 46 = +6 decision-math columns (pot_odds, hero_equity, range_equity,
             blocker_combos, top_villain_combos, stat_notes) -- the "Show
             the math" panel data, appended at the end. (June 2026.)
      * 47 = +claim_check (Layer-7 claim-checker output, opt-in). (June 2026.)
      * 48 = +exploit_notes (deterministic exploit adjustments). (June 2026.)
      * 49 = +action_ev_bb (per-action solver EV for the hero hand, bb).
             (June 2026.)
    """
    assert len(CSV_COLUMNS) == 49
    assert CSV_COLUMNS[-1] == "action_ev_bb"
    assert CSV_COLUMNS[-2] == "exploit_notes"
    assert CSV_COLUMNS[-3] == "claim_check"
    assert "hand_class" not in CSV_COLUMNS
    assert "difficulty_bumps" not in CSV_COLUMNS
    assert CSV_COLUMNS[0] == "No"
    # ip_range/oop_range are gone; ranges carries every active player's range.
    assert "ip_range" not in CSV_COLUMNS and "oop_range" not in CSV_COLUMNS
    assert "ranges" in CSV_COLUMNS
    # tag_1/2/3 are gone.
    assert "tag_1" not in CSV_COLUMNS
    # Cluster after Difficulty Rating: skills, action_frequencies, ev_gap_bb,
    # concept_tags.
    diff_i = CSV_COLUMNS.index("Difficulty Rating")
    assert CSV_COLUMNS[diff_i + 1:diff_i + 5] == [
        "skills", "action_frequencies", "ev_gap_bb", "concept_tags"]
    # Notes sits right after concept_tags (June 2026), then Position Matchup,
    # then ranges. The QA/provenance cluster follows the strategic frame:
    # archetype -> board_texture -> solver_reference -> validation_status.
    assert CSV_COLUMNS[CSV_COLUMNS.index("concept_tags") + 1] == "Notes"
    assert CSV_COLUMNS[CSV_COLUMNS.index("Notes") + 1] == "Position Matchup"
    assert CSV_COLUMNS[CSV_COLUMNS.index("Position Matchup") + 1] == "ranges"
    arch_i = CSV_COLUMNS.index("archetype")
    assert CSV_COLUMNS[arch_i:arch_i + 4] == [
        "archetype", "board_texture", "solver_reference", "validation_status"]
    # The four difficulty diagnostic axes, followed by the six decision-math
    # columns that now close out the row (June 2026).
    assert CSV_COLUMNS[-13:-9] == [
        "easy_freq", "easy_ev", "easy_concept", "easy_hand"]
    assert CSV_COLUMNS[-9:-3] == [
        "pot_odds", "hero_equity", "range_equity", "blocker_combos",
        "top_villain_combos", "stat_notes"]
    assert CSV_COLUMNS[-3:] == ["claim_check", "exploit_notes", "action_ev_bb"]
    # Header casing fixes (unchanged from the 35-column era).
    assert ["option 1", "option 2", "option 3", "option 4"] == CSV_COLUMNS[12:16]
    assert "Live or Online" in CSV_COLUMNS and "Live/Online" not in CSV_COLUMNS
    row = build_row(_spot(), 1500, 1)
    assert set(row) == set(CSV_COLUMNS) and len(row) == 49
    # Postflop rows always have empty archetype (the column is only
    # populated by the preflop writer).
    assert row["archetype"] == ""
    # Relative Position is hero's IP/OOP standing; the seat matchup moved
    # to Position Matchup. The _spot() fixture has hero_in_position=False.
    assert row["Relative Position"] == "Out of Position"
    assert row["Position Matchup"] == "BB_vs_BTN"


def test_no_and_validation_status():
    assert build_row(_spot(), 1500, 7)["No"] == "7"
    assert build_row(_spot(), 1500, 1)["validation_status"] == "auto_approved"


def test_new_pipeline_columns():
    row = build_row(_spot(), 1500, 1)
    assert row["concept_tags"] == "single_raised_pot, range_advantage_hero"
    # board_texture is the 3-axis string, not the composite word.
    assert row["board_texture"] == "two_tone_disconnected_middling"
    # solver_reference is a descriptive cache path, not a Pio node id.
    assert row["solver_reference"] == (
        "PioSolver_Cash_100bb/BB_vs_BTN/single_raised_pot/flop_2cJs7s/AhKh")
    assert row["ev_gap_bb"] == "1.37"


def test_action_frequencies_column_renders_descending_percentages():
    """Fix 3: action_frequencies summarises Pio's range strategy as
    '<verb>: <integer>%' entries, descending by frequency."""
    from pipeline.fact_extractor.spot_data import DecisionData
    spot = _spot()
    # Replace the decision with one that has an explicit strategy.
    object.__setattr__(spot, "decision_data", DecisionData(
        correct_action="call",
        range_aggregate_strategy={"call": 0.601, "fold": 0.199,
                                  "raise": 0.10, "all-in": 0.10},
        ev_gap_bb=0.74))
    row = build_row(spot, 1500, 1)
    assert row["action_frequencies"] == \
        "call: 60%, fold: 20%, raise: 10%, all-in: 10%"


def test_action_frequencies_column_handles_empty_strategy():
    """When the strategy is empty, the column is the empty string -- the
    row is still valid, the field is just blank."""
    from pipeline.fact_extractor.spot_data import DecisionData
    spot = _spot()
    object.__setattr__(spot, "decision_data", DecisionData(
        correct_action="bet", range_aggregate_strategy={}, ev_gap_bb=0.0))
    row = build_row(spot, 1500, 1)
    assert row["action_frequencies"] == ""


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
    """Without a scenario, the scenario-derived columns stay as `[TBD]` and the
    LLM-written columns stay as `[TBD by Layer 6]` -- the legacy default the
    pre-Layer-1 pipeline relied on."""
    row = build_row(_spot(), 1500, 1)
    for column in ("Context", "Question", "option 1", "option 4",
                   "Answer Explanation"):
        assert row[column] == "[TBD by Layer 6]", column
    for column in ("Table Size", "Seats", "POT", "Live or Online"):
        assert row[column] == "[TBD]", column


def test_scenario_populated_columns():
    """With a scenario, the seven previously-TBD columns now fill from real data.

    Context comes from scenario.context; Question is the deterministic action
    history with suit emojis; Table Size, Default Stack, Live or Online, Seats,
    POT all come from scenario fields (or scenario * spot_metadata math).
    """
    scenario = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    # Flop spot, hero (BB) facing a BTN cbet of 41 chips (~$0.23).
    spot = _spot(action_sequence=[("OOP", "check"), ("IP", "bet 41")],
                 pot_bb=5.96)              # ~$2.98 pot after the cbet
    row = build_row(spot, 1500, 1, scenario=scenario)

    # No more [TBD] in any of the seven scenario-driven columns.
    for column in ("Context", "Question", "Table Size", "Default Stack",
                   "Live or Online", "Seats", "POT"):
        assert "[TBD" not in row[column], (column, row[column])

    # Spot-check each value.
    # Whole-dollar stack drops trailing '.00' per Ryan-feedback Fix 1 (May 2026).
    assert row["Context"] == "6-Handed, $0.25/$0.50, Stacks $50"
    assert row["Table Size"] == "6"
    assert row["Default Stack"] == "$50"
    assert row["Live or Online"] == "Online"
    assert row["Seats"].startswith("BTN-$")                 # villain seat + stack
    assert row["POT"].startswith("$")                       # dollar-prefixed
    # Question starts with the team's voice; references hero hole cards with
    # suit emoji; includes the preflop line; includes a Flop section.
    q = row["Question"]
    assert q.startswith("You're in the Big Blind with A")
    assert "The Button opens to $1.25" in q                 # _VILLAIN_REF rendering
    assert "You call." in q
    assert "Flop ($2.75)" in q                              # preflop pot rendered
    # Postflop conversion fires: 41 chips ~ $0.234 raw -> snaps to $0.25
    # under Fix 1's round-to-nearest-SB step (Ryan-feedback Apr 2026).
    assert "$0.25" in q


def test_scenario_question_has_no_tbd_token():
    """The Question column under a scenario must never contain '[TBD' --
    that's how the v3 batch counts 'TBDs eliminated'."""
    scenario = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    row = build_row(_spot(action_sequence=[("OOP", "check")]), 1500, 1,
                    scenario=scenario)
    assert "[TBD" not in row["Question"]
    assert "[TBD" not in row["Context"]


def test_write_csv_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "questions.csv"        # parent auto-created
        count = write_csv(path, [(_spot(), 1500), (_spot(), 800)])
        assert count == 2
        with open(path, newline="", encoding="utf-8-sig") as handle:
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
