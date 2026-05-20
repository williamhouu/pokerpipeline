"""Tests for pipeline.fact_extractor.spot_data.

Run directly (`python tests/test_spot_data.py`) or under pytest. Covers
construction (minimal and fully populated), validation (every guarded field),
and lossless serialization (dict and JSON round-trips).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.board_texture import classify_board       # noqa: E402
from pipeline.fact_extractor.hand_class import classify_hand           # noqa: E402
from pipeline.fact_extractor.spot_data import (                        # noqa: E402
    GAME_FORMATS, STREETS, BoardTexture, Combo, DecisionData, EquityData,
    HandClass, PopulationBaseline, RangeData, SpotData, SpotMetadata,
    lookup_population_baseline,
)


def _full_spot() -> SpotData:
    """A richly populated SpotData -- the brief's BB-vs-BTN worked example."""
    return SpotData(
        spot_metadata=SpotMetadata(
            street="flop", stack_depth_bb=100, spr=4.2,
            position_dynamic="BB_vs_BTN", hero_position="BB",
            preflop_raise_count=1, active_players_on_flop=2,
            parent_node_id="flop_node_142", action_to_reach="check_bet33_call"),
        decision_data=DecisionData(
            options=["fold", "call", "raise"],
            hero_combo_evs={"fold": 0.0, "call": 0.5, "raise": -2.0},
            hero_combo_strategy={"call": 1.0},
            range_aggregate_strategy={"fold": 0.40, "call": 0.50, "raise": 0.10},
            correct_action="call", ev_gap_bb=0.5,
            option_pot_fractions={"raise": 1.0},
            facing_bet_pot_fraction=1.0,
            street_actions=[("hero", "check"), ("villain", "bet")]),
        equity_data=EquityData(
            hero_raw_equity_vs_continuing=0.38, pot_odds_required=0.33,
            equity_realization_ratio=0.90),
        range_data=RangeData(
            villain_range=[Combo(("Ah", "Kh"), 0.6, 0.71),
                           Combo(("7d", "7s"), 1.0, 0.55)],
            villain_range_shape="polarized",
            villain_value_combos=24, villain_bluff_combos=36,
            hero_total_equity=0.42, villain_total_equity=0.58),
        hand_class=HandClass.from_cards("Ah Ad", "8c 7c 6d"),
        board_texture=BoardTexture.from_board("8c 7c 6d"),
        population_baseline=lookup_population_baseline("BB_vs_BTN", "srp_cbet"),
        concept_tags=["bluffcatch_spot", "villain_polarized"],
    )


# --- construction ------------------------------------------------------------
def test_minimal_construction_applies_defaults():
    spot = SpotData(SpotMetadata(street="flop"))
    assert spot.spot_metadata.street == "flop"
    assert spot.spot_metadata.stack_depth_bb == 100.0
    assert isinstance(spot.decision_data, DecisionData)
    assert isinstance(spot.equity_data, EquityData)
    assert isinstance(spot.range_data, RangeData)
    assert spot.hand_class is None and spot.board_texture is None
    assert spot.equity_data.equity_realization_ratio == 1.0   # neutral default
    assert spot.concept_tags == []


def test_full_construction():
    spot = _full_spot()
    assert spot.spot_metadata.position_dynamic == "BB_vs_BTN"
    assert spot.decision_data.correct_action == "call"
    assert len(spot.range_data.villain_range) == 2
    assert spot.range_data.villain_range[0].cards == ("Ah", "Kh")
    assert spot.concept_tags == ["bluffcatch_spot", "villain_polarized"]


def test_hand_class_and_board_texture_reference_the_modules():
    # The HandClass / BoardTexture wrappers must mirror the module output.
    assert HandClass.from_cards("Ah Ad", "8c 7c 6d").__dict__ == \
        classify_hand("Ah Ad", "8c 7c 6d")
    assert BoardTexture.from_board("8c 7c 6d").__dict__ == \
        classify_board("8c 7c 6d")


def test_combo_normalises_cards():
    combo = Combo(("ah", "kS"))                  # mixed case is normalised
    assert combo.cards == ("Ah", "Ks")
    assert combo.weight == 1.0 and combo.equity == 0.0


def test_population_baseline_lookup_is_unpopulated():
    baseline = lookup_population_baseline("BB_vs_BTN", "srp_cbet")
    assert baseline.populated is False
    assert baseline.top_5pct_combos == 0.0
    assert PopulationBaseline().populated is False        # bare default too


def test_constants_exposed():
    assert STREETS == ("preflop", "flop", "turn", "river")
    assert GAME_FORMATS == ("cash", "tournament")


# --- validation --------------------------------------------------------------
def _expect_error(label, error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"expected {error.__name__} for {label}")


def test_validation_rejects_bad_input():
    _expect_error("bad street", ValueError,
                  lambda: SpotMetadata(street="fifth"))
    _expect_error("bad game_format", ValueError,
                  lambda: SpotMetadata(street="flop", game_format="sng"))
    _expect_error("negative spr", ValueError,
                  lambda: SpotMetadata(street="flop", spr=-1.0))
    _expect_error("too few players", ValueError,
                  lambda: SpotMetadata(street="flop", active_players_on_flop=1))
    _expect_error("equity above 1", ValueError,
                  lambda: EquityData(hero_raw_equity_vs_continuing=1.4))
    _expect_error("EQR gap out of range", ValueError,
                  lambda: EquityData(equity_realization_gap=2.0))
    _expect_error("frequency above 1", ValueError,
                  lambda: DecisionData(hero_combo_strategy={"call": 1.5}))
    _expect_error("blocker pct above 1", ValueError,
                  lambda: RangeData(hero_blocks_value_pct=1.2))
    _expect_error("negative combo count", ValueError,
                  lambda: RangeData(villain_value_combos=-3))
    _expect_error("duplicate combo cards", ValueError,
                  lambda: Combo(("Ah", "Ah")))
    _expect_error("combo weight above 1", ValueError,
                  lambda: Combo(("Ah", "Kh"), weight=1.5))
    _expect_error("non-string concept tag", ValueError,
                  lambda: SpotData(SpotMetadata("flop"), concept_tags=[1, 2]))
    _expect_error("wrong section type", TypeError,
                  lambda: SpotData(SpotMetadata("flop"), equity_data={}))


# --- serialization -----------------------------------------------------------
def test_to_dict_has_data_block_shape():
    block = _full_spot().to_dict()
    assert set(block) == {"spot_metadata", "decision_data", "equity_data",
                          "range_data", "hand_class", "board_texture",
                          "population_baseline", "concept_tags"}
    assert block["hand_class"]["made_hand"] == "overpair"
    assert block["range_data"]["villain_range"][0]["cards"] == ("Ah", "Kh")


def test_dict_round_trip():
    for label, spot in (("minimal", SpotData(SpotMetadata("preflop"))),
                        ("full", _full_spot())):
        restored = SpotData.from_dict(spot.to_dict())
        assert restored == spot, f"{label} dict round-trip changed the spot"


def test_json_round_trip():
    for label, spot in (("minimal", SpotData(SpotMetadata("turn"))),
                        ("full", _full_spot())):
        restored = SpotData.from_json(spot.to_json())
        assert restored == spot, f"{label} JSON round-trip changed the spot"


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
