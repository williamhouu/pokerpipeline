"""Tests for the concept tag registry and compute_tags.

Run directly (`python tests/test_concept_tags_registry.py`) or under pytest.
Covers registry completeness and that compute_tags collects exactly the firing
tags in section order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.registry import (            # noqa: E402
    TAG_REGISTRY, compute_tags,
)
from pipeline.fact_extractor.spot_data import (                        # noqa: E402
    EquityData, SpotData, SpotMetadata,
)

# The 42 concept tags of Sections A-G (Section H is deferred per the brief).
EXPECTED_TAGS = {
    # A -- range characterization
    "villain_capped", "villain_uncapped", "villain_polarized", "villain_linear",
    # B -- decision class
    "thin_value_spot", "bluffcatch_spot", "protection_bet_spot",
    "merged_value_spot", "pot_control_spot", "equity_denial_spot", "bluff_spot",
    "commitment_threshold_decision", "float_call_spot",
    # C -- postflop action-type spots
    "check_raise_spot", "facing_check_raise_spot", "donk_bet_spot",
    "facing_donk_spot", "probe_bet_spot", "facing_probe_spot", "overbet_spot",
    "facing_overbet_spot",
    # D -- blocker effects
    "blocks_value_unblocks_bluffs", "blocks_bluffs_unblocks_value",
    "blocks_value", "no_blocker_effects",
    # E -- range advantage
    "range_advantage_hero", "range_advantage_villain", "nut_advantage_hero",
    "nut_advantage_villain",
    # F -- equity & math concepts
    "equity_under_realized", "equity_over_realized", "implied_odds_call",
    "reverse_implied_odds_call", "mdf_defense_threshold",
    # G -- pot and action context
    "3bet_pot", "4bet_pot", "single_raised_pot", "squeeze_pot", "multiway_pot",
    "blind_vs_blind_pot", "blind_defense_spot", "short_stack_tournament",
}


def test_registry_has_all_42_tags():
    assert len(TAG_REGISTRY) == 42
    assert set(TAG_REGISTRY) == EXPECTED_TAGS
    assert all(callable(fn) for fn in TAG_REGISTRY.values())


def test_compute_tags_collects_firing_tags_in_section_order():
    # Single raised pot, multiway, default (zero) blockers, EQR under-realised.
    spot = SpotData(
        SpotMetadata("flop", preflop_raise_count=1, active_players_on_flop=3),
        equity_data=EquityData(equity_realization_ratio=0.80),
    )
    # Sections D, F, G fire -- and the result is ordered A->G.
    assert compute_tags(spot) == [
        "no_blocker_effects",       # Section D
        "equity_under_realized",    # Section F
        "single_raised_pot",        # Section G
        "multiway_pot",             # Section G
    ]


def test_compute_tags_on_a_bare_spot():
    # A minimal spot fires only no_blocker_effects (default zero block pcts).
    tags = compute_tags(SpotData(SpotMetadata("flop")))
    assert tags == ["no_blocker_effects"]


def test_compute_tags_only_returns_registered_names():
    spot = SpotData(SpotMetadata("turn", preflop_raise_count=2,
                                 active_players_on_flop=3))
    tags = compute_tags(spot)
    assert all(tag in TAG_REGISTRY for tag in tags)
    assert "3bet_pot" in tags                    # raise count 2 -> 3-bet pot


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
