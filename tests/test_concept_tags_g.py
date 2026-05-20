"""Tests for Section G concept tags -- pot and action context.

Run directly (`python tests/test_concept_tags_g.py`) or under pytest. Each tag
has at least one positive and one negative case built from SpotData fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.section_g_pot_context import (  # noqa: E402
    blind_defense_spot, blind_vs_blind_pot, four_bet_pot, multiway_pot,
    short_stack_tournament, single_raised_pot, squeeze_pot, three_bet_pot,
)
from pipeline.fact_extractor.spot_data import SpotData, SpotMetadata  # noqa: E402


def _spot(raise_count=0, had_cold_caller=False, active_players=2,
          hero_position="", villain_position="", street="flop",
          game_format="cash", effective_stack=100.0, is_pfr=False):
    return SpotData(SpotMetadata(
        street, preflop_raise_count=raise_count, had_cold_caller=had_cold_caller,
        active_players_on_flop=active_players, hero_position=hero_position,
        villain_position=villain_position, game_format=game_format,
        effective_stack_bb=effective_stack, hero_is_preflop_raiser=is_pfr))


def test_three_bet_pot():
    assert three_bet_pot(_spot(raise_count=2)) is True
    assert three_bet_pot(_spot(raise_count=1)) is False


def test_four_bet_pot():
    assert four_bet_pot(_spot(raise_count=3)) is True
    assert four_bet_pot(_spot(raise_count=2)) is False


def test_single_raised_pot():
    assert single_raised_pot(_spot(raise_count=1)) is True
    assert single_raised_pot(_spot(raise_count=2)) is False


def test_squeeze_pot():
    # A 3-bet pot (two raises) with a cold-caller before the 3-bet.
    assert squeeze_pot(_spot(raise_count=2, had_cold_caller=True)) is True
    # A 3-bet pot with no cold-caller is a standard 3-bet pot, not a squeeze.
    assert squeeze_pot(_spot(raise_count=2, had_cold_caller=False)) is False


def test_multiway_pot():
    assert multiway_pot(_spot(active_players=3)) is True
    assert multiway_pot(_spot(active_players=2)) is False


def test_blind_vs_blind_pot():
    assert blind_vs_blind_pot(_spot(hero_position="SB",
                                    villain_position="BB")) is True
    # Opponent is the Button -> not a blind-vs-blind pot.
    assert blind_vs_blind_pot(_spot(hero_position="BB",
                                    villain_position="BTN")) is False


def test_blind_defense_spot():
    # Hero in the BB, preflop, facing a single open, hasn't raised.
    assert blind_defense_spot(_spot(hero_position="BB", street="preflop",
                                    raise_count=1)) is True
    # Two preflop raises -> hero faces a 3-bet, not an open.
    assert blind_defense_spot(_spot(hero_position="BB", street="preflop",
                                    raise_count=2)) is False


def test_short_stack_tournament():
    assert short_stack_tournament(_spot(game_format="tournament",
                                        effective_stack=15)) is True
    # A 50bb tournament stack is too deep for push/fold math to dominate.
    assert short_stack_tournament(_spot(game_format="tournament",
                                        effective_stack=50)) is False


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
