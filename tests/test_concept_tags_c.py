"""Tests for Section C concept tags -- postflop action-type spots.

Run directly (`python tests/test_concept_tags_c.py`) or under pytest. Each tag
has at least one positive and one negative case built from SpotData fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.section_c_action_types import (  # noqa: E402
    check_raise_spot, donk_bet_spot, facing_check_raise_spot, facing_donk_spot,
    facing_overbet_spot, facing_probe_spot, overbet_spot, probe_bet_spot,
)
from pipeline.fact_extractor.spot_data import (                     # noqa: E402
    DecisionData, SpotData, SpotMetadata,
)


def _spot(street=None, prior=None, options=None, evs=None,
          option_pot_fractions=None, facing_bet_pot_fraction=None,
          in_position=False, is_pfr=False):
    return SpotData(
        SpotMetadata("flop", hero_in_position=in_position,
                     hero_is_preflop_raiser=is_pfr),
        decision_data=DecisionData(
            street_actions=street or [], prior_street_actions=prior or [],
            options=options or [], hero_combo_evs=evs or {},
            option_pot_fractions=option_pot_fractions or {},
            facing_bet_pot_fraction=facing_bet_pot_fraction),
    )


def test_check_raise_spot():
    line = [("hero", "check"), ("villain", "bet")]
    # Hero checked, villain bet, raise option is the best EV.
    assert check_raise_spot(_spot(street=line, options=["fold", "call", "raise"],
                                  evs={"fold": 0.0, "call": 0.3, "raise": 0.5})) is True
    # The raise option's EV is far below the best -> not a check-raise spot.
    assert check_raise_spot(_spot(street=line, options=["fold", "call", "raise"],
                                  evs={"fold": 0.0, "call": 2.0, "raise": -1.0})) is False


def test_facing_check_raise_spot():
    # Villain checked, hero bet, villain raised.
    assert facing_check_raise_spot(_spot(street=[("villain", "check"),
                                                 ("hero", "bet"),
                                                 ("villain", "raise")])) is True
    # No villain raise -> hero isn't facing a check-raise.
    assert facing_check_raise_spot(_spot(street=[("hero", "check"),
                                                 ("villain", "bet")])) is False


def test_donk_bet_spot():
    evs = {"check": 0.2, "bet": 0.5}
    # Hero OOP, not the preflop raiser, first to act, bet is near-best.
    assert donk_bet_spot(_spot(options=["check", "bet"], evs=evs,
                               in_position=False, is_pfr=False)) is True
    # Hero is in position -> leading isn't a donk bet.
    assert donk_bet_spot(_spot(options=["check", "bet"], evs=evs,
                               in_position=True, is_pfr=False)) is False


def test_facing_donk_spot():
    # Hero is the preflop raiser and in position; villain led out.
    assert facing_donk_spot(_spot(street=[("villain", "bet")],
                                  in_position=True, is_pfr=True)) is True
    # Villain checked rather than betting -> not a donk.
    assert facing_donk_spot(_spot(street=[("villain", "check")],
                                  in_position=True, is_pfr=True)) is False


def test_probe_bet_spot():
    evs = {"check": 0.1, "bet": 0.4}
    # Villain checked the previous street; hero considers a near-best bet.
    assert probe_bet_spot(_spot(prior=[("hero", "check"), ("villain", "check")],
                                options=["check", "bet"], evs=evs)) is True
    # Villain bet the previous street -> no revealed weakness to probe.
    assert probe_bet_spot(_spot(prior=[("hero", "check"), ("villain", "bet")],
                                options=["check", "bet"], evs=evs)) is False


def test_facing_probe_spot():
    # Hero checked back last street; villain bet into hero this street.
    assert facing_probe_spot(_spot(prior=[("villain", "check"), ("hero", "check")],
                                   street=[("villain", "bet")])) is True
    # Villain checked again -> hero isn't facing a probe.
    assert facing_probe_spot(_spot(prior=[("villain", "check"), ("hero", "check")],
                                   street=[("villain", "check")])) is False


def test_overbet_spot():
    # A 125%-pot option exists and its EV is the best.
    assert overbet_spot(_spot(option_pot_fractions={"overbet": 1.25},
                              evs={"check": 0.2, "overbet": 0.5})) is True
    # The largest sizing is 75% pot -> no overbet option.
    assert overbet_spot(_spot(option_pot_fractions={"bet": 0.75},
                              evs={"check": 0.2, "bet": 0.5})) is False


def test_facing_overbet_spot():
    # Villain bet 125% of the pot.
    assert facing_overbet_spot(_spot(facing_bet_pot_fraction=1.25)) is True
    # Villain bet 75% of the pot -- a normal-sized bet.
    assert facing_overbet_spot(_spot(facing_bet_pot_fraction=0.75)) is False


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
