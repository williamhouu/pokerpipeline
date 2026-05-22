"""Tests for Section B concept tags -- decision class.

Run directly (`python tests/test_concept_tags_b.py`) or under pytest. Each tag
has at least one positive and one negative case built from SpotData fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.section_b_decision_class import (  # noqa: E402
    bluff_spot, bluffcatch_spot, commitment_threshold_decision,
    equity_denial_spot, float_call_spot, merged_value_spot, pot_control_spot,
    protection_bet_spot, thin_value_spot,
)
from pipeline.fact_extractor.spot_data import (                      # noqa: E402
    DecisionData, EquityData, HandClass, RangeData, SpotData, SpotMetadata,
)


def _spot(evs=None, equity=None, range_data=None, hand_class=None,
          metadata=None, **decision_kw):
    return SpotData(
        metadata or SpotMetadata("flop"),
        decision_data=DecisionData(range_mean_evs_per_action=evs or {}, **decision_kw),
        equity_data=equity or EquityData(),
        range_data=range_data or RangeData(),
        hand_class=hand_class,
    )


def test_thin_value_spot():
    range_data = RangeData(villain_calling_range_worse_pct=0.80)
    equity = EquityData(hero_raw_equity_vs_calling=0.60)
    # bet beats check by 0.6bb, calling range 80% worse, equity in 50-70%.
    assert thin_value_spot(_spot(evs={"bet": 1.0, "check": 0.4},
                                 equity=equity, range_data=range_data)) is True
    # EV gain of 2.6bb is outside the thin 0.3-1.0bb window.
    assert thin_value_spot(_spot(evs={"bet": 3.0, "check": 0.4},
                                 equity=equity, range_data=range_data)) is False


def test_bluffcatch_spot():
    equity = EquityData(hero_raw_equity_vs_continuing=0.42)
    polarized = RangeData(villain_range_shape="polarized")
    # 0.42 equity, no later-street value, polarized villain.
    assert bluffcatch_spot(_spot(equity=equity, range_data=polarized,
                                 hero_has_later_street_value=False)) is True
    # Hero has later-street value -> not a pure bluffcatch.
    assert bluffcatch_spot(_spot(equity=equity, range_data=polarized,
                                 hero_has_later_street_value=True)) is False


def test_protection_bet_spot():
    vulnerable = HandClass("top_pair_weak_kicker", strength_bucket="vulnerable")
    evs = {"bet": 1.0, "check": 0.5}
    assert protection_bet_spot(_spot(
        evs=evs, hand_class=vulnerable,
        range_data=RangeData(villain_draw_equity_pct=0.40))) is True
    # Villain has little draw equity -> nothing to protect against.
    assert protection_bet_spot(_spot(
        evs=evs, hand_class=vulnerable,
        range_data=RangeData(villain_draw_equity_pct=0.10))) is False


def test_merged_value_spot():
    evs = {"bet": 1.0, "check": 0.5}
    range_data = RangeData(villain_calling_range_has_worse_made_hands=True,
                           villain_calling_range_has_draws=True)
    # Small sizing (33% pot) -> merged value.
    assert merged_value_spot(_spot(evs=evs, range_data=range_data,
                                   option_pot_fractions={"bet": 0.33})) is True
    # 75% pot is not a small, merged sizing.
    assert merged_value_spot(_spot(evs=evs, range_data=range_data,
                                   option_pot_fractions={"bet": 0.75})) is False


def test_pot_control_spot():
    medium = HandClass("two_pair_mid", strength_bucket="medium")
    evs = {"check": 1.0, "bet": 0.4}
    # Check beats bet, medium hand, hero OOP.
    assert pot_control_spot(_spot(evs=evs, hand_class=medium,
                                  metadata=SpotMetadata("flop"))) is True
    # In position with a deep SPR -> betting/barrelling isn't committing.
    assert pot_control_spot(_spot(
        evs=evs, hand_class=medium,
        metadata=SpotMetadata("flop", hero_in_position=True, spr=8.0))) is False


def test_equity_denial_spot():
    marginal = HandClass("bottom_pair", strength_bucket="marginal")
    evs = {"bet": 1.0, "check": 0.5}
    # Checking vs the whole range beats betting vs only the callers.
    assert equity_denial_spot(_spot(evs=evs, hand_class=marginal,
                                    check_ev_vs_villain_range=1.0,
                                    bet_ev_vs_calling_range=0.5)) is True
    # Betting vs callers wins -> this is value, not equity denial.
    assert equity_denial_spot(_spot(evs=evs, hand_class=marginal,
                                    check_ev_vs_villain_range=0.2,
                                    bet_ev_vs_calling_range=0.5)) is False


def test_bluff_spot():
    evs = {"bet": 1.0, "check": 0.3}
    low_equity = EquityData(hero_raw_equity_vs_continuing=0.20)
    # Low equity, fold equity dominates showdown value.
    assert bluff_spot(_spot(evs=evs, equity=low_equity,
                            estimated_fold_equity=2.0,
                            estimated_showdown_value=0.3)) is True
    # Showdown value dominates -> betting isn't a bluff.
    assert bluff_spot(_spot(evs=evs, equity=low_equity,
                            estimated_fold_equity=0.1,
                            estimated_showdown_value=0.5)) is False


def test_commitment_threshold_decision():
    # Post-decision SPR below 1.5.
    assert commitment_threshold_decision(_spot(post_decision_spr=1.0)) is True
    # More than half the stack committed by the call.
    assert commitment_threshold_decision(
        _spot(stack_committed_fraction=0.60)) is True
    # Deep SPR, little of the stack in -> not a commitment decision.
    assert commitment_threshold_decision(
        _spot(post_decision_spr=4.0, stack_committed_fraction=0.20)) is False


def test_float_call_spot():
    low_equity = EquityData(hero_raw_equity_vs_continuing=0.30)
    evs = {"call": 0.5, "fold": 0.0}
    # Call beats fold, low equity, the float line is +EV.
    assert float_call_spot(_spot(evs=evs, equity=low_equity,
                                 float_line_is_positive=True)) is True
    # Float line not +EV -> the weak call isn't a float.
    assert float_call_spot(_spot(evs=evs, equity=low_equity,
                                 float_line_is_positive=False)) is False


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
