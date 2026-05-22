"""Tests for Section F concept tags -- equity & math concepts.

Run directly (`python tests/test_concept_tags_f.py`) or under pytest. Each tag
has at least one positive and one negative case built from SpotData fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.section_f_equity_math import (  # noqa: E402
    equity_over_realized, equity_under_realized, implied_odds_call,
    mdf_defense_threshold, reverse_implied_odds_call,
)
from pipeline.fact_extractor.spot_data import (                       # noqa: E402
    DecisionData, EquityData, HandClass, SpotData, SpotMetadata,
)


def _spot(eqr=1.0, raw_equity=0.0, pot_odds=0.0, mdf=0.0, call_ev=None,
          range_strategy=None, facing_bet=None, hand_class=None):
    evs = {} if call_ev is None else {"call": call_ev}
    return SpotData(
        SpotMetadata("flop"),
        decision_data=DecisionData(range_mean_evs_per_action=evs,
                                   range_aggregate_strategy=range_strategy or {},
                                   facing_bet_pot_fraction=facing_bet),
        equity_data=EquityData(equity_realization_ratio=eqr,
                               hero_raw_equity_vs_continuing=raw_equity,
                               pot_odds_required=pot_odds, mdf=mdf),
        hand_class=hand_class,
    )


def test_equity_under_realized():
    assert equity_under_realized(_spot(eqr=0.80)) is True
    assert equity_under_realized(_spot(eqr=1.00)) is False


def test_equity_over_realized():
    assert equity_over_realized(_spot(eqr=1.10)) is True
    assert equity_over_realized(_spot(eqr=1.00)) is False


def test_implied_odds_call():
    drawing = HandClass("no_pair_air", draws=["flush_draw_nut"],
                        strength_bucket="air")
    made = HandClass("top_pair_good_kicker", strength_bucket="medium")
    # Raw equity below pot odds, the solver call EV is positive, hero is drawing.
    assert implied_odds_call(_spot(raw_equity=0.25, pot_odds=0.33, call_ev=0.5,
                                   hand_class=drawing)) is True
    # Same numbers, but a made hand -> not an implied-odds call.
    assert implied_odds_call(_spot(raw_equity=0.25, pot_odds=0.33, call_ev=0.5,
                                   hand_class=made)) is False


def test_reverse_implied_odds_call():
    # Raw odds say easy call (0.50 > 0.46), but the solver call EV is thin and
    # equity realises poorly.
    assert reverse_implied_odds_call(_spot(raw_equity=0.50, pot_odds=0.40,
                                           call_ev=0.1, eqr=0.80)) is True
    # A healthy solver call EV -> the solver doesn't disagree with the raw math.
    assert reverse_implied_odds_call(_spot(raw_equity=0.50, pot_odds=0.40,
                                           call_ev=0.5, eqr=0.80)) is False


def test_mdf_defense_threshold():
    # Facing a half-pot bet; the range defends 67%, MDF is 67%.
    assert mdf_defense_threshold(_spot(facing_bet=0.5, mdf=0.67,
        range_strategy={"fold": 0.33, "call": 0.67})) is True
    # The range defends 90% -- far above the 67% MDF.
    assert mdf_defense_threshold(_spot(facing_bet=0.5, mdf=0.67,
        range_strategy={"fold": 0.10, "call": 0.90})) is False


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
