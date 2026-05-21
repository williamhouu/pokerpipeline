"""Tests for pipeline.question_extractor (Layer 4).

Run directly (`python tests/test_question_extractor.py`) or under pytest.
Covers the two filters at their boundaries and the difficulty formula.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.spot_data import (                       # noqa: E402
    DecisionData, SpotData, SpotMetadata,
)
from pipeline.question_extractor import (                             # noqa: E402
    difficulty_score, evaluate_spot, is_question_worthy,
    top_action_frequency,
)


def _spot(top_frequency: float, ev_gap_bb: float) -> SpotData:
    """A spot whose most-played action has exactly the given frequency.

    The remainder is split into pieces no larger than `top_frequency`, so the
    given value is genuinely the maximum even when it is below 50%.
    """
    strategy = {"top": top_frequency}
    remaining = 1.0 - top_frequency
    piece = 0
    while remaining > 1e-9:
        chunk = min(top_frequency, remaining)
        strategy[f"other_{piece}"] = chunk
        remaining -= chunk
        piece += 1
    return SpotData(
        SpotMetadata("flop"),
        decision_data=DecisionData(range_aggregate_strategy=strategy,
                                   ev_gap_bb=ev_gap_bb),
    )


def test_top_action_frequency():
    assert top_action_frequency(_spot(0.72, 1.0)) == 0.72
    assert top_action_frequency(SpotData(SpotMetadata("flop"))) == 0.0   # no data


def test_frequency_filter_boundaries():
    # 55% and 95% are inside the window (inclusive); just outside fails.
    assert is_question_worthy(_spot(0.55, 1.0)) is True
    assert is_question_worthy(_spot(0.95, 1.0)) is True
    assert is_question_worthy(_spot(0.5499, 1.0)) is False   # just below 55%
    assert is_question_worthy(_spot(0.9501, 1.0)) is False   # just above 95%


def test_ev_gap_filter_boundary():
    # The EV gap filter is "at least" -- exactly 0.5bb passes.
    assert is_question_worthy(_spot(0.70, 0.50)) is True
    assert is_question_worthy(_spot(0.70, 0.49)) is False
    assert is_question_worthy(_spot(0.70, 2.0)) is True


def test_both_filters_must_pass():
    # Good frequency but a thin EV gap -> not worthy.
    assert is_question_worthy(_spot(0.70, 0.1)) is False
    # Good EV gap but a too-obvious frequency -> not worthy.
    assert is_question_worthy(_spot(0.98, 3.0)) is False
    # Both pass.
    assert is_question_worthy(_spot(0.70, 1.0)) is True


def test_difficulty_formula():
    # 55% -> hardest (3000); 95% -> easiest (500); 75% -> midpoint (1750).
    assert difficulty_score(_spot(0.55, 1.0)) == 3000
    assert difficulty_score(_spot(0.95, 1.0)) == 500
    assert difficulty_score(_spot(0.75, 1.0)) == 1750


def test_difficulty_clamped_outside_range():
    # Frequencies outside the 55-95% window clamp to the score bounds.
    assert difficulty_score(_spot(0.40, 1.0)) == 3000
    assert difficulty_score(_spot(0.99, 1.0)) == 500


def test_evaluate_spot():
    verdict = evaluate_spot(_spot(0.70, 1.2))
    assert verdict.is_worthy is True
    assert verdict.top_action_frequency == 0.70
    assert verdict.ev_gap_bb == 1.2
    assert verdict.difficulty_score == difficulty_score(_spot(0.70, 1.2))


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
