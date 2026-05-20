"""Tests for Section E concept tags -- range advantage.

Run directly (`python tests/test_concept_tags_e.py`) or under pytest. Each tag
has at least one positive and one negative case built from SpotData fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.section_e_range_advantage import (  # noqa: E402
    nut_advantage_hero, nut_advantage_villain, range_advantage_hero,
    range_advantage_villain,
)
from pipeline.fact_extractor.spot_data import (                       # noqa: E402
    RangeData, SpotData, SpotMetadata,
)


def _spot(hero_eq=0.0, villain_eq=0.0, hero_strong=0.0, villain_strong=0.0,
          hero_top5=0.0, villain_top5=0.0):
    return SpotData(
        SpotMetadata("flop"),
        range_data=RangeData(
            hero_total_equity=hero_eq, villain_total_equity=villain_eq,
            hero_strong_hand_count=hero_strong,
            villain_strong_hand_count=villain_strong,
            hero_top_5pct_combos=hero_top5,
            villain_top_5pct_combos=villain_top5),
    )


def test_range_advantage_hero():
    # 60% equity, 20 strong hands vs 10 (well over the 1.3x bar).
    assert range_advantage_hero(_spot(hero_eq=0.60, hero_strong=20,
                                      villain_strong=10)) is True
    # Equity only 50% -> no range advantage despite the strong-hand edge.
    assert range_advantage_hero(_spot(hero_eq=0.50, hero_strong=20,
                                      villain_strong=10)) is False


def test_range_advantage_villain():
    assert range_advantage_villain(_spot(villain_eq=0.62, villain_strong=22,
                                         hero_strong=10)) is True
    assert range_advantage_villain(_spot(villain_eq=0.50, villain_strong=22,
                                         hero_strong=10)) is False


def test_nut_advantage_hero():
    # 10 nutted combos vs 4 -> more than 1.5x.
    assert nut_advantage_hero(_spot(hero_top5=10, villain_top5=4)) is True
    # 5 vs 4 is not a 1.5x nut advantage.
    assert nut_advantage_hero(_spot(hero_top5=5, villain_top5=4)) is False


def test_nut_advantage_villain():
    assert nut_advantage_villain(_spot(villain_top5=12, hero_top5=4)) is True
    assert nut_advantage_villain(_spot(villain_top5=5, hero_top5=4)) is False


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
