"""Tests for Section A concept tags -- range characterization.

Run directly (`python tests/test_concept_tags_a.py`) or under pytest. Each tag
has at least one positive and one negative case built from SpotData fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.section_a_range import (   # noqa: E402
    villain_capped, villain_linear, villain_polarized, villain_uncapped,
)
from pipeline.fact_extractor.spot_data import (                      # noqa: E402
    Combo, PopulationBaseline, RangeData, SpotData, SpotMetadata,
)


def _spot(villain_range=None, shape="", villain_top_5pct=0.0, baseline=None):
    return SpotData(
        SpotMetadata("flop"),
        range_data=RangeData(villain_range=villain_range or [],
                             villain_range_shape=shape,
                             villain_top_5pct_combos=villain_top_5pct),
        population_baseline=baseline or PopulationBaseline(),
    )


def _baseline(top_5pct):
    return PopulationBaseline(top_5pct_combos=top_5pct, populated=True)


# Bimodal: 40% strong / 15% middle / 45% weak.
_POLARIZED = [Combo(("Ah", "Kh"), 0.40, 0.90),
              Combo(("9s", "8s"), 0.15, 0.50),
              Combo(("7c", "2d"), 0.45, 0.10)]
# Smooth: 25% strong / 50% middle / 25% weak, every band represented.
_LINEAR = [Combo(("Ah", "Kh"), 0.25, 0.85),
           Combo(("Js", "Ts"), 0.50, 0.55),
           Combo(("7c", "2d"), 0.25, 0.20)]


def test_villain_capped():
    base = _baseline(20)
    # 10 < 20 * 0.70 (= 14) -> capped.
    assert villain_capped(_spot(villain_top_5pct=10, baseline=base)) is True
    # 18 sits above the 70% floor -> not capped.
    assert villain_capped(_spot(villain_top_5pct=18, baseline=base)) is False
    # No populated baseline -> cannot judge.
    assert villain_capped(_spot(villain_top_5pct=0)) is False


def test_villain_uncapped():
    base = _baseline(20)
    # 22 >= baseline 20 -> uncapped.
    assert villain_uncapped(_spot(villain_top_5pct=22, baseline=base)) is True
    # 10 < baseline -> not uncapped.
    assert villain_uncapped(_spot(villain_top_5pct=10, baseline=base)) is False
    # No populated baseline -> cannot judge.
    assert villain_uncapped(_spot(villain_top_5pct=99)) is False


def test_villain_polarized():
    assert villain_polarized(_spot(villain_range=_POLARIZED)) is True
    assert villain_polarized(_spot(villain_range=_LINEAR)) is False
    # Fallback to the pre-computed shape when no combo range is present.
    assert villain_polarized(_spot(shape="polarized")) is True
    assert villain_polarized(_spot(shape="linear")) is False


def test_villain_linear():
    assert villain_linear(_spot(villain_range=_LINEAR)) is True
    assert villain_linear(_spot(villain_range=_POLARIZED)) is False
    # Fallback to the pre-computed shape when no combo range is present.
    assert villain_linear(_spot(shape="linear")) is True
    assert villain_linear(_spot(shape="polarized")) is False


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
