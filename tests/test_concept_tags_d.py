"""Tests for Section D concept tags -- blocker effects.

Run directly (`python tests/test_concept_tags_d.py`) or under pytest. Each tag
has at least one positive and one negative case built from SpotData fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.concept_tags.section_d_blockers import (  # noqa: E402
    blocks_bluffs_unblocks_value, blocks_value, blocks_value_unblocks_bluffs,
    no_blocker_effects,
)
from pipeline.fact_extractor.spot_data import (                       # noqa: E402
    RangeData, SpotData, SpotMetadata,
)


def _spot(value_pct=0.0, bluffs_pct=0.0):
    return SpotData(
        SpotMetadata("flop"),
        range_data=RangeData(hero_blocks_value_pct=value_pct,
                             hero_blocks_bluffs_pct=bluffs_pct),
    )


def test_blocks_value_unblocks_bluffs():
    # Blocks 20% of value, only 2% of bluffs.
    assert blocks_value_unblocks_bluffs(_spot(value_pct=0.20, bluffs_pct=0.02)) is True
    # Blocks too many bluffs (10%) to count as unblocking them.
    assert blocks_value_unblocks_bluffs(_spot(value_pct=0.20, bluffs_pct=0.10)) is False


def test_blocks_bluffs_unblocks_value():
    # Blocks 20% of bluffs, only 2% of value.
    assert blocks_bluffs_unblocks_value(_spot(value_pct=0.02, bluffs_pct=0.20)) is True
    # Blocks too much value (10%) to count as unblocking it.
    assert blocks_bluffs_unblocks_value(_spot(value_pct=0.10, bluffs_pct=0.20)) is False


def test_blocks_value():
    # 25% of villain's value combos removed.
    assert blocks_value(_spot(value_pct=0.25)) is True
    # 15% is below the 20% significance threshold.
    assert blocks_value(_spot(value_pct=0.15)) is False


def test_no_blocker_effects():
    # Both block percentages are negligible.
    assert no_blocker_effects(_spot(value_pct=0.05, bluffs_pct=0.05)) is True
    # A 15% value blocker effect is not negligible.
    assert no_blocker_effects(_spot(value_pct=0.15, bluffs_pct=0.05)) is False


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
