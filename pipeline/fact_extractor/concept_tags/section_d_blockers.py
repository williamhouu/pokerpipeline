"""Concept tags -- Section D: Blocker Effects (4 tags).

From docs/engineering_brief.docx, "Concept Tag Library Specification",
Section D. Each tag is a pure function: SpotData -> bool.

Every rule is a threshold check on the blocker percentages the fact extractor
pre-computes from combo counts on villain's continuing range:
range_data.hero_blocks_value_pct and range_data.hero_blocks_bluffs_pct -- the
fraction of villain's value / bluff combos that hero's specific cards remove.

Numeric thresholds are the brief's starting values; they will be tuned against
the ~800-explanation gold pool before the tagger goes to production.
"""
from __future__ import annotations

from pipeline.fact_extractor.spot_data import SpotData


def blocks_value_unblocks_bluffs(spot: SpotData) -> bool:
    """Hero removes value combos while leaving bluff combos intact.

    Brief definition: hero's specific hand removes value combos from villain's
    range while leaving bluff combos intact; shifts the bluff-to-value ratio in
    hero's favour when facing aggression.

    Brief rule: hero_blocks_value_pct > 0.15 AND hero_blocks_bluffs_pct < 0.05.
    """
    range_data = spot.range_data
    return (range_data.hero_blocks_value_pct > 0.15
            and range_data.hero_blocks_bluffs_pct < 0.05)


def blocks_bluffs_unblocks_value(spot: SpotData) -> bool:
    """Hero removes bluff combos while leaving value intact.

    Brief definition: the reverse -- hero removes bluff combos while leaving
    value intact; makes a call less profitable.

    Brief rule: hero_blocks_bluffs_pct > 0.15 AND hero_blocks_value_pct < 0.05.
    """
    range_data = spot.range_data
    return (range_data.hero_blocks_bluffs_pct > 0.15
            and range_data.hero_blocks_value_pct < 0.05)


def blocks_value(spot: SpotData) -> bool:
    """Hero has a significant value blocker effect.

    Brief definition: significant value blocker effect without strong
    implications for bluffs.

    Brief rule: hero_blocks_value_pct > 0.20.
    """
    return spot.range_data.hero_blocks_value_pct > 0.20


def no_blocker_effects(spot: SpotData) -> bool:
    """Hero's combo neither blocks nor unblocks meaningfully.

    Brief definition: hero's specific combo doesn't significantly block or
    unblock either value or bluffs.

    Brief rule: hero_blocks_value_pct < 0.10 AND hero_blocks_bluffs_pct < 0.10.
    """
    range_data = spot.range_data
    return (range_data.hero_blocks_value_pct < 0.10
            and range_data.hero_blocks_bluffs_pct < 0.10)
