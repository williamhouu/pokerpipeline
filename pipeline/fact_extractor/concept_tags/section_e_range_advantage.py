"""Concept tags -- Section E: Range Advantage (4 tags).

From docs/engineering_brief.docx, "Concept Tag Library Specification",
Section E. Each tag is a pure function: SpotData -> bool.

These tags compare the two players' ranges -- overall equity and the counts of
strong / nutted combos -- using the aggregates the fact extractor pre-computes
into range_data.

Numeric thresholds are the brief's starting values; they will be tuned against
the ~800-explanation gold pool before the tagger goes to production.
"""
from __future__ import annotations

from pipeline.fact_extractor.spot_data import SpotData

# "more strong hands by at least 30%" -> at least 1.30x the other count;
# "more nutted combos x 1.5" -> at least 1.5x. Both use a strict `>` (matching
# the brief's notation and avoiding a degenerate fire when both counts are 0).
_STRONG_HAND_RATIO = 1.30
_NUT_COMBO_RATIO = 1.5


def range_advantage_hero(spot: SpotData) -> bool:
    """Hero's overall range is stronger than villain's.

    Brief definition: hero's overall range is stronger than villain's. Common
    postflop when hero is the preflop raiser and the flop favours the raising
    range.

    Brief rule: hero_total_equity > 0.55 AND hero_strong_hand_count >
    villain_strong_hand_count by at least 30%.
    """
    rd = spot.range_data
    return (rd.hero_total_equity > 0.55
            and rd.hero_strong_hand_count
            > rd.villain_strong_hand_count * _STRONG_HAND_RATIO)


def range_advantage_villain(spot: SpotData) -> bool:
    """Villain's overall range is stronger than hero's.

    Brief definition: villain's range is stronger.

    Brief rule: villain_total_equity > 0.55 AND villain_strong_hand_count >
    hero_strong_hand_count by at least 30%.
    """
    rd = spot.range_data
    return (rd.villain_total_equity > 0.55
            and rd.villain_strong_hand_count
            > rd.hero_strong_hand_count * _STRONG_HAND_RATIO)


def nut_advantage_hero(spot: SpotData) -> bool:
    """Hero's range holds far more nutted combos than villain's.

    Brief definition: hero's range has more nutted combos than villain's, even
    when overall equity is close; enables credible large bet sizings.

    Brief rule: hero_top_5pct_combos > villain_top_5pct_combos x 1.5.
    """
    rd = spot.range_data
    return rd.hero_top_5pct_combos > rd.villain_top_5pct_combos * _NUT_COMBO_RATIO


def nut_advantage_villain(spot: SpotData) -> bool:
    """Villain's range holds far more nutted combos than hero's.

    Brief definition: villain has the nuts more often than hero in the
    comparable spot.

    Brief rule: villain_top_5pct_combos > hero_top_5pct_combos x 1.5.
    """
    rd = spot.range_data
    return rd.villain_top_5pct_combos > rd.hero_top_5pct_combos * _NUT_COMBO_RATIO
