"""Concept tags -- Section A: Range Characterization (4 tags).

From docs/engineering_brief.docx, "Concept Tag Library Specification",
Section A. Each tag is a pure function: SpotData -> bool.

Numeric thresholds are the brief's starting values; they will be tuned against
the ~800-explanation gold pool before the tagger goes to production.
"""
from __future__ import annotations

from pipeline.fact_extractor.spot_data import SpotData

# Equity-band boundaries for range-shape analysis. The equity scale splits into
# a top band [0.75, 1.0], a middle band (0.30, 0.75), and a bottom band
# [0.0, 0.30] -- the "top 25%" / "bottom 30%" buckets the brief refers to.
_TOP_BAND_MIN = 0.75
_BOTTOM_BAND_MAX = 0.30


def _equity_band_shares(spot: SpotData):
    """(top, middle, bottom) shares of villain's range weight, split by equity.

    Returns None when villain's combo-level range is unavailable, so callers
    can fall back to the pre-computed range-shape label.
    """
    combos = spot.range_data.villain_range
    total = sum(c.weight for c in combos)
    if total <= 0:
        return None
    top = sum(c.weight for c in combos if c.equity >= _TOP_BAND_MIN) / total
    bottom = sum(c.weight for c in combos if c.equity <= _BOTTOM_BAND_MAX) / total
    return top, 1.0 - top - bottom, bottom


def villain_capped(spot: SpotData) -> bool:
    """Villain's range is missing the strongest hands it could hold.

    Brief definition: villain's range is missing the strongest hands it could
    mathematically hold, because their action sequence excluded those hands
    (e.g. a flat-call vs a 3-bet usually excludes AA/KK).

    Brief rule: if villain's range at this node has top-5% combos at less than
    70% of the population baseline, fire the tag. (0.70: starting value.)
    """
    baseline = spot.population_baseline
    if not baseline.populated:                  # no baseline -> cannot judge
        return False
    return spot.range_data.villain_top_5pct_combos < baseline.top_5pct_combos * 0.70


def villain_uncapped(spot: SpotData) -> bool:
    """Villain's range contains the full top of the rank distribution.

    Brief definition: villain took a passive action that doesn't filter
    premiums (cold-calling a single raise), or an action that includes
    premiums by construction.

    Brief rule: the inverse of villain_capped -- top-5% combos present at or
    above the population baseline.
    """
    baseline = spot.population_baseline
    if not baseline.populated:
        return False
    return spot.range_data.villain_top_5pct_combos >= baseline.top_5pct_combos


def villain_polarized(spot: SpotData) -> bool:
    """Villain's range is bimodal -- strong hands and bluffs, little between.

    Brief definition: bimodal range -- strong made hands and weak/bluff hands,
    with little in the middle. Typical of a turn check-raise or river overbet.

    Brief rule: bucket villain's continuing range by equity vs hero; fires when
    >=30% sits in the top-25% equity band AND >=30% in the bottom-30% band AND
    <40% in the middle. (Thresholds: starting values.) When no combo-level
    range is available, falls back to the pre-computed villain_range_shape.
    """
    shares = _equity_band_shares(spot)
    if shares is None:
        return spot.range_data.villain_range_shape == "polarized"
    top, middle, bottom = shares
    return top >= 0.30 and bottom >= 0.30 and middle < 0.40


def villain_linear(spot: SpotData) -> bool:
    """Villain's range is a smooth equity gradient with no gap.

    Brief definition: villain's range follows a smooth equity gradient with no
    gap. Typical of value-heavy passive lines like cold-calling a raise.

    Brief rule: bucket by equity; fires when no major gap exists -- no equity
    bucket has less than 10% representation. v1: every equity band carries
    >=10% of the range AND the range is not polarized (polarized and linear are
    mutually exclusive shapes). When no combo-level range is available, falls
    back to the pre-computed villain_range_shape.
    """
    shares = _equity_band_shares(spot)
    if shares is None:
        return spot.range_data.villain_range_shape == "linear"
    if villain_polarized(spot):                 # the two shapes are exclusive
        return False
    return all(share >= 0.10 for share in shares)
