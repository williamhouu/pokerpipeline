"""Equity, range, and blocker fact extraction (Fact Extractor / Layer 5).

Turns a Path Sampler SpotContext into the SpotData.equity_data and
SpotData.range_data sections -- the strategic numbers the concept tagger reads.

`compute_equity_data` and `compute_range_data` take the per-spot equity already
computed by pipeline.fact_extractor.equity, so the expensive runout enumeration
runs once per spot (the orchestrator in __init__.py does it).

v1 limitations, documented where they bite:
  * equity_realization_ratio is left at its neutral default -- deriving it
    cleanly needs PioSolver's calc_ev convention pinned down, which warrants
    its own verification pass. equity_under_realized / equity_over_realized
    therefore stay dormant for now.
  * Villain's calling-only sub-range is not modelled separately from the
    continuing range; hero_raw_equity_vs_calling reuses the continuing-range
    equity.
  * "Top 5%" combo counts use the hand_class `premium` bucket as a proxy for
    the nutted hands; "value" / "bluff" use the premium+strong / air buckets.
"""
from __future__ import annotations

from pipeline.fact_extractor.equity import range_vs_range_equity
from pipeline.fact_extractor.hand_class import classify_hand
from pipeline.fact_extractor.spot_data import Combo, EquityData, RangeData

_VALUE_BUCKETS = ("premium", "strong")     # made hands that bet/call for value
_BLUFF_BUCKET = "air"                      # no showdown value
# Equity bands for villain range-shape analysis (mirror Section A's bands).
_TOP_BAND = 0.75
_BOTTOM_BAND = 0.30


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _cards(combo: str) -> tuple[str, str]:
    """Split a combo string: 'AhKh' -> ('Ah', 'Kh')."""
    return combo[:2], combo[2:]


def _on_board(combo: str, board) -> bool:
    """Whether a combo shares a card with the board (an impossible holding)."""
    return bool({combo[:2], combo[2:]} & set(board))


def compute_equity_data(spot_context, hero_hand: str,
                        hero_equity: float) -> EquityData:
    """Populate EquityData from the spot and hero's pre-computed range equity.

    Pot odds and MDF come straight from the pot field: with a bet of B already
    folded into the pot total P, the call is B/P and MDF is 1 - B/P.
    """
    node = spot_context.node
    to_call, pot = node.amount_to_call, node.pot
    pot_odds = _clamp01(to_call / pot) if pot > 0 and to_call > 0 else 0.0
    return EquityData(
        hero_raw_equity_vs_continuing=_clamp01(hero_equity),
        hero_raw_equity_vs_calling=_clamp01(hero_equity),   # v1: same range
        pot_odds_required=pot_odds,
        mdf=(1.0 - pot_odds) if to_call > 0 else 0.0,
        equity_realization_ratio=1.0,                       # v1: deferred
    )


def compute_range_data(spot_context, hero_hand: str,
                       villain_combo_equity: dict[str, float]) -> RangeData:
    """Populate RangeData: ranges, blockers, range shape, advantage counts.

    `villain_combo_equity` maps each villain combo to its equity vs hero (the
    by-product of the equity pass), used for the villain Combo list and the
    polarized/linear range-shape read.
    """
    board = spot_context.node.board
    hero_cards = set(_cards(hero_hand))

    # Classify every (playable) villain combo once; reuse for blockers etc.
    villain_class = {combo: classify_hand(combo, board)
                     for combo in spot_context.villain_range
                     if not _on_board(combo, board)}

    # Blocker effect -- how much value / bluff weight hero's cards remove.
    value_total = value_blocked = bluff_total = bluff_blocked = 0.0
    draw_weight = draw_total = 0.0
    for combo, weight in spot_context.villain_range.items():
        info = villain_class.get(combo)
        if info is None:                        # combo conflicts with the board
            continue
        blocked = bool(set(_cards(combo)) & hero_cards)
        if info["strength_bucket"] in _VALUE_BUCKETS:
            value_total += weight
            value_blocked += weight if blocked else 0.0
        elif info["strength_bucket"] == _BLUFF_BUCKET:
            bluff_total += weight
            bluff_blocked += weight if blocked else 0.0
        draw_total += weight
        draw_weight += weight if info["draws"] else 0.0

    hero_strong, hero_top = _strength_counts(spot_context.hero_range, board)
    villain_strong, villain_top = _strength_counts(spot_context.villain_range,
                                                   board, villain_class)

    # Villain range as Combos (hero-blocked combos dropped), with per-combo
    # equity; hero range carries weights only (no tag needs hero-combo equity).
    villain_combos = [Combo(cards=_cards(combo), weight=_clamp01(weight),
                            equity=_clamp01(villain_combo_equity[combo]))
                      for combo, weight in spot_context.villain_range.items()
                      if combo in villain_combo_equity]
    hero_combos = [Combo(cards=_cards(combo), weight=_clamp01(weight))
                   for combo, weight in spot_context.hero_range.items()
                   if not _on_board(combo, board)]

    hero_total = range_vs_range_equity(spot_context.hero_range,
                                       spot_context.villain_range, board)

    return RangeData(
        villain_range=villain_combos,
        hero_range=hero_combos,
        villain_range_shape=_range_shape(villain_combos),
        villain_value_combos=value_total,
        villain_bluff_combos=bluff_total,
        hero_blocks_value_pct=_clamp01(value_blocked / value_total)
        if value_total else 0.0,
        hero_blocks_bluffs_pct=_clamp01(bluff_blocked / bluff_total)
        if bluff_total else 0.0,
        hero_total_equity=_clamp01(hero_total),
        villain_total_equity=_clamp01(1.0 - hero_total),
        hero_strong_hand_count=hero_strong,
        villain_strong_hand_count=villain_strong,
        hero_top_5pct_combos=hero_top,
        villain_top_5pct_combos=villain_top,
        villain_draw_equity_pct=_clamp01(draw_weight / draw_total)
        if draw_total else 0.0,
    )


def _strength_counts(combo_range, board, classified=None):
    """Weighted counts of (strong-bucket combos, premium-bucket combos)."""
    strong = top = 0.0
    for combo, weight in combo_range.items():
        if classified is not None:
            info = classified.get(combo)        # None -> conflicts with board
        elif _on_board(combo, board):
            info = None
        else:
            info = classify_hand(combo, board)
        if info is None:
            continue
        if info["strength_bucket"] in _VALUE_BUCKETS:
            strong += weight
        if info["strength_bucket"] == "premium":
            top += weight
    return strong, top


def _range_shape(villain_combos) -> str:
    """'polarized' / 'linear' / '' from villain combos bucketed by equity."""
    total = sum(combo.weight for combo in villain_combos)
    if total <= 0:
        return ""
    top = sum(c.weight for c in villain_combos if c.equity >= _TOP_BAND) / total
    bottom = sum(c.weight for c in villain_combos
                 if c.equity <= _BOTTOM_BAND) / total
    middle = 1.0 - top - bottom
    if top >= 0.30 and bottom >= 0.30 and middle < 0.40:
        return "polarized"
    if top >= 0.10 and bottom >= 0.10 and middle >= 0.10:
        return "linear"
    return ""
