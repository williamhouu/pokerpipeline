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
  * "value" / "bluff" use the premium+strong / air strength buckets. The
    "top 5%" combo count was originally a premium-bucket proxy too; it now
    ranks the universe of board-legal combos by hand strength (rank_hand on
    each combo + board) and counts the weight each player carries in the
    top 5% (Section E nut_advantage tags read these counts).
"""
from __future__ import annotations

from itertools import combinations

from pipeline.fact_extractor.equity import (
    FULL_DECK, range_vs_range_equity, rank_hand,
)
from pipeline.fact_extractor.hand_class import classify_hand
from pipeline.fact_extractor.spot_data import Combo, EquityData, RangeData

_VALUE_BUCKETS = ("premium", "strong")     # made hands that bet/call for value
_BLUFF_BUCKET = "air"                      # no showdown value
# Equity bands for villain range-shape analysis (mirror Section A's bands).
_TOP_BAND = 0.75
_BOTTOM_BAND = 0.30
# Nut-advantage threshold: the fraction of board-legal combos considered "top".
_NUT_PCT = 0.05

# Ryan-feedback Fix 4 (May 2026): hand-class ranking constants.
# How many hand-class entries to surface to Layer 6 from villain's continuing
# range. The user spec asked for "top 5-8 hand classes"; 6 is the centre of
# that range and matches what the LLM's 2-3-combo citation in answer_explanation
# can reasonably ground itself on.
_VILLAIN_TOP_COMBOS_COUNT = 6
# How many specific (rank-suit, rank-suit) combos to include per hand class
# entry, for the LLM to optionally cite directly in prose.
_EXAMPLES_PER_CLASS = 3
# Strength-bucket score used in the ranking key (weight * bucket_score).
# Premium-tier value combos rise to the top of the list; air-tier combos fall
# to the bottom even when their range weight is non-trivial.
_BUCKET_SCORE = {
    "premium": 6, "strong": 5, "medium": 4,
    "vulnerable": 3, "marginal": 2, "air": 1,
}
# Card suit -> emoji (mirrors action_history._SUIT_EMOJI so the data block's
# example combos render identically to the Question column's hole-card prose).
_SUIT_EMOJI = {"s": "♠️", "h": "❤️", "d": "♦️", "c": "♣️"}


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

    hero_strong, _ = _strength_counts(spot_context.hero_range, board)
    villain_strong, _ = _strength_counts(spot_context.villain_range,
                                         board, villain_class)
    # Top-5% combo counts use the absolute strength of every board-legal combo
    # so the comparison ranks ranges against the same universal pool -- the
    # earlier premium-bucket proxy fired too readily because the bucket holds
    # ~20% of typical ranges, not 5%.
    top_combos = _board_top_combos(board, pct=_NUT_PCT)
    hero_top = _range_weight_in(spot_context.hero_range, top_combos)
    villain_top = _range_weight_in(spot_context.villain_range, top_combos)

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

    top_value_combos = _villain_top_value_combos(spot_context.villain_range,
                                                  villain_class)
    hero_disposition = _compute_hero_range_disposition(
        spot_context.hero_range, hero_top, board)

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
        villain_top_value_combos=top_value_combos,
        hero_range_disposition=hero_disposition,
    )


def _strength_counts(combo_range, board, classified=None):
    """Weighted count of value-bucket combos in a range.

    The second tuple slot used to also return the premium-bucket weight as a
    "top 5%" proxy; that is now computed against the universal top-of-board
    pool via `_board_top_combos` (see below). The slot stays for back-compat
    but the caller discards it.
    """
    strong = 0.0
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
    return strong, 0.0


def _board_top_combos(board, pct: float = _NUT_PCT) -> set[frozenset]:
    """The {a, b} combos in the top `pct` strongest on this board.

    Ranks every 2-card combo that does not share a card with the board by
    `rank_hand(combo + board)` (the same poker evaluator the equity calc uses)
    and returns the strongest pct slice as a set of frozensets, so a player's
    range membership can be tested by frozenset({card_a, card_b}) in the set.

    Used by Section E nut_advantage to count each player's weight in the
    board's nut-tier combos. v1 ranks at the current street -- on the flop and
    turn, hands strong now usually stay strong (sets, two pair); a more careful
    runout-weighted score is a tune-against-gold improvement.
    """
    board_set = set(board)
    deck = [c for c in FULL_DECK if c not in board_set]
    ranked: list[tuple[tuple, frozenset]] = []
    for a, b in combinations(deck, 2):
        strength = rank_hand([a, b] + list(board))
        ranked.append((strength, frozenset((a, b))))
    ranked.sort(key=lambda item: item[0], reverse=True)
    cutoff = max(1, int(round(len(ranked) * pct)))
    return {combo for _, combo in ranked[:cutoff]}


def _range_weight_in(combo_range, top_combos: set[frozenset]) -> float:
    """Total weight in `combo_range` of combos that lie in `top_combos`."""
    weight = 0.0
    for combo, w in combo_range.items():
        if frozenset(_cards(combo)) in top_combos:
            weight += w
    return weight


# Ryan-feedback Fix 5 (May 2026): hero range-disposition thresholds.
# Hero's range is "capped" when its weight in the universal top-5% pool is
# below the lower threshold (very few nuts available given the action line);
# "uncapped" above the upper threshold; "linear" in between. The "polarized"
# variant fires when hero ALSO carries a large air slice -- a top+bottom shape
# with little middle.
_CAPPED_TOP_FRAC = 0.05
_UNCAPPED_TOP_FRAC = 0.15
_POLARIZED_AIR_FRAC = 0.30
_POLARIZED_MIDDLE_MAX = 0.40


def _compute_hero_range_disposition(hero_range, hero_top_5pct: float,
                                     board) -> str:
    """Hero's range shape at this node: capped / uncapped / polarized / linear.

    Per Ryan-feedback Fix 5: Layer 6 uses this to frame whether hero's range
    can credibly bet for value, must bluff catch, etc. "polarized" trumps
    "uncapped" when the air slice is large -- a polarized range has nuts AND
    air, not a smooth value distribution.
    """
    total = sum(w for w in hero_range.values()) if hero_range else 0.0
    if total <= 0:
        return ""
    top_frac = hero_top_5pct / total

    # Count air-bucket combos in hero's range for the polarized check.
    air_weight = 0.0
    middle_weight = 0.0
    for combo, weight in hero_range.items():
        if _on_board(combo, board) or weight <= 0:
            continue
        bucket = classify_hand(combo, board)["strength_bucket"]
        if bucket == "air":
            air_weight += weight
        elif bucket in ("medium", "vulnerable", "marginal"):
            middle_weight += weight
    air_frac = air_weight / total
    middle_frac = middle_weight / total

    # Polarized: meaningful top + meaningful air + thin middle.
    if (top_frac >= _CAPPED_TOP_FRAC
            and air_frac >= _POLARIZED_AIR_FRAC
            and middle_frac <= _POLARIZED_MIDDLE_MAX):
        return "polarized"
    if top_frac >= _UNCAPPED_TOP_FRAC:
        return "uncapped"
    if top_frac < _CAPPED_TOP_FRAC:
        return "capped"
    return "linear"


def _emoji_combo(combo: str) -> str:
    """'KsKh' -> 'K♠️K❤️'. Used for example_combos in villain_top_value_combos."""
    a, b = combo[:2], combo[2:]
    return f"{a[0]}{_SUIT_EMOJI.get(a[1], a[1])}{b[0]}{_SUIT_EMOJI.get(b[1], b[1])}"


def _villain_top_value_combos(combo_range, villain_class,
                               top_n: int = _VILLAIN_TOP_COMBOS_COUNT) -> list[dict]:
    """The top N hand classes in villain's continuing range, by total weight *
    strength-bucket score, with 2-3 highest-weighted concrete combos per class
    rendered in emoji form for direct citation by Layer 6.

    Per Ryan-feedback Fix 4 (May 2026): Layer 6's prior prose described villain's
    range abstractly ("villain has value hands and bluffs"); Ryan wants 2-3
    SPECIFIC combos named ("K♠️K♣️, J♦️J♣️, and 7♠️7♣️ for sets"). This field
    surfaces the data the LLM needs to comply without inventing combos.

    Empty input or empty class info -> empty list (the LLM degrades to abstract
    prose for that spot, same as pre-Fix-4 behaviour). Air-bucket classes are
    NOT excluded -- they may legitimately rise to the top of villain's bluff
    range and the LLM may want to cite them; the bucket field lets the prompt
    distinguish bluffs from value calls.
    """
    if not combo_range or not villain_class:
        return []

    # Aggregate combos by hand-class label, tracking total weight, bucket,
    # and per-combo weight list (for example-combo selection).
    by_label: dict[str, dict] = {}
    for combo, weight in combo_range.items():
        info = villain_class.get(combo)
        if info is None or weight <= 0:
            continue
        label = info["label"]
        if label not in by_label:
            by_label[label] = {
                "hand_class_label": label,
                "bucket": info["strength_bucket"],
                "total_weight": 0.0,
                "_combos": [],
            }
        by_label[label]["total_weight"] += weight
        by_label[label]["_combos"].append((combo, weight))

    # Build each entry's example_combos + combo_count, then drop the working
    # list so the dict serialises cleanly.
    entries: list[dict] = []
    for entry in by_label.values():
        combos_by_weight = sorted(entry["_combos"], key=lambda cw: -cw[1])
        entry["combo_count"] = len(entry["_combos"])
        entry["example_combos"] = [_emoji_combo(c)
                                   for c, _ in combos_by_weight[:_EXAMPLES_PER_CLASS]]
        entry["total_weight"] = round(entry["total_weight"], 3)
        del entry["_combos"]
        entries.append(entry)

    entries.sort(
        key=lambda e: e["total_weight"] * _BUCKET_SCORE.get(e["bucket"], 0),
        reverse=True,
    )
    return entries[:top_n]


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
