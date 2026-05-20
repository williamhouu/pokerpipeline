"""Board texture classification (Fact Extractor / Layer 5).

A pure function of the community cards -- no solver data, no LLM. Produces the
`board_texture` field of the Layer 5 data block: five descriptive axes plus a
derived composite descriptor.

    >>> classify_board("Jh Th 9h")
    {'suit_distribution': 'monotone', 'pair_status': 'unpaired',
     'connectedness': 'connected', 'rank_distribution': 'broadway_heavy',
     'composite': 'very_wet'}

See docs/engineering_brief.docx, "Board Texture". The brief's axes overlap in
places and it states the rules are starting points to tune against real solves;
where a judgement call was needed it is marked `v1:` inline.
"""
from __future__ import annotations

import itertools

from pipeline.cards import BROADWAY, card_suit, parse_board, rank_value


def _suit_distribution(cards: list[str]) -> str:
    counts: dict[str, int] = {}
    for card in cards:
        counts[card_suit(card)] = counts.get(card_suit(card), 0) + 1
    top = max(counts.values())
    if len(cards) == 3:                       # flop
        return {3: "monotone", 2: "two_tone", 1: "rainbow"}[top]
    if top >= 4:                              # turn / river
        return "flush_completed"
    if top == 3:
        return "monotone"
    if top == 2:
        return "two_tone"
    return "rainbow"


def _pair_status(values: list[int]) -> str:
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    if top >= 3:
        return "trips_on_board"
    if top >= 2:
        return "paired"
    return "unpaired"


def _connectedness(values: list[int]) -> str:
    """Closest 3-card straightness across the board's distinct ranks.

    v1: measured over distinct ranks via the most-connected 3-card subset, with
    aces counted high or low. Boards with fewer than 3 distinct ranks (paired /
    trips) report 'disconnected' -- 3-card straight potential is not meaningful
    there, and the composite descriptor does not rely on it for paired boards.
    """
    distinct = sorted(set(values))
    if len(distinct) < 3:
        return "disconnected"
    best_gaps: int | None = None
    for combo in itertools.combinations(distinct, 3):
        variants = [combo]
        if 14 in combo:                       # also try the ace as low (wheel)
            variants.append(tuple(1 if v == 14 else v for v in combo))
        for var in variants:
            gaps = (max(var) - min(var)) - 2  # internal ranks missing from span
            best_gaps = gaps if best_gaps is None else min(best_gaps, gaps)
    return {0: "connected", 1: "one_gap", 2: "two_gap"}.get(best_gaps, "disconnected")


def _rank_distribution(values: list[int]) -> str:
    """Single rank-height label. v1 priority order (the brief lists overlapping
    categories; ace beats king beats broadway-heavy beats the rest)."""
    n_broadway = sum(1 for v in values if v in BROADWAY)
    if 14 in values:
        return "ace_high"
    if all(v >= 10 for v in values):
        return "high"
    if n_broadway >= 2:
        return "broadway_heavy"
    if max(values) == 13:
        return "king_high"
    if all(v <= 8 for v in values):
        return "low"
    return "middling"


def _composite(suit: str, pair: str, connectedness: str, n_broadway: int) -> str:
    """Derive the single wetness descriptor from the other axes.

    v1 priority ladder, tuned to reproduce every labelled example in the brief.
    """
    if pair == "trips_on_board":
        return "static"
    if pair == "paired":
        if suit in ("monotone", "flush_completed"):
            return "very_wet"
        if suit == "two_tone":
            return "semi_wet"
        return "static"                       # paired + rainbow
    if suit in ("monotone", "flush_completed"):
        return "very_wet"
    if connectedness == "connected":
        if suit == "two_tone":
            # low/middling connected boards churn equity hardest -> dynamic
            return "dynamic" if n_broadway <= 1 else "wet"
        return "wet"                          # connected + rainbow (straights)
    if connectedness == "one_gap":
        return "wet" if suit == "two_tone" else "semi_wet"
    if suit == "two_tone":
        return "semi_wet"
    return "static" if n_broadway >= 2 else "dry"   # spread, draw-light rainbow


def classify_board(board) -> dict:
    """Classify a board (list of cards, or a space-separated string).

    Returns the `board_texture` dict: suit_distribution, pair_status,
    connectedness, rank_distribution, and the derived composite descriptor.
    """
    cards = parse_board(board)
    values = [rank_value(c) for c in cards]
    n_broadway = sum(1 for v in values if v in BROADWAY)

    suit = _suit_distribution(cards)
    pair = _pair_status(values)
    connectedness = _connectedness(values)
    return {
        "suit_distribution": suit,
        "pair_status": pair,
        "connectedness": connectedness,
        "rank_distribution": _rank_distribution(values),
        "composite": _composite(suit, pair, connectedness, n_broadway),
    }
