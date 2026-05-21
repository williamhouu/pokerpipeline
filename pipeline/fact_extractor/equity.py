"""Poker hand evaluation and equity calculation (Fact Extractor / Layer 5).

The equity calculator the engineering brief scopes as its own component: a
7-card hand evaluator plus runout enumeration. Pure poker math -- cards in,
numbers out -- with no dependency on the solver or the rest of the pipeline.

`rank_hand` turns 5-7 cards into a comparable strength tuple (bigger beats
smaller). `hand_equity` enumerates the remaining board for one matchup;
`equity_vs_range` and `range_vs_range_equity` build on it.

Flop equity exhaustively enumerates the ~990 turn+river runouts per matchup, so
range-level calls are deliberately bounded -- exact where it is cheap (turn,
river), sampled where it is not.
"""
from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

_RANKS = "23456789TJQKA"
_SUITS = "shdc"
_RANK_VALUE = {rank: value for value, rank in enumerate(_RANKS, start=2)}
FULL_DECK = tuple(rank + suit for rank in _RANKS for suit in _SUITS)


def _straight_high(values) -> int:
    """Top card of the best 5-card straight in `values` (wheel aware); 0 if none."""
    present = set(values)
    if 14 in present:
        present.add(1)                              # ace plays low for the wheel
    for top in range(14, 4, -1):
        if all((top - step) in present for step in range(5)):
            return top
    return 0


def rank_hand(cards) -> tuple:
    """A comparable strength tuple for a 5-7 card hand (larger tuple = better).

    The first element is the hand category (8 straight flush ... 0 high card);
    the rest are tiebreakers. Tuples within a category share a shape, and the
    category element always settles cross-category comparisons.
    """
    values = [_RANK_VALUE[c[0]] for c in cards]
    suits = [c[1] for c in cards]
    rank_count = Counter(values)
    suit_count = Counter(suits)

    flush_suit = next((s for s, n in suit_count.items() if n >= 5), None)
    flush_values: list[int] = []
    if flush_suit is not None:
        flush_values = sorted((values[i] for i in range(len(cards))
                               if suits[i] == flush_suit), reverse=True)
        straight_flush = _straight_high(flush_values)
        if straight_flush:
            return (8, straight_flush)

    quads = [v for v, n in rank_count.items() if n == 4]
    if quads:
        quad = max(quads)
        return (7, quad, max(v for v in values if v != quad))

    trips = sorted((v for v, n in rank_count.items() if n >= 3), reverse=True)
    pairs = sorted((v for v, n in rank_count.items() if n >= 2), reverse=True)
    if trips:
        side = [v for v in pairs if v != trips[0]]
        if side:
            return (6, trips[0], side[0])

    if flush_suit is not None:
        return (5, tuple(flush_values[:5]))

    straight = _straight_high(values)
    if straight:
        return (4, straight)

    if trips:
        kickers = sorted((v for v in values if v != trips[0]), reverse=True)
        return (3, trips[0], tuple(kickers[:2]))

    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        return (2, high, low, max(v for v in values if v not in (high, low)))

    if pairs:
        kickers = sorted((v for v in values if v != pairs[0]), reverse=True)
        return (1, pairs[0], tuple(kickers[:3]))

    return (0, tuple(sorted(values, reverse=True)[:5]))


def hand_equity(hero, villain, board, *, max_runouts=None, rng=None) -> float:
    """Hero's equity (wins + ties/2) vs one villain hand on `board` (3-5 cards).

    Remaining board cards are enumerated exhaustively, or sampled when the
    runout count exceeds `max_runouts`. A card conflict yields 0.0.
    """
    hero, villain, board = list(hero), list(villain), list(board)
    known = set(hero) | set(villain) | set(board)
    if len(known) != len(hero) + len(villain) + len(board):
        return 0.0                                  # impossible matchup
    deck = [c for c in FULL_DECK if c not in known]
    need = 5 - len(board)
    if need <= 0:
        runouts = [()]
    elif need == 1:
        runouts = [(c,) for c in deck]
    else:
        runouts = list(combinations(deck, 2))
    if max_runouts is not None and len(runouts) > max_runouts:
        runouts = (rng or random).sample(runouts, max_runouts)

    wins = ties = 0
    for extra in runouts:
        full_board = board + list(extra)
        hero_rank = rank_hand(hero + full_board)
        villain_rank = rank_hand(villain + full_board)
        if hero_rank > villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1
    return (wins + ties / 2) / len(runouts) if runouts else 0.0


def equity_vs_range(hero, villain_range, board, *, max_runouts=None, rng=None):
    """Hero's equity vs a weighted villain range.

    `villain_range` maps a combo string ('AhKh') to its weight. Returns
    (hero_equity, per_combo) where per_combo maps each villain combo to its own
    equity vs hero. Combos sharing a card with hero or the board are skipped.
    """
    hero = list(hero)
    blockers = set(hero) | set(board)
    weighted_equity = total_weight = 0.0
    per_combo: dict[str, float] = {}
    for combo, weight in villain_range.items():
        cards = [combo[:2], combo[2:]]
        if set(cards) & blockers:
            continue
        hero_eq = hand_equity(hero, cards, board, max_runouts=max_runouts, rng=rng)
        per_combo[combo] = 1.0 - hero_eq
        weighted_equity += hero_eq * weight
        total_weight += weight
    return (weighted_equity / total_weight if total_weight else 0.0), per_combo


def range_vs_range_equity(hero_range, villain_range, board, *,
                          max_matchups=600, max_runouts=200, seed=0) -> float:
    """Sampled hero-range vs villain-range equity.

    Matchups are drawn in proportion to combo weight; each one's equity is
    exact unless the runout count exceeds `max_runouts`. The estimate carries a
    few percent of sampling noise -- adequate for driving tag thresholds, not
    for a published equity figure.
    """
    if not hero_range or not villain_range:
        return 0.0
    rng = random.Random(seed)
    hero_combos, hero_weights = zip(*hero_range.items())
    villain_combos, villain_weights = zip(*villain_range.items())
    heroes = rng.choices(hero_combos, weights=hero_weights, k=max_matchups)
    villains = rng.choices(villain_combos, weights=villain_weights, k=max_matchups)

    total_equity = 0.0
    matchups = 0
    board_set = set(board)
    for hero, villain in zip(heroes, villains):
        hero_cards = [hero[:2], hero[2:]]
        villain_cards = [villain[:2], villain[2:]]
        if set(hero_cards) & set(villain_cards):
            continue
        if (set(hero_cards) | set(villain_cards)) & board_set:
            continue
        total_equity += hand_equity(hero_cards, villain_cards, board,
                                    max_runouts=max_runouts, rng=rng)
        matchups += 1
    return total_equity / matchups if matchups else 0.0
