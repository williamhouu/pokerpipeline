"""Preflop equity (hand vs hand, hand vs range).

The postflop equity engine in ``pipeline/fact_extractor/equity.py`` only
enumerates 1- or 2-card runouts (turn / river completion). Preflop has
5 cards left to come, so we need a separate function that samples
5-card boards.

Approach: Monte Carlo. For each (hero, villain) pair, sample N random
5-card boards from the remaining deck, compare hand ranks, average.
Sampling noise at N=200 is ~1-2 percentage points -- adequate for
filter thresholds and prose, not for a published equity figure.

For range-vs-hand work, weight by villain combo weight.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from pipeline.fact_extractor.equity import FULL_DECK, rank_hand


def preflop_hand_equity(
    hero: Iterable[str],
    villain: Iterable[str],
    *,
    n_samples: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Hero's preflop equity vs one villain hand. No board cards.

    Args:
        hero: 2-tuple of card strings (e.g. ``('Ah', 'Kh')``).
        villain: 2-tuple of card strings (e.g. ``('Qs', 'Qd')``).
        n_samples: How many 5-card boards to sample. 200 = ~1-2%
            noise; bump to 1000 for higher fidelity.
        rng: Optional ``random.Random`` for determinism (tests).

    Returns:
        Hero's equity in [0.0, 1.0] (wins + ties/2). Returns 0.0 if
        hero and villain share a card.
    """
    hero_cards = list(hero)
    villain_cards = list(villain)
    known = set(hero_cards) | set(villain_cards)
    if len(known) != len(hero_cards) + len(villain_cards):
        return 0.0  # card conflict
    deck = [c for c in FULL_DECK if c not in known]
    rng = rng or random.Random()

    wins = ties = 0
    for _ in range(n_samples):
        board = rng.sample(deck, 5)
        hero_rank = rank_hand(hero_cards + board)
        villain_rank = rank_hand(villain_cards + board)
        if hero_rank > villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1
    return (wins + ties / 2) / n_samples


def preflop_equity_vs_range(
    hero: Iterable[str],
    villain_range: dict[str, float],
    *,
    n_samples: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Hero's preflop equity vs a weighted villain range.

    Args:
        hero: 2-tuple of card strings.
        villain_range: ``{combo_label: weight}`` for villain's range
            (e.g. ``{'AhAd': 1.0, 'AhKh': 0.8, ...}``). Combos sharing
            a card with hero are skipped (weight effectively zero).
        n_samples: Boards-per-villain-combo sample count.

    Returns:
        Weighted average of hero's equity vs each non-conflicting
        villain combo, in [0.0, 1.0]. Returns 0.0 if the range has no
        non-conflicting combos with positive weight.
    """
    hero_cards = list(hero)
    blockers = set(hero_cards)
    rng = rng or random.Random()

    weighted_total = 0.0
    total_weight = 0.0
    for combo, weight in villain_range.items():
        if weight <= 0:
            continue
        cards = [combo[:2], combo[2:]]
        if set(cards) & blockers:
            continue
        eq = preflop_hand_equity(
            hero_cards,
            cards,
            n_samples=n_samples,
            rng=rng,
        )
        weighted_total += eq * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_total / total_weight


def preflop_range_vs_range_equity(
    hero_range: dict[str, float],
    villain_range: dict[str, float],
    *,
    max_matchups: int = 200,
    n_samples_per_matchup: int = 50,
    rng: random.Random | None = None,
) -> float:
    """Hero range vs villain range, preflop (no board).

    Samples ``max_matchups`` (hero, villain) pairs in proportion to combo
    weights, computes each pair's equity over ``n_samples_per_matchup``
    random boards, returns the average.

    Carries more noise than the hand-vs-range function (two layers of
    sampling); 200 matchups × 50 boards is a reasonable speed/accuracy
    tradeoff for batch generation. Bump both for a tighter estimate.

    Args:
        hero_range: ``{combo: weight}`` for hero's range at the node
            (typically the union of all of hero's action ranges).
        villain_range: same shape, for the villain.
        max_matchups: how many hero/villain combo pairs to sample.
        n_samples_per_matchup: how many 5-card boards to sample for each
            matchup's equity.
        rng: optional ``random.Random`` for determinism.

    Returns:
        Hero's range equity vs villain's range, in [0.0, 1.0]. Returns
        0.0 if either range is empty or all matchups conflict.
    """
    if not hero_range or not villain_range:
        return 0.0
    rng = rng or random.Random()

    # Filter to non-zero-weight combos (saves work + makes choices() valid).
    hero_items = [(c, w) for c, w in hero_range.items() if w > 0]
    villain_items = [(c, w) for c, w in villain_range.items() if w > 0]
    if not hero_items or not villain_items:
        return 0.0

    hero_combos, hero_weights = zip(*hero_items, strict=True)
    villain_combos, villain_weights = zip(*villain_items, strict=True)
    heroes = rng.choices(
        list(hero_combos),
        weights=list(hero_weights),
        k=max_matchups,
    )
    villains = rng.choices(
        list(villain_combos),
        weights=list(villain_weights),
        k=max_matchups,
    )

    total = 0.0
    matched = 0
    for hero, villain in zip(heroes, villains, strict=False):
        hero_cards = [hero[:2], hero[2:]]
        villain_cards = [villain[:2], villain[2:]]
        if set(hero_cards) & set(villain_cards):
            continue
        total += preflop_hand_equity(
            hero_cards,
            villain_cards,
            n_samples=n_samples_per_matchup,
            rng=rng,
        )
        matched += 1

    return total / matched if matched else 0.0
