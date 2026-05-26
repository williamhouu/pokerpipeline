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
