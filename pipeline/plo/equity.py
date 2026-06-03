"""PLO preflop equity -- 4-card "best 2-of-4 hole + 3-of-5 board".

The defining Omaha rule, and the only thing that makes this different from
the Hold'em equity engine in :mod:`pipeline.preflop.equity`: a player's
five-card hand must use **exactly two** of their four hole cards and
**exactly three** of the five board cards. You cannot "play the board," and
you cannot play one or three hole cards. Getting this constraint right is
the whole game -- a naive "best 5 of 9" evaluator silently breaks it (it
will make a flush from four board cards, etc.).

Implementation: reuse the existing, tested 5-card ranker
(:func:`pipeline.fact_extractor.equity.rank_hand`, categories 8 straight
flush ... 0 high card, bigger tuple = better) and enumerate the
``C(4,2) = 6`` hole pairs x ``C(5,3) = 10`` board triples, taking the max
over the 60 resulting five-card hands. No new dependency.

The three public functions mirror :mod:`pipeline.preflop.equity` exactly
(``preflop_hand_equity`` / ``preflop_equity_vs_range`` /
``preflop_range_vs_range_equity``) so the forked fact extractor / EV engine
call them with no change -- the 4-card evaluator is a drop-in behind a
stable interface. Combo strings are 8 chars (``'AhKhQsJs'`` = four cards).

Performance note: each board evaluation does 60 ``rank_hand`` calls per
player (vs 1 for Hold'em best-5-of-7), so this is ~60x heavier than the NLHE
engine per sample. Equity is therefore computed lazily, only on spots that
already passed the cheap worthiness gate (same pattern as preflop). If batch
times prove too slow, optimise here (memoise / lookup-table evaluator)
behind this interface; correctness first.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from pipeline.fact_extractor.equity import FULL_DECK, rank_hand

# The six ways to choose 2 of 4 hole cards. Board triples are enumerated per
# board (the board changes every sample), but the hole-pair index set is
# constant, so precompute it once.
_HOLE_PAIRS: tuple[tuple[int, int], ...] = tuple(combinations(range(4), 2))

# A four-card PLO combo is encoded as an 8-character string: two chars per
# card, e.g. "AhKhQsJs". Hole / villain hand arguments may instead be passed
# as an iterable of four 2-char card strings.
_PLO_HAND_LEN = 4


def split_combo(combo: str) -> list[str]:
    """Split an 8-char PLO combo string into its four 2-char cards.

    ``"AhKhQsJs" -> ["Ah", "Kh", "Qs", "Js"]``.
    """
    return [combo[i : i + 2] for i in range(0, 8, 2)]


def omaha_best_rank(hole: list[str], board: list[str]) -> tuple[Any, ...]:
    """Best five-card rank using EXACTLY 2 of 4 hole + 3 of 5 board cards.

    Enumerates all 6 hole pairs x ``C(len(board), 3)`` board triples and
    returns the maximum :func:`rank_hand` value. ``board`` is normally the
    five sampled community cards (giving 6 x 10 = 60 candidate hands).

    This is the function that enforces the Omaha constraint; everything else
    in this module is sampling around it.
    """
    triples = list(combinations(board, 3))
    return max(
        rank_hand([hole[i], hole[j], *triple])
        for i, j in _HOLE_PAIRS
        for triple in triples
    )


def _as_four_cards(hand: Iterable[str]) -> list[str]:
    cards = list(hand)
    if len(cards) != _PLO_HAND_LEN:
        msg = f"PLO hand needs exactly 4 cards, got {len(cards)}: {cards!r}"
        raise ValueError(msg)
    return cards


def preflop_hand_equity(
    hero: Iterable[str],
    villain: Iterable[str],
    *,
    n_samples: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Hero's preflop equity vs one villain PLO hand. No board cards.

    Args:
        hero: four card strings (e.g. ``('Ah', 'Kh', 'Qs', 'Js')``).
        villain: four card strings.
        n_samples: how many 5-card boards to sample.
        rng: optional ``random.Random`` for determinism (tests).

    Returns:
        Hero's equity in [0.0, 1.0] (wins + ties/2). Returns 0.0 if hero
        and villain share a card.
    """
    hero_cards = _as_four_cards(hero)
    villain_cards = _as_four_cards(villain)
    known = set(hero_cards) | set(villain_cards)
    if len(known) != len(hero_cards) + len(villain_cards):
        return 0.0  # card conflict
    deck = [c for c in FULL_DECK if c not in known]
    rng = rng or random.Random()

    wins = ties = 0
    for _ in range(n_samples):
        board = rng.sample(deck, 5)
        hero_rank = omaha_best_rank(hero_cards, board)
        villain_rank = omaha_best_rank(villain_cards, board)
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
    """Hero's preflop equity vs a weighted villain PLO range.

    Args:
        hero: four card strings.
        villain_range: ``{combo: weight}`` where ``combo`` is an 8-char PLO
            combo (``'AhKhQsJs'``). Combos sharing a card with hero are
            skipped.
        n_samples: boards-per-villain-combo sample count.

    Returns:
        Weighted average of hero's equity vs each non-conflicting villain
        combo, in [0.0, 1.0]. Returns 0.0 if no non-conflicting combos with
        positive weight remain.
    """
    hero_cards = _as_four_cards(hero)
    blockers = set(hero_cards)
    rng = rng or random.Random()

    weighted_total = 0.0
    total_weight = 0.0
    for combo, weight in villain_range.items():
        if weight <= 0:
            continue
        cards = split_combo(combo)
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

    Samples ``max_matchups`` (hero, villain) combo pairs in proportion to
    weight, computes each pair's equity over ``n_samples_per_matchup``
    boards, returns the average. Two layers of sampling -> noisier than the
    hand-vs-range function; bump both for a tighter estimate.

    Args:
        hero_range: ``{combo: weight}`` (8-char PLO combos) for hero.
        villain_range: same shape for villain.
        max_matchups: how many hero/villain combo pairs to sample.
        n_samples_per_matchup: boards per matchup.
        rng: optional ``random.Random`` for determinism.

    Returns:
        Hero's range equity vs villain's, in [0.0, 1.0]. Returns 0.0 if
        either range is empty or every sampled matchup conflicts.
    """
    if not hero_range or not villain_range:
        return 0.0
    rng = rng or random.Random()

    hero_items = [(c, w) for c, w in hero_range.items() if w > 0]
    villain_items = [(c, w) for c, w in villain_range.items() if w > 0]
    if not hero_items or not villain_items:
        return 0.0

    hero_combos, hero_weights = zip(*hero_items, strict=True)
    villain_combos, villain_weights = zip(*villain_items, strict=True)
    heroes = rng.choices(
        list(hero_combos), weights=list(hero_weights), k=max_matchups
    )
    villains = rng.choices(
        list(villain_combos), weights=list(villain_weights), k=max_matchups
    )

    total = 0.0
    matched = 0
    for hero, villain in zip(heroes, villains, strict=False):
        hero_cards = split_combo(hero)
        villain_cards = split_combo(villain)
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
