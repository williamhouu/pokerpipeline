"""Tests for pipeline.plo.equity (4-card "best 2-of-4 + 3-of-5").

The constraint tests below are the important ones: they pin the Omaha rule
that a naive "best 5 of 9" evaluator silently breaks -- you must use exactly
two hole cards and exactly three board cards, and you cannot play the board.
Each constructs a spot where the naive answer and the correct Omaha answer
land in *different rank categories*, so a regression can't hide.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.equity import (  # noqa: E402
    omaha_best_rank,
    preflop_equity_vs_range,
    preflop_hand_equity,
    preflop_range_vs_range_equity,
    split_combo,
)

# rank_hand categories: 8 straight flush, 7 quads, 6 full house, 5 flush,
# 4 straight, 3 trips, 2 two pair, 1 pair, 0 high card.
_STRAIGHT_FLUSH = 8
_FULL_HOUSE = 6
_FLUSH = 5
_STRAIGHT = 4
_ONE_PAIR = 1
_HIGH_CARD = 0


# --- the Omaha constraint (deterministic, the crux) -----------------------
def test_cannot_play_the_board():
    """Quad aces in hand, a straight flush ON the board.

    Naive best-5-of-9 = straight flush (8). Correct Omaha: the two playable
    aces have no diamond, so the board's straight flush is unreachable ->
    pair of aces (1).
    """
    rank = omaha_best_rank(
        ["As", "Ah", "Ad", "Ac"], ["Kd", "Qd", "Jd", "Td", "9d"]
    )
    assert rank[0] == _ONE_PAIR


def test_uses_exactly_two_hole_cards_for_a_flush():
    """Two suited hole cards that complete the board flush -> royal flush.

    Confirms the evaluator DOES use exactly two hole + three board when that
    is the nut line (Ad Kd + Qd Jd Td).
    """
    rank = omaha_best_rank(
        ["Ad", "Kd", "7c", "2h"], ["Qd", "Jd", "Td", "3s", "4s"]
    )
    assert rank[0] == _STRAIGHT_FLUSH


def test_one_suited_hole_card_cannot_make_a_flush():
    """A single diamond in hand cannot make a diamond flush (needs two).

    Naive best-5-of-9 = royal flush (Ad + four board diamonds). Correct
    Omaha: no pair, no straight, no flush -> high card (0).
    """
    rank = omaha_best_rank(
        ["Ad", "7c", "2h", "3s"], ["Kd", "Qd", "Jd", "Td", "9s"]
    )
    assert rank[0] == _HIGH_CARD


def test_straight_uses_exactly_two_hole_cards():
    """9-8 in hand + 7-6-5 on board makes a straight (positive control)."""
    rank = omaha_best_rank(
        ["9h", "8c", "2d", "3s"], ["7h", "6c", "5d", "Kh", "Qs"]
    )
    assert rank[0] == _STRAIGHT


def test_full_house_from_pair_in_hand_plus_board():
    """Two aces in hand + A K K on board -> aces full of kings (6)."""
    rank = omaha_best_rank(
        ["Ah", "Ad", "2c", "3s"], ["Ac", "Kh", "Kd", "7s", "8c"]
    )
    assert rank[0] == _FULL_HOUSE


def test_flush_needs_two_from_hand_present():
    """Two hearts in hand + three board hearts -> a real flush (5)."""
    rank = omaha_best_rank(
        ["Ah", "Th", "2c", "3s"], ["Kh", "8h", "4h", "9s", "2d"]
    )
    assert rank[0] == _FLUSH


# --- equity invariants (Monte Carlo, deterministic via fixed seed) --------
def test_complement_is_exact_with_same_seed():
    """eq(A vs B) + eq(B vs A) == 1.0 exactly.

    Same two hands -> same known cards -> same deck -> same seed -> identical
    sampled boards, so every board is a win for exactly one side (ties split
    evenly). A non-1.0 sum means the winner test is asymmetric.
    """
    hero = ["As", "Ks", "Qh", "Jh"]
    villain = ["Ad", "Kd", "Qc", "Jc"]
    eq_ab = preflop_hand_equity(hero, villain, n_samples=300, rng=random.Random(7))
    eq_ba = preflop_hand_equity(villain, hero, n_samples=300, rng=random.Random(7))
    assert eq_ab + eq_ba == pytest.approx(1.0)


def test_suit_isomorphic_hands_are_a_coinflip():
    """Two double-suited AKQJ hands (different suits) should be ~50/50."""
    eq = preflop_hand_equity(
        ["As", "Ks", "Qh", "Jh"],
        ["Ad", "Kd", "Qc", "Jc"],
        n_samples=600,
        rng=random.Random(11),
    )
    assert 0.44 < eq < 0.56


def test_premium_hand_dominates_junk():
    """AA-double-suited crushes disconnected rainbow junk (wide margin)."""
    eq = preflop_hand_equity(
        ["As", "Ah", "Ks", "Kh"],
        ["9d", "4c", "3h", "8s"],
        n_samples=600,
        rng=random.Random(3),
    )
    assert eq > 0.60


def test_card_conflict_returns_zero():
    """Hero and villain sharing a card -> 0.0 (impossible matchup)."""
    eq = preflop_hand_equity(
        ["As", "Ah", "Ks", "Kh"],
        ["As", "Qd", "Jc", "Tc"],  # shares As
        n_samples=50,
        rng=random.Random(1),
    )
    assert eq == 0.0


# --- range helpers --------------------------------------------------------
def test_split_combo():
    assert split_combo("AhKhQsJs") == ["Ah", "Kh", "Qs", "Js"]


def test_hand_rejects_wrong_card_count():
    with pytest.raises(ValueError, match="exactly 4 cards"):
        preflop_hand_equity(["As", "Ah", "Ks"], ["Ad", "Kd", "Qc", "Jc"])


def test_equity_vs_range_in_unit_interval():
    eq = preflop_equity_vs_range(
        ["As", "Ah", "Ks", "Kh"],
        {"9d4c3h8s": 1.0, "TdTc9h8c": 1.0},
        n_samples=80,
        rng=random.Random(5),
    )
    assert 0.0 < eq < 1.0


def test_equity_vs_range_all_conflicts_returns_zero():
    """Every villain combo shares a card with hero -> 0.0."""
    eq = preflop_equity_vs_range(
        ["As", "Ah", "Ks", "Kh"],
        {"AsAhKsKh": 1.0},  # identical to hero
        n_samples=50,
        rng=random.Random(5),
    )
    assert eq == 0.0


def test_range_vs_range_dominance_and_bounds():
    eq = preflop_range_vs_range_equity(
        {"AsAhKsKh": 1.0},
        {"9d4c3h8s": 1.0},
        max_matchups=40,
        n_samples_per_matchup=40,
        rng=random.Random(9),
    )
    assert 0.0 <= eq <= 1.0
    assert eq > 0.5  # premium hero range beats junk villain range


def test_range_vs_range_empty_returns_zero():
    assert preflop_range_vs_range_equity({}, {"9d4c3h8s": 1.0}) == 0.0
