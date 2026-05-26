"""Tests for pipeline.preflop.equity (Monte Carlo preflop equity)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.equity import (                                       # noqa: E402
    preflop_equity_vs_range,
    preflop_hand_equity,
)


# --- preflop_hand_equity ----------------------------------------------------
def test_AA_dominates_72o():
    """Pocket aces beat 7-2 offsuit ~85% preflop. Sampled at N=500 the
    estimate should be within ~3% of that figure."""
    rng = random.Random(0)
    eq = preflop_hand_equity(
        ("As", "Ac"), ("7h", "2d"), n_samples=500, rng=rng,
    )
    assert 0.80 < eq < 0.90, f"AA vs 72o expected ~85%, got {eq:.2%}"


def test_AKs_vs_QQ_approximately_coinflip():
    """AKs vs QQ is the canonical "race." Expected ~46% for AKs."""
    rng = random.Random(0)
    eq = preflop_hand_equity(
        ("As", "Ks"), ("Qh", "Qd"), n_samples=500, rng=rng,
    )
    assert 0.42 < eq < 0.50, f"AKs vs QQ expected ~46%, got {eq:.2%}"


def test_card_conflict_returns_zero():
    """If hero and villain share a card, equity is 0.0 by convention."""
    eq = preflop_hand_equity(("As", "Ks"), ("As", "Qd"))
    assert eq == 0.0


def test_deterministic_with_same_seed():
    """Same RNG seed -> identical equity (Monte Carlo is reproducible)."""
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    eq_a = preflop_hand_equity(("As", "Ac"), ("Kh", "Kd"), n_samples=100, rng=rng_a)
    eq_b = preflop_hand_equity(("As", "Ac"), ("Kh", "Kd"), n_samples=100, rng=rng_b)
    assert eq_a == eq_b


# --- preflop_equity_vs_range ------------------------------------------------
def test_AA_vs_tight_range():
    """AA vs {KK: 1.0, QQ: 1.0}: should crush both (~80%)."""
    rng = random.Random(0)
    eq = preflop_equity_vs_range(
        hero=("As", "Ac"),
        villain_range={"KsKd": 1.0, "QhQs": 1.0},
        n_samples=300, rng=rng,
    )
    assert eq > 0.75


def test_72o_vs_premium_range():
    """72o vs {AA, KK, QQ}: should lose badly."""
    rng = random.Random(0)
    eq = preflop_equity_vs_range(
        hero=("7c", "2d"),
        villain_range={"AhAs": 1.0, "KhKs": 1.0, "QhQs": 1.0},
        n_samples=300, rng=rng,
    )
    assert eq < 0.20


def test_skips_card_conflict_combos():
    """If hero blocks one combo, it's skipped; remaining combos drive the avg."""
    rng = random.Random(0)
    # Hero has As; villain range has 'AhAd' (no conflict) and 'AsAc' (conflict).
    # The 'AsAc' entry is skipped; equity is purely vs AhAd.
    eq = preflop_equity_vs_range(
        hero=("As", "Kh"),
        villain_range={"AhAd": 1.0, "AsAc": 1.0},
        n_samples=200, rng=rng,
    )
    # AK off vs AA = ~7% equity (dominated)
    assert eq < 0.15


def test_zero_total_weight_returns_zero():
    """All villain combos zero-weighted -> equity is 0.0."""
    eq = preflop_equity_vs_range(
        hero=("As", "Ks"),
        villain_range={"QhQd": 0.0, "JhJd": 0.0},
    )
    assert eq == 0.0


def test_negative_weights_treated_as_zero():
    """Negative weights are ignored (defensive against weird input)."""
    eq = preflop_equity_vs_range(
        hero=("As", "Ks"),
        villain_range={"QhQd": -1.0, "JhJd": -1.0},
    )
    assert eq == 0.0


def test_weighted_average():
    """Heavily-weighted bad combos drag down equity even if there's a token AA."""
    rng = random.Random(0)
    eq = preflop_equity_vs_range(
        hero=("Ks", "Kc"),
        villain_range={"AhAs": 0.01, "5h2d": 1.0},  # ~99% trash, ~1% AA
        n_samples=300, rng=rng,
    )
    # KK crushes 52o (~90%) and loses to AA (~20%). Heavy weight on 52o ->
    # near 90%.
    assert eq > 0.80
