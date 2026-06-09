"""Tests for pipeline.plo.hand_model.

The suit / pair / connectedness / flag fields are exact properties of the
four cards, so they're asserted precisely. ``strength`` is a tunable
heuristic, so it's only checked on clear, directional cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.hand_model import (  # noqa: E402
    classify_plo_hand,
    describe_flush_potential,
    flush_suits,
)


def _c(hand: str):
    return classify_plo_hand(hand)


# --- flush nut-ranking -----------------------------------------------------
def test_king_high_flush_is_second_nut_not_nut():
    # The reported bug: K-high diamond flush must read "second-nut", not "nut".
    suits = {f.suit: f for f in flush_suits("8c Qc Td Kd")}
    assert suits["d"].high_rank == "K"
    assert suits["d"].nut_label == "second-nut"
    assert suits["d"].is_nut is False
    assert suits["c"].nut_label == "third-nut"  # Q-high clubs
    assert (
        describe_flush_potential("8c Qc Td Kd")
        == "diamonds K-high (second-nut flush), clubs Q-high (third-nut flush)"
    )


def test_ace_high_flush_is_the_nut():
    (f,) = flush_suits("Ad Kd 7c 2s")
    assert f.suit_word == "diamonds"
    assert f.nut_label == "nut"
    assert f.is_nut is True


def test_low_flush_is_weak():
    suits = {f.suit: f for f in flush_suits("3c Tc 6d Kd")}
    assert suits["c"].nut_label == "weak"  # T-high clubs
    assert suits["d"].nut_label == "second-nut"  # K-high diamonds


def test_rainbow_hand_has_no_flush_potential():
    assert flush_suits("As Kh Qd Jc") == ()
    assert describe_flush_potential("As Kh Qd Jc") == "none (no two cards share a suit)"


# --- suit patterns --------------------------------------------------------
def test_double_suited():
    assert _c("AsKsQhJh").suit_pattern == "double_suited"
    assert _c("AsKsQhJh").double_suited is True


def test_single_suited():
    assert _c("AsKsQhJd").suit_pattern == "single_suited"
    assert _c("AsKsQhJd").double_suited is False


def test_rainbow():
    assert _c("AsKhQcJd").suit_pattern == "rainbow"


def test_three_suited():
    assert _c("AsKsQsJh").suit_pattern == "three_suited"


def test_monotone():
    assert _c("AsKsQsJs").suit_pattern == "monotone"


# --- pair patterns --------------------------------------------------------
def test_unpaired():
    h = _c("AsKhQcJd")
    assert h.pair_pattern == "unpaired"
    assert h.pair_ranks == ()


def test_one_pair():
    h = _c("AsAhKsQh")
    assert h.pair_pattern == "one_pair"
    assert h.pair_ranks == (14,)


def test_two_pair():
    h = _c("KsKhQsQh")
    assert h.pair_pattern == "two_pair"
    assert h.pair_ranks == (13, 12)


def test_trips():
    h = _c("AhAdAcKs")
    assert h.pair_pattern == "trips"
    assert h.pair_ranks == (14,)


def test_quads():
    h = _c("AhAdAcAs")
    assert h.pair_pattern == "quads"


# --- connectedness / wrap (ace high or low) -------------------------------
def test_broadway_rundown_ace_high():
    h = _c("AsKhQcJd")
    assert h.connectedness == "rundown"
    assert h.span == 3
    assert h.wrap_potential is True


def test_wheel_rundown_uses_ace_low():
    """A-2-3-4 is a rundown only with the ace played low."""
    h = _c("Ah2c3d4s")
    assert h.connectedness == "rundown"
    assert h.span == 3
    assert h.has_dangler is False


def test_one_gapper():
    h = _c("JcTd9h7s")  # J T 9 _ 7
    assert h.connectedness == "one_gapper"
    assert h.wrap_potential is True


def test_two_gapper():
    h = _c("JcTd8h6s")  # span 5 across 4 distinct ranks
    assert h.connectedness == "two_gapper"
    assert h.wrap_potential is False  # only rundown / one-gapper flop big wraps


def test_loosely_connected_no_dangler():
    h = _c("JcTd8h5s")  # J T 8 5 -- spread but each card has a near neighbour
    assert h.connectedness == "connected"
    assert h.has_dangler is False


def test_disconnected_with_dangler():
    h = _c("KsQhJc4d")  # the 4 is >4 ranks from K/Q/J, no ace to rescue it
    assert h.connectedness == "disconnected"
    assert h.has_dangler is True
    assert h.wrap_potential is False


# --- nut / high-card flags ------------------------------------------------
def test_suited_ace_flag():
    assert _c("AsKsQhJh").suited_ace is True  # As shares spades with Ks
    assert _c("AhKsQcJd").suited_ace is False  # lone ace of hearts


def test_has_ace_and_broadway_count():
    h = _c("AsKsQhJh")
    assert h.has_ace is True
    assert h.broadway_count == 4
    assert _c("9c8d7h6s").has_ace is False
    assert _c("9c8d7h6s").broadway_count == 0


# --- strength buckets (heuristic, clear cases only) -----------------------
def test_premium_hands():
    assert _c("AsAhKsKh").strength == "premium"  # aces double-suited w/ KK
    assert _c("AsKsQhJh").strength == "premium"  # broadway rundown ds


def test_strong_hand():
    assert _c("JsTs9h8h").strength == "strong"  # JT98 double-suited


def test_low_rundown_is_demoted():
    """5432 double-suited is a real hand but non-nut -- not 'strong'."""
    s = _c("5s4s3h2h").strength
    assert s in {"medium", "marginal"}


def test_junk_is_weak_or_trash():
    assert _c("9c4d3h8s").strength in {"weak", "trash", "marginal"}
    assert _c("Kc7d2h8s").strength in {"weak", "trash"}
    assert _c("2c2d3h3s").strength == "trash"  # low rainbow double pair


# --- descriptor (prose) ---------------------------------------------------
def test_descriptor_broadway_rundown():
    assert _c("AsKsQhJh").descriptor == "double-suited broadway rundown"


def test_descriptor_pair():
    assert _c("AsAhKsQh").descriptor == "double-suited aces"


def test_descriptor_two_pair():
    assert _c("KsKhQsQh").descriptor == "double-suited kings and queens"


def test_descriptor_dangler_note():
    assert "with a dangler" in _c("AhAdJc2s").descriptor


# --- normalisation --------------------------------------------------------
def test_accepts_iterable_and_string_equivalently():
    from_str = _c("AsKsQhJh")
    from_list = classify_plo_hand(["As", "Ks", "Qh", "Jh"])
    assert from_str == from_list


def test_rejects_wrong_card_count():
    with pytest.raises(ValueError, match="exactly 4 cards"):
        classify_plo_hand("AsKsQh")


def test_rejects_duplicate_cards():
    with pytest.raises(ValueError, match="duplicate"):
        classify_plo_hand("AsAsKhQd")
