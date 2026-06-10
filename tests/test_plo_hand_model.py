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
    describe_card_redundancy,
    describe_flush_potential,
    describe_suit_redundancy,
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
    assert describe_flush_potential("8c Qc Td Kd") == (
        "diamonds can make at best a K-high flush (second-nut), "
        "clubs can make at best a Q-high flush (third-nut)"
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


def test_jack_high_flush_is_weak_not_fourth_nut():
    # The J="fourth-nut" vs T="weak" label cliff between adjacent ranks fed
    # the "diamonds are a real flush, clubs are a backup" LLM invention.
    suits = {f.suit: f for f in flush_suits("7c Tc 9d Jd")}
    assert suits["d"].nut_label == "weak"
    assert suits["c"].nut_label == "weak"


def test_two_weak_suits_read_close_in_strength():
    # Draw tense ("can make at best") + the comparability clause, so the LLM
    # can neither claim a made flush nor rank one weak suit far above another.
    assert describe_flush_potential("7c Tc 9d Jd") == (
        "diamonds can make at best a J-high flush (weak), "
        "clubs can make at best a T-high flush (weak). "
        "Neither suit makes the nut flush, and the two are close in strength"
    )


def test_two_weak_suits_far_apart_skip_the_closeness_clause():
    # J-high vs 5-high: both weak and neither is the nut, but they are NOT
    # close in strength, so that clause must not appear.
    text = describe_flush_potential("Jc Tc 5d 2d")
    assert text.endswith("Neither suit makes the nut flush")
    assert "close in strength" not in text


def test_single_weak_suit_says_not_the_nut():
    assert describe_flush_potential("3h Th Kd 2s") == (
        "hearts can make at best a T-high flush (weak). Not the nut flush"
    )


def test_nut_suit_keeps_its_ranking_no_closeness_clause():
    # A real hierarchy (nut vs weak) is true and stays; no comparability talk.
    text = describe_flush_potential("As 2s 3h 4h")
    assert "spades can make at best an A-high flush (nut)" in text
    assert "close in strength" not in text
    assert "Neither suit" not in text


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


def test_double_paired_hands_have_no_dangler():
    # KK44: the fours are a set-mining PAIR, not a dead card -- no dangler
    # (and no descriptor note), though the hand is still "disconnected" as a
    # straight shape.
    h = _c("KcKd4c4d")
    assert h.pair_pattern == "two_pair"
    assert h.has_dangler is False
    assert h.connectedness == "disconnected"
    assert "with a dangler" not in h.descriptor


def test_one_pair_side_card_is_still_a_dangler():
    # QQ J 2: the unpaired deuce is far from everything -- a real dangler.
    h = _c("QcQdJh2s")
    assert h.has_dangler is True
    assert "with a dangler" in h.descriptor


def test_quads_have_no_dangler():
    # All four ranks paired -- nothing to dangle; the badness is redundancy.
    h = _c("KcKdKhKs")
    assert h.has_dangler is False
    assert h.connectedness == "disconnected"  # category unchanged


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


# --- card redundancy (trips/quads -> deterministic dead-card count) ---------
def test_trips_aces_has_exactly_one_redundant_ace():
    # The reported bug hand: 9c Ad Ah As. The LLM claimed "two of your three
    # aces are doing nothing" and invented a "fourth ace" -- the truth is the
    # AA pair uses two, so exactly ONE ace is redundant.
    text = describe_card_redundancy("9c Ad Ah As")
    assert text is not None
    assert "three aces" in text
    assert "ONE of the three is redundant" in text
    assert "bare pair of aces" in text
    assert "fourth" not in text


def test_quads_have_two_dead_cards_and_no_set_outs():
    text = describe_card_redundancy("Ac Ad Ah As")
    assert text is not None
    assert "all four aces" in text
    assert "two are pure dead weight" in text
    assert "never flop a set" in text


def test_trips_sixes_pluralization():
    text = describe_card_redundancy("6c 6d 6h Ks")
    assert text is not None
    assert "three sixes" in text
    assert "only one six is left in the deck" in text


def test_no_redundancy_for_pairs_or_unpaired():
    assert describe_card_redundancy("Ac Ad 9h 9s") is None  # two pair
    assert describe_card_redundancy("Ac Ad Kh 9s") is None  # one pair
    assert describe_card_redundancy("Ac Kd Qh Js") is None  # unpaired


# --- suit redundancy (3+ of one suit -> dead flush card) --------------------
def test_three_suited_has_one_dead_suit_card():
    # The KQJT failure hand: three diamonds, only two can ever play.
    text = describe_suit_redundancy("Jc Td Qd Kd")
    assert text is not None
    assert "three diamonds" in text
    assert "third diamond is a dead card" in text


def test_monotone_has_two_dead_suit_cards():
    text = describe_suit_redundancy("As Ks Qs Js")
    assert text is not None
    assert "all four of your cards are spades" in text
    assert "two are dead cards" in text


def test_no_suit_redundancy_for_two_or_fewer_per_suit():
    assert describe_suit_redundancy("As Ks Ah Kh") is None  # double-suited
    assert describe_suit_redundancy("As Kh Qd Jc") is None  # rainbow
    assert describe_suit_redundancy("As Ks Qh Jd") is None  # single-suited
    # Trips are rank redundancy, never suit redundancy (three suits occupied).
    assert describe_suit_redundancy("9c Ad Ah As") is None
