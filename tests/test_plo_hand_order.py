"""Tests for pipeline.plo.hand_order (the authoritative .rng index->hand map)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.hand_order import (  # noqa: E402
    HAND_COUNT,
    canonical_form,
    cards_at,
    hand_order,
    monker_label,
    parse_monker_label,
)


def test_order_loads_16432_hands():
    order = hand_order()
    assert len(order) == HAND_COUNT == 16432  # noqa: PLR2004
    assert order[0] == "AAAA"
    assert order[-1] == "KKKK"


def test_monker_label_indexing():
    assert monker_label(0) == "AAAA"
    assert monker_label(HAND_COUNT - 1) == "KKKK"


def test_parse_label_rainbow_quads():
    # AAAA = four aces, all different suits.
    cards = parse_monker_label("AAAA")
    assert len(cards) == 4
    assert all(c[0] == "A" for c in cards)
    assert len({c[1] for c in cards}) == 4  # four distinct suits


def test_parse_label_suit_groups():
    # (AK)(AK) = double-suited: A&K share one suit, the other A&K share another.
    cards = parse_monker_label("(AK)(AK)")
    by_suit: dict[str, list[str]] = {}
    for c in cards:
        by_suit.setdefault(c[1], []).append(c[0])
    sizes = sorted(len(v) for v in by_suit.values())
    assert sizes == [2, 2]  # two suited pairs

    # AA(2A) = the 2 shares a suit with one ace; three aces total.
    cards = parse_monker_label("AA(2A)")
    ranks = sorted(c[0] for c in cards)
    assert ranks == ["2", "A", "A", "A"]


def test_cards_at_are_four_distinct_valid_cards():
    for i in (0, 1, 2, 5000, HAND_COUNT - 1):
        cards = cards_at(i)
        assert len(cards) == 4
        assert len(set(cards)) == 4  # no duplicate cards


def test_order_is_a_bijection_onto_the_plo_hand_set():
    """Every entry parses to 4 distinct cards and all 16,432 canonical forms
    are distinct. Since there are *exactly* 16,432 suit-isomorphic PLO hands,
    distinct + valid + count-16432 proves the order is the full set, by
    pigeonhole -- no need to enumerate all 270k combos here."""
    seen = set()
    for i in range(HAND_COUNT):
        cards = cards_at(i)
        assert len(set(cards)) == 4, monker_label(i)  # valid hand
        seen.add(canonical_form(cards))
    assert len(seen) == HAND_COUNT


def test_canonical_form_is_suit_isomorphic():
    # Same hand, different concrete suits -> same canonical form.
    assert canonical_form(["As", "Ks", "Qh", "Jh"]) == canonical_form(
        ["Ah", "Kh", "Qd", "Jd"]
    )
    # Different hand -> different form.
    assert canonical_form(["As", "Ks", "Qh", "Jh"]) != canonical_form(
        ["As", "Kh", "Qd", "Jc"]
    )


def test_parse_rejects_non_four_card_label():
    with pytest.raises(ValueError, match="4-card"):
        parse_monker_label("AAA")
