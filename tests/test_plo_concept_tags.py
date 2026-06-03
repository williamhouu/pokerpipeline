"""Tests for pipeline.plo.concept_tags.

Per-tag firing checks, plus a sampled invariant loop mirroring the exhaustive
audit in scripts/plo_tag_simulation.py (which covers all 270,725 combos).
"""

from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cards import RANKS, SUITS  # noqa: E402
from pipeline.plo.concept_tags import (  # noqa: E402
    CONNECTEDNESS_TAGS,
    PAIR_TAGS,
    SUIT_TAGS,
    compute_plo_hand_tags,
)
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402


def _tags(hand: str) -> set[str]:
    return set(compute_plo_hand_tags(classify_plo_hand(hand)))


# --- one representative hand per tag --------------------------------------
def test_each_tag_fires_on_a_representative_hand():
    cases = {
        "double_suited": "AsKsQhJh",
        "single_suited": "AsKsQhJd",
        "rainbow": "AsKhQcJd",
        "three_suited": "AsKsQsJh",
        "monotone": "AsKsQsJs",
        "unpaired_hand": "AsKhQcJd",
        "single_pair": "AsAhKsQh",
        "double_paired": "KsKhQsQh",
        "trips_in_hand": "AhAdAcKs",
        "quads_in_hand": "AhAdAcAs",
        "pocket_aces": "AhAdKsQh",
        "pocket_kings": "KhKdAsQh",
        "rundown": "AsKhQcJd",
        "one_gap_rundown": "JcTd9h7s",
        "two_gap_rundown": "JcTd8h6s",
        "connected_hand": "JcTd8h5s",
        "disconnected_hand": "KsQhJc4d",
        "wrap_potential": "JcTd9h8s",
        "has_dangler": "KsQhJc4d",
        "broadway_rundown": "AsKsQhJh",
        "nut_flush_potential": "AsKsQhJh",
        "bare_ace": "AhKsQsJd",
        "all_broadway": "AsKsQhJh",
        "broadway_heavy": "AsKsQh9d",
        "low_cards": "8c7d5h3s",
        "premium_hand": "AsAhKsKh",
        "trash_hand": "2c2d3h3s",
    }
    for tag, hand in cases.items():
        assert tag in _tags(hand), f"{tag} should fire on {hand}, got {_tags(hand)}"


# --- negative / disambiguation cases --------------------------------------
def test_bare_ace_excludes_nut_flush_and_double_suited():
    t = _tags("AhKsQsJd")  # lone ace of hearts; spades are the suited pair
    assert "bare_ace" in t
    assert "nut_flush_potential" not in t
    assert "double_suited" not in t


def test_low_cards_excludes_high_card_tags():
    t = _tags("8c7d5h3s")
    assert "low_cards" in t
    assert t.isdisjoint(
        {"nut_flush_potential", "bare_ace", "all_broadway", "broadway_heavy",
         "pocket_aces", "pocket_kings"}
    )


def test_rainbow_is_not_suited():
    t = _tags("AsKhQcJd")
    assert "rainbow" in t
    assert t.isdisjoint({"double_suited", "single_suited", "three_suited", "monotone"})


def test_wheel_rundown_fires_rundown_and_wrap():
    t = _tags("Ah2c3d4s")
    assert "rundown" in t
    assert "wrap_potential" in t


# --- sampled invariants (exhaustive version in scripts/plo_tag_simulation) -
def test_partition_and_consistency_invariants_on_a_sample():
    deck = [r + s for r in RANKS for s in SUITS]
    all_combos = list(combinations(deck, 4))
    sample = random.Random(1234).sample(all_combos, 4000)

    suit_names = {fn.__name__ for fn in SUIT_TAGS}
    pair_names = {fn.__name__ for fn in PAIR_TAGS}
    connect_names = {fn.__name__ for fn in CONNECTEDNESS_TAGS}

    for combo in sample:
        tags = set(compute_plo_hand_tags(classify_plo_hand(combo)))
        label = "".join(combo)
        # exactly-one partitions
        assert len(tags & suit_names) == 1, label
        assert len(tags & pair_names) == 1, label
        assert len(tags & connect_names) == 1, label
        assert len(tags) >= 3, label  # noqa: PLR2004 -- the 3 partition tags
        # consistency
        assert not ("bare_ace" in tags and "nut_flush_potential" in tags), label
        assert not ("bare_ace" in tags and "double_suited" in tags), label
        assert not ("premium_hand" in tags and "trash_hand" in tags), label
        assert not ("pocket_aces" in tags and "unpaired_hand" in tags), label
        if "broadway_rundown" in tags:
            assert {"rundown", "all_broadway"} <= tags, label
        if "wrap_potential" in tags:
            assert "unpaired_hand" in tags, label
            assert tags & {"rundown", "one_gap_rundown"}, label
