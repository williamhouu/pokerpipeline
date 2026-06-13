"""Pins Zach's exact hand-bucket taxonomy. If a hand moves buckets, this
fails loudly -- the taxonomy is a product decision, not an implementation
detail."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.preflop.hand_categories import (  # noqa: E402
    HAND_CATEGORIES,
    categorize_hand_class,
)

# The source-of-truth table, transcribed from Zach's message.
EXPECTED: dict[str, list[str]] = {
    "premium_pairs": ["AA", "KK", "QQ"],
    "medium_pairs": ["JJ", "TT", "99"],
    "small_pairs": ["88", "77", "66", "55", "44", "33", "22"],
    "premium_broadways": ["AKs", "AKo", "AQs", "AQo"],
    "ace_broadways": ["AJs", "AJo", "ATs", "ATo"],
    "suited_broadways": ["KQs", "KJs", "QJs", "KTs", "QTs", "JTs"],
    "offsuit_broadways": ["KQo", "KJo", "QJo", "KTo", "QTo", "JTo"],
    "wheel_aces": ["A5s", "A4s", "A3s", "A2s"],
    "other_suited_aces": ["A9s", "A8s", "A7s", "A6s"],
    "weak_offsuit_aces": ["A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o"],
    "suited_connectors": ["T9s", "98s", "87s", "76s", "65s", "54s"],
    "suited_one_gappers": ["J9s", "T8s", "97s", "86s", "75s", "64s", "53s"],
}


def test_every_listed_hand_lands_in_its_bucket() -> None:
    for bucket, hands in EXPECTED.items():
        for hand in hands:
            assert categorize_hand_class(hand) == bucket, f"{hand} -> wrong bucket"


def test_buckets_match_the_module_exactly() -> None:
    # No extra or missing members vs the transcribed table.
    assert {k: set(v) for k, v in EXPECTED.items()} == {
        k: set(v) for k, v in HAND_CATEGORIES.items()
    }


def test_buckets_are_disjoint() -> None:
    seen: set[str] = set()
    for hands in HAND_CATEGORIES.values():
        assert not (seen & hands), "a hand is in two buckets"
        seen |= hands


def test_unlisted_hands_are_other() -> None:
    # The screenshot's leaning hands: 72s/74s are junk, A5s is a wheel ace.
    assert categorize_hand_class("72s") == "other"
    assert categorize_hand_class("74s") == "other"
    assert categorize_hand_class("A5s") == "wheel_aces"
    # A few more non-members.
    for junk in ("K9s", "Q8s", "95o", "32s", "T7o"):
        assert categorize_hand_class(junk) == "other"
