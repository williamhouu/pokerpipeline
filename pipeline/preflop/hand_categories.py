"""Bucket a 169-class hand label into a named category (Zach's taxonomy).

Deterministic, explicit membership -- every class is listed by hand so there
is zero rule-inference to get subtly wrong. Used to describe a set of hands
(e.g. the leaning examples in the SOLVER DATA block) by TYPE rather than as a
raw list, so the LLM can say "your suited one-gappers and wheel aces lean
toward folding" instead of "72s, 74s, A5s".

Anything not explicitly listed falls into ``other`` (e.g. 72s, K9s, weak
offsuit junk).
"""
from __future__ import annotations

# Ordered so a human can scan it against the source table. Each class appears
# in exactly one bucket; lookup is first-match (they are disjoint anyway).
HAND_CATEGORIES: dict[str, frozenset[str]] = {
    "premium_pairs": frozenset({"AA", "KK", "QQ"}),
    "medium_pairs": frozenset({"JJ", "TT", "99"}),
    "small_pairs": frozenset({"88", "77", "66", "55", "44", "33", "22"}),
    "premium_broadways": frozenset({"AKs", "AKo", "AQs", "AQo"}),
    "ace_broadways": frozenset({"AJs", "AJo", "ATs", "ATo"}),
    "suited_broadways": frozenset({"KQs", "KJs", "QJs", "KTs", "QTs", "JTs"}),
    "offsuit_broadways": frozenset({"KQo", "KJo", "QJo", "KTo", "QTo", "JTo"}),
    "wheel_aces": frozenset({"A5s", "A4s", "A3s", "A2s"}),
    "other_suited_aces": frozenset({"A9s", "A8s", "A7s", "A6s"}),
    "weak_offsuit_aces": frozenset(
        {"A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o"}
    ),
    "suited_connectors": frozenset({"T9s", "98s", "87s", "76s", "65s", "54s"}),
    "suited_one_gappers": frozenset(
        {"J9s", "T8s", "97s", "86s", "75s", "64s", "53s"}
    ),
}

OTHER = "other"

# Reverse index for O(1) lookup, built once.
_CATEGORY_OF: dict[str, str] = {
    hand: category for category, hands in HAND_CATEGORIES.items() for hand in hands
}

# Human-readable singular form, for prose ("a wheel ace", "a suited
# one-gapper"). Keyed by category id.
CATEGORY_LABEL: dict[str, str] = {
    "premium_pairs": "a premium pair",
    "medium_pairs": "a medium pair",
    "small_pairs": "a small pair",
    "premium_broadways": "a premium broadway",
    "ace_broadways": "an ace-broadway",
    "suited_broadways": "a suited broadway",
    "offsuit_broadways": "an offsuit broadway",
    "wheel_aces": "a wheel ace",
    "other_suited_aces": "a suited ace",
    "weak_offsuit_aces": "a weak offsuit ace",
    "suited_connectors": "a suited connector",
    "suited_one_gappers": "a suited one-gapper",
    OTHER: "a hand",
}


# Plural form for describing a whole bucket ("your wheel aces fold 64%").
CATEGORY_LABEL_PLURAL: dict[str, str] = {
    "premium_pairs": "premium pairs",
    "medium_pairs": "medium pairs",
    "small_pairs": "small pairs",
    "premium_broadways": "premium broadways",
    "ace_broadways": "ace-broadways",
    "suited_broadways": "suited broadways",
    "offsuit_broadways": "offsuit broadways",
    "wheel_aces": "wheel aces",
    "other_suited_aces": "middle suited aces",
    "weak_offsuit_aces": "weak offsuit aces",
    "suited_connectors": "suited connectors",
    "suited_one_gappers": "suited one-gappers",
}


def categorize_hand_class(hand_class: str) -> str:
    """The bucket id for a 169-class label (e.g. 'A5s' -> 'wheel_aces').

    Returns ``'other'`` for anything not in the taxonomy.
    """
    return _CATEGORY_OF.get(hand_class, OTHER)


def combos_in_class(hand_class: str) -> int:
    """Combos in a 169-class label: pairs 6, suited 4, offsuit 12.

    Used to weight a class by how often it's actually dealt when rolling a
    bucket up to a group frequency (AKo's 12 combos count 3x a suited hand's).
    """
    if len(hand_class) == 2:  # a pair, e.g. "AA"
        return 6
    return 4 if hand_class.endswith("s") else 12
