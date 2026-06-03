"""Exhaustive audit of the PLO hand model + concept tags.

Enumerates all C(52,4) = 270,725 four-card combos, classifies each, runs the
hand-structure tagger, and (1) asserts structural invariants on every hand,
(2) prints the tag + strength distribution so we can eyeball that the tag set
is complete and no tag is dead or always-on.

Run:  venv/bin/python scripts/plo_tag_simulation.py
Exits non-zero if any invariant is violated (prints the offending hands).
"""

from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cards import RANKS, SUITS  # noqa: E402
from pipeline.plo.concept_tags import (  # noqa: E402
    _HAND_TAG_REGISTRY,
    CONNECTEDNESS_TAGS,
    PAIR_TAGS,
    SUIT_TAGS,
    compute_plo_hand_tags,
)
from pipeline.plo.hand_model import PloHandClass, classify_plo_hand  # noqa: E402

_SUIT_NAMES = {fn.__name__ for fn in SUIT_TAGS}
_PAIR_NAMES = {fn.__name__ for fn in PAIR_TAGS}
_CONNECT_NAMES = {fn.__name__ for fn in CONNECTEDNESS_TAGS}
_MIN_TAGS = 3  # the three partition tags always fire


def check_invariants(tags: set[str], hand: PloHandClass) -> list[str]:
    """Return a list of invariant-violation messages (empty == all good)."""
    errs: list[str] = []

    def want_one(group: set[str], label: str) -> None:
        hit = tags & group
        if len(hit) != 1:
            errs.append(f"{label} partition not exactly-one: {sorted(hit)}")

    want_one(_SUIT_NAMES, "suit")
    want_one(_PAIR_NAMES, "pair")
    want_one(_CONNECT_NAMES, "connectedness")

    if len(tags) < _MIN_TAGS:
        errs.append(f"coverage < {_MIN_TAGS}: {sorted(tags)}")
    if "bare_ace" in tags and "nut_flush_potential" in tags:
        errs.append("bare_ace AND nut_flush_potential")
    if "bare_ace" in tags and "double_suited" in tags:
        errs.append("bare_ace AND double_suited (an ace in a ds hand is suited)")
    if "broadway_rundown" in tags and not (
        "rundown" in tags and "all_broadway" in tags
    ):
        errs.append("broadway_rundown without rundown+all_broadway")
    if "all_broadway" in tags and "broadway_heavy" not in tags:
        errs.append("all_broadway without broadway_heavy")
    if "wrap_potential" in tags and not (
        "unpaired_hand" in tags
        and ("rundown" in tags or "one_gap_rundown" in tags)
    ):
        errs.append("wrap_potential without unpaired rundown/one-gapper")
    if "premium_hand" in tags and "trash_hand" in tags:
        errs.append("premium_hand AND trash_hand")
    if "pocket_aces" in tags and "unpaired_hand" in tags:
        errs.append("pocket_aces AND unpaired_hand")
    if "low_cards" in tags and (
        tags
        & {"nut_flush_potential", "bare_ace", "all_broadway", "broadway_heavy",
           "pocket_aces", "pocket_kings"}
    ):
        errs.append("low_cards with a high-card/ace tag")
    return errs


def main() -> int:
    deck = [r + s for r in RANKS for s in SUITS]
    counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    total = 0
    violations: list[tuple[str, list[str]]] = []

    for combo in combinations(deck, 4):
        hand = classify_plo_hand(combo)
        tags = compute_plo_hand_tags(hand)
        tagset = set(tags)
        total += 1
        strength_counts[hand.strength] += 1
        for t in tags:
            counts[t] += 1
            examples.setdefault(t, "".join(combo))
        errs = check_invariants(tagset, hand)
        if errs:
            violations.append(("".join(combo), errs))

    print(f"Enumerated {total:,} PLO combos\n")
    print(f"{'tag':<22}{'count':>9}{'pct':>8}   example")
    print("-" * 60)
    for fn in _HAND_TAG_REGISTRY:
        name = fn.__name__
        c = counts[name]
        print(f"{name:<22}{c:>9,}{c / total:>8.1%}   {examples.get(name, '-')}")

    print("\nstrength buckets:")
    for label in ("premium", "strong", "medium", "marginal", "weak", "trash"):
        c = strength_counts[label]
        print(f"  {label:<10}{c:>9,}{c / total:>8.1%}")

    dead = [fn.__name__ for fn in _HAND_TAG_REGISTRY if counts[fn.__name__] == 0]
    if dead:
        print(f"\nWARNING -- tags that never fired: {dead}")

    if violations:
        print(f"\nFAIL -- {len(violations):,} invariant violations. First 15:")
        for combo_label, errs in violations[:15]:
            print(f"  {combo_label}: {'; '.join(errs)}")
        return 1
    print(f"\nOK -- all {total:,} hands satisfy every invariant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
