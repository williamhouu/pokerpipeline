"""Authoritative MonkerViewer Omaha hand order — the index→hand map for `.rng`.

A MonkerSolver/MonkerViewer `.rng` range file stores one ``p;ev`` payload per
suit-isomorphic PLO hand, in a FIXED order of **16,432 hands**. That order is
the list ``q.k`` hardcoded in ``monkerviewer-1.4.jar``; this module loads it
from ``data/monker_hand_order.txt`` and exposes the index→hand mapping the
`.rng` reader needs. The `.rng`'s i-th payload corresponds to ``hand_order()[i]``.

Validated three independent ways (see ``scripts/plo_hand_order_audit.py`` and
``docs/plo_rng_format.md``): MonkerViewer's own display code maps a loaded
index via ``Client.a.b(i) == q.k.get(i)``; the list is a bijection onto the
16,432 canonical hands; and on a real open-raise node it reads correctly
(AAKK-ds opens 100%, rainbow trash folds 100%).

**Monker hand notation:** ranks ``23456789TJQKA``; parentheses group same-suit
cards. ``AAKK`` = rainbow aces+kings; ``(AK)(AK)`` = double-suited; ``AA(2A)``
= the 2 suited to one ace. We parse each label into a *representative* concrete
4-card hand (the SHARING pattern is what matters; the hand is suit-isomorphic,
so any concrete suit assignment classifies identically).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from pathlib import Path

from pipeline.cards import RANK_VALUE, SUITS

_ORDER_FILE = Path(__file__).parent / "data" / "monker_hand_order.txt"
HAND_COUNT = 16432


@lru_cache(maxsize=1)
def hand_order() -> tuple[str, ...]:
    """The 16,432 Monker hand labels, in `.rng` index order."""
    labels = tuple(
        line.strip()
        for line in _ORDER_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if len(labels) != HAND_COUNT:
        msg = f"expected {HAND_COUNT} hands in {_ORDER_FILE.name}, got {len(labels)}"
        raise ValueError(msg)
    return labels


def monker_label(index: int) -> str:
    """The Monker hand string at a `.rng` index (0..16431)."""
    return hand_order()[index]


def parse_monker_label(label: str) -> list[str]:
    """Parse a Monker hand string into 4 representative cards (``'Ac'`` form).

    Cards inside the same parentheses share a suit; every other card gets a
    fresh suit. Suits are handed out from :data:`pipeline.cards.SUITS` in group
    order — an arbitrary but valid choice (Monker never groups two same-rank
    cards, so a representative is always four distinct cards).
    """
    cards: list[str] = []
    suit_idx = 0
    i = 0
    while i < len(label):
        if label[i] == "(":
            close = label.index(")", i)
            suit = SUITS[suit_idx]
            suit_idx += 1
            cards.extend(rank + suit for rank in label[i + 1:close])
            i = close + 1
        else:
            cards.append(label[i] + SUITS[suit_idx])
            suit_idx += 1
            i += 1
    if len(cards) != 4:
        msg = f"not a 4-card Omaha hand: {label!r} -> {cards}"
        raise ValueError(msg)
    return cards


def cards_at(index: int) -> list[str]:
    """A representative concrete 4-card hand for a `.rng` index."""
    return parse_monker_label(monker_label(index))


def canonical_form(cards: list[str]) -> tuple[tuple[int, int], ...]:
    """Suit-isomorphic canonical key for 4 cards (for dedup / equality).

    Two hands share a canonical form iff they are the same up to a relabeling
    of suits. Used to verify the order is a bijection onto the PLO hand set and
    to compare representatives against concrete combos.
    """
    parsed = [(RANK_VALUE[c[0]], SUITS.index(c[1])) for c in cards]
    best: tuple[tuple[int, int], ...] | None = None
    for perm in permutations(range(4)):
        mapped = tuple(sorted((r, perm[s]) for r, s in parsed))
        if best is None or mapped < best:
            best = mapped
    assert best is not None
    return best
