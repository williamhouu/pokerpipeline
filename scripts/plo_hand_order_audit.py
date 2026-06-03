"""Audit the authoritative MonkerViewer hand order (pipeline.plo.hand_order).

Two checks:
  1. BIJECTION (pack-independent): the 16,432 loaded hands, canonicalized,
     equal the full set of suit-isomorphic PLO hands enumerated from scratch
     over all C(52,4) combos.
  2. STRATEGY (needs the pack): on the LJ open-raise node `40100.rng`, premium
     hands must open ~100% and rainbow trash ~0% under the index->hand map.

Run:  venv/bin/python scripts/plo_hand_order_audit.py
"""

from __future__ import annotations

import sys
from itertools import combinations, permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cards import RANK_VALUE, SUITS  # noqa: E402
from pipeline.plo.hand_order import (  # noqa: E402
    HAND_COUNT,
    canonical_form,
    cards_at,
)

_PACK = Path("plo_ranges/ranges/Omaha/6-way/100bb(5p-1bb)")


def _generate_canonical_set() -> set[tuple[tuple[int, int], ...]]:
    deck = [(r, s) for r in range(2, 15) for s in range(4)]
    out = set()
    for combo in combinations(deck, 4):
        best = None
        for perm in permutations(range(4)):
            m = tuple(sorted((r, perm[s]) for r, s in combo))
            if best is None or m < best:
                best = m
        out.add(best)
    return out


def _p_at(node: Path, index: int) -> float:
    lines = [ln for ln in node.read_text().split("\n") if ln != ""]
    return float(lines[2 * index + 1].split(";")[0])


def main() -> int:
    loaded = {canonical_form(cards_at(i)) for i in range(HAND_COUNT)}
    print(f"loaded distinct canonical hands: {len(loaded)}")
    generated = _generate_canonical_set()
    print(f"generated canonical PLO hands:   {len(generated)}")
    if loaded != generated:
        print("FAIL: order is not a bijection onto the PLO hand set")
        return 1
    print("OK: bijection confirmed.\n")

    node = _PACK / "40100.rng"
    if not node.exists():
        print(f"(skipping strategy check -- {node} not present)")
        return 0

    idx = {canonical_form(cards_at(i)): i for i in range(HAND_COUNT)}

    def hand(*cards: str) -> int:
        return idx[canonical_form(list(cards))]

    s, h, d, c = SUITS[3], SUITS[2], SUITS[1], SUITS[0]  # noqa: F841 -- readability
    checks = [
        ("AAKK double-suited", ["A" + s, "A" + h, "K" + s, "K" + h], 1.0),
        ("JT98 double-suited", ["J" + s, "T" + s, "9" + h, "8" + h], 1.0),
        ("2233 rainbow trash", ["2" + s, "2" + h, "3" + d, "3" + c], 0.0),
        ("K723 rainbow trash", ["K" + s, "7" + h, "2" + d, "3" + c], 0.0),
    ]
    print("40100.rng (LJ open-raise range):")
    ok = True
    for name, cards, expected in checks:
        assert all(card[0] in RANK_VALUE for card in cards)  # sanity
        p = _p_at(node, hand(*cards))
        flag = "ok" if abs(p - expected) < 0.05 else "MISMATCH"  # noqa: PLR2004
        ok = ok and flag == "ok"
        print(f"  {name:22} p={p:<5} (expected ~{expected})  {flag}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
