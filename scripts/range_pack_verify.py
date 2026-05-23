"""Range-pack verification helper for docs/ryan_range_pack_index.md.

Parses a Ryan-pack .txt range file (single line Hand:weight CSV) and prints
composition metrics. Used during reconnaissance only -- not part of the
pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Hand -> number of 4-suit combos. Pairs=6, suited=4, offsuit=12.
def combos(hand: str) -> int:
    if len(hand) == 2:                   # AA, KK, ..., 22
        return 6
    if hand.endswith("s"):
        return 4
    if hand.endswith("o"):
        return 12
    raise ValueError(f"unknown hand format: {hand}")


def parse(path: Path) -> dict[str, float]:
    raw = path.read_text(encoding="utf-8").strip()
    out: dict[str, float] = {}
    for pair in raw.split(","):
        h, w = pair.split(":")
        out[h.strip()] = float(w)
    return out


def summary(path: Path) -> str:
    r = parse(path)
    n_hands_total = len(r)
    weighted_combos = sum(w * combos(h) for h, w in r.items())
    pct_of_range = 100.0 * weighted_combos / 1326
    full_weight = [h for h, w in r.items() if w >= 0.999]
    partial = [(h, w) for h, w in r.items() if 0.001 < w < 0.999]
    zero = sum(1 for w in r.values() if w <= 0.001)
    top5 = sorted(r.items(), key=lambda kv: -kv[1])[:8]
    top5_str = ", ".join(f"{h}:{w:.2f}" for h, w in top5)
    out = [
        f"  {pct_of_range:>5.1f}% of all hands (weighted)",
        f"  {len(full_weight):>3d} hand classes at full weight, "
        f"{len(partial):>3d} partial, {zero:>3d} at zero",
        f"  top-weighted hands: {top5_str}",
    ]
    return "\n".join(out)


def main(argv):
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            print(f"{arg}: NOT FOUND"); continue
        print(f"--- {path.name}")
        print(summary(path))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
