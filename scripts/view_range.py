"""Render a Ryan-pack preflop range file as a 13x13 grid in the terminal.

The pack stores ONE FILE PER ACTION at each node, where every hand's value
is the frequency that hand takes that action. This prints any such file as
the standard poker range grid so you can eyeball it on a Mac (no PioSolver
/ PioViewer needed).

Layout: rows + columns run A,K,Q,...,2. Upper-right triangle = suited,
diagonal = pairs, lower-left = offsuit. Numbers are percentages; '.' = 0.

Usage:
    python scripts/view_range.py <path-to-range.txt>

Examples:
    # The HJ's opening range:
    python scripts/view_range.py \\
      "ranges/ryan_preflop_tree/PioViewer - NLH 6max 100bb 2.5x Open/HJ/UTG_Fold_HJ_60%.txt"

    # The CO's CALL range facing that open (i.e. what the CO flats):
    python scripts/view_range.py \\
      "ranges/ryan_preflop_tree/PioViewer - NLH 6max 100bb 2.5x Open/CO/UTG_Fold_HJ_60%_CO_Call.txt"
"""

from __future__ import annotations

import sys
from pathlib import Path

RANKS = "AKQJT98765432"


def load(path: str) -> dict[str, float]:
    """Parse a 'Hand:weight,Hand:weight,...' range file into a dict."""
    out: dict[str, float] = {}
    for tok in Path(path).read_text(encoding="utf-8").strip().split(","):
        if ":" in tok:
            hand, weight = tok.split(":", 1)
            try:
                out[hand.strip()] = float(weight)
            except ValueError:
                continue
    return out


def hand_at(i: int, j: int) -> str:
    """The 169-class label for grid cell (row i, col j)."""
    hi, lo = RANKS[i], RANKS[j]
    if i == j:
        return hi + lo
    if i < j:
        return hi + lo + "s"  # upper-right = suited
    return lo + hi + "o"  # lower-left = offsuit


def _combos(i: int, j: int) -> int:
    if i == j:
        return 6
    return 4 if i < j else 12


def _cell(v: float) -> str:
    if v >= 0.995:
        return "100"
    if v <= 0.005:
        return "  ."
    return f"{v * 100:3.0f}"


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(argv) == 2 else 1
    path = argv[1]
    weights = load(path)
    print(f"Range: {Path(path).name}")
    print(
        "(upper-right = suited, diagonal = pairs, lower-left = offsuit; "
        "numbers are %, '.' = 0)\n"
    )
    print("     " + "  ".join(f"{r:>3}" for r in RANKS))
    total = 0.0
    for i, r in enumerate(RANKS):
        cells = []
        for j in range(13):
            v = weights.get(hand_at(i, j), 0.0)
            cells.append(_cell(v))
            total += v * _combos(i, j)
        print(f"{r:>3}  " + "  ".join(cells))
    n_frac = sum(1 for v in weights.values() if 0.005 < v < 0.995)
    print(
        f"\nTotal: {total / 1326 * 100:.1f}% of all hands ({total:.0f} combos) · "
        f"{n_frac} hand classes are mixed (a fraction, not pure 0/1)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
