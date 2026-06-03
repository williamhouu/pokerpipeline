"""Generate (and verify) the PLO combo-multiplicity table.

Each of the 16,432 suit-isomorphic Monker hands represents a different number
of *concrete* 4-card combos: a rainbow 4-distinct-rank hand is 24 combos, quad
aces is 1, a double-suited hand somewhere between. The total over all 16,432
hands is exactly C(52,4) = 270,725. That per-hand count -- the "multiplicity"
-- is what turns a `.rng` range (weights over suit-iso hands) into a real combo
count and a correctly-combo-weighted "% of hands", and what weights an equity
range so rainbow hands aren't under-counted vs double-suited ones.

The table is fully DERIVED from the hand order + suit-isomorphic canonical form
(no external source), but enumerating C(52,4) and canonicalising each combo
takes ~5 s -- too slow for import or the test suite. So we bake it to
``pipeline/plo/data/monker_combo_multiplicity.txt`` (one int per line, in `.rng`
index order) and load it cheaply at runtime, the same pattern as the hand order.

Run:  ``venv/bin/python scripts/plo_combo_multiplicity_gen.py``
(writes the data file, then verifies sum == 270,725 and a full bijection).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.equity import FULL_DECK  # noqa: E402
from pipeline.plo.hand_order import (  # noqa: E402
    HAND_COUNT,
    canonical_form,
    hand_order,
    parse_monker_label,
)

_OUT = Path(__file__).resolve().parent.parent / "pipeline" / "plo" / "data" / (
    "monker_combo_multiplicity.txt"
)
_TOTAL_COMBOS = 270_725


def compute_multiplicities() -> list[int]:
    """Return the concrete-combo count for each of the 16,432 hand indices."""
    cf_to_index = {
        canonical_form(parse_monker_label(label)): i
        for i, label in enumerate(hand_order())
    }
    if len(cf_to_index) != HAND_COUNT:
        msg = f"hand order is not a bijection: {len(cf_to_index)} distinct forms"
        raise ValueError(msg)

    counts = [0] * HAND_COUNT
    for combo in combinations(FULL_DECK, 4):
        counts[cf_to_index[canonical_form(list(combo))]] += 1
    return counts


def _verify(counts: list[int]) -> None:
    total = sum(counts)
    if total != _TOTAL_COMBOS:
        msg = f"multiplicities sum to {total}, expected {_TOTAL_COMBOS}"
        raise ValueError(msg)
    if min(counts) < 1:
        missing = counts.index(min(counts))
        msg = f"hand index {missing} has multiplicity {counts[missing]} (< 1)"
        raise ValueError(msg)


def main() -> None:
    counts = compute_multiplicities()
    _verify(counts)
    header = (
        "# Concrete-combo multiplicity per Monker hand index (parallel to\n"
        "# monker_hand_order.txt). One int per line; sums to C(52,4)=270725.\n"
        "# Regenerate with scripts/plo_combo_multiplicity_gen.py.\n"
    )
    _OUT.write_text(header + "\n".join(str(c) for c in counts) + "\n", encoding="utf-8")
    print(f"wrote {len(counts)} multiplicities to {_OUT}")
    print(f"  sum={sum(counts)} min={min(counts)} max={max(counts)}")


if __name__ == "__main__":
    main()
