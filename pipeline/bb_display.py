"""Shared big-blind display rounding (NLHE + PLO + postflop).

Solver bet sizes are pot fractions (33%, 67%, ...), so in big-blind terms they
land on ugly values like ``2.14bb`` or ``4.36bb``. For DISPLAY we snap
bb-denominated amounts to the nearest **0.5bb** so questions read cleanly
(``2bb`` / ``4.5bb`` / ``8bb``) -- the same 0.5bb grid the preflop range packs
already quantize their raise sizes to (``size_round_bb``).

This is **display-only**. The strategic facts (equity, pot odds, EV, SPR,
worthiness, difficulty, concept tags) are always computed from the EXACT
amounts -- rounding the underlying geometry would shift the pot-odds price and
could flip a borderline tag, which we explicitly do not want. It is also a no-op
for amounts already on the 0.5bb grid (so applying it to already-clean preflop
output changes nothing), and it never touches the dollar path (dollar amounts
keep their cents).

Caveat (acceptable, cosmetic): because each amount is snapped independently, a
multi-street pot built from several rounded wagers can read up to 0.5bb away
from the literal sum of the rounded wagers shown. Single-bet spots (the common
case) stay exactly consistent.
"""

from __future__ import annotations

# The display grid: snap to the nearest this-many big blinds.
HALF_BB = 0.5


def round_to_half_bb(amount_bb: float) -> float:
    """Snap a big-blind amount to the nearest 0.5bb (display only).

    ``2.14 -> 2.0``, ``4.36 -> 4.5``, ``7.8 -> 8.0``; ``2.5 -> 2.5`` (no-op for
    amounts already on the grid). Negative inputs are handled symmetrically.
    """
    return round(amount_bb / HALF_BB) * HALF_BB


__all__ = ["HALF_BB", "round_to_half_bb"]
