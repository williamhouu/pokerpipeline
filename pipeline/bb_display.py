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


def exact_amount_str(
    amount_bb: float,
    *,
    display_in_bb: bool = True,
    bb_in_dollars: float | None = None,
) -> str:
    """An EXACT amount string for the math panel's written-out equations.

    Deliberately NOT snapped to the 0.5bb display grid: an equation whose
    printed inputs don't reproduce its printed result destroys trust in the
    panel, so these amounts stay exact (to the cent / 0.01bb) even where the
    Question prose shows the rounded 2bb/4.5bb form (team decision, July
    2026). ``"2bb"`` / ``"9.5bb"`` / ``"2.33bb"``; dollars when the batch
    displays dollars: ``"$6"`` / ``"$7.50"``.
    """
    if display_in_bb or not bb_in_dollars:
        return f"{round(amount_bb, 2):g}bb"
    v = round(amount_bb * bb_in_dollars, 2)
    return f"${int(v)}" if v == int(v) else f"${v:.2f}"


__all__ = ["HALF_BB", "exact_amount_str", "round_to_half_bb"]
