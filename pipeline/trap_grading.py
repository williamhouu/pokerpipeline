"""Graded trap-difficulty floor, shared by the preflop and postflop pipelines.

July 2026, replacing the flat 2400 floor. A "trap" spot (the solver's
dominant action contradicts the naive equity-vs-price pot-odds baseline)
used to be floored to a single constant, so at high frequencies EVERY trap
rated exactly 2400 -- no gradation for the app's Elo matching, and nothing
in the Medium band. This module maps HOW counterintuitive the spot is to a
floor between :data:`TRAP_FLOOR_MIN` and :data:`TRAP_FLOOR_MAX`, so mild
traps land upper-Medium and extreme ones land deep-Hard.

The grading margin is the RAKE-BLIND contradiction ``|equity - price|``:
what a player naively reasoning from pot odds actually feels. Trap
DETECTION (in each pipeline's ``_is_counterintuitive_spot``) keeps its
noise margin and rake cushion -- those decide WHETHER a spot is a trap;
this module only decides HOW HARD a detected trap rates. Measured on the
8-max packs (July 2026) trap margins run ~0.07-0.22 with a median ~0.16,
which grades to ~2430 -- so the typical trap keeps rating where the old
flat floor put it.

Pure functions, no pipeline imports -- safe as a leaf for the
self-contained postflop package (same precedent as ``bb_display``).
PLO is NOT wired up: it has no trap mode and no break-even price fact,
and the equity-vs-price baseline misfires there (PLO equities compress
toward 50%, so routinely-correct folds hold equity above the naive price).
"""

from __future__ import annotations

# A trap that barely qualifies (margin at the detection threshold) rates
# upper-Medium: counterintuitive, but only mildly.
TRAP_FLOOR_MIN: int = 1800
# A maximally-contradictory trap (equity ~a quarter of the pot's worth on
# the wrong side of the price) rates deep-Hard.
TRAP_FLOOR_MAX: int = 2900
# Margin anchors for the linear map. AT_MIN matches the detectors' noise
# margin (a spot can't fire below it); AT_MAX is where grading saturates.
TRAP_MARGIN_AT_MIN: float = 0.04
TRAP_MARGIN_AT_MAX: float = 0.25


def graded_trap_floor(margin: float) -> int:
    """Difficulty floor for a detected trap with the given equity margin.

    Args:
        margin: The naive contradiction size ``|hero equity - break-even
            price|``, in equity fraction (0.15 = fifteen points). Rake-blind
            on purpose: it measures what the pot odds LOOK like to a player,
            while the detectors' rake cushion handles whether the spot is a
            trap at all.

    Returns:
        An integer floor in [TRAP_FLOOR_MIN, TRAP_FLOOR_MAX], linear in the
        margin between the two anchors and clipped outside them. The caller
        applies it as ``score = max(natural score, floor)`` so a trap can
        never rate BELOW its ungraded score.
    """
    if margin <= TRAP_MARGIN_AT_MIN:
        return TRAP_FLOOR_MIN
    if margin >= TRAP_MARGIN_AT_MAX:
        return TRAP_FLOOR_MAX
    frac = (margin - TRAP_MARGIN_AT_MIN) / (TRAP_MARGIN_AT_MAX - TRAP_MARGIN_AT_MIN)
    return round(TRAP_FLOOR_MIN + frac * (TRAP_FLOOR_MAX - TRAP_FLOOR_MIN))


__all__ = [
    "TRAP_FLOOR_MAX",
    "TRAP_FLOOR_MIN",
    "TRAP_MARGIN_AT_MAX",
    "TRAP_MARGIN_AT_MIN",
    "graded_trap_floor",
]
