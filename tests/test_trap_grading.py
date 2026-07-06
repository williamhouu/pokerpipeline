"""Tests for the shared graded trap-difficulty floor (pipeline.trap_grading).

The leaf is consumed by BOTH the preflop and postflop difficulty modules
(their own suites cover the wiring); these tests pin the map itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.trap_grading import (  # noqa: E402
    TRAP_FLOOR_MAX,
    TRAP_FLOOR_MIN,
    TRAP_MARGIN_AT_MAX,
    TRAP_MARGIN_AT_MIN,
    graded_trap_floor,
)


def test_anchor_values() -> None:
    """The two anchors map exactly to the two floor endpoints."""
    assert graded_trap_floor(TRAP_MARGIN_AT_MIN) == TRAP_FLOOR_MIN
    assert graded_trap_floor(TRAP_MARGIN_AT_MAX) == TRAP_FLOOR_MAX


def test_clips_outside_the_anchors() -> None:
    """Below the detection threshold clips to the min floor (defensive --
    detectors can't fire below it); above saturation clips to the max."""
    assert graded_trap_floor(0.0) == TRAP_FLOOR_MIN
    assert graded_trap_floor(0.01) == TRAP_FLOOR_MIN
    assert graded_trap_floor(0.50) == TRAP_FLOOR_MAX
    assert graded_trap_floor(1.0) == TRAP_FLOOR_MAX


def test_linear_midpoint() -> None:
    """Halfway between the anchors grades halfway between the floors."""
    mid_margin = (TRAP_MARGIN_AT_MIN + TRAP_MARGIN_AT_MAX) / 2
    assert graded_trap_floor(mid_margin) == round(
        (TRAP_FLOOR_MIN + TRAP_FLOOR_MAX) / 2
    )


def test_monotonic_nondecreasing() -> None:
    """A bigger contradiction never grades EASIER."""
    margins = [i / 100 for i in range(0, 41)]
    floors = [graded_trap_floor(m) for m in margins]
    assert all(a <= b for a, b in zip(floors, floors[1:]))


def test_typical_8max_trap_keeps_rating_near_the_old_flat_floor() -> None:
    """Calibration guard: the median trap measured on the 8-max packs
    (~16 points of equity-vs-price contradiction) grades to ~2430, i.e.
    right where the old flat 2400 floor put every trap. If a retune moves
    this materially, re-check the Medium/Hard preset boundaries."""
    assert abs(graded_trap_floor(0.16) - 2400) <= 50


def test_bands_split_sensibly() -> None:
    """Mild traps land in the Medium preset band (1300-2100), strong ones
    in Hard (2100-3200) -- the gradation the flat floor couldn't give."""
    assert 1300 <= graded_trap_floor(0.05) < 2100
    assert 2100 <= graded_trap_floor(0.16) <= 3200
