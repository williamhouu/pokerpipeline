"""PLO question extractor (Layer 4) -- the worthiness gate.

Decides whether a :class:`~pipeline.plo.spot_sampler.PloSpot` is worth a
question. Ports :mod:`pipeline.preflop.question_extractor`: the gate is a pure
function of the spot's dominant-action frequency (the 55-95% window -- below,
no clear best answer; above, too obvious) and its presence (a hand that never
reaches the node isn't a real spot).

The canonical CSV difficulty rating comes from
:func:`pipeline.plo.difficulty.compute_plo_difficulty` (4-axis, post-facts);
:func:`difficulty_estimate` here is only the cheap freq-only estimate used at
the gate, before facts are extracted.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.plo.spot_sampler import PloSpot

MIN_TOP_FREQUENCY = 0.55  # below: no clear best answer to teach
MAX_TOP_FREQUENCY = 0.95  # above: the answer is too obvious
MIN_PRESENCE = 0.01  # below: hand doesn't actually reach the node
# The 90-95% "mostly-but-feels-like-always" labelling trap. The exclusion
# punches a HOLE here, [FLOOR, CEILING), rather than capping the ceiling, so a
# genuinely-pure 95-100% spot (a clear "Always X", at/above the options
# builder's 0.95 "Always" line) still qualifies when the window reaches it.
AMBIGUOUS_BAND_FLOOR = 0.90
AMBIGUOUS_BAND_CEILING = 0.95

_DIFFICULTY_CEILING = 3000
_DIFFICULTY_FLOOR = 500
_FREQ_FLOOR = 0.55
_FREQ_SPAN = 0.45
_DIFFICULTY_SPAN = 2500


def in_ambiguous_band(frequency: float) -> bool:
    """True if a dominant-action frequency sits in the 90-95% trap band."""
    return AMBIGUOUS_BAND_FLOOR <= frequency < AMBIGUOUS_BAND_CEILING


def is_question_worthy(
    spot: PloSpot,
    *,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_presence: float = MIN_PRESENCE,
    exclude_ambiguous_band: bool = False,
) -> bool:
    """True if the spot passes the presence filter AND the frequency window.

    The frequency window is inclusive on both ends. When
    ``exclude_ambiguous_band`` is True, the 90-95% band is additionally
    removed as a HOLE in the window (see :data:`AMBIGUOUS_BAND_FLOOR` /
    :data:`AMBIGUOUS_BAND_CEILING`): a 90-95% spot reads as "mostly" but sits
    just under the 0.95 "always" line, so it is a labelling trap. Punching a
    hole (rather than capping the ceiling at 90%) means a genuinely-pure
    95-100% spot still qualifies when the window's max reaches it -- e.g. a
    100% slider yields 55-90% PLUS 95-100%, skipping only the trap.

    Override the thresholds via keyword args -- a difficulty preset / custom
    slider passes through to ``min_frequency`` / ``max_frequency``.
    """
    if spot.presence < min_presence:
        return False
    freq = spot.dominant_frequency
    if not (min_frequency <= freq <= max_frequency):
        return False
    return not (exclude_ambiguous_band and in_ambiguous_band(freq))


def difficulty_estimate(spot: PloSpot) -> int:
    """Cheap pre-facts difficulty ESTIMATE from the frequency axis only.

    Linear from 55% (hardest = 3000) to 100% (easiest = 500), matching the
    freq axis of the canonical 4-axis rating so the estimate isn't far off.
    The CSV-bound score is the full :func:`compute_plo_difficulty`.
    """
    freq_easy = max(0.0, min(1.0, (spot.dominant_frequency - _FREQ_FLOOR) / _FREQ_SPAN))
    score = _DIFFICULTY_CEILING - freq_easy * _DIFFICULTY_SPAN
    return round(max(_DIFFICULTY_FLOOR, min(_DIFFICULTY_CEILING, score)))


@dataclass(frozen=True)
class PloQuestionEvaluation:
    """The extractor's verdict on a spot."""

    is_worthy: bool
    top_action_frequency: float
    total_presence: float
    difficulty_estimate: int


def evaluate_spot(
    spot: PloSpot,
    *,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_presence: float = MIN_PRESENCE,
    exclude_ambiguous_band: bool = False,
) -> PloQuestionEvaluation:
    """Run the worthiness gate + the cheap difficulty estimate."""
    return PloQuestionEvaluation(
        is_worthy=is_question_worthy(
            spot,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_presence=min_presence,
            exclude_ambiguous_band=exclude_ambiguous_band,
        ),
        top_action_frequency=spot.dominant_frequency,
        total_presence=spot.presence,
        difficulty_estimate=difficulty_estimate(spot),
    )


__all__ = [
    "AMBIGUOUS_BAND_CEILING",
    "AMBIGUOUS_BAND_FLOOR",
    "MAX_TOP_FREQUENCY",
    "MIN_PRESENCE",
    "MIN_TOP_FREQUENCY",
    "PloQuestionEvaluation",
    "difficulty_estimate",
    "evaluate_spot",
    "in_ambiguous_band",
    "is_question_worthy",
]
