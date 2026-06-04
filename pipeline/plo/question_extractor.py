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
AMBIGUOUS_BAND_FLOOR = 0.90  # the 90-95% "mostly-but-feels-like-always" trap

_DIFFICULTY_CEILING = 3000
_DIFFICULTY_FLOOR = 500
_FREQ_FLOOR = 0.55
_FREQ_SPAN = 0.45
_DIFFICULTY_SPAN = 2500


def is_question_worthy(
    spot: PloSpot,
    *,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_presence: float = MIN_PRESENCE,
) -> bool:
    """True if the spot passes the presence filter AND the frequency window.

    Both windows inclusive. Override the thresholds via keyword args -- a
    difficulty preset / custom slider passes through to
    ``min_frequency`` / ``max_frequency``.
    """
    if spot.presence < min_presence:
        return False
    return min_frequency <= spot.dominant_frequency <= max_frequency


def effective_max_frequency(
    slider_max: float, *, exclude_ambiguous_band: bool
) -> float:
    """The worthiness ceiling to use, given the ambiguous-band toggle.

    When ``exclude_ambiguous_band`` is True, caps the ceiling at
    :data:`AMBIGUOUS_BAND_FLOOR` (0.90) so the 90-95% band is filtered out
    before any LLM spend. A no-op when the ceiling is already <= 0.90.
    """
    if exclude_ambiguous_band:
        return min(slider_max, AMBIGUOUS_BAND_FLOOR)
    return slider_max


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
) -> PloQuestionEvaluation:
    """Run the worthiness gate + the cheap difficulty estimate."""
    return PloQuestionEvaluation(
        is_worthy=is_question_worthy(
            spot,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_presence=min_presence,
        ),
        top_action_frequency=spot.dominant_frequency,
        total_presence=spot.presence,
        difficulty_estimate=difficulty_estimate(spot),
    )


__all__ = [
    "AMBIGUOUS_BAND_FLOOR",
    "MAX_TOP_FREQUENCY",
    "MIN_PRESENCE",
    "MIN_TOP_FREQUENCY",
    "PloQuestionEvaluation",
    "difficulty_estimate",
    "effective_max_frequency",
    "evaluate_spot",
    "is_question_worthy",
]
