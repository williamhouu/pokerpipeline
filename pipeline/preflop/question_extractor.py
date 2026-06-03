"""Preflop question extractor (Layer 4, preflop edition).

Decides whether a ``PreflopSpot`` is worth turning into a training question,
and rates how hard it is on the brief's 500-3000 difficulty scale.

Mirrors ``pipeline.question_extractor`` for postflop, with two
preflop-specific differences:

  * **EV-gap filter is NOT applied at the worthiness gate.** Postflop's
    worthiness filter requires ``ev_gap_bb >= 0.5`` as a third gate
    alongside the frequency window. Preflop intentionally drops this
    gate from sampling because computing the EV gap needs equity (~1.5s
    per spot), which would 5-10x the cost of the cheap candidate filter
    that runs over thousands of spots. EV gap IS still consumed -- the
    batch loop computes it after sampling and feeds it into
    :func:`difficulty_score` to refine the displayed Difficulty Rating
    on every kept spot.

  * **Presence filter (new).** A hand that never reaches the decision
    node (zero or near-zero weight across all actions) shouldn't be a
    question even if technically the dict has a "dominant action" --
    that's just whichever zero-weight entry max() picked first. The
    ``min_presence`` filter excludes those.

Difficulty scoring uses :func:`difficulty_score`, which combines the
dominant-action frequency (weight 0.6) and the EV gap (weight 0.4,
capped at 3 bb of credit) when both are available. When ev_gap_bb is
None (raise-involved spots that the v1 EV engine doesn't model) the
score falls back to freq-only. The score is computed twice per spot:
once during worthy-spot collection with ev_gap_bb=None (cheap), then
again in the batch loop after facts compute (refined). The refined
score is what reaches the CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.preflop.spot_sampler import PreflopSpot

# Phase A thresholds -- starting values, expect tuning against review.
MIN_TOP_FREQUENCY = 0.55  # below: no clear best answer to teach
MAX_TOP_FREQUENCY = 0.95  # above: the answer is too obvious
MIN_PRESENCE = 0.01  # below: hand doesn't actually reach the node

# A dominant action at 90-95% reads as "mostly" but sits just under the
# 95% "always" label boundary (frequency_to_verb_prefix /
# options._PURE_STRATEGY_THRESHOLD). A player with the right read can pick
# the "always" option and still be marked wrong -- a labelling trap, not a
# strategy test. The Generate page excludes this band by default; see
# effective_max_frequency. Tune alongside MAX_TOP_FREQUENCY.
AMBIGUOUS_BAND_FLOOR = 0.90

# Difficulty-bounds + freq-axis constants. Kept here so the pre-facts
# freq-only ESTIMATE in :class:`PreflopQuestionEvaluation` doesn't
# depend on the full pipeline.preflop.difficulty module (which needs
# PreflopFacts -- not yet built at the worthiness-gate stage).
# The canonical CSV-bound score is computed downstream in batch.py
# via pipeline.preflop.difficulty.compute_difficulty().
_DIFFICULTY_CEILING = 3000
_DIFFICULTY_FLOOR = 500


def top_action_frequency(spot: PreflopSpot) -> float:
    """The frequency of the spot's dominant action, in [0.0, 1.0]."""
    return spot.dominant_frequency


def total_presence(spot: PreflopSpot) -> float:
    """How often hero reaches this decision with this hand, in [0.0, 1.0].

    Reads ``spot.presence`` directly (the sum of Pio's raw presence-
    weighted action weights before normalisation). Summing
    ``action_frequencies`` would not work here: those values are
    CONDITIONAL on reaching the node, so they sum to ~1.0 for any
    present hand and to 0 for absent ones -- losing the gradient
    between "barely reaches" and "always reaches".
    """
    return spot.presence


def difficulty_score(spot: PreflopSpot) -> int:
    """Cheap pre-facts difficulty ESTIMATE using only the freq axis.

    This is the value stored on :class:`PreflopQuestionEvaluation` for
    the worthiness-gate evaluation, which runs BEFORE facts are
    extracted (the canonical full-axis rating needs PreflopFacts).

    Formula: linear interpolation from 55% (hardest = 3000) to 100%
    (easiest = 500). Matches the freq axis used in the canonical
    algorithm (:func:`pipeline.preflop.difficulty.compute_difficulty`)
    so the estimate isn't wildly different.

    The canonical CSV-bound score, which blends freq + EV gap +
    archetype/concept + hand class, is computed downstream in
    :mod:`pipeline.preflop.batch` after facts are extracted. THIS
    function is just for the cheap pre-facts estimate.
    """
    frequency = top_action_frequency(spot)
    freq_easy = max(0.0, min(1.0, (frequency - 0.55) / 0.45))
    score = 3000 - freq_easy * 2500
    return round(max(_DIFFICULTY_FLOOR, min(_DIFFICULTY_CEILING, score)))


def is_question_worthy(
    spot: PreflopSpot,
    *,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_presence: float = MIN_PRESENCE,
) -> bool:
    """True if the spot passes the presence filter AND the frequency window.

    Both windows inclusive at both ends. Override the thresholds via
    keyword args -- the admin panel's difficulty preset / custom slider
    passes through to ``min_frequency`` / ``max_frequency``, e.g. "Hard"
    preset -> (0.55, 0.70); "Easy" preset -> (0.85, 0.95).
    """
    if total_presence(spot) < min_presence:
        return False
    frequency = top_action_frequency(spot)
    return min_frequency <= frequency <= max_frequency


def effective_max_frequency(
    slider_max: float, *, exclude_ambiguous_band: bool
) -> float:
    """The worthiness ceiling to actually use, given the ambiguous-band toggle.

    When ``exclude_ambiguous_band`` is True, caps the ceiling at
    :data:`AMBIGUOUS_BAND_FLOOR` (0.90) so the 90-95% "mostly-but-feels-
    like-always" band is filtered out before any LLM spend. A no-op when
    the caller's ceiling is already at or below 0.90.
    """
    if exclude_ambiguous_band:
        return min(slider_max, AMBIGUOUS_BAND_FLOOR)
    return slider_max


@dataclass(frozen=True)
class PreflopQuestionEvaluation:
    """The preflop question extractor's verdict on a spot.

    Layer 4 returns one of these per spot, downstream code uses
    ``is_worthy`` for the gate and ``difficulty_score`` for the
    ``Difficulty Rating`` CSV column.
    """

    is_worthy: bool
    top_action_frequency: float
    total_presence: float
    difficulty_score: int


def evaluate_spot(
    spot: PreflopSpot,
    *,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_presence: float = MIN_PRESENCE,
) -> PreflopQuestionEvaluation:
    """Run the filter + the difficulty rating, return the full verdict."""
    return PreflopQuestionEvaluation(
        is_worthy=is_question_worthy(
            spot,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_presence=min_presence,
        ),
        top_action_frequency=top_action_frequency(spot),
        total_presence=total_presence(spot),
        difficulty_score=difficulty_score(spot),
    )
