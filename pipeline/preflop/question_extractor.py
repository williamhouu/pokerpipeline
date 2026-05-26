"""Preflop question extractor (Layer 4, preflop edition).

Decides whether a ``PreflopSpot`` is worth turning into a training question,
and rates how hard it is on the brief's 500-3000 difficulty scale.

Mirrors ``pipeline.question_extractor`` for postflop, with two differences:

  * **No EV-gap filter (Phase A).** Pio's per-action EV values are baked
    into a postflop ``.cfr``; preflop only carries frequencies in the
    range files. Pulling equivalent EV estimates needs an equity-driven
    EV engine -- worth ~1 day of work, deferred to Phase B. The
    frequency-window filter alone is enough to drop "trivial" and
    "coin-flip" spots; review will tell us whether we miss the EV gap.

  * **Presence filter (new).** A hand that never reaches the decision
    node (zero or near-zero weight across all actions) shouldn't be a
    question even if technically the dict has a "dominant action" --
    that's just whichever zero-weight entry max() picked first. The
    ``min_presence`` filter excludes those.

Difficulty uses the brief's MVP formula: harder spots = top action's
frequency closer to the 55% floor; easier spots = closer to 95% ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.preflop.spot_sampler import PreflopSpot

# Phase A thresholds -- starting values, expect tuning against review.
MIN_TOP_FREQUENCY = 0.55  # below: no clear best answer to teach
MAX_TOP_FREQUENCY = 0.95  # above: the answer is too obvious
MIN_PRESENCE = 0.01  # below: hand doesn't actually reach the node

_DIFFICULTY_CEILING = 3000  # hardest
_DIFFICULTY_FLOOR = 500  # easiest


def top_action_frequency(spot: PreflopSpot) -> float:
    """The frequency of the spot's dominant action, in [0.0, 1.0]."""
    return spot.dominant_frequency


def total_presence(spot: PreflopSpot) -> float:
    """Sum of all action frequencies -- a proxy for "does hero ever reach
    this decision with this hand?". Zero or near-zero = hand folded
    earlier in the tree and the spot is bogus."""
    return sum(spot.action_frequencies.values())


def difficulty_score(spot: PreflopSpot) -> int:
    """The spot's difficulty on the brief's 500-3000 scale.

    Brief MVP formula: 3000 - ((freq - 0.55) / 0.40) * 2500, where ``freq``
    is the top action's frequency. A barely-dominant 55% spot scores 3000
    (hardest); a near-pure 95% spot scores 500 (easiest). Clamped to the
    500-3000 range.
    """
    frequency = top_action_frequency(spot)
    score = 3000 - ((frequency - 0.55) / 0.40) * 2500
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
