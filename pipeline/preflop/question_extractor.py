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

Difficulty (May 2026 update): combines BOTH the dominant-action frequency
AND the EV gap between top-2 actions, when EV gap is available. The
EV gap is computed downstream in the batch loop (needs facts/equity),
then re-fed into :func:`difficulty_score` for a refined rating. When
EV gap isn't available (raise-involved spots in v1), the score falls
back to freq-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.preflop.spot_sampler import PreflopSpot

# Phase A thresholds -- starting values, expect tuning against review.
MIN_TOP_FREQUENCY = 0.55  # below: no clear best answer to teach
MAX_TOP_FREQUENCY = 0.95  # above: the answer is too obvious
MIN_PRESENCE = 0.01  # below: hand doesn't actually reach the node

# Difficulty-blend constants. The "easy" scalar is a weighted average of
# a frequency component and an EV-gap component, then linearly mapped to
# the 500-3000 score range. Tuned against the first review batch; expect
# further adjustment as more batches land.
_DIFFICULTY_CEILING = 3000  # hardest
_DIFFICULTY_FLOOR = 500  # easiest
_FREQ_WEIGHT = 0.6  # primary signal: how obvious is the correct answer
_EV_GAP_WEIGHT = 0.4  # secondary: how crisp is the right-vs-wrong distinction
# EV gap above this is "fully easy" on the EV axis. 3bb covers the
# realistic preflop call/fold EV-gap range (a 2.5bb-open BB-defend spot
# with terrible equity tops out around ~2bb; deeper trees can hit 3bb).
_EV_GAP_FULL_CREDIT_BB = 3.0


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


def difficulty_score(
    spot: PreflopSpot,
    ev_gap_bb: float | None = None,
) -> int:
    """The spot's difficulty on the brief's 500-3000 scale.

    Two signals when both are available:

      * **Frequency component**  -- ``(freq - 0.55) / 0.45``, clipped to
        [0, 1]. 0 at the 55% worthiness floor (most ambiguous), 1 at
        100% pure (most obvious).
      * **EV-gap component**     -- ``min(ev_gap_bb / 3.0, 1)``,
        clipped. 0 at 0 bb (close decision), 1 at 3 bb (clearly costly
        mistake). 3 bb is the "fully easy" ceiling; above that adds
        no further easiness credit.

    Blended ``easy = 0.6 * freq + 0.4 * ev_gap`` (freq weighted higher
    because it's the more direct signal of "obvious correct answer";
    EV gap is the modifier for "how costly is the wrong call").
    Mapped to the 500-3000 score range with ``3000 - easy * 2500``.

    When ``ev_gap_bb`` is ``None`` (e.g. raise-involved spots -- the v1
    EV engine only handles call/fold), falls back to the freq-only
    formula. Old behavior preserved.

    Examples (freq, ev_gap_bb -> score):

      * 55%, None       -> 3000  (hardest, no EV signal)
      * 100%, None      -> 500   (easiest, no EV signal)
      * 95%, None       -> 778   (very easy, no EV signal)
      * 66%, 1.37       -> 2175  (3♣3♦ vs 3-bet: meaningful both ways)
      * 81%, 1.38       -> 1672  (A♠8♠ vs 3-bet: high freq + modest gap)
      * 95%, 3.0        -> 666   (very easy + clearly costly mistake)
      * 55%, 3.0        -> 2000  (mixed strategy but big gap → still hard
                                  to identify the right answer)
    """
    frequency = top_action_frequency(spot)
    freq_easy = max(0.0, min(1.0, (frequency - 0.55) / 0.45))

    if ev_gap_bb is None:
        easy = freq_easy
    else:
        ev_easy = max(0.0, min(1.0, ev_gap_bb / _EV_GAP_FULL_CREDIT_BB))
        easy = _FREQ_WEIGHT * freq_easy + _EV_GAP_WEIGHT * ev_easy

    score = 3000 - easy * 2500
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
