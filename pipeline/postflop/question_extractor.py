"""Layer 4 (postflop): decide whether a spot is worth a question.

Same worthiness contract the brief specifies for the whole project: keep a
spot only when the solver's top action sits in a band that is neither a
coin-flip nor a forced/trivial decision, AND the EV gap to the best wrong
answer is large enough that there is a genuinely "correct" answer to teach.

    * dominant-action frequency in ``[min_frequency, max_frequency]``
      (default 55%-95%): below 55% it's a near-coin-flip with no clear answer;
      above 95% it's a pure/forced spot that teaches nothing.
    * EV gap to the second-best action >= ``min_ev_gap_bb`` (default 0.5bb):
      ensures the wrong answer actually costs something.

These same signals also feed the difficulty rating (see ``difficulty.py``).
The EV gap is taken from :func:`pipeline.postflop.spot_sampler.spot_ev_gap_bb`
(per-combo when the solve exposes per-hand EVs, else range-mean); a spot whose
EV gap is unknown (``None``) passes the EV check -- we don't drop a spot for
missing data, we just can't use that axis (mirrors the preflop pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.postflop.spot_sampler import PostflopSpot, spot_ev_gap_bb

MIN_FREQUENCY = 0.55
MAX_FREQUENCY = 0.95
MIN_EV_GAP_BB = 0.5


@dataclass(frozen=True)
class PostflopQuestionEvaluation:
    """Outcome of the worthiness gate for one spot."""

    is_worthy: bool
    reason: str  # human-readable: why it was kept or dropped
    dominant_frequency: float
    ev_gap_bb: float | None


def evaluate_spot(
    spot: PostflopSpot,
    *,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    min_ev_gap_bb: float = MIN_EV_GAP_BB,
) -> PostflopQuestionEvaluation:
    """Apply the worthiness gate to ``spot``.

    Returns a :class:`PostflopQuestionEvaluation`. ``is_worthy`` is True only
    when the dominant frequency is in the window AND (the EV gap is unknown OR
    meets the minimum).
    """
    freq = spot.dominant_frequency
    ev_gap = spot_ev_gap_bb(spot)

    if not (min_frequency <= freq <= max_frequency):
        return PostflopQuestionEvaluation(
            is_worthy=False,
            reason=(
                f"dominant freq {freq:.0%} outside worthiness window "
                f"[{min_frequency:.0%}, {max_frequency:.0%}]"
            ),
            dominant_frequency=freq,
            ev_gap_bb=ev_gap,
        )

    if ev_gap is not None and ev_gap < min_ev_gap_bb:
        return PostflopQuestionEvaluation(
            is_worthy=False,
            reason=(
                f"EV gap {ev_gap:.2f}bb below minimum {min_ev_gap_bb:.2f}bb "
                "(no clearly-correct answer)"
            ),
            dominant_frequency=freq,
            ev_gap_bb=ev_gap,
        )

    return PostflopQuestionEvaluation(
        is_worthy=True,
        reason=(
            f"dominant {spot.dominant_action} at {freq:.0%}"
            + (f", EV gap {ev_gap:.2f}bb" if ev_gap is not None else ", EV gap n/a")
        ),
        dominant_frequency=freq,
        ev_gap_bb=ev_gap,
    )


__all__ = [
    "MAX_FREQUENCY",
    "MIN_EV_GAP_BB",
    "MIN_FREQUENCY",
    "PostflopQuestionEvaluation",
    "evaluate_spot",
]
