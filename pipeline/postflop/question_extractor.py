"""Layer 4 (postflop): decide whether a spot is worth a question.

A spot is worth a question when the solver's top action sits in a band that is
neither a coin-flip nor a forced/trivial decision -- the **frequency window**
is the worthiness gate:

    * dominant-action frequency in ``[min_frequency, max_frequency]``
      (default 65%-99%): below 65% it's a near-coin-flip with no clear answer;
      above 99% it's a pure/forced spot that teaches nothing.

The EV gap to the second-best action is an **optional** quality filter, OFF by
default (``min_ev_gap_bb=None``) -- mirroring the preflop pipeline. The brief
proposed a hard 0.5bb floor, but postflop GTO mixes heavily: a *worthy*
(mixed-frequency) spot is, by definition, close to EV-indifferent, so its EV
gap is ~0 by construction (genuine indifference, not solver noise). A hard
floor therefore throws out exactly the interesting decisions (c-bet/lead/size)
and leaves only high-variance facing-a-big-bet call/folds. So worthiness is
frequency-only by default; enabling ``min_ev_gap_bb`` is an opt-in way to keep
only the higher-consequence spots (the admin "advanced filter", same as
preflop). ``MIN_EV_GAP_BB`` below is the suggested value WHEN enabled.

The EV gap still feeds the difficulty rating unconditionally (see
``difficulty.py``); it is taken from
:func:`pipeline.postflop.spot_sampler.spot_ev_gap_bb` (per-combo when the solve
exposes per-hand EVs, else range-mean). A spot whose EV gap is unknown
(``None``) always passes -- we never drop a spot for missing data.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.postflop.spot_sampler import PostflopSpot, spot_ev_gap_bb

MIN_FREQUENCY = 0.65
MAX_FREQUENCY = 0.99
# Suggested EV-gap value WHEN the optional filter is enabled (the gate is OFF
# by default -- see the module docstring). Not a hard worthiness default.
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
    min_ev_gap_bb: float | None = None,
) -> PostflopQuestionEvaluation:
    """Apply the worthiness gate to ``spot``.

    Returns a :class:`PostflopQuestionEvaluation`. ``is_worthy`` is True when
    the dominant frequency is in the window AND (the optional ``min_ev_gap_bb``
    filter is off, OR the EV gap is unknown, OR it meets the minimum). The EV
    filter is OFF by default (``min_ev_gap_bb=None``) -- mirroring preflop; see
    the module docstring.
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

    if min_ev_gap_bb is not None and ev_gap is not None and ev_gap < min_ev_gap_bb:
        return PostflopQuestionEvaluation(
            is_worthy=False,
            reason=(
                f"EV gap {ev_gap:.2f}bb below optional filter {min_ev_gap_bb:.2f}bb"
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
