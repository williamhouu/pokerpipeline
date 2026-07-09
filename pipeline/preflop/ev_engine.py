"""Preflop EV engine (Phase B, step 4).

Computes per-action EV in big blinds and the EV gap between the dominant
and second-most-frequent action. Pure Python -- the LLM never sees the
math, only the resulting ``ev_gap_bb`` number in its solver-data block
(same way it sees ``hero_equity_vs_villain`` today).

Per the brief, ``ev_gap_bb >= 0.5`` is the postflop worthiness filter --
spots where the right answer is barely better than the wrong answer
aren't useful training questions. For preflop, this engine adds the
same filter (call/fold spots only -- see scope below).

## Scope: what this engine computes precisely vs what it skips

**Precise (computed):**
- ``EV(Fold) = 0`` -- folding gives up nothing more than already
  committed; we measure relative to the status quo.
- ``EV(Call) = equity × (current_pot + call_cost) - call_cost`` --
  classic pot-odds vs equity math.

**Skipped (returns None):**
- ``EV(Raise X)`` -- the EV of a raise depends on villain's response
  distribution (how often villain folds, calls, or re-raises) which
  needs the full game tree to model. PioSolver's preflop EVs would
  give us this; until we have them, raise EVs aren't reliable.

This means ``compute_ev_gap_bb`` returns:
- A number when both dominant and 2nd-most-frequent actions are
  in {Fold, Call} (the classic "should you call?" preflop spot).
- ``None`` when either of the top-2 is a Raise / All-in (we'd be
  approximating).

## Worthiness-filter integration -- DEFERRED

The brief calls for an ``ev_gap_bb >= 0.5`` worthiness gate on top of
the frequency window. We don't apply that gate in v1 because computing
EV gap requires running the equity calculator on every candidate spot
before filtering -- doubling the per-spot cost of batch generation.
For now the ``ev_gap_bb`` value is exposed in the CSV column so
reviewers can sort / filter by it manually; a future iteration can
wire the filter into ``pipeline.preflop.question_extractor`` if the
review batches show small-gap spots cluttering the output.

## Future direction

Once we have PioSolver postflop solves backing each preflop scenario,
we can read per-action EVs from those solves directly (the way the
postflop side already does). Then ``EV(Raise X)`` becomes precise and
this engine becomes a thin reader rather than an estimator. Until
then: call/fold-only.
"""

from __future__ import annotations

import logging

from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.options import canonicalize_strategy
from pipeline.preflop.pack import PreflopPack

logger = logging.getLogger(__name__)


def _pot_and_call_cost_bb(
    facts: PreflopFacts,
    pack: PreflopPack,
) -> tuple[float, float]:
    """Return ``(pot_bb, hero_call_cost_bb)`` at hero's decision point.

    Thin wrapper over the shared pot accounting in
    :func:`pipeline.preflop.action_history.resolve_preflop_history`, so
    the EV math, the Question prose, and the POT column can't drift
    apart. Hero's call cost = the current bet minus hero's existing
    commit (0 if hero is already up to the max).
    """
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        resolve_preflop_history,
    )

    state = resolve_preflop_history(facts.spot.node.history_before, pack)
    return state.pot_bb, state.call_cost_bb(facts.spot.node.actor)


def ev_fold_bb() -> float:
    """EV of folding, in bb. Always 0 -- we measure relative to status quo.

    Sunk cost (chips already committed via blinds / prior raises) is
    ignored: it's the same for every action so it doesn't affect the
    GAP between actions, which is what the worthiness filter cares
    about.
    """
    return 0.0


def ev_call_bb(facts: PreflopFacts, pack: PreflopPack) -> float | None:
    """EV of calling, in bb. ``None`` if we lack the inputs.

    Formula::

        EV(Call) = equity × (pot + call_cost) - call_cost
                 = equity × pot - (1 - equity) × call_cost

    Where:
      * ``equity`` -- hero's equity vs villain's current range
        (precomputed in ``facts.hero_equity_vs_villain``).
      * ``pot`` -- chips already in the middle BEFORE hero's call.
      * ``call_cost`` -- chips hero adds to match the current bet.

    Returns ``None`` if no equity data is available (e.g. open spot
    where there's no villain to compute equity against).

    On a MULTI-WAY all-in (2+ players all-in) the correct equity is hero's
    share of the whole field -- ``hero_equity_vs_field`` -- not the heads-up
    number, which overstates it. (For packs with solver per-action EVs the
    batch uses those instead of this analytic EV; this keeps the analytic
    fallback right too.)
    """
    eq = _equity_for_call(facts)
    if eq is None:
        return None
    pot_bb, call_cost_bb = _pot_and_call_cost_bb(facts, pack)
    return eq * (pot_bb + call_cost_bb) - call_cost_bb


def _equity_for_call(facts: PreflopFacts) -> float | None:
    """Hero's equity to use in the call-EV: the multi-way FIELD equity when
    2+ players are all-in (hero must beat everyone), else the heads-up number.
    None when no equity was computed."""
    if facts.hero_equity_vs_field is not None and len(facts.showdown_opponents) >= 2:
        return facts.hero_equity_vs_field
    return facts.hero_equity_vs_villain


def compute_price_geometry(
    facts: PreflopFacts,
    pack: PreflopPack,
) -> tuple[float | None, float | None, float | None]:
    """``(pot_bb, call_bb, break_even)`` at hero's decision point, or Nones.

    ONE computation for all three so the break-even number and the amounts
    shown in the math panel's written-out equation can never disagree.
    All-None when hero faces no bet to call (no price exists).
    """
    pot_bb, call_cost_bb = _pot_and_call_cost_bb(facts, pack)
    total = pot_bb + call_cost_bb
    if call_cost_bb <= 0 or total <= 0:
        return None, None, None
    return pot_bb, call_cost_bb, call_cost_bb / total


def compute_break_even_equity(
    facts: PreflopFacts,
    pack: PreflopPack,
) -> float | None:
    """Pot-odds threshold to call, in [0, 1]: ``cost / (pot + cost)``.

    The minimum equity hero needs for a call to break even against the
    chips already in the middle. Layer 6 cites this instead of doing the
    pot-odds arithmetic itself.

    Returns ``None`` when hero faces no bet to call (call_cost is 0 -- e.g.
    hero is the aggressor or can check), since there's no price to meet.
    """
    return compute_price_geometry(facts, pack)[2]


def compute_ev_gap_bb(
    facts: PreflopFacts,
    pack: PreflopPack,
) -> float | None:
    """EV gap between the dominant and 2nd-most-frequent action, in bb.

    Returns ``None`` when:
      * the spot has < 2 meaningful actions to compare,
      * either of the top-2 actions involves a Raise / All-in (we
        don't model raise EVs in v1 -- see module docstring),
      * the call/fold EV computation needs equity data we don't have.

    A non-None return is the absolute gap (always >= 0) -- by definition
    the dominant action has the higher Pio frequency, so we treat it as
    the "best" action and the 2nd as the "wrong answer".
    """
    canonical = canonicalize_strategy(facts)
    if len(canonical) < 2:
        return None
    ranked = sorted(canonical.items(), key=lambda kv: -kv[1])
    dominant_label = ranked[0][0]
    secondary_label = ranked[1][0]

    # Only compute when both top-2 are Fold or Call. Raise EVs need
    # villain's response distribution -- skipped in v1.
    if dominant_label not in ("Fold", "Call") or secondary_label not in (
        "Fold",
        "Call",
    ):
        return None
    if dominant_label == secondary_label:
        # Shouldn't happen after canonicalisation, but defensive.
        return None

    fold_ev = ev_fold_bb()
    call_ev = ev_call_bb(facts, pack)
    if call_ev is None:
        return None

    # Map labels to EVs and compute the gap.
    evs = {"Fold": fold_ev, "Call": call_ev}
    gap = abs(evs[dominant_label] - evs[secondary_label])
    return gap


__all__ = [
    "compute_ev_gap_bb",
    "ev_call_bb",
    "ev_fold_bb",
]
