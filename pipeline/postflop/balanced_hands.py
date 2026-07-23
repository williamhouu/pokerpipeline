"""Fully-balanced FULL-HAND selection (July 2026) -- the Hold'em analog of
PLO's 🎛️ Fully balanced batch, adapted to play-through hands.

Uses the shared greedy marginal-balance leaf (:mod:`pipeline.balanced_select`
-- NOT ``pipeline.plo``; the postflop package is self-contained). Differences
from the PLO preflop mode, on purpose:

* **Ending street is NOT a balance axis.** The length profile
  (river_heavy production default) owns the ending-street mix via quotas --
  a product choice balance must not fight. This module only PRE-ORDERS the
  candidate pool; :func:`~pipeline.postflop.play_through.balanced_length_mix`
  keeps input order within each street bucket, so the quota picks inherit
  the balanced ordering.
* **Hand difficulty is balanced by the batch driver, not here.** A hand's
  difficulty needs per-leg equity-priced facts (expensive), so the driver
  runs a BOUNDED swap-refinement after the street mix instead of scoring
  the whole pool. This module's axes are all computable with pure range
  math -- cheap enough for a full-pool pre-order.

The cheap per-hand axes (weights in :data:`POSTFLOP_HAND_AXES`):

* **final answer verb** (fold / call-check / raise) -- the hand's LAST
  decision's dominant action; a preflop fold-ender is ``fold``, a
  raise-ender ``raise``. Read from the dominant action, never the rendered
  option text, so basic and GTO option styles balance identically (the
  user's rule).
* **decision type** of the deepest leg (C-bet / Facing a bet / ... via
  :func:`~pipeline.postflop.spot_selection.spot_decision_type`; preflop
  enders bucket as ``Preflop only``).
* **hero hand strength** at the deepest leg
  (:func:`~pipeline.postflop.spot_selection.spot_strength_bucket`).
* **hero seat**.

Pure and deterministic, like the shared leaf.
"""

from __future__ import annotations

from typing import Any, Sequence

from pipeline.balanced_select import (
    answer_verb,
    balance_report,
    balanced_order,
    format_balance_report,
)
from pipeline.postflop.facts import preflop_aggressor
from pipeline.postflop.spot_selection import (
    spot_decision_type,
    spot_strength_bucket,
)

# (attr key, plain-English label, weight). Order = display order.
POSTFLOP_HAND_AXES: tuple[tuple[str, str, float], ...] = (
    ("answer_verb", "Final answer", 1.00),
    ("decision_type", "Situation", 0.80),
    ("strength", "Hand strength", 0.60),
    ("hero", "Hero seat", 0.50),
)

#: The difficulty axis is reported (and swap-refined by the batch driver)
#: separately -- see the module docstring.
HAND_DIFFICULTY_AXIS: tuple[str, str, float] = (
    "difficulty_band", "Hand difficulty", 1.00,
)

# Hand-difficulty band edges -- same bands the admin full-hand mode radio
# uses (Easy 400-1300 / Medium 1300-2100 / Hard 2100+).
def hand_difficulty_band(score: float) -> str:
    if score < 1300:  # noqa: PLR2004
        return "Easy"
    if score < 2100:  # noqa: PLR2004
        return "Medium"
    return "Hard"


def hand_balance_attrs(hand: Any, solve: Any) -> dict[str, str]:
    """The cheap balance-axis values for one play-through hand."""
    deepest = hand.legs[-1]
    if getattr(deepest, "spot", None) is not None:
        spot = deepest.spot
        verb = answer_verb(spot.dominant_action)
        strength = spot_strength_bucket(spot)
        decision = spot_decision_type(
            spot,
            aggressor=preflop_aggressor(solve),
            ip_position=solve.ip_position,
        )
    else:
        # Preflop-only ender (terminal fold / terminal re-raise) or the
        # entry leg: no postflop spot to bucket.
        if getattr(deepest, "terminal_raise", False):
            verb = "raise"
        elif getattr(deepest, "terminal_fold", False):
            verb = "fold"
        else:
            verb = "call/check"
        strength = "Preflop only"
        decision = "Preflop only"
    return {
        "answer_verb": verb,
        "decision_type": decision,
        "strength": strength,
        "hero": str(hand.hero),
    }


def order_pool_for_balance(hands: Sequence[Any], solve: Any) -> list[Any]:
    """Reorder the whole candidate pool by greedy marginal balance.

    The street-quota mixer keeps input order within each ending-street
    bucket, so feeding it this ordering balances the quota picks on every
    cheap axis; the post-quota remainder (the reserve) continues the same
    ordering, so atomicity backfill stays balanced too.
    """
    if not hands:
        return []
    attrs = [hand_balance_attrs(h, solve) for h in hands]
    order = balanced_order(
        attrs,
        POSTFLOP_HAND_AXES,
        spread_keys=[str(h.legs[-1].node_id) for h in hands],
    )
    return [hands[i] for i in order]


def full_hand_balance_report(
    selected_hands: Sequence[Any],
    pool_hands: Sequence[Any],
    solve: Any,
    *,
    selected_difficulties: dict[str, int],
    scored_difficulties: dict[str, int],
) -> dict:
    """The meta ``balance_report`` for a fully-balanced full-hand batch.

    Cheap axes are reported against the FULL candidate pool; the difficulty
    axis against the SCORED subset only (scoring needs equity-priced facts,
    so the driver never scores the whole pool) -- each axis's ``pool``
    counts are honest for what was actually examined. The ending-street mix
    is quota-owned (length profile) and reported by the existing counters,
    so it is deliberately absent here.
    """
    sel_attrs = [hand_balance_attrs(h, solve) for h in selected_hands]
    pool_attrs = [hand_balance_attrs(h, solve) for h in pool_hands]
    report = balance_report(sel_attrs, pool_attrs, POSTFLOP_HAND_AXES)

    sel_bands = [
        {"difficulty_band": hand_difficulty_band(selected_difficulties[h.hand_id])}
        for h in selected_hands
        if h.hand_id in selected_difficulties
    ]
    scored_bands = [
        {"difficulty_band": hand_difficulty_band(s)}
        for s in scored_difficulties.values()
    ]
    diff_report = balance_report(
        sel_bands, scored_bands, (HAND_DIFFICULTY_AXIS,)
    )
    # Difficulty leads the display, matching its weight in the mode's intent.
    report["axes"] = diff_report["axes"] + report["axes"]
    return report


__all__ = [
    "HAND_DIFFICULTY_AXIS",
    "POSTFLOP_HAND_AXES",
    "format_balance_report",
    "full_hand_balance_report",
    "hand_balance_attrs",
    "hand_difficulty_band",
    "order_pool_for_balance",
]
