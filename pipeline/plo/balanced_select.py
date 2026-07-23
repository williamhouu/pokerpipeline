"""PLO adapter for the shared fully-balanced batch selector (July 2026).

The greedy marginal-balance ALGORITHM lives in the shared leaf
:mod:`pipeline.balanced_select` (so the self-contained postflop package can
use it too); this module owns the PLO-specific schema:

* **difficulty band** (Easy / Medium / Hard)            weight 1.00
* **situation** (the PLO_ACTION_CONTEXTS bucket)        weight 0.90
* **correct-answer verb** (fold / call-check / raise)   weight 0.80
* **hero position bucket** (early/middle/late/sb/bb)    weight 0.50
* **hand shape family** (paired x suitedness)           weight 0.25

The answer verb comes from the solver's DOMINANT ACTION, never the rendered
option string -- so basic labels ("Call") and GTO labels ("Mostly Call")
balance identically, per the user's rule.

HONESTY RULE (no silent caps): when the pool simply lacks a value (e.g. only
two Hard spots exist under the current filters), the selector ships what
exists and :func:`balance_report` records achieved-vs-target per axis so the
shortfall is visible in the meta and the admin done-panel.

Pure and deterministic, like the shared leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pipeline.balanced_select import answer_verb  # noqa: F401 - re-export
from pipeline.balanced_select import balance_report as _core_report
from pipeline.balanced_select import balanced_order as _core_order
from pipeline.balanced_select import format_balance_report

# Score edges shared with the admin Generate page's presets (Easy 400-1300,
# Medium 1300-2100, Hard 2100-3200). Scores below/above clip into the end
# bands so every spot classifies.
DIFFICULTY_BANDS: tuple[tuple[str, int, int], ...] = (
    ("Easy", 400, 1300),
    ("Medium", 1300, 2100),
    ("Hard", 2100, 3200),
)

# (attr name, plain-English label, weight). Order = display order.
BALANCE_AXES: tuple[tuple[str, str, float], ...] = (
    ("difficulty_band", "Difficulty", 1.00),
    ("action_context", "Situation", 0.90),
    ("answer_verb", "Correct answer", 0.80),
    ("position", "Position", 0.50),
    ("hand_shape", "Hand shape", 0.25),
)


def difficulty_band(score: float) -> str:
    """``Easy`` / ``Medium`` / ``Hard`` for a difficulty score."""
    if score < DIFFICULTY_BANDS[1][1]:
        return "Easy"
    if score < DIFFICULTY_BANDS[2][1]:
        return "Medium"
    return "Hard"


def hand_shape_family(pair_pattern: str, suit_pattern: str) -> str:
    """Coarse shape family: paired-ness x suitedness (6 values).

    Deliberately coarser than the range-breakdown buckets -- this is the
    lowest-weight axis and finer buckets would be unfillable at batch sizes.
    """
    paired = "unpaired" if pair_pattern == "unpaired" else "paired"
    if suit_pattern == "double_suited":
        suited = "double-suited"
    elif suit_pattern in ("single_suited", "three_suited", "monotone"):
        suited = "suited"
    else:
        suited = "rainbow"
    return f"{paired} {suited}"


@dataclass(frozen=True)
class BalanceAttrs:
    """One candidate spot's value on each balance axis."""

    difficulty_band: str
    action_context: str
    answer_verb: str
    position: str
    hand_shape: str
    node_id: str = ""

    def value(self, axis: str) -> str:
        return str(getattr(self, axis))

    def as_dict(self) -> dict[str, str]:
        return {key: self.value(key) for key, _label, _w in BALANCE_AXES}


def balanced_order(attrs: Sequence[BalanceAttrs], count: int) -> list[int]:
    """Greedy balanced ordering of the WHOLE pool; consume front-to-back.

    Thin wrapper over :func:`pipeline.balanced_select.balanced_order` with
    the PLO axes and a node-reuse spread penalty. ``count`` only matters to
    callers slicing the result -- the ordering rule is uniform.
    """
    del count  # the ordering rule is uniform; callers slice
    return _core_order(
        [a.as_dict() for a in attrs],
        BALANCE_AXES,
        spread_keys=[a.node_id for a in attrs],
    )


def balance_report(
    selected: Sequence[BalanceAttrs],
    pool: Sequence[BalanceAttrs],
) -> dict:
    """Achieved-vs-target distribution per axis (see the shared leaf)."""
    return _core_report(
        [a.as_dict() for a in selected],
        [a.as_dict() for a in pool],
        BALANCE_AXES,
    )


__all__ = [
    "BALANCE_AXES",
    "DIFFICULTY_BANDS",
    "BalanceAttrs",
    "answer_verb",
    "balance_report",
    "balanced_order",
    "difficulty_band",
    "format_balance_report",
    "hand_shape_family",
]
