"""PLO adapter for the shared fully-balanced batch selector (July 2026).

The greedy marginal-balance ALGORITHM lives in the shared leaf
:mod:`pipeline.balanced_select` (so the self-contained postflop package can
use it too); this module owns the PLO-specific schema:

* **difficulty band** (Easy / Medium / Hard)            weight 1.00
* **situation** (the PLO_ACTION_CONTEXTS bucket)        weight 0.90
* **correct-answer verb** (fold / call-check / raise)   weight 0.80
* **Always/Mostly qualifier** (GTO-capable styles ONLY) weight 0.70
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

# Always/Mostly qualifier axis (Aug 2026, user ask: a 44/56 Always/Mostly
# split with strong per-verb skews let players meta-game the prefix). Only
# ACTIVE when the batch's answer style can render GTO-style options
# (qualifier_axis_active in pipeline.plo.options); with basic labels the
# axis is omitted entirely so selection is byte-identical to before.
# INVARIANT (the user's July-21 rule, extended): the qualifier value is
# derived from the SOLVER's dominant-action frequency via
# pipeline.plo.options.answer_qualifier (the exact mapping
# build_options_gto renders), NEVER by parsing rendered option text.
QUALIFIER_AXIS: tuple[str, str, float] = ("qualifier", "Always/Mostly", 0.70)


def balance_axes(
    include_qualifier: bool = False,
) -> tuple[tuple[str, str, float], ...]:
    """The active axis schema: :data:`BALANCE_AXES`, plus the qualifier
    axis slotted between the answer-verb and position axes (its weight
    order) when the batch's answer style can render qualifiers."""
    if not include_qualifier:
        return BALANCE_AXES
    axes = list(BALANCE_AXES)
    axes.insert(3, QUALIFIER_AXIS)  # after answer_verb (0.80), before position
    return tuple(axes)


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
    """One candidate spot's value on each balance axis.

    ``qualifier`` is always carried (it is cheap to compute) but only
    BALANCES when the caller passes ``include_qualifier=True`` -- the axis
    list decides, so basic-style batches stay byte-identical.
    """

    difficulty_band: str
    action_context: str
    answer_verb: str
    position: str
    hand_shape: str
    node_id: str = ""
    qualifier: str = ""

    def value(self, axis: str) -> str:
        return str(getattr(self, axis))

    def as_dict(self) -> dict[str, str]:
        out = {key: self.value(key) for key, _label, _w in BALANCE_AXES}
        out["qualifier"] = self.qualifier
        return out


def balanced_order(
    attrs: Sequence[BalanceAttrs],
    count: int,
    *,
    include_qualifier: bool = False,
) -> list[int]:
    """Greedy balanced ordering of the WHOLE pool; consume front-to-back.

    Thin wrapper over :func:`pipeline.balanced_select.balanced_order` with
    the PLO axes and a node-reuse spread penalty. ``count`` only matters to
    callers slicing the result -- the ordering rule is uniform.
    ``include_qualifier`` adds the Always/Mostly axis (GTO-capable answer
    styles only; see :data:`QUALIFIER_AXIS`).
    """
    del count  # the ordering rule is uniform; callers slice
    return _core_order(
        [a.as_dict() for a in attrs],
        balance_axes(include_qualifier),
        spread_keys=[a.node_id for a in attrs],
    )


def balance_report(
    selected: Sequence[BalanceAttrs],
    pool: Sequence[BalanceAttrs],
    *,
    include_qualifier: bool = False,
) -> dict:
    """Achieved-vs-target distribution per axis (see the shared leaf)."""
    return _core_report(
        [a.as_dict() for a in selected],
        [a.as_dict() for a in pool],
        balance_axes(include_qualifier),
    )


__all__ = [
    "BALANCE_AXES",
    "DIFFICULTY_BANDS",
    "QUALIFIER_AXIS",
    "BalanceAttrs",
    "answer_verb",
    "balance_axes",
    "balance_report",
    "balanced_order",
    "difficulty_band",
    "format_balance_report",
    "hand_shape_family",
]
