"""Shared greedy marginal-balance selector (July 2026) -- the engine behind
every "🎛️ Fully balanced batch" mode.

A SHARED LEAF (like ``artifact_strip`` / ``bb_display`` / ``trap_grading``):
pure, dependency-free, importable by every pipeline. The PLO adapter is
:mod:`pipeline.plo.balanced_select` (fixed five-axis schema); the postflop
full-hand adapter is :mod:`pipeline.postflop.balanced_hands`. The postflop
package may not import ``pipeline.plo`` (self-containment rule), which is
exactly why the algorithm lives here.

HOW IT BALANCES: greedy *marginal* balance, not a filled cross-product. At
each step the selector picks the candidate that most reduces the current
imbalance summed over the weighted axes. Each individual axis lands close
to uniform without needing every cross-product cell populated.

HONESTY RULE (no silent caps): when the pool simply lacks a value, the
selector ships what exists and :func:`balance_report` records
achieved-vs-target per axis so the shortfall is visible downstream.

Pure and deterministic: no I/O, no randomness (ties break on pool order,
which is the caller's seeded shuffle), so batches stay byte-exactly
rebuildable and everything here is browserless-testable.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Mild penalty per prior pick sharing a candidate's ``spread_key`` (e.g. the
# solver node id), so a balanced batch also spreads across distinct spots.
_SPREAD_REUSE_PENALTY = 0.15

#: Axis schema: (attr key, plain-English label, weight).
Axis = tuple[str, str, float]


def answer_verb(dominant_action: str) -> str:
    """``fold`` / ``call/check`` / ``raise`` from the solver's dominant action.

    Deliberately reads the RAW dominant action label, not the rendered
    option text, so the same spot classifies identically under basic
    ("Call") and GTO ("Mostly Call") option styles -- the user's rule.
    Every aggressive label (Bet 33%, Raise 72%, 3-bet, All-in, ...) is the
    raise family. Game-agnostic: shared by the PLO and postflop adapters.
    """
    label = dominant_action.strip().lower()
    if label.startswith("fold"):
        return "fold"
    if label.startswith(("call", "check")):
        return "call/check"
    return "raise"


def balanced_order(
    attrs: Sequence[Mapping[str, str]],
    axes: Sequence[Axis],
    *,
    spread_keys: Sequence[str] | None = None,
) -> list[int]:
    """Greedy balanced ordering of the WHOLE pool; consume front-to-back.

    Returns every pool index exactly once. The first ``count`` picks (the
    caller's slice) are the intended batch; the remainder continues the same
    greedy rule so that backfill after a failure stays balanced too.

    ``attrs[i]`` maps each axis key to candidate *i*'s value on that axis.
    ``spread_keys`` (optional, same length) adds a mild penalty for reusing
    the same key (e.g. one solver node contributing many picks). Ties break
    on pool order, so the ordering is fully deterministic.
    """
    n = len(attrs)
    if n == 0:
        return []
    # Uniform target share over the values PRESENT in the pool -- you cannot
    # ask for a value that does not exist under the current filters.
    target_share = {
        key: 1.0 / len({str(a.get(key, "")) for a in attrs})
        for key, _label, _w in axes
    }

    picked: list[int] = []
    remaining = list(range(n))
    axis_counts: dict[str, dict[str, int]] = {key: {} for key, _l, _w in axes}
    spread_counts: dict[str, int] = {}

    while remaining:
        k = len(picked)
        best_i = None
        best_score = float("-inf")
        for i in remaining:
            a = attrs[i]
            score = 0.0
            for key, _label, weight in axes:
                val = str(a.get(key, ""))
                share = axis_counts[key].get(val, 0) / k if k else 0.0
                score += weight * (target_share[key] - share)
            if spread_keys is not None and spread_keys[i]:
                score -= _SPREAD_REUSE_PENALTY * spread_counts.get(
                    spread_keys[i], 0
                )
            if score > best_score:  # strict > keeps first-in-pool on ties
                best_score = score
                best_i = i
        assert best_i is not None
        picked.append(best_i)
        remaining.remove(best_i)
        a = attrs[best_i]
        for key, _label, _w in axes:
            val = str(a.get(key, ""))
            axis_counts[key][val] = axis_counts[key].get(val, 0) + 1
        if spread_keys is not None and spread_keys[best_i]:
            spread_counts[spread_keys[best_i]] = (
                spread_counts.get(spread_keys[best_i], 0) + 1
            )
    return picked


def balance_report(
    selected: Sequence[Mapping[str, str]],
    pool: Sequence[Mapping[str, str]],
    axes: Sequence[Axis],
) -> dict:
    """Achieved-vs-target distribution per axis, for the meta + admin panel.

    ``selected`` is what actually shipped; ``pool`` is every scored
    candidate, so a shortfall reads as "the pool only had N". Shape::

        {"axes": [{"axis", "label", "values": [
            {"value", "achieved", "target", "pool"}]}],
         "selected": len, "pool": len}
    """
    out = []
    for key, label, _w in axes:
        vals = sorted(
            {str(a.get(key, "")) for a in pool}
            | {str(a.get(key, "")) for a in selected}
        )
        target = round(len(selected) / len(vals), 1) if vals else 0.0
        out.append(
            {
                "axis": key,
                "label": label,
                "values": [
                    {
                        "value": v,
                        "achieved": sum(
                            1 for a in selected if str(a.get(key, "")) == v
                        ),
                        "target": target,
                        "pool": sum(
                            1 for a in pool if str(a.get(key, "")) == v
                        ),
                    }
                    for v in vals
                ],
            }
        )
    return {"axes": out, "selected": len(selected), "pool": len(pool)}


def format_balance_report(report: dict) -> list[str]:
    """Plain-English one-liners for the admin done-panel, one per axis.

    Example: ``Difficulty — Easy 8 · Medium 8 · Hard 8 (target ~8 each)``.
    A value the pool ran short on gets an honest suffix: ``Hard 3 (pool
    only had 3)``.
    """
    lines: list[str] = []
    for ax in report.get("axes", []):
        parts = []
        target = None
        for v in ax.get("values", []):
            target = v["target"]
            short = v["achieved"] < v["target"] and v["achieved"] >= v["pool"]
            parts.append(
                f"{v['value']} {v['achieved']}"
                + (f" (pool only had {v['pool']})" if short else "")
            )
        tail = f" (target ~{target} each)" if target is not None else ""
        lines.append(f"{ax['label']} — " + " · ".join(parts) + tail)
    return lines


__all__ = [
    "Axis",
    "balance_report",
    "balanced_order",
    "format_balance_report",
]
