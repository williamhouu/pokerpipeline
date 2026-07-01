"""Build the per-question ``ranges`` CSV column for postflop questions.

The postflop analogue of the preflop ``ranges`` column: it attaches each ACTIVE
player's range AT THE CURRENT STREET to the question, so the app's range UI can
render both players' holdings for a postflop / play-through spot -- exactly the
data the Review page already shows, surfaced into the output row.

It is computed ONLY from the decision node, so it is always accurate: both
players' ranges are the reach-weighted ranges that actually arrive at the node
(aggregated to the 169 hand classes the grid uses, board-blocked combos dropped
by the aggregator), and the hero's strategy is the reach-weighted action mix at
the node. The villain's current-street ACTION mix is intentionally not included
here -- at the decision node the villain is not the one to act, so only their
range (holdings) is well-defined; their earlier action this street is already in
the question's action history.

Accuracy boundary (the caller's responsibility): this is for POSTFLOP nodes,
where the solve carries accurate node ranges. The preflop-ENTRY leg of a
play-through does NOT get a ranges value -- a postflop solve has no accurate
preflop range (only the flat-call frequency, missing the 3-bet/fold split), so
:func:`pipeline.postflop.preflop_entry.build_preflop_entry_row` leaves the column
empty rather than show a half-true range.

Schema (one JSON object, sorted keys so the CSV stays byte-identical)::

    {
      "<POS>": {
        "acting": true|false,            # is this the player to act at the node
        "street": "flop"|"turn"|"river",
        "range":    {"<hand169>": <weight 0-1>},          # holdings here
        "strategy": {"<hand169>": {"<action label>": <freq 0-1>}}  # actor only
      },
      ...
    }
"""

from __future__ import annotations

import json
from typing import Any

# Pure leaf (combo-range + board -> mean weight per 169 class), reused exactly
# like the batch meta range snapshots + the preflop range charts.
from pipeline.preflop_ranges import aggregate_combo_range_to_classes

_WEIGHT_EPS = 0.004  # drop classes that are essentially not in the range


def _range_classes(combo_range, board: list[str]) -> dict[str, float]:
    return {
        h: round(w, 4)
        for h, w in aggregate_combo_range_to_classes(dict(combo_range), board).items()
        if w > _WEIGHT_EPS
    }


def _actor_strategy_classes(node, board: list[str]) -> dict[str, dict[str, float]]:
    """The actor's reach-weighted action mix per 169 class: ``{hand: {action:
    freq}}``. ``freq`` is ``reach(class) x P(action|class)`` -- a class's per-
    action segments sum to its presence (same convention as the Review grids)."""
    out: dict[str, dict[str, float]] = {}
    for action in {a.label for a in node.actions}:
        per_combo = {
            c: node.hero_range.get(c, 0.0) * node.strategy.get(c, {}).get(action, 0.0)
            for c in node.hero_range
        }
        for h, w in aggregate_combo_range_to_classes(per_combo, board).items():
            if w > _WEIGHT_EPS:
                out.setdefault(h, {})[action] = round(w, 4)
    return out


def _villain_decision_node(node, solve):
    """The most recent node where ``node.villain`` acted -- the longest node-id
    PREFIX of the current node whose actor is the villain. ``None`` when the
    villain has not acted on this line yet, or that node was down-sampled out.

    (Lives in this leaf, not batch.py, so both the CSV ``ranges`` builder and the
    batch's Review-panel strategy snapshots can resolve the villain's own
    decision without a circular import.)"""
    villain = node.villain
    nid = node.node_id
    best = None
    for cid, cand in solve.nodes.items():
        if cand.actor == villain and nid.startswith(cid + ":"):
            if best is None or len(cid) > len(best.node_id):
                best = cand
    return best


def build_active_ranges(node, solve=None) -> dict[str, Any]:
    """The ``ranges`` dict for one postflop decision node (see module docstring).

    When ``solve`` is given and the villain ALREADY acted on THIS street, the
    villain entry carries their range AND strategy at the node where THEY acted
    (their "current action point"), not just their holdings here -- so the column
    shows every player at their own action point. When the villain has not acted
    on this street yet (e.g. they checked back the previous street and it is now
    hero's turn first), only their holdings here are well-defined, so the entry
    stays range-only."""
    board = list(node.board)
    out: dict[str, Any] = {
        node.actor: {
            "acting": True,
            "street": node.street,
            "range": _range_classes(node.hero_range, board),
            "strategy": _actor_strategy_classes(node, board),
        },
        node.villain: {
            "acting": False,
            "street": node.street,
            "range": _range_classes(node.villain_range, board),
        },
    }
    if solve is not None:
        vnode = _villain_decision_node(node, solve)
        if vnode is not None and vnode.street == node.street:
            vboard = list(vnode.board)
            # vnode.actor == villain, so vnode.hero_range IS the villain's range
            # and _actor_strategy_classes(vnode) IS the villain's action mix.
            out[node.villain] = {
                "acting": False,
                "acted_this_street": True,
                "street": vnode.street,
                "range": _range_classes(vnode.hero_range, vboard),
                "strategy": _actor_strategy_classes(vnode, vboard),
            }
    return out


def build_active_ranges_json(node, solve=None) -> str:
    """``build_active_ranges`` as a compact, sorted (byte-stable) JSON string."""
    return json.dumps(
        build_active_ranges(node, solve), separators=(",", ":"), sort_keys=True
    )


__all__ = ["build_active_ranges", "build_active_ranges_json", "_villain_decision_node"]
