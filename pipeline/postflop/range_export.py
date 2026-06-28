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


def build_active_ranges(node) -> dict[str, Any]:
    """The ``ranges`` dict for one postflop decision node (see module docstring)."""
    board = list(node.board)
    return {
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


def build_active_ranges_json(node) -> str:
    """``build_active_ranges`` as a compact, sorted (byte-stable) JSON string."""
    return json.dumps(build_active_ranges(node), separators=(",", ":"), sort_keys=True)


__all__ = ["build_active_ranges", "build_active_ranges_json"]
