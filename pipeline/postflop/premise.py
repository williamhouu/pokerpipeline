"""Premise-realism gate for postflop -- the explicit port of preflop's gate.

Preflop drops a question when the villain's line is taken ~0% of the time
(``min_villain_line_pct``) or hero's own earlier action is one a good player
almost never makes (``min_hero_premise_freq``): a question built on a ghost.
Postflop's node-reach quality gate (:func:`pipeline.postflop.quality.
node_quality_issue`) catches the WORST such lines via a combo COUNT, but not an
explicit per-action frequency. This module adds the explicit check: walk every
PRIOR action on the line to hero's decision and read its range-aggregate
frequency; the minimum is the "weakest link" in the premise. A spot below the
threshold is built on an action someone almost never takes.

How the per-action frequency is read, scale- and token-free: the frequency of
the action taken from a parent decision node to its child is the reach mass that
continued, ``sum(child.villain_range) / sum(parent.hero_range)``. At the child
the parent's actor is the *villain* (the other player is now to act), and its
reach weights are the parent actor's range filtered by the action taken; the
parent's ``hero_range`` is that actor's full range at the parent. The ratio is
therefore exactly the reach-weighted frequency of the action for a same-street
step, and a close approximation across a chance card (the dealt card board-masks
a few percent of combos) -- which only nudges high-frequency street-closing
calls/checks, never the rare aggressive lines (a donk lead, a check-raise, an
overbet) this gate exists to catch.

Pure + deterministic (reach weights are fixed solver output), so it is safe for
the byte-identical-CSV guarantee and runs BEFORE any equity sim or LLM spend. It
depends only on the node id + the solve's nodes -- the same combo for every hero
hand at a node, so it is a per-NODE gate (compute once, skip the whole node).
"""
from __future__ import annotations

from pipeline.postflop.solve import PostflopNode, PostflopSolve

# Default floor: every prior action on the line must be taken at least this often
# (range-aggregate). 0.5% is permissive -- it drops only clear ghost lines (a
# turn overbet-jam taken ~0.1%, a never-used bet size), leaving uncommon-but-real
# lines (a 2% donk lead, a check-raise). Raise it for stricter premise realism.
DEFAULT_MIN_PREMISE_FREQ = 0.005


def _line_decision_nodes(node: PostflopNode, solve: PostflopSolve) -> list[PostflopNode]:
    """The decision nodes on the path to ``node`` (inclusive), in line order.

    Each prefix of the node id that is a real decision node, in order. A prefix
    that is a chance-card transition or a street-closed state is absent from
    ``solve.nodes`` and so is skipped -- leaving exactly the player-decision
    nodes, parent-then-child."""
    tokens = node.node_id.split(":")
    chain: list[PostflopNode] = []
    for i in range(2, len(tokens) + 1):  # prefixes after the "r:0" root
        ancestor = solve.nodes.get(":".join(tokens[:i]))
        if ancestor is not None:
            chain.append(ancestor)
    return chain


def line_premise_min_freq(node: PostflopNode, solve: PostflopSolve) -> float | None:
    """Min range-aggregate frequency across every PRIOR action on the line.

    Walks the decision nodes from the root to ``node`` and, for each
    parent->child step, reads the reach-weighted frequency of the action taken
    (see the module docstring). Returns the minimum over the whole line (both
    players' prior actions), or ``None`` when there is no prior action -- a
    first-to-act flop node -- so the gate passes. The hero's OWN decision at
    ``node`` is never gated (it is the question; worthiness handles it)."""
    chain = _line_decision_nodes(node, solve)
    min_freq: float | None = None
    for parent, child in zip(chain, chain[1:], strict=False):
        denom = sum(parent.hero_range.values())
        if denom <= 0:
            continue
        freq = sum(child.villain_range.values()) / denom
        min_freq = freq if min_freq is None else min(min_freq, freq)
    return min_freq


# --- artifact all-ins (July 2026, team standing rule) -----------------------
# The deep-stack trees include "bet your whole stack" branches because the
# TREE offers them, not because a 10x-pot jam is a realistic line. A question
# built on such a line trains against play nobody sees. An all-in wager is an
# ARTIFACT when it is both huge in absolute terms AND a large multiple of the
# pot it goes into; realistic short-stack jams (a 20bb stack shoving into a
# 15bb pot) pass both tests naturally, so no per-solve switch is needed here.
ARTIFACT_ALLIN_MIN_BB = 40.0
ARTIFACT_ALLIN_POT_MULT = 1.5


def line_contains_artifact_allin(node: PostflopNode, solve: PostflopSolve) -> bool:
    """True when any postflop action on this node's line is an artifact jam.

    Walks the node's history with the pot (same math as the animation walk):
    a step with ``all_in`` set is an artifact when its wager exceeds
    :data:`ARTIFACT_ALLIN_MIN_BB` and :data:`ARTIFACT_ALLIN_POT_MULT` times
    the pot it was fired into. Covers hero FACING the jam (it is the last
    history step) and jams earlier in the line alike."""
    pot = solve.starting_pot_bb
    invested: dict[str, float] = {}
    street = None
    for step in node.history:
        if step.street != street:
            street = step.street
            invested = {}
        if step.verb in ("bet", "raise") and step.to_bb is not None:
            wager = step.to_bb - invested.get(step.position, 0.0)
            pot_before = pot
            if step.all_in and wager > ARTIFACT_ALLIN_MIN_BB and (
                pot_before > 0 and wager > ARTIFACT_ALLIN_POT_MULT * pot_before
            ):
                return True
            pot += wager
            invested[step.position] = step.to_bb
        elif step.verb == "call":
            need = max(invested.values(), default=0.0) - invested.get(step.position, 0.0)
            pot += need
            invested[step.position] = invested.get(step.position, 0.0) + need
    return False


__all__ = [
    "ARTIFACT_ALLIN_MIN_BB",
    "ARTIFACT_ALLIN_POT_MULT",
    "DEFAULT_MIN_PREMISE_FREQ",
    "line_contains_artifact_allin",
    "line_premise_min_freq",
]
