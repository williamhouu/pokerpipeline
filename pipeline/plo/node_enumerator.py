"""PLO preflop decision-node enumeration.

A :class:`PloDecisionNode` is one decision point in the PLO preflop game
tree: the actor (whose turn it is), the history of actions before them, and
the set of actions they can take -- each backed by one `.rng` file.

The PLO Monker pack stores **one file per action** at each decision node, the
same convention as the NLHE Ryan pack (one `.txt` per action). So a node is a
group of sibling `.rng` files that share an action history and differ only in
the actor's last action::

    40100.0.0.0.0.40100.0.rng        LJ opens, folds to BB, BB 3-bets, LJ folds
    40100.0.0.0.0.40100.1.rng        ... LJ calls the 3-bet
    40100.0.0.0.0.40100.40100.rng    ... LJ 4-bets

Those three files are one node: actor ``LJ``, history "open / fold around /
BB 3-bet", actions {Fold, Call, Raise 100%}. The open node has just two
siblings (``0.rng`` fold, ``40100.rng`` raise); the pack stores a fold-range
file as well, so fold + raise partition every hand.

This mirrors :mod:`pipeline.preflop.node_enumerator`. Enumeration reads only
filenames (via :func:`pipeline.plo.pack.parse_node_path`), never file
contents -- the per-hand strategy is loaded lazily by
:mod:`pipeline.plo.spot_sampler`.

Typical use::

    from pipeline.plo.pack import discover_plo_pack
    from pipeline.plo.node_enumerator import enumerate_plo_nodes

    pack = discover_plo_pack(Path("plo_ranges"))
    nodes = enumerate_plo_nodes(pack)
    print(f"{len(nodes)} PLO decision nodes")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pipeline.plo.pack import (
    PloAction,
    PloActionType,
    PloPack,
    node_actor,
    parse_node_path,
)

logger = logging.getLogger(__name__)


def _verb(action: PloAction) -> str:
    """Compact action verb for node ids, e.g. ``'raise100'``, ``'call'``."""
    if action.action is PloActionType.RAISE:
        return f"raise{action.raise_pct:g}"
    return action.action.value.replace("_", "")


def action_label(action: PloAction) -> str:
    """Stable, human-readable label for an action, e.g. ``'Raise 100%'``.

    Unique within a node (one file per distinct action), so it doubles as the
    key in a spot's ``action_frequencies`` / ``ev_by_action`` dicts. Raise
    sizes are pot-relative (``'Raise 100%'`` = a pot-sized raise). Shared by
    :attr:`PloActionOption.label` and the fact extractor (which labels the
    villain's action).
    """
    t = action.action
    if t is PloActionType.RAISE:
        return f"Raise {action.raise_pct:g}%"
    if t is PloActionType.MIN_RAISE:
        return "Min-raise"
    if t is PloActionType.ALL_IN:
        return "All-in"
    if t is PloActionType.CALL:
        return "Call"
    return "Fold"


@dataclass(frozen=True)
class PloActionOption:
    """One action available at a PLO decision node.

    Each option is one `.rng` file. The per-hand weights/EVs are loaded
    lazily by :func:`pipeline.plo.spot_sampler.sample_plo_spot` from
    :attr:`path`.
    """

    action: PloAction  # the actor's action (seat, type, raise %)
    path: Path  # the .rng file backing this action

    @property
    def stem(self) -> str:
        """The node-path stem (filename without extension)."""
        return self.path.stem

    @property
    def label(self) -> str:
        """Stable, human-readable key for this action at its node.

        Unique within a node (one file per distinct action), so it doubles as
        the key in a spot's ``action_frequencies`` / ``ev_by_action`` dicts.
        Raise sizes are pot-relative (``'Raise 100%'`` = a pot-sized raise),
        matching the NLHE convention.
        """
        return action_label(self.action)


@dataclass(frozen=True)
class PloDecisionNode:
    """One decision point in the PLO preflop tree.

    Fields:
        actor: The seat whose turn it is, e.g. ``'LJ'``.
        history_before: The actions before this decision, in order. Empty
            tuple = the actor is first to act (the open node).
        actions: All actions the actor can take here (>= 1; usually 2-4),
            ordered by backing filename for determinism.
        history_stem: The shared `.rng` filename prefix that reaches this node
            (``''`` for the open node). The natural ``solver_reference`` path
            back to the node; every option's stem is ``history_stem`` plus the
            action's own token.
    """

    actor: str
    history_before: tuple[PloAction, ...]
    actions: tuple[PloActionOption, ...]
    history_stem: str

    @property
    def node_id(self) -> str:
        """Stable, human-readable id, e.g. ``'LJ_raise100_HJ_decision'``.

        Built from the action history and the actor. The open node (no
        history) is ``'<actor>_decision'``. Useful for logging / dedup; pair
        with the pack label for global uniqueness.
        """
        parts = [f"{a.seat}_{_verb(a)}" for a in self.history_before]
        parts.append(f"{self.actor}_decision")
        return "_".join(parts)

    def has_action(self, action_type: PloActionType) -> bool:
        """True if this node offers at least one option of the given type."""
        return any(opt.action.action is action_type for opt in self.actions)


def _history_stem(stem: str) -> str:
    """The node-history prefix of a file stem (drop the actor's own token).

    ``'40100.0.0.0.0.40100.1'`` -> ``'40100.0.0.0.0.40100'``; a single-token
    open stem (``'0'``, ``'40100'``) -> ``''`` (the root, LJ to act).
    """
    return stem.rsplit(".", 1)[0] if "." in stem else ""


def enumerate_plo_nodes(pack: PloPack) -> tuple[PloDecisionNode, ...]:
    """Walk a pack's `.rng` files and group siblings into decision nodes.

    A node = a unique ``(actor, history_before)`` pair; the files with that
    pair become its ``actions``. Only filenames are parsed -- no file contents
    are read, so this is cheap even on the full 12k-file pack.

    Malformed filenames are logged at WARNING level and skipped (one bad file
    can't abort the run). Nodes are returned sorted by ``(actor, node_id)`` for
    determinism.
    """
    groups: dict[
        tuple[str, tuple[PloAction, ...]],
        list[tuple[PloAction, Path]],
    ] = defaultdict(list)

    file_count = 0
    skip_count = 0
    for rng_path in sorted(pack.root.glob("*.rng")):
        file_count += 1
        try:
            actions = parse_node_path(rng_path.stem)
        except ValueError as exc:
            skip_count += 1
            logger.warning("node_enumerator: skipping %s: %s", rng_path.name, exc)
            continue
        actor = node_actor(actions)
        history_before = actions[:-1]
        groups[(actor, history_before)].append((actions[-1], rng_path))

    nodes: list[PloDecisionNode] = []
    for (actor, history_before), items in groups.items():
        options = tuple(
            PloActionOption(action=action, path=path)
            for action, path in sorted(items, key=lambda ap: ap[1].name)
        )
        nodes.append(
            PloDecisionNode(
                actor=actor,
                history_before=history_before,
                actions=options,
                history_stem=_history_stem(options[0].stem),
            )
        )

    logger.info(
        "node_enumerator: walked %d files (skipped %d), produced %d nodes",
        file_count,
        skip_count,
        len(nodes),
    )
    return tuple(sorted(nodes, key=lambda n: (n.actor, n.node_id)))


def enumerate_plo_nodes_by_actor(
    pack: PloPack,
) -> dict[str, tuple[PloDecisionNode, ...]]:
    """Same as :func:`enumerate_plo_nodes`, pre-grouped by actor seat.

    Maps each seat (``'LJ'``, ``'HJ'``, ...) to its decision nodes -- handy
    for a "filter by hero position" UI.
    """
    by_actor: dict[str, list[PloDecisionNode]] = defaultdict(list)
    for node in enumerate_plo_nodes(pack):
        by_actor[node.actor].append(node)
    return {actor: tuple(ns) for actor, ns in by_actor.items()}
