"""Preflop decision-node enumeration.

A ``PreflopDecisionNode`` is one decision point in the preflop game tree:
the actor (whose turn it is), the history of actions before them, and the
set of actions they can take (each backed by a range file).

The enumerator walks one or more ``PreflopPack`` instances, parses every
range file via the per-pack filename grammar, and groups files by
``(pack_id, actor, history_before)``. Each group becomes one
``PreflopDecisionNode``.

Typical use::

    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import discover_packs

    packs = discover_packs("ranges")
    nodes = enumerate_nodes(packs)
    print(f"{len(nodes)} preflop decision nodes")

Errors during parsing (malformed filenames, unknown actions, etc.) are
logged as warnings and the offending file is skipped -- one bad file
doesn't abort the whole enumeration.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from pipeline.preflop.grammars import parse
from pipeline.preflop.grammars.types import (
    ParsedAction,
    ParsedRangeFile,
    PreflopActionType,
)
from pipeline.preflop.pack import PreflopPack

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreflopActionOption:
    """One action available at a decision node.

    Each option corresponds to one range file in the pack (``.txt`` for
    PioViewer packs, ``.rng`` for Monker-export packs). The
    ``range_file`` field points at the parsed metadata; load the
    actual 169-class weights via
    ``pipeline.preflop_ranges.parse_range_file(option.range_file.path)``.
    """

    action_type: PreflopActionType
    raise_size_pct: float | None
    range_file: ParsedRangeFile

    @property
    def label(self) -> str:
        """Short human-readable label, e.g. ``'Fold'`` or ``'Raise 60%'``."""
        if self.action_type is PreflopActionType.RAISE:
            return f"Raise {self.raise_size_pct:g}%"
        return self.action_type.value


@dataclass(frozen=True)
class PreflopDecisionNode:
    """One decision point in the preflop tree.

    Fields:
        pack_id: Source pack (so multi-pack enumerations stay traceable).
        actor: The position whose turn it is, e.g. ``'BTN'``.
        history_before: The actions that happened before this decision,
            in order. Empty tuple = the actor is first to act preflop.
        actions: All actions the actor can take here. At least one entry;
            usually 2-4. Order is sorted by the underlying filename for
            stability.
    """

    pack_id: str
    actor: str
    history_before: tuple[ParsedAction, ...]
    actions: tuple[PreflopActionOption, ...]

    @property
    def node_id(self) -> str:
        """A stable, human-readable id for the node.

        Built by concatenating the action history (the pre-actor part) and
        the actor's position, similar to the Ryan-pack filename style.
        Useful for logging / debugging / dedup. Not guaranteed unique
        across packs -- pair with ``pack_id`` for global uniqueness.
        """
        parts = []
        for a in self.history_before:
            verb = (
                f"{a.raise_size_pct:g}%"
                if a.action_type is PreflopActionType.RAISE
                else a.action_type.value
            )
            parts.append(f"{a.position}_{verb}")
        parts.append(f"{self.actor}_decision")
        return "_".join(parts)

    def has_action(self, action_type: PreflopActionType) -> bool:
        """True if this node has at least one option of the given type."""
        return any(opt.action_type is action_type for opt in self.actions)


def enumerate_nodes(
    packs: Iterable[PreflopPack],
) -> tuple[PreflopDecisionNode, ...]:
    """Walk the given packs, parse all range files, group into decision nodes.

    A node = a unique ``(pack_id, actor, history_before)`` triple. Files
    with that triple are collected as the node's ``actions``.

    Args:
        packs: One or more ``PreflopPack`` instances (from ``discover_packs``
            or hand-registered for tests).

    Returns:
        Tuple of all decision nodes found, ordered by ``(pack_id,
        actor, node_id)`` for determinism.

    Raises:
        Nothing -- individual file parse failures are logged at WARNING
        level and skipped so a single malformed file can't abort the run.
    """
    groups: dict[
        tuple[str, str, tuple[ParsedAction, ...]],
        list[ParsedRangeFile],
    ] = defaultdict(list)

    file_count = 0
    skip_count = 0
    for pack in packs:
        # Each pack declares its own file pattern (PioViewer packs are
        # .txt, Monker-export packs are .rng).
        for txt_path in sorted(pack.root_path.rglob(pack.file_glob)):
            file_count += 1
            try:
                parsed = parse(txt_path, pack=pack)
            except (ValueError, KeyError) as exc:
                skip_count += 1
                logger.warning(
                    "node_enumerator: skipping %s: %s",
                    txt_path.relative_to(pack.root_path),
                    exc,
                )
                continue
            history_before = parsed.action_history[:-1]
            key = (pack.pack_id, parsed.actor, history_before)
            groups[key].append(parsed)

    nodes: list[PreflopDecisionNode] = []
    for (pack_id, actor, history_before), files in groups.items():
        options = tuple(
            PreflopActionOption(
                action_type=f.actor_action,
                raise_size_pct=f.actor_raise_size_pct,
                range_file=f,
            )
            for f in sorted(files, key=lambda x: x.path.name)
        )
        nodes.append(
            PreflopDecisionNode(
                pack_id=pack_id,
                actor=actor,
                history_before=history_before,
                actions=options,
            )
        )

    logger.info(
        "node_enumerator: walked %d files (skipped %d), produced %d nodes",
        file_count,
        skip_count,
        len(nodes),
    )
    return tuple(sorted(nodes, key=lambda n: (n.pack_id, n.actor, n.node_id)))


def enumerate_nodes_by_actor(
    packs: Iterable[PreflopPack],
) -> dict[str, tuple[PreflopDecisionNode, ...]]:
    """Convenience: same as ``enumerate_nodes`` but pre-grouped by actor.

    Returns a dict mapping each actor position (``'UTG'``, ``'HJ'``, ...)
    to the tuple of nodes where that actor is making the decision.
    Handy for the admin panel's "filter by hero position" UI.
    """
    nodes = enumerate_nodes(packs)
    by_actor: dict[str, list[PreflopDecisionNode]] = defaultdict(list)
    for n in nodes:
        by_actor[n.actor].append(n)
    return {actor: tuple(ns) for actor, ns in by_actor.items()}
