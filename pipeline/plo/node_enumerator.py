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
from functools import cached_property, lru_cache
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
    # Table size of the pack this node came from (stamped by the enumerator).
    # Rides on the node so every downstream consumer -- seat display names,
    # position phrases, the app table tokens -- knows the seat dialect without
    # threading the pack itself. Default 6 = the pre-registry behavior.
    table_size: int = 6
    # EV unit scale to SB-units (stamped by the enumerator from the pack
    # spec): 1.0 for the cash packs (files already in milli-sb), 2.0 for the
    # MTT bb-ante packs (files in milli-BB; x2 converts bb -> sb so every
    # downstream EV consumer stays unit-safe). See PloPackSpec.ev_in_bb.
    ev_scale: float = 1.0
    # Effective stack of the pack this node came from (stamped by the
    # enumerator, like table_size/ev_scale) -- so stack-depth FACTS (the
    # short/standard/deep concept tags, their difficulty modifiers) never
    # need the pack threaded to them. The tags were hardcoded "always
    # standard" from the single-100bb-pack era, so every 10-25bb MTT /
    # short-stack question shipped a wrong standard_stack tag (July 2026).
    stack_bb: float = 100.0

    @cached_property
    def node_id(self) -> str:
        """Stable, human-readable id, e.g. ``'LJ_raise100_HJ_decision'``.

        Built from the action history and the actor. The open node (no
        history) is ``'<actor>_decision'``. Useful for logging / dedup; pair
        with the pack label for global uniqueness. Cached (admin pages key
        thousands of nodes by id per render; works on the frozen dataclass
        because cached_property writes via ``__dict__``).
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


@lru_cache(maxsize=4)
def enumerate_plo_nodes(pack: PloPack) -> tuple[PloDecisionNode, ...]:
    """Walk a pack's `.rng` files and group siblings into decision nodes.

    A node = a unique ``(actor, history_before)`` pair; the files with that
    pair become its ``actions``. Only filenames are parsed -- no file contents
    are read -- but the 9-max pack has 327k of them (~13s per walk), so the
    result is memoized per pack (July 2026): the admin page enumerates once
    and every batch generation in the same process starts instantly instead
    of re-walking. The value is an immutable tuple; ``PloPack`` is a frozen
    dataclass, so equal packs share the entry. Caveat (same as the admin's
    ``cache_resource``): re-extracting a pack IN PLACE needs a process
    restart to be seen.

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
            actions = parse_node_path(rng_path.stem, seats=pack.seats)
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
                table_size=pack.table_size,
                ev_scale=2.0 if pack.spec.ev_in_bb else 1.0,
                stack_bb=pack.spec.stack_bb,
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


# --- node categorization (for the Generate page's filters) ------------------
_AGGRESSIVE = frozenset(
    {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
)

#: The action-context buckets, in escalation order. Mirrors the NLHE
#: ``pipeline.preflop.batch.ACTION_CONTEXTS`` so the two Generate pages offer
#: the same "Action faced" choices.
PLO_ACTION_CONTEXTS: tuple[str, ...] = (
    "Opening",
    "Facing single raise",
    "Facing 3-bet",
    "Facing 4-bet+",
    # "After call(s)" was split (June 2026) to match the NLHE buckets: one live
    # flat-caller is a tame, well-solved spot (a squeeze opportunity / a single
    # overcall); two or more is the noisy multiway territory.
    "After one call",
    "After multiple calls",
)


def plo_node_action_context(node: PloDecisionNode) -> str:
    """Categorize a node by the action hero is facing.

    One of :data:`PLO_ACTION_CONTEXTS`. Port of
    :func:`pipeline.preflop.batch.node_action_context` (same raise-level
    precedence): a 3-bet / 4-bet pot is labeled by the raise hero faces even
    when someone flat-called earlier; the "After ... call(s)" buckets are
    reserved for a SINGLE open that went multiway via flats.

      * ``"Opening"``              -- no prior raise in the history
      * ``"Facing 4-bet+"``        -- three or more raises (callers ignored)
      * ``"Facing 3-bet"``         -- exactly two raises (callers ignored)
      * ``"After one call"``       -- a SINGLE open + at most one live flat-caller
      * ``"After multiple calls"`` -- a SINGLE open + two or more live flat-callers
      * ``"Facing single raise"``  -- a single open, no flat-callers
    """
    n_raises = sum(1 for a in node.history_before if a.action in _AGGRESSIVE)
    n_calls = sum(
        1 for a in node.history_before if a.action is PloActionType.CALL
    )
    if n_raises == 0:
        return "Opening"
    # Raise level wins over any earlier flat (mirrors NLHE): a 3-bet/4-bet pot
    # is labeled by the raise hero faces, not lumped into an after-call bucket.
    if n_raises >= 3:  # noqa: PLR2004
        return "Facing 4-bet+"
    if n_raises == 2:  # noqa: PLR2004
        return "Facing 3-bet"
    # Exactly one raise (a single open). Flat-callers make it a squeeze /
    # overcall spot; split by the number of LIVE flat-callers -- distinct
    # non-hero seats whose LAST action is a call (hero's own call and a caller
    # who later folded don't count; raw n_calls over-counts both).
    if n_calls > 0:
        last: dict[str, PloActionType] = {}
        for a in node.history_before:
            last[a.seat] = a.action
        live_callers = sum(
            1
            for seat, t in last.items()
            if t is PloActionType.CALL and seat != node.actor
        )
        return "After one call" if live_callers <= 1 else "After multiple calls"
    return "Facing single raise"


def plo_pending_after_hero(node: PloDecisionNode) -> tuple[tuple[str, ...], bool]:
    """``(still_to_act, closes_action)`` for a decision node.

    ``still_to_act`` = the seats that have NOT yet responded to the current
    bet and act after hero this round (in order); ``closes_action`` is True
    when a hero call/fold ends the preflop betting (nobody pending, and the
    aggressor can't raise their own bet). Computed by replaying the SAME
    seat-queue walk :func:`pipeline.plo.pack.parse_node_path` uses -- after
    the history, the queue front is hero, the seats behind hero up to the
    last aggressor are the pending ones (the aggressor and anyone behind them
    only re-act if someone raises).

    This is the PLO port of the NLHE multiway-awareness facts (the SOLVER
    DATA lines that stop the LLM inventing "UTG still to act behind you"
    about a seat that already folded -- the exact residual error in the first
    live 9-max batch).
    """
    from pipeline.plo.pack import SEATS, SEATS_9MAX  # noqa: PLC0415

    seats = SEATS_9MAX if node.table_size == 9 else SEATS  # noqa: PLR2004
    queue = list(seats)
    last_aggressor: str | None = None
    for a in node.history_before:
        seat = queue.pop(0)
        if seat != a.seat:  # malformed history; fail safe (no pending info)
            return (), False
        if a.action in (PloActionType.CALL, PloActionType.RAISE, PloActionType.MIN_RAISE):
            queue.append(seat)
        if a.action in (
            PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN
        ):
            last_aggressor = seat
    if not queue or queue[0] != node.actor:
        return (), False
    behind = queue[1:]
    if last_aggressor in behind:
        behind = behind[: behind.index(last_aggressor)]
    return tuple(behind), not behind


def plo_filter_meta(
    nodes: tuple[PloDecisionNode, ...],
) -> tuple[tuple[str, str, int], ...]:
    """Per-node ``(actor, action_context, pot_entrant_count)`` triples.

    The admin Generate page's live "N nodes match your filters" recount used
    to re-derive context + player count for all ~160k nodes on EVERY
    Streamlit rerun (i.e. every widget change, ~0.3s each). This one 0.5s
    pass -- built once alongside enumeration, in the background loader --
    turns that recount into a ~4ms scan of plain tuples. Same functions,
    same values: a filter over this meta MUST equal a filter over the nodes
    (pinned by a test), so the count can never drift from what generation
    actually selects. The player element is the ENTRANT count (the filter
    semantics -- see :func:`plo_pot_entrant_count`), not the live count.
    """
    return tuple(
        (n.actor, plo_node_action_context(n), plo_pot_entrant_count(n))
        for n in nodes
    )


def plo_active_player_count(node: PloDecisionNode) -> int:
    """Players still in the pot at this decision (incl. hero).

    A seat counts only if its LAST action is a non-fold -- a player who
    entered the pot and then folded (e.g. opened, then folded to a squeeze)
    is out of the hand, even though their dead money remains. Hero always
    counts. 2 = heads-up, 3 = three-way, etc. Port of
    :func:`pipeline.preflop.batch.active_player_count`; lets the Generate
    page ask for clean 3-/4-way spots instead of the deep multiway tail.
    """
    last_action: dict[str, PloActionType] = {}
    for a in node.history_before:
        last_action[a.seat] = a.action
    seats = {
        seat
        for seat, action in last_action.items()
        if action is not PloActionType.FOLD
    }
    seats.add(node.actor)
    return len(seats)


# Actions that put chips in the pot voluntarily (blind posts are not in the
# history; a FOLD commits nothing).
_VOLUNTARY = {
    PloActionType.CALL,
    PloActionType.RAISE,
    PloActionType.MIN_RAISE,
    PloActionType.ALL_IN,
}


def plo_pot_entrant_count(node: PloDecisionNode) -> int:
    """Players who have VOLUNTARILY put chips in the pot (incl. hero).

    A seat counts once it calls or raises anywhere in the history, EVEN IF
    it later folds -- its dead money still shapes the pot and its call still
    clutters the action line. Hero always counts. This is the counting the
    "Players in the pot" filter and the clean-lines player cap use (July 16
    2026 fix): the old live-seat counting
    (:func:`plo_active_player_count`) read a COLLAPSED squeeze pot as
    "heads-up" -- LJ opens, four players call, BB squeezes, everyone folds
    back = 6 entrants but only 2 live -- so caller-heavy monsters sailed
    through a "1-2 players" filter. Prose facts about who is STILL in the
    hand keep using the live count; this one answers "how multiway is this
    pot really".
    """
    seats = {a.seat for a in node.history_before if a.action in _VOLUNTARY}
    seats.add(node.actor)
    return len(seats)
