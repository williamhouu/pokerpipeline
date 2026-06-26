"""The solver-agnostic postflop intermediate representation (IR).

Everything in :mod:`pipeline.postflop` is built on these dataclasses, NOT on
any solver's native format. A thin *adapter* (one per solve source: PioSolver
``.cfr`` via the UPI client, a third-party SQLite ``.db`` export, ...) is the
only code that knows a vendor format; it populates a :class:`PostflopSolve`
and hands it to the pipeline. The synthetic fixture in
:mod:`pipeline.postflop.fixtures` builds the same IR by hand so the whole
pipeline runs with no external file.

Design notes
------------
* **Heads-up for v1.** The IR models a two-player tree (the trial solve is
  BTN vs BB). The shapes leave room for multiway later (ranges are a mapping
  keyed by position would be the extension), but v1 carries exactly two
  ranges per node.
* **"Hero" is node-relative.** At each node the player *to act* is the hero of
  any question sampled there; the other player is the villain. Across nodes
  the actor alternates (BB acts, then BTN acts), so the same solve yields both
  BB-hero and BTN-hero questions.
* **Everything is in big blinds.** Chip amounts (pot, stacks, wager sizes) are
  bb-denominated so the IR is stake-independent; dollar/stake rendering is a
  display concern handled by the format writer from ``bb_in_dollars`` +
  ``stakes``.
* **Per-combo strategy is the source of truth for questions.** ``strategy``
  maps each of the actor's combos to its action mix. Worthiness and the
  correct answer are per-combo (a node can be 50/50 in aggregate while a
  specific hand bets 80%). ``actions`` carries the *range-aggregate* view
  (frequency + range-mean EV per action) used for the options block and the
  EV gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# Canonical postflop action verbs. Bets/raises/calls carry a bb wager size;
# check/fold do not. The display LABEL (e.g. "Bet 33%") can differ per action
# even when the verb is the same, so options can distinguish two bet sizes.
POSTFLOP_VERBS = frozenset({"check", "bet", "call", "raise", "fold"})
STREETS = ("flop", "turn", "river")


@dataclass(frozen=True)
class PreflopStep:
    """One preflop action that built the pot, for action-history rendering.

    The postflop pipeline does not re-solve preflop; it only needs the line
    that created the starting pot so the question can show the full hand
    context. ``to_bb`` is the amount raised *to* (an open to 2.5bb, a 3-bet to
    9bb); it is ``None`` for calls/folds/checks.
    """

    position: str
    verb: str  # "open" | "call" | "3-bet" | "4-bet" | "fold" | "check"
    to_bb: float | None = None


@dataclass(frozen=True)
class PostflopStep:
    """One postflop action taken BEFORE the current decision node.

    The ordered tuple of these on a node is what the action-history renderer
    walks to produce "BB checks, BTN bets 2bb, BB calls" for each street up to
    the decision. ``to_bb`` is the total wager (None for check/fold).
    """

    street: str
    position: str
    verb: str
    to_bb: float | None = None


@dataclass(frozen=True)
class NodeAction:
    """One available action at a decision node (range-aggregate view).

    ``label`` is the display string the option builder uses (e.g. "Check",
    "Bet 33%", "Bet 75%", "Call", "Raise", "Fold"); ``verb`` is the canonical
    family. ``freq`` is the action's frequency across the actor's whole range
    and ``ev_bb`` its range-mean EV in bb (used for the EV gap and difficulty);
    ``ev_bb`` is ``None`` when the source solve doesn't expose per-action EVs.
    """

    label: str
    verb: str
    freq: float
    to_bb: float | None = None
    pot_fraction: float | None = None  # bet/raise size as a fraction of the pot
    ev_bb: float | None = None


@dataclass(frozen=True)
class PostflopNode:
    """A single postflop decision point in a solved tree.

    ``strategy[combo][action_label]`` is the per-combo action mix (sums to ~1
    per combo). ``hero_range`` / ``villain_range`` map ``combo -> weight`` (the
    reach probability of each combo at this node) and drive equity. ``pot_bb``
    is the pot hero is playing for INCLUDING any unmatched bet hero faces;
    ``to_call_bb`` is what hero must add to call (0 when first to act / checked
    to). ``history`` is every postflop action before this node, across streets.
    """

    node_id: str
    street: str
    board: tuple[str, ...]  # all revealed community cards at this node (3-5)
    actor: str  # position to act (the hero of any spot sampled here)
    villain: str  # the other player's position
    pot_bb: float
    effective_stack_bb: float
    actions: tuple[NodeAction, ...]
    strategy: Mapping[str, Mapping[str, float]]
    hero_range: Mapping[str, float]
    villain_range: Mapping[str, float]
    history: tuple[PostflopStep, ...] = ()
    to_call_bb: float = 0.0  # chips hero must add to call (0 if not facing a bet)
    # OPTIONAL per-combo action EVs: {combo: {action_label: ev_bb}}. Solvers
    # expose per-hand EVs; when present they give a hero-specific EV gap (the
    # cost of the best wrong answer for THIS hand) instead of the range-mean
    # gap from ``actions[*].ev_bb``, which mis-signals for hands that deviate
    # from the range's modal action. The fact extractor prefers these and
    # falls back to the range-mean action EVs, then to None.
    combo_evs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def is_facing_bet(self) -> bool:
        """True when hero must call a wager to continue (vs first-to-act)."""
        return self.to_call_bb > 0.0

    @property
    def spr(self) -> float:
        """Stack-to-pot ratio at the decision (effective stack / current pot).

        A rough descriptor for difficulty / commitment framing, not a precise
        solver figure. ``pot_bb`` includes any bet hero faces, so this is the
        SPR hero is actually weighing.
        """
        return self.effective_stack_bb / self.pot_bb if self.pot_bb else 0.0

    def action_by_label(self, label: str) -> NodeAction | None:
        for a in self.actions:
            if a.label == label:
                return a
        return None


@dataclass(frozen=True)
class PostflopSolve:
    """A whole solved postflop tree in the IR, plus the preflop context.

    ``nodes`` is keyed by ``node_id``. ``flop`` is the three-card flop;
    individual nodes carry their own (possibly longer) board. ``positions`` is
    ``(oop_position, ip_position)`` -- OOP acts first postflop.
    ``preflop_summary`` is the line that created ``starting_pot_bb`` (entering
    the flop). Display metadata (``stakes``, ``bb_in_dollars``,
    ``live_or_online``) is used only by the format writer.
    """

    solve_id: str
    positions: tuple[str, str]  # (oop_position, ip_position)
    effective_stack_bb: float
    starting_pot_bb: float  # pot entering the flop (after the preflop line)
    flop: tuple[str, str, str]
    preflop_summary: tuple[PreflopStep, ...]
    nodes: Mapping[str, PostflopNode]
    game_format: str = "cash"  # "cash" | "tournament"
    bb_in_dollars: float = 1.0
    stakes: str = ""
    live_or_online: str = "Online"
    table_size: int = 6
    # Rake structure the solve was solved with, e.g. "8% cap 2bb" (already baked
    # into the solver's EVs; carried for DISPLAY so the question Context can state
    # the structure -- different solves use different rake/ante). "" / "none" = none.
    rake: str = ""
    # Free-form provenance carried into the CSV solver_reference column.
    source_reference: str = ""

    @property
    def oop_position(self) -> str:
        return self.positions[0]

    @property
    def ip_position(self) -> str:
        return self.positions[1]


# --- light validation -------------------------------------------------------
def validate_solve(solve: PostflopSolve) -> list[str]:
    """Return a list of structural problems with ``solve`` (empty == clean).

    A defensive sanity check an adapter (or a fixture author) can run before
    feeding a solve to the pipeline -- catches the classes of bug that would
    otherwise surface as confusing failures three layers deep. NOT a poker
    correctness check; purely structural.
    """
    problems: list[str] = []
    if len(solve.flop) != 3:
        problems.append(f"flop must have 3 cards, got {solve.flop!r}")
    if solve.oop_position == solve.ip_position:
        problems.append("oop and ip positions are identical")
    for node_id, node in solve.nodes.items():
        if node_id != node.node_id:
            problems.append(f"node key {node_id!r} != node.node_id {node.node_id!r}")
        if node.street not in STREETS:
            problems.append(f"{node_id}: bad street {node.street!r}")
        if not node.actions:
            problems.append(f"{node_id}: no actions")
        labels = {a.label for a in node.actions}
        for verb in (a.verb for a in node.actions):
            if verb not in POSTFLOP_VERBS:
                problems.append(f"{node_id}: bad verb {verb!r}")
        # Every action a combo plays must be a real node action.
        for combo, mix in node.strategy.items():
            for label in mix:
                if label not in labels:
                    problems.append(
                        f"{node_id}: combo {combo} plays unknown action {label!r}"
                    )
        if node.actor not in (solve.oop_position, solve.ip_position):
            problems.append(f"{node_id}: actor {node.actor!r} not in solve positions")
    return problems


__all__ = [
    "POSTFLOP_VERBS",
    "STREETS",
    "NodeAction",
    "PostflopNode",
    "PostflopSolve",
    "PostflopStep",
    "PreflopStep",
    "validate_solve",
]
