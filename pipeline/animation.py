"""Shared builder for the ``animation_script`` CSV column (ALL pipelines).

One JSON timeline per question row: ordered events (blind posts, ONE event
per fold, raises/calls with amounts, explicit street deals) with the pot and
the actor's stack AFTER every chip event, ending at a ``decision`` event
where the question pauses. The app animates from this instead of parsing
prose -- see :mod:`pipeline.postflop.animation_script` for the full design
notes (one-event-per-fold rationale, no-showdown rationale, exact-numerics
policy).

This is the pure shared leaf (like :mod:`pipeline.chat_context`): the
postflop, preflop, and PLO writers each adapt their own history shapes onto
:class:`AnimationTable` + :func:`walk_explicit_preflop`, so the event
grammar and JSON shape stay identical across all four writers.
"""

from __future__ import annotations

import json
from typing import Any

from pipeline.action_history import preflop_order  # pure, game-agnostic leaf

ANIMATION_SCRIPT_VERSION = 1

SB_BB = 0.5
BB_BB = 1.0


def _r2(x: float) -> float:
    """2dp round that never emits -0.0 (JSON noise)."""
    v = round(float(x) + 0.0, 2)
    return 0.0 if v == 0 else v


class AnimationTable:
    """The running table state + event list for one timeline.

    ``bb_in_dollars=None`` (tournaments) omits every dollar twin. ``seats``
    defaults to the standard preflop order for the table size; pass an
    explicit list when the pipeline's seat labels differ (PLO's display
    names)."""

    def __init__(
        self, *, table_size: int, starting_stack_bb: float,
        bb_in_dollars: float | None = None, seats: list[str] | None = None,
        ante_bb: float = 0.0, display_dollars: bool = False,
    ):
        self.table_size = table_size
        self.seats = list(seats) if seats is not None else preflop_order(table_size)
        self.starting_stack_bb = float(starting_stack_bb)
        self.ante_bb = float(ante_bb)
        self.dollars = bb_in_dollars
        # The question's DISPLAY currency (Aug 2026, the app's bb/$ mixed-
        # rendering bug): when True the header carries "display": "dollars",
        # telling the renderer this row's prose + answer options are in
        # dollars, so EVERY animated surface must render the _dollars twins.
        # Absent = render bb everywhere (the default; keeps every existing
        # bb/tournament batch byte-identical). ADDITIVE -- version unchanged,
        # per the format's render-what-you-understand rule.
        self.display_dollars = bool(display_dollars)
        self.stacks = {s: self.starting_stack_bb for s in self.seats}
        self.committed = {s: 0.0 for s in self.seats}  # this street's chips
        self.pot = 0.0
        self.events: list[dict[str, Any]] = []

    def money(self, event: dict, **bb_fields: float) -> None:
        for name, bb in bb_fields.items():
            event[name] = _r2(bb)
            if self.dollars is not None:
                event[name.replace("_bb", "_dollars")] = _r2(bb * self.dollars)

    def emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {"i": len(self.events) + 1, "type": type_}
        event.update(fields)
        self.events.append(event)
        return event

    def post_blinds(self) -> None:
        self.wager_to("SB", "post", SB_BB)
        self.wager_to("BB", "post", BB_BB)
        if self.ante_bb > 0:
            self.post_ante()

    def post_ante(self) -> None:
        """The big-blind ante (MTT bb-ante formats, July 2026): DEAD money --
        it joins the pot and leaves the BB's stack, but is NOT part of the
        BB's callable street commitment. Additive event shape: type "post"
        with an ``ante: true`` marker, so a renderer that predates antes
        just shows another post."""
        self.pot += self.ante_bb
        self.stacks["BB"] -= self.ante_bb
        event = self.emit("post", seat="BB", ante=True)
        self.money(event, amount_bb=self.ante_bb, pot_bb=self.pot,
                    stack_bb=self.stacks["BB"])

    def wager_to(
        self, seat: str, type_: str, to_bb: float, *, all_in: bool = False,
    ) -> None:
        """A bet/raise/post that brings ``seat``'s street total to ``to_bb``.

        CHIP-CONSERVATION CLAMP (Aug 2026, MTT animation audit): a player
        can never wager more than their remaining stack. The Monker
        bb-ante packs label a BB jam by the PRE-ante stack ("all-in to
        10bb" when the posted ante left only 9), which drove ``stack_bb``
        to -1 in 89 shipped animation events. The wager is clamped to the
        chips actually behind, and a clamped (or stack-emptying) wager is
        marked ``all_in`` -- amounts in the timeline are now always
        physically possible, whatever the source data claims.
        """
        added = to_bb - self.committed[seat]
        if added > self.stacks[seat]:
            added = self.stacks[seat]
            to_bb = self.committed[seat] + added
            all_in = True
        self.pot += added
        self.committed[seat] = to_bb
        self.stacks[seat] -= added
        if self.stacks[seat] <= 1e-9:
            self.stacks[seat] = 0.0
            all_in = True
        event = self.emit(type_, seat=seat)
        self.money(event, amount_bb=added, to_bb=to_bb,
                    pot_bb=self.pot, stack_bb=self.stacks[seat])
        if all_in:
            event["all_in"] = True

    def call(self, seat: str, *, all_in: bool = False) -> None:
        need = max(self.committed.values()) - self.committed[seat]
        # Same chip-conservation clamp as wager_to: a call for more than
        # the stack is a call-for-less all-in.
        if need > self.stacks[seat]:
            need = self.stacks[seat]
            all_in = True
        self.pot += need
        self.committed[seat] += need
        self.stacks[seat] -= need
        if self.stacks[seat] <= 1e-9:
            self.stacks[seat] = 0.0
            all_in = True
        event = self.emit("call", seat=seat)
        self.money(event, amount_bb=need, pot_bb=self.pot,
                    stack_bb=self.stacks[seat])
        if all_in:
            event["all_in"] = True

    def new_street(self) -> None:
        self.committed = {s: 0.0 for s in self.seats}

    def render(self, hero_seat: str) -> str:
        payload: dict[str, Any] = {
            "version": ANIMATION_SCRIPT_VERSION,
            "table_size": self.table_size,
            "seats": self.seats,
            "hero_seat": hero_seat,
            "sb_bb": SB_BB,
            "bb_bb": BB_BB,
            "starting_stack_bb": _r2(self.starting_stack_bb),
            "events": self.events,
        }
        if self.ante_bb > 0:
            payload["ante_bb"] = _r2(self.ante_bb)
        if self.dollars is not None:
            payload["bb_dollars"] = _r2(self.dollars)
        if self.display_dollars:
            payload["display"] = "dollars"
        return json.dumps(payload, separators=(",", ":"))


def walk_explicit_preflop(
    table: AnimationTable,
    actions: list[tuple[str, str, float | None, bool]],
    *,
    decision_seat: str,
) -> None:
    """Blind posts + an EXPLICIT prior-action sequence (folds included, as
    the preflop/PLO pack histories carry them), ending at hero's decision.

    ``actions`` items are ``(seat, verb, to_bb, all_in)`` with verb in
    fold / call / check / raise (raise carries the raise-TO size in bb;
    an all-in is a raise/call with ``all_in=True``).
    """
    table.post_blinds()
    for seat, verb, to_bb, all_in in actions:
        if verb == "fold":
            table.emit("fold", seat=seat)
        elif verb == "check":
            table.emit("check", seat=seat)
        elif verb == "call":
            table.call(seat, all_in=all_in)
        elif verb == "raise" and to_bb is not None:
            table.wager_to(seat, "raise", float(to_bb), all_in=all_in)
        else:  # a raise with no size cannot be animated truthfully
            raise ValueError(f"unanimatable preflop action {verb!r} for {seat}")
    table.emit("decision", seat=decision_seat, street="preflop")


def hero_stack_at_preflop_decision(
    *,
    table_size: int,
    starting_stack_bb: float,
    ante_bb: float,
    actions: list[tuple[str, str, float | None, bool]],
    hero_seat: str,
    seats: list[str] | None = None,
) -> float:
    """Hero's remaining stack when the preflop decision arrives.

    Pure replay of :func:`walk_explicit_preflop` with no rendering -- the
    VACUOUS-DECISION GATE (Aug 2026 MTT audit): a hero with 0bb behind has
    no legal action, yet Monker's tree emits such decision nodes (side-pot
    formalities, complete with meaningless fold/call frequencies), and 47
    short-stack MTT questions shipped asking them ("you call the shove
    all-in; villain re-jams; what do you do?"). Callers skip the node when
    this returns 0.
    """
    table = AnimationTable(
        table_size=table_size,
        starting_stack_bb=float(starting_stack_bb),
        ante_bb=float(ante_bb),
        seats=seats,
    )
    walk_explicit_preflop(table, actions, decision_seat=hero_seat)
    return table.stacks[hero_seat]


__all__ = [
    "ANIMATION_SCRIPT_VERSION",
    "BB_BB",
    "SB_BB",
    "AnimationTable",
    "walk_explicit_preflop",
]
