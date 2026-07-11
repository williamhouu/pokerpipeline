"""The ``animation_script`` column for POSTFLOP + FULL-HAND rows.

The app animates each question (chips flying, cards mucking, streets being
dealt) from this ONE self-contained JSON blob instead of regex-parsing the
Question prose -- the exact failure mode the old ``gto-formatter`` had and
``app_table_format`` was built to remove. Built from the SAME resolved
amounts as the prose and the seat tokens, so the animation can never
disagree with the question text.

Shape (version 1)::

    {"version": 1, "table_size": 8, "seats": ["UTG", ..., "BB"],
     "hero_seat": "BTN", "sb_bb": 0.5, "bb_bb": 1.0,
     "bb_dollars": 2.0,            # omitted for tournaments
     "starting_stack_bb": 200,
     "events": [
       {"i": 1, "type": "post", "seat": "SB", "amount_bb": 0.5,
        "pot_bb": 0.5, "stack_bb": 199.5},
       {"i": 3, "type": "fold", "seat": "UTG"},          # one event PER fold
       {"i": 8, "type": "raise", "seat": "BTN", "to_bb": 3.0,
        "amount_bb": 3.0, "pot_bb": 4.5, "stack_bb": 197.0},
       {"i": 11, "type": "deal", "street": "flop", "cards": ["Kd","7s","3s"]},
       {"i": 12, "type": "bet", "seat": "BB", "amount_bb": 11.5, ...},
       {"i": 13, "type": "decision", "seat": "BTN", "street": "flop"}]}

Design decisions (agreed July 2026):

* **One event per fold**, never a grouped seat list: the renderer's loop
  stays uniform, the ORDER is preserved (in a 3-bet pot the SB folds after
  the open, not with the early seats), and consecutive folds are trivially
  coalesced into one sweep by the renderer if it wants.
* Every chip event carries the pot and the actor's stack AFTER it, so the
  renderer does no arithmetic. ``amount_bb`` is the chips ADDED by the
  event; raises also carry ``to_bb`` (the street total in front).
* Amounts are exact numerics (2dp); dollar twins (``*_dollars``) appear on
  cash games. Display rounding (the 0.5bb grid) is the renderer's choice.
* The timeline ends at a ``decision`` event -- the seat and street the
  question asks about. No showdown: the play-through ends at the last
  decision, and the villain holds a range, not a hand (deliberate: a
  sampled showdown would inject outcome bias into a training product).
* Preflop folds and blind posts are SYNTHESIZED from the table's seat
  order (the prose deliberately skips them; the animation must not).

The chip-walk/event grammar lives in the shared leaf
:mod:`pipeline.animation` (used identically by the preflop and PLO
writers); this module adapts the postflop IR (the ``PreflopStep`` line,
which carries NO folds, and the per-node street history) onto it. Pure and
deterministic -- covered by the batch re-verifiers like every other column.
"""

from __future__ import annotations

from pipeline.animation import AnimationTable
from pipeline.postflop.solve import PostflopNode, PostflopSolve

_RAISE_VERBS = ("open", "raise", "3-bet", "4-bet", "5-bet")


def _table(solve: PostflopSolve) -> AnimationTable:
    return AnimationTable(
        table_size=solve.table_size,
        starting_stack_bb=float(solve.effective_stack_bb),
        bb_in_dollars=(
            float(solve.bb_in_dollars)
            if solve.game_format != "tournament" else None
        ),
    )


def _walk_preflop(
    table: AnimationTable, solve: PostflopSolve, *, stop_before_step: int | None,
) -> None:
    """Blind posts + the preflop orbit: per-seat folds synthesized from the
    seat order (the solve's summary carries only the LINE actions), the
    summary's actions in place, stopping (with a ``decision`` event) before
    the hero's step when asked.

    The cursor walks the still-active seats in order; a seat that is not
    the next line actor folds (first orbit only, by construction -- folded
    seats leave the rotation, so later orbits contain only line actors).
    """
    table.post_blinds()
    steps = list(enumerate(solve.preflop_summary))
    active = list(table.seats)
    i = 0
    while steps and active:
        i %= len(active)
        seat = active[i]
        idx, step = steps[0]
        if seat == step.position:
            if stop_before_step is not None and idx == stop_before_step:
                table.emit("decision", seat=seat, street="preflop")
                return
            steps.pop(0)
            if step.verb in _RAISE_VERBS and step.to_bb is not None:
                table.wager_to(seat, "raise", float(step.to_bb))
            elif step.verb == "call":
                table.call(seat)
            else:  # check (unused in the current solves, kept for closure)
                table.emit("check", seat=seat)
            i += 1
        else:
            table.emit("fold", seat=seat)
            active.pop(i)  # the next seat shifts into slot i


def _street_cards(board: tuple[str, ...], street: str) -> list[str]:
    if street == "flop":
        return list(board[:3])
    if street == "turn":
        return [board[3]] if len(board) > 3 else []
    return [board[4]] if len(board) > 4 else []


def _walk_postflop(table: AnimationTable, node: PostflopNode) -> None:
    """Deal each street up to the decision street, replay its history steps,
    then emit the ``decision`` event at the node's actor."""
    order = ("flop", "turn", "river")
    for street in order:
        if order.index(street) > order.index(node.street):
            break
        cards = _street_cards(node.board, street)
        if not cards:
            continue
        table.new_street()
        table.emit("deal", street=street, cards=cards)
        for step in (s for s in node.history if s.street == street):
            if step.verb in ("bet", "raise") and step.to_bb is not None:
                table.wager_to(
                    step.position, step.verb, float(step.to_bb),
                    all_in=step.all_in,
                )
            elif step.verb == "call":
                table.call(step.position)
            elif step.verb == "check":
                table.emit("check", seat=step.position)
            elif step.verb == "fold":
                table.emit("fold", seat=step.position)
    table.emit("decision", seat=node.actor, street=node.street)


def build_postflop_animation_script(node: PostflopNode, solve: PostflopSolve) -> str:
    """The full timeline for a postflop question: the whole preflop line
    (folds + blinds synthesized), each street's deal + actions, ending at
    the decision the question asks."""
    table = _table(solve)
    _walk_preflop(table, solve, stop_before_step=None)
    _walk_postflop(table, node)
    return table.render(node.actor)


def build_preflop_animation_script(
    solve: PostflopSolve, hero_seat: str, *, step_index: int | None = None,
) -> str:
    """The timeline for a PREFLOP question (an entry leg or a pack line
    leg): blinds, the folds and raises before hero's turn, ending at
    hero's ``decision``. ``step_index`` names WHICH of hero's decisions on
    a multi-raise line (None = hero's first step, the SRP entry case)."""
    if step_index is None:
        step_index = next(
            (i for i, st in enumerate(solve.preflop_summary)
             if st.position == hero_seat),
            0,
        )
    table = _table(solve)
    _walk_preflop(table, solve, stop_before_step=step_index)
    return table.render(hero_seat)


__all__ = [
    "build_postflop_animation_script",
    "build_preflop_animation_script",
]
