"""PLO action-history + context rendering (deterministic, no LLM).

Produces the two text blocks before every question -- the context line and the
action history (the "Question" prose) -- for a PLO spot. Forks
:mod:`pipeline.action_history` (the NLHE renderer can't take the PLO seat set:
it has no ``BU`` and its 6-max order is ``UTG,HJ,CO,BTN,SB,BB`` not
``LJ,HJ,CO,BU,SB,BB``). It reuses the shared, card-agnostic
:func:`pipeline.action_history.format_card`.

The genuinely new piece is :func:`resolve_pot_limit` -- a pot-limit betting
calculator. Unlike NLHE (a per-token raise-size lookup table), every PLO raise
size is *computed* from the pot-limit rule, so no table is maintained:

    to_call    = high_bet - raiser's current commitment
    raise_to   = high_bet + raise%/100 * (pot + to_call)

With SB = 0.5 bb, BB = 1 bb, a 100%-pot open is ``1 + 1.0*(1.5 + 1) = 3.5`` bb;
a pot 3-bet over it is ``3.5 + 1.0*(5 + 3.5) = 12`` bb. An all-in is the full
stack. The resolved amounts feed both this renderer and the app-table format, so
the prose and the chip tokens always agree.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.action_history import format_card
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.node_enumerator import PloDecisionNode
from pipeline.plo.pack import PloAction, PloActionType

# The pack speaks Monker's full-ring dialect (LJ for the first seat, BU for
# the button); every PLAYER-FACING surface uses the NLHE/app convention
# instead (UTG, BTN) so both games read the same. Pack internals -- filenames,
# node ids, solver_reference -- keep the Monker codes untouched.
DISPLAY_SEAT = {"LJ": "UTG", "BU": "BTN"}


def display_seat(seat: str) -> str:
    """Player-facing seat code for a pack seat code (LJ -> UTG, BU -> BTN)."""
    return DISPLAY_SEAT.get(seat, seat)


# Position phrases mirror the NLHE table exactly (UTG takes no article).
_HERO_PHRASE = {
    "LJ": "UTG",
    "HJ": "in the Hijack",
    "CO": "in the Cutoff",
    "BU": "on the Button",
    "SB": "in the Small Blind",
    "BB": "in the Big Blind",
}
_VILLAIN_REF = {
    "LJ": "UTG",
    "HJ": "The Hijack",
    "CO": "The Cutoff",
    "BU": "The Button",
    "SB": "The Small Blind",
    "BB": "The Big Blind",
}
_RAISE_LEVEL = {1: "open", 2: "3-bet", 3: "4-bet", 4: "5-bet"}
_RAISE_VERBS = frozenset({"open", "3-bet", "4-bet", "5-bet", "raise"})
_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}

_SB_BB = 0.5  # small blind in big-blind units
_BB = 1.0


@dataclass(frozen=True)
class ResolvedAction:
    """One prior action with its pot-limit-resolved size (in big blinds)."""

    seat: str
    verb: str  # open / 3-bet / 4-bet / 5-bet / raise / call / fold / all-in
    to_bb: float | None  # the "raise to" / all-in total in bb; None for fold/call


def resolve_pot_limit(
    history: tuple[PloAction, ...], *, stack_bb: float = 100.0
) -> tuple[tuple[ResolvedAction, ...], float]:
    """Resolve a prior-action sequence into per-action bb sizes + the pot (bb).

    Walks the actions after posting the blinds, tracking each seat's commitment,
    the current high bet, and the pot, computing each raise's ``raise_to`` via
    the pot-limit rule. Raise levels (open / 3-bet / ...) count every aggressive
    action, including all-ins.
    """
    committed: dict[str, float] = {"SB": _SB_BB, "BB": _BB}
    high_bet = _BB
    pot = _SB_BB + _BB
    raise_level = 0
    out: list[ResolvedAction] = []

    for action in history:
        seat = action.seat
        prev = committed.get(seat, 0.0)
        if action.action is PloActionType.FOLD:
            out.append(ResolvedAction(seat, "fold", None))
        elif action.action is PloActionType.CALL:
            pot += high_bet - prev
            committed[seat] = high_bet
            out.append(ResolvedAction(seat, "call", None))
        elif action.action is PloActionType.ALL_IN:
            raise_level += 1
            pot += stack_bb - prev
            committed[seat] = stack_bb
            high_bet = max(high_bet, stack_bb)
            out.append(ResolvedAction(seat, "all-in", stack_bb))
        else:  # RAISE / MIN_RAISE
            raise_level += 1
            to_call = high_bet - prev
            pct = (action.raise_pct or 100) / 100.0
            raise_to = high_bet + pct * (pot + to_call)
            pot += raise_to - prev
            committed[seat] = raise_to
            high_bet = raise_to
            out.append(
                ResolvedAction(seat, _RAISE_LEVEL.get(raise_level, "raise"), raise_to)
            )
    return tuple(out), pot


# --- formatting ------------------------------------------------------------
def _fmt_num(value: float) -> str:
    """Comma-grouped number; drop a trailing ``.0`` (``100.0`` -> ``'100'``)."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,}"


def _money(amount: float, *, render_bb: bool) -> str:
    """``3.5`` -> ``'$3.50'`` (cash) or ``'3.5bb'`` (bb display)."""
    if render_bb:
        return f"{_fmt_num(amount)}bb"
    if float(amount).is_integer():
        return f"${int(amount):,}"
    return f"${amount:,.2f}"


def _conjugate(base: str, *, is_hero: bool) -> str:
    """Hero uses the base verb; a villain takes the third-person form."""
    if is_hero:
        return base
    return "moves all-in" if base == "move all-in" else base + "s"


def _hole_cards(cards: tuple[str, ...]) -> str:
    return " ".join(format_card(c) for c in cards)


def _sentence(act: ResolvedAction, hero: str, *, render_bb: bool) -> str:
    is_hero = act.seat == hero
    subject = "You" if is_hero else _VILLAIN_REF[act.seat]
    if act.verb in _RAISE_VERBS:
        amount = _money(act.to_bb or 0.0, render_bb=render_bb)
        return f"{subject} {_conjugate(act.verb, is_hero=is_hero)} to {amount}."
    if act.verb == "all-in":
        amount = _money(act.to_bb or 0.0, render_bb=render_bb)
        return f"{subject} {_conjugate('move all-in', is_hero=is_hero)} for {amount}."
    if act.verb == "call":
        return f"{subject} {_conjugate('call', is_hero=is_hero)}."
    if act.verb == "fold":
        # Only rendered when include_folds is set (the LLM-facing variant);
        # the player-facing question drops folds per the brief's Fold Rule.
        return f"{subject} {_conjugate('fold', is_hero=is_hero)}."
    raise ValueError(f"unexpected verb {act.verb!r}")


def format_plo_action_history(
    facts: PloFacts,
    *,
    stakes_bb_dollars: float = 1.0,
    game_format: str = "cash",
    display_in_bb: bool = False,
    stack_bb: float = 100.0,
    include_folds: bool = False,
) -> str:
    """The action-history ("Question") block for a PLO spot.

    ``You're <position> with <4 cards>.`` then one sentence per prior non-fold
    action (preflop folds are dropped, per the brief's Fold Rule). Amounts are
    pot-limit-resolved; rendered in dollars for cash, or bb for tournament /
    when ``display_in_bb``.

    ``include_folds=True`` renders the folds too ("The Hijack folds.") -- the
    LLM-facing variant. The player sees the fold on the app's table render
    (Seats marks an entrant who folded), but the LLM sees only this prose, so
    without the folds it cannot tell "facing a squeeze after the opener
    folded" (heads-up) from "opener still in" (multiway) -- sibling nodes
    whose correct answers differ.
    """
    render_bb = display_in_bb or game_format != "cash"
    actions, _pot = resolve_pot_limit(
        facts.spot.node.history_before, stack_bb=stack_bb
    )
    hero = facts.spot.node.actor
    sentences = [f"You're {_HERO_PHRASE[hero]} with {_hole_cards(facts.spot.hero_cards)}."]
    sentences += [
        _sentence(act, hero, render_bb=render_bb)
        for act in actions
        if include_folds or act.verb != "fold"
    ]
    return " ".join(sentences)


def format_plo_context(
    *,
    stakes_bb_dollars: float = 1.0,
    stack_bb: float = 100.0,
    game_format: str = "cash",
    display_in_bb: bool = False,
    live_or_online: str = "Online",
) -> str:
    """The context line: stakes + effective stacks.

    Table size is intentionally NOT shown -- the dedicated Table Size column
    already carries it, so repeating it here was redundant (dropped June 2026
    per the team's feedback).
    """
    if game_format != "cash":
        return f"PLO tournament. {_fmt_num(stack_bb)}bb effective stacks."
    venue = live_or_online.capitalize()
    if display_in_bb:
        return f"{venue} PLO cash. {_fmt_num(stack_bb)}bb effective stacks."
    sb = stakes_bb_dollars * _SB_BB
    eff = stack_bb * stakes_bb_dollars
    return (
        f"${_fmt_num(sb)}/${_fmt_num(stakes_bb_dollars)} {venue} PLO cash. "
        f"${_fmt_num(eff)} effective stacks."
    )


def pot_bb(node: PloDecisionNode, *, stack_bb: float = 100.0) -> float:
    """The pot (in bb) at the moment hero must act."""
    _actions, pot = resolve_pot_limit(node.history_before, stack_bb=stack_bb)
    return pot


__all__ = [
    "ResolvedAction",
    "format_plo_action_history",
    "format_plo_context",
    "pot_bb",
    "resolve_pot_limit",
]
