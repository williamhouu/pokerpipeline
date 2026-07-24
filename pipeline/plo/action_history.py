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
from pipeline.bb_display import round_to_half_bb  # 0.5bb display grid (leaf)
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.node_enumerator import PloDecisionNode
from pipeline.plo.pack import PloAction, PloActionType

# The 6-max pack speaks Monker's full-ring dialect (LJ for its FIRST seat, BU
# for the button); every PLAYER-FACING surface uses the NLHE/app convention
# instead (UTG, BTN) so both games read the same. Pack internals -- filenames,
# node ids, solver_reference -- keep the Monker codes untouched.
# The 9-max pack's seats are already the app names (UTG, UTG+1, ..., BTN, and
# a REAL Lojack), so its remap is the identity -- which is exactly why every
# lookup here is table-size aware: 6-max LJ means UTG, 9-max LJ means Lojack.
DISPLAY_SEAT = {"LJ": "UTG", "BU": "BTN"}


def display_seat(seat: str, *, table_size: int = 6) -> str:
    """Player-facing seat code for a pack seat code.

    6-max: the Monker-dialect remap (LJ -> UTG, BU -> BTN). Other table
    sizes: identity (their pack seats are already the app names).
    """
    if table_size == 6:  # noqa: PLR2004
        return DISPLAY_SEAT.get(seat, seat)
    return seat


# Position phrases mirror the shared NLHE table exactly (UTG-family seats
# take no article), keyed by DISPLAY seat name; look up via _hero_phrase /
# _villain_ref so the 6-max dialect is remapped first.
_HERO_PHRASE = {
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "UTG+2": "UTG+2",
    "LJ": "in the Lojack",
    "HJ": "in the Hijack",
    "CO": "in the Cutoff",
    "BTN": "on the Button",
    "SB": "in the Small Blind",
    "BB": "in the Big Blind",
}
_VILLAIN_REF = {
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "UTG+2": "UTG+2",
    "LJ": "The Lojack",
    "HJ": "The Hijack",
    "CO": "The Cutoff",
    "BTN": "The Button",
    "SB": "The Small Blind",
    "BB": "The Big Blind",
}


def _hero_phrase(seat: str, *, table_size: int) -> str:
    return _HERO_PHRASE[display_seat(seat, table_size=table_size)]


def _villain_ref(seat: str, *, table_size: int) -> str:
    return _VILLAIN_REF[display_seat(seat, table_size=table_size)]
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


def _walk(
    history: tuple[PloAction, ...],
    *,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
) -> tuple[tuple[ResolvedAction, ...], float, dict[str, float], float]:
    """THE pot-limit betting walk -- the single source of truth.

    Returns ``(resolved_actions, pot_bb, committed_by_seat, high_bet_bb)``.

    INVARIANT (July 23 2026): every consumer of resolved sizes, the pot,
    per-seat commitments, or the call price MUST derive from this ONE walk.
    This module used to carry three hand-copied walks (resolve_pot_limit /
    resolved_commitment_bb / call_price); the July-22 min-raise fix landed
    in only two of them, so `call_price` kept pricing a 2bb min-raise open
    as a 3.5bb pot open -- 109 shipped 20bb-pack questions carried pot-odds
    numbers for a different action sequence than the one displayed. Never
    reintroduce a parallel walk; extend this one.

    Walks the actions after posting the blinds, tracking each seat's
    commitment, the current high bet, and the pot, computing each raise's
    ``raise_to`` via the pot-limit rule. Raise levels (open / 3-bet / ...)
    count every aggressive action, including all-ins.

    ``ante_bb`` (MTT bb-ante packs, July 2026): the BB's dead ante joins the
    pot before any action -- the packs' sized-raise tokens only resolve to
    their advertised sizes with it -- but it is NOT part of the BB's callable
    commitment, and it comes off the BB's stack, so the BB's all-in total is
    ``stack_bb - ante_bb``.
    """
    committed: dict[str, float] = {"SB": _SB_BB, "BB": _BB}
    high_bet = _BB
    pot = _SB_BB + _BB + ante_bb
    raise_level = 0
    # The min-raise rule needs the size of the LAST bet/raise increment: a
    # min-raise is "match the high bet, then raise by at least that much
    # again". Preflop the BB post is the opening bet, so the first-in
    # min-raise goes to exactly 2bb.
    last_increment = _BB
    out: list[ResolvedAction] = []

    def _allin_total(seat: str) -> float:
        return stack_bb - ante_bb if seat == "BB" else stack_bb

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
            total = _allin_total(seat)
            pot += total - prev
            committed[seat] = total
            if total > high_bet:
                last_increment = total - high_bet
                high_bet = total
            out.append(ResolvedAction(seat, "all-in", total))
        else:  # RAISE / MIN_RAISE
            raise_level += 1
            to_call = high_bet - prev
            if action.action is PloActionType.MIN_RAISE:
                # A REAL min-raise: high bet + the last raise increment.
                # July 22 2026 fact-correctness fix: this used to fall into
                # the pot-raise arm below (raise_pct None -> "or 100"), so a
                # 2bb min-raise open rendered as a 3.5bb pot open and every
                # size after it compounded the error (a pot 3-bet showed
                # 12bb where the real line is 7.5bb) -- while the EV panel,
                # reading the solver file, showed the truth (fold EV -2bb).
                # Caught by the user comparing the two, July 22.
                raise_to = high_bet + last_increment
            else:
                pct = (action.raise_pct or 100) / 100.0
                raise_to = high_bet + pct * (pot + to_call)
            last_increment = raise_to - high_bet
            pot += raise_to - prev
            committed[seat] = raise_to
            high_bet = raise_to
            out.append(
                ResolvedAction(seat, _RAISE_LEVEL.get(raise_level, "raise"), raise_to)
            )
    return tuple(out), pot, committed, high_bet


def resolve_pot_limit(
    history: tuple[PloAction, ...],
    *,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
) -> tuple[tuple[ResolvedAction, ...], float]:
    """Resolve a prior-action sequence into per-action bb sizes + the pot (bb).

    A view over :func:`_walk` (see its invariant); feeds the prose, the app
    tokens, and the animation script.
    """
    actions, pot, _committed, _high_bet = _walk(
        history, stack_bb=stack_bb, ante_bb=ante_bb
    )
    return actions, pot


def resolved_commitment_bb(
    history: tuple[PloAction, ...],
    seat: str,
    *,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
) -> float:
    """How many bb ``seat`` has voluntarily committed (blind posts included,
    the dead ante excluded) after ``history`` -- a view over :func:`_walk`,
    exposed for the fold-EV consistency guard."""
    _actions, _pot, committed, _high_bet = _walk(
        history, stack_bb=stack_bb, ante_bb=ante_bb
    )
    return committed.get(seat, 0.0)


def fold_ev_consistency_issue(
    spot: object,
    *,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
    tolerance_bb: float = 0.05,
) -> str | None:
    """The solver file's own fold EV must equal MINUS what hero invested.

    The size-resolution tripwire (July 22 2026, born from the min-raise bug:
    a 2bb open rendered as 3.5bb while the file's fold EV said -2.0bb).
    Folding surrenders exactly the chips voluntarily committed (blinds
    included; the MTT ante is dead before the EV baseline and excluded), so
    for EVERY hand the pack's fold EV is -(hero's invested amount). If our
    resolved sizes ever disagree with that, the display arithmetic is wrong
    for this line -- for ANY pack, ANY token type, automatically. Returns a
    plain-English issue string, or None when consistent / no fold on the
    menu. ``spot`` needs ``ev_by_action`` (sb units) and ``node``.
    """
    fold_ev_sb = getattr(spot, "ev_by_action", {}).get("Fold")
    if fold_ev_sb is None:
        return None
    node = spot.node
    invested = resolved_commitment_bb(
        node.history_before, node.actor, stack_bb=stack_bb, ante_bb=ante_bb
    )
    fold_ev_bb = fold_ev_sb / 2.0
    if abs(fold_ev_bb + invested) > tolerance_bb:
        return (
            f"fold-EV consistency: the solver file says folding costs "
            f"{fold_ev_bb:+.2f}bb, but the resolved action sizes say hero "
            f"invested {invested:.2f}bb -- a displayed bet size on this "
            f"line is wrong (size-resolution bug, not a solver issue)."
        )
    return None


# --- formatting ------------------------------------------------------------
def _fmt_num(value: float) -> str:
    """Comma-grouped number; drop a trailing ``.0`` (``100.0`` -> ``'100'``)."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,}"


def _money(amount: float, *, render_bb: bool) -> str:
    """``3.5`` -> ``'$3.50'`` (cash) or ``'3.5bb'`` (bb display).

    The bb path snaps to the 0.5bb display grid (pot-relative PLO sizes are
    otherwise ugly, e.g. 7.34bb); the dollar path keeps exact cents. Display-only
    -- strategic facts use the exact amount. See :mod:`pipeline.bb_display`."""
    if render_bb:
        return f"{_fmt_num(round_to_half_bb(amount))}bb"
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


def _sentence(
    act: ResolvedAction, hero: str, *, render_bb: bool, table_size: int
) -> str:
    is_hero = act.seat == hero
    subject = "You" if is_hero else _villain_ref(act.seat, table_size=table_size)
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
    ante_bb: float = 0.0,
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
        facts.spot.node.history_before, stack_bb=stack_bb, ante_bb=ante_bb
    )
    hero = facts.spot.node.actor
    table_size = facts.spot.node.table_size
    sentences = [
        f"You're {_hero_phrase(hero, table_size=table_size)} "
        f"with {_hole_cards(facts.spot.hero_cards)}."
    ]
    sentences += [
        _sentence(act, hero, render_bb=render_bb, table_size=table_size)
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
    table_size: int = 6,
    ante_bb: float = 0.0,
    venue_neutral: bool = False,
) -> str:
    """The context line: stakes + effective stacks.

    Table size is intentionally NOT shown -- the dedicated Table Size column
    already carries it, so repeating it here was redundant (dropped June 2026
    per the team's feedback).

    9-max pack questions (July 2026, team ask): the Context carries ONLY the
    effective stack size -- no stakes, venue, or game-type framing. The
    6-max format is unchanged. ``venue_neutral`` (July 22 2026, team ask)
    gives a 6-max cash pack the same stacks-only treatment -- set for the
    short-stack 12bb/20bb packs, whose venue is deliberately unspecified.
    """
    if table_size == 9 or (venue_neutral and game_format == "cash"):  # noqa: PLR2004
        if display_in_bb or game_format != "cash":
            return f"{_fmt_num(stack_bb)}bb effective stacks."
        return f"${_fmt_num(stack_bb * stakes_bb_dollars)} effective stacks."
    if game_format != "cash":
        ante = f" Big blind ante {_fmt_num(ante_bb)}bb." if ante_bb else ""
        # No game-type prefix (July 2026, team ask): the Holdem and PLO apps
        # are separate products, so "PLO" is dead weight in the Context line.
        return f"Tournament. {_fmt_num(stack_bb)}bb effective stacks.{ante}"
    venue = live_or_online.capitalize()
    if display_in_bb:
        return f"{venue} PLO cash. {_fmt_num(stack_bb)}bb effective stacks."
    sb = stakes_bb_dollars * _SB_BB
    eff = stack_bb * stakes_bb_dollars
    return (
        f"${_fmt_num(sb)}/${_fmt_num(stakes_bb_dollars)} {venue} PLO cash. "
        f"${_fmt_num(eff)} effective stacks."
    )


def pot_bb(
    node: PloDecisionNode, *, stack_bb: float = 100.0, ante_bb: float = 0.0
) -> float:
    """The pot (in bb) at the moment hero must act."""
    _actions, pot = resolve_pot_limit(
        node.history_before, stack_bb=stack_bb, ante_bb=ante_bb
    )
    return pot


def call_price(
    history: tuple[PloAction, ...],
    hero_seat: str,
    *,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
) -> tuple[float, float]:
    """``(pot_bb, to_call_bb)`` at hero's decision -- the raw pot-odds inputs.

    Runs the SAME walk as :func:`resolve_pot_limit` (blinds + any bb ante
    posted, pot-limit raise sizes, per-seat commitments) and reads off what
    hero must add to call: the current high bet minus hero's own committed
    chips (a blind counts; an ante does NOT -- it is dead money hero is
    priced to win, not part of anyone's bet). ``to_call`` is 0 when hero is
    first in unraised (the BB option). Break-even equity =
    ``to_call / (pot + to_call)`` -- the "show the math" numbers for the
    SOLVER DATA block and the stat_notes column.

    A view over :func:`_walk` (July 23 2026). This function used to carry
    its OWN copy of the walk, which missed the July-22 min-raise fix: every
    min-raise was still priced as a 100%-pot raise, so the prose said
    "opens to 2bb" while the price block computed the 3.5bb-open line (109
    shipped 20bb-pack questions). See the invariant on :func:`_walk`.
    """
    _actions, pot, committed, high_bet = _walk(
        history, stack_bb=stack_bb, ante_bb=ante_bb
    )
    return pot, max(0.0, high_bet - committed.get(hero_seat, 0.0))


def display_call_price(
    history: tuple[PloAction, ...],
    hero_seat: str,
    *,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
) -> tuple[float, float, float]:
    """The pot-odds price AS THE PLAYER SEES IT: ``(pot_bb, to_call_bb,
    break_even)`` on the 0.5bb display grid.

    Every player-facing surface rounds bb amounts to the 0.5bb display grid
    (Question prose sizes, the POT column, the app chips), so the pot-odds
    math must quote THOSE numbers or the panel reads as wrong (user report,
    July 23 2026: the question said "open to 3.5bb ... 3-bets to 11.5bb"
    while the note said "facing 7.9bb into the 16.4bb pot" -- the exact
    resolved sizes are 3.52/11.38). ``to_call`` is the subtraction the
    player does from the DISPLAYED sizes (rounded high bet minus hero's
    rounded commitment -- NOT the rounded exact difference, which can land
    0.5bb away from it); the pot matches the displayed POT column (rounded
    exact pot); break-even is computed FROM the two displayed numbers so
    the printed equation is self-checking.

    Display-only: worthiness, difficulty, EV and every strategic gate keep
    the exact :func:`call_price`. INVARIANT: the SOLVER DATA price block
    and the CSV pot_odds stat_note MUST both read THIS function -- they
    used to quote the exact amounts and visibly disagreed with the
    question on every off-grid (pot-relative MTT) size.
    """
    _actions, pot, committed, high_bet = _walk(
        history, stack_bb=stack_bb, ante_bb=ante_bb
    )
    pot_disp = round_to_half_bb(pot)
    to_call_disp = max(
        0.0,
        round_to_half_bb(high_bet)
        - round_to_half_bb(committed.get(hero_seat, 0.0)),
    )
    break_even = (
        to_call_disp / (pot_disp + to_call_disp) if to_call_disp > 0 else 0.0
    )
    return pot_disp, to_call_disp, break_even


def price_is_live(history: tuple[PloAction, ...], hero_seat: str) -> bool:
    """Whether pot-odds price facts APPLY at hero's decision.

    True when hero faces a raise/jam, or is the SB deciding whether to
    complete first-in (a genuine priced call). False for every other open
    decision: a non-SB seat technically "owes" the 1bb blind, but the tree
    offers no first-in call, so a price there would push pot-odds framing
    onto a pure open/fold choice. ONE rule shared by the SOLVER DATA block
    and the CSV writer (pot_odds / stat_notes), so they can never disagree.
    """
    facing_aggression = any(
        a.action in (
            PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN
        )
        for a in history
    )
    return facing_aggression or (not facing_aggression and hero_seat == "SB")


__all__ = [
    "ResolvedAction",
    "call_price",
    "display_call_price",
    "display_seat",
    "format_plo_action_history",
    "format_plo_context",
    "pot_bb",
    "resolve_pot_limit",
]
