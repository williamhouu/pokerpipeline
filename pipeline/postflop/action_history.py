"""Deterministic Context + Question prose for a postflop spot.

This is the layer that makes a postflop question self-contained: it renders
the ENTIRE line that came before the decision, across every street. A turn
question shows the preflop action AND the flop action AND the turn card before
asking hero to act; a river question shows preflop + flop + turn + river. No
LLM is involved -- this is pure Python from the structured solve (positions,
the preflop summary, the per-node postflop history, and the board), so it is
zero-variance and trivially testable.

Output is two pieces (both consumed by the format writer):

* :func:`build_context_line` -- the short framing line, e.g.
  ``"Online · 6-max · $0.50/$1"``.
* :func:`format_question` -- the narrative, e.g. for a BTN c-bet spot::

      You're on the Button with A♣️J♣️.
      You open to 2.5bb and the Big Blind calls.
      The flop is 2♣️ J♠️ 7♠️. The Big Blind checks.

  and for a turn spot (hero = BB) the flop action is fully rendered first::

      You're in the Big Blind with J❤️9♣️.
      The Button opens to 2.5bb and you call.
      The flop is 2♣️ J♠️ 7♠️. You check, the Button bets
      1.8bb, and you call.
      The turn is 2❤️.

Amounts are rendered in big blinds (the IR is bb-denominated); the format
writer renders the dollar/seat table-state columns separately.
"""

from __future__ import annotations

from pipeline.action_history import format_card  # pure, game-agnostic leaf
from pipeline.postflop.solve import (
    POSTFLOP_VERBS,
    PostflopSolve,
    PostflopStep,
    PreflopStep,
)
from pipeline.postflop.spot_sampler import PostflopSpot

# Where each seat sits in the "You're {phrase}" intro. Button takes "on"; the
# blinds and the named seats take "in"; UTG takes the idiom "under the gun".
_SEAT_INTRO = {
    "BTN": "on the Button",
    "SB": "in the Small Blind",
    "BB": "in the Big Blind",
    "CO": "in the Cutoff",
    "HJ": "in the Hijack",
    "LJ": "in the Lojack",
    "UTG": "under the gun",
    "UTG+1": "in UTG+1",
    "UTG+2": "in UTG+2",
}
# Mid-sentence reference to a non-hero seat ("the Button bets ...").
_SEAT_SUBJECT = {
    "BTN": "the Button",
    "SB": "the Small Blind",
    "BB": "the Big Blind",
    "CO": "the Cutoff",
    "HJ": "the Hijack",
    "LJ": "the Lojack",
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "UTG+2": "UTG+2",
}
_STREET_LABEL = {"flop": "flop", "turn": "turn", "river": "river"}


def _bb(amount: float) -> str:
    """A bb amount: '2.5bb', '1.8bb', '6bb' (drop a trailing '.0')."""
    rounded = round(amount, 2)
    text = f"{rounded:g}"
    return f"{text}bb"


def _cards(combo_or_cards) -> str:
    """Render a 4-char combo or a card iterable as emoji cards (no spaces)."""
    if isinstance(combo_or_cards, str):
        cards = [combo_or_cards[:2], combo_or_cards[2:]]
    else:
        cards = list(combo_or_cards)
    return "".join(format_card(c) for c in cards)


def _board_phrase(cards: list[str]) -> str:
    """Space-separated emoji cards for a board, e.g. '2♣️ J♠️ 7♠️'."""
    return " ".join(format_card(c) for c in cards)


def _subject(position: str, hero: str, *, capitalized: bool) -> str:
    """'You'/'you' for hero, else the seat subject phrase.

    When ``capitalized`` (the subject starts a sentence/clause), the seat
    phrase's leading article is capitalised too ('the Button' -> 'The
    Button'); mid-sentence it stays lowercase.
    """
    if position == hero:
        return "You" if capitalized else "you"
    phrase = _SEAT_SUBJECT.get(position, position)
    return phrase[0].upper() + phrase[1:] if capitalized else phrase


def _preflop_verb_phrase(step: PreflopStep, *, is_hero: bool) -> str:
    """Conjugated preflop action: 'open to 2.5bb' / 'opens to 2.5bb', etc."""
    base = {
        "open": "open", "raise": "raise", "3-bet": "3-bet",
        "4-bet": "4-bet", "5-bet": "5-bet", "call": "call",
        "fold": "fold", "check": "check",
    }.get(step.verb, step.verb)
    verb = base if is_hero else _third_person(base)
    if step.to_bb is not None and step.verb in ("open", "raise", "3-bet", "4-bet", "5-bet"):
        return f"{verb} to {_bb(step.to_bb)}"
    return verb


def _postflop_verb_phrase(step: PostflopStep, *, is_hero: bool) -> str:
    """Conjugated postflop action: 'check' / 'checks', 'bet 1.8bb' / 'bets 1.8bb',
    'call' / 'calls', 'raise to 6bb' / 'raises to 6bb'."""
    if step.verb not in POSTFLOP_VERBS:
        raise ValueError(f"unknown postflop verb {step.verb!r}")
    verb = step.verb if is_hero else _third_person(step.verb)
    if step.verb == "bet" and step.to_bb is not None:
        return f"{verb} {_bb(step.to_bb)}"
    if step.verb == "raise" and step.to_bb is not None:
        return f"{verb} to {_bb(step.to_bb)}"
    return verb


def _third_person(verb: str) -> str:
    """'check' -> 'checks', '3-bet' -> '3-bets', 'raise' -> 'raises'."""
    if verb.endswith(("s", "x", "z")):
        return verb + "es"
    return verb + "s"


def _preflop_sentence(solve: PostflopSolve, hero: str) -> str:
    """The preflop line as one sentence, e.g.
    'You open to 2.5bb and the Big Blind calls.'"""
    clauses: list[str] = []
    for i, step in enumerate(solve.preflop_summary):
        is_hero = step.position == hero
        subject = _subject(step.position, hero, capitalized=(i == 0))
        clauses.append(f"{subject} {_preflop_verb_phrase(step, is_hero=is_hero)}")
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0] + "."
    body = clauses[0]
    for clause in clauses[1:-1]:
        body += f", {clause}"
    body += f" and {clauses[-1]}"
    return body + "."


def _street_cards(solve_flop, board: tuple[str, ...], street: str) -> list[str]:
    """The cards revealed AT a given street (flop = 3 cards, turn/river = 1)."""
    if street == "flop":
        return list(board[:3])
    if street == "turn":
        return [board[3]] if len(board) >= 4 else []
    if street == "river":
        return [board[4]] if len(board) >= 5 else []
    return []


def _street_sentences(spot: PostflopSpot, hero: str) -> list[str]:
    """One sentence per street up to (and including) the decision street.

    Each sentence reveals that street's board card(s) and then renders the
    actions taken on that street BEFORE the decision (from the node history).
    """
    node = spot.node
    board = node.board
    order = ["flop", "turn", "river"]
    sentences: list[str] = []
    for street in order:
        # Stop after the decision street.
        if order.index(street) > order.index(node.street):
            break
        cards = _street_cards(node.board[:3], board, street)
        if not cards:
            continue
        label = _STREET_LABEL[street]
        if street == "flop":
            reveal = f"The flop is {_board_phrase(cards)}."
        else:
            reveal = f"The {label} is {_board_phrase(cards)}."

        steps = [s for s in node.history if s.street == street]
        if steps:
            clauses = [
                f"{_subject(s.position, hero, capitalized=(i == 0))} "
                f"{_postflop_verb_phrase(s, is_hero=(s.position == hero))}"
                for i, s in enumerate(steps)
            ]
            if len(clauses) == 1:
                action = clauses[0] + "."
            else:
                action = clauses[0]
                for clause in clauses[1:-1]:
                    action += f", {clause}"
                action += f", and {clauses[-1]}."
            sentences.append(f"{reveal} {action}")
        else:
            sentences.append(reveal)
    return sentences


def format_question(spot: PostflopSpot, solve: PostflopSolve) -> str:
    """The full multi-street question narrative for ``spot``.

    Renders, in order: hero's seat + cards, the preflop line, then each street
    up to the decision (board reveal + the action taken before hero acts).
    Ends right where hero must act (the options carry "what do you do").
    """
    hero = spot.node.actor
    intro = (
        f"You're {_SEAT_INTRO.get(hero, hero)} with {_cards(spot.hero_combo)}."
    )
    lines = [intro]
    preflop = _preflop_sentence(solve, hero)
    if preflop:
        lines.append(preflop)
    lines.extend(_street_sentences(spot, hero))
    return "\n".join(lines)


def build_context_line(solve: PostflopSolve) -> str:
    """The short framing line, e.g. 'Online · $0.50/$1'.

    Cash shows the stakes, a tournament shows the stack depth instead. The
    table size is intentionally NOT shown -- the dedicated Table Size column
    already carries it, so repeating it in the Context was redundant (dropped
    June 2026 per the team's feedback).
    """
    venue = solve.live_or_online
    if solve.game_format == "tournament":
        tail = f"{solve.effective_stack_bb:g}bb"
    else:
        tail = solve.stakes or "cash"
    return f"{venue} · {tail}"


__all__ = ["build_context_line", "format_card", "format_question"]
