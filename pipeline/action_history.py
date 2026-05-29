"""Action history & context formatting (deterministic, no LLM).

Turns one structured hand dict into the two text blocks that precede every
training question:

  * the context block  -- e.g. "$1/$2 Online cash. 6-handed. $200 effective stacks."
  * the action history -- e.g. "You're on the Button with <hole cards>.
                                You open to $5. The Big Blind calls. ..."

See docs/engineering_brief.docx, "Action History & Context Format
Specification". The format is fully deterministic: identical input always
yields identical output, with zero LLM involvement.

Pot tracking sums blinds, antes, and every chip committed (including by
players who later fold). `hand["ante"]` is read as the *total* chips the antes
contribute to the pot -- a standard ante and a big-blind ante are equivalent
for pot math (per the brief), so the caller passes the already-summed total.
"""
from __future__ import annotations

from pipeline.cards import parse_card

# Position phrases (brief, "Position phrases"). UTG/UTG+1/UTG+2 take no article.
_HERO_PHRASE = {
    "UTG": "UTG", "UTG+1": "UTG+1", "UTG+2": "UTG+2",
    "LJ": "in the Lojack", "HJ": "in the Hijack", "CO": "in the Cutoff",
    "BTN": "on the Button", "SB": "in the Small Blind", "BB": "in the Big Blind",
}
_VILLAIN_REF = {
    "UTG": "UTG", "UTG+1": "UTG+1", "UTG+2": "UTG+2",
    "LJ": "The Lojack", "HJ": "The Hijack", "CO": "The Cutoff",
    "BTN": "The Button", "SB": "The Small Blind", "BB": "The Big Blind",
}

# Preflop position order, longest to shortest. A table shrinks by dropping the
# early seats (LJ, then UTG+1, then UTG+2); UTG and the last five always stay.
_FULL_ORDER = ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB")

# Preflop raise levels (brief, "Action verbs"): 1 open, 2 3-bet, 3 4-bet,
# 4 5-bet, 5+ fall back to a plain "raise".
_RAISE_LEVEL = {1: "open", 2: "3-bet", 3: "4-bet", 4: "5-bet"}
_RAISE_VERBS = frozenset({"open", "3-bet", "4-bet", "5-bet", "raise"})

# Card suit -> emoji: the suit codepoint followed by variation selector 16
# (U+FE0F); hearts uses U+2764 (heavy heart). Brief, "Card notation".
_SUIT_EMOJI = {
    "s": "♠️", "h": "❤️",
    "d": "♦️", "c": "♣️",
}


# --- small formatting helpers ------------------------------------------------
def _num(value) -> str:
    """A number as a comma-grouped string: 1000 -> '1,000', 2.5 -> '2.5'."""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,}"
    return f"{int(value):,}"


def _chips(amount, is_tournament: bool) -> str:
    """A chip amount: '$50' for cash, '25bb' for tournaments.

    Non-integer cash amounts (e.g. $0.23, $4.50) render with 2-decimal
    precision so a chip-to-dollar conversion in `scenario_config.spot_to_hand`
    doesn't produce stray "$0.5" / "$1.25" inconsistencies. Integer dollars
    stay as "$50" -- matches the brief's worked examples 1-10.
    """
    if is_tournament:
        return f"{_num(amount)}bb"
    if isinstance(amount, float) and not amount.is_integer():
        return f"${amount:,.2f}"
    return f"${int(amount):,}"


def format_card(card: str) -> str:
    """One card as rank + suit emoji, e.g. 'Th' -> 'T<heart>'."""
    norm = parse_card(card)
    return norm[0] + _SUIT_EMOJI[norm[1]]


def preflop_order(table_size: int) -> list[str]:
    """The seats, in preflop action order, for a table of this size (6-9)."""
    present = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}
    if table_size >= 7:
        present.add("LJ")
    if table_size >= 8:
        present.add("UTG+1")
    if table_size >= 9:
        present.add("UTG+2")
    return [p for p in _FULL_ORDER if p in present]


# --- action verb conjugation -------------------------------------------------
def _conjugate(base: str, is_hero: bool) -> str:
    """Hero uses the base verb; a villain uses the third-person form."""
    if is_hero:
        return base
    return "moves all-in" if base == "move all-in" else base + "s"


def _format_action(player, verb, amount, hero_pos, is_preflop,
                    is_tournament, raise_counter) -> str:
    """Format one action as its own sentence (ending with a period)."""
    is_hero = player == hero_pos
    subject = "You" if is_hero else _VILLAIN_REF[player]

    if verb in _RAISE_VERBS:
        if is_preflop:
            raise_counter[0] += 1
            word = _RAISE_LEVEL.get(raise_counter[0], "raise")
        else:
            word = "raise"                       # postflop raises take no level
        return f"{subject} {_conjugate(word, is_hero)} to {_chips(amount, is_tournament)}."
    if verb == "bet":
        return f"{subject} {_conjugate('bet', is_hero)} {_chips(amount, is_tournament)}."
    if verb == "all-in":
        return (f"{subject} {_conjugate('move all-in', is_hero)} "
                f"for {_chips(amount, is_tournament)}.")
    if verb == "limp":
        if not is_preflop:
            raise ValueError("limp is a preflop-only action")
        return f"{subject} {_conjugate('limp', is_hero)}."
    if verb in ("call", "fold", "check"):
        return f"{subject} {_conjugate(verb, is_hero)}."
    raise ValueError(f"unknown action verb: {verb!r}")


def _format_street(actions, hero_pos, is_preflop, is_tournament, raise_counter) -> str:
    """Join a street's actions into one line. Preflop folds are dropped; postflop
    folds are kept (they change the pot structure -- brief, "The Fold Rule")."""
    sentences = []
    for act in actions:
        player, verb = act[0], act[1]
        amount = act[2] if len(act) > 2 else None
        if is_preflop and verb == "fold":
            continue
        sentences.append(_format_action(player, verb, amount, hero_pos,
                                         is_preflop, is_tournament, raise_counter))
    return " ".join(sentences)


# --- pot tracking ------------------------------------------------------------
def _preflop_pot(hand) -> float:
    """Chips in the middle at the start of the flop: blinds + antes + all bets."""
    sb, bb = hand["stakes"]["sb"], hand["stakes"]["bb"]
    contrib = {"SB": sb, "BB": bb}               # each seat's running total
    for act in hand.get("preflop_actions") or []:
        player, verb = act[0], act[1]
        amount = act[2] if len(act) > 2 else None
        if verb in _RAISE_VERBS or verb == "all-in":
            contrib[player] = amount             # raises state a "to" total
        elif verb == "call":
            contrib[player] = max(contrib.values())
        elif verb == "limp":
            contrib[player] = bb
    return sum(contrib.values()) + (hand.get("ante") or 0)


def _street_added(actions) -> float:
    """Chips added on a postflop street, to roll the pot forward."""
    contrib: dict = {}
    for act in actions or []:
        player, verb = act[0], act[1]
        amount = act[2] if len(act) > 2 else None
        if verb in _RAISE_VERBS or verb in ("bet", "all-in"):
            contrib[player] = amount
        elif verb == "call":
            contrib[player] = max(contrib.values()) if contrib else 0
    return sum(contrib.values())


# --- validation --------------------------------------------------------------
def _validate(hand) -> None:
    """Catch the structural errors seen in past hand data before formatting."""
    table_size = hand["table_size"]
    if not 6 <= table_size <= 9:
        raise ValueError(f"table_size must be 6-9, got {table_size}")
    seats = set(preflop_order(table_size))
    if hand["hero_position"] not in seats:
        raise ValueError(f"{hand['hero_position']!r} is not a seat at a "
                          f"{table_size}-handed table")
    if hand["format"] not in ("cash", "tournament"):
        raise ValueError(f"format must be 'cash' or 'tournament', "
                          f"got {hand['format']!r}")
    for key in ("preflop_actions", "flop_actions", "turn_actions", "river_actions"):
        for act in hand.get(key) or []:
            if act[0] not in seats:
                raise ValueError(f"{act[0]!r} is not a seat at a "
                                 f"{table_size}-handed table")
            if len(act) > 2 and act[2] is not None and act[2] <= 0:
                raise ValueError(f"non-positive bet amount in action {act!r}")

    cards = list(hand["hero_cards"])
    board = hand.get("board") or {}
    if board.get("flop"):
        cards += list(board["flop"])
    if board.get("turn"):
        if not board.get("flop"):
            raise ValueError("board has a turn card but no flop")
        cards.append(board["turn"])
    if board.get("river"):
        if not board.get("turn"):
            raise ValueError("board has a river card but no turn")
        cards.append(board["river"])
    normalised = [parse_card(c) for c in cards]
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"duplicate cards in hand: {sorted(normalised)}")


# --- public API --------------------------------------------------------------
def format_context(hand) -> str:
    """The context block: stakes, table size, effective stacks."""
    _validate(hand)
    table_size = hand["table_size"]
    if hand["format"] == "cash":
        sb, bb = hand["stakes"]["sb"], hand["stakes"]["bb"]
        venue = hand["venue"].capitalize()
        return (f"${_num(sb)}/${_num(bb)} {venue} cash. {table_size}-handed. "
                f"${_num(hand['effective_stack'])} effective stacks.")
    return (f"${_num(hand['buy_in'])} {hand['stage']} tournament. "
            f"{table_size}-handed. "
            f"{_num(hand['effective_stack'])}bb effective stacks.")


def _street_block(name, pot, cards, actions, hero, is_tournament) -> str:
    """One postflop street: the header line, then its action line if any."""
    header = (f"{name} ({_chips(pot, is_tournament)}): "
              + "".join(format_card(c) for c in cards))
    body = _format_street(actions, hero, False, is_tournament, [0])
    return f"{header}\n{body}" if body else header


def format_action_history(hand) -> str:
    """The action history block: position, hole cards, and the action sequence."""
    _validate(hand)
    # `is_tournament` here really means "render amounts in bb (with a 'bb'
    # suffix) rather than dollars". Tournaments always do; a cash game can
    # opt in via hand["display_unit"] == "bb" (the admin "Display amounts
    # as: Big blinds" toggle) without changing its cash semantics.
    is_tournament = (
        hand["format"] == "tournament"
        or hand.get("display_unit") == "bb"
    )
    hero = hand["hero_position"]
    raise_counter = [0]                          # shared across the preflop street

    hole = "".join(format_card(c) for c in hand["hero_cards"])
    preflop_section = f"You're {_HERO_PHRASE[hero]} with {hole}."
    preflop = _format_street(hand.get("preflop_actions") or [], hero, True,
                             is_tournament, raise_counter)
    if preflop:
        preflop_section += "\n" + preflop
    sections = [preflop_section]

    board = hand.get("board") or {}
    flop, turn, river = board.get("flop"), board.get("turn"), board.get("river")
    if flop:
        pot = _preflop_pot(hand)
        sections.append(_street_block("Flop", pot, flop,
                                      hand.get("flop_actions") or [],
                                      hero, is_tournament))
        if turn:
            pot += _street_added(hand.get("flop_actions") or [])
            sections.append(_street_block("Turn", pot, [turn],
                                          hand.get("turn_actions") or [],
                                          hero, is_tournament))
            if river:
                pot += _street_added(hand.get("turn_actions") or [])
                sections.append(_street_block("River", pot, [river],
                                              hand.get("river_actions") or [],
                                              hero, is_tournament))
    return "\n\n".join(sections)


def format_hand(hand) -> tuple[str, str]:
    """Format both blocks at once: (context, action_history)."""
    return format_context(hand), format_action_history(hand)
