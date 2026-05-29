"""Layer 8 helper: the 7 "table-state" CSV columns in the Runout app's
exact format, so the pipeline CSV feeds the app's poker-table renderer
(seats + chips + actions) directly.

Columns produced (B-H in the team's template):

    User Seat       POS-$<remaining>[-$<amount>-<action>]
    User Cards      RANK-suitword, RANK-suitword
    Cards on Table  RANK-suitword, ...  (empty preflop)
    Table Size      bare int (6 / 8 / 9)
    Default Stack   $<n> (cash) / <n>BB (tournament)
    Seats           comma-separated villain tokens, same grammar as User Seat
    Pot             $<n> (cash) / <n>BB (tournament)

The team's `gto-formatter` engine (public/formatter-engine.js) produces
this format by REGEX-PARSING the LLM's prose. We already hold the same
facts structured, so we build the output directly -- no prose round-trip.
This module is a faithful port of that engine's cash/tournament preflop
path (`parseQuestion` build-hero-seat + build-seats blocks), driven by the
resolved dollar amounts that :func:`pipeline.preflop.action_history.
build_hand_dict` computes (the SAME amounts the Question column renders,
so the table tokens never disagree with the prose or the pot).

Format conventions copied from the engine, exactly:

  * **Cash** -- remaining stacks are rounded to whole dollars (JS
    ``Math.round``, i.e. half-up); wager amounts keep cents (``$1.25``).
    A zero amount renders as the empty string (so it's omitted).
  * **Seat token** -- ``POS-$<remaining>`` plus, when the seat has acted,
    ``-$<amount>-<action>``. ``<amount>`` is the total committed that
    street (raise-TO), not the increment.
  * **Action verbs** -- raises are leveled: 1st raise = ``raise`` (open),
    2nd = ``3-bet``, 3rd = ``4-bet``, 4th = ``5-bet``, 5+ = ``raise``.
    Other verbs: ``call``, ``all-in``, ``check``, and ``FOLD`` (the only
    upper-cased verb).
  * **Seats inclusion** -- hero is excluded (he's the User Seat). Blinds
    are ALWAYS shown (with their posted blind, plus ``-FOLD`` if folded).
    Non-blind seats are shown only if they took a non-fold action OR are
    still to act behind hero; a non-blind that simply folded is omitted.
  * **Seats order** -- ascending by the seat's action amount (the JS
    ``sortAmt`` sort).
  * **Pot** -- the sum of every seat's committed chips (blinds are dead
    money even when folded), plus antes in tournaments.

Tournament mode mirrors the engine's BB-unit path; the Tier-1 packs are
cash, so cash is the thoroughly-exercised path.
"""

from __future__ import annotations

import math
from typing import Any

from pipeline.preflop.action_history import build_hand_dict
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.pack import PreflopPack

# Suit letter (as stored in hero_card_combo, e.g. "AhKc") -> the engine's
# suit word.
_SUIT_WORD = {"s": "spades", "h": "hearts", "d": "diamonds", "c": "clubs"}

# Table position order by seat count (mirrors POSITIONS_6/8/9 in the JS
# engine). Drives seat ordering + the "behind hero" test.
_POSITIONS: dict[int, tuple[str, ...]] = {
    6: ("UTG", "HJ", "CO", "BTN", "SB", "BB"),
    8: ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"),
    9: ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"),
}

# Raise level (1 = open) -> verb. 5+ falls back to "raise".
_RAISE_VERB: dict[int, str] = {1: "raise", 2: "3-bet", 3: "4-bet", 4: "5-bet"}


def _round_half_up(x: float) -> int:
    """Match JS ``Math.round`` (round half UP), unlike Python's
    round-half-to-even, so remaining-stack dollars agree with the engine."""
    return math.floor(x + 0.5)


def _trim_num(n: float) -> str:
    """A number as the JS ``"" + n`` would render it: integers lose the
    decimal ("6"), fractions keep up to 2 dp with trailing zeros trimmed
    ("1.25", "7.5"). Amounts are pre-rounded to 2 dp upstream."""
    n = round(n, 2)
    if n == int(n):
        return str(int(n))
    return f"{n:.2f}".rstrip("0").rstrip(".")


def _fmt_chips(chips: float, *, is_tournament: bool, bb_unit: float) -> str:
    """The engine's ``fmtAmt``: 0 -> "" (omitted); cash -> ``$<n>``;
    tournament -> ``<bb>BB``."""
    if chips == 0:
        return ""
    if is_tournament:
        return _fmt_bb(chips / bb_unit)
    return "$" + _trim_num(chips)


def _fmt_bb(n: float, *, allow_zero: bool = False) -> str:
    """The engine's ``fmtBB``: ``<n>BB`` with trimmed decimals; empty for
    zero unless ``allow_zero``."""
    if n == 0 and not allow_zero:
        return ""
    return _trim_num(n) + "BB"


def _format_user_cards(combo: str) -> str:
    """Hero's two cards as ``RANK-suitword, RANK-suitword`` (e.g.
    ``A-spades, T-diamonds``). ``combo`` is the 4-char ``rank+suit`` x2
    form (``AsTd``); ten is already the ``T`` token."""
    if len(combo) != 4:
        raise ValueError(f"combo must be 4 chars, got {combo!r}")
    first, second = combo[:2], combo[2:]
    return (
        f"{first[0]}-{_SUIT_WORD[first[1].lower()]}, "
        f"{second[0]}-{_SUIT_WORD[second[1].lower()]}"
    )


def build_app_table_columns(
    facts: PreflopFacts,
    pack: PreflopPack,
    *,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
) -> dict[str, str]:
    """Build the 7 table-state CSV columns for one preflop spot.

    Returns a dict with keys ``user_seat``, ``user_cards``,
    ``cards_on_table``, ``table_size``, ``default_stack``, ``seats``,
    ``pot`` -- each a ready-to-write CSV string in the app's format.

    Reuses :func:`pipeline.preflop.action_history.build_hand_dict` for
    the resolved per-action dollar amounts so these tokens always agree
    with the Question prose and the pot.
    """
    hand = build_hand_dict(
        facts,
        pack=pack,
        stakes_bb_dollars=stakes_bb_dollars,
        live_or_online=live_or_online,
        game_format=game_format,
        display_in_bb=display_in_bb,
    )
    # Render amounts in bb when it's a tournament OR the caller asked for
    # bb display on a cash game. (`is_tournament` below is really the
    # "render in bb" switch -- the unit, not the game type.)
    is_tournament = display_in_bb or game_format != "cash"
    # Amounts in `hand` are already in the display unit (dollars or bb --
    # see build_hand_dict), so the bb_unit divisor is 1.0; bb formatting
    # still routes through fmtBB.
    bb_unit = 1.0
    # Blinds must be in the SAME unit as the action amounts: dollars for
    # dollar display (from the stakes config), big blinds for bb display
    # (where the amounts in `hand` are already bb). hand["stakes"] is
    # always dollars, so don't read it for the bb path.
    if is_tournament:
        bb_amt = 1.0
        sb_amt = round(pack.sb_to_bb_ratio, 2)
    else:
        sb_amt = hand["stakes"]["sb"]
        bb_amt = hand["stakes"]["bb"]
    table_size: int = hand["table_size"]
    positions = _POSITIONS.get(table_size, _POSITIONS[6])
    hero_pos: str = hand["hero_position"]
    stack_chips: float = hand["effective_stack"]

    # --- reconstruct per-seat commitments + last action -----------------
    # money_in[pos] = total chips committed (raise-TO replaces; blinds are
    # seeded as dead money). info[pos] = last action descriptor.
    money_in: dict[str, float] = {}
    if sb_amt:
        money_in["SB"] = sb_amt
    if bb_amt:
        money_in["BB"] = bb_amt
    info: dict[str, dict[str, Any]] = {}
    raise_level = 0
    for action in hand["preflop_actions"]:
        pos = action[0]
        verb = action[1]
        amount = float(action[2]) if len(action) > 2 else 0.0
        if verb == "raise":
            raise_level += 1
            money_in[pos] = amount
            info[pos] = {
                "action": _RAISE_VERB.get(raise_level, "raise"),
                "amount": amount,
                "folded": False,
                "all_in": False,
            }
        elif verb == "all-in":
            money_in[pos] = amount
            info[pos] = {
                "action": "all-in", "amount": amount,
                "folded": False, "all_in": True,
            }
        elif verb == "call":
            call_to = max(money_in.values(), default=0.0)
            money_in[pos] = call_to
            info[pos] = {
                "action": "call", "amount": call_to,
                "folded": False, "all_in": False,
            }
        elif verb == "fold":
            prior = info.get(pos)
            if prior is not None and prior["amount"] > 0:
                # Acted (e.g. raised) then folded to a re-raise.
                prior["folded"] = True
            else:
                info[pos] = {
                    "action": "FOLD", "amount": 0.0,
                    "folded": True, "all_in": False,
                }
        # build_hand_dict emits no check/limp tokens for preflop spots.

    fmt = lambda c: _fmt_chips(  # noqa: E731
        c, is_tournament=is_tournament, bb_unit=bb_unit)

    def _remaining(invested: float) -> str:
        if is_tournament:
            return _fmt_bb(stack_chips - invested / bb_unit, allow_zero=True)
        return "$" + str(_round_half_up(stack_chips - invested))

    # --- hero seat (User Seat) ------------------------------------------
    hero_info = info.get(hero_pos)
    user_seat = f"{hero_pos}-{_remaining(money_in.get(hero_pos, 0.0))}"
    hero_acted = (
        hero_info is not None
        and not hero_info["folded"]
        and hero_info["action"] != "check"
        and hero_info["amount"] > 0
    )
    if hero_acted:
        assert hero_info is not None
        user_seat += "-" + fmt(hero_info["amount"]) + "-" + hero_info["action"]
    elif hero_pos == "SB":
        user_seat += "-" + fmt(sb_amt)
    elif hero_pos == "BB":
        user_seat += "-" + fmt(bb_amt)

    # --- other seats (Seats) --------------------------------------------
    hero_idx = positions.index(hero_pos)
    entries: list[tuple[float, str]] = []
    for pos in positions:
        if pos == hero_pos:
            continue
        is_blind = pos in ("SB", "BB")
        pos_info = info.get(pos)
        silently_folded = pos_info is not None and pos_info["action"] == "FOLD"
        if silently_folded and not is_blind:
            continue  # a non-blind that simply folded -- nothing to render
        is_behind_hero = positions.index(pos) > hero_idx
        if pos_info is None and not is_blind and not is_behind_hero:
            continue  # folded earlier / not involved

        seat_str = f"{pos}-{_remaining(money_in.get(pos, 0.0))}"
        sort_amt = 0.0
        if pos_info is not None:
            if pos_info["action"] == "FOLD":
                blind_amt = sb_amt if pos == "SB" else (
                    bb_amt if pos == "BB" else 0.0)
                if blind_amt > 0:
                    seat_str += "-" + fmt(blind_amt)
                seat_str += "-FOLD"
                sort_amt = blind_amt
            elif pos_info["folded"] and pos_info["amount"] > 0:
                seat_str += (
                    "-" + fmt(pos_info["amount"])
                    + "-" + pos_info["action"] + "-FOLD"
                )
                sort_amt = pos_info["amount"]
            elif pos_info["action"] == "check":
                seat_str += "-" + ("0BB" if is_tournament else "$0") + "-check"
            elif pos_info["all_in"]:
                seat_str += "-" + fmt(pos_info["amount"]) + "-all-in"
                sort_amt = pos_info["amount"]
            elif pos_info["amount"] > 0:
                seat_str += "-" + fmt(pos_info["amount"]) + "-" + pos_info["action"]
                sort_amt = pos_info["amount"]
        elif is_blind:
            blind_amt = sb_amt if pos == "SB" else bb_amt
            seat_str += "-" + fmt(blind_amt)
            sort_amt = blind_amt
        entries.append((sort_amt, seat_str))

    entries.sort(key=lambda e: e[0])
    seats = ", ".join(seat_str for _amt, seat_str in entries)

    # --- pot + the trivial columns --------------------------------------
    pot_chips = sum(money_in.values())
    ante = hand.get("ante") or 0
    if is_tournament and ante:
        pot_chips += ante * table_size
    pot = (_fmt_bb(pot_chips / bb_unit, allow_zero=True)
           if is_tournament else "$" + _trim_num(pot_chips))

    default_stack = (
        _fmt_bb(stack_chips, allow_zero=True)
        if is_tournament else "$" + _trim_num(stack_chips)
    )
    return {
        "user_seat": user_seat,
        "user_cards": _format_user_cards(facts.spot.hero_card_combo),
        "cards_on_table": "",  # preflop: no board has come yet

        "table_size": str(table_size),
        "default_stack": default_stack,
        "seats": seats,
        "pot": pot,
    }


__all__ = ["build_app_table_columns"]
