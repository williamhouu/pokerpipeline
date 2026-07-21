"""Layer 8 helper: the Runout app's "table-state" CSV columns for PLO.

Builds the seven table-state columns in the app's poker-table format so the CSV
feeds the chip/seat renderer directly. A PLO port of
:mod:`pipeline.preflop.app_table_format` (itself a port of the team's
``gto-formatter`` engine), driven by :func:`pipeline.plo.action_history.
resolve_pot_limit` so the chip tokens always agree with the Question prose + pot.

Columns:

    User Seat       POS-$<remaining>[-$<amount>-<action>]
    User Cards      RANK-suitword, RANK-suitword, RANK-suitword, RANK-suitword
    Cards on Table  (empty preflop)
    Table Size      bare int (6)
    Default Stack   $<n> (cash) / <n>BB (bb display)
    Seats           comma-separated villain tokens, same grammar as User Seat
    Pot             $<n> (cash) / <n>BB

Format conventions copied from the engine: cash remaining stacks round to whole
dollars (half-up), wagers keep cents; a level-1 raise's verb is ``raise`` (the
prose says "open"); blinds are always shown; a non-blind that simply folded is
omitted; seats sort ascending by committed amount.
"""

from __future__ import annotations

import math
from typing import Any

from pipeline.bb_display import round_to_half_bb  # 0.5bb display grid (leaf)
from pipeline.plo.action_history import display_seat, resolve_pot_limit
from pipeline.plo.fact_extractor import PloFacts

_SUIT_WORD = {"s": "spades", "h": "hearts", "d": "diamonds", "c": "clubs"}
# Seat layouts in preflop acting order, per table size (pack-internal codes;
# mirrors pack.SEATS / pack.SEATS_9MAX -- kept literal here so this module
# stays a formatting leaf).
_POSITIONS_BY_TABLE: dict[int, tuple[str, ...]] = {
    6: ("LJ", "HJ", "CO", "BU", "SB", "BB"),
    9: ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"),
}
_RAISE_VERBS = frozenset({"open", "3-bet", "4-bet", "5-bet", "raise"})
_SB_BB = 0.5
_BB = 1.0


def _round_half_up(x: float) -> int:
    """JS ``Math.round`` (round half up), so remaining-stack dollars match."""
    return math.floor(x + 0.5)


def _trim_num(n: float) -> str:
    """``6.0`` -> ``'6'``, ``3.5`` -> ``'3.5'``, ``1.25`` -> ``'1.25'``."""
    n = round(n, 2)
    if n == int(n):
        return str(int(n))
    return f"{n:.2f}".rstrip("0").rstrip(".")


def _fmt_bb(n: float, *, allow_zero: bool = False) -> str:
    if n == 0 and not allow_zero:
        return ""
    # Snap to the 0.5bb display grid so the app tokens read cleanly (display-only).
    return _trim_num(round_to_half_bb(n)) + "BB"


def _fmt_chips(chips: float, *, render_bb: bool) -> str:
    """``0`` -> ``''`` (omitted); cash -> ``$<n>``; bb -> ``<n>BB``."""
    if chips == 0:
        return ""
    if render_bb:
        return _fmt_bb(chips)
    return "$" + _trim_num(chips)


def _format_user_cards(cards: tuple[str, ...]) -> str:
    """``('Ac','Ad','4h','8h')`` -> ``'A-clubs, A-diamonds, 4-hearts, 8-hearts'``."""
    return ", ".join(f"{c[0]}-{_SUIT_WORD[c[1].lower()]}" for c in cards)


def build_plo_app_table_columns(
    facts: PloFacts,
    *,
    stakes_bb_dollars: float = 1.0,
    game_format: str = "cash",
    display_in_bb: bool = False,
    stack_bb: float = 100.0,
    table_size: int = 6,
    ante_bb: float = 0.0,
) -> dict[str, str]:
    """Build the seven table-state columns for one PLO spot.

    Returns a dict keyed ``user_seat``, ``user_cards``, ``cards_on_table``,
    ``table_size``, ``default_stack``, ``seats``, ``pot``.
    """
    render_bb = display_in_bb or game_format != "cash"
    mult = 1.0 if render_bb else stakes_bb_dollars  # bb -> display unit
    actions, _pot = resolve_pot_limit(
        facts.spot.node.history_before, stack_bb=stack_bb, ante_bb=ante_bb
    )
    hero_pos = facts.spot.node.actor
    positions = _POSITIONS_BY_TABLE[table_size]

    sb_amt = _SB_BB * mult
    bb_amt = _BB * mult
    stack_chips = stack_bb * mult

    # --- per-seat commitment + last action ------------------------------
    money_in: dict[str, float] = {"SB": sb_amt, "BB": bb_amt}
    info: dict[str, dict[str, Any]] = {}
    for act in actions:
        pos = act.seat
        amount = (act.to_bb or 0.0) * mult
        if act.verb in _RAISE_VERBS:
            money_in[pos] = amount
            info[pos] = {
                "action": "raise" if act.verb == "open" else act.verb,
                "amount": amount,
                "folded": False,
                "all_in": False,
            }
        elif act.verb == "all-in":
            money_in[pos] = amount
            info[pos] = {"action": "all-in", "amount": amount, "folded": False, "all_in": True}
        elif act.verb == "call":
            call_to = max(money_in.values(), default=0.0)
            money_in[pos] = call_to
            info[pos] = {"action": "call", "amount": call_to, "folded": False, "all_in": False}
        else:  # fold
            prior = info.get(pos)
            if prior is not None and prior["amount"] > 0:
                prior["folded"] = True  # acted, then folded to a re-raise
            else:
                info[pos] = {"action": "FOLD", "amount": 0.0, "folded": True, "all_in": False}

    def _fmt(chips: float) -> str:
        return _fmt_chips(chips, render_bb=render_bb)

    def _remaining(invested: float, pos: str) -> str:
        # The BB's ante (MTT packs) is dead money: not in money_in (it is
        # not callable) but gone from the BB's stack all the same.
        dead = ante_bb * mult if pos == "BB" else 0.0
        if render_bb:
            return _fmt_bb(stack_chips - invested - dead, allow_zero=True)
        return "$" + str(_round_half_up(stack_chips - invested - dead))

    # --- hero seat (User Seat) ------------------------------------------
    # Tokens carry the NLHE/app seat codes (UTG/BTN), not the pack's LJ/BU.
    hero_info = info.get(hero_pos)
    user_seat = (
        f"{display_seat(hero_pos, table_size=table_size)}"
        f"-{_remaining(money_in.get(hero_pos, 0.0), hero_pos)}"
    )
    if hero_info is not None and not hero_info["folded"] and hero_info["amount"] > 0:
        user_seat += "-" + _fmt(hero_info["amount"]) + "-" + hero_info["action"]
    elif hero_pos == "SB":
        user_seat += "-" + _fmt(sb_amt)
    elif hero_pos == "BB":
        user_seat += "-" + _fmt(bb_amt)

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
            continue
        is_behind_hero = positions.index(pos) > hero_idx
        if pos_info is None and not is_blind and not is_behind_hero:
            continue

        seat_str = (
            f"{display_seat(pos, table_size=table_size)}"
            f"-{_remaining(money_in.get(pos, 0.0), pos)}"
        )
        sort_amt = 0.0
        if pos_info is not None:
            if pos_info["action"] == "FOLD":
                blind_amt = sb_amt if pos == "SB" else (bb_amt if pos == "BB" else 0.0)
                if blind_amt > 0:
                    seat_str += "-" + _fmt(blind_amt)
                seat_str += "-FOLD"
                sort_amt = blind_amt
            elif pos_info["folded"] and pos_info["amount"] > 0:
                seat_str += "-" + _fmt(pos_info["amount"]) + "-" + pos_info["action"] + "-FOLD"
                sort_amt = pos_info["amount"]
            elif pos_info["all_in"]:
                seat_str += "-" + _fmt(pos_info["amount"]) + "-all-in"
                sort_amt = pos_info["amount"]
            elif pos_info["amount"] > 0:
                seat_str += "-" + _fmt(pos_info["amount"]) + "-" + pos_info["action"]
                sort_amt = pos_info["amount"]
        elif is_blind:
            blind_amt = sb_amt if pos == "SB" else bb_amt
            seat_str += "-" + _fmt(blind_amt)
            sort_amt = blind_amt
        entries.append((sort_amt, seat_str))

    entries.sort(key=lambda e: e[0])
    seats = ", ".join(seat_str for _amt, seat_str in entries)

    pot_chips = sum(money_in.values()) + ante_bb * mult
    pot = _fmt_bb(pot_chips, allow_zero=True) if render_bb else "$" + _trim_num(pot_chips)
    default_stack = (
        _fmt_bb(stack_chips, allow_zero=True) if render_bb else "$" + _trim_num(stack_chips)
    )

    return {
        "user_seat": user_seat,
        "user_cards": _format_user_cards(facts.spot.hero_cards),
        "cards_on_table": "",  # preflop: no board
        "table_size": str(table_size),
        "default_stack": default_stack,
        "seats": seats,
        "pot": pot,
    }


__all__ = ["build_plo_app_table_columns"]
