"""Preflop action-history rendering for the Question column.

The brief (docs/engineering_brief.docx "Action History & Context Format
Specification") already has a faithful implementation in
:mod:`pipeline.action_history` -- it drops preflop folds, renders raise
levels (open / 3-bet / 4-bet / 5-bet), conjugates hero vs villain verbs,
formats cards with suit emojis, and handles preflop-only hands when the
``board`` dict is empty.

This module bridges from PreflopFacts to that renderer:

1. :func:`build_hand_dict` converts a PreflopFacts + pack + stake config
   into the brief's input ``hand`` dict shape.
2. :func:`format_preflop_action_history` wraps the conversion + call to
   :func:`pipeline.action_history.format_action_history`.

The one piece that needs invention is the **dollar amount per raise**.
Open size is known from ``pack.open_size_bb``. Subsequent raise sizes
are looked up in :data:`_RYAN_PACK_RAISE_SIZES_BB`, which mirrors the
documented table in ``docs/ryan_range_pack_index.md``. Unknown tokens
fall back to a multiplicative heuristic (3-bet ≈ 3× open, 4-bet ≈ 2×
3-bet, 5-bet ≈ 2× 4-bet) so the renderer never crashes on an
unrecognised token.

The lookup table is Ryan-pack-specific by design; future packs (9-max,
MTT) will need their own entries added (or, ideally, the % token →
bb-size converter generalised once we have more pack data).
"""

from __future__ import annotations

from typing import Any

from pipeline.action_history import format_action_history
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType
from pipeline.preflop.pack import PreflopPack

# Documented Ryan-pack raise sizes. Indexed by (raise_size_pct, raise_level)
# where raise_level is the position of this raise in the sequence: 1=open,
# 2=3-bet, 3=4-bet, 4=5-bet. Sourced from docs/ryan_range_pack_index.md
# "Sizing tokens" and cross-referenced with
# pipeline/scenario_config.py:SCENARIOS (which encodes the same sizes for
# the postflop solves spawned from this pack).
#
# When a token isn't here, `_raise_size_bb` falls back to a multiplicative
# heuristic so generation never crashes — but rare tokens in production
# output will read approximately rather than precisely.
_RYAN_PACK_RAISE_SIZES_BB: dict[tuple[float, int], float] = {
    # OPENS
    (60.0, 1): 2.5,  # UTG/HJ/CO/BTN
    (76.0, 1): 3.0,  # SB BvB open (76% pot in a 1bb-OOP spot ≈ 3bb)
    # 3-BETS
    (77.0, 2): 8.0,  # HJ/CO/BTN 3-bet vs prior open
    (79.0, 2): 9.0,  # 3-bet with a caller between opener and 3-bettor
    (150.0, 2): 10.0,  # SB 3-bet vs BTN open
    (155.0, 2): 10.0,  # BB 3-bet vs UTG open
    (182.0, 2): 12.0,  # BB 3-bet vs HJ/CO/BTN open
    # SQUEEZES (a 3-bet after an open + at least one caller)
    (85.0, 2): 9.0,  # SB squeeze after open + call
    (162.0, 2): 12.0,  # BB squeeze after open + 1 call
    (198.0, 2): 13.0,  # BB squeeze after open + multi calls
    # 4-BETS
    (49.0, 3): 22.0,  # UTG 4-bet over BB's 3-bet
    (50.0, 3): 25.0,  # HJ/CO/BTN 4-bet over 3-bet
    (54.0, 3): 22.0,  # BB 4-bet over SB 3-bet (in 3-way after BTN open)
    (95.0, 3): 28.0,  # CO 4-bet over BTN 3-bet
}


def _raise_size_bb(
    parsed: ParsedAction,
    raise_level: int,
    pack: PreflopPack,
) -> float:
    """Bb amount of one raise action.

    Args:
        parsed: The ParsedAction with the raise_size_pct token.
        raise_level: 1 = open, 2 = 3-bet, 3 = 4-bet, etc.
        pack: Source pack (for the open-size convention).

    Returns:
        Bb size. Uses :data:`_RYAN_PACK_RAISE_SIZES_BB` when the
        ``(pct_token, level)`` pair is registered; otherwise falls back
        to a multiplicative heuristic so unknown tokens don't crash.
    """
    # Try the pack-specific token lookup first -- this is where
    # position-dependent open sizes (e.g. SB BvB opens 3bb vs EP 2.5bb)
    # are registered. The level-1 table entries override the pack's
    # generic open_size_bb when a specific token matches.
    key = (
        (parsed.raise_size_pct, raise_level)
        if parsed.raise_size_pct is not None
        else None
    )
    if key is not None and key in _RYAN_PACK_RAISE_SIZES_BB:
        return _RYAN_PACK_RAISE_SIZES_BB[key]

    if raise_level == 1:
        # Open with no specific table entry: fall back to the pack's
        # documented open size. Covers the majority of EP/MP/LP opens
        # the table already has registered at this value too.
        return pack.open_size_bb

    # Fallback heuristic: 3-bet ≈ 3× open, 4-bet ≈ 2× 3-bet, 5-bet ≈ 2× 4-bet.
    # Used when the pack-specific lookup misses a token.
    fallback = pack.open_size_bb
    multipliers = {2: 3.0, 3: 2.0, 4: 2.0}
    for level in range(2, raise_level + 1):
        fallback *= multipliers.get(level, 2.0)
    return fallback


def _to_action_tuple(
    parsed: ParsedAction,
    raise_level: int,
    bb_amount_multiplier: float,
    pack: PreflopPack,
) -> tuple:
    """Convert one ParsedAction to the brief's ``(position, verb, [amount])`` shape.

    Args:
        parsed: The ParsedAction.
        raise_level: 0 if this action is not itself a raise; 1..N if it is
            (1 = open, 2 = 3-bet, etc).
        bb_amount_multiplier: Multiplier from bb-count to the units used in
            the action history. For cash, this is the dollar value of one
            big blind (so a 2.5bb open becomes ``$1.25``). For tournament,
            pass ``1.0`` (so a 2.5bb open stays as ``2.5bb``).
        pack: Source pack.

    Returns:
        A 2-tuple ``(position, verb)`` for fold/call/check, or a 3-tuple
        ``(position, verb, amount)`` for raise/all-in.
    """
    pos = parsed.position
    if parsed.action_type is PreflopActionType.FOLD:
        return (pos, "fold")
    if parsed.action_type is PreflopActionType.CALL:
        return (pos, "call")
    if parsed.action_type is PreflopActionType.ALL_IN:
        # All-in caps at effective stack -- approximate the size as
        # the stack depth in the chosen unit.
        amount = round(pack.stack_depth_bb * bb_amount_multiplier, 2)
        return (pos, "all-in", amount)
    if parsed.action_type is PreflopActionType.RAISE:
        bb_size = _raise_size_bb(parsed, raise_level, pack)
        return (pos, "raise", round(bb_size * bb_amount_multiplier, 2))
    raise ValueError(f"unknown action type: {parsed.action_type!r}")


def build_hand_dict(
    facts: PreflopFacts,
    *,
    pack: PreflopPack,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
) -> dict[str, Any]:
    """Construct the brief's ``hand`` dict from a PreflopFacts + pack.

    The resulting dict is the input shape :func:`pipeline.action_history.
    format_action_history` consumes. Preflop-only: ``board`` is empty,
    and the postflop action lists are empty so the renderer emits only
    the preflop section.

    Args:
        facts: The Layer 5 preflop data block.
        pack: Source pack (for open size + sb-to-bb ratio + table size).
        stakes_bb_dollars: BB size in dollars. Default 0.50 = Tier 1.
        live_or_online: "Online" or "Live" (cash only; cosmetic for cash).
        game_format: "cash" or "tournament".

    Returns:
        Dict in the brief's input shape, ready for format_action_history.
    """
    bb_dollars = stakes_bb_dollars
    sb_dollars = round(bb_dollars * pack.sb_to_bb_ratio, 2)

    # Render amounts in bb (with a "bb" suffix) when the game is a
    # tournament OR the caller asked for bb display on a cash game
    # (the admin "Display amounts as: Big blinds" toggle). Otherwise
    # cash renders dollars.
    render_in_bb = display_in_bb or game_format != "cash"

    # For dollar display, bb amounts in the action history get converted
    # to dollars (so a 2.5bb open becomes "$1.25"). For bb display the
    # amounts stay in bb (the `_chips` formatter appends "bb"), so the
    # multiplier is 1.0.
    bb_amount_multiplier = 1.0 if render_in_bb else bb_dollars

    # Walk history_before, tracking raise count to assign raise levels.
    actions: list[tuple] = []
    raise_level = 0
    for parsed in facts.spot.node.history_before:
        if parsed.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN):
            raise_level += 1
            actions.append(
                _to_action_tuple(parsed, raise_level, bb_amount_multiplier, pack)
            )
        else:
            actions.append(_to_action_tuple(parsed, 0, bb_amount_multiplier, pack))

    # Effective stack in the display unit: bb when rendering bb, else dollars.
    eff_stack: float = (
        float(pack.stack_depth_bb)
        if render_in_bb
        else round(pack.stack_depth_bb * bb_dollars, 2)
    )

    # Hero cards: split the 4-char combo into the action_history shape.
    combo = facts.spot.hero_card_combo
    if len(combo) != 4:
        raise ValueError(f"hero_card_combo must be 4 chars, got {combo!r}")
    hero_cards = [combo[:2], combo[2:]]

    return {
        "stakes": {"sb": sb_dollars, "bb": bb_dollars},
        "format": game_format,
        # Tells pipeline.action_history.format_action_history to render the
        # "bb" suffix even on a cash game (without changing cash semantics).
        "display_unit": "bb" if render_in_bb else "dollars",
        "venue": live_or_online.lower(),
        "stage": None,
        "buy_in": None,
        "ante": 0,
        "table_size": pack.table_size,
        "effective_stack": eff_stack,
        "hero_position": facts.spot.node.actor,
        "hero_cards": hero_cards,
        "preflop_actions": actions,
        "board": {},
        "flop_actions": [],
        "turn_actions": [],
        "river_actions": [],
    }


def format_preflop_action_history(
    facts: PreflopFacts,
    *,
    pack: PreflopPack,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
) -> str:
    """Preflop action history per the brief's spec.

    Renders:

      * ``You're [POSITION_PHRASE] with [HOLE_CARDS].``
      * One sentence per non-fold prior action, period-terminated.

    Preflop folds are DROPPED (implied by absence, per the brief's
    "Fold Rule"). Raise verbs follow the level: 1st raise = "open",
    2nd = "3-bet", 3rd = "4-bet", 4th = "5-bet", 5+ = "raise". Hero
    uses the base verb ("you open"); villain uses third-person
    ("UTG opens").

    Args:
        facts: The Layer 5 data block.
        pack: Source pack (for table size, open size, sb ratio).
        stakes_bb_dollars: BB size in dollars. Default 0.50 = Tier 1.
        live_or_online: Cosmetic; passed through (cash only).
        game_format: "cash" or "tournament".

    Returns:
        Multi-line action-history string, ready for the Question CSV
        column.
    """
    hand = build_hand_dict(
        facts,
        pack=pack,
        stakes_bb_dollars=stakes_bb_dollars,
        live_or_online=live_or_online,
        game_format=game_format,
        display_in_bb=display_in_bb,
    )
    return format_action_history(hand)


__all__ = [
    "build_hand_dict",
    "format_preflop_action_history",
]
