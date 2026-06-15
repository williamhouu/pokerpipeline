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

from dataclasses import dataclass
from typing import Any

from pipeline.action_history import format_action_history, preflop_order
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.grammars.types import (
    MIN_RAISE_PCT,
    ParsedAction,
    PreflopActionType,
)
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


# --- shared history resolution (sizes + pot accounting) ----------------------
# Monker-export packs (grammar "monker_nlhe") don't need a size lookup at
# all: the raise token IS the size, as a percent of pot under Monker's
# pot-relative rule. Resolving it requires the running pot state, so the
# walk below is the single source of truth for preflop pot math -- the
# Question prose, the POT column, the EV engine, and the ranges-column
# action mixes all read from it and can't drift apart.
_MONKER_GRAMMAR = "monker_nlhe"


@dataclass(frozen=True)
class ResolvedPreflopState:
    """Pot accounting after a sequence of preflop actions.

    Fields:
        sizes_bb: Per-action "to" amount in bb, aligned with the input
            history: the raise-to / all-in total for aggressive actions,
            ``None`` for folds and calls.
        committed_bb: Each seat's total commitment so far, blinds
            included. Folded seats keep their dead money here (it is in
            the pot).
        high_bet_bb: The current bet level a caller must match.
        raise_level: Aggressive actions so far (1 = open, 2 = 3-bet, ...).
    """

    sizes_bb: tuple[float | None, ...]
    committed_bb: dict[str, float]
    high_bet_bb: float
    raise_level: int

    @property
    def pot_bb(self) -> float:
        """Chips in the middle: every seat's commitment, dead money included."""
        return sum(self.committed_bb.values())

    def call_cost_bb(self, position: str) -> float:
        """What ``position`` must add to match the current bet."""
        return max(0.0, self.high_bet_bb - self.committed_bb.get(position, 0.0))


def _monker_min_raise_to_bb(high_bet: float) -> float:
    """Monker's ``5`` min-raise, in bb: raise to twice the current bet.

    Unlike the ``40<pct>`` pot-relative tokens, the min-raise is sized
    *relative to the running bet*, not as a fraction of the pot. For an
    open (the current bet is the 1bb big blind) this is the 2bb min-raise;
    ``5`` is only ever the open in the registered short-stack 6-max packs,
    so in practice this resolves the 2bb opens. The "twice the current bet"
    rule is exact for an open and is the documented convention for these
    packs; a true min-raise (current bet + last raise increment) would only
    diverge on a *re-raise* by ``5``, which does not occur.
    """
    return 2.0 * high_bet


def _monker_raise_to_bb(
    raise_size_pct: float,
    *,
    committed: dict[str, float],
    high_bet: float,
    position: str,
) -> float:
    """Monker's pot-relative raise rule, in bb.

    ``raise_to = high_bet + pct × (pot + to_call)`` where ``to_call`` is
    what the raiser owes before raising. Same math as the PLO pack's
    :func:`pipeline.plo.action_history.resolve_pot_limit` (Monker uses one
    sizing rule across games; NLHE just isn't capped at 100%). First-in at
    a 9-max table: ``1 + 1.2 × (1.5 + 1) = 4.0`` -- the pack's 40120 token
    is the documented 4bb open.
    """
    prev = committed.get(position, 0.0)
    to_call = high_bet - prev
    pot = sum(committed.values())
    return high_bet + (raise_size_pct / 100.0) * (pot + to_call)


def _quantize_raise(raise_to: float, pack: PreflopPack) -> float:
    """Snap a resolved raise-to amount to the pack's display quantum.

    Pot-percentage trees produce sizes like 13.625bb; packs with
    ``size_round_bb`` set (the Monker 9-max pack uses 0.5) render the
    nearest clean amount instead, and the CALLER's pot accounting then
    accumulates from the rounded value -- so the prose, POT column, and
    pot-odds all describe the same (quantized) game. All-ins never pass
    through here.
    """
    q = pack.size_round_bb
    if not q:
        return raise_to
    return round(raise_to / q) * q


def resolve_preflop_history(
    history: tuple[ParsedAction, ...],
    pack: PreflopPack,
) -> ResolvedPreflopState:
    """Walk a prior-action sequence into per-action bb sizes + pot state.

    Starts from the posted blinds and applies each action in order. Raise
    sizes come from the pack's sizing convention: Monker-grammar packs use
    the pot-relative formula on the running state; PioViewer packs use the
    :data:`_RYAN_PACK_RAISE_SIZES_BB` token lookup (with its documented
    fallbacks). All-ins commit the effective stack -- precise side-pot
    math would need per-seat stack tracking, deferred.
    """
    is_monker = pack.grammar_name == _MONKER_GRAMMAR
    committed: dict[str, float] = {"SB": pack.sb_to_bb_ratio, "BB": 1.0}
    high_bet = 1.0  # everyone has to match the BB to enter
    raise_level = 0
    sizes: list[float | None] = []
    for parsed in history:
        pos = parsed.position
        if parsed.action_type is PreflopActionType.FOLD:
            # Fold adds no chips; any blind already posted stays in the pot.
            sizes.append(None)
            continue
        if parsed.action_type is PreflopActionType.CALL:
            committed[pos] = high_bet
            sizes.append(None)
            continue
        raise_level += 1
        if parsed.action_type is PreflopActionType.ALL_IN:
            stack = float(pack.stack_depth_bb)
            committed[pos] = stack
            high_bet = max(high_bet, stack)
            sizes.append(stack)
            continue
        # RAISE
        if is_monker:
            if parsed.raise_size_pct == MIN_RAISE_PCT:
                raise_to = _quantize_raise(
                    _monker_min_raise_to_bb(high_bet), pack
                )
            else:
                raise_to = _quantize_raise(
                    _monker_raise_to_bb(
                        parsed.raise_size_pct or 100.0,
                        committed=committed,
                        high_bet=high_bet,
                        position=pos,
                    ),
                    pack,
                )
        else:
            raise_to = _raise_size_bb(parsed, raise_level, pack)
        committed[pos] = raise_to
        high_bet = raise_to
        sizes.append(raise_to)
    return ResolvedPreflopState(
        sizes_bb=tuple(sizes),
        committed_bb=committed,
        high_bet_bb=high_bet,
        raise_level=raise_level,
    )


def compute_action_pending(
    history: tuple[ParsedAction, ...],
    actor: str,
    table_size: int,
) -> tuple[list[str], list[str], bool]:
    """Who is still in the hand, and who still acts after hero.

    Returns ``(others_still_in, still_to_act_after_hero, hero_closes)``:

      * ``others_still_in`` -- every non-folded seat except hero, in seat
        order, with all-in seats marked ``"HJ (all-in)"``. These players
        hold cards; the LLM may reference their recorded actions but has
        no range data for them.
      * ``still_to_act_after_hero`` -- the subset with a PENDING decision
        if hero calls or folds: seats that owe chips to the current bet,
        plus seats that haven't acted at all since the bet reached its
        current level (the BB option in unraised pots). All-in and folded
        seats never appear.
      * ``hero_closes`` -- True when that list is empty: hero's call or
        fold ends the preflop action. (A raise by hero always reopens
        it -- this bool is about the passive lines.)

    Built for the Layer-6 SOLVER DATA block (June 2026 audit findings #1
    and #3: the model invented still-live players who had already folded,
    and missed that hero closed the action). Pure bookkeeping from the
    recorded history -- no file reads.
    """
    seats = preflop_order(table_size)
    folded: set[str] = set()
    all_in: set[str] = set()
    committed: dict[str, float] = {"SB": 0.5, "BB": 1.0}
    high_bet = 1.0
    acted_since_raise: set[str] = set()
    for a in history:
        if a.action_type is PreflopActionType.FOLD:
            folded.add(a.position)
            acted_since_raise.add(a.position)
        elif a.action_type is PreflopActionType.CALL:
            committed[a.position] = high_bet
            acted_since_raise.add(a.position)
        elif a.action_type is PreflopActionType.ALL_IN:
            # Jams commit the effective stack (sentinel: actual size is
            # irrelevant for pending logic). A jam over an existing jam is
            # a CALL-OFF at equal stacks, not a new bet level -- it must
            # not reset who has already acted.
            all_in.add(a.position)
            jam = 1e9
            if high_bet < jam:
                high_bet = jam
                acted_since_raise = {a.position}
            else:
                acted_since_raise.add(a.position)
            committed[a.position] = jam
        else:  # RAISE -- exact size irrelevant here; only "a new bet level"
            high_bet += 1.0
            committed[a.position] = high_bet
            acted_since_raise = {a.position}

    others_in = [
        s + (" (all-in)" if s in all_in else "")
        for s in seats
        if s != actor and s not in folded
    ]
    pending = [
        s
        for s in seats
        if s != actor
        and s not in folded
        and s not in all_in
        and (
            committed.get(s, 0.0) < high_bet or s not in acted_since_raise
        )
    ]
    return others_in, pending, not pending


def raise_to_bb(
    state: ResolvedPreflopState,
    *,
    position: str,
    raise_size_pct: float | None,
    pack: PreflopPack,
) -> float:
    """Size (bb) of a hypothetical NEXT raise by ``position`` given ``state``.

    Used to size the actions *offered at* a node (hero's raise option, a
    villain chart's raise column) rather than actions already in the
    history. Monker packs apply the pot-relative rule to the resolved
    state (quantized like the history sizes, so an option reading "raise
    to 13.5bb" matches what the prose would say if taken); PioViewer
    packs hit the token lookup at ``state.raise_level + 1``.
    """
    if pack.grammar_name == _MONKER_GRAMMAR:
        if raise_size_pct == MIN_RAISE_PCT:
            return _quantize_raise(
                _monker_min_raise_to_bb(state.high_bet_bb), pack
            )
        return _quantize_raise(
            _monker_raise_to_bb(
                raise_size_pct or 100.0,
                committed=state.committed_bb,
                high_bet=state.high_bet_bb,
                position=position,
            ),
            pack,
        )
    parsed = ParsedAction(
        position=position,
        action_type=PreflopActionType.RAISE,
        raise_size_pct=raise_size_pct,
    )
    return _raise_size_bb(parsed, state.raise_level + 1, pack)


def _to_action_tuple(
    parsed: ParsedAction,
    size_bb: float | None,
    bb_amount_multiplier: float,
) -> tuple:
    """Convert one ParsedAction to the brief's ``(position, verb, [amount])`` shape.

    Args:
        parsed: The ParsedAction.
        size_bb: The action's resolved "to" amount in bb (from
            :func:`resolve_preflop_history`); ``None`` for fold/call.
        bb_amount_multiplier: Multiplier from bb-count to the units used in
            the action history. For cash, this is the dollar value of one
            big blind (so a 2.5bb open becomes ``$1.25``). For tournament,
            pass ``1.0`` (so a 2.5bb open stays as ``2.5bb``).

    Returns:
        A 2-tuple ``(position, verb)`` for fold/call/check, or a 3-tuple
        ``(position, verb, amount)`` for raise/all-in.
    """
    pos = parsed.position
    if parsed.action_type is PreflopActionType.FOLD:
        return (pos, "fold")
    if parsed.action_type is PreflopActionType.CALL:
        return (pos, "call")
    if size_bb is None:
        raise ValueError(f"missing resolved size for {parsed.action_type!r}")
    if parsed.action_type is PreflopActionType.ALL_IN:
        return (pos, "all-in", round(size_bb * bb_amount_multiplier, 2))
    if parsed.action_type is PreflopActionType.RAISE:
        return (pos, "raise", round(size_bb * bb_amount_multiplier, 2))
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

    # Resolve every prior action's bb size in one walk (the shared pot
    # accounting), then render each as the brief's action tuple.
    history = facts.spot.node.history_before
    state = resolve_preflop_history(history, pack)
    actions: list[tuple] = [
        _to_action_tuple(parsed, size_bb, bb_amount_multiplier)
        for parsed, size_bb in zip(history, state.sizes_bb, strict=True)
    ]

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
    "ResolvedPreflopState",
    "build_hand_dict",
    "compute_action_pending",
    "format_preflop_action_history",
    "raise_to_bb",
    "resolve_preflop_history",
]
