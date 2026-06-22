"""Layer 5 (postflop): the fact extractor -- the most important layer.

Turns a :class:`~pipeline.postflop.spot_sampler.PostflopSpot` (a node + a hero
combo) into a fully-resolved :class:`PostflopFacts` block: hero's made-hand
class and the board texture (via the repo's pure classifiers), hero's equity
vs villain's range on the board (via the reused 7-card evaluator), pot
geometry (SPR, pot odds / break-even when facing a bet), the EV gap, the
strategic archetype, and the concept tags. Every strategic claim in the final
explanation must trace back to a field here.

This module is the postflop analogue of
:mod:`pipeline.preflop.fact_extractor`. It reuses only *pure, game-agnostic
leaf functions* from elsewhere in the repo:

    * :func:`pipeline.fact_extractor.hand_class.classify_hand`
    * :func:`pipeline.fact_extractor.board_texture.classify_board`
    * :func:`pipeline.fact_extractor.equity.equity_vs_range`

It does not import any other pipeline's batch driver, fact extractor, or
validators, and it never mutates a shared module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from pipeline.fact_extractor.board_texture import classify_board
from pipeline.fact_extractor.equity import equity_vs_range
from pipeline.fact_extractor.hand_class import classify_hand
from pipeline.postflop.concept_tags import (
    PostflopTagInput,
    classify_postflop_archetype,
    compute_postflop_tags,
)
from pipeline.postflop.solve import PostflopSolve
from pipeline.postflop.spot_sampler import PostflopSpot, spot_ev_gap_bb

# Per-villain-combo board runouts sampled when computing hero equity. The
# estimate is seeded per-spot (see ``_spot_rng``) so recomputing the SAME spot
# is byte-identical -- threshold-fed values (archetype via strength, the
# concept tags, difficulty) can't flip between runs. 200 keeps a full batch
# fast; raise for a published-precision figure.
DEFAULT_EQUITY_RUNOUTS = 200

# Preflop verbs that count as putting in a raise (for the SRP/3bet/4bet tag and
# "who is the preflop aggressor").
_RAISE_VERBS = frozenset({"open", "raise", "3-bet", "4-bet", "5-bet"})

# The street immediately before each decision street (flop has none).
_PREV_STREET = {"turn": "flop", "river": "turn"}


def _prior_street_context(
    history: tuple, hero: str, street: str
) -> tuple[bool, bool]:
    """(hero_bet_prev_street, prev_street_checked_through) from the node history.

    Drives the street-aware action-context tags: whether hero put in a bet/raise
    on the immediately preceding street (a turn barrel continues flop aggression)
    and whether that street had NO bet at all (a delayed c-bet / probe follows a
    checked-through street). Flop decisions have no prior street -> (False, False).
    """
    prev = _PREV_STREET.get(street)
    if prev is None:
        return False, False
    prev_bets = [
        s for s in history if s.street == prev and s.verb in ("bet", "raise")
    ]
    hero_bet_prev = any(s.position == hero for s in prev_bets)
    return hero_bet_prev, not prev_bets


@dataclass(frozen=True)
class PostflopFacts:
    """Pre-computed facts for one postflop spot. Layer 6's only input.

    Every field is solver-derived and deterministic. The LLM turns this into
    prose; it never recomputes or second-guesses any of it.
    """

    spot: PostflopSpot

    # --- who / where ---
    hero_position: str
    villain_position: str
    hero_in_position: bool
    hero_is_preflop_aggressor: bool
    street: str
    board: tuple[str, ...]

    # --- hand + board (reused pure classifiers) ---
    made_hand: str
    draws: tuple[str, ...]
    strength_bucket: str
    hand_label: str  # composite, e.g. "top_pair_top_kicker_with_flush_draw"
    board_texture: dict[str, str]  # 5 axes + composite, from classify_board

    # --- equity / pot geometry ---
    hero_equity_vs_villain: float  # hero hand vs villain's full range, [0,1]
    villain_range_combos: int  # live villain combos (not blocked by board/hero)
    hero_blocks_combos: int  # villain combos hero removes by card sharing
    spr: float
    pot_bb: float
    to_call_bb: float
    break_even_equity: float | None  # pot-odds threshold when facing a bet
    equity_runouts_used: int

    # --- decision signals ---
    dominant_action: str  # the action LABEL (e.g. "Bet 33%")
    dominant_verb: str
    dominant_frequency: float
    ev_gap_bb: float | None
    archetype: str
    concept_tags: list[str] = field(default_factory=list)

    # --- pot context (for tags / writer) ---
    preflop_raise_count: int = 1
    n_players: int = 2


def _spot_rng(spot: PostflopSpot) -> random.Random:
    """Deterministic RNG keyed by the spot's identity (node + combo)."""
    return random.Random(f"{spot.node.node_id}|{spot.hero_combo}")


def preflop_aggressor(solve: PostflopSolve) -> str:
    """Position of the last preflop raiser (the c-bet "aggressor"). '' if none."""
    aggressor = ""
    for step in solve.preflop_summary:
        if step.verb in _RAISE_VERBS:
            aggressor = step.position
    return aggressor


def preflop_raise_count(solve: PostflopSolve) -> int:
    """Number of preflop raises (open=1 SRP, +3-bet=2, +4-bet=3 ...)."""
    return sum(1 for step in solve.preflop_summary if step.verb in _RAISE_VERBS)


def extract_facts(
    spot: PostflopSpot,
    solve: PostflopSolve,
    *,
    equity_runouts: int = DEFAULT_EQUITY_RUNOUTS,
) -> PostflopFacts:
    """Compute the full fact block for one postflop spot.

    Args:
        spot: the (node, hero combo) spot from the sampler.
        solve: the parent solve (for positions + the preflop line).
        equity_runouts: per-villain-combo board samples for the equity calc.

    Returns:
        A fully-populated :class:`PostflopFacts`.
    """
    node = spot.node
    board = list(node.board)
    hero_cards = spot.hero_cards

    # --- hand + board via the reused pure classifiers ---
    hand = classify_hand(hero_cards, board)
    texture = classify_board(board)

    # --- equity (seeded for determinism) ---
    hero_equity, _per_combo = equity_vs_range(
        hero_cards,
        dict(node.villain_range),
        board,
        max_runouts=equity_runouts,
        rng=_spot_rng(spot),
    )

    # --- light blocker / range-size signals ---
    hero_card_set = set(hero_cards)
    live = 0
    blocked = 0
    for combo, weight in node.villain_range.items():
        if weight <= 0:
            continue
        cards = {combo[:2], combo[2:]}
        if cards & set(board):
            continue  # not a real villain holding on this board
        if cards & hero_card_set:
            blocked += 1
            continue
        live += 1

    # --- pot geometry ---
    break_even = None
    if node.to_call_bb > 0:
        break_even = node.to_call_bb / (node.pot_bb + node.to_call_bb)

    # --- positions / preflop context ---
    hero_in_position = node.actor == solve.ip_position
    aggressor = preflop_aggressor(solve)
    hero_is_aggressor = bool(aggressor) and node.actor == aggressor
    n_raises = preflop_raise_count(solve)

    # --- archetype + concept tags ---
    hero_bet_prev, prev_checked_through = _prior_street_context(
        node.history, node.actor, node.street
    )
    tag_input = PostflopTagInput(
        street=node.street,
        preflop_raise_count=n_raises,
        n_players=len(solve.positions),
        hero_is_preflop_aggressor=hero_is_aggressor,
        hero_in_position=hero_in_position,
        is_facing_bet=node.is_facing_bet,
        dominant_verb=spot.dominant_verb,
        made_hand=hand["made_hand"],
        draws=tuple(hand["draws"]),
        strength_bucket=hand["strength_bucket"],
        suit_distribution=texture["suit_distribution"],
        pair_status=texture["pair_status"],
        connectedness=texture["connectedness"],
        composite=texture["composite"],
        hero_equity=hero_equity,
        break_even_equity=break_even,
        hero_bet_prev_street=hero_bet_prev,
        prev_street_checked_through=prev_checked_through,
    )
    archetype = classify_postflop_archetype(tag_input)
    concept_tags = compute_postflop_tags(tag_input)

    return PostflopFacts(
        spot=spot,
        hero_position=node.actor,
        villain_position=node.villain,
        hero_in_position=hero_in_position,
        hero_is_preflop_aggressor=hero_is_aggressor,
        street=node.street,
        board=tuple(node.board),
        made_hand=hand["made_hand"],
        draws=tuple(hand["draws"]),
        strength_bucket=hand["strength_bucket"],
        hand_label=hand["label"],
        board_texture=texture,
        hero_equity_vs_villain=hero_equity,
        villain_range_combos=live,
        hero_blocks_combos=blocked,
        spr=node.spr,
        pot_bb=node.pot_bb,
        to_call_bb=node.to_call_bb,
        break_even_equity=break_even,
        equity_runouts_used=equity_runouts,
        dominant_action=spot.dominant_action,
        dominant_verb=spot.dominant_verb,
        dominant_frequency=spot.dominant_frequency,
        ev_gap_bb=spot_ev_gap_bb(spot),
        archetype=archetype,
        concept_tags=concept_tags,
        preflop_raise_count=n_raises,
        n_players=len(solve.positions),
    )


__all__ = [
    "DEFAULT_EQUITY_RUNOUTS",
    "PostflopFacts",
    "extract_facts",
    "preflop_aggressor",
    "preflop_raise_count",
]
