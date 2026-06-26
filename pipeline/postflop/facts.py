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
from pipeline.fact_extractor.equity import (
    equity_vs_range,
    range_vs_range_equity,
    rank_hand,
)
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

# Range-vs-range equity sampling bound (the "who has the range advantage on this
# board" stat). Node-level (same for every hero combo at the node), seeded to a
# fixed value so generation and the batch re-verifier compute it identically.
# 400 matchups keeps a batch fast while pinning the equity to ~+/-1pt.
_RANGE_EQUITY_MATCHUPS = 400
_RANGE_EQUITY_SEED = 0
# How far apart two ranges' shares must be to call it an "advantage" (vs "even").
# Range equities cluster near 50%, so a small but real edge is meaningful; nut
# shares get a flat 3-point margin.
_RANGE_ADV_MARGIN = 0.02  # on hero range equity around 0.50 -> hero >= 52%
_NUT_ADV_MARGIN = 0.03  # 3 percentage points of strong-made share

# Villain made-hand buckets that count as "value" (strong hands that bet/call
# for value) vs "bluff" (air, no showdown). The medium middle (weak pairs, etc.)
# is neither -- it's the showdown-value hands that mostly check/call.
_VALUE_BUCKETS = frozenset({"premium", "strong"})
_BLUFF_BUCKETS = frozenset({"air"})
# A blocker effect is only "real" when hero removes a non-trivial share AND one
# side is removed clearly more than the other.
_BLOCKER_MIN_PCT = 0.05
_BLOCKER_MARGIN = 0.03

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
    # Reach-weighted fraction of villain's range hero's made hand BEATS / loses to
    # at showdown on the CURRENT board (deterministic, exact -- no runouts). The
    # "where my equity comes from" fact: a made hand ahead of villain's air has
    # SHOWDOWN equity, not draw equity. See compute_currently_ahead.
    currently_ahead_pct: float
    currently_behind_pct: float
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

    # --- range-vs-range (node-level; the "who is ahead here" stat) ---
    # hero_range_equity: the ACTOR's whole range vs the villain's whole range on
    # this board, [0,1]. range_advantage / nut_advantage are the RESOLVED verdict
    # ("hero" / "villain" / "even") the LLM mirrors -- computed in Python so the
    # model never decides who has the edge. hero/villain_nut_share = weighted
    # fraction of each range that is two-pair-or-better (the value region).
    hero_range_equity: float = 0.5
    range_advantage: str = "even"
    hero_nut_share: float = 0.0
    villain_nut_share: float = 0.0
    nut_advantage: str = "even"

    # --- blocker value/bluff decomposition (do hero's cards remove villain's
    # value or bluffs?). blocker_effect ∈ "value" / "bluffs" / "neutral" is the
    # RESOLVED verdict the LLM mirrors; the pcts are the supporting evidence.
    blocked_value_pct: float = 0.0
    blocked_bluff_pct: float = 0.0
    blocker_effect: str = "neutral"

    # --- pot context (for tags / writer) ---
    preflop_raise_count: int = 1
    n_players: int = 2

    @property
    def has_strong_draw(self) -> bool:
        """True for a REAL draw worth playing as a draw -- a flush draw, an
        open-ended straight draw, or a combo draw. Excludes backdoors and bare
        gutshots (which ride in ``draws`` but are not 'drawing hands'). Single
        source of truth for the difficulty hand axis + the skills tagger."""
        return any(d in _STRONG_DRAW_TYPES for d in self.draws)


# Draw types that make a hand a genuine "drawing hand" (mirrors the strong-draw
# set the concept-tag archetype logic uses); backdoors / bare gutshots excluded.
_STRONG_DRAW_TYPES = frozenset({
    "flush_draw_nut", "flush_draw_weak", "straight_draw_open_ended", "combo_draw",
})


def _spot_rng(spot: PostflopSpot) -> random.Random:
    """Deterministic RNG keyed by the spot's identity (node + combo)."""
    return random.Random(f"{spot.node.node_id}|{spot.hero_combo}")


def _range_strong_share(combo_range, board: list[str]) -> float:
    """Weighted fraction of ``combo_range`` that is two-pair-or-better on ``board``.

    The "nutted / value region" share -- the standard proxy for nut advantage.
    Board-blocked combos (sharing a card with the board) are skipped, matching
    the live set the equity calc uses. Pure: hole cards + board -> a category,
    no runouts, so it is cheap to run over a whole range.
    """
    board_set = set(board)
    strong = total = 0.0
    for combo, weight in combo_range.items():
        if weight <= 0:
            continue
        cards = [combo[:2], combo[2:]]
        if set(cards) & board_set:
            continue
        total += weight
        if rank_hand(cards + list(board))[0] >= 2:  # two pair or better
            strong += weight
    return strong / total if total else 0.0


def _advantage_label(hero_value: float, villain_value: float, margin: float) -> str:
    """'hero' / 'villain' / 'even' from two comparable shares and a margin.

    The single place the range/nut-advantage *conclusion* is decided -- in
    Python, never the LLM -- so the explanation can only mirror the verdict the
    fact carries (the brief's anti-"range advantage to the wrong player" rule).
    """
    diff = hero_value - villain_value
    if diff >= margin:
        return "hero"
    if diff <= -margin:
        return "villain"
    return "even"


def compute_blocker_decomposition(
    hero_cards: list[str], villain_range, board: list[str]
) -> tuple[float, float, str]:
    """How much of villain's VALUE vs BLUFF range hero's cards remove.

    The pedagogically critical postflop blocker fact -- are you blocking the
    opponent's VALUE (good when bluff-catching: fewer value combos left, so a
    bet is more often a bluff) or their BLUFFS (bad: you've removed the hands
    you beat)? Computed deterministically (never the LLM): classify every
    board-unblocked villain combo's made hand, split into value (strong made
    hands) and bluff (air, no showdown), and measure the weighted fraction of
    each that hero removes by card-sharing. Reach-weighted, so on a facing-bet
    node -- where ``villain_range`` IS the reach-weighted betting range -- this
    reflects the real composition of the bet hero faces.

    Returns ``(blocked_value_pct, blocked_bluff_pct, effect)`` where ``effect``
    is ``"value"`` / ``"bluffs"`` / ``"neutral"``.
    """
    hero_set = set(hero_cards)
    board_set = set(board)
    value_total = value_blocked = bluff_total = bluff_blocked = 0.0
    for combo, weight in villain_range.items():
        if weight <= 0:
            continue
        cards = [combo[:2], combo[2:]]
        cs = set(cards)
        if cs & board_set:
            continue  # not a real villain holding on this board
        bucket = classify_hand(cards, board)["strength_bucket"]
        blocked = bool(cs & hero_set)
        if bucket in _VALUE_BUCKETS:
            value_total += weight
            value_blocked += weight if blocked else 0.0
        elif bucket in _BLUFF_BUCKETS:
            bluff_total += weight
            bluff_blocked += weight if blocked else 0.0
    value_pct = value_blocked / value_total if value_total else 0.0
    bluff_pct = bluff_blocked / bluff_total if bluff_total else 0.0
    diff = value_pct - bluff_pct
    if max(value_pct, bluff_pct) >= _BLOCKER_MIN_PCT and abs(diff) >= _BLOCKER_MARGIN:
        effect = "value" if diff > 0 else "bluffs"
    else:
        effect = "neutral"
    return value_pct, bluff_pct, effect


def compute_currently_ahead(
    hero_cards: list[str], villain_range, board: list[str]
) -> tuple[float, float]:
    """How much of villain's range hero's made hand BEATS at showdown right now.

    Returns ``(ahead_pct, behind_pct)`` -- the reach-weighted fraction of
    villain's (board- and hero-unblocked) range that hero strictly beats / loses
    to on the CURRENT board (no future cards). Ties count as neither, so
    ``ahead + behind`` can be < 1.

    The "where does my equity come from" fact (the brief's #1 failure mode --
    the LLM mis-attributing a made hand's equity to draws). A small pair that is
    *ahead* of half of villain's betting range has SHOWDOWN equity -- it beats
    villain's air right now -- which is categorically different from a draw's
    equity (which comes from IMPROVING). Deterministic: an exact 5-/6-/7-card
    showdown comparison via the shared evaluator, no runouts, so it reproduces
    byte-for-byte (unlike the sampled ``hero_equity_vs_villain``). On a
    facing-bet node ``villain_range`` IS the reach-weighted betting range, so
    this is "what fraction of the hands villain is BETTING do I currently beat".
    """
    hero_set = set(hero_cards)
    board_set = set(board)
    hero_rank = rank_hand(list(hero_cards) + list(board))
    ahead = behind = total = 0.0
    for combo, weight in villain_range.items():
        if weight <= 0:
            continue
        cards = [combo[:2], combo[2:]]
        cs = set(cards)
        if cs & board_set or cs & hero_set:
            continue  # not a real villain holding (shares a card with board/hero)
        total += weight
        villain_rank = rank_hand(cards + list(board))
        if hero_rank > villain_rank:
            ahead += weight
        elif hero_rank < villain_rank:
            behind += weight
        # equal ranks (chop) -> neither ahead nor behind
    if total <= 0:
        return 0.0, 0.0
    return ahead / total, behind / total


def compute_range_advantage(node) -> tuple[float, str, float, float, str]:
    """Node-level range-vs-range stats for the actor (hero) vs the villain.

    Returns ``(hero_range_equity, range_advantage, hero_nut_share,
    villain_nut_share, nut_advantage)``. This is a property of the two RANGES on
    the board -- identical for every hero combo at the node -- so it is the
    grounded "who is ahead here" fact. Deterministic (fixed seed) so the batch
    re-verifier reproduces it exactly.
    """
    board = list(node.board)
    hero_eq = range_vs_range_equity(
        dict(node.hero_range),
        dict(node.villain_range),
        board,
        max_matchups=_RANGE_EQUITY_MATCHUPS,
        seed=_RANGE_EQUITY_SEED,
    )
    range_adv = _advantage_label(hero_eq, 1.0 - hero_eq, _RANGE_ADV_MARGIN)
    hero_nut = _range_strong_share(node.hero_range, board)
    villain_nut = _range_strong_share(node.villain_range, board)
    nut_adv = _advantage_label(hero_nut, villain_nut, _NUT_ADV_MARGIN)
    return hero_eq, range_adv, hero_nut, villain_nut, nut_adv


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

    # --- range-vs-range advantage (node-level; "who is ahead here") ---
    (
        hero_range_equity,
        range_advantage,
        hero_nut_share,
        villain_nut_share,
        nut_advantage,
    ) = compute_range_advantage(node)

    # --- blocker value/bluff decomposition (hero-specific) ---
    blocked_value_pct, blocked_bluff_pct, blocker_effect = (
        compute_blocker_decomposition(hero_cards, dict(node.villain_range), board)
    )

    # --- currently ahead / behind (exact showdown equity composition) ---
    currently_ahead, currently_behind = compute_currently_ahead(
        hero_cards, dict(node.villain_range), board
    )

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
        range_advantage=range_advantage,
        nut_advantage=nut_advantage,
        blocker_effect=blocker_effect,
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
        currently_ahead_pct=currently_ahead,
        currently_behind_pct=currently_behind,
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
        hero_range_equity=hero_range_equity,
        range_advantage=range_advantage,
        hero_nut_share=hero_nut_share,
        villain_nut_share=villain_nut_share,
        nut_advantage=nut_advantage,
        blocked_value_pct=blocked_value_pct,
        blocked_bluff_pct=blocked_bluff_pct,
        blocker_effect=blocker_effect,
        preflop_raise_count=n_raises,
        n_players=len(solve.positions),
    )


__all__ = [
    "DEFAULT_EQUITY_RUNOUTS",
    "PostflopFacts",
    "compute_blocker_decomposition",
    "compute_currently_ahead",
    "compute_range_advantage",
    "extract_facts",
    "preflop_aggressor",
    "preflop_raise_count",
]
