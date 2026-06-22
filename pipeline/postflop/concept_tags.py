"""Postflop concept tags + strategic archetype (Layer 5 detail).

Concept tags are the computational atoms the brief calls for: boolean facts
about the spot, computed by **pure Python rules from solver-derived data** --
never written by an LLM. They are the LLM's input (the data block) and a
retrieval/QA key. The archetype is the single strategic *frame* the LLM is
told to write toward (value bet vs bluff vs pot-control check vs bluff-catch).

This is a postflop-native module: the predicates read a small
:class:`PostflopTagInput` (built by the fact extractor from the reused
hand-class / board-texture classifiers), NOT the older ``SpotData``. That
keeps the postflop pipeline decoupled -- nothing here imports
``pipeline.fact_extractor`` -- so changes can't ripple into other pipelines.
The registry is a dict of pure ``(PostflopTagInput) -> bool`` functions, so
adding a tag is a one-line, unit-testable change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Archetype labels (shared vocabulary with the repo's older postflop
# archetype module so downstream analytics line up). The LLM gets exactly one
# of these as its strategic frame.
POSTFLOP_ARCHETYPES = (
    "value_bet", "bluff", "protection_bet", "semibluff",
    "pot_control_check", "trap_check", "defensive_check",
    "bluff_catch", "call_with_showdown", "call_drawing",
    "fold_to_pressure", "value_raise", "bluff_raise",
    "unclassified",
)

# Draw groupings used by several tags + the archetype classifier.
_FLUSH_DRAWS = frozenset({"flush_draw_nut", "flush_draw_weak", "combo_draw"})
_STRAIGHT_DRAWS = frozenset({"straight_draw_open_ended", "gutshot", "combo_draw"})
# "Strong" = a draw worth semibluffing (excludes bare gutshots / backdoors).
_STRONG_DRAWS = frozenset(
    {"flush_draw_nut", "flush_draw_weak", "straight_draw_open_ended", "combo_draw"}
)
# Made hands at "two pair or better" -- the value core.
_SET_OR_BETTER = frozenset({
    "straight_flush", "quads", "full_house_set_plus_board",
    "full_house_trips_plus_pocket", "set", "trips", "flush_nut",
    "flush_second_nut", "flush_weak", "straight_nut", "straight_weak",
    "two_pair_top", "two_pair_top_and_bottom", "two_pair_mid",
})
_WET = frozenset({"wet", "very_wet", "dynamic"})


@dataclass(frozen=True)
class PostflopTagInput:
    """The minimal, solver-derived facts the tag rules + archetype read.

    Built by :func:`pipeline.postflop.facts.extract_facts`. Every field is a
    plain value (no nested solver objects) so the predicates stay trivial.
    """

    street: str
    preflop_raise_count: int
    n_players: int
    hero_is_preflop_aggressor: bool
    hero_in_position: bool
    is_facing_bet: bool
    dominant_verb: str  # "check"|"bet"|"call"|"raise"|"fold"
    made_hand: str
    draws: tuple[str, ...]
    strength_bucket: str
    suit_distribution: str
    pair_status: str
    connectedness: str
    composite: str  # board-texture composite descriptor
    hero_equity: float
    break_even_equity: float | None
    # Prior-street action context (from the node history; default False so the
    # flop case and hand-built inputs need not set them). Drive the street-aware
    # action-context tags (turn barrel vs delayed c-bet vs probe).
    hero_bet_prev_street: bool = False  # hero bet/raised the immediately prior street
    prev_street_checked_through: bool = False  # no bet on the immediately prior street

    # convenience predicates (kept off the tag registry; used by rules + archetype)
    @property
    def has_flush_draw(self) -> bool:
        return any(d in _FLUSH_DRAWS for d in self.draws)

    @property
    def has_straight_draw(self) -> bool:
        return any(d in _STRAIGHT_DRAWS for d in self.draws)

    @property
    def has_strong_draw(self) -> bool:
        return any(d in _STRONG_DRAWS for d in self.draws)


# --- the tag registry (pure predicates) -------------------------------------
TAG_REGISTRY: dict[str, Callable[[PostflopTagInput], bool]] = {
    # Pot context
    "single_raised_pot": lambda s: s.preflop_raise_count <= 1,
    "threebet_pot": lambda s: s.preflop_raise_count == 2,
    "fourbet_pot": lambda s: s.preflop_raise_count >= 3,
    "multiway_pot": lambda s: s.n_players > 2,
    "heads_up_pot": lambda s: s.n_players == 2,
    # Action context (street-aware: a flop c-bet, a turn barrel, a delayed
    # c-bet, an OOP probe, and a river bet are distinct strategic situations).
    "c_bet_spot": lambda s: (
        s.dominant_verb == "bet" and s.hero_is_preflop_aggressor and s.street == "flop"
    ),
    "donk_bet_spot": lambda s: (
        s.dominant_verb == "bet"
        and not s.hero_is_preflop_aggressor
        and not s.hero_in_position
        and s.street == "flop"  # a "donk" is a flop lead; turn/river -> probe/lead
    ),
    "turn_barrel": lambda s: (
        s.dominant_verb == "bet"
        and s.street == "turn"
        and s.hero_is_preflop_aggressor
        and s.hero_bet_prev_street  # bet the flop, betting again = second barrel
    ),
    "delayed_cbet": lambda s: (
        s.dominant_verb == "bet"
        and s.street == "turn"
        and s.hero_is_preflop_aggressor
        and s.prev_street_checked_through  # checked the flop, now bets the turn
    ),
    "probe_bet": lambda s: (
        s.dominant_verb == "bet"
        and s.street in ("turn", "river")
        and not s.hero_is_preflop_aggressor
        and not s.hero_in_position
        and s.prev_street_checked_through  # OOP leads after the aggressor declined
    ),
    "river_bet": lambda s: s.dominant_verb == "bet" and s.street == "river",
    "facing_bet_spot": lambda s: s.is_facing_bet,
    "in_position": lambda s: s.hero_in_position,
    "out_of_position": lambda s: not s.hero_in_position,
    # Decision class
    "value_bet_spot": lambda s: (
        s.dominant_verb in ("bet", "raise")
        and s.strength_bucket in ("premium", "strong")
    ),
    "bluff_spot": lambda s: (
        s.dominant_verb in ("bet", "raise")
        and s.strength_bucket in ("air", "marginal")
        and not s.has_strong_draw
    ),
    "semibluff_spot": lambda s: (
        s.dominant_verb in ("bet", "raise")
        and s.has_strong_draw
        and s.strength_bucket in ("air", "marginal", "vulnerable")
    ),
    "protection_bet_spot": lambda s: (
        s.dominant_verb == "bet"
        and s.strength_bucket in ("medium", "vulnerable")
        and s.composite in _WET
        and s.street != "river"  # nothing to protect once all cards are out
    ),
    "bluff_catch_spot": lambda s: (
        s.dominant_verb == "call"
        and s.is_facing_bet
        and s.strength_bucket in ("medium", "vulnerable", "marginal")
    ),
    "pot_control_spot": lambda s: (
        s.dominant_verb == "check" and s.strength_bucket in ("strong", "medium")
    ),
    "draw_spot": lambda s: bool(s.draws),
    # Hand class
    "top_pair": lambda s: s.made_hand.startswith("top_pair"),
    "overpair": lambda s: s.made_hand == "overpair",
    "set_or_better": lambda s: s.made_hand in _SET_OR_BETTER,
    "flush_draw": lambda s: s.has_flush_draw,
    "straight_draw": lambda s: s.has_straight_draw,
    # Board texture
    "wet_board": lambda s: s.composite in _WET,
    "paired_board": lambda s: s.pair_status in ("paired", "trips_on_board"),
    "monotone_board": lambda s: s.suit_distribution in (
        "monotone", "three_of_suit", "flush_on_board"
    ),
    "two_tone_board": lambda s: s.suit_distribution == "two_tone",
    # Equity / math
    "equity_favored": lambda s: s.hero_equity >= 0.55,
    "getting_a_price": lambda s: (
        s.is_facing_bet
        and s.break_even_equity is not None
        and s.hero_equity >= s.break_even_equity
    ),
}


def compute_postflop_tags(tag_input: PostflopTagInput) -> list[str]:
    """Every tag that fires for ``tag_input``, in registry order."""
    return [name for name, rule in TAG_REGISTRY.items() if rule(tag_input)]


def classify_postflop_archetype(tag_input: PostflopTagInput) -> str:
    """Pick the single strategic frame the LLM should write toward.

    Reads the dominant verb + hand strength (+ draws, board, facing-a-bet)
    exactly like the brief's "decision class" logic. Returns one of
    :data:`POSTFLOP_ARCHETYPES`. The frame the LLM receives in its SOLVER DATA
    block; it never decides this itself.
    """
    s = tag_input
    verb, bucket = s.dominant_verb, s.strength_bucket

    if verb == "bet":
        if bucket in ("premium", "strong"):
            return "value_bet"
        if s.has_strong_draw:  # no draws on the river, so never fires there
            return "semibluff"
        if bucket in ("medium", "vulnerable"):
            # Protection needs cards to come; on the river a medium hand betting
            # is thin value, not protection.
            protect = s.composite in _WET and s.street != "river"
            return "protection_bet" if protect else "value_bet"
        return "bluff"  # air / marginal, no real draw

    if verb == "raise":
        if bucket in ("premium", "strong"):
            return "value_raise"
        return "bluff_raise"

    if verb == "check":
        if bucket == "premium":
            return "trap_check"
        if bucket in ("strong", "medium"):
            return "pot_control_check"
        return "defensive_check"

    if verb == "call":
        if s.has_strong_draw and bucket in ("air", "marginal", "vulnerable"):
            return "call_drawing"
        if bucket in ("premium", "strong"):
            return "call_with_showdown"
        return "bluff_catch"

    if verb == "fold":
        return "fold_to_pressure"

    return "unclassified"


__all__ = [
    "POSTFLOP_ARCHETYPES",
    "TAG_REGISTRY",
    "PostflopTagInput",
    "classify_postflop_archetype",
    "compute_postflop_tags",
]
