"""User-facing skill tagger (Phase 3 of the engineering brief).

Maps the pipeline's computational outputs (preflop archetype + 38 preflop
concept tags + the postflop 42-tag library + scenario metadata) onto the
team's 42-skill catalog -- the labels the app surfaces to users when it
says "you're weak at Squeezing" or "you struggle with Facing a 3-Bet".

Per the user's strict-tagging direction: only fire a skill when the spot
clearly tests the concept. Most preflop questions get 2-4 skills, not 8.
False negatives are preferable to noise.

Rule format: each catalog entry is a ``SkillRule`` -- a Python predicate
over a :class:`SkillContext`. Lambdas for the simple cases (tag presence
or archetype membership), named functions for the few that need shape
checks. Easy to migrate to YAML later if non-engineers want to edit.

Coverage status as of v1:

* **Preflop spots, fires today (~15 skills)**: every skill in Section 1
  + the math/positional/blocker ones that derive from preflop facts.
* **Postflop spots, rules written but not exercised (~20 skills)**: maps
  to the postflop concept-tag library in
  ``pipeline.fact_extractor.concept_tags``. Will start firing when the
  postflop generation path is unblocked (waiting on PioSolver solves).
* **Not yet supported (~7 skills)**: marked ``# TODO Phase 4`` -- need
  new computational signals (tournament metadata, hand-reading
  heuristics, bet-sizing axes, etc.).

Adding/editing a skill: add or modify the row in :data:`SKILL_CATALOG`
and write a unit test in ``tests/test_skill_tagger.py``. The mapping
ships in column 39 of the output CSV ("skills", comma-separated).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pipeline.preflop.fact_extractor import PreflopFacts


# --- context dataclass ------------------------------------------------------
@dataclass(frozen=True)
class SkillContext:
    """Everything a skill predicate might inspect.

    Built once per question by either :func:`from_preflop_facts` or
    :func:`from_postflop_spot_data` (the latter is a stub until the
    postflop path lands real output). Keeping the shape unified means
    the same :data:`SKILL_CATALOG` works for both paths without an
    if-statement per rule.

    Fields are conservative: anything that *might* not be available
    in one path has a sensible default. Predicates should check
    ``path`` before reaching for postflop-only fields like
    ``board_texture``.
    """

    path: Literal["preflop", "postflop"]
    street: str  # "Preflop" / "Flop" / "Turn" / "River"

    # Hot inputs -- the boolean concept tags + the strategic archetype.
    concept_tags: frozenset[str] = frozenset()
    archetype: str = ""  # empty for postflop (no archetype layer there)

    # Spot metadata. Available on both paths.
    hand_class: str = ""
    hero_position: str = ""  # "UTG", "HJ", "CO", "BTN", "SB", "BB", ...

    # Game-format / stake context (from scenario config or pack metadata).
    game_format: str = "cash"  # "cash" or "tournament"
    stack_depth_bb: int = 100

    # Preflop action-history derivatives -- empty list / 0 for postflop.
    n_prior_raises: int = 0
    n_calls_after_open: int = 0  # calls that occurred AFTER the open AND
    # BEFORE any subsequent raise (used for facing-squeeze detection)

    # Postflop-only fields, filled in by from_postflop_spot_data.
    # Defaults keep preflop rules from accidentally tripping on these.
    board_texture: str = ""
    is_preflop_aggressor: bool = False  # hero opened/3bet/etc preflop
    dominant_is_aggressive: bool = False  # bet or raise on this street


# --- rule type --------------------------------------------------------------
SkillRule = Callable[[SkillContext], bool]


# --- helper: build a SkillContext from PreflopFacts -------------------------
def from_preflop_facts(
    facts: PreflopFacts,
    *,
    game_format: str = "cash",
    stack_depth_bb: int = 100,
) -> SkillContext:
    """Adapter: PreflopFacts -> SkillContext for the catalog to consume.

    Concept tags are recomputed here rather than read off a field --
    PreflopFacts doesn't carry them, and recomputing is cheap (pure
    Python on already-loaded data).
    """
    # Lazy import: keeps the postflop side of this module importable
    # without the preflop concept-tag stack being available.
    from pipeline.preflop.concept_tags import compute_concept_tags  # noqa: PLC0415
    from pipeline.preflop.grammars.types import PreflopActionType  # noqa: PLC0415

    history = facts.spot.node.history_before

    # Count raises (incl. all-ins).
    n_prior_raises = sum(
        1
        for a in history
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )

    # Count CALLs that happened AFTER the first raise but BEFORE any
    # subsequent raise. Non-zero => the most recent raise was preceded
    # by at least one caller => hero might be facing a squeeze (when
    # n_prior_raises >= 2) or have a squeeze opportunity (when
    # n_prior_raises == 1 and we're the last to act).
    n_calls_after_open = 0
    seen_first_raise = False
    seen_second_raise = False
    for a in history:
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN):
            if not seen_first_raise:
                seen_first_raise = True
            else:
                seen_second_raise = True
                break
        elif seen_first_raise and a.action_type is PreflopActionType.CALL:
            n_calls_after_open += 1
    # If we never saw a second raise, n_calls_after_open is "calls after
    # the open" (used for squeeze_opportunity detection); if we did, it's
    # "calls between open and 3-bet" (used for facing-squeeze detection).
    # The semantics are identical for both rules so a single field works.
    del seen_second_raise  # used implicitly via the loop break

    return SkillContext(
        path="preflop",
        street="Preflop",
        concept_tags=frozenset(compute_concept_tags(facts)),
        archetype=facts.archetype,
        hand_class=facts.spot.hero_hand_class,
        hero_position=facts.spot.node.actor,
        game_format=game_format,
        stack_depth_bb=stack_depth_bb,
        n_prior_raises=n_prior_raises,
        n_calls_after_open=n_calls_after_open,
    )


def from_postflop_spot_data(
    spot: object,  # SpotData -- typed loosely to avoid import-time coupling
    *,
    game_format: str = "cash",
    stack_depth_bb: int = 100,
) -> SkillContext:
    """Adapter: postflop SpotData -> SkillContext.

    STUB. Builds a usable context for the rules that only need
    ``concept_tags`` + ``street`` + position, but several fields
    (``is_preflop_aggressor``, ``dominant_is_aggressive``,
    ``board_texture``) will need to be wired when the postflop path
    actually generates output. Marked clearly with ``# TODO Phase 4``
    in the rules that depend on them.
    """
    # Lazy import so this module is usable preflop-only without the
    # postflop concept-tag chain being importable.
    from pipeline.fact_extractor.concept_tags.registry import (  # noqa: PLC0415
        compute_tags,
    )

    # SpotData has .metadata (SpotMetadata with street, scenario, etc.)
    # and the rest of its fields. Read defensively -- the postflop path
    # is unverified.
    metadata = getattr(spot, "metadata", None)
    street = ""
    hero_position = ""
    hand_class_label = ""
    if metadata is not None:
        street = (getattr(metadata, "street", "") or "").capitalize()
        hero_position = getattr(metadata, "hero_position", "") or ""
    hand_class_obj = getattr(spot, "hand_class", None)
    if hand_class_obj is not None:
        hand_class_label = getattr(hand_class_obj, "label", "") or ""

    tags = frozenset(compute_tags(spot)) if spot is not None else frozenset()
    board_texture_obj = getattr(spot, "board_texture", None)
    board_texture = ""
    if board_texture_obj is not None:
        board_texture = getattr(board_texture_obj, "composite", "") or ""

    return SkillContext(
        path="postflop",
        street=street,
        concept_tags=tags,
        archetype="",  # postflop has no archetype layer
        hand_class=hand_class_label,
        hero_position=hero_position,
        game_format=game_format,
        stack_depth_bb=stack_depth_bb,
        board_texture=board_texture,
        # TODO Phase 4: wire is_preflop_aggressor + dominant_is_aggressive
        # from postflop facts when the postflop generation path is verified.
    )


# --- the 42-skill catalog ---------------------------------------------------
# Each entry is (canonical skill name, predicate). Order matches the user's
# numbered list so it's easy to diff against the source doc.
#
# Predicates should be strict: tag only when the spot clearly tests the
# concept. False negatives > noisy positives.

_ARCHETYPES_3BET = frozenset({"3bet_for_value", "3bet_as_bluff"})
_ARCHETYPES_4BET = frozenset({"4bet_for_value", "4bet_as_bluff"})
_ARCHETYPES_5BET = frozenset({"5bet_for_value", "5bet_as_bluff"})
_ARCHETYPES_SQUEEZE = frozenset({"squeeze_for_value", "squeeze_as_bluff"})
_ARCHETYPES_OPEN = frozenset({"open_for_value", "fold_outranged"})
_ARCHETYPES_CALL = frozenset({"call_for_value", "call_for_implied_odds"})
_ARCHETYPES_ALL_IN = frozenset({"all_in_for_value", "all_in_as_bluff"})
_ARCHETYPES_FOLD = frozenset({"fold_dominated", "fold_pot_odds"})

_BLIND_TAGS = frozenset({"small_blind", "big_blind"})
_LATE_POS = frozenset({"CO", "BTN"})


def _is_facing_squeeze(c: SkillContext) -> bool:
    """Facing a squeeze: there were >=2 prior raises AND at least one call
    between the first two. Distinguishes from a vanilla 3-bet response."""
    return c.n_prior_raises >= 2 and c.n_calls_after_open >= 1


def _is_facing_3bet_not_squeeze(c: SkillContext) -> bool:
    """Vanilla facing-a-3-bet: 2 prior raises with no caller in between
    (otherwise it's a squeeze, which has its own skill)."""
    return "facing_3bet" in c.concept_tags and c.n_calls_after_open == 0


SKILL_CATALOG: dict[str, SkillRule] = {
    # --- Section 1: Preflop (9) ---
    # Strict: "Preflop Hand Selection" only fires on opening decisions
    # (RFI). Every other preflop spot is covered by 3-Betting / Facing
    # a 3-Bet / etc., so always-firing here would be noise.
    "Preflop Hand Selection": lambda c: c.archetype in _ARCHETYPES_OPEN,
    "3-Betting": lambda c: c.archetype in _ARCHETYPES_3BET,
    "Facing a 3-Bet": _is_facing_3bet_not_squeeze,
    "4-Betting": lambda c: c.archetype in _ARCHETYPES_4BET,
    "Facing a 4-Bet": lambda c: "facing_4bet_plus" in c.concept_tags,
    "Squeezing": lambda c: c.archetype in _ARCHETYPES_SQUEEZE,
    "Facing a Squeeze": _is_facing_squeeze,
    "Blind Defense": lambda c: (
        bool(c.concept_tags & _BLIND_TAGS)
        and ("facing_single_raise" in c.concept_tags
             or "facing_3bet" in c.concept_tags
             or "facing_4bet_plus" in c.concept_tags)
        and "bvb_spot" not in c.concept_tags  # BvB has its own skill
    ),
    "Blind vs. Blind Play": lambda c: "bvb_spot" in c.concept_tags,

    # --- Section 2: Betting & Aggression (13) -- mostly POSTFLOP ---
    # TODO Phase 4: C-Betting / Facing a C-Bet need
    # is_preflop_aggressor + dominant_is_aggressive wired through the
    # postflop adapter. Off until then.
    "C-Betting": lambda c: (
        c.path == "postflop" and c.street == "Flop"
        and c.is_preflop_aggressor and c.dominant_is_aggressive
    ),
    "Facing a C-Bet": lambda c: (
        c.path == "postflop" and c.street == "Flop"
        and not c.is_preflop_aggressor
        # The cleanest signal would be "the preflop raiser bet" but
        # we'd need previous-action data. Approximation: defender facing
        # any aggression on the flop.
        and "facing_donk_spot" not in c.concept_tags  # donk has its own skill
        and "facing_check_raise_spot" not in c.concept_tags
    ),
    "Check-Raising": lambda c: "check_raise_spot" in c.concept_tags,
    "Facing a Check-Raise": lambda c: "facing_check_raise_spot" in c.concept_tags,
    "Donk Betting": lambda c: "donk_bet_spot" in c.concept_tags,
    "Facing a Donk Bet": lambda c: "facing_donk_spot" in c.concept_tags,
    "Probe Betting": lambda c: "probe_bet_spot" in c.concept_tags,
    "Facing a Probe Bet": lambda c: "facing_probe_spot" in c.concept_tags,
    "Overbetting": lambda c: "overbet_spot" in c.concept_tags,
    "Facing an Overbet": lambda c: "facing_overbet_spot" in c.concept_tags,
    # TODO Phase 4: Bet Sizing is too broad without a sizing-choice axis.
    # Off until we have a "this spot has multiple meaningful sizings" signal.
    "Bet Sizing": lambda _c: False,
    "Value Betting": lambda c: (
        "thin_value_spot" in c.concept_tags
        or "merged_value_spot" in c.concept_tags
    ),
    "Bluffing": lambda c: "bluff_spot" in c.concept_tags,

    # --- Section 3: Defense & Response (3) -- POSTFLOP ---
    "Bluff Catching": lambda c: "bluffcatch_spot" in c.concept_tags,
    "Floating": lambda c: "float_call_spot" in c.concept_tags,
    "Pot Control": lambda c: "pot_control_spot" in c.concept_tags,

    # --- Section 4: Math & Theory (7) ---
    # Pot Odds: preflop call/fold spots where the math is meaningful
    # (dominant action is Call or Fold and the spot isn't a pure no-equity
    # snap-fold). Postflop fires on facing_donk / facing_overbet which
    # are inherently pot-odds decisions.
    "Pot Odds": lambda c: (
        (c.archetype in _ARCHETYPES_CALL or c.archetype in _ARCHETYPES_FOLD)
        or "facing_donk_spot" in c.concept_tags
        or "facing_overbet_spot" in c.concept_tags
    ),
    "Implied Odds": lambda c: (
        c.archetype == "call_for_implied_odds"
        or "implied_odds_call" in c.concept_tags
    ),
    "Reverse Implied Odds": lambda c: (
        "reverse_implied_odds_call" in c.concept_tags
    ),
    "Minimum Defense Frequency (MDF)": lambda c: (
        "mdf_defense_threshold" in c.concept_tags
    ),
    # TODO Phase 4: Combinatorics is hard to scope -- nearly every
    # spot involves counting villain combos. Off until we have a
    # narrower "this spot's right answer requires combo counting" signal.
    "Combinatorics": lambda _c: False,
    "Equity Realization": lambda c: (
        "equity_under_realized" in c.concept_tags
        or "equity_over_realized" in c.concept_tags
    ),
    # TODO Phase 4: SPR needs stack-to-pot computation. Off until then.
    "Stack-to-Pot Ratio (SPR)": lambda _c: False,

    # --- Section 5: Hand Analysis & Decision Making (3) ---
    # TODO Phase 4: Hand Reading is universal in poker; would fire on
    # every question. Need a specific "hero must read a narrow range"
    # signal before turning on. Off for now.
    "Hand Reading": lambda _c: False,
    # Blockers fires on the pedagogically MEANINGFUL blocker tags only.
    # `blocks_villain_top_value` is excluded -- it fires whenever hero's
    # cards remove ANY combo in villain's top 5, which happens on most
    # hands without actually teaching a blocker lesson. Strict tagging
    # cares about high-impact / directional blocker effects only.
    "Blockers & Card Removal": lambda c: bool(
        c.concept_tags & {
            "ace_blocker",                  # hero holds an A (preflop, high impact)
            "king_blocker",                 # hero holds a K (preflop, high impact)
            "blocks_value_unblocks_bluffs", # postflop, directional
            "blocks_bluffs_unblocks_value", # postflop, directional
        }
    ),
    "Range Polarization": lambda c: "villain_polarized" in c.concept_tags,

    # --- Section 6: Positional & Situational (4) ---
    # In Position: hero acts after villain. Preflop, this means hero is
    # last to act (BTN closes vs. open, IP caller vs. 3-bet, etc.).
    # Approximation: late position AND there's a villain action history.
    "In Position Play": lambda c: (
        c.hero_position in _LATE_POS
        and c.n_prior_raises >= 1
    ),
    "Out of Position Play": lambda c: (
        # Blinds defending, or earlier positions in a non-RFI spot.
        bool(c.concept_tags & _BLIND_TAGS)
        and c.n_prior_raises >= 1
    ),
    "Multiway Pot Strategy": lambda c: "multiway_pot" in c.concept_tags,
    # TODO Phase 4: Drawing Hand Strategy is postflop-specific and needs
    # hand_class to be a draw type (flush_draw / straight_draw / combo_draw).
    # Postflop hand_class.label could be checked here when the postflop
    # path lands.
    "Drawing Hand Strategy": lambda c: (
        c.path == "postflop"
        and ("draw" in c.hand_class.lower() if c.hand_class else False)
    ),

    # --- Section 7: Tournament (3) ---
    "Short Stack Tournament Strategy": lambda c: (
        c.game_format == "tournament"
        and ("short_stack" in c.concept_tags
             or "short_stack_tournament" in c.concept_tags)
    ),
    "Tournament Blind vs. Blind": lambda c: (
        c.game_format == "tournament" and "bvb_spot" in c.concept_tags
    ),
    # TODO Phase 4: ICM needs tournament structure data (payouts, blinds
    # remaining, stack distribution). No current source. Off until then.
    "ICM & Tournament Pressure": lambda _c: False,
}


# --- compute -----------------------------------------------------------------
def compute_skills(ctx: SkillContext) -> list[str]:
    """Run every catalog predicate and return the firing skill names.

    Order matches :data:`SKILL_CATALOG` insertion order (= the user's
    numbered list) so the CSV stays diff-friendly across batches.
    """
    return [name for name, rule in SKILL_CATALOG.items() if rule(ctx)]


# --- coverage introspection (used by the smoke test) -----------------------
@dataclass(frozen=True)
class SkillCoverageReport:
    """Summary of which skills the catalog can/can't fire today."""

    always_off: tuple[str, ...] = field(default_factory=tuple)
    # Skills hard-coded to return False (TODO markers).
    preflop_fireable: tuple[str, ...] = field(default_factory=tuple)
    postflop_fireable: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "SKILL_CATALOG",
    "SkillContext",
    "SkillCoverageReport",
    "SkillRule",
    "compute_skills",
    "from_postflop_spot_data",
    "from_preflop_facts",
]
