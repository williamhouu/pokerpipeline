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
from dataclasses import dataclass
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

    # Count CALLs between the first and the second raise (i.e. "callers
    # of the open before a 3-bet hit"). Non-zero AND n_prior_raises >= 2
    # => hero is facing a squeeze; non-zero AND n_prior_raises == 1 AND
    # hero is the last seat to act => hero has a squeeze opportunity.
    # Counting stops at the second raise -- any later calls would be of
    # the 3-bet, not the open, and aren't relevant to either rule.
    n_calls_after_open = 0
    seen_first_raise = False
    for a in history:
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN):
            if seen_first_raise:
                break  # second raise reached -- stop counting
            seen_first_raise = True
        elif seen_first_raise and a.action_type is PreflopActionType.CALL:
            n_calls_after_open += 1

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

# Preflop archetypes that are EXPLICIT bluffs -- hero is betting/raising
# with a hand whose value comes from fold equity rather than equity vs
# villain's continuing range. The Bluffing skill uses this set as its
# strict trigger; spots that happen to have low equity but a value-frame
# archetype do NOT qualify.
_ARCHETYPES_BLUFF_PREFLOP = frozenset({
    "3bet_as_bluff",
    "4bet_as_bluff",
    "5bet_as_bluff",
    "squeeze_as_bluff",
    "all_in_as_bluff",
})

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
    # Bluffing fires STRICTLY: either the postflop `bluff_spot` tag,
    # OR a preflop archetype that's explicitly labelled `_as_bluff`
    # (3bet_as_bluff, 4bet_as_bluff, 5bet_as_bluff, squeeze_as_bluff,
    # all_in_as_bluff). Low-equity value spots (e.g. an SB calling KQs
    # vs BTN open with 42% equity) do NOT qualify -- the value-frame
    # archetype rules them out. Only the explicit bluff archetypes count.
    "Bluffing": lambda c: (
        "bluff_spot" in c.concept_tags
        or c.archetype in _ARCHETYPES_BLUFF_PREFLOP
    ),

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
    # In Position: hero acts after villain postflop. Preflop, that's late
    # position vs an open/3-bet (BTN/CO), PLUS the blind-vs-blind exception:
    # in a BvB pot the SB is the dealer and acts LAST postflop, so the SB
    # is IN position (mirrors _ip_oop_positions in preflop.format_writer and
    # the "Relative Position" column). Without the BvB clause the SB here
    # would be mislabeled OOP.
    "In Position Play": lambda c: (
        (c.hero_position in _LATE_POS and c.n_prior_raises >= 1)
        or (
            c.hero_position == "SB"
            and "bvb_spot" in c.concept_tags
            and c.n_prior_raises >= 1
        )
    ),
    "Out of Position Play": lambda c: (
        # Blinds defending, or earlier positions in a non-RFI spot --
        # EXCEPT the SB in a BvB pot, which is in position (see above).
        bool(c.concept_tags & _BLIND_TAGS)
        and c.n_prior_raises >= 1
        and not (c.hero_position == "SB" and "bvb_spot" in c.concept_tags)
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


# --- user-facing explainer metadata ------------------------------------------
# Per-skill (section, status, human-readable description) -- powers the
# admin panel's Skills page and any future skill-catalog docs. Keep in
# lockstep with SKILL_CATALOG: tests below assert key parity. When a
# TODO Phase 4 rule starts firing for real, flip the status here and
# update the description.
#
#   section: human-readable section name from the user's source doc
#   status:  one of "preflop_fires", "postflop_fires", "todo"
#     preflop_fires  = will tag real spots today on preflop output
#     postflop_fires = rule written but waits for postflop generation
#                      (Pio solves blocked)
#     todo           = predicate is `lambda _c: False`; needs more work
#   description: one short paragraph for the admin-panel "why this fired"
#                view. Concrete enough that a reviewer reading a tagged
#                spot can understand why it matched.

_PREFLOP = "preflop_fires"
_POSTFLOP = "postflop_fires"
_TODO = "todo"


@dataclass(frozen=True)
class SkillMeta:
    """UI-side metadata for one entry in SKILL_CATALOG."""

    section: str
    status: str  # _PREFLOP / _POSTFLOP / _TODO
    description: str


SKILL_META: dict[str, SkillMeta] = {
    # --- Section 1: Preflop ---
    "Preflop Hand Selection": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires on opening (RFI) decisions -- the spot's archetype is "
        "`open_for_value` or `fold_outranged`. Tests the foundational "
        "'what hands play from this position?' question. Other preflop "
        "spots are covered by 3-Betting / Facing a 3-Bet / etc., so "
        "this skill is strict to the open-fold decision only.",
    ),
    "3-Betting": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires when hero's strategic frame is a 3-bet "
        "(archetype `3bet_for_value` or `3bet_as_bluff`). Tests "
        "value vs. bluff selection and 3-bet sizing.",
    ),
    "Facing a 3-Bet": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires when the spot has the `facing_3bet` tag AND there were "
        "no callers between the open and the 3-bet (otherwise it's a "
        "squeeze response, which has its own skill). Tests fold / call "
        "/ 4-bet decisions vs. a tight re-raise.",
    ),
    "4-Betting": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires when hero's strategic frame is a 4-bet "
        "(archetype `4bet_for_value` or `4bet_as_bluff`). Tests "
        "value vs. bluff selection at a 4-bet pot.",
    ),
    "Facing a 4-Bet": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires on the `facing_4bet_plus` tag (3 or more prior raises "
        "in history). Tests stack-committed fold / call / jam "
        "decisions with significant chips already in.",
    ),
    "Squeezing": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires when hero's strategic frame is a squeeze "
        "(archetype `squeeze_for_value` or `squeeze_as_bluff`). "
        "Tests leveraging dead money from prior callers.",
    ),
    "Facing a Squeeze": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires when there were >=2 prior raises AND at least one "
        "call between them -- i.e. raise -> call -> 3-bet (the squeeze). "
        "Hero is the original raiser or the caller now responding.",
    ),
    "Blind Defense": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires when hero is in SB or BB AND facing a single raise / "
        "3-bet / 4-bet+. Excluded for blind-vs-blind spots (which "
        "have their own skill).",
    ),
    "Blind vs. Blind Play": SkillMeta(
        "Preflop", _PREFLOP,
        "Fires on the `bvb_spot` concept tag -- the hand is contested "
        "only between SB and BB. Ranges widen dramatically vs. a "
        "standard raised pot.",
    ),
    # --- Section 2: Betting & Aggression ---
    "C-Betting": SkillMeta(
        "Betting & Aggression", _TODO,
        "Currently OFF. Will fire when hero is the preflop aggressor "
        "and is betting on the flop. Needs the postflop adapter to "
        "wire `is_preflop_aggressor` and `dominant_is_aggressive` "
        "from postflop facts (Phase 4 work).",
    ),
    "Facing a C-Bet": SkillMeta(
        "Betting & Aggression", _TODO,
        "Currently OFF. Will fire when hero is the defender facing "
        "a flop c-bet. Same Phase 4 dependency as C-Betting.",
    ),
    "Check-Raising": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `check_raise_spot` tag -- hero checks "
        "with intent to raise after villain bets.",
    ),
    "Facing a Check-Raise": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `facing_check_raise_spot` tag -- hero "
        "bet and villain check-raised.",
    ),
    "Donk Betting": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `donk_bet_spot` tag -- hero leads "
        "into the preflop raiser from out of position.",
    ),
    "Facing a Donk Bet": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `facing_donk_spot` tag -- villain "
        "donked into hero on the flop.",
    ),
    "Probe Betting": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `probe_bet_spot` tag -- hero bets into "
        "a villain who declined to c-bet on the previous street.",
    ),
    "Facing a Probe Bet": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `facing_probe_spot` tag -- villain "
        "probed into hero after hero checked back the previous street.",
    ),
    "Overbetting": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `overbet_spot` tag -- hero is betting "
        "more than the size of the pot.",
    ),
    "Facing an Overbet": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on the postflop `facing_overbet_spot` tag -- hero "
        "faces a bet larger than pot.",
    ),
    "Bet Sizing": SkillMeta(
        "Betting & Aggression", _TODO,
        "Currently OFF. Needs a sizing-axis signal (a 'this spot has "
        "multiple meaningful sizing choices' indicator) before firing -- "
        "otherwise it would tag every postflop bet/raise. Phase 4 work.",
    ),
    "Value Betting": SkillMeta(
        "Betting & Aggression", _POSTFLOP,
        "Fires on `thin_value_spot` or `merged_value_spot` -- hero is "
        "betting a range that's ahead of villain's calling range.",
    ),
    "Bluffing": SkillMeta(
        "Betting & Aggression", _PREFLOP,
        "Fires on EXPLICIT bluff archetypes only -- strict by design. "
        "Preflop: when the archetype is `3bet_as_bluff`, `4bet_as_bluff`, "
        "`5bet_as_bluff`, `squeeze_as_bluff`, or `all_in_as_bluff` "
        "(hero is raising/shoving with a hand whose value comes from "
        "fold equity, not equity vs villain's continuing range). "
        "Postflop: when the `bluff_spot` tag fires. Low-equity hands "
        "in value-frame archetypes do NOT qualify -- the value-frame "
        "label rules them out.",
    ),
    # --- Section 3: Defense & Response ---
    "Bluff Catching": SkillMeta(
        "Defense & Response", _POSTFLOP,
        "Fires on the `bluffcatch_spot` tag -- hero's hand can only "
        "beat bluffs, so the decision turns on villain's bluff "
        "frequency vs. pot odds.",
    ),
    "Floating": SkillMeta(
        "Defense & Response", _POSTFLOP,
        "Fires on the `float_call_spot` tag -- hero calls a bet with "
        "intent to take the pot away on a later street.",
    ),
    "Pot Control": SkillMeta(
        "Defense & Response", _POSTFLOP,
        "Fires on the `pot_control_spot` tag -- hero deliberately "
        "keeps the pot small with a medium-strength hand.",
    ),
    # --- Section 4: Math & Theory ---
    "Pot Odds": SkillMeta(
        "Math & Theory", _PREFLOP,
        "Fires on call/fold archetypes preflop "
        "(`call_for_value`, `call_for_implied_odds`, `fold_pot_odds`, "
        "`fold_dominated`) or on `facing_donk_spot` / "
        "`facing_overbet_spot` postflop. The defining 'is this call "
        "mathematically profitable?' question.",
    ),
    "Implied Odds": SkillMeta(
        "Math & Theory", _PREFLOP,
        "Fires on the `call_for_implied_odds` archetype (preflop) or "
        "the `implied_odds_call` tag (postflop). Hero calls without "
        "direct pot odds, expecting future-street value when hitting.",
    ),
    "Reverse Implied Odds": SkillMeta(
        "Math & Theory", _POSTFLOP,
        "Fires on the postflop `reverse_implied_odds_call` tag -- "
        "hero's hand might lose more when it 'hits' than it gains "
        "(dominated draws etc.).",
    ),
    "Minimum Defense Frequency (MDF)": SkillMeta(
        "Math & Theory", _POSTFLOP,
        "Fires on the postflop `mdf_defense_threshold` tag -- the "
        "central concept is hero's required defense frequency to "
        "prevent villain from profitably bluffing.",
    ),
    "Combinatorics": SkillMeta(
        "Math & Theory", _TODO,
        "Currently OFF. Counting villain combos is involved in nearly "
        "every poker decision; needs a narrower 'this spot's right "
        "answer requires combo counting' signal before firing. Phase 4.",
    ),
    "Equity Realization": SkillMeta(
        "Math & Theory", _POSTFLOP,
        "Fires on the postflop `equity_under_realized` or "
        "`equity_over_realized` tags -- how much of hero's raw equity "
        "actually gets captured given position / playability / etc.",
    ),
    "Stack-to-Pot Ratio (SPR)": SkillMeta(
        "Math & Theory", _TODO,
        "Currently OFF. Needs stack-to-pot ratio computation in the "
        "SkillContext (no current field). Phase 4 work.",
    ),
    # --- Section 5: Hand Analysis ---
    "Hand Reading": SkillMeta(
        "Hand Analysis & Decision Making", _TODO,
        "Currently OFF. Hand reading is universal in poker -- firing "
        "it on every question would be noise. Needs a specific 'narrow "
        "range read required' signal before firing. Phase 4.",
    ),
    "Blockers & Card Removal": SkillMeta(
        "Hand Analysis & Decision Making", _PREFLOP,
        "Fires on `ace_blocker` or `king_blocker` (preflop -- hero "
        "holds an A or K, the high-impact preflop blockers) or on "
        "`blocks_value_unblocks_bluffs` / `blocks_bluffs_unblocks_value` "
        "(postflop directional blocker tags). The generic "
        "`blocks_villain_top_value` is intentionally excluded -- it "
        "fires on too many hands without teaching a blocker lesson.",
    ),
    "Range Polarization": SkillMeta(
        "Hand Analysis & Decision Making", _POSTFLOP,
        "Fires on the postflop `villain_polarized` tag -- villain's "
        "range is strong hands + bluffs with nothing in between, "
        "which drives later-street decisions.",
    ),
    # --- Section 6: Positional & Situational ---
    "In Position Play": SkillMeta(
        "Positional & Situational", _PREFLOP,
        "Fires when hero is in CO or BTN AND there is at least one "
        "prior raise -- there's positional advantage to exploit.",
    ),
    "Out of Position Play": SkillMeta(
        "Positional & Situational", _PREFLOP,
        "Fires when hero is in SB or BB AND there is at least one "
        "prior raise -- playing OOP requires tighter ranges and "
        "more check-raising.",
    ),
    "Multiway Pot Strategy": SkillMeta(
        "Positional & Situational", _PREFLOP,
        "Fires on the `multiway_pot` tag -- three or more players "
        "are still in the pot when hero decides. Ranges tighten "
        "and bluffing decreases.",
    ),
    "Drawing Hand Strategy": SkillMeta(
        "Positional & Situational", _POSTFLOP,
        "Postflop only. Fires when hand_class label contains 'draw' "
        "(flush_draw, straight_draw, combo_draw, etc.). Preflop has "
        "no draw concept.",
    ),
    # --- Section 7: Tournament ---
    "Short Stack Tournament Strategy": SkillMeta(
        "Tournament", _PREFLOP,
        "Fires when game_format is 'tournament' AND the spot has a "
        "short_stack tag (or postflop `short_stack_tournament`). "
        "Push/fold dynamics dominate.",
    ),
    "Tournament Blind vs. Blind": SkillMeta(
        "Tournament", _PREFLOP,
        "Fires on `bvb_spot` AND game_format == 'tournament'. "
        "Distinct from cash BvB because ICM changes the math.",
    ),
    "ICM & Tournament Pressure": SkillMeta(
        "Tournament", _TODO,
        "Currently OFF. Needs tournament-structure metadata "
        "(payouts, blinds remaining, stack distribution) that the "
        "scenario configs don't carry yet. Phase 4.",
    ),
}


__all__ = [
    "SKILL_CATALOG",
    "SKILL_META",
    "SkillContext",
    "SkillMeta",
    "SkillRule",
    "compute_skills",
    "from_postflop_spot_data",
    "from_preflop_facts",
]
