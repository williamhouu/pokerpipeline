"""PLO user-facing skill tagger.

Maps a spot's computational outputs (archetype + concept tags + position +
action history) onto the PLO skill catalog -- the labels the app surfaces
("you're weak at Nut-Flush Awareness"). Mirrors the NLHE tagger
(:mod:`pipeline.skill_tagger`) but is **preflop-only** (PLO postflop solves
don't exist yet) and adds the PLO hand-reading skills, which are the product's
real edge.

Strict tagging, per the design: only fire a skill when the spot clearly tests
it. A typical PLO question gets 2-5 skills. The hand-reading skills key off the
hand-structure concept tags but are gated -- on the action (a folded hand isn't
testing suitedness) and against double-counting (an AAxx hand's lesson is
Big-Pair Construction, not generic Suitedness) -- so they don't over-fire into
noise.

Catalog (from docs/plo_port_assessment.md):
  * A -- carry-over preflop decisions (hand selection, 3-bet/4-bet/squeeze and
    facing them, blind play, pot odds, position, multiway).
  * B -- PLO hand-reading (Suitedness, Rundowns & Connectivity, Dangler
    Awareness, Nut-Flush Awareness, Big-Pair Construction, Nuttedness & Non-Nut
    Traps, Nut Blockers & Card Removal).
  * C -- math / dynamics (Implied Odds, Reverse Implied Odds, Pot-Limit Bet
    Sizing).
  * D -- postflop skills: out of scope until a PLO postflop path exists.

Thresholds / rule shapes are tunable; refine against graded output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.pack import PloActionType
from pipeline.plo.spot_tags import compute_plo_concept_tags

_AGGRESSIVE = {
    PloActionType.RAISE,
    PloActionType.MIN_RAISE,
    PloActionType.ALL_IN,
}
_RAISE_SIZED = {PloActionType.RAISE, PloActionType.MIN_RAISE}
_MIN_SQUEEZE_RAISES = 2


# --- context ---------------------------------------------------------------
@dataclass(frozen=True)
class PloSkillContext:
    """The signals a PLO skill predicate inspects, built from PloFacts."""

    tags: frozenset[str]
    archetype: str
    hero_position: str
    dominant_is_fold: bool
    dominant_is_aggressive: bool
    n_prior_raises: int
    n_calls_after_open: int  # calls between the open and the next raise
    multiple_raise_sizes: bool  # node offers >1 raise size (a sizing choice)


def from_plo_facts(facts: PloFacts) -> PloSkillContext:
    """Adapter: PloFacts -> PloSkillContext."""
    history = facts.spot.node.history_before
    n_prior_raises = sum(1 for a in history if a.action in _AGGRESSIVE)

    # Calls between the first and second raise -- the squeeze signal.
    n_calls_after_open = 0
    seen_first_raise = False
    for a in history:
        if a.action in _AGGRESSIVE:
            if seen_first_raise:
                break
            seen_first_raise = True
        elif seen_first_raise and a.action is PloActionType.CALL:
            n_calls_after_open += 1

    raise_sizes = {
        (a.action.action, a.action.raise_pct)
        for a in facts.spot.node.actions
        if a.action.action in _RAISE_SIZED
    }
    dom = facts.spot.dominant_action
    return PloSkillContext(
        tags=frozenset(compute_plo_concept_tags(facts)),
        archetype=facts.archetype,
        hero_position=facts.spot.node.actor,
        dominant_is_fold=dom.startswith("Fold"),
        dominant_is_aggressive=dom.startswith(("Raise", "Min-raise", "All-in")),
        n_prior_raises=n_prior_raises,
        n_calls_after_open=n_calls_after_open,
        multiple_raise_sizes=len(raise_sizes) > 1,
    )


# --- catalog ---------------------------------------------------------------
SkillRule = Callable[[PloSkillContext], bool]

_OPEN = frozenset({"open_for_value", "open_fold"})
_3BET = frozenset({"3bet_for_value", "3bet_as_bluff"})
_4BET = frozenset({"4bet_for_value", "4bet_as_bluff"})
_SQUEEZE = frozenset({"squeeze_for_value", "squeeze_as_bluff"})
_CALL = frozenset({"call_for_value", "call_for_implied_odds", "call_allin"})
_FOLD = frozenset({"fold_dominated", "fold_pot_odds"})
_BLIND_TAGS = frozenset({"small_blind", "big_blind"})
_LATE = frozenset({"CO", "BU"})


def _facing_squeeze(c: PloSkillContext) -> bool:
    return c.n_prior_raises >= _MIN_SQUEEZE_RAISES and c.n_calls_after_open >= 1


def _facing_3bet_not_squeeze(c: PloSkillContext) -> bool:
    return "facing_3bet" in c.tags and c.n_calls_after_open == 0


def _continues(c: PloSkillContext) -> bool:
    """Hero is keeping the hand in (not folding) -- the hand's shape matters."""
    return not c.dominant_is_fold


SKILL_CATALOG: dict[str, SkillRule] = {
    # --- A: carry-over preflop decisions ---
    "Preflop Hand Selection": lambda c: c.archetype in _OPEN,
    "3-Betting": lambda c: c.archetype in _3BET,
    "Facing a 3-Bet": _facing_3bet_not_squeeze,
    "4-Betting": lambda c: c.archetype in _4BET,
    "Facing a 4-Bet": lambda c: "facing_4bet_plus" in c.tags,
    "Squeezing": lambda c: c.archetype in _SQUEEZE,
    "Facing a Squeeze": _facing_squeeze,
    "Blind Defense": lambda c: (
        bool(c.tags & _BLIND_TAGS)
        and c.n_prior_raises >= 1
        and "bvb_spot" not in c.tags
    ),
    "Blind vs. Blind Play": lambda c: "bvb_spot" in c.tags,
    "Pot Odds": lambda c: c.archetype in _CALL or c.archetype in _FOLD,
    "In Position Play": lambda c: (
        (c.hero_position in _LATE and c.n_prior_raises >= 1)
        or (c.hero_position == "SB" and "bvb_spot" in c.tags and c.n_prior_raises >= 1)
    ),
    "Out of Position Play": lambda c: (
        bool(c.tags & _BLIND_TAGS)
        and c.n_prior_raises >= 1
        and not (c.hero_position == "SB" and "bvb_spot" in c.tags)
    ),
    "Multiway Pot Strategy": lambda c: "multiway_pot" in c.tags,

    # --- B: PLO hand-reading (the edge) ---
    # Suitedness is the lesson when a hand's value leans on its suits AND it's
    # not an obvious big-pair hand (those teach Big-Pair Construction instead).
    "Suitedness": lambda c: (
        "double_suited" in c.tags
        and _continues(c)
        and not (c.tags & {"pocket_aces", "pocket_kings"})
    ),
    "Rundowns & Connectivity": lambda c: (
        bool(c.tags & {"rundown", "one_gap_rundown", "two_gap_rundown", "wrap_potential"})
        and "unpaired_hand" in c.tags
        and _continues(c)
    ),
    "Dangler Awareness": lambda c: "has_dangler" in c.tags,
    "Nut-Flush Awareness": lambda c: "nut_flush_potential" in c.tags and _continues(c),
    "Big-Pair Construction": lambda c: bool(c.tags & {"pocket_aces", "pocket_kings"}),
    # Non-nut traps: a bare ace (no nut-flush potential) or an all-low holding
    # that makes non-nut flushes / straights.
    "Nuttedness & Non-Nut Traps": lambda c: bool(c.tags & {"bare_ace", "low_cards"}),
    # Nut blockers: an ace blocks villain's AA, a suited ace blocks the nut
    # flush -- the card-removal that justifies aggression (a 3-bet/4-bet/squeeze
    # leaning on blockers). Gated to aggressive spots so it fires when the
    # blocker is the REASON for the bet, not on every hand that holds an ace.
    # (The blocks_villain_* tags already require a villain, so opens never fire.)
    "Nut Blockers & Card Removal": lambda c: (
        c.dominant_is_aggressive
        and bool(c.tags & {"blocks_villain_value", "blocks_villain_nut_flush"})
    ),

    # --- C: math / dynamics ---
    # Implied odds: calling a well-shaped speculative hand for the big pots you
    # win when it connects. The call_for_implied_odds archetype IS this
    # decision; matches the NLHE 'Implied Odds' skill. The positive counterpart
    # to reverse implied odds (both fire on a trappy speculative call).
    "Implied Odds": lambda c: c.archetype == "call_for_implied_odds",
    # Reverse implied odds: calling speculatively with a non-nut hand that can
    # make second-best hands (the classic PLO money-loser).
    "Reverse Implied Odds": lambda c: (
        c.archetype == "call_for_implied_odds"
        and bool(c.tags & {"bare_ace", "low_cards", "has_dangler"})
    ),
    # Pot-limit bet sizing is only a *decision* when the tree offers more than
    # one raise size. This pack has a single pot-sized raise, so it stays off
    # here -- and lights up automatically on a multi-size tree.
    "Pot-Limit Bet Sizing": lambda c: (
        c.dominant_is_aggressive and c.multiple_raise_sizes
    ),
}


# --- display metadata (category + plain-English trigger) -------------------
# Powers the admin Skills page's PLO section. Categories mirror the catalog's
# A/B/C grouping (D = postflop is out of scope until PLO postflop solves exist).
CATEGORY_A = "A -- Carry-over preflop decisions"
CATEGORY_B = "B -- PLO hand-reading (the edge)"
CATEGORY_C = "C -- Math / dynamics"

#: Ordered (A->B->C) tuple of the category labels, for grouped display.
SKILL_CATEGORIES: tuple[str, ...] = (CATEGORY_A, CATEGORY_B, CATEGORY_C)


@dataclass(frozen=True)
class PloSkillMeta:
    """Display metadata for one PLO skill."""

    category: str
    description: str
    fires: bool = True  # False = wired but dormant on this single-raise-size pack


SKILL_META: dict[str, PloSkillMeta] = {
    # --- A: carry-over preflop decisions ---
    "Preflop Hand Selection": PloSkillMeta(
        CATEGORY_A, "Opening or open-folding first in: which 4-card hands are worth playing."
    ),
    "3-Betting": PloSkillMeta(
        CATEGORY_A, "Re-raising an open, for value or as a light / fold-equity 3-bet."
    ),
    "Facing a 3-Bet": PloSkillMeta(
        CATEGORY_A, "Responding to a 3-bet of your open (call / 4-bet / fold), not a squeeze."
    ),
    "4-Betting": PloSkillMeta(
        CATEGORY_A, "Re-raising a 3-bet, for value or light."
    ),
    "Facing a 4-Bet": PloSkillMeta(
        CATEGORY_A, "Responding to a 4-bet of your 3-bet (call / jam / fold)."
    ),
    "Squeezing": PloSkillMeta(
        CATEGORY_A, "Raising over an open plus one or more callers."
    ),
    "Facing a Squeeze": PloSkillMeta(
        CATEGORY_A, "Responding to a squeeze after you opened or called."
    ),
    "Blind Defense": PloSkillMeta(
        CATEGORY_A, "Defending the SB or BB against a raise (excludes blind-vs-blind)."
    ),
    "Blind vs. Blind Play": PloSkillMeta(
        CATEGORY_A, "Small-blind-versus-big-blind confrontations."
    ),
    "Pot Odds": PloSkillMeta(
        CATEGORY_A, "Call / fold decisions priced on pot odds, including calling an all-in."
    ),
    "In Position Play": PloSkillMeta(
        CATEGORY_A, "Acting last postflop: late seats facing a raise, or the SB in a blind battle."
    ),
    "Out of Position Play": PloSkillMeta(
        CATEGORY_A, "Acting first postflop: the blinds defending against a raise."
    ),
    "Multiway Pot Strategy": PloSkillMeta(
        CATEGORY_A, "Three or more players still in at hero's decision."
    ),
    # --- B: PLO hand-reading (the edge) ---
    "Suitedness": PloSkillMeta(
        CATEGORY_B, "Value that leans on flush potential (double-suited), not on a big pair."
    ),
    "Rundowns & Connectivity": PloSkillMeta(
        CATEGORY_B, "Connected / wrap hands that make straights and big draws."
    ),
    "Dangler Awareness": PloSkillMeta(
        CATEGORY_B, "A card disconnected from the other three -- effectively a wasted card."
    ),
    "Nut-Flush Awareness": PloSkillMeta(
        CATEGORY_B, "Holding nut-flush potential (a suited ace)."
    ),
    "Big-Pair Construction": PloSkillMeta(
        CATEGORY_B, "AAxx / KKxx hands and how well their side cards support them."
    ),
    "Nuttedness & Non-Nut Traps": PloSkillMeta(
        CATEGORY_B,
        "Hands that tend to make SECOND-best flushes or straights (a bare ace, all-low holdings).",
    ),
    "Nut Blockers & Card Removal": PloSkillMeta(
        CATEGORY_B,
        "Using your cards to remove villain's strong combos -- an ace blocks "
        "AA, a suited ace blocks the nut flush -- to justify a 3-bet / 4-bet / "
        "squeeze.",
    ),
    # --- C: math / dynamics ---
    "Implied Odds": PloSkillMeta(
        CATEGORY_C,
        "Calling a well-shaped speculative hand for the big pots you win when "
        "it connects -- the positive counterpart to reverse implied odds.",
    ),
    "Reverse Implied Odds": PloSkillMeta(
        CATEGORY_C, "Speculative calls with a non-nut hand that tends to make second-best hands."
    ),
    "Pot-Limit Bet Sizing": PloSkillMeta(
        CATEGORY_C,
        "Choosing a raise size. Only a real decision on a multi-raise-size tree, so "
        "dormant on this single-pot-sized-raise pack.",
        fires=False,
    ),
}


def compute_plo_skills(facts: PloFacts) -> list[str]:
    """The firing PLO skill names for a spot, in catalog order.

    Strict: a typical question yields 2-5 skills. Order matches
    :data:`SKILL_CATALOG` so the CSV ``skills`` column stays diff-friendly.
    """
    ctx = from_plo_facts(facts)
    return [name for name, rule in SKILL_CATALOG.items() if rule(ctx)]


__all__ = [
    "SKILL_CATALOG",
    "SKILL_CATEGORIES",
    "SKILL_META",
    "PloSkillContext",
    "PloSkillMeta",
    "SkillRule",
    "compute_plo_skills",
    "from_plo_facts",
]
