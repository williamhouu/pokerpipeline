"""PLO facts-relative concept tags + the unified concept-tag aggregator.

The hand-structure tags live in :mod:`pipeline.plo.concept_tags` (pure
functions of a :class:`~pipeline.plo.hand_model.PloHandClass`, pack-independent
and exhaustively audited). This module adds the tags that need the
:class:`~pipeline.plo.fact_extractor.PloFacts` layer -- position, decision
context, solver strategy shape, equity, range dynamics, stack, blockers -- and
:func:`compute_plo_concept_tags`, the single entry point that combines both.

It mirrors the NLHE tagger (:mod:`pipeline.preflop.concept_tags`): every tag is
a pure ``def tag(facts: PloFacts) -> bool``, the registry maps a function to its
own name, and the aggregator returns the firing names. Per the "LLM never
thinks about poker" rule, nothing here is an LLM call or a judgement that needs
strategy -- the LLM later reads the firing tags as context, it doesn't produce
them.

Thresholds (equity bands, range-advantage cutoffs) are tunable starting values;
PLO preflop equities run compressed (4-card hands run close), so the equity
bands are tighter than NLHE's. Refine against graded output.
"""

from __future__ import annotations

from collections.abc import Callable

from pipeline.plo.concept_tags import compute_plo_hand_tags
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.node_enumerator import plo_active_player_count
from pipeline.plo.pack import PloActionType

FactsTagFn = Callable[[PloFacts], bool]

_AGGRESSIVE = {
    PloActionType.RAISE,
    PloActionType.MIN_RAISE,
    PloActionType.ALL_IN,
}
_AGGRESSIVE_LABELS = ("Raise", "Min-raise", "All-in")

_MIN_MULTIWAY = 3


# --- helpers ---------------------------------------------------------------
def _raise_count(facts: PloFacts) -> int:
    """Raises (incl. min-raises and jams) in the prior action history."""
    return sum(1 for a in facts.spot.node.history_before if a.action in _AGGRESSIVE)


def _calls_after_last_raise(facts: PloFacts) -> int:
    """Calls between the most recent raise and hero's decision."""
    count = 0
    seen_raise = False
    for a in facts.spot.node.history_before:
        if a.action in _AGGRESSIVE:
            seen_raise = True
            count = 0  # reset on each new raise
        elif seen_raise and a.action is PloActionType.CALL:
            count += 1
    return count


def _active_count(facts: PloFacts) -> int:
    """Players still in the pot at hero's decision (last-action non-folders + hero)."""
    return plo_active_player_count(facts.spot.node)


# --- Position context (5) --------------------------------------------------
# Buckets come from pipeline.plo.position.position_bucket (table-size aware:
# the 6-max pack's LJ is its UTG-equivalent, the 9-max LJ is a middle seat).
def _bucket(facts: PloFacts) -> str:
    from pipeline.plo.position import position_bucket  # noqa: PLC0415

    return position_bucket(
        facts.spot.node.actor, table_size=facts.spot.node.table_size
    )


def early_position(facts: PloFacts) -> bool:
    """Hero is in an early (UTG-family) seat."""
    return _bucket(facts) == "early"


def middle_position(facts: PloFacts) -> bool:
    """Hero is in a middle seat (6-max: the Hijack; 9-max: Lojack/Hijack)."""
    return _bucket(facts) == "middle"


def late_position(facts: PloFacts) -> bool:
    """Hero is in the Cutoff or on the Button."""
    return _bucket(facts) == "late"


def small_blind(facts: PloFacts) -> bool:
    """Hero is in the Small Blind."""
    return facts.spot.node.actor == "SB"


def big_blind(facts: PloFacts) -> bool:
    """Hero is in the Big Blind."""
    return facts.spot.node.actor == "BB"


# --- Decision context (7) --------------------------------------------------
def open_decision(facts: PloFacts) -> bool:
    """No prior raise -- hero is deciding whether to open."""
    return _raise_count(facts) == 0


def facing_single_raise(facts: PloFacts) -> bool:
    """Exactly one prior raise and no caller between it and hero."""
    return _raise_count(facts) == 1 and _calls_after_last_raise(facts) == 0


def facing_3bet(facts: PloFacts) -> bool:
    """Exactly two prior raises -- hero opened/3-bet and got re-raised."""
    return _raise_count(facts) == 2  # noqa: PLR2004


def facing_4bet_plus(facts: PloFacts) -> bool:
    """Three or more prior raises -- a 4-bet+ spot."""
    return _raise_count(facts) >= 3  # noqa: PLR2004


def squeeze_opportunity(facts: PloFacts) -> bool:
    """One prior raise and at least one caller -- raising here is a squeeze."""
    return _raise_count(facts) == 1 and _calls_after_last_raise(facts) >= 1


def bvb_spot(facts: PloFacts) -> bool:
    """Only the blinds are left and hero is one of them (blind vs blind)."""
    # Deliberately ANY non-fold action (not last-action like the active-player
    # count): a non-blind who entered the pot and later folded still shaped
    # the ranges and left dead money, so it's not a pure blind battle.
    non_blind_acted = any(
        a.seat not in ("SB", "BB") and a.action is not PloActionType.FOLD
        for a in facts.spot.node.history_before
    )
    return not non_blind_acted and facts.spot.node.actor in ("SB", "BB")


def multiway_pot(facts: PloFacts) -> bool:
    """Three or more players still in the pot at hero's decision."""
    return _active_count(facts) >= _MIN_MULTIWAY


# --- Strategy shape (5) ----------------------------------------------------
_MIXED_FLOOR = 0.55
_PURE_FLOOR = 0.95


def mixed_strategy(facts: PloFacts) -> bool:
    """Dominant action's conditional frequency is 55-95% -- a real mix."""
    return _MIXED_FLOOR <= facts.spot.dominant_frequency < _PURE_FLOOR


def near_pure_strategy(facts: PloFacts) -> bool:
    """Dominant action's conditional frequency is >= 95% -- effectively pure."""
    return facts.spot.dominant_frequency >= _PURE_FLOOR


def dominant_is_aggressive(facts: PloFacts) -> bool:
    """Hero's dominant action raises the bet (raise / min-raise / jam)."""
    return facts.spot.dominant_action.startswith(_AGGRESSIVE_LABELS)


def dominant_is_passive(facts: PloFacts) -> bool:
    """Hero's dominant action is a Call."""
    return facts.spot.dominant_action == "Call"


def dominant_is_fold(facts: PloFacts) -> bool:
    """Hero's dominant action is Fold."""
    return facts.spot.dominant_action == "Fold"


# --- Equity context (4) ----------------------------------------------------
# PLO 4-card equities are compressed vs NLHE, so the bands are tighter.
_EQ_DOMINANT = 0.62
_EQ_FAVORITE = 0.54
_EQ_COINFLIP_LO = 0.46
_EQ_DOMINATED = 0.40


def equity_dominant(facts: PloFacts) -> bool:
    """Hero's hand has > 62% equity vs villain's range (big PLO edge)."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and eq > _EQ_DOMINANT


def equity_favorite(facts: PloFacts) -> bool:
    """Hero's hand has 54-62% equity vs villain's range."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and _EQ_FAVORITE <= eq <= _EQ_DOMINANT


def coinflip(facts: PloFacts) -> bool:
    """Hero's equity vs villain is 46-54% -- close to a flip."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and _EQ_COINFLIP_LO <= eq < _EQ_FAVORITE


def dominated(facts: PloFacts) -> bool:
    """Hero's equity vs villain is < 40% -- a clear underdog."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and eq < _EQ_DOMINATED


# --- Range dynamics (3) ----------------------------------------------------
_RANGE_HERO = 0.53
_RANGE_VILLAIN = 0.47


def hero_range_advantage(facts: PloFacts) -> bool:
    """Hero's range equity vs villain's range >= 53%."""
    re_ = facts.hero_range_equity_vs_villain
    return re_ is not None and re_ >= _RANGE_HERO


def villain_range_advantage(facts: PloFacts) -> bool:
    """Villain's range has the edge: hero range equity <= 47%."""
    re_ = facts.hero_range_equity_vs_villain
    return re_ is not None and re_ <= _RANGE_VILLAIN


def roughly_equal_ranges(facts: PloFacts) -> bool:
    """Range equity is within 47-53% -- no meaningful edge either way."""
    re_ = facts.hero_range_equity_vs_villain
    return re_ is not None and _RANGE_VILLAIN < re_ < _RANGE_HERO


# --- Blockers (2) ----------------------------------------------------------
# Preflop, the blockers that matter are nut blockers: holding an ace removes
# villain's AA value, and a SUITED ace is the nut-flush blocker. Read off the
# already-classified hand; only meaningful when facing a villain.
def blocks_villain_value(facts: PloFacts) -> bool:
    """Hero holds an ace, removing AA combos from villain's value range."""
    return facts.villain_stats is not None and facts.hand_class.has_ace


def blocks_villain_nut_flush(facts: PloFacts) -> bool:
    """Hero holds a suited ace -- the nut-flush blocker -- vs a villain."""
    return facts.villain_stats is not None and facts.hand_class.suited_ace


# --- Stack depth (3) -------------------------------------------------------
# Real since July 2026: the enumerator stamps the pack's effective stack on
# every node (PloDecisionNode.stack_bb), so these read it directly. They were
# hardcoded to "always standard" from the single-100bb-pack era, which shipped
# a wrong standard_stack tag (and silenced the short_stack difficulty
# modifier) on every 10-25bb MTT / short-stack question.
def short_stack(facts: PloFacts) -> bool:
    """Effective stack < 40bb."""
    return facts.spot.node.stack_bb < 40  # noqa: PLR2004


def standard_stack(facts: PloFacts) -> bool:
    """Effective stack 40-150bb."""
    return 40 <= facts.spot.node.stack_bb <= 150  # noqa: PLR2004


def deep_stack(facts: PloFacts) -> bool:
    """Effective stack > 150bb."""
    return facts.spot.node.stack_bb > 150  # noqa: PLR2004


# --- registry + aggregator -------------------------------------------------
# Context tags lead (position, decision), then the hand-structure tags, then
# the solver-derived dynamics (strategy, equity, range, blockers, stack).
_CONTEXT_TAGS: tuple[FactsTagFn, ...] = (
    early_position,
    middle_position,
    late_position,
    small_blind,
    big_blind,
    open_decision,
    facing_single_raise,
    facing_3bet,
    facing_4bet_plus,
    squeeze_opportunity,
    bvb_spot,
    multiway_pot,
)
_DYNAMICS_TAGS: tuple[FactsTagFn, ...] = (
    mixed_strategy,
    near_pure_strategy,
    dominant_is_aggressive,
    dominant_is_passive,
    dominant_is_fold,
    equity_dominant,
    equity_favorite,
    coinflip,
    dominated,
    hero_range_advantage,
    villain_range_advantage,
    roughly_equal_ranges,
    blocks_villain_value,
    blocks_villain_nut_flush,
    short_stack,
    standard_stack,
    deep_stack,
)


def compute_plo_concept_tags(facts: PloFacts) -> list[str]:
    """All firing concept-tag names for a spot, in CSV-readable order.

    Combines the position/decision context tags, the hand-structure tags from
    :func:`pipeline.plo.concept_tags.compute_plo_hand_tags`, and the solver
    dynamics tags. Deterministic; each tag's function name is its label. This
    is the entry point Layer 6 (SOLVER DATA block), Layer 8 (the ``concept_tags``
    column), and the skill mapper consume.
    """
    return [
        *(fn.__name__ for fn in _CONTEXT_TAGS if fn(facts)),
        *compute_plo_hand_tags(facts.hand_class),
        *(fn.__name__ for fn in _DYNAMICS_TAGS if fn(facts)),
    ]
