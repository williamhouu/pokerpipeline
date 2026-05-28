"""Preflop difficulty rating algorithm (May 2026 redesign).

Replaces the freq-only formula with a **4-axis weighted-sum + bump
table** so the score can span the full range regardless of strategy
form (pure-strategy `Always X` spots can now reach high difficulty
when the concept / hand context warrants it; mixed-strategy spots
can land in the easy tier when everything else is straightforward).

## Algorithm

::

    easy_freq    = (dominant_freq - 0.55) / 0.45,   clipped [0, 1]
    easy_ev      = ev_gap_bb / 3.0,                 clipped [0, 1]
    easy_concept = ARCHETYPE_BASE_EASE[archetype]
                 + sum(CONCEPT_TAG_MODIFIERS[tag] for tag firing),
                 clipped [0.05, 1.0]
    easy_hand    = HAND_CLASS_EASE.get(matched_hand_tag, default 0.55)

    easy = w_freq * easy_freq
         + w_ev * easy_ev
         + w_concept * easy_concept
         + w_hand * easy_hand

    # Optional additive bump rules apply AFTER the weighted sum.
    easy += sum(rule.easy_delta for rule in BUMP_RULES if rule.predicate(...))

    difficulty = round(clip(3000 - easy * 2500, 400, 3200))

Each axis is normalised to [0, 1] where 1 = "easy on this dimension"
and 0 = "hard". The weights sum to 1.0 by construction; when
``ev_gap_bb`` is unavailable (raise-involved spots in v1) the
EV weight is redistributed proportionally across the other three so
the blend stays a valid weighted average.

## Bounds (soft)

Most spots land in [500, 3000] -- the brief's MVP Elo range. Hard
floor / ceiling at [400, 3200] absorbs the rare outlier without
information loss (a 'really easy' spot can score 450 instead of being
capped at 500, etc.).

## Bump rules

Signed additive deltas applied after the weighted sum, captured in
:data:`BUMP_RULES`. The table is currently EMPTY -- bumps get added
here as we observe specific spot patterns the linear weighted sum
mis-scores. Each bump should be small (~+/-0.05 in easy-units, ~125
difficulty points) so the axis structure remains the primary signal.

Example future bump (NOT currently active)::

    BumpRule(
        name="advanced_squeeze_with_marginal_hand",
        description="Squeeze archetype + marginal hand class is meaningfully "
                    "harder than the axes alone suggest",
        easy_delta=-0.05,
        predicate=lambda facts, ev: (
            facts.archetype in ("squeeze_for_value", "squeeze_as_bluff")
            and facts.spot.hero_hand_class in {"22", "33", ...}
        ),
    )

## Tuning

All weights, base ease tables, modifiers, and bump rules live as
Python constants in this module. Edit the constants and re-run any
batch to retune. A future iteration may expose them via a YAML
override file or admin-panel sliders; for now they're version-
controlled here.

## Output

:func:`compute_difficulty` returns a :class:`DifficultyResult` carrying
the score AND the per-axis breakdown. Layer 8 surfaces the breakdown
in CSV diagnostic columns (``easy_freq``, ``easy_ev``,
``easy_concept``, ``easy_hand``) so reviewers can see WHY a spot got
a particular rating.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pipeline.preflop.fact_extractor import PreflopFacts

# === axis weights ============================================================
# Sum to 1.0. When ev_gap_bb is unavailable (raise-involved spots),
# W_EV is redistributed proportionally across the other three so the
# blend stays a valid weighted average.
W_FREQ: float = 0.40
W_EV: float = 0.30
W_CONCEPT: float = 0.20
W_HAND: float = 0.10
# Sanity: weights sum to 1.0 (allowing IEEE-754 rounding noise).
assert abs((W_FREQ + W_EV + W_CONCEPT + W_HAND) - 1.0) < 1e-9


# === axis 1: freq ============================================================
# Linear interpolation from 55% (hardest, easy_freq=0) to 100%
# (easiest, easy_freq=1). Mirrors the legacy freq-only formula's
# spread; the 55% floor matches the worthiness window.
_FREQ_FLOOR: float = 0.55
_FREQ_SPAN: float = 0.45  # 1.00 - 0.55


# === axis 2: EV gap ==========================================================
# Linear up to 3 bb -- gaps beyond 3bb give no additional easiness
# (the spot is already "trivial" past that point). 3bb is roughly the
# practical preflop ceiling.
_EV_GAP_FULL_CREDIT_BB: float = 3.0


# === axis 3: concept =========================================================
# Base ease per strategic archetype. Open spots are conceptually easiest
# (no villain context to track); 5-bet pots are hardest (rare + complex).
# Default 0.50 for "unclassified" / unknown -- neutral fallback.
ARCHETYPE_BASE_EASE: dict[str, float] = {
    "open_for_value":         1.00,
    "fold_outranged":         1.00,
    "fold_dominated":         0.70,
    "fold_pot_odds":          0.60,
    "call_for_value":         0.60,
    "call_for_implied_odds":  0.55,
    "3bet_for_value":         0.50,
    "3bet_as_bluff":          0.45,
    "squeeze_for_value":      0.30,
    "squeeze_as_bluff":       0.30,
    "4bet_for_value":         0.25,
    "4bet_as_bluff":          0.25,
    "5bet_for_value":         0.10,
    "5bet_as_bluff":          0.10,
    "all_in_for_value":       0.15,
    "all_in_as_bluff":        0.15,
    "unclassified":           0.50,
}
_CONCEPT_BASE_DEFAULT: float = 0.50

# Additive modifiers applied on top of the archetype base when each
# tag fires. Sum into easy_concept, then clip to [_CONCEPT_EASE_MIN,
# _CONCEPT_EASE_MAX]. Captures spot-shape effects orthogonal to the
# archetype.
CONCEPT_TAG_MODIFIERS: dict[str, float] = {
    "multiway_pot":   -0.15,  # 3+ players = more variables, harder
    "short_stack":    -0.15,  # tournament short stacks add ICM, harder
    "deep_stack":      0.05,  # standard 100bb is the simpler case
}
_CONCEPT_EASE_MIN: float = 0.05
_CONCEPT_EASE_MAX: float = 1.00


# === axis 4: hand class ======================================================
# U-shaped: extreme hands (premium OR genuine trash) are easy to play.
# Marginal hands (small pairs, suited connectors, suited aces) are
# hard because the right action requires real strategic reasoning.
# The mapping reads from the preflop concept_tag library's hand-class
# tags -- whichever tag fires sets the ease. Order matters when
# multiple tags could fire: earlier entries win the lookup.
HAND_CLASS_EASE: dict[str, float] = {
    "premium_pair":         1.00,  # AA / KK / QQ -- almost always pure
    "premium_unpaired":     0.85,  # AK / AQ -- mostly pure 3-bet or call
    "unconnected_offsuit":  0.90,  # 73o etc. -- obvious folds from most spots
    "suited_broadway":      0.65,  # KQs etc. -- generally easy decisions
    "medium_pair":          0.50,  # value vs. capped continuing ranges
    "small_pair":           0.45,  # set-mine vs. continue decisions
    "suited_ace":           0.40,  # blocker + semi-bluff candidate math
    "suited_connector":     0.40,  # implied odds + position math
}
_HAND_DEFAULT_EASE: float = 0.55


# === bump rules ==============================================================
@dataclass(frozen=True)
class BumpRule:
    """A signed additive delta applied after the weighted axis sum.

    Each bump captures a known spot pattern the linear axis blend
    mis-scores -- typically synergies between axes (e.g. mixed strategy
    AND advanced concept).

    Attributes:
        name: Short identifier surfaced in the CSV's ``difficulty_bumps``
            column when this rule fires. Keep snake_case so it reads
            cleanly in comma-separated lists.
        description: One-line human reason for the bump. Shown in the
            admin panel's difficulty explainer.
        easy_delta: Signed; positive = make easier, negative = make
            harder. Keep small (|delta| <= 0.08) so the axes remain
            the dominant signal.
        predicate: Function ``(facts, ev_gap_bb) -> bool`` returning
            True iff the bump should fire for this spot.
    """

    name: str
    description: str
    easy_delta: float
    predicate: Callable[[PreflopFacts, float | None], bool]


# Currently empty. Add rules as observed batches show specific spots
# the 4-axis blend mis-scores. Example template (NOT active)::
#
#   BUMP_RULES = (
#       BumpRule(
#           name="five_bet_with_marginal",
#           description="5-bet pot AND small/medium pair: extra hard",
#           easy_delta=-0.05,
#           predicate=lambda facts, ev: (
#               facts.archetype.startswith("5bet")
#               and facts.spot.hero_hand_class in {"22","33","44","55","66","77","88","99","TT"}
#           ),
#       ),
#   )
BUMP_RULES: tuple[BumpRule, ...] = ()


# === bounds ==================================================================
# Soft bounds: most spots land in [500, 3000] but rare outliers can
# spill slightly past either edge to preserve information at the
# extremes. Hard floor/ceiling clamp at [400, 3200].
_LINEAR_CEILING: int = 3000  # difficulty when easy=0
_LINEAR_FLOOR: int = 500     # difficulty when easy=1
_LINEAR_SPAN: int = _LINEAR_CEILING - _LINEAR_FLOOR  # 2500
_HARD_FLOOR: int = 400       # clamp; allows ~100 of "easier than easy" room
_HARD_CEILING: int = 3200    # clamp; allows ~200 of "harder than hard" room


# === result dataclass ========================================================
@dataclass(frozen=True)
class DifficultyResult:
    """Full breakdown of a preflop spot's difficulty rating.

    Carries each axis's contribution alongside the final score so the
    CSV diagnostic columns can surface them and reviewers can inspect
    WHY a spot got a particular rating.

    Attributes:
        score: Final integer difficulty in roughly [400, 3200].
            Round-half-up; most values in [500, 3000].
        easy_freq: Contribution from the frequency axis, in [0, 1].
        easy_ev: Contribution from EV gap, in [0, 1]. When ``ev_available``
            is False this is 0.0 (the value is not used in the blend
            -- the EV weight is redistributed across the other three).
        easy_concept: Contribution from the archetype + concept-tag
            modifier axis, in [_CONCEPT_EASE_MIN, _CONCEPT_EASE_MAX].
        easy_hand: Contribution from the hand class axis, in [0, 1].
        easy_blend: The weighted sum (and any applied bumps) that the
            final score was computed from. Useful for debugging.
        bumps_applied: Names of any BUMP_RULES that fired for this
            spot. Empty for now (BUMP_RULES is empty).
        ev_available: True iff ``ev_gap_bb`` was provided.
    """

    score: int
    easy_freq: float
    easy_ev: float
    easy_concept: float
    easy_hand: float
    easy_blend: float
    bumps_applied: tuple[str, ...] = field(default_factory=tuple)
    ev_available: bool = True


# === main entry point ========================================================
def compute_difficulty(
    facts: PreflopFacts,
    *,
    ev_gap_bb: float | None = None,
) -> DifficultyResult:
    """Compute the per-spot difficulty rating with full breakdown.

    Args:
        facts: The PreflopFacts -- archetype, hand class, action freqs.
        ev_gap_bb: EV gap to the second-best action in bb. Pass None
            when the EV engine couldn't compute it (raise-involved
            spots). The weight is redistributed across the other axes
            in that case rather than treating EV as neutral 0.5.

    Returns:
        DifficultyResult: score in roughly [400, 3200] (most in
        [500, 3000]) plus per-axis breakdowns + any fired bump names.
    """
    # --- axis 1: freq ---------------------------------------------------------
    freq = facts.spot.dominant_frequency
    easy_freq = _clip01((freq - _FREQ_FLOOR) / _FREQ_SPAN)

    # --- axis 2: EV gap -------------------------------------------------------
    ev_available = ev_gap_bb is not None
    if ev_available:
        # mypy: ev_gap_bb is not None inside this branch
        easy_ev = _clip01(ev_gap_bb / _EV_GAP_FULL_CREDIT_BB)  # type: ignore[operator]
    else:
        easy_ev = 0.0  # not used in blend; weight redistributed instead

    # --- axis 3 + 4: concept + hand (both need the firing concept tags) ------
    # Lazy import to avoid a circular dependency between this module
    # and pipeline.preflop.concept_tags (which imports PreflopFacts).
    from pipeline.preflop.concept_tags import compute_concept_tags  # noqa: PLC0415

    firing_tags = set(compute_concept_tags(facts))

    # axis 3: concept
    archetype = facts.archetype or "unclassified"
    base_concept = ARCHETYPE_BASE_EASE.get(archetype, _CONCEPT_BASE_DEFAULT)
    tag_delta = sum(
        CONCEPT_TAG_MODIFIERS.get(tag, 0.0) for tag in firing_tags
    )
    easy_concept = _clip(
        base_concept + tag_delta,
        _CONCEPT_EASE_MIN,
        _CONCEPT_EASE_MAX,
    )

    # axis 4: hand class. Pick the first matching tag in HAND_CLASS_EASE's
    # insertion order (so premium_pair wins over a hypothetical conflicting
    # tag). Default 0.55 when none fire.
    easy_hand = _HAND_DEFAULT_EASE
    for tag, ease in HAND_CLASS_EASE.items():
        if tag in firing_tags:
            easy_hand = ease
            break

    # --- weighted sum (with EV weight redistribution when needed) -------------
    if ev_available:
        easy_blend = (
            W_FREQ * easy_freq
            + W_EV * easy_ev
            + W_CONCEPT * easy_concept
            + W_HAND * easy_hand
        )
    else:
        # Redistribute W_EV proportionally across the other three so the
        # weights still sum to 1.0. easy_ev (= 0.0) drops out of the sum.
        remaining = W_FREQ + W_CONCEPT + W_HAND
        scale = 1.0 / remaining
        easy_blend = (
            (W_FREQ * scale) * easy_freq
            + (W_CONCEPT * scale) * easy_concept
            + (W_HAND * scale) * easy_hand
        )

    # --- bump rules -----------------------------------------------------------
    bumps_applied: list[str] = []
    for rule in BUMP_RULES:
        if rule.predicate(facts, ev_gap_bb):
            easy_blend += rule.easy_delta
            bumps_applied.append(rule.name)

    # --- map to integer difficulty with soft bounds --------------------------
    raw_difficulty = _LINEAR_CEILING - easy_blend * _LINEAR_SPAN
    score = round(_clip(raw_difficulty, _HARD_FLOOR, _HARD_CEILING))

    return DifficultyResult(
        score=score,
        easy_freq=easy_freq,
        easy_ev=easy_ev,
        easy_concept=easy_concept,
        easy_hand=easy_hand,
        easy_blend=easy_blend,
        bumps_applied=tuple(bumps_applied),
        ev_available=ev_available,
    )


# === helpers =================================================================
def _clip01(x: float) -> float:
    """Clip to [0, 1]."""
    return max(0.0, min(1.0, x))


def _clip(x: float, lo: float, hi: float) -> float:
    """Clip to [lo, hi]."""
    return max(lo, min(hi, x))


__all__ = [
    "ARCHETYPE_BASE_EASE",
    "BUMP_RULES",
    "BumpRule",
    "CONCEPT_TAG_MODIFIERS",
    "DifficultyResult",
    "HAND_CLASS_EASE",
    "W_CONCEPT",
    "W_EV",
    "W_FREQ",
    "W_HAND",
    "compute_difficulty",
]
