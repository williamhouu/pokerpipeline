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
    # Checking the option in a limped/unraised pot (the BB facing no bet) is a
    # near-trivial decision -- you check almost everything, raise a few hands.
    "bb_check":               0.95,
    # Completing the half bet first-in from the SB -- easy, not trivial
    # (same base as the PLO tagger's entry).
    "sb_complete":            0.85,
    # Folding a clearly-dominated hand is about as trivial as poker gets,
    # so this sits near the top (was 0.70, which left obvious junk-folds
    # mis-rated as medium). fold_pot_odds stays lower: folding despite
    # decent equity is a closer, more teachable decision.
    "fold_dominated":         0.95,
    # Folding a hand not strong enough to (re)raise when no call is offered
    # (e.g. SB junk-ace vs a 3-bet, fold-or-4bet) is an easy, near-trivial
    # fold -- close to fold_dominated.
    "fold_no_continue":       0.90,
    "fold_pot_odds":          0.60,
    "call_for_value":         0.60,
    "call_for_implied_odds":  0.55,
    "call_allin":             0.55,
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

# Clear suited trash (e.g. 73s, 84s): suited, both cards low, and gapped --
# an auto-fold from essentially everywhere, so easy to evaluate. Rated a
# touch below offsuit junk (0.90) since suited hands have marginally more
# playability. CONSERVATIVE by design: anything with a broadway/ace, a 9,
# or a small gap (connectors / one-gappers) is EXCLUDED, so genuinely more
# playable hands like K2s, T6s, 86s, 53s keep the neutral default.
_TRASH_SUITED_EASE: float = 0.82
_RANK_VALUE: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
# Highest card must be <= this (8) -> no broadway, no ace, no 9.
_TRASH_SUITED_MAX_RANK: int = 8
# Cards must be at least this far apart (2 = excludes connectors + one-gappers).
_TRASH_SUITED_MIN_GAP: int = 2


def _is_clear_trash_suited(hand_class: str) -> bool:
    """True for clearly-trash SUITED hands (73s, 84s, 62s ...).

    Conservative: suited only, highest card <= 8 (so no broadway/ace/9),
    and a gap of >= 2 (so suited connectors and one-gappers -- which have
    real playability -- are excluded). K2s, T6s, 86s, 53s all return False.
    """
    if len(hand_class) != 3 or hand_class[2] != "s":
        return False
    hi_rank = _RANK_VALUE.get(hand_class[0])
    lo_rank = _RANK_VALUE.get(hand_class[1])
    if hi_rank is None or lo_rank is None:
        return False
    hi, lo = max(hi_rank, lo_rank), min(hi_rank, lo_rank)
    if hi > _TRASH_SUITED_MAX_RANK:
        return False
    return (hi - lo - 1) >= _TRASH_SUITED_MIN_GAP


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


# === trap-aware difficulty (opt-in) ==========================================
# A "trap" spot is one where the solver's dominant action CONTRADICTS the
# simple pot-odds (equity-vs-price) baseline a naive player reasons from:
# folding a hand whose equity clears the price, or calling/raising a hand whose
# equity is clearly below it. These are HARD FOR A HUMAN even when the solver
# is pure (one clear-cut action, a big EV gap) -- so the frequency + EV axes,
# which both score a pure spot "easy", measure exactly the wrong thing here.
# When trap-aware difficulty is ON, such a spot's "easy" score is CAPPED so it
# lands at least at TRAP_DIFFICULTY_FLOOR (solidly in the Hard band), regardless
# of how pure / +EV the line is -- this is what lets a 100%-frequency
# ("Always X") spot be rated Hard. Opt-in (a batch flag), so the DEFAULT rating
# behaviour is unchanged. Only defined for spots facing a price to call; opening
# spots (no price) are never traps.
TRAP_DIFFICULTY_FLOOR: int = 2400
# easy_blend cap that maps to TRAP_DIFFICULTY_FLOOR via the linear score map, so
# the floored score still derives cleanly from easy_blend (no separate clamp).
_TRAP_EASY_CAP: float = (_LINEAR_CEILING - TRAP_DIFFICULTY_FLOOR) / _LINEAR_SPAN
# Noise guard: equity is Monte-Carlo sampled (~1-2% at 400 runouts), so only
# call a spot a trap when equity is CLEARLY on the wrong side of the price by
# this margin. A spot whose equity sits within the margin of the price is a
# genuine close-call at the price, not a counterintuitive trap.
_TRAP_EQUITY_MARGIN: float = 0.04


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
        trap_bump_applied: True iff trap-aware difficulty was ON AND this
            spot is a counterintuitive "trap" (so its score was floored to
            at least :data:`TRAP_DIFFICULTY_FLOOR`). Always False when the
            trap bump is off.
    """

    score: int
    easy_freq: float
    easy_ev: float
    easy_concept: float
    easy_hand: float
    easy_blend: float
    bumps_applied: tuple[str, ...] = field(default_factory=tuple)
    ev_available: bool = True
    trap_bump_applied: bool = False


# === main entry point ========================================================
def compute_difficulty(
    facts: PreflopFacts,
    *,
    ev_gap_bb: float | None = None,
    apply_trap_bump: bool = False,
) -> DifficultyResult:
    """Compute the per-spot difficulty rating with full breakdown.

    Args:
        facts: The PreflopFacts -- archetype, hand class, action freqs.
        ev_gap_bb: EV gap to the second-best action in bb. Pass None
            when the EV engine couldn't compute it (raise-involved
            spots). The weight is redistributed across the other axes
            in that case rather than treating EV as neutral 0.5.
        apply_trap_bump: Opt-in trap-aware difficulty. When True, a
            counterintuitive "trap" spot (the solver's dominant action
            contradicts the equity-vs-price baseline) is floored to at
            least :data:`TRAP_DIFFICULTY_FLOOR`, so a pure / clear-cut
            but deceptive spot rates Hard. Default False = unchanged
            behaviour. See :func:`_is_counterintuitive_spot`.

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
    # When hero is FOLDING, spot-complexity penalties (multiway / short
    # stack) don't make the decision harder -- folding trash is trivial
    # whether heads-up or multiway. So drop the negative modifiers for
    # fold spots; positive modifiers (e.g. deep_stack) still apply.
    is_fold = facts.spot.dominant_action.startswith("Fold")
    tag_delta = 0.0
    for tag in firing_tags:
        mod = CONCEPT_TAG_MODIFIERS.get(tag, 0.0)
        if is_fold and mod < 0:
            continue  # don't penalize a fold for multiway / short stack
        tag_delta += mod
    easy_concept = _clip(
        base_concept + tag_delta,
        _CONCEPT_EASE_MIN,
        _CONCEPT_EASE_MAX,
    )

    # axis 4: hand class. Pick the first matching tag in HAND_CLASS_EASE's
    # insertion order (so premium_pair wins over a hypothetical conflicting
    # tag). When no tag fires, recognize clear suited trash (73s etc.) as
    # an easy hand; otherwise fall back to the neutral default.
    easy_hand = _HAND_DEFAULT_EASE
    for tag, ease in HAND_CLASS_EASE.items():
        if tag in firing_tags:
            easy_hand = ease
            break
    else:
        if _is_clear_trash_suited(facts.spot.hero_hand_class):
            easy_hand = _TRASH_SUITED_EASE

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

    # --- trap-aware floor (opt-in) -------------------------------------------
    # A counterintuitive spot is hard for a human even when the solver is pure,
    # so cap its "easy" score (-> floor its difficulty). The freq/EV axes rated
    # it easy precisely because the line is clear-cut -- the wrong signal for a
    # trap. Spots already harder than the floor keep their (lower) easy_blend.
    trap_bump_applied = False
    if apply_trap_bump and _is_counterintuitive_spot(facts):
        trap_bump_applied = True
        easy_blend = min(easy_blend, _TRAP_EASY_CAP)

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
        trap_bump_applied=trap_bump_applied,
    )


# === helpers =================================================================
def _clip01(x: float) -> float:
    """Clip to [0, 1]."""
    return max(0.0, min(1.0, x))


def _clip(x: float, lo: float, hi: float) -> float:
    """Clip to [lo, hi]."""
    return max(lo, min(hi, x))


def _is_counterintuitive_spot(facts: PreflopFacts) -> bool:
    """True when the solver's dominant action contradicts the pot-odds
    (equity-vs-price) baseline a naive player reasons from -- a "trap".

    Defined ONLY for spots facing a price to call (``break_even_equity`` is
    set); opening spots (no price) return False. Uses the field equity in a
    multiway all-in (the equity hero must actually beat) and the heads-up
    equity otherwise. Fires only when equity is on the wrong side of the price
    by a clear margin (``_TRAP_EQUITY_MARGIN``), so a genuine close-call at the
    price -- or a sampling wobble -- is NOT mistaken for a trap:

      * equity clears the price but the solver FOLDS (looks like a call, isn't
        -- domination / reverse implied odds / poor realisation), or
      * equity is below the price but the solver CONTINUES (calls or raises --
        an implied-odds call or a low-equity 3-bet bluff that looks like a fold).

    Deliberately the SAME signal as
    :func:`pipeline.preflop.validators.soft_validate_fold_as_equity_favorite`,
    so any trap-FOLD this rates Hard is ALSO soft-flagged for human review -- a
    guard against amplifying a broken solve (e.g. the Monker 9-max QQ-fold
    artifact) into a "hard" question.

    HEADS-UP ONLY. The equity-vs-price baseline is only well defined against ONE
    opponent at ONE price. In a multiway pot ``hero_equity_vs_field`` (beat the
    WHOLE field) compared to a single call price is mis-specified -- and for a
    jam it is a category error (a raiser is not paying that price). Empirically
    those multiway "traps" are dominated by degenerate deep-all-in nodes
    (calling / jamming at ~13-26% field equity), not teachable spots, so we skip
    them. (A measured 88% of raw trap hits were multiway and ~all degenerate;
    every clean trap was heads-up.)
    """
    if facts.hero_equity_vs_field is not None:
        return False  # multiway -> field-equity-vs-price is mis-specified; skip
    price = facts.break_even_equity
    if price is None:
        return False
    eq = facts.hero_equity_vs_villain
    if eq is None:
        return False
    solver_continues = not facts.spot.dominant_action.startswith("Fold")
    if solver_continues and eq <= price - _TRAP_EQUITY_MARGIN:
        return True  # equity clearly below the price, yet the solver continues
    # A FOLD is only a "trap" if equity clears the price by the base margin PLUS
    # a rake cushion: rake (and, at depth, poor realisation) mean you win less
    # than the naive pot-odds price implies, so a fold whose raw equity barely
    # beats that price is usually just correct, not a surprising trap. Without
    # this, the detector over-flagged normal deep folds on raked packs (the
    # 8-max 200bb pack at ~10% rake). rake_pct defaults 0 -> unchanged behaviour.
    rake_cushion = getattr(facts, "rake_pct", 0.0)
    if (not solver_continues) and eq >= price + _TRAP_EQUITY_MARGIN + rake_cushion:
        return True  # equity clearly clears the (rake-adjusted) price, yet folds
    return False


__all__ = [
    "ARCHETYPE_BASE_EASE",
    "BUMP_RULES",
    "BumpRule",
    "CONCEPT_TAG_MODIFIERS",
    "DifficultyResult",
    "HAND_CLASS_EASE",
    "TRAP_DIFFICULTY_FLOOR",
    "W_CONCEPT",
    "W_EV",
    "W_FREQ",
    "W_HAND",
    "compute_difficulty",
]
