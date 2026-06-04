"""PLO preflop difficulty rating (4-axis weighted sum).

Ports the NLHE algorithm (:mod:`pipeline.preflop.difficulty`) to PLO. Same
shape -- four [0, 1] "ease" axes (1 = easy, 0 = hard) blended by fixed weights,
then mapped to a 400-3200 score -- with PLO-specific tables and two structural
differences:

1. **The EV axis is reliably available.** Monker stores per-action EV, so every
   spot carries a real ``ev_gap_bb`` (raise spots included). The NLHE code had
   to redistribute the EV weight on raise spots; here that path is a rare
   fallback (only a degenerate single-action node lacks a gap).
2. **The hand axis keys off the hand's strength bucket** (premium / strong /
   medium / marginal / weak / trash) rather than matching hand-class tags --
   the PLO hand model already produces that categorical key.

::

    easy_freq    = (dominant_freq - 0.55) / 0.45,          clip [0, 1]
    easy_ev      = ev_gap_bb / 3.0,                         clip [0, 1]
    easy_concept = ARCHETYPE_BASE_EASE[archetype]
                 + sum(CONCEPT_TAG_MODIFIERS[tag] firing),  clip [0.05, 1.0]
    easy_hand    = HAND_STRENGTH_EASE[hand_class.strength]

    easy = 0.40*easy_freq + 0.30*easy_ev + 0.20*easy_concept + 0.10*easy_hand
    difficulty = round(clip(3000 - easy*2500, 400, 3200))

All weights / tables are tunable starting values; refine against graded output.
:func:`compute_plo_difficulty` returns the score plus the per-axis breakdown for
the CSV diagnostic columns.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.spot_tags import compute_plo_concept_tags

# === axis weights (sum to 1.0) =============================================
W_FREQ: float = 0.40
W_EV: float = 0.30
W_CONCEPT: float = 0.20
W_HAND: float = 0.10
assert abs((W_FREQ + W_EV + W_CONCEPT + W_HAND) - 1.0) < 1e-9

# === axis 1: freq ==========================================================
_FREQ_FLOOR: float = 0.55  # worthiness floor -> easy_freq 0
_FREQ_SPAN: float = 0.45  # 1.00 - 0.55 -> easy_freq 1

# === axis 2: EV gap ========================================================
_EV_GAP_FULL_CREDIT_BB: float = 3.0  # gaps beyond 3 bb add no more easiness

# === axis 3: concept =======================================================
# Base ease per archetype: opens/clear folds are trivial; 5-bet / jam pots are
# hard. Mirrors the NLHE table (PLO decision complexity by bet level is similar).
ARCHETYPE_BASE_EASE: dict[str, float] = {
    "open_for_value": 1.00,
    "open_fold": 1.00,
    "fold_dominated": 0.95,
    "fold_pot_odds": 0.60,
    "call_for_value": 0.60,
    "call_for_implied_odds": 0.55,
    "call_allin": 0.55,
    "3bet_for_value": 0.50,
    "3bet_as_bluff": 0.45,
    "squeeze_for_value": 0.30,
    "squeeze_as_bluff": 0.30,
    "4bet_for_value": 0.25,
    "4bet_as_bluff": 0.25,
    "5bet_for_value": 0.10,
    "5bet_as_bluff": 0.10,
    "all_in_for_value": 0.15,
    "all_in_as_bluff": 0.15,
    "unclassified": 0.50,
}
_CONCEPT_BASE_DEFAULT: float = 0.50

# Additive tag modifiers (PLO multiway is common and genuinely harder -- more
# equity and nut considerations to track).
CONCEPT_TAG_MODIFIERS: dict[str, float] = {
    "multiway_pot": -0.15,
    "short_stack": -0.15,
    "deep_stack": 0.05,
}
_CONCEPT_EASE_MIN: float = 0.05
_CONCEPT_EASE_MAX: float = 1.00

# === axis 4: hand strength =================================================
# U-shaped: clear hands (premium value OR trash) are easy; marginal/medium hands
# (close, playable, non-nut) are the hard ones.
HAND_STRENGTH_EASE: dict[str, float] = {
    "premium": 1.00,
    "strong": 0.78,
    "medium": 0.48,
    "marginal": 0.40,
    "weak": 0.58,
    "trash": 0.88,
}
_HAND_DEFAULT_EASE: float = 0.55

# === bounds ================================================================
_LINEAR_CEILING: int = 3000  # difficulty when easy = 0
_LINEAR_SPAN: int = 2500  # 3000 - 500
_HARD_FLOOR: int = 400
_HARD_CEILING: int = 3200


@dataclass(frozen=True)
class PloDifficultyResult:
    """A PLO spot's difficulty score plus its per-axis ease breakdown."""

    score: int
    easy_freq: float
    easy_ev: float
    easy_concept: float
    easy_hand: float
    easy_blend: float
    ev_available: bool = True


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_plo_difficulty(facts: PloFacts) -> PloDifficultyResult:
    """Compute the PLO difficulty rating with its per-axis breakdown.

    Reads ``facts.ev_gap_bb`` (free from the solver). When it is ``None`` -- only
    a degenerate single-action node -- the EV weight is redistributed across the
    other three axes so the blend stays a valid weighted average.
    """
    # axis 1: freq
    easy_freq = _clip01((facts.spot.dominant_frequency - _FREQ_FLOOR) / _FREQ_SPAN)

    # axis 2: EV gap
    ev_gap_bb = facts.ev_gap_bb
    ev_available = ev_gap_bb is not None
    easy_ev = _clip01(ev_gap_bb / _EV_GAP_FULL_CREDIT_BB) if ev_gap_bb is not None else 0.0

    # axes 3 + 4 need the firing concept tags
    firing = set(compute_plo_concept_tags(facts))

    # axis 3: concept
    base = ARCHETYPE_BASE_EASE.get(facts.archetype or "unclassified", _CONCEPT_BASE_DEFAULT)
    is_fold = facts.spot.dominant_action.startswith("Fold")
    tag_delta = 0.0
    for tag in firing:
        mod = CONCEPT_TAG_MODIFIERS.get(tag, 0.0)
        if is_fold and mod < 0:
            continue  # folding trash is trivial whether multiway / short or not
        tag_delta += mod
    easy_concept = _clip(base + tag_delta, _CONCEPT_EASE_MIN, _CONCEPT_EASE_MAX)

    # axis 4: hand strength
    easy_hand = HAND_STRENGTH_EASE.get(facts.hand_class.strength, _HAND_DEFAULT_EASE)

    # weighted sum (redistribute EV weight when the gap is unavailable)
    if ev_available:
        easy_blend = (
            W_FREQ * easy_freq
            + W_EV * easy_ev
            + W_CONCEPT * easy_concept
            + W_HAND * easy_hand
        )
    else:
        scale = 1.0 / (W_FREQ + W_CONCEPT + W_HAND)
        easy_blend = (
            (W_FREQ * scale) * easy_freq
            + (W_CONCEPT * scale) * easy_concept
            + (W_HAND * scale) * easy_hand
        )

    score = round(_clip(_LINEAR_CEILING - easy_blend * _LINEAR_SPAN, _HARD_FLOOR, _HARD_CEILING))
    return PloDifficultyResult(
        score=score,
        easy_freq=easy_freq,
        easy_ev=easy_ev,
        easy_concept=easy_concept,
        easy_hand=easy_hand,
        easy_blend=easy_blend,
        ev_available=ev_available,
    )


__all__ = [
    "ARCHETYPE_BASE_EASE",
    "CONCEPT_TAG_MODIFIERS",
    "HAND_STRENGTH_EASE",
    "PloDifficultyResult",
    "W_CONCEPT",
    "W_EV",
    "W_FREQ",
    "W_HAND",
    "compute_plo_difficulty",
]
