"""PLO preflop difficulty rating (3-axis weighted sum; EV axis dropped).

Ports the NLHE algorithm (:mod:`pipeline.preflop.difficulty`) to PLO. Same
shape -- [0, 1] "ease" axes (1 = easy, 0 = hard) blended by fixed weights, then
mapped to a 400-3200 score -- with PLO-specific tables and two differences:

1. **No EV axis.** NLHE blends a fourth ``easy_ev`` axis, but PLO drops it
   (June 2026). The reason is structural, not a magnitude quirk: a *worthy*
   spot is mixed-frequency (55-90%) by definition, and a spot mixes precisely
   because its top actions are near-equal in EV. So within the worthiness
   window the solver EV gap is ~0 almost everywhere (mean ~0.06 bb) AND
   redundant with the frequency axis -- it carried no independent signal and
   just shoved every score upward ~350-500 points, which made the "Easy" tier
   unreachable for PLO (no worthy spot rated below ~1420). ``ev_gap_bb`` is
   still computed and kept for the ``easy_ev`` CSV diagnostic and the separate
   ``min_ev_gap_bb`` worthiness gate; it simply no longer feeds the rating.
   It could be reintroduced if the spot pool ever spans more EV-separated
   decisions, or if rescaled to PLO's compressed magnitude (full credit near
   ~0.5 bb, not 3 bb) -- see the diagnostic data we still emit.
2. **The hand axis keys off the hand's strength bucket** (premium / strong /
   medium / marginal / weak / trash) rather than matching hand-class tags --
   the PLO hand model already produces that categorical key.

::

    easy_freq    = (dominant_freq - 0.55) / 0.45,          clip [0, 1]
    easy_concept = ARCHETYPE_BASE_EASE[archetype]
                 + sum(CONCEPT_TAG_MODIFIERS[tag] firing),  clip [0.05, 1.0]
    easy_hand    = HAND_STRENGTH_EASE[hand_class.strength]

    easy = 0.57*easy_freq + 0.29*easy_concept + 0.14*easy_hand
    difficulty = round(clip(3000 - easy*2500, 400, 3200))

All weights / tables are tunable starting values; refine against graded output.
:func:`compute_plo_difficulty` returns the score plus the per-axis breakdown for
the CSV diagnostic columns (``easy_ev`` included, for analysis even though it no
longer affects the score).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.spot_tags import compute_plo_concept_tags

# === axis weights (sum to 1.0; 3 axes -- EV dropped, see module docstring) ==
# These are the old freq / concept / hand split (4 : 2 : 1) renormalised after
# removing the 0.30 EV weight, so their relative balance is unchanged.
W_FREQ: float = 0.57
W_CONCEPT: float = 0.29
W_HAND: float = 0.14
assert abs((W_FREQ + W_CONCEPT + W_HAND) - 1.0) < 1e-9

# === axis 1: freq ==========================================================
_FREQ_FLOOR: float = 0.55  # worthiness floor -> easy_freq 0
_FREQ_SPAN: float = 0.45  # 1.00 - 0.55 -> easy_freq 1

# === EV gap (diagnostic only -- NOT a difficulty axis; see module docstring) =
_EV_GAP_FULL_CREDIT_BB: float = 3.0  # gaps beyond 3 bb add no more easiness

# === axis 3: concept =======================================================
# Base ease per archetype: opens/clear folds are trivial; 5-bet / jam pots are
# hard. Mirrors the NLHE table (PLO decision complexity by bet level is similar).
ARCHETYPE_BASE_EASE: dict[str, float] = {
    "open_for_value": 1.00,
    "open_fold": 1.00,
    "bb_check": 0.95,  # checking the option in a limped pot -- near-trivial
    "sb_complete": 0.85,  # completing the half bet first-in -- easy, not trivial
    "open_limp": 0.75,  # limping at ante depths -- cheap, but a real choice
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
    # True when the opt-in 🪤 trap floor re-rated this spot (see plo_trap_margin).
    trap_bump_applied: bool = False


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# === 🪤 trap-aware floor (opt-in, July 2026 -- the NLHE port, RE-CALIBRATED) =
# PLO equities COMPRESS toward 50%, so a routinely-correct fold holds equity
# well above the naive break-even price -- measured on the 6-max 100bb pack
# (150 worthy heads-up closes-action spots, July 2026): ordinary folds sit at
# eq - price median +0.087, p90 +0.144, max +0.247. The NLHE fold rule
# (margin 0.04 + rake) would therefore flag essentially EVERY normal PLO
# fold. The PLO detector adds a COMPRESSION CUSHION on the fold side and a
# hand-shape gate (the hand model's own premium/strong bucket -- a
# pretty-looking hand that folds is the trap being taught), so only the far
# tail past the ordinary-fold distribution fires. The continue side fires
# only on an ALL-IN price (call closes all betting -> no implied odds can
# justify a below-price call), which ordinary continues never approach
# (measured min -0.011).
_TRAP_EQUITY_MARGIN: float = 0.04  # same noise margin as NLHE / postflop
_PLO_FOLD_COMPRESSION_CUSHION: float = 0.10  # past p90 of ordinary folds


def plo_trap_margin(
    facts: PloFacts,
    *,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
    rake_pct: float = 0.0,
) -> float | None:
    """The naive ``|equity - price|`` margin when this spot is a trap, else None.

    A trap = the solver's dominant action contradicts the equity-vs-price
    baseline in a way a PLO player would find genuinely counterintuitive:

    * FOLD-trap: a premium/strong-shaped hand folds although its equity
      clears the price by margin + rake + the compression cushion (see the
      calibration note above). Both signals must agree -- the hand model
      says "pretty", the pot odds say "way ahead of the price" -- so PLO's
      equity compression can't flag ordinary folds.
    * CONTINUE-trap: the solver calls/raises although equity sits clearly
      BELOW the price, and the price is an ALL-IN price (villain's action is
      a jam, so no implied odds exist to justify it).

    HEADS-UP closes-action spots only (mirrors NLHE: the baseline is only
    well-defined against one live opponent when hero's call/fold ends the
    betting). Returns None -- never a trap -- when equity wasn't computed
    (``compute_equity=False``), when no live price exists, or when any gate
    fails. The returned margin is RAKE-BLIND (what the player naively
    feels), matching :func:`pipeline.trap_grading.graded_trap_floor`.
    """
    from pipeline.plo.action_history import call_price, price_is_live  # noqa: PLC0415
    from pipeline.plo.fact_extractor import identify_villain  # noqa: PLC0415
    from pipeline.plo.node_enumerator import (  # noqa: PLC0415
        PloActionType,
        plo_active_player_count,
        plo_pending_after_hero,
    )

    eq = facts.hero_equity_vs_villain
    if eq is None:
        return None
    node = facts.spot.node
    if plo_active_player_count(node) != 2:  # noqa: PLR2004 -- heads-up only
        return None
    _pending, closes_action = plo_pending_after_hero(node)
    if not closes_action:
        return None
    if not price_is_live(node.history_before, node.actor):
        return None
    pot, to_call = call_price(
        node.history_before, node.actor, stack_bb=stack_bb, ante_bb=ante_bb
    )
    if to_call <= 0:
        return None
    price = to_call / (pot + to_call)
    folds = facts.spot.dominant_action.startswith("Fold")
    if folds:
        if facts.hand_class.strength not in ("premium", "strong"):
            return None
        threshold = (
            price + _TRAP_EQUITY_MARGIN + rake_pct + _PLO_FOLD_COMPRESSION_CUSHION
        )
        return abs(eq - price) if eq >= threshold else None
    villain = identify_villain(node)
    if villain is None or villain[1].action is not PloActionType.ALL_IN:
        return None  # implied odds may justify a below-price call
    return abs(eq - price) if eq <= price - _TRAP_EQUITY_MARGIN else None


def compute_plo_difficulty(
    facts: PloFacts,
    *,
    apply_trap_bump: bool = False,
    stack_bb: float = 100.0,
    ante_bb: float = 0.0,
    rake_pct: float = 0.0,
) -> PloDifficultyResult:
    """Compute the PLO difficulty rating with its per-axis breakdown.

    Three axes (freq / concept / hand); the EV axis is intentionally NOT part of
    the blend (see the module docstring for why). ``easy_ev`` and ``ev_available``
    are still computed and returned, but only for the CSV diagnostic and possible
    future re-introduction -- they do not affect ``score``.

    ``apply_trap_bump`` (opt-in) floors a counterintuitive spot (see
    :func:`plo_trap_margin`) to its GRADED trap floor
    (:func:`pipeline.trap_grading.graded_trap_floor`), so a pure-but-deceptive
    spot rates Medium-to-Hard. Requires the batch to have computed equity
    (``compute_equity=True``, the default) -- without equity no spot can fire.
    ``stack_bb`` / ``ante_bb`` / ``rake_pct`` come from the pack spec and feed
    the price walk + the fold-side rake cushion. Default off = byte-identical
    to the pre-trap behaviour.
    """
    # axis 1: freq
    easy_freq = _clip01((facts.spot.dominant_frequency - _FREQ_FLOOR) / _FREQ_SPAN)

    # EV gap -- DIAGNOSTIC ONLY, not blended into the score (module docstring).
    ev_gap_bb = facts.ev_gap_bb
    ev_available = ev_gap_bb is not None
    easy_ev = _clip01(ev_gap_bb / _EV_GAP_FULL_CREDIT_BB) if ev_gap_bb is not None else 0.0

    # axes 2 + 3 need the firing concept tags
    firing = set(compute_plo_concept_tags(facts))

    # axis 2: concept
    base = ARCHETYPE_BASE_EASE.get(facts.archetype or "unclassified", _CONCEPT_BASE_DEFAULT)
    is_fold = facts.spot.dominant_action.startswith("Fold")
    tag_delta = 0.0
    for tag in firing:
        mod = CONCEPT_TAG_MODIFIERS.get(tag, 0.0)
        if is_fold and mod < 0:
            continue  # folding trash is trivial whether multiway / short or not
        tag_delta += mod
    easy_concept = _clip(base + tag_delta, _CONCEPT_EASE_MIN, _CONCEPT_EASE_MAX)

    # axis 3: hand strength
    easy_hand = HAND_STRENGTH_EASE.get(facts.hand_class.strength, _HAND_DEFAULT_EASE)

    # weighted sum over the three live axes (weights already sum to 1.0)
    easy_blend = W_FREQ * easy_freq + W_CONCEPT * easy_concept + W_HAND * easy_hand

    score = round(_clip(_LINEAR_CEILING - easy_blend * _LINEAR_SPAN, _HARD_FLOOR, _HARD_CEILING))

    trap_applied = False
    if apply_trap_bump:
        margin = plo_trap_margin(
            facts, stack_bb=stack_bb, ante_bb=ante_bb, rake_pct=rake_pct
        )
        if margin is not None:
            from pipeline.trap_grading import graded_trap_floor  # noqa: PLC0415

            trap_applied = True
            # Floor, never lower: a trap can't rate below its natural score.
            score = max(score, graded_trap_floor(margin))

    return PloDifficultyResult(
        score=score,
        easy_freq=easy_freq,
        easy_ev=easy_ev,
        easy_concept=easy_concept,
        easy_hand=easy_hand,
        easy_blend=easy_blend,
        ev_available=ev_available,
        trap_bump_applied=trap_applied,
    )


__all__ = [
    "ARCHETYPE_BASE_EASE",
    "CONCEPT_TAG_MODIFIERS",
    "HAND_STRENGTH_EASE",
    "PloDifficultyResult",
    "W_CONCEPT",
    "W_FREQ",
    "W_HAND",
    "compute_plo_difficulty",
    "plo_trap_margin",
]
