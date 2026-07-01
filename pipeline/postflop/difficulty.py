"""Difficulty rating for a postflop spot (500-3000 Elo-style scale).

Multi-axis weighted sum, mirroring the preflop / PLO algorithms. Difficulty
falls out of how *clear* the solver's decision is, across three axes:

    easy_freq    : how dominant the top action is. 55% (a near-coin-flip, the
                   hard edge of the worthiness window) -> 0.0; 100% (pure) -> 1.0.
    easy_concept : how hard the strategic FRAME is (a top-set value bet is easy,
                   a thin river bluff-catch is hard). Base ease per archetype +
                   concept-tag modifiers.
    easy_hand    : how hard the HAND is to play (premium made hands and clear air
                   are easy; medium / marginal / vulnerable hands are hard).

    easy  = w_freq * easy_freq + w_concept * easy_concept + w_hand * easy_hand
    score = round(clip(3000 - easy * 2500, 400, 3200))

So a *clear* spot scores LOW (easy ~500) and a *close/tricky* spot scores HIGH
(~3000). The EV gap is deliberately NOT in the score: a worthy postflop spot is
a frequency MIX by construction (the EV-gap worthiness gate is off by default),
so the EV gap is ~0 across the whole worthy pool and adds no signal beyond the
frequency axis -- exactly why PLO dropped its EV axis. ``easy_ev`` is still
computed and surfaced as a CSV diagnostic, it just doesn't feed the score.

**Trap-aware floor (opt-in, mirrors preflop).** The freq axis rates a PURE spot
"easy", but a genuinely counterintuitive pure spot (the solver folds a hand
whose equity clears the price, or continues one whose equity is clearly below
it) is HARD for a human. When ``apply_trap_bump`` is on, such a spot's score is
floored to :data:`TRAP_DIFFICULTY_FLOOR` (solidly Hard). Heads-up facing-a-bet
spots only; off by default so the default rating is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.postflop.facts import PostflopFacts

# === axis weights (sum to 1) =================================================
# Frequency carries most of the signal; concept + hand refine it. Mirrors PLO's
# 3-axis split (it dropped EV for the same reason postflop does).
W_FREQ = 0.55
W_CONCEPT = 0.30
W_HAND = 0.15
assert abs((W_FREQ + W_CONCEPT + W_HAND) - 1.0) < 1e-9

# The worthiness window's hard edge -- 55% dominant is the hardest spot we keep,
# so it anchors easy_freq = 0.
_FREQ_FLOOR = 0.55
# EV gap (bb) that counts as "maximally easy" -- diagnostic only (not in score).
_EV_FULL_CREDIT_BB = 3.0

# === axis 2: concept (archetype base + tag modifiers) ========================
# Base ease per postflop archetype. Value bets / clear give-ups / folds-when-
# beaten are conceptually easiest; bluff-catches, traps, and bluff-raises are
# hardest (they hinge on the opponent's range, not your own hand).
ARCHETYPE_BASE_EASE: dict[str, float] = {
    "value_bet":          0.78,  # bet a strong hand for value -- clear
    "defensive_check":    0.72,  # give up a weak hand -- clear
    "fold_to_pressure":   0.68,  # fold when beaten -- fairly clear
    "value_raise":        0.62,
    "call_with_showdown": 0.55,
    "call_drawing":       0.55,  # pot-odds / implied-odds driven
    "protection_bet":     0.55,  # bet a vulnerable hand to deny equity
    "semibluff":          0.50,  # draw + fold equity
    "pot_control_check":  0.48,  # check a medium hand -- non-obvious to a beginner
    "bluff":              0.40,  # needs a fold-equity read
    "trap_check":         0.35,  # check a monster to trap -- counterintuitive
    "bluff_raise":        0.30,  # advanced
    "bluff_catch":        0.30,  # hinges on villain's bluff frequency vs the price
    "unclassified":       0.50,
}
_CONCEPT_BASE_DEFAULT = 0.50
# Additive modifiers on top of the archetype base. Kept small + conservative.
CONCEPT_TAG_MODIFIERS: dict[str, float] = {
    "multiway_pot":       -0.15,  # 3+ players = more variables, harder
    "wet_board":          -0.05,  # dynamic boards raise the cost of an error
    "range_disadvantage": -0.05,  # navigating from behind as a range is harder
}
_CONCEPT_EASE_MIN = 0.05
_CONCEPT_EASE_MAX = 1.00

# === axis 3: hand class (U-shaped on strength bucket) ========================
# Premium made hands and clear air are easy to evaluate; the medium / marginal /
# vulnerable middle is hard. A draw in an otherwise-weak hand is a real decision
# (semibluff / call), so it's nudged off the "clear air" easiness.
_STRENGTH_BUCKET_EASE: dict[str, float] = {
    "premium":    0.90,
    "strong":     0.75,
    "air":        0.85,  # nothing -- easy to evaluate as a give-up / bluff
    "medium":     0.45,
    "vulnerable": 0.45,
    "marginal":   0.40,
}
_HAND_DEFAULT_EASE = 0.55
# A draw in a weak/marginal hand is a medium decision, not a clear give-up.
_DRAW_HAND_EASE = 0.55

# === bounds (soft) ===========================================================
_LINEAR_CEILING = 3000  # difficulty when easy=0
_LINEAR_FLOOR = 500     # difficulty when easy=1
_LINEAR_SPAN = _LINEAR_CEILING - _LINEAR_FLOOR  # 2500
# The brief's documented scale is 500-3000 (500 = easiest). Clamp the hard floor
# to 500 so no spot prints below "easiest" -- matters for the trivial entry legs
# (see preflop_entry.difficulty, which uses the same 500 floor).
_HARD_FLOOR = 500
_HARD_CEILING = 3200

# === trap-aware difficulty (opt-in, mirrors preflop) =========================
TRAP_DIFFICULTY_FLOOR = 2400
_TRAP_EASY_CAP = (_LINEAR_CEILING - TRAP_DIFFICULTY_FLOOR) / _LINEAR_SPAN
# Equity is Monte-Carlo sampled (~1-2%), so only call a spot a trap when equity
# is CLEARLY on the wrong side of the price by this margin.
_TRAP_EQUITY_MARGIN = 0.04


@dataclass(frozen=True)
class PostflopDifficulty:
    """The difficulty score plus its per-axis diagnostics (for the CSV)."""

    score: int
    easy_freq: float
    easy_ev: float | None  # diagnostic only -- NOT in the score
    easy_concept: float
    easy_hand: float
    easy_blend: float
    trap_bump_applied: bool = False


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def easy_freq_axis(dominant_frequency: float) -> float:
    """Map a dominant frequency in [0.55, 1.0] to ease in [0, 1]."""
    return _clip01((dominant_frequency - _FREQ_FLOOR) / (1.0 - _FREQ_FLOOR))


def easy_ev_axis(ev_gap_bb: float) -> float:
    """Map an EV gap in [0, 3]bb to ease in [0, 1] (diagnostic only)."""
    return _clip01(ev_gap_bb / _EV_FULL_CREDIT_BB)


def easy_concept_axis(facts: PostflopFacts) -> float:
    """Archetype base ease + concept-tag modifiers, clipped to [0.05, 1.0]."""
    base = ARCHETYPE_BASE_EASE.get(facts.archetype or "unclassified", _CONCEPT_BASE_DEFAULT)
    tags = set(facts.concept_tags)
    # Folding gets no spot-complexity penalty -- folding is trivial whether the
    # board is wet or the pot is multiway (mirrors the preflop fold carve-out).
    is_fold = facts.dominant_verb == "fold"
    delta = 0.0
    for tag, mod in CONCEPT_TAG_MODIFIERS.items():
        if tag in tags and not (is_fold and mod < 0):
            delta += mod
    return _clip(base + delta, _CONCEPT_EASE_MIN, _CONCEPT_EASE_MAX)


def easy_hand_axis(facts: PostflopFacts) -> float:
    """Hand-class ease: U-shaped on the strength bucket; a real draw in a weak
    hand is a medium decision rather than a clear give-up (backdoors/gutshots
    don't count -- see PostflopFacts.has_strong_draw)."""
    if facts.has_strong_draw and facts.strength_bucket in ("air", "marginal", "vulnerable"):
        return _DRAW_HAND_EASE
    return _STRENGTH_BUCKET_EASE.get(facts.strength_bucket, _HAND_DEFAULT_EASE)


def _is_counterintuitive_spot(facts: PostflopFacts) -> bool:
    """True when the solver's action contradicts the equity-vs-price baseline.

    Heads-up facing-a-bet spots only (a price must exist, and the baseline is
    only well-defined against one opponent). The SAME signal the preflop trap
    floor uses: folds despite equity >= price, or continues despite equity <
    price, by a clear margin so sampling wobble isn't mistaken for a trap.
    """
    if facts.n_players != 2:  # noqa: PLR2004
        return False
    price = facts.break_even_equity
    if price is None:  # not facing a bet -> no price -> never a trap
        return False
    eq = facts.hero_equity_vs_villain
    continues = facts.dominant_verb != "fold"
    if continues and eq <= price - _TRAP_EQUITY_MARGIN:
        return True  # equity clearly below the price, yet the solver continues
    # A FOLD is a trap only if equity clears the price by the base margin PLUS a
    # rake cushion: rake (and poor realisation) mean you win less than the naive
    # price implies, so a fold whose raw equity barely beats it is usually just
    # correct, not a trap. Stops over-flagging normal folds on raked solves.
    rake_cushion = getattr(facts, "rake_pct", 0.0)
    if (not continues) and eq >= price + _TRAP_EQUITY_MARGIN + rake_cushion:
        return True  # equity clears the (rake-adjusted) price, yet the solver folds
    return False


def compute_difficulty(
    facts: PostflopFacts, *, apply_trap_bump: bool = False
) -> PostflopDifficulty:
    """Difficulty for a fully-extracted spot (3-axis: freq + concept + hand).

    ``apply_trap_bump`` (opt-in) floors a counterintuitive heads-up facing-a-bet
    spot to :data:`TRAP_DIFFICULTY_FLOOR` so a pure-but-deceptive spot rates Hard.
    Default off = unchanged behaviour.
    """
    e_freq = easy_freq_axis(facts.dominant_frequency)
    e_concept = easy_concept_axis(facts)
    e_hand = easy_hand_axis(facts)
    e_ev = easy_ev_axis(facts.ev_gap_bb) if facts.ev_gap_bb is not None else None

    easy = W_FREQ * e_freq + W_CONCEPT * e_concept + W_HAND * e_hand

    trap_applied = False
    if apply_trap_bump and _is_counterintuitive_spot(facts):
        trap_applied = True
        easy = min(easy, _TRAP_EASY_CAP)

    score = round(_clip(_LINEAR_CEILING - easy * _LINEAR_SPAN, _HARD_FLOOR, _HARD_CEILING))
    return PostflopDifficulty(
        score=score,
        easy_freq=e_freq,
        easy_ev=e_ev,
        easy_concept=e_concept,
        easy_hand=e_hand,
        easy_blend=easy,
        trap_bump_applied=trap_applied,
    )


__all__ = [
    "ARCHETYPE_BASE_EASE",
    "CONCEPT_TAG_MODIFIERS",
    "PostflopDifficulty",
    "TRAP_DIFFICULTY_FLOOR",
    "W_CONCEPT",
    "W_FREQ",
    "W_HAND",
    "compute_difficulty",
    "easy_concept_axis",
    "easy_ev_axis",
    "easy_freq_axis",
    "easy_hand_axis",
]
