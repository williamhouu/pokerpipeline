"""Difficulty rating for a postflop spot (500-3000 Elo-style scale).

Follows the brief's MVP shape -- difficulty falls out of how *clear* the
solver's decision is -- generalised to two axes the postflop pipeline can
compute today:

    easy_freq : how dominant the top action is. 55% (a near-coin-flip, the
                hard edge of the worthiness window) -> 0.0; 100% (pure) -> 1.0.
    easy_ev   : how punishing the best wrong answer is. A 0bb EV gap -> 0.0
                (hard, the wrong answer barely costs anything); a >=3bb gap
                -> 1.0 (easy, an obvious mistake to avoid).

    easy   = w_freq * easy_freq + w_ev * easy_ev
    score  = round(clip(3000 - easy * 2500, 400, 3200))

So a *clear* spot (pure-ish top action, big EV gap) scores LOW (easy ~500)
and a *close* spot (barely-dominant action, tiny gap) scores HIGH (~3000).
When the EV gap is unknown the EV weight is redistributed onto frequency, so
the score stays well-defined. Concept- and hand-class axes (as in the preflop
algorithm) are a documented future extension.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.postflop.facts import PostflopFacts

# Axis weights (sum to 1). Frequency carries most of the signal; the EV gap
# refines it. Mirrors the spirit of the preflop weighting (freq-heavy).
W_FREQ = 0.6
W_EV = 0.4

# The worthiness window's hard edge -- 55% dominant is the hardest spot we
# keep, so it anchors easy_freq = 0.
_FREQ_FLOOR = 0.55
# EV gap (bb) that counts as "maximally easy" on the EV axis.
_EV_FULL_CREDIT_BB = 3.0

_SCORE_MIN = 400
_SCORE_MAX = 3200
_SCORE_SPAN = 2500  # 3000 - easy*2500, then clipped


@dataclass(frozen=True)
class PostflopDifficulty:
    """The difficulty score plus its per-axis diagnostics (for the CSV)."""

    score: int
    easy_freq: float
    easy_ev: float | None  # None when the EV gap was unavailable


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def easy_freq_axis(dominant_frequency: float) -> float:
    """Map a dominant frequency in [0.55, 1.0] to ease in [0, 1]."""
    return _clip01((dominant_frequency - _FREQ_FLOOR) / (1.0 - _FREQ_FLOOR))


def easy_ev_axis(ev_gap_bb: float) -> float:
    """Map an EV gap in [0, 3]bb to ease in [0, 1]."""
    return _clip01(ev_gap_bb / _EV_FULL_CREDIT_BB)


def compute_difficulty(facts: PostflopFacts) -> PostflopDifficulty:
    """Difficulty for a fully-extracted spot.

    Uses ``dominant_frequency`` and ``ev_gap_bb`` from ``facts``. If the EV
    gap is unavailable, the EV weight is folded onto the frequency axis so the
    score still reflects how dominant the action is.
    """
    e_freq = easy_freq_axis(facts.dominant_frequency)
    if facts.ev_gap_bb is not None:
        e_ev: float | None = easy_ev_axis(facts.ev_gap_bb)
        easy = W_FREQ * e_freq + W_EV * e_ev
    else:
        e_ev = None
        easy = e_freq  # all weight on frequency when EV is unknown

    score = round(3000 - easy * _SCORE_SPAN)
    score = max(_SCORE_MIN, min(_SCORE_MAX, score))
    return PostflopDifficulty(score=score, easy_freq=e_freq, easy_ev=e_ev)


__all__ = [
    "PostflopDifficulty",
    "W_EV",
    "W_FREQ",
    "compute_difficulty",
    "easy_ev_axis",
    "easy_freq_axis",
]
