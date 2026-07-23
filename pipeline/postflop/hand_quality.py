"""🎬 Action-heavy hand policy: keep full-hand batches interesting.

July 2026 (user ask, from live review of v8 strict-clean batches): 37% of
all full hands generated to date had one bet or fewer in their ENTIRE
postflop line — check-check, check-check, then an easy river fold. And the
difficulty mix inverted: the hands that reach the river passively usually
entered via a MARGINAL preflop defend (54s-type), whose close mix maxes the
two dominant preflop difficulty axes (~2000-2600), while every postflop leg
on a check line is near-pure (~800-1000) — so "Hard" full hands were hard
preflop and trivial after, and the peak-anchored ``hand_difficulty`` let the
preflop spike qualify the whole hand.

This module is the PURE policy layer (no Streamlit, no solve access beyond
the hand objects, no LLM — it runs before any API call):

* :func:`is_trivial_fold_ender` — the featured final question must be a real
  decision: a fold-ender above 90% is "fold 5-high to a stab", excluded.
  Genuinely mixed folds (the bluff-catch zone the worthiness window keeps)
  survive.
* :func:`is_passive_line` — a hand ending on the turn/river with ZERO bets or
  raises on any EARLIER street is a checkdown story. Not banned outright —
  the occasional "it checked through, stab or give up?" is educational — but
  capped by :func:`apply_action_heavy_policy` (default ~15% of the batch).
  Flop- and preflop-ending hands are exempt (there are no earlier postflop
  streets for action to have happened on).
* :func:`educational_density` — the ordering score: action content + a
  genuinely-mixed ender + facing-a-bet enders + raised lines float to the
  top; passive lines sink. The selectors downstream (length quotas, the
  diversify mix, greedy balance) all preserve input order within their
  buckets, so this ordering survives every mode.
* :func:`apply_action_heavy_policy` — the composed gate: density-order the
  pool, drop trivial fold-enders, cap passive lines. Returns honest counters
  (no silent truncation).

The postflop-SPINE difficulty half of the July-2026 bundle (bands/balance
keying off the hardest POSTFLOP leg, so a preflop spike alone can't qualify
a hand as hard) lives in ``full_hand_batch`` where the leg scores exist;
:func:`postflop_leg_count` is the shared "does this hand even have a
postflop spine" helper.
"""

from __future__ import annotations

import math
from typing import Any

_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_AGGRESSIVE_VERBS = ("bet", "raise")

# A fold-ender above this frequency is a non-decision ("fold 5-high").
TRIVIAL_FOLD_ENDER_FREQ = 0.90
# Share of a batch allowed to be passive (checkdown) stories.
DEFAULT_PASSIVE_SHARE = 0.15


def postflop_leg_count(hand: Any) -> int:
    """How many legs of ``hand`` are postflop decisions (its postflop spine)."""
    return sum(
        1 for leg in hand.legs if leg.kind == "postflop" and leg.spot is not None
    )


def _deepest_postflop_leg(hand: Any):
    for leg in reversed(hand.legs):
        if leg.kind == "postflop" and leg.spot is not None:
            return leg
    return None


def _postflop_history(hand: Any) -> tuple:
    """The postflop action steps of the hand's line (from the deepest leg's
    node history — every shallower leg's history is a prefix of it)."""
    leg = _deepest_postflop_leg(hand)
    if leg is None:
        return ()
    return tuple(
        s for s in leg.spot.node.history if s.street in ("flop", "turn", "river")
    )


def aggressive_steps(hand: Any) -> int:
    """Bet/raise steps in the hand's whole postflop line (any street)."""
    return sum(
        1 for s in _postflop_history(hand) if s.verb in _AGGRESSIVE_VERBS
    )


def is_passive_line(hand: Any) -> bool:
    """True for a BORING checkdown story: the hand ends on the turn or river,
    no bet or raise happened on any street BEFORE the ending street, and the
    final question is not a genuine bluff-catch.

    A bet ON the ending street (e.g. the river stab the ender faces) does not
    by itself redeem the line — that is the "nothing happened until the very
    end" shape being capped. EXCEPTION (July 22 2026, tuned on v8): when that
    late bet sets up a real BLUFF-CATCH — the ender faces a bet and the
    correct action is a non-fold (call/raise) — the hand is a legitimate,
    common teaching story ("it checked down, villain stabs, do you pay off?")
    and escapes the cap; without this carve-out the passive cap starved the
    river bucket and batches over-rotated into turn-fold endings. Checkdowns
    into a fold (even a mixed one) or into a "do you stab?" decision stay
    capped. Flop/preflop enders are exempt (no earlier postflop street
    exists); hands with no postflop legs are never passive.
    """
    if postflop_leg_count(hand) == 0:
        return False
    ender = hand.legs[-1]
    ending = _STREET_ORDER.get(ender.street, 0)
    if ending < _STREET_ORDER["turn"]:
        return False
    if any(
        s.verb in _AGGRESSIVE_VERBS
        and _STREET_ORDER.get(s.street, 0) < ending
        for s in _postflop_history(hand)
    ):
        return False
    if (
        ender.kind == "postflop"
        and ender.spot is not None
        and ender.spot.node.to_call_bb > 0
        and ender.spot.dominant_verb != "fold"
    ):
        return False  # checkdown into a real bluff-catch: keep, uncapped
    return True


def is_trivial_fold_ender(
    hand: Any, *, max_freq: float = TRIVIAL_FOLD_ENDER_FREQ
) -> bool:
    """True when the featured final question is a near-pure fold.

    The worthiness window (65-99%) admits folds up to 99%; above ``max_freq``
    a fold facing a bet is not a decision worth ending a play-through on.
    Mixed folds — the real bluff-catch zone — pass. Preflop fold-enders are
    NOT judged here: they come from the pack's own frequency-windowed ender
    pools, a deliberate product quota.
    """
    if not hand.legs:
        return False
    leg = hand.legs[-1]
    if leg.kind != "postflop" or leg.spot is None:
        return False
    spot = leg.spot
    return spot.dominant_verb == "fold" and spot.dominant_frequency > max_freq


def educational_density(hand: Any) -> float:
    """Deterministic interestingness score for pool ordering (higher = first).

    Cheap by design (spot/node fields only — no equity sims, no LLM), so it
    can run over a 20x-oversized candidate pool:

    * up to +6 for action content (2 points per bet/raise in the line, capped
      at three — a raised multi-street pot maxes it);
    * up to +1.5 for a genuinely MIXED ender (peaks near 78% dominant
      frequency: a clear-but-close decision; pure 99% spots score ~0 here);
    * +0.75 when the ender faces a bet (bluff-catch / raise decisions);
    * +0.5 when the line contains a raise (raised pots are rare and rich);
    * −1.5 for a passive (checkdown) line.

    Hands with no postflop legs (preflop enders) score 0 and sink — harmless,
    because the length quotas pick them by street bucket regardless of
    position.
    """
    if postflop_leg_count(hand) == 0:
        return 0.0
    steps = _postflop_history(hand)
    ender = hand.legs[-1]
    score = 2.0 * min(aggressive_steps(hand), 3)
    if ender.kind == "postflop" and ender.spot is not None:
        f = ender.spot.dominant_frequency
        score += 1.5 * max(0.0, 1.0 - abs(f - 0.78) / 0.35)
        if ender.spot.node.to_call_bb > 0:
            score += 0.75
    if any(s.verb == "raise" for s in steps):
        score += 0.5
    if is_passive_line(hand):
        score -= 1.5
    return score


def apply_action_heavy_policy(
    hands: list,
    *,
    total_hands: int,
    passive_share: float = DEFAULT_PASSIVE_SHARE,
    fold_ender_max_freq: float = TRIVIAL_FOLD_ENDER_FREQ,
) -> tuple[list, dict[str, int]]:
    """Order the candidate pool by educational density and apply the gates.

    Returns ``(kept, counters)`` where ``counters`` reports what was dropped
    (never silent): ``hands_excluded_trivial_fold_ender`` and
    ``hands_excluded_passive_line`` (passive hands beyond the cap — the cap
    is ``ceil(passive_share * total_hands)``, and the BEST-scoring passive
    hands are the ones kept). Hands with no postflop spine pass through
    untouched. Deterministic: the density sort is stable, so equal-density
    hands keep their (variety-seeded) input order.
    """
    cap = math.ceil(max(0.0, passive_share) * max(0, total_hands))
    ordered = sorted(hands, key=lambda h: -educational_density(h))
    kept: list = []
    passive_kept = 0
    excluded_fold = 0
    excluded_passive = 0
    for hand in ordered:
        if postflop_leg_count(hand) == 0:
            kept.append(hand)
            continue
        if is_trivial_fold_ender(hand, max_freq=fold_ender_max_freq):
            excluded_fold += 1
            continue
        if is_passive_line(hand):
            if passive_kept >= cap:
                excluded_passive += 1
                continue
            passive_kept += 1
        kept.append(hand)
    return kept, {
        "hands_excluded_trivial_fold_ender": excluded_fold,
        "hands_excluded_passive_line": excluded_passive,
        "passive_hands_kept": passive_kept,
    }


__all__ = [
    "DEFAULT_PASSIVE_SHARE",
    "TRIVIAL_FOLD_ENDER_FREQ",
    "aggressive_steps",
    "apply_action_heavy_policy",
    "educational_density",
    "is_passive_line",
    "is_trivial_fold_ender",
    "postflop_leg_count",
]
