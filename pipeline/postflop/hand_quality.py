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
  The checkdown-into-a-river-BLUFF-CATCH subclass
  (:func:`is_bluffcatch_checkdown`) gets its own sub-quota (~25-30% of the
  river enders) instead — it was fully exempt July 22-23 and swallowed whole
  batches. Flop- and preflop-ending hands are exempt (there are no earlier
  postflop streets for action to have happened on).
* :func:`educational_density` — the ordering score: PRE-ender-street action
  content + a genuinely-mixed ender + facing-a-bet enders + raised lines
  float to the top; passive lines sink. (The ending street's own bet counts
  for nothing — otherwise a pure checkdown into a river stab scores as
  "action".) The selectors downstream (length quotas, the diversify mix,
  greedy balance) all preserve input order within their buckets, so this
  ordering survives every mode.
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
# Share of the RIVER-ender quota allowed to be checkdown-into-a-bluff-catch
# stories ("it checked down, villain stabs, do you pay off?"). July 23 2026:
# this class was fully EXEMPT from the passive cap, and with a shallow river
# sample it swallowed the batch -- the 8-hand production-validation run
# shipped 6/7 postflop hands as x/x, x/x, river-bet. Legitimate teaching
# story, so it keeps a real (~25-30%) slice of the river enders, but a
# sub-quota now, not a free pass.
BLUFFCATCH_RIVER_SHARE = 0.30


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


def _ending_street_order(hand: Any) -> int:
    return _STREET_ORDER.get(hand.legs[-1].street, 0) if hand.legs else 0


def aggressive_steps(hand: Any) -> int:
    """Bet/raise steps on streets BEFORE the hand's ending street.

    July 23 2026: was the whole line. Counting the ending street's own bet
    let a pure checkdown-into-a-river-stab score as "action content" (+2 for
    the very stab the ender faces), which floated x/x, x/x, river-bet shapes
    to the top of the density ordering — the last 8-hand batch shipped 6/7
    postflop hands in exactly that shape. Pre-ender streets only makes this
    consistent with :func:`is_passive_line`'s definition of a checkdown.
    """
    ending = _ending_street_order(hand)
    return sum(
        1
        for s in _postflop_history(hand)
        if s.verb in _AGGRESSIVE_VERBS and _STREET_ORDER.get(s.street, 0) < ending
    )


def is_passive_line(hand: Any) -> bool:
    """True for a BORING checkdown story: the hand ends on the turn or river
    and no bet or raise happened on any street BEFORE the ending street.

    A bet ON the ending street (e.g. the river stab the ender faces) does not
    redeem the line — that is the "nothing happened until the very end" shape
    being capped. A checkdown into a genuine river BLUFF-CATCH (the ender
    faces a bet and the correct action is a non-fold) IS passive too, but
    :func:`apply_action_heavy_policy` gives that class its own sub-quota
    (~25-30% of the river enders) instead of the generic passive cap — it was
    a full exemption July 22-23 and swallowed whole batches (6/7 hands as
    x/x, x/x, river-bet); see :func:`is_bluffcatch_checkdown`. Flop/preflop
    enders are exempt (no earlier postflop street exists); hands with no
    postflop legs are never passive.
    """
    if postflop_leg_count(hand) == 0:
        return False
    ending = _ending_street_order(hand)
    if ending < _STREET_ORDER["turn"]:
        return False
    return aggressive_steps(hand) == 0


def is_bluffcatch_checkdown(hand: Any) -> bool:
    """A passive line ending in a genuine RIVER bluff-catch: the hand checked
    down, villain stabbed the river, and the correct action is a non-fold
    (call/raise). A legitimate, common teaching story — kept, but under its
    own :data:`BLUFFCATCH_RIVER_SHARE` sub-quota rather than the generic
    passive cap. (Pre-river bluff-catch enders can't exist in full-hand
    batches — the no-mid-hand-endings rule makes every pre-river ender a
    fold — so this class is river-only by construction.)
    """
    if not is_passive_line(hand):
        return False
    ender = hand.legs[-1]
    return (
        ender.street == "river"
        and ender.kind == "postflop"
        and ender.spot is not None
        and ender.spot.node.to_call_bb > 0
        and ender.spot.dominant_verb != "fold"
    )


def is_exciting_hand(hand: Any) -> bool:
    """🔥 "Exciting pots" toggle (July 23 2026, user ask): the play-through's
    FEATURED FINAL decision is a big-hand spot on a genuinely heated line
    (see :func:`pipeline.postflop.spot_selection.spot_is_exciting`: hero
    holds a premium/strong made hand, and the line carries a raise or
    two-plus bets). Preflop enders are never exciting pots -- with the
    toggle on they drop out and the length quotas backfill honestly."""
    from pipeline.postflop.spot_selection import spot_is_exciting  # noqa: PLC0415

    if not hand.legs:
        return False
    ender = hand.legs[-1]
    if ender.kind != "postflop" or ender.spot is None:
        return False
    return spot_is_exciting(ender.spot)


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

    * up to +6 for PRE-ender-street action content (2 points per bet/raise
      before the ending street, capped at three — a raised multi-street pot
      maxes it; the ending street's own bet counts for nothing);
    * up to +1.5 for a genuinely MIXED ender (peaks near 78% dominant
      frequency: a clear-but-close decision; pure 99% spots score ~0 here);
    * +0.75 when the ender faces a bet (bluff-catch / raise decisions);
    * +0.5 when a pre-ender street contains a raise (raised pots are rare
      and rich);
    * −1.5 for a passive (checkdown) line.

    Hands with no postflop legs (preflop enders) score 0 and sink — harmless,
    because the length quotas pick them by street bucket regardless of
    position.
    """
    if postflop_leg_count(hand) == 0:
        return 0.0
    steps = _postflop_history(hand)
    ender = hand.legs[-1]
    ending = _ending_street_order(hand)
    score = 2.0 * min(aggressive_steps(hand), 3)
    if ender.kind == "postflop" and ender.spot is not None:
        f = ender.spot.dominant_frequency
        score += 1.5 * max(0.0, 1.0 - abs(f - 0.78) / 0.35)
        if ender.spot.node.to_call_bb > 0:
            score += 0.75
    if any(
        s.verb == "raise" and _STREET_ORDER.get(s.street, 0) < ending
        for s in steps
    ):
        score += 0.5
    if is_passive_line(hand):
        score -= 1.5
    return score


def line_shape_signature(hand: Any) -> tuple:
    """The hand's LINE SHAPE: the (street, verb) sequence of its postflop
    history plus the ender's street and dominant verb. Sizes and cards are
    deliberately ignored — "flop c-bet call, turn barrel, fold" is ONE shape
    whatever the runout, which is exactly the axis a batch must vary on
    (July 23 2026: a paid batch shipped 7/7 postflop hands as the SAME
    shape with different combos — density-identical hands clustered at the
    top of the street bucket)."""
    if not hand.legs:
        return ("empty",)
    ender = hand.legs[-1]
    sig: list = [(s.street, s.verb) for s in _postflop_history(hand)]
    verb = ""
    if ender.kind == "postflop" and ender.spot is not None:
        verb = ender.spot.dominant_verb
    sig.append(("ender", ender.street, verb))
    return tuple(sig)


def _rotate_shapes_within_streets(kept: list) -> list:
    """Re-order ``kept`` so that WITHIN each ending street the hands rotate
    across distinct line shapes (round-robin: the best hand of each shape,
    then the second-best of each, ...), while every hand keeps its original
    STREET slot — so the cross-street arrangement, and therefore every
    downstream street-bucketed selector, sees the same street sequence but
    shape variety at the top of each bucket. Passive/bluff-catch hands are
    rotated in their own tier AFTER the non-passive tier (they were capped
    by the caller, and they must keep sinking below real-action lines).
    Deterministic and stable throughout."""
    by_street: dict[str, list] = {}
    for h in kept:
        by_street.setdefault(h.legs[-1].street if h.legs else "", []).append(h)
    rotated: dict[str, list] = {}
    for street, hands in by_street.items():
        out: list = []
        for tier_test in (lambda h: not is_passive_line(h), is_passive_line):
            tier = [h for h in hands if tier_test(h)]
            groups: dict[tuple, list] = {}
            for h in tier:  # density order preserved within each group
                groups.setdefault(line_shape_signature(h), []).append(h)
            round_i, added = 0, 0
            while added < len(tier):
                for sig in groups:  # insertion order = best-density-first
                    if round_i < len(groups[sig]):
                        out.append(groups[sig][round_i])
                        added += 1
                round_i += 1
        rotated[street] = out
    feeds = {s: iter(h) for s, h in rotated.items()}
    return [next(feeds[h.legs[-1].street if h.legs else ""]) for h in kept]


def apply_action_heavy_policy(
    hands: list,
    *,
    total_hands: int,
    passive_share: float = DEFAULT_PASSIVE_SHARE,
    fold_ender_max_freq: float = TRIVIAL_FOLD_ENDER_FREQ,
    bluffcatch_river_share: float = BLUFFCATCH_RIVER_SHARE,
    river_ender_target: int | None = None,
) -> tuple[list, dict[str, int]]:
    """Order the candidate pool by educational density and apply the gates.

    Returns ``(kept, counters)`` where ``counters`` reports what was dropped
    (never silent): ``hands_excluded_trivial_fold_ender``,
    ``hands_excluded_passive_line`` (plain checkdowns beyond the generic cap
    — ``ceil(passive_share * total_hands)``), and
    ``hands_excluded_bluffcatch_checkdown`` (checkdown-into-a-river-bluff-
    catch stories beyond THEIR sub-quota —
    ``ceil(bluffcatch_river_share * river_ender_target)``, where
    ``river_ender_target`` is the number of river-ending hands the batch
    aims for; defaults to ``total_hands`` when the caller has no length
    profile). The BEST-scoring hands of each capped class are the ones kept.
    Hands with no postflop spine pass through untouched. Deterministic: the
    density sort is stable, so equal-density hands keep their
    (variety-seeded) input order.

    The returned order is density-desc, then SHAPE-ROTATED within each
    ending street (see :func:`_rotate_shapes_within_streets`): the street
    quotas downstream pick in input order, so without the rotation a batch
    fills each street with N copies of the densest line shape. Passive
    hands still sink below the non-passive tier within their street.
    """
    cap = math.ceil(max(0.0, passive_share) * max(0, total_hands))
    river_target = total_hands if river_ender_target is None else river_ender_target
    bc_cap = math.ceil(max(0.0, bluffcatch_river_share) * max(0, river_target))
    ordered = sorted(hands, key=lambda h: -educational_density(h))
    kept: list = []
    passive_kept = 0
    bluffcatch_kept = 0
    excluded_fold = 0
    excluded_passive = 0
    excluded_bluffcatch = 0
    for hand in ordered:
        if postflop_leg_count(hand) == 0:
            kept.append(hand)
            continue
        if is_trivial_fold_ender(hand, max_freq=fold_ender_max_freq):
            excluded_fold += 1
            continue
        if is_bluffcatch_checkdown(hand):
            # Its own sub-quota (July 23 2026): a legitimate teaching story,
            # but the old full exemption let it swallow the river bucket.
            if bluffcatch_kept >= bc_cap:
                excluded_bluffcatch += 1
                continue
            bluffcatch_kept += 1
        elif is_passive_line(hand):
            if passive_kept >= cap:
                excluded_passive += 1
                continue
            passive_kept += 1
        kept.append(hand)
    # Shape variety (July 23 2026): rotate distinct line shapes to the top
    # of each ending-street bucket, or the street quota fills with N copies
    # of the single densest shape (a paid batch shipped 7/7 postflop hands
    # as the same c-bet-call/turn-barrel/fold line with different combos).
    kept = _rotate_shapes_within_streets(kept)
    return kept, {
        "hands_excluded_trivial_fold_ender": excluded_fold,
        "hands_excluded_passive_line": excluded_passive,
        "hands_excluded_bluffcatch_checkdown": excluded_bluffcatch,
        "passive_hands_kept": passive_kept,
        "bluffcatch_checkdowns_kept": bluffcatch_kept,
    }


__all__ = [
    "BLUFFCATCH_RIVER_SHARE",
    "DEFAULT_PASSIVE_SHARE",
    "TRIVIAL_FOLD_ENDER_FREQ",
    "aggressive_steps",
    "apply_action_heavy_policy",
    "educational_density",
    "is_bluffcatch_checkdown",
    "is_exciting_hand",
    "is_passive_line",
    "is_trivial_fold_ender",
    "line_shape_signature",
    "postflop_leg_count",
]
