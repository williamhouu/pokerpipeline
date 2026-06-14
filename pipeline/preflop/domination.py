"""Deterministic preflop domination map.

Which specific hands in villain's range have hero dominated (the real reason a
weak-kicker hand folds), and which hands hero dominates. Pure structural
classification of a 169-class vs a 169-class -- no equity sims, no LLM. The
explanation generator feeds the LLM the resulting *lists of hand classes*, and
the prompt may only name hands from those lists, so it can never invent "you're
dominated by AK" when villain doesn't have an ace.

"Domination" here is the structural concept (a shared card with a worse kicker,
or a pair that has you crushed), distinct from merely being behind a bigger
pair where your cards are still live. AhalfX vs a bigger ace is dominated; A9
vs QQ is "behind" (the ace is live), not dominated -- the classifier keeps that
distinction because it is exactly the lesson.

Hand selection is left to the caller: pass villain's already-weight-gated
heaviest classes (``VillainRangeStats.top_combos``), so the lists only ever
name hands villain actually holds with meaningful frequency. That is the
"barrier" -- the prose names hands from a vetted in-range list and never quotes
a raw percentage (exact combo counts live in the math panel instead).
"""
from __future__ import annotations

_RANK_VAL: dict[str, int] = {r: i for i, r in enumerate("23456789TJQKA", start=2)}

DOMINATES_YOU = "dominates_you"
YOU_DOMINATE = "you_dominate"
FLIP = "flip"
AHEAD = "ahead"
BEHIND = "behind"

# How many classes to name per bucket in prose before it gets unwieldy.
_MAX_PER_BUCKET = 6


def _parse(hand_class: str) -> tuple[int, int, bool] | None:
    """(high_value, low_value, is_pair) for a 169-class, or None if malformed."""
    hc = hand_class.strip()
    if len(hc) == 2 and hc[0] in _RANK_VAL and hc[0] == hc[1]:
        v = _RANK_VAL[hc[0]]
        return v, v, True
    if len(hc) == 3 and hc[0] in _RANK_VAL and hc[1] in _RANK_VAL and hc[2] in "so":
        a, b = _RANK_VAL[hc[0]], _RANK_VAL[hc[1]]
        return max(a, b), min(a, b), False
    return None


def classify_matchup(hero_class: str, villain_class: str) -> str | None:
    """How villain_class fares against hero_class, structurally.

    Returns one of dominates_you / you_dominate / flip / ahead / behind, from
    HERO's point of view (dominates_you = villain has hero dominated). None if
    either label is malformed.
    """
    hp, vp = _parse(hero_class), _parse(villain_class)
    if hp is None or vp is None:
        return None
    h_hi, h_lo, h_pair = hp
    v_hi, v_lo, v_pair = vp

    if h_pair and v_pair:
        if v_hi > h_hi:
            return DOMINATES_YOU
        if v_hi < h_hi:
            return YOU_DOMINATE
        return FLIP

    if h_pair and not v_pair:
        p = h_hi
        if p in (v_hi, v_lo):           # villain holds one of hero's set cards
            return YOU_DOMINATE
        overs = (v_hi > p) + (v_lo > p)
        if overs == 2:
            return FLIP                 # pair vs two overcards
        if overs == 1:
            return AHEAD                # pair vs one over, one under
        return YOU_DOMINATE             # pair over both villain cards

    if v_pair and not h_pair:
        p = v_hi
        if p in (h_hi, h_lo):           # villain pairs one of hero's cards
            return DOMINATES_YOU
        overs = (h_hi > p) + (h_lo > p)
        if overs == 2:
            return FLIP
        if overs == 1:
            return BEHIND               # one live overcard, still behind the pair
        return DOMINATES_YOU            # villain overpair to both hero cards

    # neither is a pair
    hset, vset = {h_hi, h_lo}, {v_hi, v_lo}
    shared = hset & vset
    if shared:
        h_kick, v_kick = hset - shared, vset - shared
        if h_kick and v_kick:
            hk, vk = max(h_kick), max(v_kick)
            if vk > hk:
                return DOMINATES_YOU
            if hk > vk:
                return YOU_DOMINATE
        return FLIP
    # no shared card: compare the made high cards (both live)
    if h_hi != v_hi:
        return AHEAD if h_hi > v_hi else BEHIND
    if h_lo != v_lo:
        return AHEAD if h_lo > v_lo else BEHIND
    return FLIP


def dominating_map(
    hero_class: str, villain_classes: list[str] | tuple[str, ...]
) -> dict[str, list[str]]:
    """Split villain's (already in-range, weight-gated) classes by matchup.

    Pass ``[c for c, _ in villain_stats.top_combos]`` -- the heaviest classes
    villain actually holds. Returns ``{"dominated_by": [...], "you_dominate":
    [...], "coinflips": [...]}`` of villain class labels, order preserved
    (heaviest first), each capped for prose sanity. Empty lists are kept so the
    caller can decide whether to surface the fact at all.
    """
    dominated_by: list[str] = []
    you_dominate: list[str] = []
    coinflips: list[str] = []
    seen: set[str] = set()
    for vc in villain_classes:
        if vc in seen:
            continue
        seen.add(vc)
        m = classify_matchup(hero_class, vc)
        if m == DOMINATES_YOU:
            dominated_by.append(vc)
        elif m == YOU_DOMINATE:
            you_dominate.append(vc)
        elif m == FLIP:
            coinflips.append(vc)
    return {
        "dominated_by": dominated_by[:_MAX_PER_BUCKET],
        "you_dominate": you_dominate[:_MAX_PER_BUCKET],
        "coinflips": coinflips[:_MAX_PER_BUCKET],
    }
