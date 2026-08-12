"""Deterministic hand-type range breakdown for PLO -- the app's answer to
"PLO has no 13x13 grid" (the GTO Wizard "Categories" panel equivalent).

For one decision spot this aggregates each STILL-ACTIVE player's preflop range
into structural hand-type buckets (pair pattern x suit pattern, e.g. "Unpaired
Double-Suited"), plus deterministic natural-language copy relating the range to
hero's own hand. Two views, by design:

* **Hero** (the player whose decision the question asks): the full reaching
  range, bucketed, WITH the fold/call/raise action split per bucket -- "how
  your whole range plays this spot, by shape".
* **Each other still-live player**: the composition of the range they took to
  get here (their raising or calling range), bucketed -- "what that player's
  range looks like now, by shape". No action split; they have already acted.

Everything is PYTHON from solver output (never the LLM): shares are
combo-weighted (a rainbow hand is 24 combos, quads 1), so they match the app's
combo counts. Pure -- facts + pack in, a JSON-able dict out. The copy is
fact-bound (it only states the computed numbers), deterministic, and handles
every edge case (open spots with no villain, unresolvable villain nodes, rare
hero shapes, pure ranges, monotone/three-suited hands).

Reuses the sanctioned range-resolution seam: a villain's decision range is the
file at ``villain_range_stem(node, index)`` (:mod:`pipeline.plo.fact_extractor`),
loaded with ``range_at`` -- the same primitives ``range_examples`` uses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from pipeline.plo.action_history import _villain_ref, display_seat
from pipeline.plo.fact_extractor import villain_range_stem
from pipeline.plo.hand_model import classify_plo_hand
from pipeline.plo.hand_order import cards_at, combo_multiplicity
from pipeline.plo.options import (
    _action_aggression,
    _hero_raise_level,
    canonicalize_action_label,
    canonicalize_strategy,
    integer_percentages,
    is_check_spot,
)
from pipeline.plo.pack import PloActionType, range_at
from pipeline.plo.spot_sampler import _cached_node_values

if TYPE_CHECKING:
    from pipeline.plo.fact_extractor import PloFacts
    from pipeline.plo.node_enumerator import PloDecisionNode
    from pipeline.plo.pack import PloPack

# Human labels for the two structural axes (the category = "<pair> <suit>").
_PAIR_LABEL: dict[str, str] = {
    "unpaired": "Unpaired",
    "one_pair": "One Pair",
    "two_pair": "Two Pair",
    "trips": "Trips",
    "quads": "Quads",
}
_SUIT_LABEL: dict[str, str] = {
    "double_suited": "Double-Suited",
    "single_suited": "Single-Suited",
    "rainbow": "Rainbow",
    "three_suited": "Three-Suited",
    "monotone": "Monotone",
}

_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
_FOLD = PloActionType.FOLD
# Buckets below this combo-share are omitted from the cell (noise), and the
# cell keeps at most this many buckets per player (a CSV cell, not a report).
_MIN_BUCKET_PCT = 0.5
_MAX_BUCKETS = 8
# Raise-count -> the name of that raise, for a villain's role label.
_RAISE_ROLE = {1: "open", 2: "3-bet", 3: "4-bet", 4: "5-bet"}


def _category(index: int) -> str:
    """The hand-type bucket label for a hand-order index (pure, cached)."""
    return _category_for_cards(tuple(cards_at(index)))


@lru_cache(maxsize=None)
def _category_for_cards(cards: tuple[str, ...]) -> str:
    hc = classify_plo_hand(cards)
    return f"{_PAIR_LABEL[hc.pair_pattern]} {_SUIT_LABEL[hc.suit_pattern]}"


@dataclass(frozen=True)
class _Bucket:
    category: str
    combo_share: float  # % of the player's range, combo-weighted
    actions: dict[str, float] | None  # hero only: label -> % within the bucket


def _round_actions(actions: dict[str, float]) -> dict[str, int]:
    """Integer action percents that sum to 100 (largest-remainder)."""
    total = sum(actions.values())
    if total <= 0:
        return {}
    scaled = [(lbl, v / total * 100.0) for lbl, v in actions.items()]
    floors = {lbl: int(v) for lbl, v in scaled}
    deficit = 100 - sum(floors.values())
    if deficit > 0:
        order = sorted(scaled, key=lambda kv: -(kv[1] % 1))
        for lbl, _ in order[:deficit]:
            floors[lbl] += 1
    return {lbl: floors[lbl] for lbl, _ in scaled if floors[lbl] > 0}


def _finalize(
    combo: dict[str, float],
    action_combo: dict[str, dict[str, float]] | None,
) -> list[_Bucket]:
    """Turn raw combo-weight accumulators into sorted, capped buckets."""
    total = sum(combo.values())
    if total <= 0:
        return []
    ordered = sorted(combo.items(), key=lambda kv: -kv[1])
    out: list[_Bucket] = []
    for cat, w in ordered:
        pct = w / total * 100.0
        if pct < _MIN_BUCKET_PCT or len(out) >= _MAX_BUCKETS:
            continue
        actions = None
        if action_combo is not None:
            actions = _norm_actions(_round_actions(action_combo.get(cat, {})))
        out.append(_Bucket(cat, round(pct, 1), actions))
    return out


def _norm_actions(rounded: dict[str, int]) -> dict[str, int] | None:
    return rounded or None


# --- hero: reaching range, bucketed, with the action split -----------------
def _hero_buckets(
    facts: PloFacts,
) -> tuple[list[_Bucket], str]:
    """Hero's whole reaching range by hand type + hero's own hand's category.

    Combo-weighted; the per-bucket action split uses the CANONICAL labels the
    rest of the row speaks ("3-bet", not "Raise 100%"), so the breakdown
    agrees with the ``action_frequencies`` column.
    """
    node = facts.spot.node
    raise_level = _hero_raise_level(facts)
    check_spot = is_check_spot(facts)
    # Per-action per-hand weights, keyed by canonical label (two raw raise
    # sizes collapsing to one canonical label merge).
    per_action: dict[str, tuple[tuple[float, float], ...]] = {}
    canon_of: dict[str, str] = {}
    for opt in node.actions:
        vals = _cached_node_values(str(opt.path))
        canon = canonicalize_action_label(
            opt.label, raise_level=raise_level, check_spot=check_spot
        )
        canon_of[opt.label] = canon
        per_action[opt.label] = vals

    combo: dict[str, float] = defaultdict(float)
    action_combo: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    hand_count = len(next(iter(per_action.values()))) if per_action else 0
    for i in range(hand_count):
        reach = sum(per_action[lbl][i][0] for lbl in per_action)
        if reach <= 0:
            continue
        mult = combo_multiplicity(i)
        cat = _category(i)
        combo[cat] += reach * mult
        for lbl, vals in per_action.items():
            w = vals[i][0] * mult
            if w > 0:
                action_combo[cat][canon_of[lbl]] += w
    hero_cat = _category_for_cards(tuple(facts.spot.hero_cards))
    return _finalize(combo, action_combo), hero_cat


def hero_range_action_shares(facts: PloFacts) -> tuple[dict[str, float], float]:
    """Combo-weighted frequency of each canonical action across hero's WHOLE
    reaching range at this node -- the app's "what percent of hands are played"
    (range width) number.

    Returns ``({canonical_label: share_of_reaching_range}, total_reach_combos)``.
    Each share is the fraction of the hands hero can hold at this spot that take
    that action, combo-weighted (a rainbow hand is 24 combos, quads 1) so it
    matches the app's combo counts. At an opening (first-in) node hero's whole
    deal reaches, so the top action's share is the classic RFI/range width
    ("UTG opens 18% of hands"); at a facing node it reads "of the hands you get
    here with, X% take this line". Pure frequency read from the pack node
    values -- no equity sim, deterministic. ``({}, 0.0)`` if the node is
    unreachable/empty (never invents a number)."""
    node = facts.spot.node
    raise_level = _hero_raise_level(facts)
    check_spot = is_check_spot(facts)
    per_action: dict[str, tuple[tuple[float, float], ...]] = {}
    canon_of: dict[str, str] = {}
    for opt in node.actions:
        per_action[opt.label] = _cached_node_values(str(opt.path))
        canon_of[opt.label] = canonicalize_action_label(
            opt.label, raise_level=raise_level, check_spot=check_spot
        )
    action_combo: dict[str, float] = defaultdict(float)
    total_reach = 0.0
    hand_count = len(next(iter(per_action.values()))) if per_action else 0
    for i in range(hand_count):
        reach = sum(per_action[lbl][i][0] for lbl in per_action)
        if reach <= 0:
            continue
        mult = combo_multiplicity(i)
        total_reach += reach * mult
        for lbl, vals in per_action.items():
            w = vals[i][0] * mult
            if w > 0:
                action_combo[canon_of[lbl]] += w
    if total_reach <= 0:
        return {}, 0.0
    return {k: v / total_reach for k, v in action_combo.items()}, total_reach


# --- villain: the range that took the line to reach hero -------------------
@lru_cache(maxsize=256)
@lru_cache(maxsize=256)
def _villain_buckets_by_stem(pack: PloPack, stem: str) -> tuple[_Bucket, ...]:
    """Combo-weighted hand-type composition of the range at ``stem``.

    Cached per (pack, stem): a batch revisits the same villain node for many
    hero hands. Returns () when the file can't be read (never invents copy).
    """
    try:
        entries = range_at(pack, stem)
    except (OSError, ValueError):
        return ()
    combo: dict[str, float] = defaultdict(float)
    for e in entries:
        if e.p <= 0:
            continue
        combo[_category(e.index)] += e.p * combo_multiplicity(e.index)
    return tuple(_finalize(combo, None))


def _still_active_villains(
    node: PloDecisionNode,
) -> list[tuple[str, int, PloActionType]]:
    """(seat, last-action index, last action) for every still-live opponent.

    A seat counts as still in the hand only if it voluntarily put chips in
    (a call or raise, never a blind post -- blinds aren't in the history) and
    its LAST action is not a fold. Hero is excluded. Ordered by seat's last
    action index (so the earliest-committed player reads first).
    """
    last: dict[str, tuple[int, PloActionType]] = {}
    for i, a in enumerate(node.history_before):
        last[a.seat] = (i, a.action)
    out = [
        (seat, idx, act)
        for seat, (idx, act) in last.items()
        if seat != node.actor and act is not _FOLD
    ]
    return sorted(out, key=lambda t: t[1])


def _villain_role(node: PloDecisionNode, idx: int, action: PloActionType) -> str:
    """A short label for a villain's line: "open" / "3-bet" / "call" ..."""
    if action is _FOLD:  # pragma: no cover -- callers exclude folds
        return "fold"
    if action is PloActionType.CALL:
        return "call"
    # A raise: name it by how many raises had happened through this action.
    raises = sum(
        1
        for a in node.history_before[: idx + 1]
        if a.action in _AGGRESSIVE
    )
    return _RAISE_ROLE.get(raises, "raise")


# --- copy (deterministic, fact-bound) --------------------------------------
def _pct(x: float) -> str:
    return f"{x:.0f}%"


# Third-person verb for a canonical action label ("Call" -> "calls").
_ACTION_VERB = {
    "Fold": "folds", "Call": "calls", "Check": "checks", "Raise": "raises",
    "3-bet": "3-bets", "4-bet": "4-bets", "5-bet": "5-bets",
    "All-in": "jams",
}


def _verb(label: str) -> str:
    return _ACTION_VERB.get(label, label.lower())


def _lead_actions(bucket: _Bucket) -> str:
    """"raises 26%, folds 74%" for a hero bucket's action split."""
    if not bucket.actions:
        return ""
    parts = [
        f"{_verb(lbl)} {pct}%"
        for lbl, pct in sorted(bucket.actions.items(), key=lambda kv: -kv[1])
    ]
    return ", ".join(parts)


def hand_action_percentages(facts: PloFacts) -> dict[str, int]:
    """This exact hand's own action mix as house integer percents.

    Canonical labels ("3-bet", never "Raise 100%") via the SAME
    :func:`pipeline.plo.options.integer_percentages` allocation as the CSV
    ``action_frequencies`` column and the SOLVER DATA ``action_strategy``
    block, so the breakdown can never quote a percentage the question's
    other surfaces disagree with. ``{}`` when the spot carries no strategy.
    """
    strategy = canonicalize_strategy(facts)
    if not strategy or sum(strategy.values()) <= 0:
        return {}
    return integer_percentages(strategy)


def _hand_reconciliation(
    hand_actions: dict[str, int] | None,
    hero_bucket: _Bucket | None,
) -> str:
    """The sentence tying the SHAPE bucket's split back to this exact hand.

    The bucket line states how the whole shape family plays ("a shape this
    range folds 67%, calls 21%, 3-bets 12% here") while the question shows the
    hand's OWN answer (Call 94%) -- read side by side those look contradictory,
    so the summary must always end by reconciling to the hand's own mix:

    * agreement (hand's dominant action == the bucket's plurality action):
      a plain confirmation, "Your exact hand calls 94%.";
    * hand STRONGER than its bucket (more aggressive dominant action):
      "Overall this shape folds 67% here, but yours is one of the stronger
      hands in the shape: this exact hand calls 94%.";
    * hand WEAKER than its bucket (less aggressive dominant action):
      "Overall this shape calls 60% here, but this exact combination is one
      of the weaker ones in the shape: it folds 88%."

    Returns "" when the hand has no strategy to state (never invents one);
    leading space included so callers can append verbatim.
    """
    if not hand_actions:
        return ""
    hand_label = max(hand_actions, key=hand_actions.__getitem__)
    hand_pct = hand_actions[hand_label]
    plain = f" Your exact hand {_verb(hand_label)} {hand_pct}%."
    if hero_bucket is None or not hero_bucket.actions:
        return plain
    bucket_label = max(hero_bucket.actions, key=hero_bucket.actions.__getitem__)
    bucket_pct = hero_bucket.actions[bucket_label]
    if bucket_label == hand_label:
        return plain
    lead = f" Overall this shape {_verb(bucket_label)} {bucket_pct}% here, but "
    if _action_aggression(hand_label) > _action_aggression(bucket_label):
        return (
            lead + "yours is one of the stronger hands in the shape: "
            f"this exact hand {_verb(hand_label)} {hand_pct}%."
        )
    return (
        lead + "this exact combination is one of the weaker ones in the "
        f"shape: it {_verb(hand_label)} {hand_pct}%."
    )


def _shape_phrase(buckets: list[_Bucket], joiner: str) -> str:
    """"38% unpaired single-suited and 19% one pair single-suited"."""
    top = buckets[0]
    shape = f"{top.combo_share:.0f}% {top.category.lower()}"
    if len(buckets) > 1:
        shape += f"{joiner}{buckets[1].combo_share:.0f}% {buckets[1].category.lower()}"
    return shape


def _hero_copy(
    buckets: list[_Bucket],
    hero_cat: str,
    is_open: bool,
    hand_actions: dict[str, int] | None = None,
) -> str:
    """One sentence on hero's range shape + where hero's own hand sits in it.

    Hero's seat is deliberately omitted -- the question already states it, and
    naming it here would just repeat it.

    INVARIANT (Aug 2026, live user confusion): whenever ``hand_actions`` is
    known the summary ENDS with :func:`_hand_reconciliation` -- the bucket's
    shape-family split alone reads as contradicting the question's own answer
    ("the shape folds 67%" next to "Mostly Call 94%").
    """
    hero_bucket = next((b for b in buckets if b.category == hero_cat), None)
    reconcile = _hand_reconciliation(hand_actions, hero_bucket)
    if not buckets:
        return (
            f"Your hand is {hero_cat.lower()}. There is not enough of a range "
            f"at this spot to break it down by shape.{reconcile}"
        )
    lead = "Your opening range is " if is_open else "Your continuing range is "
    shape = _shape_phrase(buckets, " and ")
    if hero_bucket is not None and hero_bucket.actions:
        placement = (
            f" Your hand is {hero_cat.lower()}, a shape this range "
            f"{_lead_actions(hero_bucket)} here."
        )
    elif hero_bucket is not None:
        placement = (
            f" Your hand is {hero_cat.lower()}, "
            f"{hero_bucket.combo_share:.0f}% of the range."
        )
    else:
        placement = f" Your hand is {hero_cat.lower()}, a rare shape in this range."
    return f"{lead}{shape}.{placement}{reconcile}"


_ROLE_WORD = {
    "open": "opening", "3-bet": "3-betting", "4-bet": "4-betting",
    "5-bet": "5-betting", "call": "calling", "raise": "raising",
}


def _villain_copy(seat_ref: str, role: str, buckets: list[_Bucket]) -> str:
    if not buckets:
        return f"{seat_ref}'s range could not be broken down by shape."
    role_word = _ROLE_WORD.get(role, "continuing")
    # "is made up of" (July 22 2026, user wording ask): plainer than the bare
    # "is 37% ...", which read as if the range WERE one shape.
    return (
        f"{seat_ref}'s {role_word} range is made up of "
        f"{_shape_phrase(buckets, ', ')}."
    )


def _bucket_json(b: _Bucket) -> dict[str, object]:
    d: dict[str, object] = {"category": b.category, "pct": b.combo_share}
    if b.actions:
        d["actions"] = b.actions
    return d


def build_range_breakdown(facts: PloFacts, pack: PloPack) -> dict[str, object]:
    """The full hero + still-active-villains hand-type breakdown for a spot.

    Shape (JSON-able, the ``range_breakdown`` CSV cell)::

        {"version": 1,
         "hero": {"seat","position","role":"decision","hand_category",
                  "buckets":[{"category","pct","actions"?}...],
                  "hand_actions"?: {"Call": 94, "3-bet": 6, "Fold": 0},
                  "summary"},
         "villains": [{"seat","position","role","buckets":[...],"summary"}...],
         "copy": "<one-line range read relative to hero's hand>"}

    ``hand_actions`` (Aug 2026) is THIS EXACT HAND's own action mix -- house
    integer percents on canonical labels, the same
    :func:`pipeline.plo.options.integer_percentages` allocation as the
    ``action_frequencies`` column -- so the app can render the
    bucket-vs-hand reconciliation natively (the bucket ``actions`` split is
    the whole SHAPE family; users read the two as contradictory without it).
    Present whenever the spot carries a strategy; omitted otherwise. The hero
    ``summary`` and the top-level ``copy`` also state it in prose.

    Deterministic and total: an open spot yields an empty ``villains`` list; an
    unresolvable villain node is dropped (never a false summary); a hero hand
    of a rare shape still gets an honest placement sentence.
    """
    node = facts.spot.node
    table_size = node.table_size
    villains_meta = _still_active_villains(node)
    is_open = not villains_meta and not node.history_before

    hero_buckets, hero_cat = _hero_buckets(facts)
    hand_actions = hand_action_percentages(facts)
    hero: dict[str, object] = {
        "seat": node.actor,
        "position": display_seat(node.actor, table_size=table_size),
        "role": "decision",
        "hand_category": hero_cat,
        "buckets": [_bucket_json(b) for b in hero_buckets],
        "summary": _hero_copy(hero_buckets, hero_cat, is_open, hand_actions),
    }
    if hand_actions:
        hero["hand_actions"] = hand_actions

    villains: list[dict[str, object]] = []
    for seat, idx, action in villains_meta:
        stem = villain_range_stem(node, idx)
        vbuckets = list(_villain_buckets_by_stem(pack, stem))
        if not vbuckets:
            continue  # unresolvable / empty -- never fabricate a summary
        role = _villain_role(node, idx, action)
        # "The Button" / "UTG" -- the house subject phrasing (article for
        # positional seats, none for the UTG family).
        seat_ref = _villain_ref(seat, table_size=table_size)
        villains.append({
            "seat": seat,
            "position": display_seat(seat, table_size=table_size),
            "role": role,
            "buckets": [_bucket_json(b) for b in vbuckets],
            "summary": _villain_copy(seat_ref, role, vbuckets),
        })

    return {
        "version": 1,
        "hero": hero,
        "villains": villains,
        "copy": _overall_copy(hero_buckets, hero_cat, villains),
    }


def _overall_copy(
    hero_buckets: list[_Bucket],
    hero_cat: str,
    villains: list[dict[str, object]],
) -> str:
    """One-line read tying hero's hand to the ranges in the pot."""
    hero_bucket = next((b for b in hero_buckets if b.category == hero_cat), None)
    if hero_bucket is None or not hero_bucket.combo_share:
        base = f"Your {hero_cat.lower()} hand is an off-shape holding here."
    else:
        base = (
            f"Your hand is {hero_cat.lower()}, "
            f"{hero_bucket.combo_share:.0f}% of the range at this spot."
        )
    if villains:
        v = villains[0]
        vb = v.get("buckets") or []
        if vb:
            top = vb[0]
            # Mid-sentence: lower the leading article ("the Button"), leave
            # the UTG family untouched ("UTG").
            ref = str(v["summary"]).split("'s", 1)[0]
            ref = "the " + ref[4:] if ref.startswith("The ") else ref
            base += (
                f" You are up against {ref}, whose range is "
                f"{top['pct']:.0f}% {str(top['category']).lower()}."
            )
    return base


__all__ = ["build_range_breakdown"]
