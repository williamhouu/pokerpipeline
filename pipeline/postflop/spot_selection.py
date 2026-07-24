"""Deterministic curation of the worthy-spot pool for a postflop batch.

Two knobs the admin (and the CLI) expose per solve:

* **hero filter** -- which player's decisions to ask about (BTN / BB / both).
  A solve carries both sides; you usually want to pick one at a time.
* **diversify** -- round-robin the pool across the four flop decision types so a
  fill-to-N batch isn't dominated by one archetype (e.g. all "BTN c-bets").

Both are pure + sorted, so byte-identical output is preserved. Moved here from
``scripts/generate_postflop.py`` so the CLI and the admin (its subprocess runner)
share one implementation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

_RANKS = "23456789TJQKA"
_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")

# Curation axes the admin/CLI expose (the postflop analog of preflop's
# hand-strength + action-faced filters). Both are applied BEFORE the expensive
# equity sim, so a filtered spot costs nothing.
STRENGTH_BUCKETS: tuple[str, ...] = (
    "premium", "strong", "medium", "vulnerable", "marginal", "air",
)
DECISION_TYPES: tuple[str, ...] = (
    "C-bet / barrel spot", "Lead / probe spot", "Facing a bet", "Check-back spot",
)


def spot_strength_bucket(spot: Any) -> str:
    """Hero's made-hand strength bucket on the board (premium … air).

    The postflop analog of preflop's hand-strength filter. Uses the shared pure
    classifier (no runouts), so it is cheap to run as a pre-equity filter."""
    from pipeline.fact_extractor.hand_class import classify_hand  # noqa: PLC0415

    return classify_hand(spot.hero_cards, list(spot.node.board))["strength_bucket"]


# 🔥 "Exciting pots" (July 23 2026, user ask): hero holds a BIG hand and the
# pot has genuinely heated up. Strength = the top two made-hand buckets; action
# = a raise anywhere in the postflop line, or two-plus bets/raises (facing a
# second barrel counts; a routine single c-bet does not).
EXCITING_STRENGTH_BUCKETS: tuple[str, ...] = ("premium", "strong")
_EXCITING_MIN_AGGRESSIVE_STEPS = 2


def spot_is_exciting(spot: Any) -> bool:
    """Big hand + big action, both required.

    Pure and cheap (the shared classifier, no equity sim), so it can gate a
    pool before any compute. Used by the standalone selector AND the
    full-hand ender test (:func:`pipeline.postflop.hand_quality.
    is_exciting_hand`)."""
    if spot_strength_bucket(spot) not in EXCITING_STRENGTH_BUCKETS:
        return False
    steps = [
        s for s in spot.node.history if s.street in ("flop", "turn", "river")
    ]
    if any(s.verb == "raise" for s in steps):
        return True
    return sum(
        1 for s in steps if s.verb in ("bet", "raise")
    ) >= _EXCITING_MIN_AGGRESSIVE_STEPS


def spot_decision_type(spot: Any, *, aggressor: str, ip_position: str) -> str:
    """The decision SITUATION hero is in -- the postflop analog of preflop's
    "action faced" filter. Situation-based (never reads hero's chosen action),
    so filtering on it can't leak the answer."""
    node = spot.node
    if node.is_facing_bet:
        return "Facing a bet"
    if aggressor and node.actor == aggressor:
        return "C-bet / barrel spot"  # the preflop raiser, first to act
    if ip_position and node.actor != ip_position:
        return "Lead / probe spot"  # OOP non-aggressor leading
    return "Check-back spot"  # IP non-aggressor, acting after a check


def combo_class(combo: str) -> str:
    """``'AsKs'`` -> ``'AKs'``, ``'3s3c'`` -> ``'33'`` (suit-isomorphic 169 class)."""
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if _RANKS.index(r1) > _RANKS.index(r2) else (r2, r1)
    return hi + lo + ("s" if s1 == s2 else "o")


def node_kind(node_id: str) -> str:
    """Classify a flop node into a decision type for diversity grouping."""
    t = node_id.split(":")[2:]
    if not t:
        return "bb_lead"
    if t == ["c"]:
        return "btn_cbet"
    if len(t) == 1 and t[0].startswith("b"):
        return "btn_faces_donk"
    if len(t) == 2 and t[0] == "c" and t[1].startswith("b"):
        return "bb_faces_cbet"
    return "deeper"


# The four first-line FLOP decision types, in the round-robin order a batch
# fills. Turn/river buckets ("turn:bets", "river:faces_bet", ...) are appended
# after these in first-seen order.
DECISION_KINDS: tuple[str, ...] = ("btn_cbet", "bb_faces_cbet", "btn_faces_donk", "bb_lead")


def _max_street_raises(node_id: str) -> int:
    """The most bet/raise tokens on any single street of the line.

    Splits the node string at chance cards into per-street segments and counts
    ``b`` tokens in each. 1 = a lone bet, 2 = bet + raise, >=3 = a re-raise war
    (poor first question; this is where degenerate, never-reached lines live).
    Street-aware so a normal "barrel every street" line (one bet per street)
    isn't mistaken for a deep raise war."""
    segments: list[list[str]] = [[]]
    for token in node_id.split(":")[2:]:
        if _CARD_RE.match(token):
            segments.append([])
        else:
            segments[-1].append(token)
    return max((sum(t.startswith("b") for t in seg) for seg in segments), default=0)


def _decision_family(spot: Any) -> str:
    """Coarse decision family for turn/river bucketing: facing a bet, betting
    out, or checking."""
    if spot.node.is_facing_bet:
        return "faces_bet"
    if spot.dominant_verb in ("bet", "raise"):
        return "bets"
    return "checks"


def _spot_bucket(spot: Any, *, max_depth: int) -> str | None:
    """The diversity bucket for a spot, or ``None`` to drop it.

    Flop keeps the original four-kind classification (and its depth cap), so the
    flop path is unchanged. Turn/river bucket by ``"<street>:<family>"`` and use
    a street-aware raise-war filter instead of the flop colon-depth cap (which
    would drop every legitimately-deeper turn/river line)."""
    nid = spot.node.node_id
    if any(getattr(s, "all_in", False) for s in getattr(spot.node, "history", ()) or ()):
        # The line already contains an all-in: a poor first question on any
        # street. File-agnostic (July 2026): the adapter walk stamps ``all_in``
        # exactly under the cumulative bet-token semantics -- replaces the old
        # hardcoded v8-only "b9697" node-id match, which never fired on the
        # 200bb files.
        return None
    if spot.node.street == "flop":
        if (nid.count(":") - 1) > max_depth:
            return None
        kind = node_kind(nid)
        return None if kind == "deeper" else kind
    if _max_street_raises(nid) > 2:  # noqa: PLR2004 -- re-raise war
        return None
    return f"{spot.node.street}:{_decision_family(spot)}"


def diversify_spots(
    worthy: Sequence[Any], *, per_class: int = 1, max_depth: int = 2
) -> list[Any]:
    """Curate ``worthy`` for a varied fill-to-N batch across streets.

    Drops all-in / re-raise-war lines (poor first questions), caps each
    ``(node, 169-class)`` to ``per_class`` combos (kills near-duplicate suits),
    then round-robins across decision buckets so the batch spreads over streets
    AND decision types. Flop buckets fill first (in :data:`DECISION_KINDS`
    order), then turn/river buckets in first-seen order. Pure + sorted, so the
    output is byte-stable."""
    seen: dict[tuple[Any, ...], int] = defaultdict(int)
    buckets: dict[str, list[Any]] = defaultdict(list)
    extra_order: list[str] = []  # non-flop buckets, in first-seen order
    for spot in sorted(worthy, key=lambda s: (s.node.node_id, s.hero_combo)):
        kind = _spot_bucket(spot, max_depth=max_depth)
        if kind is None:
            continue
        key = (kind, spot.node.node_id, combo_class(spot.hero_combo))
        if seen[key] >= per_class:
            continue
        seen[key] += 1
        if kind not in buckets and kind not in DECISION_KINDS:
            extra_order.append(kind)
        buckets[kind].append(spot)

    ordered_kinds = [k for k in DECISION_KINDS if k in buckets] + extra_order
    out: list[Any] = []
    i = 0
    while any(i < len(buckets[k]) for k in ordered_kinds):
        for k in ordered_kinds:
            if i < len(buckets[k]):
                out.append(buckets[k][i])
        i += 1
    return out


def make_spot_selector(
    *,
    heroes: Sequence[str] | None = None,
    diversify: bool = False,
    strength_buckets: Sequence[str] | None = None,
    decision_types: Sequence[str] | None = None,
    aggressor: str = "",
    ip_position: str = "",
    exciting: bool = False,
) -> Callable[[Sequence[Any]], list[Any]]:
    """A ``spot_selector`` for ``generate_postflop_batch``.

    Filters the worthy pool, in order, by:

    * ``heroes`` -- acting positions to keep (``None`` / empty = both players);
    * ``strength_buckets`` -- hero's made-hand bucket (premium … air; the
      postflop analog of preflop's hand-strength filter; ``None`` = all);
    * ``decision_types`` -- the decision situation (the "action faced" analog;
      ``None`` = all; needs ``aggressor`` + ``ip_position`` to classify);

    then optionally :func:`diversify_spots`. All filters run BEFORE the equity
    sim, so a filtered spot costs no compute. Pure + sorted; built fresh per run,
    so it's safe inside a subprocess worker.
    """
    hero_set = {h for h in heroes} if heroes else None
    strength_set = {b for b in strength_buckets} if strength_buckets else None
    decision_set = {d for d in decision_types} if decision_types else None

    def _select(worthy: Sequence[Any]) -> list[Any]:
        pool: list[Any] = []
        for s in worthy:
            if hero_set is not None and s.node.actor not in hero_set:
                continue
            if strength_set is not None and spot_strength_bucket(s) not in strength_set:
                continue
            if decision_set is not None and spot_decision_type(
                s, aggressor=aggressor, ip_position=ip_position
            ) not in decision_set:
                continue
            if exciting and not spot_is_exciting(s):
                continue  # 🔥 toggle: big hand + big action only
            pool.append(s)
        if diversify:
            pool = diversify_spots(pool)
        return list(pool)

    return _select


__all__ = [
    "DECISION_KINDS",
    "DECISION_TYPES",
    "STRENGTH_BUCKETS",
    "combo_class",
    "diversify_spots",
    "make_spot_selector",
    "node_kind",
    "spot_decision_type",
    "spot_strength_bucket",
]
