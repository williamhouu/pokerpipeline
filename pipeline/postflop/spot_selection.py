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

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

_RANKS = "23456789TJQKA"


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


# The four first-line decision types, in the round-robin order a batch fills.
DECISION_KINDS: tuple[str, ...] = ("btn_cbet", "bb_faces_cbet", "btn_faces_donk", "bb_lead")


def diversify_spots(
    worthy: Sequence[Any], *, per_class: int = 1, max_depth: int = 2
) -> list[Any]:
    """Curate ``worthy`` for a varied fill-to-N batch.

    Drops all-in / deep multi-raise lines (poor first questions), caps each
    ``(node, 169-class)`` to ``per_class`` combos (kills near-duplicate suits),
    then round-robins across the four decision types. Pure + sorted.
    """
    seen: dict[tuple[Any, ...], int] = defaultdict(int)
    buckets: dict[str, list[Any]] = defaultdict(list)
    for spot in sorted(worthy, key=lambda s: (s.node.node_id, s.hero_combo)):
        nid = spot.node.node_id
        if "b9697" in nid or (nid.count(":") - 1) > max_depth:
            continue
        kind = node_kind(nid)
        if kind == "deeper":
            continue
        key = (kind, nid, combo_class(spot.hero_combo))
        if seen[key] >= per_class:
            continue
        seen[key] += 1
        buckets[kind].append(spot)

    out: list[Any] = []
    i = 0
    while any(i < len(buckets[k]) for k in DECISION_KINDS):
        for k in DECISION_KINDS:
            if i < len(buckets[k]):
                out.append(buckets[k][i])
        i += 1
    return out


def make_spot_selector(
    *, heroes: Sequence[str] | None = None, diversify: bool = False
) -> Callable[[Sequence[Any]], list[Any]]:
    """A ``spot_selector`` for ``generate_postflop_batch``.

    Filters the worthy pool to ``heroes`` (the acting positions to keep; ``None``
    / empty = both players), then optionally :func:`diversify_spots`. Built fresh
    per run, so it's safe to construct inside a subprocess worker.
    """
    hero_set = {h for h in heroes} if heroes else None

    def _select(worthy: Sequence[Any]) -> list[Any]:
        pool = [s for s in worthy if hero_set is None or s.node.actor in hero_set]
        if diversify:
            pool = diversify_spots(pool)
        return list(pool)

    return _select


__all__ = [
    "DECISION_KINDS",
    "combo_class",
    "diversify_spots",
    "make_spot_selector",
    "node_kind",
]
