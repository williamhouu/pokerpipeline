"""PLO preflop chart export -- the app team's filterable hand-list JSON.

The PLO equivalent of :mod:`pipeline.preflop.chart_export`, but a DIFFERENT
app surface: NLHE ships a 13x13 grid per node; PLO has 16,432 suit-isomorphic
hand classes, so the app UI is a filterable hand LIST plus a "check any hand"
lookup. Per pack the export is a directory::

    <pack_id>/
        index.json            small: pack header + the full node tree
        nodes/<safe_key>.json.gz   one lazily-loadable hand list per node

Tree-link contract (same pathKey scheme as the shipped NLH SolverPack
exports): a node's map key is its dot-joined history action tokens; the root
(first decision) is the empty string ``""``; a child's key is
``parentKey + "." + act.c``. For this pack family that falls straight out of
the range-file naming (a node's key == its ``history_stem``).

Sizes are NEVER computed here: every ``pot_bb`` / ``to_call_bb`` / ``to_bb``
comes from the shared pot-limit resolution walk
(:func:`pipeline.plo.action_history.resolve_pot_limit` /
:func:`~pipeline.plo.action_history.call_price`), so the chart can never
disagree with question prose built from the same pack. The MTT packs' 1bb BB
ante is already inside that walk's pot (and is never re-added here).

Hand-entry classification reuses :mod:`pipeline.plo.hand_model`
(:func:`classify_plo_hand`) so chart tags can never contradict question
prose; the display bucket taxonomy (AAxx .. Other) is defined here (it is a
chart-only display grouping, not a strategy fact).

WHY ``.json.gz`` node files: measured on the real packs, the 13-pack tree is
~200M hand entries (~25 GB of plain JSON) -- far over the disk budget --
while the entry content is extremely repetitive and gzips ~10-15x. The
per-node files are therefore gzip-compressed JSON (deterministic: mtime=0).
``index.json`` stays plain. For R2, either upload the ``.gz`` bytes with
``Content-Encoding: gzip`` (browsers decompress transparently) or decompress
on upload; see the export README.

Pure functions + local file I/O only (no Streamlit), so everything is
browserless-testable (fix-durability rule). The thin CLI lives in
``scripts/export_plo_chart_packs.py``.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# SEAM NOTE: _walk is action_history's private single-source pot-limit walk
# (see its INVARIANT comment). Importing it here (an in-repo consumer, like
# the NLHE chart export importing format_writer._normalize_action) keeps the
# chart's pot / to-call / high-bet single-sourced with the Question prose and
# the SOLVER DATA price block -- they can never disagree. If action_history
# ever renames it, tests/test_plo_chart_export.py fails first.
from pipeline.plo.action_history import _walk, display_seat, resolve_pot_limit
from pipeline.plo.hand_model import (
    _STRAIGHT_REACH,
    PloHandClass,
    classify_plo_hand,
)
from pipeline.plo.hand_order import (
    HAND_COUNT,
    cards_at,
    combo_multiplicities,
    hand_order,
)
from pipeline.plo.node_enumerator import (
    PloDecisionNode,
    action_label,
    enumerate_plo_nodes,
)
from pipeline.plo.options import integer_percentages
from pipeline.plo.pack import (
    KNOWN_PLO_PACKS,
    PLO_PACK_6MAX_100BB,
    PLO_PACK_9MAX_100BB,
    PloActionType,
    PloPack,
    PloPackSpec,
    parse_node_path,
    read_rng_values,
)

#: The 6-max wave-1 packs this exporter ships (informational default for the
#: CLI; any registered 6-max pack with an extracted folder exports fine, so
#: the deferred 40/150/200bb cash depths run later with the same command).
PLO_CHART_EXPORT_PACK_IDS: tuple[str, ...] = (
    "plo_6max_12bb",
    "plo_6max_20bb",
    "plo_6max_30bb",
    "plo_6max_50bb",
    "plo_6max_75bb",
    "plo_6max_100bb",
    "plo_mtt_6max_10bb",
    "plo_mtt_6max_15bb",
    "plo_mtt_6max_20bb",
    "plo_mtt_6max_25bb",
    "plo_mtt_6max_30bb",
    "plo_mtt_6max_40bb",
    "plo_mtt_6max_75bb",
)

#: Display buckets, in PRECEDENCE order (first match wins). User-supplied
#: taxonomy -- the app's hand-list filter chips.
CHART_BUCKETS: tuple[str, ...] = (
    "AAxx",
    "KKxx",
    "QQ-TT",
    "Two pair",
    "Low pair",
    "Rundown",
    "Broadway",
    "Dangler",
    "Other",
)

#: 3-way suit partition for the hand list (coarser than hand_model's exact
#: 5-way suit_pattern, which ships alongside in the structural facts).
CHART_SUITS: tuple[str, ...] = ("double-suited", "single-suited", "rainbow")

# acts[] display order: fold, call, raise(s) ascending by to_bb, all-in last
# (the aggression ladder; mirrors the NLHE SolverPack export contract).
_KIND_ORDER = {"fold": 0, "call": 1, "raise": 2, "allin": 3}

_KIND_BY_TYPE = {
    PloActionType.FOLD: "fold",
    PloActionType.CALL: "call",
    PloActionType.RAISE: "raise",
    PloActionType.MIN_RAISE: "raise",
    PloActionType.ALL_IN: "allin",
}

# Highest paired rank T..Q -> the QQ-TT bucket (A/K pairs are caught earlier
# by precedence; a single pair that falls through is 99 or below = Low pair).
_QQ_TT_MIN = 10

_SIZE_TOLERANCE_BB = 1e-3  # stored sizes are rounded to 4dp


# --- bucket + suits classification (display-only, hand_model-derived) -------
def _ace_layouts(distinct_high: tuple[int, ...]) -> list[list[int]]:
    """Ace-high and (when an ace is present) ace-low rank layouts.

    The same dual-layout convention as hand_model's connectedness: the ace
    plays high or low, and a shape claim holds if SOME placement shows it.
    """
    layouts = [list(distinct_high)]
    if 14 in distinct_high:
        layouts.append([1 if r == 14 else r for r in distinct_high])
    return layouts


def _is_dangler_shape(distinct_high: tuple[int, ...]) -> bool:
    """Exactly one disconnected card next to three coordinated ones.

    Documented rule (chart README): with the ace tried high AND low, the
    hand is a Dangler when some placement shows EXACTLY ONE card whose
    nearest other rank is more than :data:`_STRAIGHT_REACH` (4) away --
    hand_model's straight-reach constant, the "these two hole cards can
    share a 5-card straight" span -- while the other THREE ranks span at
    most 4 (so all three genuinely coordinate for straights). KQJ2, AKQ2
    and Q753 qualify; KQ72 (two dead cards) and QT86 (evenly spread, no
    card fully disconnected) do not. Unpaired hands only (every paired
    hand is caught by an earlier bucket).
    """
    if len(distinct_high) != 4:  # noqa: PLR2004 -- paired hands never reach here
        return False
    for layout in _ace_layouts(distinct_high):
        s = sorted(layout)
        far = [
            v
            for i, v in enumerate(s)
            if min(abs(v - o) for j, o in enumerate(s) if j != i) > _STRAIGHT_REACH
        ]
        if len(far) == 1:
            trio = [v for v in s if v != far[0]]
            if max(trio) - min(trio) <= _STRAIGHT_REACH:
                return True
    return False


def chart_bucket(cls: PloHandClass) -> str:
    """The display bucket for a classified hand (first match wins).

    Precedence and exact rules (mirrored in the export README):

    1. ``AAxx``     two or more aces (AAKK, AAAx included).
    2. ``KKxx``     two or more kings.
    3. ``QQ-TT``    highest paired rank is Q, J or T (QQJJ lands here).
    4. ``Two pair`` two distinct paired ranks (both 99 or below by now).
    5. ``Low pair`` exactly one paired rank, 99 or below (999x / 9999 too:
       trips and quads count as "paired" at their rank).
    6. ``Rundown``  unpaired, at most ONE total missing rank across the
       four cards (ace high or low, best placement) -- perfect rundowns
       (KQJT, JT98, A234) and one-gappers (J987) -- EXCEPT all-broadway
       one-gappers, which fall through to Broadway.
    7. ``Broadway`` all four cards T or higher, unpaired, not already a
       Rundown (AKQT, AKJT, AQJT; AKQJ and KQJT are Rundowns).
    8. ``Dangler``  three coordinated cards plus one disconnected card
       (see :func:`_is_dangler_shape`).
    9. ``Other``    everything else (two-gappers, double danglers, ...).
    """
    pr = cls.pair_ranks  # ranks appearing 2+ times, high -> low
    if 14 in pr:
        return "AAxx"
    if 13 in pr:
        return "KKxx"
    if pr and pr[0] >= _QQ_TT_MIN:
        return "QQ-TT"
    if len(pr) >= 2:  # noqa: PLR2004
        return "Two pair"
    if pr:
        return "Low pair"
    all_broadway = cls.broadway_count == 4  # noqa: PLR2004
    if cls.connectedness == "rundown" or (
        cls.connectedness == "one_gapper" and not all_broadway
    ):
        return "Rundown"
    if all_broadway:
        return "Broadway"
    if _is_dangler_shape(cls.distinct_ranks):
        return "Dangler"
    return "Other"


def chart_suits(cls: PloHandClass) -> str:
    """The 3-way suit partition (README rule, hand_model-consistent).

    ``double-suited`` = exactly (2,2) suit counts (two flush-capable
    suits); ``rainbow`` = all four suits distinct (no flush possible);
    ``single-suited`` = everything else, i.e. exactly ONE flush-capable
    suit: the classic (2,1,1) AND three-of-a-suit AND monotone (a flush
    uses exactly two hole cards, so 3 or 4 of one suit still make just
    one flush suit; hand_model's exact 5-way ``suit_pattern`` ships in
    the structural facts for apps that want the distinction). Equivalent
    to counting :func:`pipeline.plo.hand_model.flush_suits` entries
    (2 / 1 / 0) -- pinned by tests.
    """
    if cls.suit_pattern == "double_suited":
        return "double-suited"
    if cls.suit_pattern == "rainbow":
        return "rainbow"
    return "single-suited"


@lru_cache(maxsize=1)
def _static_entry_prefixes() -> tuple[str, ...]:
    """Per-class JSON prefix (everything but the mix), by hand-order index.

    A hand entry's static fields (class string, combos, bucket, suits,
    structural facts) depend only on the class, so they are serialised once
    here and reused for every node of every pack -- the export hot loop only
    appends the per-node mix object.
    """
    labels = hand_order()
    mults = combo_multiplicities()
    prefixes: list[str] = []
    for i in range(HAND_COUNT):
        cls = classify_plo_hand(cards_at(i))
        facts = (
            f'{{"pair":"{cls.pair_pattern}","suit":"{cls.suit_pattern}",'
            f'"conn":"{cls.connectedness}"}}'
        )
        prefixes.append(
            f'{{"h":{json.dumps(labels[i])},"n":{mults[i]},'
            f'"b":"{chart_bucket(cls)}","s":"{chart_suits(cls)}",'
            f'"f":{facts},"m":'
        )
    return tuple(prefixes)


# --- keys / filenames -------------------------------------------------------
def safe_node_key(key: str) -> str:
    """Filesystem-safe node filename stem: dots -> underscores; root -> 'root'.

    Reversible (action tokens never contain underscores) and collision-free
    (tokens are numeric, so no real key maps to 'root').
    """
    return key.replace(".", "_") if key else "root"


def _num(value: float) -> float | int:
    """Round to 4dp and collapse integral floats for compact, stable JSON."""
    r = round(value, 4)
    return int(r) if float(r).is_integer() else r


# --- per-node export --------------------------------------------------------
@dataclass(frozen=True)
class _ActInfo:
    c: str  # the action token (child path-key suffix)
    kind: str  # fold | call | raise | allin
    to_bb: float
    label: str  # house action label (mix/aggregate key), e.g. "Raise 100%"
    path: Path  # the backing range file


def _node_acts(
    node: PloDecisionNode, *, stack_bb: float, ante_bb: float
) -> tuple[list[_ActInfo], float, float]:
    """Ladder-ordered acts with resolved sizes + (pot_bb, to_call_bb).

    All sizes come from the ONE shared pot-limit walk. Call ``to_bb`` is the
    total the caller matches (the current high bet), mirroring the NLH
    SolverPack call-TO convention; fold is 0.
    """
    _actions, pot, committed, high_bet = _walk(
        node.history_before, stack_bb=stack_bb, ante_bb=ante_bb
    )
    to_call = max(0.0, high_bet - committed.get(node.actor, 0.0))
    acts: list[_ActInfo] = []
    for opt in node.actions:
        kind = _KIND_BY_TYPE[opt.action.action]
        if kind == "fold":
            to_bb = 0.0
        elif kind == "call":
            to_bb = high_bet
        else:
            resolved, _pot_after = resolve_pot_limit(
                (*node.history_before, opt.action),
                stack_bb=stack_bb,
                ante_bb=ante_bb,
            )
            to_bb = resolved[-1].to_bb or 0.0
        token = opt.stem.rsplit(".", 1)[-1]
        acts.append(_ActInfo(token, kind, to_bb, opt.label, opt.path))
    acts.sort(key=lambda a: (_KIND_ORDER[a.kind], a.to_bb, a.c))
    return acts, pot, to_call


@dataclass
class PackExportStats:
    """What one pack's export produced (the CLI's per-pack report line)."""

    pack_id: str
    nodes_written: int = 0
    nodes_pruned_zero_reach: int = 0
    hand_entries: int = 0
    bytes_written: int = 0


def _write_gz(path: Path, text: str) -> int:
    """Write deterministic gzip (mtime=0) -> compressed bytes written."""
    data = text.encode("utf-8")
    with path.open("wb") as fh:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=fh, compresslevel=6, mtime=0
        ) as gz:
            gz.write(data)
    return path.stat().st_size


def export_pack(
    pack: PloPack,
    out_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
    validate: bool = True,
) -> PackExportStats:
    """Export one pack to ``out_root/<pack_id>/`` and validate it.

    Nodes are processed shallow-first so zero-reach pruning cascades: a node
    none of whose hands can reach it (total strategy mass 0 across every
    action file) is dropped along with its whole subtree; its parent still
    lists the act (aggregate share 0), and the walk simply ends there --
    exactly the README's absent-child contract.
    """
    spec = pack.spec
    stats = PackExportStats(pack_id=spec.pack_id)
    pack_dir = out_root / spec.pack_id
    nodes_dir = pack_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    mults = np.array(combo_multiplicities(), dtype=np.float64)
    prefixes = _static_entry_prefixes()

    nodes = sorted(
        enumerate_plo_nodes(pack),
        key=lambda n: (len(n.history_before), n.history_stem),
    )
    all_keys = {n.history_stem for n in nodes}
    live: set[str] = set()
    index_nodes: dict[str, dict[str, Any]] = {}

    for done, node in enumerate(nodes, start=1):
        key = node.history_stem
        if key:
            parent_key = key.rsplit(".", 1)[0] if "." in key else ""
            if parent_key not in live:
                if parent_key not in all_keys:
                    # A child exists but its parent decision has no range
                    # files at all: a structural pack gap, not a pruned
                    # zero-frequency branch. Fail loudly.
                    raise AssertionError(
                        f"{spec.pack_id}: node {key!r} has no parent node "
                        f"{parent_key!r} in the pack"
                    )
                stats.nodes_pruned_zero_reach += 1
                continue
        acts, pot_bb, to_call_bb = _node_acts(
            node, stack_bb=spec.stack_bb, ante_bb=spec.ante_bb
        )
        labels = [a.label for a in acts]
        if len(set(labels)) != len(labels):  # one file per distinct action
            raise AssertionError(
                f"{spec.pack_id} {key!r}: duplicate act labels {labels}"
            )

        plists = [
            np.fromiter(
                (pe[0] for pe in read_rng_values(a.path)),
                dtype=np.float64,
                count=HAND_COUNT,
            )
            for a in acts
        ]
        reach = plists[0].copy()
        for arr in plists[1:]:
            reach += arr
        live_idx = np.flatnonzero(reach > 0.0)
        if live_idx.size == 0:
            stats.nodes_pruned_zero_reach += 1
            continue

        # aggregate: combo-weighted share of the acting player's reaching
        # range, per act, via the house largest-remainder allocation.
        masses = [float(np.dot(arr, mults)) for arr in plists]
        total_mass = sum(masses)
        aggregate = integer_percentages(
            {labels[j]: masses[j] / total_mass for j in range(len(acts))}
        )
        if sum(aggregate.values()) != 100:  # noqa: PLR2004
            raise AssertionError(
                f"{spec.pack_id} {key!r}: aggregate sums to {sum(aggregate.values())}"
            )

        # hand list: ONLY classes actually in range here (weight > 0 in at
        # least one action file), mix conditional on reach.
        pure_mix = [f"{{{json.dumps(lbl)}:100}}" for lbl in labels]
        n_acts = len(acts)
        parts: list[str] = []
        cols = [arr.tolist() for arr in plists]  # python floats: fast loop
        for i in live_idx.tolist():
            ps = [cols[j][i] for j in range(n_acts)]
            taken = [j for j, p in enumerate(ps) if p > 0.0]
            if len(taken) == 1:
                mix_str = pure_mix[taken[0]]
            else:
                tot = sum(ps)
                alloc = integer_percentages({labels[j]: ps[j] / tot for j in taken})
                mix_str = json.dumps(alloc, separators=(",", ":"))
            parts.append(prefixes[i] + mix_str + "}")
        stats.hand_entries += len(parts)

        hands_rel = f"nodes/{safe_node_key(key)}.json.gz"
        body = (
            f'{{"node":{json.dumps(key)},"pack":{json.dumps(spec.pack_id)},'
            f'"acts":{json.dumps(labels, separators=(",", ":"))},'
            f'"hands":[{",".join(parts)}]}}'
        )
        stats.bytes_written += _write_gz(pack_dir / hands_rel, body)

        index_nodes[key] = {
            "pos": display_seat(node.actor, table_size=node.table_size),
            "bl": sum(
                1
                for a in node.history_before
                if a.action
                in (
                    PloActionType.RAISE,
                    PloActionType.MIN_RAISE,
                    PloActionType.ALL_IN,
                )
            ),
            "pot_bb": _num(pot_bb),
            "to_call_bb": _num(to_call_bb),
            "acts": [{"c": a.c, "k": a.kind, "to_bb": _num(a.to_bb)} for a in acts],
            "aggregate": aggregate,
            "hands_file": hands_rel,
        }
        live.add(key)
        stats.nodes_written += 1
        if progress is not None and done % 500 == 0:
            progress(
                f"{spec.pack_id}: {done}/{len(nodes)} nodes scanned, "
                f"{stats.nodes_written} written"
            )

    if "" not in index_nodes:
        raise AssertionError(f"{spec.pack_id}: no root node exported")

    header: dict[str, Any] = {
        "pack_id": spec.pack_id,
        "game": "PLO",
        "format": "MTT" if spec.game_format == "tournament" else "Cash",
        "table": "6-Max",
        "stack_bb": _num(spec.stack_bb),
        "rake": spec.rake_note,
        "seats": [display_seat(s, table_size=spec.table_size) for s in spec.seats],
        "hand_class_count": HAND_COUNT,
        "node_count": len(index_nodes),
        "format_version": 1,
    }
    if spec.ante_bb:
        # The 1bb BB ante is already inside every pot_bb/to_bb (the shared
        # walk posts it before any action); this field is informational.
        header["ante_bb"] = _num(spec.ante_bb)
    header["nodes"] = index_nodes

    index_path = pack_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(header, fh, separators=(",", ":"), ensure_ascii=True)
    stats.bytes_written += index_path.stat().st_size

    if validate:
        validate_pack_export(pack_dir, spec)
    return stats


# --- validation (fail loudly) ----------------------------------------------
def _expected_sizes(
    key: str, spec: PloPackSpec, acts: list[dict[str, Any]]
) -> tuple[float, float, list[float]]:
    """Recompute (pot, to_call, per-act to_bb) from the path key alone.

    Decodes the key's tokens through :func:`parse_node_path` + the shared
    walk -- an independent path from the exporter's node objects (which came
    from file grouping), so a mis-keyed node or wrong size fails here.
    """
    history = parse_node_path(key, seats=spec.seats) if key else ()
    _a, pot, committed, high_bet = _walk(
        history, stack_bb=spec.stack_bb, ante_bb=spec.ante_bb
    )
    seats_queue = list(spec.seats)
    for a in history:
        seats_queue.pop(0)
        if a.action in (
            PloActionType.CALL,
            PloActionType.RAISE,
            PloActionType.MIN_RAISE,
        ):
            seats_queue.append(a.seat)
    actor = seats_queue[0]
    to_call = max(0.0, high_bet - committed.get(actor, 0.0))
    to_bbs: list[float] = []
    for act in acts:
        stem = f"{key}.{act['c']}" if key else act["c"]
        full = parse_node_path(stem, seats=spec.seats)
        kind = _KIND_BY_TYPE[full[-1].action]
        if kind == "fold":
            to_bbs.append(0.0)
        elif kind == "call":
            to_bbs.append(high_bet)
        else:
            resolved, _p = resolve_pot_limit(
                full, stack_bb=spec.stack_bb, ante_bb=spec.ante_bb
            )
            to_bbs.append(resolved[-1].to_bb or 0.0)
    return pot, to_call, to_bbs


def _validate_hands_file(
    path: Path,
    key: str,
    act_labels: list[str],
    *,
    parity_sample: int,
    label_index: dict[str, int],
) -> None:
    """Round-trip one node's hand list; spot-check bucket/suits parity."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc["node"] != key:
        raise AssertionError(f"{path.name}: node {doc['node']!r} != {key!r}")
    if doc["acts"] != act_labels:
        raise AssertionError(f"{path.name}: acts mismatch vs index")
    hands = doc["hands"]
    if not hands:
        raise AssertionError(f"{path.name}: empty hand list")
    allowed = set(act_labels)
    step = max(1, len(hands) // max(parity_sample, 1))
    for pos, entry in enumerate(hands):
        mix = entry["m"]
        if sum(mix.values()) != 100:  # noqa: PLR2004
            raise AssertionError(
                f"{path.name}: hand {entry['h']} mix sums to {sum(mix.values())}"
            )
        if not set(mix) <= allowed:
            raise AssertionError(
                f"{path.name}: hand {entry['h']} mix keys {set(mix)} not in acts"
            )
        if pos % step == 0:
            idx = label_index.get(entry["h"])
            if idx is None:
                raise AssertionError(f"{path.name}: unknown class {entry['h']!r}")
            cls = classify_plo_hand(cards_at(idx))
            if entry["b"] != chart_bucket(cls) or entry["s"] != chart_suits(cls):
                raise AssertionError(
                    f"{path.name}: {entry['h']} bucket/suits drift "
                    f"({entry['b']}/{entry['s']})"
                )


def validate_pack_export(
    pack_dir: Path,
    spec: PloPackSpec,
    *,
    hands_file_sample: int = 200,
    parity_sample: int = 25,
) -> None:
    """Fail-loud validation of one exported pack directory.

    Checks, per the export contract: tree connectivity (every non-root key =
    parent + "." + a parent act token, parent present); aggregate sums of
    exactly 100; every act/pot/to-call size re-derived from the path key
    through the shared pot-limit walk; every node's hands_file exists; and a
    deterministic sample of hands files round-trips (mix sums 100, keys
    subset of acts, bucket/suits parity vs hand_model on sampled entries).
    """
    index = json.loads((pack_dir / "index.json").read_text(encoding="utf-8"))
    nodes: dict[str, dict[str, Any]] = index["nodes"]
    if "" not in nodes:
        raise AssertionError("root node missing from index")
    label_index = {lbl: i for i, lbl in enumerate(hand_order())}

    keys = sorted(nodes, key=lambda k: (len(k.split(".")), k))
    sample_step = max(1, len(keys) // max(hands_file_sample, 1))
    for pos, key in enumerate(keys):
        node = nodes[key]
        acts = node["acts"]
        if key:
            parent_key, _, c = key.rpartition(".")
            parent = nodes.get(parent_key)
            if parent is None:
                raise AssertionError(f"{key!r}: parent {parent_key!r} missing")
            if c not in {a["c"] for a in parent["acts"]}:
                raise AssertionError(
                    f"{key!r}: reached by {c!r}, not an act of its parent"
                )
        if sum(node["aggregate"].values()) != 100:  # noqa: PLR2004
            raise AssertionError(f"{key!r}: aggregate does not sum to 100")
        kinds = [_KIND_ORDER[a["k"]] for a in acts]
        if kinds != sorted(kinds):
            raise AssertionError(f"{key!r}: acts out of ladder order")
        pot, to_call, to_bbs = _expected_sizes(key, spec, acts)
        if abs(node["pot_bb"] - pot) > _SIZE_TOLERANCE_BB:
            raise AssertionError(f"{key!r}: pot_bb {node['pot_bb']} != walk {pot}")
        if abs(node["to_call_bb"] - to_call) > _SIZE_TOLERANCE_BB:
            raise AssertionError(
                f"{key!r}: to_call_bb {node['to_call_bb']} != walk {to_call}"
            )
        for act, expected in zip(acts, to_bbs, strict=True):
            if abs(act["to_bb"] - expected) > _SIZE_TOLERANCE_BB:
                raise AssertionError(
                    f"{key!r}: act {act['c']} to_bb {act['to_bb']} != walk {expected}"
                )
        hands_path = pack_dir / node["hands_file"]
        if not hands_path.is_file():
            raise AssertionError(f"{key!r}: hands file missing {hands_path}")
        if pos % sample_step == 0 or len(key.split(".")) <= 1:
            # Re-derive the house act labels from the tokens themselves
            # (through the same action_label), independent of the exporter's
            # in-memory objects.
            act_labels = [
                action_label(
                    parse_node_path(
                        f"{key}.{a['c']}" if key else a["c"],
                        seats=spec.seats,
                    )[-1]
                )
                for a in acts
            ]
            _validate_hands_file(
                hands_path,
                key,
                act_labels,
                parity_sample=parity_sample,
                label_index=label_index,
            )


def spec_for_pack_id(pack_id: str) -> PloPackSpec:
    """The registered spec for a pack id (fail loudly on unknown ids)."""
    for spec in KNOWN_PLO_PACKS:
        if spec.pack_id == pack_id:
            return spec
    raise KeyError(f"unknown PLO pack id: {pack_id!r}")


# ===========================================================================
# Developer per-class chart (plo-charts-data.json)
# ===========================================================================
# The app developer's Charts-tab data contract (plo-charts.md sections 6-8):
# ONE JSON file, skeleton
#   {"version":1,"charts":{"<packId>":{"<depth>":{"<heroSeat>":{"<nodeKey>":
#       {"<handKey>":[fold,call,raise]}}}}}}
# derived from the SAME pack walk + mixes as the per-hand directory export
# above. The classifier here is HIS (doc section 6) -- deliberately separate
# from chart_bucket above (different precedence: AAKK is Two pair for him,
# AAxx for the per-hand list; AAAK is Trips for him, AAxx above).

#: packId in the developer contract: cash 6-max = "c6", cash 9-max = "c9",
#: MTT 6-max = "t6".
DEV_PACK_ID_CASH6 = "c6"
DEV_PACK_ID_CASH9 = "c9"
DEV_PACK_ID_MTT6 = "t6"

# The developer's c9 seat list (plo-charts.md section 4) is
#   UTG, UTG+1, MP, LJ, HJ, CO, BTN, SB, BB
# while our 9-max pack's seats are
#   UTG, UTG+1, UTG+2, LJ, HJ, CO, BTN, SB, BB
# -- the SAME seat, third to act; only the display name differs.
_DEV_SEAT_RENAME_9MAX = {"UTG+2": "MP"}


def dev_seat_name(seat: str, *, table_size: int) -> str:
    """A seat's name in the developer chart contract.

    INVARIANT: EVERY dev-chart seat surface -- heroSeat map keys, nodeKey
    ``seat:code`` tokens, and the replay validator's expected queue -- must
    route through this ONE function, so the 9-max UTG+2 -> MP rename can
    never half-apply (a heroSeat saying MP while its nodeKeys say UTG+2
    would break the developer's engine walk). 6-max names are the shared
    ``display_seat`` output, byte-identical to the pre-c9 exports.
    """
    name = display_seat(seat, table_size=table_size)
    if table_size == 9:  # noqa: PLR2004
        return _DEV_SEAT_RENAME_9MAX.get(name, name)
    return name

_DEV_SUIT_SUFFIX = {
    "double-suited": "ds",
    "single-suited": "ss",
    "rainbow": "rbw",
}

# nodeKey codes (doc section 7.1): raise ladder r -> 3 -> 4 -> 5, cap 4
# raises (a history containing a 5th raise cannot be encoded; the node is
# omitted -- his engine's raise cap never walks there).
_DEV_RAISE_CODES = {1: "r", 2: "3", 3: "4", 4: "5"}
_DEV_MAX_RAISES = 4


def dev_hand_class(cls: PloHandClass) -> str:
    """The developer's hand classifier (plo-charts.md section 6).

    First match wins: Trips (3-4 of one rank; AAAK and AAAA land here) ->
    Two pair (AAKK lands here, unlike the per-hand bucket taxonomy) ->
    single pair split AAxx / KKxx / QQ-TT / Low pair -> Rundown (4 distinct
    ranks, span <= 4, ace always high) -> Broadway (4 distinct, all T+;
    structurally unreachable: every such hand has span <= 4 and is already
    a Rundown -- kept to mirror his spec byte-for-byte) -> Dangler (4
    distinct, gap of 5+ between the 3rd and 4th ranks sorted descending)
    -> Other. NOTE: no ace-low reading anywhere (A234 is Other), per the
    literal spec.
    """
    if cls.pair_pattern in ("trips", "quads"):
        return "Trips"
    if cls.pair_pattern == "two_pair":
        return "Two pair"
    if cls.pair_pattern == "one_pair":
        top = cls.pair_ranks[0]
        if top == 14:  # noqa: PLR2004
            return "AAxx"
        if top == 13:  # noqa: PLR2004
            return "KKxx"
        if top >= _QQ_TT_MIN:
            return "QQ-TT"
        return "Low pair"
    d = cls.distinct_ranks  # 4 distinct, high -> low
    if d[0] - d[3] <= 4:  # noqa: PLR2004
        return "Rundown"
    if d[3] >= 10:  # noqa: PLR2004  # pragma: no cover -- see docstring
        return "Broadway"
    if d[2] - d[3] >= 5:  # noqa: PLR2004
        return "Dangler"
    return "Other"


def dev_suit_suffix(cls: PloHandClass) -> str:
    """``ds`` / ``ss`` / ``rbw`` -- his suffixes, same partition as
    :func:`chart_suits` (two suits with 2+ each / exactly one suit with 2+ /
    all four suits)."""
    return _DEV_SUIT_SUFFIX[chart_suits(cls)]


@lru_cache(maxsize=1)
def _dev_class_ids() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Per hand-order index: suffixed + bare class group ids (for bincount).

    Returns ``(suffixed_labels, suffixed_id_per_hand, bare_labels,
    bare_id_per_hand)``.
    """
    suffixed_labels: list[str] = []
    bare_labels: list[str] = []
    suffixed_of: dict[str, int] = {}
    bare_of: dict[str, int] = {}
    suff_ids = np.empty(HAND_COUNT, dtype=np.int64)
    bare_ids = np.empty(HAND_COUNT, dtype=np.int64)
    for i in range(HAND_COUNT):
        cls = classify_plo_hand(cards_at(i))
        bare = dev_hand_class(cls)
        suffixed = f"{bare}|{dev_suit_suffix(cls)}"
        if suffixed not in suffixed_of:
            suffixed_of[suffixed] = len(suffixed_labels)
            suffixed_labels.append(suffixed)
        if bare not in bare_of:
            bare_of[bare] = len(bare_labels)
            bare_labels.append(bare)
        suff_ids[i] = suffixed_of[suffixed]
        bare_ids[i] = bare_of[bare]
    return suffixed_labels, suff_ids, bare_labels, bare_ids


def dev_node_key(history: tuple[Any, ...], *, table_size: int = 6) -> str | None:
    """Translate a pack action history into the developer's nodeKey.

    ``seat:code`` tokens joined by ``.``, acting order; root = ``""``.
    Codes: ``f`` fold; ``l`` limp (call before any raise; the unopened BB's
    call is ``x`` check); ``c`` call after a raise; ``r``/``3``/``4``/``5``
    the raise ladder (an all-in counts at its ladder position -- his leaf
    has no all-in slot, doc section 7.3). Returns None when the history
    contains a 5th raise (his raise cap is 4; such nodes are omitted).
    Seat names go through :func:`dev_seat_name` (the 9-max UTG+2 -> MP
    rename; 6-max unchanged).
    """
    raise_count = 0
    parts: list[str] = []
    for a in history:
        seat = dev_seat_name(a.seat, table_size=table_size)
        if a.action is PloActionType.FOLD:
            code = "f"
        elif a.action is PloActionType.CALL:
            if raise_count == 0:
                code = "x" if a.seat == "BB" else "l"
            else:
                code = "c"
        else:  # RAISE / MIN_RAISE / ALL_IN
            raise_count += 1
            if raise_count > _DEV_MAX_RAISES:
                return None
            code = _DEV_RAISE_CODES[raise_count]
        parts.append(f"{seat}:{code}")
    return ".".join(parts)


def replay_dev_node_key(key: str, *, seats: tuple[str, ...]) -> str:
    """Replay a nodeKey through an independent seat-order walk.

    Returns the seat (display name) next to act -- the node's heroSeat --
    and raises if any token's seat does not match the acting order (folded
    seats leave the queue; every other action re-queues its seat). The
    validation half of the nodeKey round-trip. The queue is built through
    :func:`dev_seat_name`, so a c9 key's ``MP:`` tokens replay against MP
    (the mapping is validated, not assumed).
    """
    queue = [dev_seat_name(s, table_size=len(seats)) for s in seats]
    if key:
        for token in key.split("."):
            seat, _, code = token.partition(":")
            expected = queue.pop(0)
            if seat != expected:
                raise AssertionError(
                    f"nodeKey {key!r}: token {token!r} out of order "
                    f"(expected {expected})"
                )
            if code != "f":
                queue.append(seat)
    if not queue:
        raise AssertionError(f"nodeKey {key!r}: no seat left to act")
    return queue[0]


def _dev_triple(fold: float, call: float, raise_: float) -> list[int]:
    """``[fold, call, raise]`` ints summing to exactly 100 (house
    largest-remainder allocation)."""
    total = fold + call + raise_
    alloc = integer_percentages(
        {"fold": fold / total, "call": call / total, "raise": raise_ / total}
    )
    triple = [alloc["fold"], alloc["call"], alloc["raise"]]
    if sum(triple) != 100 or any(v < 0 for v in triple):  # noqa: PLR2004
        raise AssertionError(f"bad class triple {triple}")
    return triple


def clean_line_node(node: PloDecisionNode) -> bool:
    """The pack family's clean-line convention, as a walk-connected filter.

    True when the action line BEFORE the decision holds at most 2 raises AND
    at most 3 seats that voluntarily entered the pot -- the same caps as
    ``generate_plo_batch``'s defaults (``max_prior_raises=2`` /
    ``max_active_players=3``), with ONE deliberate difference: the entrant
    count here reads the HISTORY only, not history-plus-hero
    (:func:`plo_pot_entrant_count` adds the acting hero, because a question
    hero enters the pot). A chart node is a lookup along a walked line, so
    the hero-inclusive count would (a) drop the natural "4th seat deciding
    against a 3-way pot" lookups and (b) orphan nodes mid-line (a blind's
    fold node excluded while the opener's next decision on the same line is
    included). History-only counting is monotone along a line, so the
    emitted tree is guaranteed connected: every included node's parent is
    included (pinned by test).

    The 9-max chart export uses this as its ``node_filter``: the full 9-max
    tree would emit a six-figure node count (>100MB of chart JSON) whose
    excluded portion is the unconverged 4-way+ limped/deep-raise tail the
    question pipeline already gates out.
    """
    raises = 0
    entrants: set[str] = set()
    for a in node.history_before:
        if a.action in (
            PloActionType.RAISE,
            PloActionType.MIN_RAISE,
            PloActionType.ALL_IN,
        ):
            raises += 1
            entrants.add(a.seat)
        elif a.action is PloActionType.CALL:
            entrants.add(a.seat)
    return raises <= 2 and len(entrants) <= 3  # noqa: PLR2004


def build_pack_class_chart(
    pack: PloPack,
    *,
    progress: Callable[[str], None] | None = None,
    node_filter: Callable[[PloDecisionNode], bool] | None = None,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    """One pack -> ``{heroSeat: {nodeKey: {handKey: [f, c, r]}}}``.

    Reuses the same node walk and range files as :func:`export_pack`. Per
    node, each hand's strategy mass is combo-weighted and folded into three
    slots (fold / call / raise, all-in merged into raise per the contract),
    then aggregated per suffixed class (``AAxx|ds``) AND per bare class
    (``AAxx`` = the combo-weighted merge of the three suit variants -- his
    class-fallback rule reads the bare entries). Classes with zero reach
    mass at a node are omitted (not in range there). Nodes beyond the
    4-raise cap are omitted (not encodable as a nodeKey). ``node_filter``
    (e.g. :func:`clean_line_node` for the 9-max pack) drops nodes BEFORE
    any range file is read; omitted nodes hit the developer's "No chart for
    this line yet" state, never a fabricated fallback.
    """
    spec = pack.spec
    mults = np.array(combo_multiplicities(), dtype=np.float64)
    suffixed_labels, suff_ids, bare_labels, bare_ids = _dev_class_ids()
    n_suff, n_bare = len(suffixed_labels), len(bare_labels)

    out: dict[str, dict[str, dict[str, list[int]]]] = {}
    nodes = sorted(
        enumerate_plo_nodes(pack),
        key=lambda n: (len(n.history_before), n.history_stem),
    )
    for done, node in enumerate(nodes, start=1):
        if node_filter is not None and not node_filter(node):
            continue
        key = dev_node_key(node.history_before, table_size=node.table_size)
        if key is None:
            continue
        # Three slot mass vectors over the 16,432 classes (combo-weighted).
        slot_mass = [np.zeros(HAND_COUNT, dtype=np.float64) for _ in range(3)]
        slot_of = {"fold": 0, "call": 1, "raise": 2, "allin": 2}
        for opt in node.actions:
            arr = np.fromiter(
                (pe[0] for pe in read_rng_values(opt.path)),
                dtype=np.float64,
                count=HAND_COUNT,
            )
            slot_mass[slot_of[_KIND_BY_TYPE[opt.action.action]]] += arr * mults
        totals = slot_mass[0] + slot_mass[1] + slot_mass[2]
        if not np.any(totals > 0.0):
            continue  # zero-reach node: never emitted (matches export_pack)

        node_entry: dict[str, list[int]] = {}
        for labels, ids, n_groups in (
            (bare_labels, bare_ids, n_bare),
            (suffixed_labels, suff_ids, n_suff),
        ):
            sums = [
                np.bincount(ids, weights=slot_mass[s], minlength=n_groups)
                for s in range(3)
            ]
            group_totals = sums[0] + sums[1] + sums[2]
            for g in np.flatnonzero(group_totals > 0.0).tolist():
                node_entry[labels[g]] = _dev_triple(
                    float(sums[0][g]), float(sums[1][g]), float(sums[2][g])
                )

        hero = dev_seat_name(node.actor, table_size=node.table_size)
        if replay_dev_node_key(key, seats=spec.seats) != hero:
            raise AssertionError(
                f"{spec.pack_id} {key!r}: replay hero mismatch vs {hero}"
            )
        out.setdefault(hero, {})[key] = node_entry
        if progress is not None and done % 2000 == 0:
            progress(f"{spec.pack_id}: {done}/{len(nodes)} nodes")
    return out


def dev_pack_id_for_spec(spec: PloPackSpec) -> str:
    """The developer packId for a registered pack spec (fail loudly else)."""
    if spec.table_size == 6:  # noqa: PLR2004
        return (
            DEV_PACK_ID_MTT6 if spec.game_format == "tournament" else DEV_PACK_ID_CASH6
        )
    if spec.table_size == 9 and spec.game_format == "cash":  # noqa: PLR2004
        return DEV_PACK_ID_CASH9
    raise AssertionError(f"{spec.pack_id}: no dev packId for this table/format")


def build_dev_charts(
    packs: tuple[PloPack, ...],
    *,
    progress: Callable[[str], None] | None = None,
    node_filter: Callable[[PloDecisionNode], bool] | None = None,
) -> dict[str, Any]:
    """The full plo-charts-data.json document for the given packs.

    Cash 6-max packs land under ``charts.c6.<depth>``; MTT 6-max packs under
    ``charts.t6.<depth>``; cash 9-max under ``charts.c9.<depth>`` (seats in
    the developer's c9 names via :func:`dev_seat_name`, i.e. our UTG+2 is
    his MP). Only nodes that exist in the packs are emitted (his loader
    shows "No chart for this line yet" on a miss and must never be fed a
    fabricated node). ``node_filter`` is forwarded per pack.
    """
    charts: dict[str, dict[str, Any]] = {}
    for pack in packs:
        spec = pack.spec
        pack_id = dev_pack_id_for_spec(spec)
        depth = str(int(spec.stack_bb))
        chart = build_pack_class_chart(
            pack, progress=progress, node_filter=node_filter
        )
        charts.setdefault(pack_id, {})[depth] = chart
    return {"version": 1, "charts": charts}


#: packId -> pack-internal seat order, for replaying any document's nodeKeys.
_DEV_PACK_SEATS: dict[str, tuple[str, ...]] = {
    DEV_PACK_ID_CASH6: PLO_PACK_6MAX_100BB.seats,
    DEV_PACK_ID_MTT6: PLO_PACK_6MAX_100BB.seats,
    DEV_PACK_ID_CASH9: PLO_PACK_9MAX_100BB.seats,
}


def validate_dev_charts(doc: dict[str, Any]) -> int:
    """Fail-loud validation of a full plo-charts-data.json document.

    The contract checks (plo-charts.md section 7), applied to EVERY entry:
    version 1; known packIds; every heroSeat a legal dev seat name for the
    pack (9-max via the UTG+2 -> MP mapping); every nodeKey replays through
    the independent seat-queue walk to exactly its heroSeat; every handKey a
    class the dev classifier can produce (bare or ``|ds|ss|rbw`` suffixed);
    every leaf ``[fold, call, raise]`` = 3 non-negative ints summing to
    exactly 100. Returns the number of triples checked.
    """
    if doc.get("version") != 1:
        raise AssertionError(f"version {doc.get('version')!r} != 1")
    suffixed_labels, _si, bare_labels, _bi = _dev_class_ids()
    legal_classes = set(suffixed_labels) | set(bare_labels)
    checked = 0
    for dev_pack, depths in doc["charts"].items():
        seats = _DEV_PACK_SEATS.get(dev_pack)
        if seats is None:
            raise AssertionError(f"unknown dev packId {dev_pack!r}")
        legal_seats = {dev_seat_name(s, table_size=len(seats)) for s in seats}
        for depth, seat_map in depths.items():
            int(depth)  # depth keys are number-as-string
            for hero, node_map in seat_map.items():
                if hero not in legal_seats:
                    raise AssertionError(
                        f"{dev_pack} {depth}: illegal heroSeat {hero!r}"
                    )
                for key, entry in node_map.items():
                    if replay_dev_node_key(key, seats=seats) != hero:
                        raise AssertionError(
                            f"{dev_pack} {depth} {key!r}: replay != {hero}"
                        )
                    if not entry:
                        raise AssertionError(
                            f"{dev_pack} {depth} {key!r}: empty node entry"
                        )
                    for hand_key, triple in entry.items():
                        if hand_key not in legal_classes:
                            raise AssertionError(
                                f"{dev_pack} {depth} {key!r}: "
                                f"illegal handKey {hand_key!r}"
                            )
                        if (
                            len(triple) != 3  # noqa: PLR2004
                            or any(
                                not isinstance(v, int) or v < 0 for v in triple
                            )
                            or sum(triple) != 100  # noqa: PLR2004
                        ):
                            raise AssertionError(
                                f"{dev_pack} {depth} {key!r} {hand_key}: "
                                f"bad triple {triple}"
                            )
                        checked += 1
    return checked


__all__ = [
    "CHART_BUCKETS",
    "CHART_SUITS",
    "DEV_PACK_ID_CASH6",
    "DEV_PACK_ID_CASH9",
    "DEV_PACK_ID_MTT6",
    "PLO_CHART_EXPORT_PACK_IDS",
    "PackExportStats",
    "build_dev_charts",
    "build_pack_class_chart",
    "chart_bucket",
    "chart_suits",
    "clean_line_node",
    "dev_hand_class",
    "dev_node_key",
    "dev_pack_id_for_spec",
    "dev_seat_name",
    "dev_suit_suffix",
    "export_pack",
    "replay_dev_node_key",
    "safe_node_key",
    "spec_for_pack_id",
    "validate_dev_charts",
    "validate_pack_export",
]
