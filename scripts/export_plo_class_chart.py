#!/usr/bin/env python
"""Emit the app developer's per-class PLO chart file (plo-charts-data.json).

The contract is plo-charts.md sections 6-8 (his Charts-tab data layer): ONE
JSON file, skeleton ``{"version":1,"charts":{pack:{depth:{heroSeat:{nodeKey:
{handKey:[fold,call,raise]}}}}}}``, per-CLASS triples (his classifier +
``|ds|ss|rbw`` suffixes, plus the bare-class merge his fallback rule reads).
Derived from the SAME pack data as the per-hand directory export
(``scripts/export_plo_chart_packs.py``); logic in
``pipeline/plo/chart_export.py`` (build_dev_charts and friends).

Usage::

    venv/bin/python scripts/export_plo_class_chart.py
    venv/bin/python scripts/export_plo_class_chart.py --packs plo_6max_12bb
    venv/bin/python scripts/export_plo_class_chart.py --out elsewhere.json
    venv/bin/python scripts/export_plo_class_chart.py \
        --packs plo_6max_150bb --merge   # add depths into an existing file

``--merge`` loads the existing ``--out`` file, adds ONLY the built packs'
depth subtrees, deep-compares every untouched pack/depth subtree before vs
after (additive by construction, verified anyway), validates the whole
merged document (triples, nodeKey replay, class keys), and writes
atomically (tmp + replace). ``--clean-lines`` restricts the node walk to
the pack family's clean-line convention (<= 2 raises, <= 3 pot entrants)
-- required for the 9-max pack, whose full tree would emit >100MB.

Includes a build-time spot-check against the per-hand export (root-node
class aggregation must match the emitted triples within rounding).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.plo.chart_export import (  # noqa: E402
    PLO_CHART_EXPORT_PACK_IDS,
    build_dev_charts,
    clean_line_node,
    dev_hand_class,
    dev_pack_id_for_spec,
    dev_suit_suffix,
    spec_for_pack_id,
    validate_dev_charts,
)
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.hand_order import parse_monker_label  # noqa: E402
from pipeline.plo.pack import discover_plo_pack  # noqa: E402

DEFAULT_OUT = Path("/Users/zackdelson/Downloads/plo-charts-data.json")
PER_HAND_DIR = REPO_ROOT / "test_output" / "plo_chart_export"

# Spot-check tolerance (points per slot): the per-hand mixes are integers,
# so re-aggregating them can drift from the exact-mass triples by rounding.
_SPOT_CHECK_TOLERANCE = 2


def _spot_check_root(doc: dict, pack_id: str) -> float:
    """Aggregate the per-hand export's ROOT mixes per class and compare.

    At the root every hand's reach is 1, so class mass per slot ==
    sum(combos * mix/100) over the root hand list. Returns the max abs
    drift seen (must be within rounding tolerance).
    """
    spec = spec_for_pack_id(pack_id)
    dev_pack = "t6" if spec.game_format == "tournament" else "c6"
    depth = str(int(spec.stack_bb))
    root_file = PER_HAND_DIR / pack_id / "nodes" / "root.json.gz"
    with gzip.open(root_file, "rt", encoding="utf-8") as fh:
        hands = json.load(fh)["hands"]

    slot_of = {"Fold": 0, "Call": 1}  # raises + all-ins -> slot 2
    masses: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for entry in hands:
        cls = classify_plo_hand(parse_monker_label(entry["h"]))
        key = f"{dev_hand_class(cls)}|{dev_suit_suffix(cls)}"
        for act_label, pct in entry["m"].items():
            slot = slot_of.get(act_label, 2)
            masses[key][slot] += entry["n"] * pct / 100.0

    # Find the root node under any hero seat.
    seat_map = doc["charts"][dev_pack][depth]
    root = next(v[""] for v in seat_map.values() if "" in v)
    max_drift = 0.0
    for key, m in masses.items():
        total = sum(m)
        if total <= 0 or key not in root:
            continue
        expected = [100.0 * x / total for x in m]
        for got, exp in zip(root[key], expected, strict=True):
            drift = abs(got - exp)
            max_drift = max(max_drift, drift)
            if drift > _SPOT_CHECK_TOLERANCE:
                raise AssertionError(
                    f"{pack_id} root {key}: emitted {root[key]} vs per-hand "
                    f"aggregate {[round(e, 2) for e in expected]}"
                )
    return max_drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", nargs="+", default=list(PLO_CHART_EXPORT_PACK_IDS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="add the built packs' depths into the existing --out file "
        "(other subtrees stay byte-identical; atomic write)",
    )
    parser.add_argument(
        "--clean-lines",
        action="store_true",
        help="restrict to clean-line nodes (<=2 raises, <=3 pot entrants); "
        "required for the 9-max pack",
    )
    args = parser.parse_args()

    packs = []
    for pack_id in args.packs:
        spec = spec_for_pack_id(pack_id)
        pack = discover_plo_pack(REPO_ROOT / spec.default_base)
        if pack.pack_id != pack_id:
            print(f"FAIL: {spec.default_base} -> {pack.pack_id} != {pack_id}")
            return 1
        packs.append(pack)

    t0 = time.time()
    doc = build_dev_charts(
        tuple(packs),
        progress=lambda m: print(f"    {m}", flush=True),
        node_filter=clean_line_node if args.clean_lines else None,
    )

    if args.merge:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("version") != doc["version"]:
            print(f"FAIL: existing version {existing.get('version')!r} != 1")
            return 1
        new_subtrees = {
            (dev_pack_id_for_spec(p.spec), str(int(p.spec.stack_bb)))
            for p in packs
        }
        untouched_before = {
            (dp, depth): copy.deepcopy(subtree)
            for dp, depths in existing["charts"].items()
            for depth, subtree in depths.items()
            if (dp, depth) not in new_subtrees
        }
        for dev_pack, depths in doc["charts"].items():
            for depth, subtree in depths.items():
                existing["charts"].setdefault(dev_pack, {})[depth] = subtree
        for (dp, depth), before in untouched_before.items():
            if existing["charts"][dp][depth] != before:
                print(f"FAIL: untouched subtree {dp}/{depth} changed in merge")
                return 1
        print(
            f"merge: kept {len(untouched_before)} existing depth subtrees "
            f"unchanged (deep-equal verified), added/replaced "
            f"{sorted(new_subtrees)}"
        )
        doc = existing

    # Report per-pack node counts + root menus (the developer data flags).
    for dev_pack, depths in doc["charts"].items():
        for depth, seat_map in sorted(depths.items(), key=lambda kv: int(kv[0])):
            n_nodes = sum(len(v) for v in seat_map.values())
            root_seat = next(s for s, v in seat_map.items() if "" in v)
            print(f"{dev_pack} {depth}bb: nodes={n_nodes} root_hero={root_seat}")

    for pack_id in ("plo_6max_12bb", "plo_6max_100bb", "plo_mtt_6max_15bb"):
        if pack_id in args.packs and (PER_HAND_DIR / pack_id).exists():
            drift = _spot_check_root(doc, pack_id)
            print(
                f"spot-check vs per-hand export OK ({pack_id}, "
                f"max drift {drift:.2f} pts)"
            )

    n_triples = validate_dev_charts(doc)
    print(f"validate_dev_charts OK ({n_triples:,} triples checked)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"), ensure_ascii=True)
    os.replace(tmp, args.out)
    size_mb = args.out.stat().st_size / 1024**2
    print(f"DONE: {args.out} ({size_mb:.1f}MB, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
