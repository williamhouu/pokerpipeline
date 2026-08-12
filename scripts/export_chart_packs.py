#!/usr/bin/env python
"""Export preflop packs to the app team's SolverPack JSON.

One file per pack, in the Runout chart engine's native shape (Option B of
the developer's ``mtt-chart-export-spec.md``): the NLH MTT 8-max depths
by default, the NLHE cash 6-max packs via ``--cash``. All the logic is in
``pipeline/preflop/chart_export.py`` (pure, unit-tested); this is the thin
CLI.

Usage::

    venv/bin/python scripts/export_chart_packs.py                 # all 7 MTT depths
    venv/bin/python scripts/export_chart_packs.py --depths 10 15  # a subset
    venv/bin/python scripts/export_chart_packs.py --cash          # the cash 6-max packs
    venv/bin/python scripts/export_chart_packs.py --cash8         # the 3 cash 8-max IMPROVED packs
    venv/bin/python scripts/export_chart_packs.py --out somewhere/else
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.preflop.chart_export import (  # noqa: E402
    CASH8_CHART_EXPORT_PACK_IDS,
    CASH_CHART_EXPORT_PACK_IDS,
    build_solver_pack,
    pack_export_id,
    write_solver_pack_json,
)
from pipeline.preflop.pack import discover_packs  # noqa: E402

DEFAULT_DEPTHS = (10, 15, 20, 30, 50, 75, 300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=None,
        help=(
            f"MTT depths (bb) to export (default: {DEFAULT_DEPTHS} "
            "unless --cash is given)"
        ),
    )
    parser.add_argument(
        "--cash",
        action="store_true",
        help=(
            "Export the NLHE cash 6-max packs "
            f"({', '.join(CASH_CHART_EXPORT_PACK_IDS)}). Combine with "
            "--depths to also export MTT depths in the same run."
        ),
    )
    parser.add_argument(
        "--cash8",
        action="store_true",
        help=(
            "Export the 8-max LIVE cash IMPROVED packs "
            f"({', '.join(CASH8_CHART_EXPORT_PACK_IDS)}) as "
            "nlh-8max-cash-<depth>bb.json. Combine with --cash and/or "
            "--depths to export those in the same run."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "test_output" / "chart_export_packs",
        help="Output directory (default: test_output/chart_export_packs)",
    )
    parser.add_argument(
        "--ranges-root",
        type=Path,
        default=REPO_ROOT / "ranges",
        help="Repo ranges/ root used by pack discovery (default: ranges)",
    )
    args = parser.parse_args()

    # --depths keeps its old default behaviour when neither cash flag is
    # given; with --cash/--cash8, MTT depths export only when explicitly
    # requested.
    depths = args.depths
    if depths is None:
        depths = [] if (args.cash or args.cash8) else list(DEFAULT_DEPTHS)

    discovered = discover_packs(args.ranges_root)
    packs = {
        int(p.stack_depth_bb): p
        for p in discovered
        if p.pack_id.startswith("monker_mtt8_")
    }
    missing = [d for d in depths if d not in packs]
    if missing:
        print(f"ERROR: no extracted mtt8 pack for depth(s) {missing}")
        return 1

    todo = [packs[d] for d in sorted(depths)]
    by_id = {p.pack_id: p for p in discovered}
    for flag, wanted in (
        (args.cash, CASH_CHART_EXPORT_PACK_IDS),
        (args.cash8, CASH8_CHART_EXPORT_PACK_IDS),
    ):
        if not flag:
            continue
        missing_cash = [pid for pid in wanted if pid not in by_id]
        if missing_cash:
            print(f"ERROR: cash pack(s) not discovered: {missing_cash}")
            return 1
        todo.extend(by_id[pid] for pid in wanted)

    for pack in todo:
        started = time.time()
        last_report = [0.0]

        def progress(done: int, total: int) -> None:
            if time.time() - last_report[0] >= 10.0:
                last_report[0] = time.time()
                print(f"  ... {done}/{total} nodes", flush=True)

        print(f"[{pack.pack_id}] exporting ...", flush=True)
        solver_pack = build_solver_pack(pack, progress=progress)
        out_path = args.out / f"{pack_export_id(pack)}.json"
        size = write_solver_pack_json(solver_pack, out_path)
        print(
            f"[{pack.pack_id}] wrote {out_path.name}: "
            f"{solver_pack['decisionNodes']} nodes, "
            f"{size / 1_000_000:.1f} MB in {time.time() - started:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
