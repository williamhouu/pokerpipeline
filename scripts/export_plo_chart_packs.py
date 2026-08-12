#!/usr/bin/env python
"""Export PLO preflop packs to the app team's chart JSON directories.

One directory per pack (index.json + gzipped per-node hand lists), in the
format documented by the README this script writes to the output root. All
the logic is in ``pipeline/plo/chart_export.py`` (pure, unit-tested); this
is the thin CLI.

Usage::

    venv/bin/python scripts/export_plo_chart_packs.py                # wave-1 13 packs
    venv/bin/python scripts/export_plo_chart_packs.py --packs plo_6max_12bb
    venv/bin/python scripts/export_plo_chart_packs.py --out somewhere/else

Packs are exported smallest-first (by range-file count). The run stops
gracefully (finished packs are kept and reported) if the cumulative output
would exceed ``--max-total-gb``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.plo.chart_export import (  # noqa: E402
    PLO_CHART_EXPORT_PACK_IDS,
    export_pack,
    spec_for_pack_id,
)
from pipeline.plo.chart_export_readme import README_MD  # noqa: E402
from pipeline.plo.pack import discover_plo_pack  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "test_output" / "plo_chart_export"


def _pack_file_count(pack) -> int:  # noqa: ANN001
    return sum(1 for _ in pack.root.glob("*.rng"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packs",
        nargs="+",
        default=list(PLO_CHART_EXPORT_PACK_IDS),
        help="pack ids to export (default: the 13 wave-1 6-max packs)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-total-gb",
        type=float,
        default=6.0,
        help="stop (keeping finished packs) if cumulative output exceeds this",
    )
    args = parser.parse_args()

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text(README_MD, encoding="utf-8")

    # Resolve + discover every requested pack up front (fail before any work).
    packs = []
    for pack_id in args.packs:
        spec = spec_for_pack_id(pack_id)
        pack = discover_plo_pack(REPO_ROOT / spec.default_base)
        if pack.pack_id != pack_id:
            print(
                f"FAIL: {spec.default_base} resolves to {pack.pack_id}, "
                f"expected {pack_id}"
            )
            return 1
        packs.append(pack)
    packs.sort(key=_pack_file_count)  # smallest first

    total_bytes = 0
    budget = int(args.max_total_gb * 1024**3)
    done: list[str] = []
    for pack in packs:
        t0 = time.time()
        print(f"=== {pack.pack_id} ({_pack_file_count(pack)} range files) ===")
        stats = export_pack(
            pack, out_root, progress=lambda m: print(f"    {m}", flush=True)
        )
        total_bytes += stats.bytes_written
        done.append(pack.pack_id)
        print(
            f"    nodes={stats.nodes_written} "
            f"pruned_zero_reach={stats.nodes_pruned_zero_reach} "
            f"hand_entries={stats.hand_entries:,} "
            f"size={stats.bytes_written / 1024**2:.1f}MB "
            f"cumulative={total_bytes / 1024**3:.2f}GB "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if total_bytes > budget:
            print(
                f"STOPPING: cumulative output {total_bytes / 1024**3:.2f}GB "
                f"exceeds the {args.max_total_gb}GB budget. "
                f"Finished packs kept: {', '.join(done)}"
            )
            return 2
    print(f"DONE: {len(done)} packs, total {total_bytes / 1024**3:.2f}GB at {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
