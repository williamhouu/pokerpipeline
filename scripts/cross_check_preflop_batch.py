"""Re-run the deterministic first-principles cross-checks on ANY existing
preflop batch (new batches run them automatically at generation time).

Usage:
    python scripts/cross_check_preflop_batch.py <batch.csv> [more.csv ...]

Prints each row's findings; exit code = total findings (0 = clean). See
pipeline/preflop/batch_cross_check.py for what is checked and why the
checks deliberately avoid the pipeline's own position/domination code.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.batch_cross_check import cross_check_batch  # noqa: E402


def check_one(csv_path: Path) -> int:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    meta = json.loads(
        csv_path.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    rs = meta.get("run_settings", {})
    findings = cross_check_batch(
        rows,
        meta["questions"],
        min_difficulty=rs.get("min_difficulty", 400),
        max_difficulty=rs.get("max_difficulty", 3200),
    )
    total = 0
    print(f"===== {csv_path.name} ({len(rows)} rows) =====")
    for idx, found in sorted(findings.items()):
        for f in found:
            total += 1
            print(f"  PROBLEM row #{rows[idx].get('No', idx + 1)}: {f}")
    print(f"  -> {total} problems")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:  # noqa: PLR2004
        print(__doc__)
        sys.exit(2)
    grand = sum(check_one(Path(a)) for a in sys.argv[1:])
    sys.exit(min(grand, 120))
