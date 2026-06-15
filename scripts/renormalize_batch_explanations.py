"""One-time migration: re-run the Layer-6 prose normalizer over saved batch
CSVs so older explanations get the inline-bullet fix (June 2026).

Generation now normalizes formatting at the source (``_normalize_prose`` in
``pipeline.preflop.explanation_generator``), but batches written before that
landed can have run-on explanations where the LLM emitted inline ``-``
bullets instead of newline-separated ones. This applies the SAME normalizer
to the ``Answer Explanation`` column in place, preserving every other column.
It is idempotent (already-clean rows are untouched), so it is safe to re-run.

Usage::

    # default: every *.csv under test_output/preflop_batches/
    venv/bin/python scripts/renormalize_batch_explanations.py
    # or specific files
    venv/bin/python scripts/renormalize_batch_explanations.py path/to/batch.csv
    # preview only, write nothing
    venv/bin/python scripts/renormalize_batch_explanations.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.explanation_generator import _normalize_prose  # noqa: E402

_COL = "Answer Explanation"
_DEFAULT_DIR = (
    Path(__file__).resolve().parent.parent / "test_output" / "preflop_batches"
)


def renormalize_csv(path: Path, *, dry_run: bool) -> tuple[int, int]:
    """Normalize the explanation column in ``path``. Returns (changed, total).

    Reads with ``csv.DictReader`` and rewrites with ``csv.DictWriter`` using
    the file's own header order, so quoting/columns are preserved exactly.
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if _COL not in fieldnames:
        return (0, len(rows))

    changed = 0
    for row in rows:
        before = row.get(_COL, "") or ""
        after = _normalize_prose(before) if before else before
        if after != before:
            row[_COL] = after
            changed += 1

    if changed and not dry_run:
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return (changed, len(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="CSV files (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    paths = args.paths or sorted(_DEFAULT_DIR.glob("*.csv"))
    if not paths:
        sys.exit(f"no CSVs found (looked in {_DEFAULT_DIR})")

    total_changed = 0
    for path in paths:
        if not path.is_file():
            print(f"  skip (missing): {path}")
            continue
        changed, total = renormalize_csv(path, dry_run=args.dry_run)
        total_changed += changed
        flag = " (dry-run)" if args.dry_run and changed else ""
        if changed:
            print(f"  {path.name}: {changed}/{total} explanations re-formatted{flag}")
        else:
            print(f"  {path.name}: clean ({total} rows)")
    verb = "would re-format" if args.dry_run else "re-formatted"
    print(f"\n{verb} {total_changed} explanation(s) across {len(paths)} file(s).")


if __name__ == "__main__":
    main()
