#!/usr/bin/env python
"""Flip vacuous-decision questions to validation_status=rejected, in place.

A question is VACUOUS when its animation_script shows the hero with 0bb
behind at the decision event -- the hero already called all-in and has no
legal action (Aug 2026 MTT audit; 47 shipped across 5 batches). The
generation-side gate (pipeline/preflop/batch.py, spots_skipped_hero_all_in)
prevents new ones; this script retro-marks the shipped rows so the approved
pool and app exports exclude them. Rows are NEVER deleted (hand IDs and
row numbering stay stable); grades in the .review.json sidecars for these
rows are dropped so a stale approval can't resurrect them.

Usage:
    python scripts/reject_vacuous_questions.py [--dry-run] [glob ...]
Default globs: test_output/*batches*/*.csv
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
csv.field_size_limit(10_000_000)


def is_vacuous(row: dict) -> bool:
    cell = (row.get("animation_script") or "").strip()
    if not cell.startswith("{"):
        return False
    try:
        a = json.loads(cell)
    except json.JSONDecodeError:
        return False
    hero = a.get("hero_seat")
    stack = None
    for e in a.get("events", []):
        if e.get("seat") == hero and "stack_bb" in e:
            stack = e["stack_bb"]
        if e.get("type") == "decision" and e.get("seat") == hero and stack == 0:
            return True
    return False


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    globs = [a for a in argv if not a.startswith("-")] or [
        "test_output/*batches*/*.csv"
    ]
    total = 0
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            p = Path(path)
            try:
                with p.open(newline="", encoding="utf-8-sig") as fh:
                    reader = csv.DictReader(fh)
                    fieldnames = reader.fieldnames
                    rows = list(reader)
            except (OSError, csv.Error):
                continue
            if not fieldnames or "validation_status" not in fieldnames:
                continue
            hit = [r for r in rows if is_vacuous(r)
                   and r.get("validation_status") != "rejected"]
            if not hit:
                continue
            total += len(hit)
            nos = [str(r.get("No", "")) for r in hit]
            print(f"{p.name}: {len(hit)} vacuous -> rejected  (No: {', '.join(nos)})")
            if dry:
                continue
            for r in hit:
                r["validation_status"] = "rejected"
            with p.open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            sidecar = p.with_suffix(".review.json")
            if sidecar.is_file():
                try:
                    reviews = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    reviews = None
                if isinstance(reviews, dict):
                    changed = False
                    for no in nos:
                        if no in reviews:
                            del reviews[no]
                            changed = True
                    if changed:
                        sidecar.write_text(
                            json.dumps(reviews, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
    print(f"\n{'would reject' if dry else 'rejected'}: {total} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
