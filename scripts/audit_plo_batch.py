"""Deterministic PLO batch re-verifier (July 2026).

The PLO analogue of ``scripts/audit_preflop_batch.py`` / ``audit_postflop_batch
.py``: rebuild EVERY row of a generated PLO batch from the pack + the batch's
``.meta.json`` sidecar and diff it against the CSV as written. Everything
outside the LLM prose is deterministic -- the meta records the RESOLVED seed
(generation resolves ``seed=None`` to a concrete value precisely so this
script exists), and generation builds each spot's facts with a fresh
``random.Random(seed)``, which we reproduce here -- so recomputation is
byte-identical and ANY drift is a regression (a code change, a pack change,
or a corrupted file). The ``Answer Explanation`` column is passed through
verbatim (LLM prose is not recomputable) which makes the diff EXACT on every
column.

Usage::

    python scripts/audit_plo_batch.py test_output/plo_batches/<batch>.csv \
        [--pack-dir plo_ranges]

Requires the batch's ``.meta.json`` sidecar next to the CSV (batches generated
before July 2026 have none -- regenerate them; there is nothing to join on).
Exit code 0 = every row rebuilt byte-identical; 1 = drift found.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.plo.difficulty import compute_plo_difficulty  # noqa: E402
from pipeline.plo.fact_extractor import extract_plo_facts  # noqa: E402
from pipeline.plo.format_writer import PLO_CSV_COLUMNS, build_plo_row  # noqa: E402
from pipeline.plo.node_enumerator import enumerate_plo_nodes  # noqa: E402
from pipeline.plo.options import build_options  # noqa: E402
from pipeline.plo.pack import discover_plo_pack  # noqa: E402
from pipeline.plo.spot_sampler import (  # noqa: E402
    sample_plo_spot,
    strip_artifact_allins,
)

# Columns we cannot recompute: the LLM prose, the Layer-7 checker's verdict
# (also LLM output), and validation_status (set from the checker's verdict at
# generation time and later mutated by reviewers). Passed through verbatim so
# the row diff stays exact everywhere else.
_PASSTHROUGH_COLS = ("Answer Explanation", "claim_check", "validation_status")

_DEEP_STACK_BB = 40.0  # mirrors batch.py's allins_ok threshold


def _load(csv_path: Path) -> tuple[list[dict[str, str]], dict]:
    meta_path = csv_path.with_suffix(".meta.json")
    if not meta_path.exists():
        sys.exit(
            f"no meta sidecar at {meta_path} -- this batch predates the "
            "July 2026 meta sidecar; regenerate it to make it auditable"
        )
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads(meta_path.read_text())
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_csv", type=Path)
    ap.add_argument(
        "--pack-dir", default="plo_ranges",
        help="Folder holding the pack's .rng files (default: plo_ranges)",
    )
    args = ap.parse_args()

    rows, meta = _load(args.batch_csv)
    rs = meta["run_settings"]
    questions = meta["questions"]
    if len(rows) != len(questions):
        print(f"row/record count mismatch: {len(rows)} CSV vs {len(questions)} meta")
        return 1

    pack = discover_plo_pack(Path(args.pack_dir))
    nodes_by_id = {n.node_id: n for n in enumerate_plo_nodes(pack)}
    seed = rs["seed"]
    stack_bb = float(rs["stack_bb"])
    allins_ok = stack_bb <= _DEEP_STACK_BB

    exact_failures = 0
    for row, q in zip(rows, questions, strict=True):
        no = row.get("No", "?")
        header = f"--- #{no} {q['hero_label']} @ {q['node_id']}"
        print(header)
        node = nodes_by_id.get(q["node_id"])
        if node is None:
            print("  EXACT-FAIL: node_id not found in pack enumeration")
            exact_failures += 1
            continue
        spot = sample_plo_spot(node, int(q["hero_index"]))
        # Mirror generation's ARTIFACT-STRIP (deep stacks only).
        if not allins_ok:
            spot = strip_artifact_allins(spot)
            if spot.artifact_material:
                print("  EXACT-FAIL: spot is artifact-material (should never ship)")
                exact_failures += 1
                continue
        # Mirror generation's per-spot equity RNG: a fresh Random(batch seed).
        facts = extract_plo_facts(
            spot, pack,
            compute_equity=bool(rs["compute_equity"]),
            rng=random.Random(seed),
        )
        difficulty = compute_plo_difficulty(facts)
        options, correct = build_options(facts, style=rs["answer_style"])
        rebuilt = build_plo_row(
            facts,
            difficulty=difficulty,
            options=options,
            correct_answer=correct,
            explanation=row.get("Answer Explanation", ""),
            number=int(q["number"]),
            pack_label=meta["pack_label"],
            stakes_bb_dollars=float(rs["stakes_bb_dollars"]),
            game_format=rs["game_format"],
            display_in_bb=bool(rs["display_in_bb"]),
            stack_bb=stack_bb,
        )
        for col in PLO_CSV_COLUMNS:
            if col in _PASSTHROUGH_COLS:
                continue
            got, want = rebuilt.get(col, ""), row.get(col, "")
            if str(got) != str(want):
                print(f"  EXACT-FAIL {col}:")
                print(f"    csv:     {want[:160]!r}")
                print(f"    rebuilt: {str(got)[:160]!r}")
                exact_failures += 1

    print()
    print(f"=== {len(rows)} rows | EXACT failures: {exact_failures} ===")
    return 1 if exact_failures else 0


if __name__ == "__main__":
    sys.exit(main())
