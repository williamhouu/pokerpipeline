"""Refresh a kept PLO batch in place after a deterministic-writer change.

Rebuilds every DETERMINISTIC column of a generated PLO batch through the
CURRENT writer (exactly like ``scripts/audit_plo_batch.py``, same meta join,
same seeded RNG) and rewrites the CSV, preserving the LLM passthrough columns
verbatim (``Answer Explanation``, ``claim_check``). Use it when a writer
change (new stat_notes rows, a column tweak) would otherwise make old kept
batches fail the byte-exact re-verifier: refresh, then re-run the audit for
0/0.

Safety rails:
* The rebuilt ``Correct Answer`` and options must MATCH the CSV -- a changed
  answer means the change was not display-only, and the batch must be
  regenerated instead (the explanation prose no longer matches the spot).
* Batches without a ``.meta.json`` sidecar are refused (nothing to join on).

Usage::

    python scripts/refresh_plo_batch.py test_output/plo_batches/<batch>.csv \
        [--pack-dir plo_ranges]
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
from pipeline.plo.format_writer import build_plo_row, write_plo_csv  # noqa: E402
from pipeline.plo.node_enumerator import enumerate_plo_nodes  # noqa: E402
from pipeline.plo.options import build_options  # noqa: E402
from pipeline.plo.pack import discover_plo_pack  # noqa: E402
from pipeline.plo.spot_sampler import (  # noqa: E402
    sample_plo_spot,
    strip_artifact_allins,
)

_PASSTHROUGH_COLS = ("Answer Explanation", "claim_check")
# Answer-key columns that must survive a refresh unchanged; drift here means
# the change was strategic, not display-only -> regenerate, don't refresh.
_ANSWER_COLS = ("Correct Answer", "option 1", "option 2", "option 3", "option 4")
_DEEP_STACK_BB = 40.0  # mirrors batch.py's allins_ok threshold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_csv", type=Path)
    ap.add_argument("--pack-dir", default=None)
    args = ap.parse_args()

    meta_path = args.batch_csv.with_suffix(".meta.json")
    if not meta_path.exists():
        sys.exit(f"no meta sidecar at {meta_path} -- pre-meta batches can't be refreshed")
    with args.batch_csv.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads(meta_path.read_text())
    rs = meta["run_settings"]
    questions = meta["questions"]
    if len(rows) != len(questions):
        sys.exit(f"row/record mismatch: {len(rows)} CSV vs {len(questions)} meta")

    pack_dir = args.pack_dir
    if pack_dir is None:
        from pipeline.plo.pack import KNOWN_PLO_PACKS  # noqa: PLC0415

        pack_id = meta.get("pack_id") or meta.get("pack_label")
        pack_dir = next(
            (s.default_base for s in KNOWN_PLO_PACKS if s.pack_id == pack_id),
            "plo_ranges",
        )
    pack = discover_plo_pack(Path(pack_dir))
    nodes_by_id = {n.node_id: n for n in enumerate_plo_nodes(pack)}
    seed = rs["seed"]
    stack_bb = float(rs["stack_bb"])
    allins_ok = stack_bb <= _DEEP_STACK_BB

    rebuilt_rows: list[dict[str, str]] = []
    for row, q in zip(rows, questions, strict=True):
        node = nodes_by_id.get(q["node_id"])
        if node is None:
            sys.exit(f"#{row.get('No')}: node {q['node_id']} not in pack -- aborting")
        spot = sample_plo_spot(node, int(q["hero_index"]))
        if not allins_ok:
            spot = strip_artifact_allins(spot)
        facts = extract_plo_facts(
            spot, pack,
            compute_equity=bool(rs["compute_equity"]),
            rng=random.Random(seed),
        )
        options, correct = build_options(facts, style=rs["answer_style"])
        rebuilt = build_plo_row(
            facts,
            difficulty=compute_plo_difficulty(facts),
            options=options,
            correct_answer=correct,
            explanation=row.get("Answer Explanation", ""),
            number=int(q["number"]),
            pack_label=meta["pack_label"],
            pack=pack,
            stakes_bb_dollars=float(rs["stakes_bb_dollars"]),
            game_format=rs["game_format"],
            ante_bb=float(rs.get("ante_bb", 0.0)),
            display_in_bb=bool(rs["display_in_bb"]),
            stack_bb=stack_bb,
        )
        for col in _ANSWER_COLS:
            if str(rebuilt.get(col, "")) != str(row.get(col, "")):
                sys.exit(
                    f"#{row.get('No')}: {col} changed "
                    f"({row.get(col)!r} -> {rebuilt.get(col)!r}); the change is "
                    "not display-only -- REGENERATE this batch instead"
                )
        for col in _PASSTHROUGH_COLS:
            rebuilt[col] = row.get(col, "")
        rebuilt_rows.append(rebuilt)

    write_plo_csv(rebuilt_rows, args.batch_csv)
    print(f"refreshed {len(rebuilt_rows)} rows in {args.batch_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
