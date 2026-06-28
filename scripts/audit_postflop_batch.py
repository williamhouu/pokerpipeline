"""Deterministic re-verification of a generated POSTFLOP batch (Layer-7 precursor).

The postflop analogue of ``scripts/audit_preflop_batch.py``. For every CSV row
it reloads the SOURCE SOLVE (from the ``.db`` recorded in the batch's
``.meta.json`` provenance), rebuilds the spot from the node id + hero combo, and
recomputes everything Python is responsible for, then diffs against what the CSV
says. Two classes of check:

* EXACT -- pure-deterministic fields must match byte-for-byte: the Question
  prose, the table-state columns (User Seat / Cards on Table / POT / ...), the
  Context, action_frequencies, action_ev_bb, options + correct answer,
  neutral_credit, archetype, board_texture, solver_reference, SPR, pot_odds,
  and the non-equity concept tags.
* TOLERANCED -- anything downstream of the Monte-Carlo equity estimate (hero
  equity, range equity, the equity-threshold concept tags, the final difficulty
  score). These are seeded and so usually reproduce exactly, but are diffed with
  a generous tolerance so genuine drift is flagged while threshold jitter is not.

The explanation prose (LLM-written) and claim_check / validation_status columns
are NOT deterministic, so they are skipped.

Usage::

    venv/bin/python scripts/audit_postflop_batch.py \\
        "test_output/postflop_batches/postflop_QsJd9s_20260624_101500.csv"

    # older batch without provenance, or to override the solve:
    venv/bin/python scripts/audit_postflop_batch.py <batch.csv> /path/to/solve.db

Prints a per-row report; exits 1 if any EXACT check failed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.postflop.adapters.sqlite_db import load_postflop_db  # noqa: E402
from pipeline.postflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.postflop.explanation_generator import (  # noqa: E402
    placeholder_explanation,
)
from pipeline.postflop.facts import DEFAULT_EQUITY_RUNOUTS, extract_facts  # noqa: E402
from pipeline.postflop.format_writer import build_postflop_row  # noqa: E402
from pipeline.postflop.options import build_options  # noqa: E402
from pipeline.postflop.spot_sampler import sample_spot  # noqa: E402

# Pure-deterministic columns: must match exactly given (node, combo, solve).
EXACT_COLS = (
    "Hand Stage", "Context", "User Seat", "User Cards", "Cards on Table",
    "Table Size", "Default Stack", "Seats", "POT", "Question", "Question Type",
    "Relative Position", "Position Matchup", "Cash/Tourney", "Live or Online",
    "action_frequencies", "action_ev_bb", "solver_reference", "archetype",
    "board_texture", "pot_odds", "spr", "neutral_credit", "Notes", "ranges",
)
# Concept tags driven by the MC equity / sampled range-equity estimates --
# diffed with tolerance, not exactness. Everything else in concept_tags is exact.
EQUITY_TAGS = frozenset({
    "equity_favored", "getting_a_price",
    "range_advantage", "range_disadvantage", "nut_advantage", "nut_disadvantage",
})


def _pct(cell: str) -> float | None:
    cell = (cell or "").strip().rstrip("%")
    try:
        return float(cell)
    except ValueError:
        return None


def audit_batch(csv_path: Path, db_override: str | None) -> int:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    questions = meta["questions"]
    if len(rows) != len(questions):
        print(f"  STRUCTURE-FAIL: {len(rows)} CSV rows vs {len(questions)} meta records")
        return 1

    prov = meta.get("provenance", {}) or {}
    db_path = db_override or prov.get("db_path")
    if not db_path or not Path(db_path).is_file():
        print(
            "  CANNOT AUDIT: no solve .db found. The batch meta has no "
            f"provenance.db_path (or it is missing on disk: {db_path!r}). "
            "Pass the .db path as the second argument."
        )
        return 1

    rs = meta.get("run_settings", {})
    solve = load_postflop_db(
        db_path,
        streets=tuple(prov.get("streets") or ("flop", "turn", "river")),
        max_nodes_per_street=prov.get("max_nodes_per_street"),
        stakes=prov.get("stakes", ""),
        live_or_online=prov.get("live_or_online", "Online"),
        bb_in_dollars=prov.get("bb_in_dollars", 1.0),
    )
    answer_style = rs.get("answer_style", "auto")
    display_in_bb = bool(rs.get("display_in_bb", True))
    equity_runouts = int(rs.get("equity_runouts", DEFAULT_EQUITY_RUNOUTS))

    print("=" * 72)
    print(f"POSTFLOP BATCH AUDIT: {csv_path.name}")
    print(f"  solve={Path(db_path).name}  rows={len(rows)}  "
          f"style={answer_style}  bb={display_in_bb}  runouts={equity_runouts}")
    print("=" * 72)

    exact_failures = 0
    tolerance_notes = 0
    for row, q in zip(rows, questions, strict=True):
        no = row["No"]
        node_id, combo = q["node_id"], q["hero_combo"]
        print(f"--- #{no} {combo} @ {node_id[:60]}")
        node = solve.nodes.get(node_id)
        if node is None:
            print("  EXACT-FAIL: node_id not found in the reloaded solve "
                  "(different streets / down-sampling?)")
            exact_failures += 1
            continue

        spot = sample_spot(node, combo)
        facts = extract_facts(spot, solve, equity_runouts=equity_runouts)
        options, correct = build_options(spot, style=answer_style)
        rebuilt = build_postflop_row(
            facts, placeholder_explanation(facts, options, correct),
            solve, compute_difficulty(facts), int(no), display_in_bb=display_in_bb,
        )

        # 1. EXACT column diffs.
        for col in EXACT_COLS:
            if rebuilt.get(col, "") != row.get(col, ""):
                print(f"  EXACT-FAIL {col}:")
                print(f"    csv:     {row.get(col, '')[:160]!r}")
                print(f"    rebuilt: {rebuilt.get(col, '')[:160]!r}")
                exact_failures += 1

        # 2. Options + correct answer (deterministic).
        csv_opts = [row.get(f"option {i}", "") for i in (1, 2, 3, 4)]
        if (options + ["", "", "", ""])[:4] != csv_opts:
            print(f"  EXACT-FAIL options: rebuilt {options!r} vs csv {csv_opts!r}")
            exact_failures += 1
        if correct != row.get("Correct Answer", ""):
            print(f"  EXACT-FAIL correct: rebuilt {correct!r} vs csv "
                  f"{row.get('Correct Answer')!r}")
            exact_failures += 1

        # 3. Concept tags: equity-bucket tags toleranced, the rest exact.
        csv_tags = {t.strip() for t in row.get("concept_tags", "").split(",") if t.strip()}
        new_tags = {t.strip() for t in rebuilt.get("concept_tags", "").split(",") if t.strip()}
        hard_diff = (csv_tags ^ new_tags) - EQUITY_TAGS
        soft_diff = (csv_tags ^ new_tags) & EQUITY_TAGS
        if hard_diff:
            print(f"  EXACT-FAIL concept_tags differ on: {sorted(hard_diff)}")
            print(f"    csv-only: {sorted(csv_tags - new_tags)}; "
                  f"rebuilt-only: {sorted(new_tags - csv_tags)}")
            exact_failures += 1
        if soft_diff:
            tolerance_notes += 1
            print(f"  tolerance: equity-bucket tags flipped on re-estimate: "
                  f"{sorted(soft_diff)} (borderline, not necessarily a bug)")

        # 4. Equity / range-equity / difficulty within tolerance.
        for col, tol in (("hero_equity", 4.0), ("range_equity", 4.0)):
            a, b = _pct(row.get(col, "")), _pct(rebuilt.get(col, ""))
            if a is not None and b is not None and abs(a - b) > tol:
                tolerance_notes += 1
                print(f"  tolerance: {col} csv {a:.0f}% vs re-estimate {b:.0f}% "
                      "(MC equity noise)")
        try:
            ddrift = abs(int(row["Difficulty Rating"]) - int(rebuilt["Difficulty Rating"]))
            if ddrift > 200:  # noqa: PLR2004
                print(f"  TOLERANCE-FAIL difficulty {row['Difficulty Rating']} vs "
                      f"rebuilt {rebuilt['Difficulty Rating']} (drift {ddrift})")
                exact_failures += 1
            elif ddrift > 40:  # noqa: PLR2004
                tolerance_notes += 1
                print(f"  tolerance: difficulty {row['Difficulty Rating']} -> "
                      f"rebuilt {rebuilt['Difficulty Rating']}")
        except (ValueError, KeyError):
            pass

    print(f"\n=== {len(rows)} rows | EXACT failures: {exact_failures} "
          f"| borderline notes: {tolerance_notes} ===")
    return 1 if exact_failures else 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    db = sys.argv[2] if len(sys.argv) == 3 else None
    sys.exit(audit_batch(Path(sys.argv[1]), db))
