"""Deterministic re-verification of a generated FULL-HAND (play-through) batch.

The play-through analogue of ``scripts/audit_postflop_batch.py``. A full-hand
batch interleaves two leg types in one CSV -- a PREFLOP-ENTRY leg (read from the
solve's flop-entry frequencies) and POSTFLOP legs (flop/turn/river nodes) -- and
links them with ``hand_id`` / ``sequence_index``. This script reloads the source
``.db`` (from the batch ``.meta.json`` provenance, line-closed so deep nodes
resolve), rebuilds EACH leg from its meta record, and diffs against the CSV.

* EXACT (byte-for-byte): the sequence tags (``hand_id`` / ``sequence_index``;
  ``sequence_total`` was dropped from the CSV July 2026), the Question prose,
  the table-state columns, Context,
  options + correct answer, action_frequencies / action_ev_bb, neutral_credit,
  archetype, board_texture, solver_reference, and the non-equity concept
  tags. Preflop legs are fully deterministic (no Monte-Carlo equity).
* TOLERANCED: the equity-bucket concept tags and the difficulty score
  (seeded, usually exact; diffed with a tolerance).

Full-hand CSVs use the TRIMMED schema (July 2026): the pot_odds / hero_equity
/ range_equity / spr / easy_* diagnostic columns are not written, so they are
not audited here (their values still live inside the kept ``stat_notes``).

The LLM prose + claim_check / validation_status columns are non-deterministic
and skipped. Usage::

    venv/bin/python scripts/audit_full_hand_batch.py <full_hand_batch.csv>
    venv/bin/python scripts/audit_full_hand_batch.py <batch.csv> /path/to/solve.db

Exits 1 if any EXACT check failed.
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
from pipeline.postflop.preflop_entry import (  # noqa: E402
    build_preflop_entry_facts,
    build_preflop_entry_options,
    build_preflop_entry_row,
    placeholder_preflop_entry_explanation,
)
from pipeline.postflop.spot_sampler import sample_spot  # noqa: E402

# Pure-deterministic columns that must match exactly for BOTH leg types.
EXACT_COLS = (
    "hand_id", "sequence_index",
    "Hand Stage", "Context", "User Seat", "User Cards", "Cards on Table",
    "Table Size", "Default Stack", "Seats", "POT", "Question", "Question Type",
    "Relative Position", "Position Matchup", "Cash/Tourney", "Live or Online",
    "action_frequencies", "action_ev_bb", "archetype",
    "board_texture", "neutral_credit", "Notes", "ranges",
    # Shared-schema classification columns (deterministic; no MC equity).
    "Preflop Pot Type", "Pot Participant", "Stack Depth",
    # The app's animation timeline (July 2026) -- fully deterministic.
    "animation_script",
)
# Concept tags driven by the MC equity estimate -- toleranced on postflop legs.
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


def _rebuild_postflop(row, q, solve, *, answer_style, display_in_bb,
                      equity_runouts, trap_difficulty=False):
    """Rebuild a postflop leg's row from the solve, using the meta record's
    node_id + combo + sequence tags. Returns (rebuilt_row, options, correct)."""
    node = solve.nodes.get(q["node_id"])
    if node is None:
        return None, None, None
    spot = sample_spot(node, q["hero_combo"])
    facts = extract_facts(spot, solve, equity_runouts=equity_runouts)
    options, correct = build_options(spot, style=answer_style)
    if not display_in_bb:
        # Mirror generation's currency-consistency rule (Aug 2026):
        # dollar batches dollarized their option labels at build time.
        from pipeline.postflop.options import dollarize_options  # noqa: PLC0415

        options, correct = dollarize_options(
            options, correct, bb_in_dollars=solve.bb_in_dollars
        )
    rebuilt = build_postflop_row(
        facts, placeholder_explanation(facts, options, correct), solve,
        compute_difficulty(facts, apply_trap_bump=trap_difficulty),
        int(row["No"]), display_in_bb=display_in_bb,
        hand_id=q.get("hand_id", ""), sequence_index=q.get("sequence_index", ""),
        sequence_total=q.get("sequence_total", ""),
    )
    return rebuilt, options, correct


def _rebuild_preflop(row, q, solve, *, answer_style, display_in_bb):
    """Rebuild a preflop-entry leg's row from the solve's flop-entry ranges."""
    facts = build_preflop_entry_facts(solve, q["hero_position"], q["hero_combo"])
    as_played = bool(q.get("as_played", True))
    options, correct = build_preflop_entry_options(
        facts, style=answer_style, display_in_bb=display_in_bb, as_played=as_played
    )
    expl = placeholder_preflop_entry_explanation(
        facts, options, correct, display_in_bb=display_in_bb, as_played=as_played
    )
    rebuilt = build_preflop_entry_row(
        facts, expl, int(row["No"]), display_in_bb=display_in_bb,
        hand_id=q.get("hand_id", ""), sequence_index=q.get("sequence_index", ""),
        sequence_total=q.get("sequence_total", ""), as_played=as_played,
    )
    return rebuilt, options, correct


def audit_batch(csv_path: Path, db_override: str | None) -> int:  # noqa: C901
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if meta.get("mode") != "full_hand":
        print(f"  NOTE: meta.mode is {meta.get('mode')!r}, not 'full_hand'. "
              "Use scripts/audit_postflop_batch.py for an independent-spots batch.")
    questions = meta.get("questions", [])
    # Each hand's final sequence_index (the leg carrying the showdown).
    final_seq: dict[str, int] = {}
    for q in questions:
        hid = q.get("hand_id", "")
        if hid:
            final_seq[hid] = max(
                final_seq.get(hid, 0), int(q.get("sequence_index") or 0)
            )
    if len(rows) != len(questions):
        print(f"  STRUCTURE-FAIL: {len(rows)} CSV rows vs {len(questions)} meta records")
        return 1

    prov = meta.get("provenance", {}) or {}
    db_path = db_override or prov.get("db_path")
    if not db_path or not Path(db_path).is_file():
        print(f"  CANNOT AUDIT: no solve .db found (provenance.db_path={db_path!r}). "
              "Pass the .db path as the second argument.")
        return 1

    rs = meta.get("run_settings", {})
    solve = load_postflop_db(
        db_path,
        streets=tuple(prov.get("streets") or ("flop", "turn", "river")),
        max_nodes_per_street=prov.get("max_nodes_per_street"),
        include_ancestors=bool(prov.get("include_ancestors", True)),
        stakes=prov.get("stakes", ""),
        live_or_online=prov.get("live_or_online", "Online"),
        bb_in_dollars=prov.get("bb_in_dollars", 1.0),
    )
    answer_style = rs.get("answer_style", "auto")
    display_in_bb = bool(rs.get("display_in_bb", True))
    equity_runouts = int(rs.get("equity_runouts", DEFAULT_EQUITY_RUNOUTS))
    # Mirror the batch's trap-aware difficulty flag (postflop legs only;
    # preflop-entry legs use a frequency-only difficulty with no trap mode).
    trap_difficulty = bool(rs.get("trap_difficulty", False))
    razor_difficulty = bool(rs.get("razor_difficulty", False))
    # Pack-backed preflop legs (July 2026): re-resolve the SAME pack source
    # the batch used (deterministic) so those legs rebuild via the preflop
    # pipeline instead of the entry ranges.
    pack_source = None
    if rs.get("preflop_leg_pack"):
        from pipeline.postflop.preflop_leg_pack import (  # noqa: PLC0415
            find_pack_leg_source,
        )

        _root = Path(__file__).resolve().parent.parent / "ranges"
        pack_source = find_pack_leg_source(solve, _root)
        if pack_source is None or pack_source.pack_id != rs["preflop_leg_pack"]:
            print(f"  NOTE: batch used pack {rs['preflop_leg_pack']!r} but "
                  f"re-resolution found {getattr(pack_source, 'pack_id', None)!r}; "
                  "pack legs will EXACT-FAIL if the packs changed.")

    print("=" * 72)
    print(f"FULL-HAND BATCH AUDIT: {csv_path.name}")
    print(f"  solve={Path(db_path).name}  rows={len(rows)}  hands="
          f"{len(meta.get('hands', []))}  style={answer_style}  bb={display_in_bb}")
    print("=" * 72)

    exact_failures = 0
    tolerance_notes = 0
    for row, q in zip(rows, questions, strict=True):
        no = row["No"]
        is_preflop = (q.get("street") == "preflop") or not q.get("node_id")
        kind = "preflop" if is_preflop else "postflop"
        print(f"--- #{no} [{kind}] {q.get('hero_combo')} "
              f"hand={str(q.get('hand_id'))[:28]} seq={q.get('sequence_index')}")

        if is_preflop and q.get("preflop_leg_source") == "pack":
            from pipeline.postflop.preflop_leg_pack import (  # noqa: PLC0415
                build_pack_preflop_leg_row,
            )

            rebuilt = options = correct = None
            if pack_source is not None:
                rebuilt, _rec, _fail = build_pack_preflop_leg_row(
                    pack_source, q.get("hero_position", ""),
                    q.get("hero_combo", ""), solve,
                    number=int(row["No"]), hand_id=q.get("hand_id", ""),
                    sequence_index=q.get("sequence_index", ""),
                    sequence_total=q.get("sequence_total", ""),
                    use_placeholder=True, client=None, model="",
                    temperature=0.0, max_tokens=0,
                    answer_style=answer_style, display_in_bb=display_in_bb,
                    equity_runouts=equity_runouts,
                    trap_difficulty=trap_difficulty,
                    razor_difficulty=razor_difficulty,
                    # Which decision of a multi-step line (a 3-bet pot's
                    # opener has TWO legs); absent on older SRP batches ->
                    # None = the hero's unique step.
                    step_index=q.get("preflop_step_index"),
                    # Preflop-ENDING hand (balanced lengths, July 2026):
                    # the coherence gate flips to require dominant Fold.
                    terminal_fold=bool(q.get("terminal_fold")),
                    terminal_raise=bool(q.get("terminal_raise")),
                )
                if _rec is not None:
                    options, correct = _rec["options"], _rec["correct_answer"]
            if rebuilt is None:
                print("  EXACT-FAIL: pack leg could not be rebuilt (pack "
                      "missing/changed or coherence gate flipped)")
                exact_failures += 1
                continue
            # Preflop fold resolution: the batch attaches the raiser's
            # vindication reveal to a preflop-ENDING hand's (only) leg;
            # re-attach with the same seeded inputs for byte-exact compare.
            if (
                bool(q.get("terminal_fold")) or bool(q.get("terminal_raise"))
            ) and pack_source is not None:
                import json as _json  # noqa: PLC0415

                from pipeline.postflop.preflop_leg_pack import (  # noqa: PLC0415
                    build_preflop_fold_resolution,
                    build_preflop_raise_resolution,
                )

                _builder = (
                    build_preflop_raise_resolution
                    if q.get("terminal_raise") else build_preflop_fold_resolution
                )
                _res = _builder(
                    pack_source, q.get("hero_position", ""),
                    q.get("hero_combo", ""), solve,
                    hand_id=q.get("hand_id", ""),
                    step_index=q.get("preflop_step_index"),
                )
                if _res is not None:
                    _payload = _json.loads(rebuilt["animation_script"])
                    _payload["version"] = 2
                    _payload["resolution"] = _res
                    rebuilt["animation_script"] = _json.dumps(
                        _payload, separators=(",", ":")
                    )
        elif is_preflop:
            rebuilt, options, correct = _rebuild_preflop(
                row, q, solve, answer_style=answer_style, display_in_bb=display_in_bb
            )
        else:
            rebuilt, options, correct = _rebuild_postflop(
                row, q, solve, answer_style=answer_style,
                display_in_bb=display_in_bb, equity_runouts=equity_runouts,
                trap_difficulty=trap_difficulty,
            )
            if rebuilt is None:
                print("  EXACT-FAIL: node_id not found in the reloaded solve "
                      "(different streets / down-sampling / no include_ancestors?)")
                exact_failures += 1
                continue
            # Showdown resolution (July 2026): the batch attaches it to each
            # hand's FINAL leg; re-attach with the same seeded inputs so the
            # animation_script column compares byte-exact.
            hand_id = q.get("hand_id", "")
            if hand_id and int(q.get("sequence_index") or 0) == final_seq.get(hand_id):
                from pipeline.postflop.showdown import (  # noqa: PLC0415
                    attach_showdown_resolution,
                )

                attach_showdown_resolution(
                    rebuilt, node=solve.nodes[q["node_id"]], solve=solve,
                    hero_combo=q["hero_combo"], correct_answer=correct,
                    hand_id=hand_id,
                )

        # 1. EXACT columns.
        for col in EXACT_COLS:
            if rebuilt.get(col, "") != row.get(col, ""):
                print(f"  EXACT-FAIL {col}:")
                print(f"    csv:     {row.get(col, '')[:160]!r}")
                print(f"    rebuilt: {rebuilt.get(col, '')[:160]!r}")
                exact_failures += 1

        # 2. Options + correct answer.
        csv_opts = [row.get(f"option {i}", "") for i in (1, 2, 3, 4)]
        if (options + ["", "", "", ""])[:4] != csv_opts:
            print(f"  EXACT-FAIL options: rebuilt {options!r} vs csv {csv_opts!r}")
            exact_failures += 1
        if correct != row.get("Correct Answer", ""):
            print(f"  EXACT-FAIL correct: rebuilt {correct!r} vs csv "
                  f"{row.get('Correct Answer')!r}")
            exact_failures += 1

        # 3. Concept tags (preflop: all exact; postflop: equity tags toleranced).
        csv_tags = {t.strip() for t in row.get("concept_tags", "").split(",") if t.strip()}
        new_tags = {t.strip() for t in rebuilt.get("concept_tags", "").split(",") if t.strip()}
        equity_tags = frozenset() if is_preflop else EQUITY_TAGS
        hard_diff = (csv_tags ^ new_tags) - equity_tags
        soft_diff = (csv_tags ^ new_tags) & equity_tags
        if hard_diff:
            print(f"  EXACT-FAIL concept_tags differ on: {sorted(hard_diff)}")
            exact_failures += 1
        if soft_diff:
            tolerance_notes += 1
            print(f"  tolerance: equity-bucket tags flipped: {sorted(soft_diff)}")

        # 4. Difficulty within tolerance. (The hero/range equity columns are
        # not in the trimmed full-hand schema, so there is nothing to diff;
        # the equity-bucket concept tags above still catch equity drift.)
        try:
            ddrift = abs(int(row["Difficulty Rating"]) - int(rebuilt["Difficulty Rating"]))
            if ddrift > 200:  # noqa: PLR2004
                print(f"  TOLERANCE-FAIL difficulty {row['Difficulty Rating']} vs "
                      f"rebuilt {rebuilt['Difficulty Rating']} (drift {ddrift})")
                exact_failures += 1
            elif ddrift > 40:  # noqa: PLR2004
                tolerance_notes += 1
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
