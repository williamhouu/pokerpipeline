"""Deterministic re-verification of a generated preflop batch (Layer-7 precursor).

For every CSV row, rebuilds the spot from the SOURCE PACK (node by
node_id from the meta sidecar, hand class, exact combo) and recomputes
everything Python is responsible for, then diffs against what the CSV
says. Two classes of check:

* EXACT -- pure-deterministic fields must match byte-for-byte: the
  Question prose, table-state tokens (User Seat / Seats / POT), Context,
  action_frequencies, Position Matchup, Relative Position, Table Size,
  solver_reference, options + correct answer, the structural validity of
  the ranges JSON, and the combinatoric concept tags.
* TOLERANCED -- anything downstream of the Monte-Carlo equity estimate
  (hero equity, ev_gap_bb, the equity bucket tags, value/bluff archetype
  variants, equity-coupled skills, the final difficulty score). These are
  recomputed with a larger sample and flagged only when the stored value
  sits outside a generous tolerance of the re-estimate.

Usage::

    venv/bin/python scripts/audit_preflop_batch.py "test_output/preflop_batches/AUDIT BATCH_20260612_104653.csv"

Prints a per-row report; exits 1 if any EXACT check failed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.batch import _placeholder_explanation  # noqa: E402
from pipeline.preflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.preflop.batch import (  # noqa: E402
    _NEAR_PURE_DOMINANT_FREQ,
    _NEAR_PURE_EV_CREDIT_BB,
    ev_gap_from_action_evs,
)
from pipeline.preflop.ev_engine import compute_ev_gap_bb  # noqa: E402
from pipeline.preflop.fact_extractor import extract_facts  # noqa: E402
from pipeline.preflop.format_writer import build_preflop_row  # noqa: E402
from pipeline.preflop.node_enumerator import enumerate_nodes  # noqa: E402
from pipeline.preflop.options import build_options  # noqa: E402
from pipeline.preflop.pack import clear_registry, discover_packs, get_pack  # noqa: E402
from pipeline.preflop.spot_sampler import sample_spot  # noqa: E402

# Fields whose recomputation is pure-deterministic given (node, hand, combo).
EXACT_COLS = (
    "Question", "Context", "User Seat", "User Cards", "Cards on Table",
    "Table Size", "Default Stack", "Seats", "POT", "Relative Position",
    "Position Matchup", "action_frequencies", "solver_reference",
    "archetype", "board_texture", "Cash/Tourney", "Live or Online",
    "Preflop Pot Type", "Pot Participant", "Stack Depth",
)
# Concept tags driven by the MC equity estimate -- diffed with tolerance,
# not exactness. Everything else in concept_tags must match exactly.
EQUITY_TAGS = frozenset({
    "equity_favorite", "coinflip", "equity_underdog",
    "priced_in", "priced_out", "dominated",
    # Range-vs-range advantage tags come from the SAME Monte-Carlo equity
    # machinery (preflop_range_vs_range_equity) as the hand-equity tags above,
    # so a borderline range can flip its bucket on re-estimate just like a
    # borderline hand. Tolerated (not a hard rebuild failure) on that basis.
    "hero_range_advantage", "villain_range_advantage", "roughly_equal_ranges",
})

_SUIT = {"spades": "s", "hearts": "h", "diamonds": "d", "clubs": "c"}
_RANK = {"10": "T"}


def combo_from_user_cards(user_cards: str) -> str:
    """'K-spades, 6-spades' -> 'Ks6s' (the renderer's inverse)."""
    out = []
    for part in user_cards.split(","):
        rank, _, suit = part.strip().partition("-")
        out.append(_RANK.get(rank, rank) + _SUIT[suit.strip()])
    return "".join(out)


def audit_batch(csv_path: Path) -> int:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    questions = meta["questions"]
    assert len(rows) == len(questions), (len(rows), len(questions))

    clear_registry()
    discover_packs(Path(__file__).resolve().parent.parent / "ranges")
    pack = get_pack(meta["pack_id"])
    nodes_by_id = {n.node_id: n for n in enumerate_nodes([pack])}

    # Mirror the batch's trap-aware + razor's-edge difficulty flags (recorded
    # in meta) or the rebuild false-flags Difficulty Rating drift on every
    # re-rated row. Older metas without run_settings default to off,
    # matching their generation.
    trap_difficulty = bool(
        meta.get("run_settings", {}).get("trap_difficulty", False)
    )
    razor_difficulty = bool(
        meta.get("run_settings", {}).get("razor_difficulty", False)
    )

    # Reverse-engineer the batch's display settings from the CSV itself.
    display_in_bb = rows[0]["POT"].endswith("BB")
    live_or_online = rows[0]["Live or Online"]
    stakes = 2.0 if live_or_online == "Live" else 0.50  # only affects $ mode

    exact_failures = 0
    tolerance_notes = 0
    for row, q in zip(rows, questions, strict=True):
        no = row["No"]
        node = nodes_by_id.get(q["node_id"])
        header = f"--- #{no} {q['hand_class']} @ {q['node_id'][:70]}"
        print(header)
        if node is None:
            print("  EXACT-FAIL: node_id not found in pack enumeration")
            exact_failures += 1
            continue
        combo = combo_from_user_cards(row["User Cards"])
        spot = sample_spot(node, q["hand_class"], combo=combo)
        facts = extract_facts(spot, pack, equity_runouts=300)
        # Mirror generation: the solver's own per-action EVs (Monker packs),
        # analytic fallback only for EV-less packs. Must match batch.py or this
        # re-verification false-flags ev_gap_bb on every Monker row.
        ev_gap = ev_gap_from_action_evs(facts, pack)
        if ev_gap is None:
            ev_gap = compute_ev_gap_bb(facts, pack)
        # Mirror generation's near-pure EV-credit override (batch.py): a >=96%
        # dominant spot is an EASY decision, and its tiny mixed action is GTO
        # balancing at ~0 EV gap -- which would mis-rate it HARD on the EV axis.
        # Generation gives the EV axis full credit for difficulty there; the
        # audit must do the same or it false-flags every near-pure spot's
        # difficulty (it only bit once packs shipped per-action file EVs, which
        # make ev_gap_from_action_evs non-None for these spots).
        ev_gap_for_difficulty = ev_gap
        if spot.dominant_frequency >= _NEAR_PURE_DOMINANT_FREQ:
            ev_gap_for_difficulty = _NEAR_PURE_EV_CREDIT_BB
        # Infer the batch's answer style from the option shape (meta does
        # not record it -- noted as a gap): the gto style is the 4-option
        # Always/Mostly spectrum.
        csv_opts_probe = [row[f"option {i}"] for i in (1, 2, 3, 4) if row[f"option {i}"]]
        style = (
            "gto"
            if csv_opts_probe
            and all(o.startswith(("Always ", "Mostly ")) for o in csv_opts_probe)
            else "auto"
        )
        options, correct = build_options(facts, style=style, pack=pack)
        rebuilt = build_preflop_row(
            facts,
            _placeholder_explanation(options, correct),
            pack=pack,
            difficulty=compute_difficulty(
                facts,
                ev_gap_bb=ev_gap_for_difficulty,
                apply_trap_bump=trap_difficulty,
                apply_razor_bump=razor_difficulty,
            ),
            number=int(no),
            stakes_bb_dollars=stakes,
            live_or_online=live_or_online,
            display_in_bb=display_in_bb,
        )

        # 1. EXACT column diffs.
        for col in EXACT_COLS:
            if rebuilt.get(col, "") != row.get(col, ""):
                print(f"  EXACT-FAIL {col}:")
                print(f"    csv:     {row.get(col, '')[:160]!r}")
                print(f"    rebuilt: {rebuilt.get(col, '')[:160]!r}")
                exact_failures += 1

        # 2. Options/correct: the meta snapshot must agree with the CSV, and
        #    the deterministic rebuild must reproduce the option set.
        csv_opts = [row[f"option {i}"] for i in (1, 2, 3, 4)]
        if [o or "" for o in (q["options"] + [""] * 4)[:4]] != csv_opts:
            print(f"  EXACT-FAIL options: meta {q['options']!r} vs csv {csv_opts!r}")
            exact_failures += 1
        if q["correct_answer"] != row["Correct Answer"]:
            print(f"  EXACT-FAIL correct: meta {q['correct_answer']!r} "
                  f"vs csv {row['Correct Answer']!r}")
            exact_failures += 1
        if options != [o for o in csv_opts if o]:
            print(f"  EXACT-FAIL rebuilt options differ: {options!r} vs {csv_opts!r}")
            exact_failures += 1
        if row["Correct Answer"] not in csv_opts:
            print("  EXACT-FAIL correct answer not among options")
            exact_failures += 1

        # 3. Concept tags: equity-bucket tags toleranced, the rest exact.
        csv_tags = {t.strip() for t in row["concept_tags"].split(",") if t.strip()}
        new_tags = {t.strip() for t in rebuilt["concept_tags"].split(",") if t.strip()}
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
                  f"{sorted(soft_diff)} (borderline equity, not necessarily a bug)")

        # 4. Equity / EV-gap / difficulty within tolerance of the re-estimate.
        sd = q["solver_data"]
        orig_eq = sd.get("your_hand_equity_vs_villain_range")
        new_eq = facts.hero_equity_vs_villain
        if orig_eq is not None and new_eq is not None:
            drift = abs(float(orig_eq) - float(new_eq))
            if drift > 0.04:
                print(f"  TOLERANCE-FAIL equity drift {drift:.3f} "
                      f"(stored {orig_eq}, re-estimate {new_eq:.3f})")
                exact_failures += 1
            elif drift > 0.02:
                tolerance_notes += 1
                print(f"  tolerance: equity {orig_eq} vs re-estimate {new_eq:.3f}")
        # ev_gap_bb was dropped from the NLHE preflop CSV (June 2026); older
        # batches still carry it, so read it defensively and skip the check
        # when absent (the gap is still recomputed above for difficulty).
        csv_gap = row.get("ev_gap_bb", "")
        if csv_gap and ev_gap is not None:
            # The analytic call-EV amplifies Monte-Carlo equity noise by
            # (pot + cost): on a 200bb jam pot, +/-1.5% equity is +/-3bb of
            # gap. Tolerance scales with magnitude; beyond it = real drift.
            a, b = float(csv_gap), ev_gap
            if abs(a - b) > max(0.75, 0.30 * max(abs(a), abs(b))):
                print(f"  TOLERANCE-FAIL ev_gap {csv_gap} vs re-estimate {ev_gap:.2f}")
                exact_failures += 1
            elif abs(a - b) > 0.5:
                tolerance_notes += 1
                print(f"  tolerance: ev_gap {csv_gap} vs re-estimate {ev_gap:.2f} "
                      "(MC equity noise amplified by pot size)")
        diff_drift = abs(int(row["Difficulty Rating"]) - int(rebuilt["Difficulty Rating"]))
        if diff_drift > 500:
            print(f"  TOLERANCE-FAIL difficulty {row['Difficulty Rating']} vs "
                  f"rebuilt {rebuilt['Difficulty Rating']} (drift {diff_drift})")
            exact_failures += 1
        elif diff_drift > 60:
            tolerance_notes += 1
            print(f"  tolerance: difficulty {row['Difficulty Rating']} -> "
                  f"rebuilt {rebuilt['Difficulty Rating']}")

        # 5. ranges JSON structural validity.
        try:
            ranges = json.loads(row["ranges"])
            assert isinstance(ranges, dict) and ranges
            for chart in ranges.values():
                assert isinstance(chart, dict)
        except (json.JSONDecodeError, AssertionError):
            print("  EXACT-FAIL ranges column is not valid position-keyed JSON")
            exact_failures += 1

    print(f"\n=== {len(rows)} rows | exact/tolerance FAILURES: {exact_failures} "
          f"| borderline notes: {tolerance_notes} ===")
    return 1 if exact_failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(audit_batch(Path(sys.argv[1])))
