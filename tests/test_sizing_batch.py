"""📏 Bet-sizing trainer batches (pipeline/postflop/sizing_batch.py).

Browserless tests on the in-memory fixture solves: the sizing-viability
selector, the class dedupe, exact difficulty banding, the multi-solve merge
(order, renumbering, per-question solve keys, balance report), determinism,
and honest shortfall. The admin checkbox wiring is pinned in
``test_postflop_generate_page.py``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.postflop.batch import _collect_worthy  # noqa: E402
from pipeline.postflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.postflop.facts import extract_facts  # noqa: E402
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.sizing_batch import (  # noqa: E402
    SIZING_BALANCE_AXES,
    collect_sizing_pool,
    difficulty_band,
    generate_sizing_batch,
    hand_class_key,
    is_sizing_spot,
    size_family,
)


def test_sizing_pool_only_multi_size_bet_dominant_spots() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    pool = collect_sizing_pool(solve)
    assert pool, "fixture must yield at least one sizing spot"
    for spot in pool:
        assert is_sizing_spot(spot)
        sizes = [a for a in spot.live_actions if a.label.startswith("Bet ")]
        assert len(sizes) >= 2
        assert spot.dominant_action.startswith("Bet ")
        assert 0.65 <= spot.dominant_frequency <= 0.99


def test_sizing_pool_dedupes_suit_twins_by_hand_class() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    pool = collect_sizing_pool(solve)
    keys = [(s.node.node_id, hand_class_key(s.hero_combo)) for s in pool]
    assert len(keys) == len(set(keys)), "one combo per (node, hand class)"


def test_hand_class_key() -> None:
    assert hand_class_key("5h5d") == "55"
    assert hand_class_key("Kc7c") == "K7s"
    assert hand_class_key("Kc7d") == "K7o"


def test_size_family_buckets() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    pool = collect_sizing_pool(solve)
    fams = {size_family(s) for s in pool}
    assert fams <= {
        "small (<45% pot)", "medium (45-90%)", "big/overbet (90%+)", "unsized",
    }


def test_pool_difficulty_band_is_exact() -> None:
    """The band from reduced-runout pool facts == the band generation computes.

    INVARIANT (module docstring): postflop difficulty reads nothing derived
    from the sampled hero equity, so scoring the pool at few runouts changes
    no band. If this ever fails, a difficulty input grew an equity
    dependency -- the pool scorer must then move to full runouts.
    """
    solve = btn_vs_bb_srp_2cJs7s()
    worthy, *_ = _collect_worthy(
        solve, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None
    )
    assert worthy
    for spot in worthy:
        few = compute_difficulty(extract_facts(spot, solve, equity_runouts=20))
        full = compute_difficulty(extract_facts(spot, solve, equity_runouts=400))
        assert few.score == full.score, (
            spot.node.node_id, spot.hero_combo, few.score, full.score
        )
        assert difficulty_band(few.score) == difficulty_band(full.score)


def _dry_batch(tmp_path: Path, name: str = "sz.csv", total: int = 4):
    out = tmp_path / name
    result = generate_sizing_batch(
        [],
        out,
        total_questions=total,
        dry_run=True,
        solves_loaded={
            "FlopA SRP": btn_vs_bb_srp_2cJs7s(),
            "FlopB SRP": btn_vs_bb_srp_2cJs7s(),
        },
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    return result, rows, meta


def test_multi_solve_merge_renumbers_and_tags_solve_keys(tmp_path: Path) -> None:
    result, rows, meta = _dry_batch(tmp_path)
    assert result.questions_written == len(rows) == len(meta["questions"])
    assert [r["No"] for r in rows] == [str(i + 1) for i in range(len(rows))]
    keys = {q["solve_key"] for q in meta["questions"]}
    assert keys == {"FlopA SRP", "FlopB SRP"}, "both solves contribute"
    assert meta["provenance"]["mode"] == "sizing_multi"
    assert set(meta["provenance"]["solves"]) == keys
    # Balance report covers every declared axis.
    labels = [a["label"] for a in meta["balance_report"]["axes"]]
    assert labels == [lbl for _k, lbl, _w in SIZING_BALANCE_AXES]


def test_multi_solve_batch_is_deterministic(tmp_path: Path) -> None:
    _r1, rows1, meta1 = _dry_batch(tmp_path, "a.csv")
    _r2, rows2, meta2 = _dry_batch(tmp_path, "b.csv")
    assert rows1 == rows2
    assert meta1["questions"] == meta2["questions"]
    assert meta1["balance_report"] == meta2["balance_report"]


def test_shortfall_is_honest_not_padded(tmp_path: Path) -> None:
    """Asking for more than the pool holds ships the pool, never duplicates."""
    result, rows, _meta = _dry_batch(tmp_path, "short.csv", total=50)
    assert result.questions_written == len(rows) == result.pool_scored
    seen = {(r["User Cards"], r["Cards on Table"], r["Question"]) for r in rows}
    # Two identical fixture solves => pairs of identical spots is expected;
    # but each (solve, node, combo) must appear at most once.
    keyset = [
        (q["solve_key"], q["node_id"], q["hero_combo"])
        for q in _meta["questions"]
    ]
    assert len(keyset) == len(set(keyset))
    assert seen  # rows are real


def test_all_options_are_check_plus_sizes(tmp_path: Path) -> None:
    """Sizing batches ask 'which size?': options = Check + the bet sizes."""
    _result, rows, _meta = _dry_batch(tmp_path, "opts.csv")
    for r in rows:
        opts = [r[f"option {i}"] for i in (1, 2, 3, 4) if r.get(f"option {i}")]
        assert any(o.startswith("Bet ") for o in opts), opts
        assert r["Correct Answer"].startswith("Bet "), r["Correct Answer"]


# --- currency consistency + even split (Aug 2026) ----------------------------
def test_dollarize_options_matches_prose_convention():
    from pipeline.postflop.options import dollarize_label, dollarize_options

    opts, corr = dollarize_options(
        ["Check", "Bet 11.5bb", "Bet 23bb", ""], "Bet 11.5bb", bb_in_dollars=2.0
    )
    assert opts == ["Check", "Bet $23", "Bet $46", ""]
    assert corr == "Bet $23"
    # Cents kept when not whole (the make_amount_fmt convention).
    assert dollarize_label("Raise to 2.17bb", 2.0) == "Raise to $4.34"
    # Verb-only labels untouched.
    assert dollarize_label("All-in", 2.0) == "All-in"
