"""Tests for admin_panel.review -- the Review page's pure logic.

No Streamlit needed (the UI lives in admin_panel.app); these cover the
sidecar round-trip, the progress summary, and the defensive parsing.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel import review  # noqa: E402


def test_sidecar_path_is_next_to_csv(tmp_path: Path) -> None:
    csv = tmp_path / "batch_20260530.csv"
    assert review.review_sidecar_path(csv) == tmp_path / "batch_20260530.review.json"


def test_load_reviews_missing_returns_empty(tmp_path: Path) -> None:
    assert review.load_reviews(tmp_path / "nope.csv") == {}


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    csv = tmp_path / "b.csv"
    review.save_review(csv, 1, "approved", "clean")
    review.save_review(csv, "2", "rejected", "wrong strategy")
    loaded = review.load_reviews(csv)
    assert loaded == {
        "1": {"status": "approved", "note": "clean"},
        "2": {"status": "rejected", "note": "wrong strategy"},
    }


def test_save_review_upserts(tmp_path: Path) -> None:
    csv = tmp_path / "b.csv"
    review.save_review(csv, 1, "needs_review", "first")
    review.save_review(csv, 1, "approved", "changed my mind")
    assert review.load_reviews(csv)["1"] == {
        "status": "approved", "note": "changed my mind"}


def test_save_review_rejects_unknown_status(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        review.save_review(tmp_path / "b.csv", 1, "looks_good", "")


def test_load_reviews_tolerates_corrupt_sidecar(tmp_path: Path) -> None:
    csv = tmp_path / "b.csv"
    review.review_sidecar_path(csv).write_text("{ not json", encoding="utf-8")
    assert review.load_reviews(csv) == {}


def test_load_reviews_drops_malformed_entries(tmp_path: Path) -> None:
    csv = tmp_path / "b.csv"
    review.review_sidecar_path(csv).write_text(
        json.dumps({"1": {"status": "approved", "note": "ok"}, "2": "garbage"}),
        encoding="utf-8",
    )
    assert review.load_reviews(csv) == {
        "1": {"status": "approved", "note": "ok"}}


def test_summarize_counts_and_quality() -> None:
    reviews = {
        "1": {"status": "approved", "note": ""},
        "2": {"status": "approved", "note": ""},
        "3": {"status": "rejected", "note": ""},
        "4": {"status": "needs_review", "note": ""},
    }
    s = review.summarize([1, 2, 3, 4, 5], reviews)  # #5 ungraded
    assert (s.total, s.reviewed, s.approved, s.rejected, s.needs_review) == (
        5, 4, 2, 1, 1)
    # quality = approved / (approved + rejected) = 2/3, needs_review excluded.
    assert s.quality_pct == pytest.approx(100 * 2 / 3)


def test_summarize_quality_none_when_no_decided_grades() -> None:
    s = review.summarize([1, 2], {"1": {"status": "needs_review", "note": ""}})
    assert s.quality_pct is None


def test_range_player_count() -> None:
    assert review.range_player_count('{"UTG":"AA:1","BB":"KK:1","SB":"..."}') == 3
    assert review.range_player_count("") == 0
    assert review.range_player_count("not json") == 0
    assert review.range_player_count("[1,2,3]") == 0  # not a dict


# --- remove_question / remove_review --------------------------------------
def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["No", "User Cards", "Question"])
        writer.writeheader()
        writer.writerows(rows)


def _read_nos(path: Path) -> list[str]:
    import csv
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [r["No"] for r in csv.DictReader(fh)]


def test_remove_question_drops_row_keeps_others(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_csv(csv_path, [
        {"No": "1", "User Cards": "A-spades, K-spades", "Question": "q1"},
        {"No": "2", "User Cards": "7-hearts, 5-clubs", "Question": "q2"},
        {"No": "3", "User Cards": "Q-diamonds, Q-clubs", "Question": "q3"},
    ])
    assert review.remove_question(csv_path, 2) is True
    # #2 gone; #1 and #3 keep their original numbers (no renumbering).
    assert _read_nos(csv_path) == ["1", "3"]


def test_remove_question_preserves_columns_and_commas(tmp_path: Path) -> None:
    """A removed-then-rewritten CSV keeps the header + quotes fields with
    commas (User Cards) so it doesn't corrupt the remaining rows."""
    csv_path = tmp_path / "b.csv"
    _write_csv(csv_path, [
        {"No": "1", "User Cards": "A-spades, K-spades", "Question": "q1"},
        {"No": "2", "User Cards": "7-hearts, 5-clubs", "Question": "q2"},
    ])
    review.remove_question(csv_path, 1)
    import csv as _csv
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows == [{"No": "2", "User Cards": "7-hearts, 5-clubs",
                     "Question": "q2"}]


def test_remove_question_missing_no_returns_false(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_csv(csv_path, [
        {"No": "1", "User Cards": "AA", "Question": "q1"}])
    assert review.remove_question(csv_path, 99) is False
    assert _read_nos(csv_path) == ["1"]


def test_remove_question_also_drops_the_grade(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_csv(csv_path, [
        {"No": "1", "User Cards": "AA", "Question": "q1"},
        {"No": "2", "User Cards": "KK", "Question": "q2"}])
    review.save_review(csv_path, 1, "approved", "ok")
    review.save_review(csv_path, 2, "rejected", "bad")
    review.remove_question(csv_path, 2)
    assert set(review.load_reviews(csv_path)) == {"1"}  # #2's grade gone too


def test_remove_review_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    review.save_review(csv_path, 1, "approved", "ok")
    review.remove_review(csv_path, 1)
    assert review.load_reviews(csv_path) == {}
    # No-op when absent (must not raise).
    review.remove_review(csv_path, 1)


# --- update_explanation (edit one cell in place) ---------------------------
def test_update_explanation_rewrites_only_that_cell(tmp_path: Path) -> None:
    import csv
    csv_path = tmp_path / "b.csv"
    fields = ["No", "User Cards", "Answer Explanation", "ev_gap_bb"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"No": "1", "User Cards": "A-spades, K-spades",
             "Answer Explanation": "old one", "ev_gap_bb": "0.40"},
            {"No": "2", "User Cards": "7-hearts, 5-clubs",
             "Answer Explanation": "old two", "ev_gap_bb": "0.10"},
        ])
    assert review.update_explanation(csv_path, 2, "edited TWO") is True
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # Only #2's explanation changed; other cells + column order intact.
    assert rows[0]["Answer Explanation"] == "old one"
    assert rows[1]["Answer Explanation"] == "edited TWO"
    assert rows[1]["ev_gap_bb"] == "0.10"
    assert list(rows[0].keys()) == fields


def test_update_explanation_missing_no_returns_false(tmp_path: Path) -> None:
    import csv
    csv_path = tmp_path / "b.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["No", "Answer Explanation"])
        writer.writeheader()
        writer.writerow({"No": "1", "Answer Explanation": "x"})
    assert review.update_explanation(csv_path, 99, "y") is False
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        assert list(csv.DictReader(fh))[0]["Answer Explanation"] == "x"


def test_update_difficulty_rewrites_only_that_cell(tmp_path: Path) -> None:
    import csv
    csv_path = tmp_path / "b.csv"
    fields = ["No", "Difficulty Rating", "Answer Explanation"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"No": "1", "Difficulty Rating": "1500", "Answer Explanation": "a"},
            {"No": "2", "Difficulty Rating": "1800", "Answer Explanation": "b"},
        ])
    assert review.update_difficulty(csv_path, 2, "900") is True
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["Difficulty Rating"] == "1500"        # untouched
    assert rows[1]["Difficulty Rating"] == "900"         # edited
    assert rows[1]["Answer Explanation"] == "b"          # other cells intact


def test_update_difficulty_missing_column_returns_false(tmp_path: Path) -> None:
    import csv
    csv_path = tmp_path / "b.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["No", "Answer Explanation"])
        writer.writeheader()
        writer.writerow({"No": "1", "Answer Explanation": "x"})
    assert review.update_difficulty(csv_path, 1, "900") is False


# --- batch meta sidecar (prompt tag + per-spot inputs) ---------------------
def test_batch_meta_path_is_next_to_csv(tmp_path: Path) -> None:
    csv = tmp_path / "batch_20260601.csv"
    assert review.batch_meta_path(csv) == tmp_path / "batch_20260601.meta.json"


def test_load_batch_meta_missing_returns_none(tmp_path: Path) -> None:
    assert review.load_batch_meta(tmp_path / "nope.csv") is None


def test_load_batch_meta_malformed_returns_none(tmp_path: Path) -> None:
    csv = tmp_path / "b.csv"
    review.batch_meta_path(csv).write_text("{not json", encoding="utf-8")
    assert review.load_batch_meta(csv) is None


def test_load_batch_meta_round_trip(tmp_path: Path) -> None:
    csv = tmp_path / "b.csv"
    review.batch_meta_path(csv).write_text(
        json.dumps({"prompt_name": "P", "questions": []}), encoding="utf-8"
    )
    meta = review.load_batch_meta(csv)
    assert meta is not None
    assert meta["prompt_name"] == "P"


def test_meta_question_for_matches_on_node_and_hand() -> None:
    meta: dict[str, object] = {
        "questions": [
            {"node_id": "NODE_A", "user_cards": "Ah, Ks", "live_block": "LA"},
            {"node_id": "NODE_B", "user_cards": "Qh, Qd", "live_block": "LB"},
        ],
    }
    q = review.meta_question_for(
        meta, user_cards="Qh, Qd", solver_reference="pack/BTN/NODE_B"
    )
    assert q is not None
    assert q["live_block"] == "LB"


def test_meta_question_for_no_match_returns_none() -> None:
    meta: dict[str, object] = {
        "questions": [{"node_id": "NODE_A", "user_cards": "Ah, Ks", "live_block": "LA"}],
    }
    # Right hand, wrong node.
    assert (
        review.meta_question_for(
            meta, user_cards="Ah, Ks", solver_reference="pack/BTN/NODE_Z"
        )
        is None
    )
    # Right node, wrong hand.
    assert (
        review.meta_question_for(
            meta, user_cards="Qh, Qd", solver_reference="pack/BTN/NODE_A"
        )
        is None
    )


def test_meta_question_for_handles_missing_questions() -> None:
    assert (
        review.meta_question_for(
            {}, user_cards="Ah, Ks", solver_reference="pack/BTN/NODE_A"
        )
        is None
    )


def test_assembled_prompt_orders_system_gold_then_live() -> None:
    meta = {"prompt_text": "SYSTEXT", "gold_block": "GOLDTEXT"}
    question = {"live_block": "\n\nLIVETEXT"}
    out = review.assembled_prompt(meta, question)
    assert "SYSTEM PROMPT" in out
    assert out.index("SYSTEXT") < out.index("GOLDTEXT") < out.index("LIVETEXT")


# --- collect_approved_rows / approved_rows_to_csv --------------------------
_COLS = ["No", "User Cards", "solver_reference", "Correct Answer"]


def _write_batch(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLS)
        writer.writeheader()
        writer.writerows(rows)


def test_collect_approved_empty_when_dir_absent(tmp_path: Path) -> None:
    fields, rows = review.collect_approved_rows(tmp_path / "nope")
    assert (fields, rows) == ([], [])


def test_collect_approved_only_approved_rows(tmp_path: Path) -> None:
    b = tmp_path / "plo_20260601_120000.csv"
    _write_batch(
        b,
        [
            {"No": "1", "User Cards": "A,A", "solver_reference": "p/BB/n1", "Correct Answer": "Check"},
            {"No": "2", "User Cards": "K,K", "solver_reference": "p/BB/n2", "Correct Answer": "Fold"},
            {"No": "3", "User Cards": "Q,Q", "solver_reference": "p/BB/n3", "Correct Answer": "Call"},
        ],
    )
    review.save_review(b, 1, "approved", "")
    review.save_review(b, 2, "rejected", "")
    # No 3 left ungraded.
    fields, rows = review.collect_approved_rows(tmp_path)
    assert fields == _COLS
    assert [r["No"] for r in rows] == ["1"]


def test_collect_approved_dedupes_across_batches(tmp_path: Path) -> None:
    spot = {"No": "1", "User Cards": "A,A", "solver_reference": "p/BB/n1", "Correct Answer": "Check"}
    older = tmp_path / "plo_20260601_120000.csv"
    newer = tmp_path / "plo_20260602_120000.csv"
    _write_batch(older, [spot])
    _write_batch(newer, [spot])
    review.save_review(older, 1, "approved", "")
    review.save_review(newer, 1, "approved", "")
    _, rows = review.collect_approved_rows(tmp_path)
    assert len(rows) == 1  # same (solver_reference, User Cards) -> one copy


def test_collect_approved_excludes_compare_artifacts(tmp_path: Path) -> None:
    cmp = tmp_path / "compare_20260601_A.csv"
    _write_batch(cmp, [
        {"No": "1", "User Cards": "A,A", "solver_reference": "p/BB/n1", "Correct Answer": "Check"}
    ])
    review.save_review(cmp, 1, "approved", "")
    _, rows = review.collect_approved_rows(tmp_path)
    assert rows == []


def test_approved_rows_to_csv_round_trips(tmp_path: Path) -> None:
    rows = [
        {"No": "1", "User Cards": "A,A", "solver_reference": "p/BB/n1", "Correct Answer": "Check", "extra": "ignored"},
    ]
    text = review.approved_rows_to_csv(_COLS, rows)
    parsed = list(csv.DictReader(text.splitlines()))
    assert parsed[0]["User Cards"] == "A,A"
    assert "extra" not in parsed[0]  # keys outside fieldnames are dropped
