"""Tests for admin_panel.review -- the Review page's pure logic.

No Streamlit needed (the UI lives in admin_panel.app); these cover the
sidecar round-trip, the progress summary, and the defensive parsing.
"""

from __future__ import annotations

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
