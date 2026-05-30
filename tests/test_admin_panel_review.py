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
