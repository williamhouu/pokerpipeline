"""Tests for hand-level review of full-hand (play-through) batches.

The unit a reviewer ships is the HAND: one click grades every leg, the
filtered download exports only whole hands, and hand removal drops all
legs atomically. Pure-function tests -- no Streamlit runtime (the
fix-durability rule: the UI is a thin shell over what these tests pin).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from admin_panel import review  # noqa: E402

COLS = ["No", "hand_id", "sequence_index", "Answer Explanation", "User Cards"]


def _write_batch(path: Path) -> None:
    """Two 2-leg hands + one standalone row."""
    rows = [
        {"No": "1", "hand_id": "h1", "sequence_index": "1",
         "Answer Explanation": "e1", "User Cards": "As Ks"},
        {"No": "2", "hand_id": "h1", "sequence_index": "2",
         "Answer Explanation": "e2", "User Cards": "As Ks"},
        {"No": "3", "hand_id": "h2", "sequence_index": "1",
         "Answer Explanation": "e3", "User Cards": "Qd Qc"},
        {"No": "4", "hand_id": "h2", "sequence_index": "2",
         "Answer Explanation": "e4", "User Cards": "Qd Qc"},
        {"No": "5", "hand_id": "", "sequence_index": "",
         "Answer Explanation": "e5", "User Cards": "7h 2c"},
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_default_postflop_venue_by_table_size() -> None:
    """Full-ring (8/9-max) solves default to Live framing; 6-max to Online.
    Regression: the old >= 9 rule framed the 8-max live-cash solves as
    'Online $1/$2 with a capped live rake' in every generated Context."""
    from admin_panel.app import _default_postflop_venue

    assert _default_postflop_venue(8) == "Live"
    assert _default_postflop_venue(9) == "Live"
    assert _default_postflop_venue(6) == "Online"
    assert _default_postflop_venue(None) == "Live"  # unknown = full-ring


def test_hand_status_aggregation() -> None:
    reviews = {
        "1": {"status": "approved"}, "2": {"status": "approved"},
        "3": {"status": "approved"}, "4": {"status": "rejected"},
        "5": {"status": "needs_review"}, "6": {"status": "approved"},
    }
    assert review.hand_status(["1", "2"], reviews) == "approved"
    # WHOLE-HAND ATOMICITY: one rejected leg rejects the hand, even if
    # another leg is approved.
    assert review.hand_status(["3", "4"], reviews) == "rejected"
    assert review.hand_status(["5", "6"], reviews) == "needs_review"
    assert review.hand_status(["1", "9"], reviews) == ""  # partial = pending
    assert review.hand_status([], reviews) == ""


def test_save_reviews_bulk_preserves_notes_and_explanations(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_batch(csv_path)
    review.save_review(csv_path, "1", "needs_review", "AUDIT (Claude): flag",
                       explanation="edited text")
    review.save_reviews_bulk(csv_path, ["1", "2"], "approved")
    grades = review.load_reviews(csv_path)
    assert grades["1"]["status"] == "approved"
    assert grades["1"]["note"] == "AUDIT (Claude): flag"  # preserved
    assert grades["1"]["explanation"] == "edited text"    # preserved
    assert grades["2"] == {"status": "approved", "note": ""}
    # An explicit note replaces; invalid status raises before any write.
    review.save_reviews_bulk(csv_path, ["1"], "rejected", note="bad hand")
    assert review.load_reviews(csv_path)["1"]["note"] == "bad hand"
    try:
        review.save_reviews_bulk(csv_path, ["1"], "nonsense")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_remove_hand_drops_all_legs_atomically(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_batch(csv_path)
    review.save_review(csv_path, "3", "approved", "")
    assert review.remove_hand(csv_path, ["3", "4"]) == 2
    remaining = [r["No"] for r in _read_rows(csv_path)]
    assert remaining == ["1", "2", "5"]  # ids stable, gaps fine
    assert "3" not in review.load_reviews(csv_path)
    assert review.remove_hand(csv_path, ["99"]) == 0


def test_group_hand_rows_orders_and_singletons() -> None:
    rows = [
        {"No": "1", "hand_id": "h1", "sequence_index": "2"},
        {"No": "2", "hand_id": "h1", "sequence_index": "1"},
        {"No": "3", "hand_id": "", "sequence_index": ""},
    ]
    groups = review.group_hand_rows(rows)
    assert [k for k, _ in groups] == ["h1", "__row_3"]
    # legs sorted by sequence_index within the hand
    assert [r["No"] for r in groups[0][1]] == ["2", "1"]


def test_filter_hand_rows_modes_and_renumbering(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_batch(csv_path)
    rows = _read_rows(csv_path)
    review.save_reviews_bulk(csv_path, ["1", "2"], "approved")
    review.save_reviews_bulk(csv_path, ["3", "4"], "rejected")
    review.save_review(csv_path, "1", "approved", "",
                       explanation="reviewer text")
    grades = review.load_reviews(csv_path)

    out, kept, total = review.filter_hand_rows(rows, grades, "all")
    assert (kept, total, len(out)) == (3, 3, 5)
    assert [r["No"] for r in out] == ["1", "2", "3", "4", "5"]

    out, kept, total = review.filter_hand_rows(rows, grades, "drop_rejected")
    assert (kept, total) == (2, 3)
    # h2 gone WHOLE (never a partial hand); Nos renumbered contiguously.
    assert [r["hand_id"] for r in out] == ["h1", "h1", ""]
    assert [r["No"] for r in out] == ["1", "2", "3"]
    # The reviewer-edited explanation ships in place of the generated one.
    assert out[0]["Answer Explanation"] == "reviewer text"
    assert rows[0]["Answer Explanation"] == "e1"  # input untouched

    out, kept, total = review.filter_hand_rows(rows, grades, "approved_only")
    assert (kept, total) == (1, 3)  # the ungraded standalone is excluded too
    assert [r["hand_id"] for r in out] == ["h1", "h1"]

    try:
        review.filter_hand_rows(rows, grades, "bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_meta_question_for_leg_joins_preflop_legs() -> None:
    """REGRESSION: pack PREFLOP legs' solver_reference ends in the node id
    (not the combo), so the old node/combo join missed them and the Review
    page hid their Layer-7 panels. The (hand_id, sequence_index) join must
    find every leg kind; blank hand_id (standalone) returns None."""
    meta = {"questions": [
        {"hand_id": "h1", "sequence_index": 1, "street": "preflop",
         "revise": {"status": "clean"}},
        {"hand_id": "h1", "sequence_index": 2, "street": "flop"},
        {"hand_id": "", "sequence_index": "", "street": "flop"},
    ]}
    q = review.meta_question_for_leg(meta, hand_id="h1", sequence_index="1")
    assert q is not None and q["street"] == "preflop"
    assert q["revise"]["status"] == "clean"
    assert review.meta_question_for_leg(meta, hand_id="h1", sequence_index="2")["street"] == "flop"
    assert review.meta_question_for_leg(meta, hand_id="", sequence_index="1") is None
    assert review.meta_question_for_leg(None, hand_id="h1", sequence_index="1") is None
    assert review.meta_question_for_leg(meta, hand_id="h9", sequence_index="1") is None
