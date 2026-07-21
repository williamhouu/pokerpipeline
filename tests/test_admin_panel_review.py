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
from pipeline.provenance import build_notes  # noqa: E402


def _notes(ref: str) -> str:
    """Notes carrying `ref` in the Node: field -- the July-2026 home of the
    node reference (was the dropped solver_reference column)."""
    return build_notes("t.", node_ref=ref)


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
        meta, user_cards="Qh, Qd", node_reference="pack/BTN/NODE_B"
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
            meta, user_cards="Ah, Ks", node_reference="pack/BTN/NODE_Z"
        )
        is None
    )
    # Right node, wrong hand.
    assert (
        review.meta_question_for(
            meta, user_cards="Qh, Qd", node_reference="pack/BTN/NODE_A"
        )
        is None
    )


def test_meta_question_for_handles_missing_questions() -> None:
    assert (
        review.meta_question_for(
            {}, user_cards="Ah, Ks", node_reference="pack/BTN/NODE_A"
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
_COLS = ["No", "User Cards", "Notes", "Correct Answer"]


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
            {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check"},
            {"No": "2", "User Cards": "K,K", "Notes": _notes("p/BB/n2"), "Correct Answer": "Fold"},
            {"No": "3", "User Cards": "Q,Q", "Notes": _notes("p/BB/n3"), "Correct Answer": "Call"},
        ],
    )
    review.save_review(b, 1, "approved", "")
    review.save_review(b, 2, "rejected", "")
    # No 3 left ungraded.
    fields, rows = review.collect_approved_rows(tmp_path)
    assert fields == _COLS
    assert [r["No"] for r in rows] == ["1"]


def test_collect_approved_dedupes_across_batches(tmp_path: Path) -> None:
    spot = {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check"}
    older = tmp_path / "plo_20260601_120000.csv"
    newer = tmp_path / "plo_20260602_120000.csv"
    _write_batch(older, [spot])
    _write_batch(newer, [spot])
    review.save_review(older, 1, "approved", "")
    review.save_review(newer, 1, "approved", "")
    _, rows = review.collect_approved_rows(tmp_path)
    assert len(rows) == 1  # same (solver_reference, User Cards) -> one copy


def test_collect_approved_includes_compare_when_approved(tmp_path: Path) -> None:
    # A question finalized on the Compare page (a compare_*.csv) feeds the
    # SAME pool as Review-page approvals.
    cmp = tmp_path / "compare_20260601_A.csv"
    _write_batch(cmp, [
        {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check"}
    ])
    review.save_review(cmp, 1, "approved", "")
    _, rows = review.collect_approved_rows(tmp_path)
    assert [r["No"] for r in rows] == ["1"]


def test_collect_approved_can_exclude_by_prefix(tmp_path: Path) -> None:
    cmp = tmp_path / "compare_20260601_A.csv"
    _write_batch(cmp, [
        {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check"}
    ])
    review.save_review(cmp, 1, "approved", "")
    _, rows = review.collect_approved_rows(tmp_path, exclude_prefix="compare_")
    assert rows == []


def test_save_review_with_explanation_round_trip(tmp_path: Path) -> None:
    b = tmp_path / "compare_20260610_A.csv"
    _write_batch(b, [
        {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Call"}
    ])
    review.save_review(b, 1, "approved", "finalized", explanation="Edited prose.")
    assert review.load_reviews(b)["1"]["explanation"] == "Edited prose."
    # Re-saving without an explanation clears the override.
    review.save_review(b, 1, "approved", "finalized")
    assert "explanation" not in review.load_reviews(b)["1"]


def test_collect_approved_applies_explanation_override(tmp_path: Path) -> None:
    cols = [*_COLS, "Answer Explanation"]
    b = tmp_path / "compare_20260610_A.csv"
    with b.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows([
            {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"),
             "Correct Answer": "Call", "Answer Explanation": "Generated."},
            {"No": "2", "User Cards": "K,K", "Notes": _notes("p/BB/n2"),
             "Correct Answer": "Fold", "Answer Explanation": "Generated."},
        ])
    review.save_review(b, 1, "approved", "", explanation="Edited.")
    review.save_review(b, 2, "approved", "")
    _, rows = review.collect_approved_rows(tmp_path)
    by_no = {r["No"]: r for r in rows}
    assert by_no["1"]["Answer Explanation"] == "Edited."
    assert by_no["2"]["Answer Explanation"] == "Generated."  # no edit -> untouched
    # The batch CSV itself is never mutated; the override lives in the sidecar.
    with b.open(newline="", encoding="utf-8-sig") as fh:
        raw = list(csv.DictReader(fh))
    assert raw[0]["Answer Explanation"] == "Generated."


def test_collect_approved_sources_returns_provenance(tmp_path: Path) -> None:
    b = tmp_path / "plo_20260601_120000.csv"
    _write_batch(
        b,
        [
            {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check"},
            {"No": "2", "User Cards": "K,K", "Notes": _notes("p/BB/n2"), "Correct Answer": "Fold"},
        ],
    )
    review.save_review(b, 1, "approved", "")
    sources = review.collect_approved_sources(tmp_path)
    assert len(sources) == 1
    csv_path, no, row = sources[0]
    assert csv_path == b
    assert no == "1"
    assert row["User Cards"] == "A,A"


def test_clear_all_approved_removes_approved_only(tmp_path: Path) -> None:
    b = tmp_path / "plo_20260601_120000.csv"
    _write_batch(
        b,
        [
            {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check"},
            {"No": "2", "User Cards": "K,K", "Notes": _notes("p/BB/n2"), "Correct Answer": "Fold"},
        ],
    )
    review.save_review(b, 1, "approved", "")
    review.save_review(b, 2, "rejected", "")
    cleared = review.clear_all_approved(tmp_path)
    assert cleared == 1
    assert review.collect_approved_rows(tmp_path) == ([], [])
    # The rejected grade is untouched.
    assert review.load_reviews(b)["2"]["status"] == "rejected"


def test_clear_all_approved_clears_duplicate_spots_in_every_batch(tmp_path: Path) -> None:
    # A spot approved in two batches dedupes to one in the pool, but clear-all
    # must un-approve BOTH so it can't reappear.
    spot = {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check"}
    older = tmp_path / "plo_20260601_120000.csv"
    newer = tmp_path / "plo_20260602_120000.csv"
    _write_batch(older, [spot])
    _write_batch(newer, [spot])
    review.save_review(older, 1, "approved", "")
    review.save_review(newer, 1, "approved", "")
    assert review.clear_all_approved(tmp_path) == 2
    assert review.collect_approved_rows(tmp_path) == ([], [])


def test_approved_rows_to_csv_round_trips(tmp_path: Path) -> None:
    rows = [
        {"No": "1", "User Cards": "A,A", "Notes": _notes("p/BB/n1"), "Correct Answer": "Check", "extra": "ignored"},
    ]
    text = review.approved_rows_to_csv(_COLS, rows)
    parsed = list(csv.DictReader(text.splitlines()))
    assert parsed[0]["User Cards"] == "A,A"
    assert "extra" not in parsed[0]  # keys outside fieldnames are dropped


# --- load_failures / promote_failure (routed-to-human-review recovery) -----
_FAIL_COLS = [
    "No", "User Cards", "Notes", "Correct Answer",
    "validation_status", "Answer Explanation",
]


def _write_fail_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FAIL_COLS)
        writer.writeheader()
        writer.writerows(rows)


def _write_meta(csv_path: Path, failures: list[dict]) -> None:
    meta = {"prompt_name": "p", "questions": [], "failures": failures}
    review.batch_meta_path(csv_path).write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _failure(row: dict | None = None, **over: object) -> dict:
    f: dict = {
        "node_id": "UTG_120%_HJ_Call_CO_Call_BTN_AllIn_HJ_Call_CO_decision",
        "hand_class": "QQ",
        "hero_position": "CO",
        "options": ["Always Fold", "Mostly Fold", "Mostly Call", "Always Call"],
        "correct_answer": "Mostly Call",
        "failed_explanation": "Mostly call: the price clears your equity.",
        "error_message": "ALL-IN spot framing ... Spot routed to human review.",
        "row": row,
    }
    f.update(over)
    return f


def test_load_failures_missing_meta_returns_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_fail_csv(csv_path, [])
    assert review.load_failures(csv_path) == []


def test_load_failures_reads_meta(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_fail_csv(csv_path, [])
    _write_meta(csv_path, [_failure()])
    out = review.load_failures(csv_path)
    assert len(out) == 1
    assert out[0]["hand_class"] == "QQ"


def test_promote_failure_appends_row_and_approves(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_fail_csv(csv_path, [{
        "No": "1", "User Cards": "A,A", "Notes": _notes("p/CO/n0"),
        "Correct Answer": "Call", "validation_status": "auto_approved",
        "Answer Explanation": "good",
    }])
    row = {
        "No": "0", "User Cards": "Q,Q", "Notes": _notes("p/CO/qq"),
        "Correct Answer": "Mostly Call", "validation_status": "needs_review",
        "Answer Explanation": "Mostly call: the price clears.",
    }
    fail = _failure(row=row)

    ok, msg = review.promote_failure(csv_path, fail)
    assert ok, msg
    # Appended with a fresh No (max 1 -> 2) and validation_status flipped to
    # approved.
    assert _read_nos(csv_path) == ["1", "2"]
    with csv_path.open(encoding="utf-8-sig") as fh:
        added = [r for r in csv.DictReader(fh) if r["User Cards"] == "Q,Q"][0]
    assert added["No"] == "2"
    assert added["validation_status"] == "approved"
    # Graded approved in the sidecar, and flows into the approved pool with the
    # kept explanation.
    assert review.load_reviews(csv_path)["2"]["status"] == "approved"
    _f, pool = review.collect_approved_rows(tmp_path)
    assert any(r["User Cards"] == "Q,Q" for r in pool)


def test_promote_failure_is_idempotent(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_fail_csv(csv_path, [])
    row = {
        "No": "0", "User Cards": "Q,Q", "Notes": _notes("p/CO/qq"),
        "Correct Answer": "Mostly Call", "validation_status": "needs_review",
        "Answer Explanation": "Mostly call.",
    }
    fail = _failure(row=row)
    ok1, _m1 = review.promote_failure(csv_path, fail)
    ok2, msg2 = review.promote_failure(csv_path, fail)
    assert ok1
    assert not ok2
    assert "already" in msg2.lower()
    assert _read_nos(csv_path).count("1") == 1  # added exactly once


def test_promote_failure_without_row_returns_false(tmp_path: Path) -> None:
    csv_path = tmp_path / "b.csv"
    _write_fail_csv(csv_path, [])
    ok, msg = review.promote_failure(csv_path, _failure(row=None))
    assert not ok
    assert "nothing to promote" in msg.lower()


# --- revise_summary_line (the experimental 4-call banner) ------------------
def test_revise_summary_none_when_revise_pass_off() -> None:
    assert review.revise_summary_line(None) is None
    assert review.revise_summary_line({"run_settings": {"revise_pass": False}}) is None
    assert review.revise_summary_line({"run_settings": {}}) is None


def test_revise_summary_reports_counts() -> None:
    meta = {
        "run_settings": {"revise_pass": True, "final_audit": True},
        "counters": {
            "revise_flagged": 3, "revise_fixed": 1,
            "revise_discarded": 1, "revise_unchanged": 1,
        },
    }
    line = review.revise_summary_line(meta)
    assert line is not None
    assert "flagged **3**" in line
    assert "1 auto-fixed" in line
    assert "discarded" in line and "unchanged" in line
    assert "Final audit: on" in line


def test_revise_summary_zero_flagged() -> None:
    meta = {
        "run_settings": {"revise_pass": True, "final_audit": False},
        "counters": {"revise_flagged": 0, "revise_fixed": 0,
                     "revise_discarded": 0, "revise_unchanged": 0},
    }
    line = review.revise_summary_line(meta)
    assert line is not None and "flagged 0 questions" in line
    assert "Final audit: off" in line


def test_revise_summary_old_batch_without_counters() -> None:
    # revise_pass on, but counters predate the revise_* keys -> generic note,
    # not a misleading "0 flagged".
    meta = {"run_settings": {"revise_pass": True, "final_audit": True},
            "counters": {"questions_written": 4}}
    line = review.revise_summary_line(meta)
    assert line is not None and "after this feature update" in line


# --- one-click "approve all fully-clean" (bulk approve) --------------------

def test_row_is_fully_clean_requires_audit_to_have_run() -> None:
    # Blank claim_check cell = the audit never ran -> NOT eligible, even with
    # a spotless meta record. (A never-audited row must not be auto-approved.)
    assert review.row_is_fully_clean("", {}) is False
    assert review.row_is_fully_clean("   ", None) is False


def test_row_is_fully_clean_true_when_ran_and_all_green() -> None:
    assert review.row_is_fully_clean("[]", {}) is True
    assert review.row_is_fully_clean("[]", None) is True
    # Absent flag fields read as clean (a pipeline without cross-check/sanity).
    assert review.row_is_fully_clean("[]", {"revise": {"status": "clean"}}) is True


def test_row_is_fully_clean_false_when_claim_checker_flagged() -> None:
    cell = json.dumps([{"claim": "you block the nut flush", "problem": "reversed"}])
    assert review.row_is_fully_clean(cell, {}) is False


def test_row_is_fully_clean_false_on_any_other_flag_source() -> None:
    for field in (
        "validator_warnings",
        "sanity_check_issues",
        "cross_check_issues",
    ):
        assert review.row_is_fully_clean("[]", {field: ["something"]}) is False, field


def test_row_is_fully_clean_claim_check_issues_alone_do_not_disqualify() -> None:
    # In auto-fix mode claim_check_issues holds the PRE-rewrite gate flags, so a
    # fixed-then-clean row still carries them; they must NOT disqualify on their
    # own when the shipped verdict (column or revise) is clean.
    assert review.row_is_fully_clean("[]", {"claim_check_issues": ["old flag"]}) is True


def test_row_is_fully_clean_false_on_malformed_cell() -> None:
    assert review.row_is_fully_clean("not json", {}) is False
    assert review.row_is_fully_clean("{}", {}) is False  # dict, not empty list


# --- auto-fix (revise_pass) awareness: the shipped verdict lives in `revise` ---

def test_row_is_fully_clean_revise_clean_with_empty_column() -> None:
    # THE REGRESSION: a "clean" revise row leaves claim_check BLANK (the gate ran
    # but its result isn't written to the column in revise mode). It is green.
    assert review.row_is_fully_clean("", {"revise": {"status": "clean"}}) is True


def test_row_is_fully_clean_revise_fixed_final_audit_clean() -> None:
    # Rewritten and the 4th-call audit found nothing -> green.
    assert review.row_is_fully_clean(
        "[]", {"revise": {"status": "fixed", "final_audit_issues": []}}
    ) is True


def test_row_is_fully_clean_revise_fixed_with_final_audit_flags_is_not_green() -> None:
    # Rewritten but the 4th-call audit still flagged something (the blue box).
    assert review.row_is_fully_clean(
        "[]", {"revise": {"status": "fixed", "final_audit_issues": ["still off"]}}
    ) is False


def test_row_is_fully_clean_revise_discarded_or_unchanged_is_not_green() -> None:
    for status in ("discarded", "unchanged"):
        assert review.row_is_fully_clean(
            "[]", {"revise": {"status": status}}
        ) is False, status


def test_fully_clean_ungraded_counts_both_clean_and_fixed_rows() -> None:
    # Mirrors the real 12q auto-fix batch: a "clean" row (blank column) and a
    # "fixed→final-audit-clean" row are BOTH eligible; "fixed-with-flags" is not.
    rows = [
        {"No": "2", "claim_check": "[]"},   # fixed, final audit clean
        {"No": "8", "claim_check": ""},     # clean, blank column
        {"No": "1", "claim_check": "[{}]"}, # fixed but final audit still flags
    ]
    qrecs = {
        "2": {"revise": {"status": "fixed", "final_audit_issues": []}},
        "8": {"revise": {"status": "clean"}},
        "1": {"revise": {"status": "fixed", "final_audit_issues": ["x", "y"]}},
    }
    nos = review.fully_clean_ungraded_nos(
        rows, {}, qrec_for=lambda r: qrecs.get(str(r["No"]))
    )
    assert nos == ["2", "8"]


def test_fully_clean_ungraded_skips_graded_and_flagged() -> None:
    rows = [
        {"No": "1", "claim_check": "[]"},                       # clean, ungraded -> IN
        {"No": "2", "claim_check": ""},                          # not audited -> out
        {"No": "3", "claim_check": "[]"},                        # clean but graded -> out
        {"No": "4", "claim_check": json.dumps([{"claim": "x"}])},# flagged -> out
        {"No": "5", "claim_check": "[]"},                        # clean but soft-flagged -> out
        {"No": "6", "claim_check": "[]"},                        # clean, ungraded -> IN
    ]
    reviews = {"3": {"status": "approved", "note": ""}}
    qrecs = {"5": {"validator_warnings": ["position wording"]}}
    nos = review.fully_clean_ungraded_nos(
        rows, reviews, qrec_for=lambda r: qrecs.get(str(r["No"]))
    )
    assert nos == ["1", "6"]


def test_bulk_approve_writes_and_is_idempotent(tmp_path: Path) -> None:
    csv = tmp_path / "b.csv"
    review.save_review(csv, "3", "rejected", "human said no")
    added = review.bulk_approve(csv, ["1", "6", "3"])  # 3 already graded -> skipped
    assert added == 2
    saved = review.load_reviews(csv)
    assert saved["1"]["status"] == "approved"
    assert saved["6"]["status"] == "approved"
    assert saved["3"]["status"] == "rejected"  # human decision untouched
    # Re-running approves nothing new.
    assert review.bulk_approve(csv, ["1", "6"]) == 0
