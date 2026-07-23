"""Smoke test: the admin postflop Generate page renders without crashing.

Drives the real Streamlit app via ``AppTest`` (no browser): navigate to
Generate, switch to Postflop mode, and assert no exception. Catches UI-level
breakage (bad widget calls, missing imports, a crash in the solve picker) that
the pipeline unit tests don't see. Skips cleanly if Streamlit's test harness
isn't importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_APP = Path(__file__).resolve().parents[1] / "admin_panel" / "app.py"


def test_postflop_generate_page_renders() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.session_state["nav_page"] = "Generate"
    at.session_state["generate_mode"] = "Postflop"
    at.run()
    assert not at.exception, at.exception
    # The picker should have rendered the page heading (a solve found, or the
    # friendly "no solves" info) -- either way, no crash and some markdown.
    assert at.title  # "Generate questions"


def test_postflop_review_page_renders() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.session_state["nav_page"] = "Postflop Review"
    at.run()
    assert not at.exception, at.exception


def test_prompt_page_postflop_mode_renders() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.session_state["nav_page"] = "Prompt"
    at.session_state["prompt_mode"] = "Postflop (editable)"
    at.run()
    assert not at.exception, at.exception


def test_hand_level_keep_button_grades_all_legs() -> None:
    """One click on '✅ Keep hand' grades EVERY leg of that hand (the
    hand-level review workflow). Full AppTest wiring check on a synthetic
    2-hand batch staged into the real batches dir, cleaned up after. The
    pure logic lives in admin_panel/review.py (browserless tests in
    test_hand_level_review.py); this pins the UI wiring."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    import csv as _csv
    import json
    import os
    import time

    batch_dir = Path(__file__).resolve().parents[1] / "test_output" / "postflop_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = batch_dir / "zz_apptest_hand_review.csv"
    cols = ["No", "hand_id", "sequence_index", "Question", "Correct Answer",
            "Answer Explanation", "User Cards", "Cards on Table", "Context",
            "Difficulty Rating", "Notes"]
    rows = [
        {"No": str(i), "hand_id": hid, "sequence_index": seq,
         "Question": f"q{i}", "Correct Answer": "Call",
         "Answer Explanation": f"e{i}", "User Cards": "As Ks",
         "Cards on Table": "", "Context": "test", "Difficulty Rating": "1000",
         "Notes": f"Auto. Node: ref/{i}"}
        for i, (hid, seq) in enumerate(
            [("hA", "1"), ("hA", "2"), ("hB", "1"), ("hB", "2")], start=1
        )
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # newest mtime -> the review page's selectbox defaults to this batch
    late = time.time() + 60
    os.utime(csv_path, (late, late))
    try:
        at = AppTest.from_file(str(_APP), default_timeout=120)
        at.session_state["nav_page"] = "Postflop Review"
        at.run()
        assert not at.exception, at.exception
        keeps = [b for b in at.button if str(b.key or "").startswith("hand_keep::")]
        assert len(keeps) == 2, f"expected 2 hand cards, got {len(keeps)}"
        target = next(b for b in keeps if b.key.endswith("::hA"))
        target.click()
        at.run()
        assert not at.exception, at.exception
        grades = json.loads(
            csv_path.with_suffix(".review.json").read_text(encoding="utf-8")
        )
        assert {k for k, v in grades.items() if v["status"] == "approved"} == {"1", "2"}
    finally:
        csv_path.unlink(missing_ok=True)
        csv_path.with_suffix(".review.json").unlink(missing_ok=True)


def test_approved_pool_remove_and_clear_controls() -> None:
    """The postflop approved pool has remove controls (July 2026, user ask:
    there was previously NO way to un-approve anything on the Postflop
    Review page). Wiring check on a synthetic approved hand staged into the
    real batches dir: the clear-all button renders, the per-question 🗑
    renders, and clicking 🗑 on one LEG un-approves the WHOLE hand (pure
    logic in review.approved_removal_group, browserless-tested in
    test_admin_panel_review.py). Clear-all itself is NOT clicked here -- it
    sweeps every batch in the real dir, which may hold real grades."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    import csv as _csv
    import json
    import os
    import time

    batch_dir = Path(__file__).resolve().parents[1] / "test_output" / "postflop_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = batch_dir / "zz_apptest_approved_pool.csv"
    cols = ["No", "hand_id", "sequence_index", "Question", "Correct Answer",
            "Answer Explanation", "User Cards", "Cards on Table", "Context",
            "Difficulty Rating", "board_texture", "Notes"]
    rows = [
        {"No": str(i), "hand_id": "hA", "sequence_index": str(i),
         "Question": f"q{i}", "Correct Answer": "Call",
         "Answer Explanation": f"e{i}", "User Cards": "As Ks",
         "Cards on Table": "", "Context": "test", "Difficulty Rating": "1000",
         "board_texture": "", "Notes": f"Auto. Node: zzpool/{i}"}
        for i in (1, 2)
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    sidecar = csv_path.with_suffix(".review.json")
    sidecar.write_text(json.dumps({
        "1": {"status": "approved", "note": ""},
        "2": {"status": "approved", "note": ""},
    }), encoding="utf-8")
    late = time.time() + 60
    os.utime(csv_path, (late, late))
    try:
        at = AppTest.from_file(str(_APP), default_timeout=120)
        at.session_state["nav_page"] = "Postflop Review"
        at.run()
        assert not at.exception, at.exception
        # Clear-all entry point renders (2-step confirm; not clicked here).
        assert any(str(b.key or "") == "pf_clear_approved" for b in at.button)
        # Per-question removal: click leg 1's 🗑 -> the WHOLE hand un-approves.
        dels = [
            b for b in at.button
            if str(b.key or "").startswith(f"pf_appr_del_{csv_path.name}_")
        ]
        assert len(dels) == 2, f"expected 2 removal buttons, got {len(dels)}"
        next(b for b in dels if b.key.endswith("_1")).click()
        at.run()
        assert not at.exception, at.exception
        grades = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        assert not any(
            g.get("status") == "approved" for g in grades.values()
        ), f"hand not fully un-approved: {grades}"
    finally:
        csv_path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
