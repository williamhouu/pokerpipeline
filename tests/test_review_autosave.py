"""The Review page's edit-persistence invariant.

WHY THIS FILE EXISTS: the "edit an answer explanation, navigate to the next
question, click back, and the edit is gone" bug has recurred repeatedly. Each
prior fix lived in the fragile Streamlit layer (an ``on_change`` callback that
has to fire before a navigation button short-circuits the rerun), was verified
by hand in the browser, and had NO automated coverage -- so the next refactor of
the Review page silently reintroduced it.

``admin_panel.app._flush_review_edit`` makes persistence robust-by-construction:
it writes the pending ``session_state`` edit to the CSV on navigation, regardless
of callback timing. These tests pin that contract at a layer that needs no
browser, so a reintroduction is caught by CI, not by the user weeks later.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel import app as admin_app  # noqa: E402
from admin_panel import review  # noqa: E402


def _write(csv_path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_flush_persists_pending_edit_on_navigation(tmp_path, monkeypatch) -> None:
    """The exact bug: an in-flight edit sitting in session_state is written to
    the CSV when the user navigates away -- not dependent on on_change firing."""
    csv_path = tmp_path / "b.csv"
    fields = ["No", "Answer Explanation", "Difficulty Rating"]
    _write(csv_path, [
        {"No": "1", "Answer Explanation": "old one", "Difficulty Rating": "1500"},
        {"No": "2", "Answer Explanation": "old two", "Difficulty Rating": "1800"},
    ], fields)

    # Streamlit copies the user's just-typed value into session_state BEFORE the
    # script body runs; simulate that state at the moment a nav button is clicked.
    fake_state = {
        f"review_expl::{csv_path.name}::1": "EDITED one",
        f"review_diff::{csv_path.name}::1": 2600,
    }
    monkeypatch.setattr(admin_app.st, "session_state", fake_state, raising=False)

    admin_app._flush_review_edit(csv_path, "1")

    rows = {r["No"]: r for r in _read(csv_path)}
    assert rows["1"]["Answer Explanation"] == "EDITED one"
    assert rows["1"]["Difficulty Rating"] == "2600"
    assert rows["2"]["Answer Explanation"] == "old two"  # other rows untouched


def test_flush_is_noop_when_no_pending_edit(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "b.csv"
    _write(csv_path, [
        {"No": "1", "Answer Explanation": "old", "Difficulty Rating": "1500"},
    ], ["No", "Answer Explanation", "Difficulty Rating"])
    monkeypatch.setattr(admin_app.st, "session_state", {}, raising=False)

    admin_app._flush_review_edit(csv_path, "1")

    assert _read(csv_path)[0]["Answer Explanation"] == "old"


def test_unchanged_update_does_not_rewrite_the_file(tmp_path) -> None:
    """The unconditional flush must be free when nothing changed: an unchanged
    write returns True WITHOUT rewriting, so mtime doesn't bump (which would
    invalidate the read cache on every navigation)."""
    csv_path = tmp_path / "b.csv"
    _write(csv_path, [{"No": "1", "Answer Explanation": "same"}],
           ["No", "Answer Explanation"])
    before_bytes = csv_path.read_bytes()
    before_mtime = csv_path.stat().st_mtime_ns

    assert review.update_explanation(csv_path, "1", "same") is True

    assert csv_path.read_bytes() == before_bytes
    assert csv_path.stat().st_mtime_ns == before_mtime  # never rewritten


def test_changed_update_still_rewrites(tmp_path) -> None:
    csv_path = tmp_path / "b.csv"
    _write(csv_path, [{"No": "1", "Answer Explanation": "same"}],
           ["No", "Answer Explanation"])
    assert review.update_explanation(csv_path, "1", "different") is True
    assert _read(csv_path)[0]["Answer Explanation"] == "different"


def test_md_lines_escapes_dollar_amounts() -> None:
    """st.markdown treats $...$ as inline LaTeX, so two dollar amounts in one
    line ("opens to $6 ... 3-bets to $24") rendered the span between them as
    math. _md_lines must escape $ (and still preserve line breaks)."""
    out = admin_app._md_lines("You open to $6.\nThe Small Blind 3-bets to $24.")
    assert "\\$6" in out and "\\$24" in out
    assert "  \n" in out  # newline -> markdown hard break


def test_flush_postflop_key_prefix(tmp_path, monkeypatch) -> None:
    """The postflop Review page namespaces its widget keys differently; the
    shared flush must honour key_prefix so postflop submits save too."""
    csv_path = tmp_path / "pf.csv"
    _write(csv_path, [
        {"No": "7", "Answer Explanation": "old", "Difficulty Rating": "1200"},
    ], ["No", "Answer Explanation", "Difficulty Rating"])
    fake_state = {
        f"postflop_review_expl::{csv_path.name}::7": "EDITED pf",
        f"postflop_review_diff::{csv_path.name}::7": 1900,
    }
    monkeypatch.setattr(admin_app.st, "session_state", fake_state, raising=False)

    admin_app._flush_review_edit(csv_path, "7", key_prefix="postflop_review")

    row = _read(csv_path)[0]
    assert row["Answer Explanation"] == "EDITED pf"
    assert row["Difficulty Rating"] == "1900"


def test_review_pages_have_no_racy_nav_buttons() -> None:
    """Structural guard for the EDIT-LOSS INVARIANT: the Review pages' nav /
    grade / remove controls must be form_submit_buttons (atomic value
    delivery), never plain st.button (which races the editor's blur commit --
    the recurring lost-edit bug). This greps the source so a refactor that
    sneaks a navigating st.button back in fails CI, not the user."""
    import inspect

    for fn in (admin_app.render_review_page,
               admin_app._render_postflop_question_card):
        src = inspect.getsource(fn)
        assert "form_submit_button" in src, fn.__name__
        for label in ("◀ Prev", "Next ▶", "✅ Approve", "❌ Reject", "🗑"):
            for line in src.splitlines():
                if label in line and "st.button" in line.replace(
                    "form_submit_button", ""
                ):
                    raise AssertionError(
                        f"{fn.__name__} renders {label!r} via plain st.button -- "
                        "must be a form_submit_button (EDIT-LOSS INVARIANT)"
                    )


def test_ranges_conditional_toggle_lives_outside_the_form() -> None:
    """Structural guard for the RERUN INVARIANT (July 2026): a widget inside
    an st.form only takes effect on the NEXT submit click, so the ranges
    panel's Conditional-view toggle (a pure viewer control) must render
    OUTSIDE the review card's form -- inside it, the toggle flipped visually
    but the grids never re-rendered ("the toggle doesn't work")."""
    import inspect

    card_src = inspect.getsource(admin_app._render_postflop_question_card)
    assert "st.toggle" not in card_src, (
        "reactive viewer widget inside the review card's st.form -- it will "
        "appear dead until a submit click (RERUN INVARIANT)"
    )
    assert "_render_postflop_ranges_panel(" in card_src
    panel_src = inspect.getsource(admin_app._render_postflop_ranges_panel)
    assert "st.toggle" in panel_src
    assert "st.form(" not in panel_src


def test_read_csv_cache_respects_mtime(tmp_path) -> None:
    """THE true root cause of the recurring 'my edit vanished' bug: the CSV
    read cache took `_mtime`, and st.cache_data EXCLUDES underscore-prefixed
    params from the cache key -- so the first parse of a batch was served for
    the entire session and every saved edit looked lost on navigate-back.
    This pins the contract: same path + newer mtime => fresh content."""
    csv_path = tmp_path / "b.csv"
    _write(csv_path, [{"No": "1", "Answer Explanation": "before"}],
           ["No", "Answer Explanation"])
    df1 = admin_app._read_csv_cached(str(csv_path), csv_path.stat().st_mtime)
    assert df1.iloc[0]["Answer Explanation"] == "before"

    review.update_explanation(csv_path, "1", "after")
    df2 = admin_app._read_csv_cached(str(csv_path), csv_path.stat().st_mtime)
    assert df2.iloc[0]["Answer Explanation"] == "after", (
        "stale cache: _read_csv_cached ignored the mtime -- if this fails, "
        "check that the mtime parameter has NOT been renamed with a leading "
        "underscore (st.cache_data skips underscore-prefixed params)"
    )


def test_blur_save_replaced_the_save_button():
    """REGRESSION PIN (July 22 2026, user ask): explanation + difficulty
    edits save when the user clicks OUT of the box (live widgets with a
    compare-and-write, the PLO page's proven pattern) -- there is no Save
    button to forget. The nav/grade controls stay form submits, and
    _flush_review_edit still runs in the handlers as belt-and-suspenders.
    If a refactor reintroduces an in-form editor with a Save button, edits
    are one missed click from vanishing again."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "admin_panel" / "app.py"
    ).read_text(encoding="utf-8")
    assert "Save edits" not in src
    assert "_save_clicked" not in src
    # Both cards blur-save through the shared pure writers.
    assert src.count("review.update_explanation(csv_path, no, _live_expl)") >= 2
    assert src.count("review.update_difficulty(csv_path, no, str(int(_live_diff)))") >= 2
    # The belt-and-suspenders flush survives on both pages.
    assert src.count("_flush_review_edit(csv_path, no") >= 2
