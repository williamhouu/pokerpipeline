"""End-to-end verification of the Review editor save path (headless, no browser).

Drives the REAL admin app via Streamlit's official AppTest harness: opens the
Review page, edits the explanation, clicks the real form-submit "Next", comes
back with "Prev", and checks BOTH that the CSV was written AND that the editor
displays the edit. This is the exact user gesture behind the recurring "my
edit vanished" bug (two compounding root causes, both fixed 2026-07-04: the
plain-button blur race -> per-question st.forms; and _read_csv_cached's
underscore-prefixed mtime param -> stale display for the whole session).

Run after ANY change to the Review pages or the CSV cache (takes ~3 min --
that's why it lives here and not in the default pytest suite; the fast unit
tests in tests/test_review_autosave.py pin both root causes in CI):

    venv/bin/python scripts/verify_review_editor_e2e.py

Expects the QC IMP200 batch to exist; edits are restored afterwards.
Prints "VERDICT: SAVE+DISPLAY OK" on success."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path("/Users/zackdelson/Documents/Bandits code stuff/pokerpipeline")
sys.path.insert(0, str(REPO))

from streamlit.testing.v1 import AppTest

BATCH = "QC IMP200_20260703.csv"
CSV_PATH = REPO / "test_output" / "preflop_batches" / BATCH

at = AppTest.from_file(str(REPO / "admin_panel" / "app.py"), default_timeout=180)
at.run()
print("initial run ok; exceptions:", [str(e.value)[:200] for e in at.exception])

# Navigate to the Review page via the sidebar radio.
radio = at.sidebar.radio[0]
print("radio options sample:", list(radio.options)[:6], "...")
radio.set_value("Review").run()
print("on Review page; exceptions:", [str(e.value)[:200] for e in at.exception])

# Pick the batch.
pick = next(sb for sb in at.selectbox if sb.key == "review_batch_pick")
pick.set_value(BATCH).run()
print("batch picked; exceptions:", [str(e.value)[:300] for e in at.exception])

# Find the editor and note its question No.
ta = next(t for t in at.text_area if t.key and t.key.startswith("review_expl::"))
no1 = ta.key.rsplit("::", 1)[-1]
orig = ta.value
print(f"question #{no1}, editor starts with: {orig[:50]!r}")

# EDIT the explanation, then click the real Next form-submit.
EDIT = orig + "\n\n[APPTEST EDIT MARKER]"
ta.set_value(EDIT)
buttons = {b.label: b for b in at.button}
print("buttons present:", sorted(buttons)[:10])
buttons["Next ▶"].click().run()
print("after Next; exceptions:", [str(e.value)[:300] for e in at.exception])

# Verify the CSV was written.
rows = {r["No"]: r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8-sig"))}
in_csv = "[APPTEST EDIT MARKER]" in rows[no1]["Answer Explanation"]
print(f"CSV contains the edit after Next: {in_csv}")

# We should now be on question 2; go back with Prev.
ta2 = next(t for t in at.text_area if t.key and t.key.startswith("review_expl::"))
print(f"now on #{ta2.key.rsplit('::', 1)[-1]}")
buttons = {b.label: b for b in at.button}
buttons["◀ Prev"].click().run()
print("after Prev; exceptions:", [str(e.value)[:300] for e in at.exception])

ta3 = next(t for t in at.text_area if t.key and t.key.startswith("review_expl::"))
back_no = ta3.key.rsplit("::", 1)[-1]
shows_edit = "[APPTEST EDIT MARKER]" in (ta3.value or "")
print(f"back on #{back_no}; editor SHOWS the edit: {shows_edit}")
print("VERDICT:", "SAVE+DISPLAY OK" if (in_csv and shows_edit and back_no == no1)
      else f"BROKEN (in_csv={in_csv}, shows_edit={shows_edit}, back_no={back_no})")

# Clean up: restore the original text so the QC batch isn't polluted.
from admin_panel import review as _rev
_rev.update_explanation(CSV_PATH, no1, orig)
print("csv restored")
