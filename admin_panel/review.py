"""Pure logic for the admin panel's Review page.

Kept separate from :mod:`admin_panel.app` (which imports Streamlit) so it
can be unit-tested without a Streamlit runtime -- same split as
:mod:`admin_panel.jobs` and :mod:`admin_panel.usage`.

Review grades live in a **sidecar JSON** next to each batch CSV
(``<batch>.review.json``) so the generated CSV is never mutated. The
sidecar maps ``str(No) -> {"status": ..., "note": ...}``.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

# Grade vocabulary (a subset of the CSV's validation_status values).
REVIEW_STATUSES = ("approved", "needs_review", "rejected")


def review_sidecar_path(csv_path: Path) -> Path:
    """Path to the review sidecar JSON for a batch CSV."""
    return csv_path.with_suffix(".review.json")


def load_reviews(csv_path: Path) -> dict[str, dict[str, str]]:
    """Return ``{str(No): {"status", "note"}}`` for a batch.

    Returns ``{}`` when the sidecar is missing, unreadable, or malformed --
    a corrupt sidecar must never crash the Review page. Only well-shaped
    entries survive.
    """
    path = review_sidecar_path(csv_path)
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            out[str(key)] = {
                "status": str(val.get("status", "")),
                "note": str(val.get("note", "")),
            }
    return out


def save_review(csv_path: Path, no: str | int, status: str, note: str) -> None:
    """Upsert one question's grade + note into the sidecar JSON.

    Raises:
        ValueError: if ``status`` isn't one of :data:`REVIEW_STATUSES` --
            so a UI typo can't write junk into the store.
    """
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"status must be one of {REVIEW_STATUSES}, got {status!r}"
        )
    reviews = load_reviews(csv_path)
    reviews[str(no)] = {"status": status, "note": note}
    path = review_sidecar_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(reviews, fh, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class ReviewSummary:
    """Tally of review progress across one batch."""

    total: int
    reviewed: int
    approved: int
    needs_review: int
    rejected: int

    @property
    def quality_pct(self) -> float | None:
        """Approved share of decided (approve+reject) grades; None if none
        decided yet. `needs_review` is intentionally excluded -- it's
        'undecided', not a quality vote."""
        decided = self.approved + self.rejected
        if decided == 0:
            return None
        return 100.0 * self.approved / decided


def summarize(
    nos: list[object],
    reviews: dict[str, dict[str, str]],
) -> ReviewSummary:
    """Count review progress across a batch's question numbers (``nos``)."""
    approved = needs_review = rejected = reviewed = 0
    for n in nos:
        grade = reviews.get(str(n))
        if not grade:
            continue
        reviewed += 1
        status = grade.get("status")
        if status == "approved":
            approved += 1
        elif status == "rejected":
            rejected += 1
        elif status == "needs_review":
            needs_review += 1
    return ReviewSummary(
        total=len(nos),
        reviewed=reviewed,
        approved=approved,
        needs_review=needs_review,
        rejected=rejected,
    )


def remove_review(csv_path: Path, no: str | int) -> None:
    """Drop one question's grade from the sidecar (no-op if absent)."""
    reviews = load_reviews(csv_path)
    if str(no) in reviews:
        del reviews[str(no)]
        path = review_sidecar_path(csv_path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(reviews, fh, indent=2, ensure_ascii=False)


def remove_question(csv_path: Path, no: str | int) -> bool:
    """Remove a question (by its ``No``) from a batch CSV, in place.

    Rewrites the CSV with the stdlib csv module (utf-8-sig, preserving the
    exact column order) and drops the question's review grade. The ``No``
    of the remaining rows is left UNCHANGED -- gaps are fine, and keeping
    the ids stable means review grades + any external references don't
    silently shift to a different question.

    Returns True iff a row was actually removed.
    """
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    kept = [row for row in rows if str(row.get("No")) != str(no)]
    if len(kept) == len(rows):
        return False  # no row matched that No
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    remove_review(csv_path, no)
    return True


def _update_cell(csv_path: Path, no: str | int, column: str, value: str) -> bool:
    """Overwrite one row's ``column`` cell in a batch CSV, in place.

    Mirrors :func:`remove_question`: rewrites with the stdlib csv module
    (utf-8-sig, exact column order preserved). Only the matched row's
    ``column`` cell changes -- every other cell, every other row, and the
    column order are untouched -- so the file on disk stays the full,
    directly-downloadable batch with the edit baked in.

    Returns True iff a row with that ``No`` was found and ``column`` exists.
    """
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if column not in fieldnames:
        return False
    found = False
    for row in rows:
        if str(row.get("No")) == str(no):
            row[column] = value
            found = True
            break
    if not found:
        return False
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def update_explanation(csv_path: Path, no: str | int, new_text: str) -> bool:
    """Overwrite one question's ``Answer Explanation`` cell in place.

    Thin wrapper over :func:`_update_cell`. Returns True iff the row was
    found and updated.
    """
    return _update_cell(csv_path, no, "Answer Explanation", new_text)


def update_difficulty(csv_path: Path, no: str | int, new_value: str) -> bool:
    """Overwrite one question's ``Difficulty Rating`` cell in place.

    Thin wrapper over :func:`_update_cell` -- used by the Review page's
    inline, auto-saving difficulty editor. ``new_value`` is the rating as a
    string (the column is integer-valued). Returns True iff the row was
    found and updated.
    """
    return _update_cell(csv_path, no, "Difficulty Rating", new_value)


def collect_approved_rows(
    batch_dir: Path, *, exclude_prefix: str = "compare_"
) -> tuple[list[str], list[dict[str, str]]]:
    """Every row graded ``approved``, gathered across all batches in a dir.

    Scans each batch CSV under ``batch_dir`` (excluding ``compare_*`` A/B
    artifacts) together with its ``.review.json`` sidecar, keeps the rows whose
    ``No`` is graded ``approved``, and dedupes across batches by
    ``(solver_reference, User Cards)`` so the same spot approved in two batches
    appears once. Batches are scanned newest-first (by mtime), so the most
    recent copy of a duplicated spot wins.

    The grades are the single source of truth -- nothing is moved or copied on
    disk -- so this view always reflects the latest approvals and an un-approve
    drops the row automatically.

    Returns ``(fieldnames, rows)`` ready to serialize as one CSV;
    ``fieldnames`` is the column order of the first contributing batch. Returns
    ``([], [])`` when the dir is absent or nothing is approved.
    """
    if not batch_dir.is_dir():
        return [], []
    csvs = sorted(
        (p for p in batch_dir.glob("*.csv") if not p.name.startswith(exclude_prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    seen: set[tuple[str, str]] = set()
    fieldnames: list[str] = []
    out_rows: list[dict[str, str]] = []
    for csv_path in csvs:
        reviews = load_reviews(csv_path)
        if not any(g.get("status") == "approved" for g in reviews.values()):
            continue  # cheap skip: no approvals in this batch
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                cols = list(reader.fieldnames or [])
                rows = list(reader)
        except OSError:
            continue
        if cols and not fieldnames:
            fieldnames = cols
        for row in rows:
            grade = reviews.get(str(row.get("No", "")))
            if not grade or grade.get("status") != "approved":
                continue
            key = (row.get("solver_reference", ""), row.get("User Cards", ""))
            if key in seen:
                continue
            seen.add(key)
            out_rows.append(row)
    return fieldnames, out_rows


def approved_rows_to_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    """Serialize approved rows to a CSV string (header + rows).

    Uses ``fieldnames`` for column order; keys outside it are dropped and
    missing cells are blank, so rows from a slightly different schema still
    write cleanly. Returns just a header when ``rows`` is empty.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in fieldnames})
    return buf.getvalue()


def batch_meta_path(csv_path: Path) -> Path:
    """Path to the prompt/inputs metadata sidecar for a batch CSV."""
    return csv_path.with_suffix(".meta.json")


def load_batch_meta(csv_path: Path) -> dict[str, object] | None:
    """Return the parsed ``<batch>.meta.json`` for a batch, or None.

    None when the sidecar is missing or malformed -- batches made before
    prompt tracking simply have no meta, and the Review page treats that as
    "inputs unavailable" rather than erroring.
    """
    path = batch_meta_path(csv_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def meta_question_for(
    meta: dict[str, object],
    *,
    user_cards: str,
    solver_reference: str,
) -> dict[str, object] | None:
    """Find the meta question record matching one CSV row.

    Matches on (node_id, User Cards) -- a spot's unique key -- rather than
    the ``No`` column, so the join survives the Review page's
    remove-question feature (which leaves gaps in ``No``). ``node_id`` is
    matched as a substring of the row's ``solver_reference``
    (``<pack>/<actor>/<node_id>``), which is robust to path-format changes.
    Keyed on the hero hole cards (``user_cards``) rather than the old
    ``hand_class`` label, which was dropped from the CSV June 2026; batches
    generated before that change have no ``user_cards`` in their meta and so
    won't match (the inspector shows "no metadata", which is graceful).
    """
    questions = meta.get("questions")
    if not isinstance(questions, list):
        return None
    for q in questions:
        if not isinstance(q, dict):
            continue
        q_node = str(q.get("node_id", ""))
        if (
            q_node
            and q_node in solver_reference
            and str(q.get("user_cards", "")) == user_cards
        ):
            return q
    return None


def assembled_prompt(meta: dict[str, object], question: dict[str, object]) -> str:
    """Reconstruct the full prompt for one question: system + gold + live.

    Mirrors the ``assembled`` layout of
    ``pipeline.preflop.explanation_generator.build_explanation_prompt_parts``
    but from the stored snapshot, so it shows exactly what was sent at
    generation time (even if the prompt was edited or renamed since).
    """
    system_text = str(meta.get("prompt_text", ""))
    gold_block = str(meta.get("gold_block", ""))
    live_block = str(question.get("live_block", ""))
    return (
        "===== SYSTEM PROMPT =====\n"
        f"{system_text}\n\n"
        "===== GOLD EXAMPLES (cached) =====\n"
        f"{gold_block}"
        f"{live_block}"
    )


def range_player_count(ranges_json: str) -> int:
    """Number of players in a ``ranges`` JSON cell (0 if empty/malformed)."""
    if not ranges_json:
        return 0
    try:
        data = json.loads(ranges_json)
    except (json.JSONDecodeError, TypeError):
        return 0
    return len(data) if isinstance(data, dict) else 0


__all__ = [
    "REVIEW_STATUSES",
    "ReviewSummary",
    "approved_rows_to_csv",
    "assembled_prompt",
    "batch_meta_path",
    "collect_approved_rows",
    "load_batch_meta",
    "load_reviews",
    "meta_question_for",
    "range_player_count",
    "remove_question",
    "remove_review",
    "review_sidecar_path",
    "save_review",
    "summarize",
    "update_difficulty",
    "update_explanation",
]
