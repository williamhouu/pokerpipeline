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
    "load_reviews",
    "range_player_count",
    "remove_question",
    "remove_review",
    "review_sidecar_path",
    "save_review",
    "summarize",
]
