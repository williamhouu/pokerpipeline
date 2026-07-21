"""Pure logic for the admin panel's Review page.

Kept separate from :mod:`admin_panel.app` (which imports Streamlit) so it
can be unit-tested without a Streamlit runtime -- same split as
:mod:`admin_panel.jobs` and :mod:`admin_panel.usage`.

Review grades live in a **sidecar JSON** next to each batch CSV
(``<batch>.review.json``) so the generated CSV is never mutated. The
sidecar maps ``str(No) -> {"status": ..., "note": ...}`` plus an optional
``"explanation"`` -- a reviewer-edited Answer Explanation that overrides
the generated one wherever approved rows are collected (the CSV keeps the
original, so an A/B verdict's context is never rewritten).
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from pipeline.provenance import node_reference_from_row

# Grade vocabulary (a subset of the CSV's validation_status values).
REVIEW_STATUSES = ("approved", "needs_review", "rejected")


def _node_ref(row: dict) -> str:
    """The row's machine node reference (the old ``solver_reference`` value).

    Lives in the Notes ``Node:`` field since July 2026; this is the single
    seam Review uses to key spots (cross-batch dedup, promote idempotency).
    Pre-July-2026 batches resolve via their legacy ``solver_reference``
    column inside ``node_reference_from_row``.
    """
    return node_reference_from_row(row)


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
            entry = {
                "status": str(val.get("status", "")),
                "note": str(val.get("note", "")),
            }
            if val.get("explanation"):
                entry["explanation"] = str(val["explanation"])
            out[str(key)] = entry
    return out


def save_review(
    csv_path: Path,
    no: str | int,
    status: str,
    note: str,
    *,
    explanation: str | None = None,
) -> None:
    """Upsert one question's grade + note into the sidecar JSON.

    ``explanation`` (optional) stores a reviewer-edited Answer Explanation
    alongside the grade; :func:`collect_approved_rows` substitutes it for the
    generated one. ``None`` leaves no override (the generated text ships).

    Raises:
        ValueError: if ``status`` isn't one of :data:`REVIEW_STATUSES` --
            so a UI typo can't write junk into the store.
    """
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"status must be one of {REVIEW_STATUSES}, got {status!r}"
        )
    reviews = load_reviews(csv_path)
    entry: dict[str, str] = {"status": status, "note": note}
    if explanation:
        entry["explanation"] = explanation
    reviews[str(no)] = entry
    path = review_sidecar_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(reviews, fh, indent=2, ensure_ascii=False)


# --- one-click "approve all fully-clean" (bulk approve) --------------------
# A question is "fully clean" when it is GREEN on EVERY flag source: the
# Layer-7 claim checker actually ran and came back empty, AND no deterministic
# soft validator, deterministic cross-check, or AI sanity audit flagged it.
# The Review pages expose a button that sweeps every such row that the reviewer
# hasn't already graded into the approved pool in one click. The eligibility
# logic lives HERE (pure, browserless-tested) so the Streamlit button is a thin
# shell -- see the fix-durability rule in CLAUDE.md.

# Meta question-record fields that flag the SHIPPED row regardless of audit
# mode. NOTE: ``claim_check_issues`` is deliberately NOT here -- in an auto-fix
# batch it holds the PRE-rewrite gate flags, so a fixed-then-clean row still
# carries them; the shipped-text claim verdict comes from the ``revise`` record
# (auto-fix mode) or the ``claim_check`` column (flag-only mode) instead. A
# field simply absent from a given pipeline's meta reads as clean.
_FLAG_FIELDS = (
    "validator_warnings",   # deterministic soft validators (position wording, ...)
    "sanity_check_issues",  # AI sanity audit (challenges the solver facts)
    "cross_check_issues",   # deterministic first-principles cross-check
)


def row_is_fully_clean(claim_check_cell: str, qrec: dict | None) -> bool:
    """True when a shipped question is green on EVERY flag source.

    "Fully clean" = the Layer-7 audit ran on this row and the SHIPPED text
    passed it with zero outstanding flags from any source. The claim verdict is
    read MODE-AWARELY, because where "did it pass" is recorded differs:

    * **Auto-fix batch** (``revise_pass`` -- the meta record has a ``revise``
      block): the shipped-text verdict lives in that block, NOT the
      ``claim_check`` column (which in this mode may hold the pre-rewrite gate
      flags, or be blank even when the gate ran clean). Green means
      ``status`` is ``"clean"`` (never flagged) or ``"fixed"`` (rewritten) AND
      the 4th-call ``final_audit_issues`` is empty. ``"discarded"`` /
      ``"unchanged"`` ship the flagged original, and a ``"fixed"`` row that still
      has final-audit flags carries an outstanding flag -- none are green.
    * **Flag-only / un-revised batch** (no ``revise`` block): the ``claim_check``
      column is authoritative. ``""`` = the audit never ran (not eligible);
      ``"[]"`` = ran and clean; a populated list = flagged.

    In BOTH modes the deterministic + sanity flag sources (:data:`_FLAG_FIELDS`)
    on the shipped row must also be empty. This mirrors exactly what the Review
    card shows as "green" (see ``_render_revise_panel`` in the app).
    """
    if isinstance(qrec, dict):
        for field in _FLAG_FIELDS:
            if qrec.get(field):
                return False
        revise = qrec.get("revise")
        if isinstance(revise, dict):
            if revise.get("status") not in ("clean", "fixed"):
                return False  # discarded / unchanged ship the flagged original
            if revise.get("final_audit_issues"):
                return False  # rewrite the 4th-call audit still flags -> not green
            return True
    # No revise record: the claim_check column is the shipped-text verdict.
    cell = (claim_check_cell or "").strip()
    if not cell:
        return False  # audit did not run -> cannot be "fully passed"
    try:
        parsed = json.loads(cell)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, list) and not parsed


def fully_clean_ungraded_nos(
    rows: Iterable[dict],
    reviews: dict[str, dict[str, str]],
    qrec_for: Callable[[dict], dict | None],
) -> list[str]:
    """The ``No``s eligible for one-click bulk approve, in row order.

    A row qualifies when it is :func:`row_is_fully_clean` AND has NO existing
    reviewer grade -- so the sweep can only add approvals, never overturn a
    human approve / reject / needs-review decision. ``qrec_for`` resolves a
    row's meta question record (each Review page joins differently: preflop by
    node+cards, full-hand by hand_id+sequence, PLO by ``No``)."""
    out: list[str] = []
    for row in rows:
        no = str(row.get("No", "")).strip()
        if not no or no in reviews:
            continue
        if row_is_fully_clean(str(row.get("claim_check", "")), qrec_for(row)):
            out.append(no)
    return out


def bulk_approve(
    csv_path: Path,
    nos: Iterable[str | int],
    *,
    note: str = "Bulk-approved: fully clean (Layer-7 audit + all validators).",
) -> int:
    """Mark every ``No`` in ``nos`` ``approved`` in one sidecar write.

    Skips any ``No`` that already has a grade (idempotent, and never overwrites
    a human decision -- the caller already filters to ungraded rows, this is the
    belt-and-suspenders guard). Returns the number newly approved."""
    reviews = load_reviews(csv_path)
    added = 0
    for no in nos:
        key = str(no)
        if key in reviews:
            continue
        reviews[key] = {"status": "approved", "note": note}
        added += 1
    if added:
        path = review_sidecar_path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(reviews, fh, indent=2, ensure_ascii=False)
    return added


# --- PLO range hand-type breakdown panel (July 2026) -----------------------
# Pure formatter for the `range_breakdown` CSV cell -> display-ready rows, so
# the Streamlit expander is a thin shell (fix-durability rule: the logic is
# browserless-tested, the UI card just renders what this returns).
_VERB_PAST = {
    "Fold": "folds", "Call": "calls", "Check": "checks", "Raise": "raises",
    "3-bet": "3-bets", "4-bet": "4-bets", "5-bet": "5-bets", "All-in": "jams",
}
_ROLE_LABEL = {
    "open": "opening range", "3-bet": "3-betting range",
    "4-bet": "4-betting range", "5-bet": "5-betting range",
    "call": "calling range", "raise": "raising range",
    "decision": "range",
}


def _fmt_actions(actions: dict) -> str:
    """"calls 38%, folds 33%, 4-bets 29%" from a bucket's action split."""
    if not isinstance(actions, dict) or not actions:
        return ""
    ordered = sorted(actions.items(), key=lambda kv: -_as_num(kv[1]))
    return ", ".join(
        f"{_VERB_PAST.get(lbl, str(lbl).lower())} {_as_num(v):.0f}%"
        for lbl, v in ordered
    )


def _as_num(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def range_breakdown_panel(cell: str) -> dict | None:
    """Parse the ``range_breakdown`` cell into display-ready sections.

    Returns ``None`` for an empty / malformed / non-PLO cell (so the panel
    no-ops). Otherwise::

        {"copy": "<overall read>",
         "players": [
            {"heading": "Your range · Small Blind",
             "summary": "<sentence>",
             "hero": True,
             "rows": [{"category": "One Pair Single-Suited",
                       "pct": 36.0, "detail": "calls 38%, folds 33%, ..."}]},
            {"heading": "The Big Blind · 3-betting range", "hero": False, ...},
         ]}

    Hero rows carry the fold/call/raise split in ``detail``; villain rows
    leave ``detail`` empty (they have already acted -- composition only).
    """
    if not cell or not cell.strip():
        return None
    try:
        data = json.loads(cell)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "hero" not in data:
        return None

    def _rows(buckets: object) -> list[dict]:
        out: list[dict] = []
        if not isinstance(buckets, list):
            return out
        for b in buckets:
            if not isinstance(b, dict):
                continue
            out.append({
                "category": str(b.get("category", "")),
                "pct": _as_num(b.get("pct")),
                "detail": _fmt_actions(b.get("actions", {})),
            })
        return out

    players: list[dict] = []
    hero = data.get("hero")
    if isinstance(hero, dict):
        pos = str(hero.get("position", ""))
        players.append({
            "heading": f"Your range · {pos}" if pos else "Your range",
            "summary": str(hero.get("summary", "")),
            "hero": True,
            "rows": _rows(hero.get("buckets")),
        })
    for v in data.get("villains", []) or []:
        if not isinstance(v, dict):
            continue
        ref = str(v.get("position", ""))
        role = _ROLE_LABEL.get(str(v.get("role", "")), "range")
        players.append({
            "heading": f"{ref} · {role}" if ref else role,
            "summary": str(v.get("summary", "")),
            "hero": False,
            "rows": _rows(v.get("buckets")),
        })
    if not any(p["rows"] for p in players):
        return None
    return {"copy": str(data.get("copy", "")), "players": players}


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
            if row.get(column, "") == value:
                return True  # already up to date -- skip the rewrite (and the
                # mtime bump), so an unconditional flush on every navigation is
                # free when nothing changed.
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


def _scan_approved(
    batch_dir: Path, exclude_prefix: str | None
) -> tuple[list[str], list[tuple[Path, str, dict[str, str]]]]:
    """``(fieldnames, [(csv_path, No, row), ...])`` for every approved row.

    Scans each CSV under ``batch_dir`` + its ``.review.json`` sidecar, keeps
    rows graded ``approved``, and dedupes across batches by ``(solver_reference,
    User Cards)`` so a spot approved in two places appears once (newest batch by
    mtime wins). ``fieldnames`` is the column order of the first contributing
    batch. The shared core behind the public collectors; the ``(csv_path, No)``
    provenance lets the Review page delete an individual approved row.
    """
    if not batch_dir.is_dir():
        return [], []
    csvs = sorted(
        (
            p
            for p in batch_dir.glob("*.csv")
            if not (exclude_prefix and p.name.startswith(exclude_prefix))
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    seen: set[tuple[str, str]] = set()
    fieldnames: list[str] = []
    out: list[tuple[Path, str, dict[str, str]]] = []
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
            no = str(row.get("No", ""))
            grade = reviews.get(no)
            if not grade or grade.get("status") != "approved":
                continue
            key = (_node_ref(row), row.get("User Cards", ""))
            if key in seen:
                continue
            seen.add(key)
            # A reviewer-edited explanation (saved with the approval) ships
            # in place of the generated one; the CSV itself stays untouched.
            if grade.get("explanation"):
                row = {**row, "Answer Explanation": grade["explanation"]}
            out.append((csv_path, no, row))
    return fieldnames, out


def collect_approved_rows(
    batch_dir: Path, *, exclude_prefix: str | None = None
) -> tuple[list[str], list[dict[str, str]]]:
    """Every row graded ``approved``, gathered across all batches in a dir.

    Dedupes across batches by ``(solver_reference, User Cards)`` (newest wins),
    and includes ``compare_*`` A/B artifacts: a question finalized on the
    Compare page is marked ``approved`` in that compare CSV's sidecar and
    belongs in the same pool as Review-page approvals. Only rows actually graded
    ``approved`` surface. Pass ``exclude_prefix`` to skip a filename family.

    The grades are the single source of truth -- nothing is moved on disk -- so
    this view always reflects the latest approvals and an un-approve (or
    :func:`clear_all_approved`) drops rows automatically. Returns
    ``(fieldnames, rows)`` ready to serialize as one CSV.
    """
    fieldnames, sources = _scan_approved(batch_dir, exclude_prefix)
    return fieldnames, [row for _csv, _no, row in sources]


def collect_approved_sources(
    batch_dir: Path, *, exclude_prefix: str | None = None
) -> list[tuple[Path, str, dict[str, str]]]:
    """Like :func:`collect_approved_rows`, but each row paired with its source
    ``(csv_path, No, row)`` so a single approved question can be un-approved in
    place (the Review page's per-row delete)."""
    return _scan_approved(batch_dir, exclude_prefix)[1]


def clear_all_approved(
    batch_dir: Path, *, exclude_prefix: str | None = None
) -> int:
    """Un-approve EVERY approved row across all batches; return the count.

    Removes the ``approved`` grade from each sidecar (not deduped -- so a spot
    approved in two batches is cleared in both, and won't reappear). Used by the
    Review page's "Clear all approved" after a finalized set has been exported.
    Other grades (rejected / needs_review) are untouched.
    """
    if not batch_dir.is_dir():
        return 0
    cleared = 0
    for csv_path in batch_dir.glob("*.csv"):
        if exclude_prefix and csv_path.name.startswith(exclude_prefix):
            continue
        reviews = load_reviews(csv_path)
        for no, grade in list(reviews.items()):
            if grade.get("status") == "approved":
                remove_review(csv_path, no)
                cleared += 1
    return cleared


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


# --- full-hand (play-through) hand-level review ------------------------------
# The unit a reviewer ships is the HAND, not the leg: the app only ever
# receives complete preflop->river sequences, so one bad leg poisons its whole
# hand. These helpers are pure/file-level (no Streamlit) so the hand-level UI
# is a thin shell over browserlessly-tested logic (the fix-durability rule).

HAND_FILTER_MODES = ("all", "drop_rejected", "approved_only")


def hand_status(leg_nos: list[object], reviews: dict[str, dict[str, str]]) -> str:
    """Aggregate one full hand's review status from its legs' grades.

    WHOLE-HAND ATOMICITY: any rejected leg -> ``"rejected"`` (the hand can't
    ship with a bad leg, and shipping it without that leg would be a partial
    hand); else any needs_review leg -> ``"needs_review"``; else ALL legs
    approved -> ``"approved"``; anything else -> ``""`` (pending).
    """
    statuses = [reviews.get(str(n), {}).get("status", "") for n in leg_nos]
    if "rejected" in statuses:
        return "rejected"
    if "needs_review" in statuses:
        return "needs_review"
    if statuses and all(s == "approved" for s in statuses):
        return "approved"
    return ""


def save_reviews_bulk(
    csv_path: Path,
    nos: list[object],
    status: str,
    note: str | None = None,
) -> None:
    """Grade many questions in ONE sidecar write (the hand-level buttons).

    ``note=None`` PRESERVES each leg's existing note (audit flags live
    there); a string replaces it. Existing reviewer-edited explanation
    overrides are always preserved.
    """
    if status not in REVIEW_STATUSES:
        raise ValueError(f"status must be one of {REVIEW_STATUSES}, got {status!r}")
    reviews = load_reviews(csv_path)
    for no in nos:
        old = reviews.get(str(no), {})
        entry: dict[str, str] = {
            "status": status,
            "note": old.get("note", "") if note is None else note,
        }
        if old.get("explanation"):
            entry["explanation"] = old["explanation"]
        reviews[str(no)] = entry
    path = review_sidecar_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(reviews, fh, indent=2, ensure_ascii=False)


def remove_hand(csv_path: Path, nos: list[object]) -> int:
    """Remove ALL of a hand's legs from a batch CSV in one rewrite.

    The hand-level analogue of :func:`remove_question` -- removing legs
    one-by-one could leave an orphaned partial hand if interrupted; this
    drops the whole set atomically and clears their sidecar grades.
    Remaining rows keep their ``No`` (stable ids; gaps are fine). Returns
    the number of rows removed.
    """
    if not csv_path.is_file():
        return 0
    wanted = {str(n) for n in nos}
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    kept = [row for row in rows if str(row.get("No")) not in wanted]
    removed = len(rows) - len(kept)
    if removed == 0:
        return 0
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    reviews = load_reviews(csv_path)
    if any(str(n) in reviews for n in nos):
        for n in nos:
            reviews.pop(str(n), None)
        with review_sidecar_path(csv_path).open("w", encoding="utf-8") as fh:
            json.dump(reviews, fh, indent=2, ensure_ascii=False)
    return removed


def group_hand_rows(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    """Group batch rows into hands, preserving CSV order.

    Rows sharing a non-blank ``hand_id`` form one hand (legs sorted by
    ``sequence_index``); a blank ``hand_id`` row is its own singleton group
    (standalone question). Group key is the hand_id or ``"__row_<No>"``.
    """
    groups: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for row in rows:
        hid = str(row.get("hand_id", "") or "").strip()
        key = hid if hid else f"__row_{row.get('No', '')}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    def _seq(r: dict[str, str]) -> int:
        try:
            return int(float(str(r.get("sequence_index", "") or 0)))
        except ValueError:
            return 0

    return [(k, sorted(groups[k], key=_seq)) for k in order]


def filter_hand_rows(
    rows: list[dict[str, str]],
    reviews: dict[str, dict[str, str]],
    mode: str,
) -> tuple[list[dict[str, str]], int, int]:
    """Filter a batch to WHOLE hands by review status, for export.

    ``mode``: ``"all"`` keeps everything; ``"drop_rejected"`` keeps every
    hand except rejected ones; ``"approved_only"`` keeps only fully-approved
    hands (see :func:`hand_status`). Output rows are COPIES with ``No``
    renumbered contiguously from 1 (the on-disk batch is untouched) and any
    reviewer-edited explanation override substituted, so the export always
    carries the latest text. Hands are never split.

    Returns ``(rows, kept_hands, total_hands)``.
    """
    if mode not in HAND_FILTER_MODES:
        raise ValueError(f"mode must be one of {HAND_FILTER_MODES}, got {mode!r}")
    out: list[dict[str, str]] = []
    kept_hands = total_hands = 0
    next_no = 1
    for _key, legs in group_hand_rows(rows):
        total_hands += 1
        status = hand_status([r.get("No", "") for r in legs], reviews)
        keep = (
            mode == "all"
            or (mode == "drop_rejected" and status != "rejected")
            or (mode == "approved_only" and status == "approved")
        )
        if not keep:
            continue
        kept_hands += 1
        for r in legs:
            grade = reviews.get(str(r.get("No", "")), {})
            row = dict(r)
            if grade.get("explanation"):
                row["Answer Explanation"] = grade["explanation"]
            row["No"] = str(next_no)
            next_no += 1
            out.append(row)
    return out, kept_hands, total_hands


def meta_question_for_leg(
    meta: dict[str, object] | None,
    *,
    hand_id: str,
    sequence_index: str,
) -> dict[str, object] | None:
    """Find a FULL-HAND row's meta question record by (hand_id,
    sequence_index) -- the one key that is unique and present for EVERY leg
    kind. The older (node_id, last-path-segment-as-combo) join silently
    missed pack PREFLOP legs, whose solver_reference ends in the node id,
    hiding their Layer-7 panels (the audit ran; Review showed nothing).
    Returns None for blank hand_id (standalone rows use the node/combo join).
    """
    if not isinstance(meta, dict) or not str(hand_id).strip():
        return None
    questions = meta.get("questions")
    if not isinstance(questions, list):
        return None
    for q in questions:
        if (
            isinstance(q, dict)
            and str(q.get("hand_id", "")) == str(hand_id)
            and str(q.get("sequence_index", "")) == str(sequence_index)
        ):
            return q
    return None


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
    node_reference: str,
) -> dict[str, object] | None:
    """Find the meta question record matching one CSV row.

    Matches on (node_id, User Cards) -- a spot's unique key -- rather than
    the ``No`` column, so the join survives the Review page's
    remove-question feature (which leaves gaps in ``No``). ``node_id`` is
    matched as a substring of the row's node reference
    (``<pack>/<actor>/<node_id>``), which now lives in the Notes ``Node:``
    field (was the dropped ``solver_reference`` column, July 2026) -- the
    value is byte-identical, so the substring match is unchanged. Keyed on
    the hero hole cards (``user_cards``) rather than the old ``hand_class``
    label, which was dropped from the CSV June 2026; batches generated before
    that change have no ``user_cards`` in their meta and so won't match (the
    inspector shows "no metadata", which is graceful).
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
            and q_node in node_reference
            and str(q.get("user_cards", "")) == user_cards
        ):
            return q
    return None


def preflop_leg_provenance(
    meta: dict[str, object] | None, *, hand_id: str
) -> str | None:
    """Small provenance caption for a full-hand PREFLOP leg: which source
    built it (a matched range pack, or the solve's entry ranges).

    Joins the meta ``questions`` on ``(hand_id, street == "preflop")`` -- a
    hand has at most one preflop leg, so the pair is unique. Returns None
    when it can't be determined (no meta, blank hand_id, no preflop record),
    which renders as no caption -- graceful for older batches. Pure function
    (no Streamlit) so it is testable without a browser.
    """
    if not isinstance(meta, dict) or not str(hand_id).strip():
        return None
    questions = meta.get("questions")
    if not isinstance(questions, list):
        return None
    for q in questions:
        if (
            isinstance(q, dict)
            and str(q.get("hand_id", "")) == str(hand_id)
            and q.get("street") == "preflop"
        ):
            if q.get("preflop_leg_source") == "pack":
                pack = str(q.get("pack_id") or "").strip()
                if pack:
                    return f"Preflop leg from range pack: {pack}"
                return "Preflop leg from a range pack"
            return (
                "Preflop leg from the solve's entry ranges "
                "(no matching range pack)"
            )
    return None


def batch_pack_id(meta: dict[str, object] | None) -> str | None:
    """The range pack a standalone PREFLOP batch was generated from, or None.

    Reads the top-level ``pack_id`` the preflop batch meta records (one pack
    per batch). Returns None for blank / missing (older batches, postflop
    batches), which renders as no caption. Pure function for browserless
    tests.
    """
    if not isinstance(meta, dict):
        return None
    pack = str(meta.get("pack_id") or "").strip()
    return pack or None


def empty_batch_diagnosis(meta: dict[str, object] | None) -> str | None:
    """Plain-English reason a batch CSV came out EMPTY, or None if unknown.

    A silent 0-question batch reads as a mystery failure ("Job done", empty
    CSV); the meta counters usually record exactly which gate ate everything.
    Pure function (no Streamlit) so it is testable without a browser.
    """
    if not isinstance(meta, dict):
        return None
    counters = meta.get("counters")
    rs = meta.get("run_settings")
    if not isinstance(counters, dict) or not isinstance(rs, dict):
        return None
    if meta.get("mode") != "full_hand":
        return None
    if int(counters.get("questions_written", 0) or 0) > 0:
        return None

    assembled = int(counters.get("hands_assembled", 0) or 0)
    filtered = int(counters.get("hands_difficulty_filtered", 0) or 0)
    lo = rs.get("min_hand_difficulty")
    hi = rs.get("max_hand_difficulty")

    if assembled == 0:
        return (
            "No hands could be assembled: no worthy decision inside the "
            "frequency window seeded a play-through. Widen the worthiness "
            "window (Advanced filters) or check the heroes/streets selection."
        )
    if filtered >= assembled and (lo is not None or hi is not None):
        band = f"{lo if lo is not None else 400}-{hi if hi is not None else 3200}"
        observed = counters.get("hand_difficulty_observed_max")
        observed_bit = (
            f" The hardest hand scanned rated **{observed}**."
            if isinstance(observed, int) else ""
        )
        return (
            f"The hand-difficulty band ({band}) filtered out every hand: "
            f"{assembled} scanned, none landed in the band.{observed_bit} "
            "Try a wider band, or enable 🪤 trap-aware (and 🔪 razor's-edge) "
            "difficulty -- deceptive pure decisions then rate 1800+, which "
            "is where most genuinely hard hands live."
        )
    return None


def revise_summary_line(meta: dict[str, object] | None) -> str | None:
    """One-line summary of the 4-call audit & auto-fix pass, or ``None`` when
    the batch did not use it (``revise_pass`` off).

    Drives the prominent Review-page banner that marks an experimental batch.
    Reads ``run_settings.revise_pass`` + the ``revise_*`` counters the batch
    records. Older batches (generated before per-question recording existed)
    return a generic note rather than misleading zero counts.
    """
    rs = (meta or {}).get("run_settings") or {}
    if not isinstance(rs, dict) or not rs.get("revise_pass"):
        return None
    counters = (meta or {}).get("counters") or {}
    audit = "on" if rs.get("final_audit") else "off"
    if not isinstance(counters, dict) or "revise_flagged" not in counters:
        return (
            "Per-question auto-fix details are recorded on batches generated "
            f"after this feature update. (Final audit: {audit}.)"
        )
    flagged = int(counters.get("revise_flagged", 0) or 0)
    fixed = int(counters.get("revise_fixed", 0) or 0)
    discarded = int(counters.get("revise_discarded", 0) or 0)
    unchanged = int(counters.get("revise_unchanged", 0) or 0)
    if not flagged:
        return (
            "The claim-check gate flagged 0 questions, so no rewrites were "
            f"needed. (Final audit: {audit}.)"
        )
    parts = [f"**{fixed} auto-fixed**"]
    if discarded:
        parts.append(f"{discarded} discarded (rewrite broke a rule, original kept)")
    if unchanged:
        parts.append(f"{unchanged} unchanged (reviser made no edit)")
    return (
        f"The gate flagged **{flagged}** question(s): "
        + " · ".join(parts)
        + f". (Final audit: {audit}.)"
    )


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


def load_failures(csv_path: Path) -> list[dict[str, object]]:
    """Return the routed-to-human-review failures for a batch.

    Read from the batch's ``.meta.json`` (the ``failures`` key, persisted by
    the generator). Each entry carries the deterministic question/options/
    correct-answer context, the LLM's rejected explanation, the rejection
    reason, and an optional ``row`` -- the COMPLETE CSV row built from the
    rejected explanation, which :func:`promote_failure` appends. Returns
    ``[]`` when there is no meta, no failures, or it's malformed (a batch made
    before failure-persistence simply has none, handled gracefully).
    """
    meta = load_batch_meta(csv_path)
    if not meta:
        return []
    failures = meta.get("failures")
    if not isinstance(failures, list):
        return []
    return [f for f in failures if isinstance(f, dict)]


def promote_failure(csv_path: Path, failure: dict[str, object]) -> tuple[bool, str]:
    """Append a routed-to-review spot into its batch CSV, keeping the exact
    rejected explanation, and mark it approved in the sidecar.

    Uses the ``row`` the generator pre-built from the rejected explanation
    (every CSV column). Assigns the next ``No`` (max existing + 1), stamps
    ``validation_status='approved'``, appends it, and records an ``approved``
    grade in ``.review.json`` so it flows into the approved pool like any
    other approval -- with the rejected explanation stored as the override so
    that exact text ships. Idempotent: a spot already present (same
    ``solver_reference`` + ``User Cards``) is not added twice.

    Returns ``(ok, message)``.
    """
    row = failure.get("row")
    if not isinstance(row, dict) or not row:
        return False, "no rebuilt row for this spot -- nothing to promote"
    if not csv_path.is_file():
        return False, f"batch CSV not found: {csv_path.name}"
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        return False, "batch CSV has no header row"
    # Idempotency: same spot (node reference + User Cards) already present?
    key = (_node_ref(row), str(row.get("User Cards", "")))
    for existing in rows:
        if (_node_ref(existing), str(existing.get("User Cards", ""))) == key:
            return False, "this spot is already in the batch"
    # Next No = max existing integer No + 1 (fall back to count+1). Gaps from
    # removed questions are fine; we only need a fresh, unused id.
    nums: list[int] = []
    for r in rows:
        try:
            nums.append(int(str(r.get("No", "")).strip()))
        except (TypeError, ValueError):
            continue
    new_no = (max(nums) + 1) if nums else (len(rows) + 1)
    new_row = {col: str(row.get(col, "")) for col in fieldnames}
    new_row["No"] = str(new_no)
    if "validation_status" in fieldnames:
        new_row["validation_status"] = "approved"
    rows.append(new_row)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_review(
        csv_path,
        new_no,
        "approved",
        "Promoted from the human-review queue (kept the rejected explanation).",
        explanation=str(row.get("Answer Explanation", "")) or None,
    )
    return True, f"added as #{new_no} and approved"


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
    "batch_pack_id",
    "empty_batch_diagnosis",
    "preflop_leg_provenance",
    "clear_all_approved",
    "collect_approved_rows",
    "collect_approved_sources",
    "load_batch_meta",
    "load_failures",
    "load_reviews",
    "promote_failure",
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


# --- flagged-phrase highlighting (July 2026) ---------------------------------
# The Layer-7 checkers quote the EXACT offending phrase in every flag, so the
# Review page can render the explanation with those phrases highlighted -- the
# reviewer scans the marks instead of hunting through prose. Pure functions
# (browserless-tested, per the fix-durability rule); the Streamlit card is a
# thin shell over these.

_MARK_STYLE = (
    "background:#ffe08a;color:#111;border-radius:3px;padding:0 2px"
)


def flagged_claims_for_row(
    claim_check_cell: str, qrec: dict | None = None
) -> list[str]:
    """The audit-flagged claim phrases that refer to the row's SHIPPED prose.

    Primary source: the ``claim_check`` CSV column -- by construction it
    always describes the shipped text (flag-only -> the gate on the shipped
    original; auto-fixed -> the final audit on the shipped rewrite;
    discarded/unchanged -> the gate on the shipped original). Fallback /
    supplement: the meta question record's ``claim_check_issues`` (either
    ``"claim -- problem"`` strings or ``{claim, problem}`` dicts) and, when
    the revise record says the REWRITE shipped, its ``final_audit_issues``.
    Deduped, order kept.
    """
    claims: list[str] = []

    def _add(claim: str) -> None:
        claim = (claim or "").strip()
        if claim and claim not in claims:
            claims.append(claim)

    if claim_check_cell and claim_check_cell.strip():
        try:
            data = json.loads(claim_check_cell)
        except (ValueError, TypeError):
            data = []
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict):
                    _add(str(d.get("claim", "")))

    def _add_mixed(items: object) -> None:
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict):
                _add(str(it.get("claim", "")))
            elif isinstance(it, str):
                _add(it.split(" -- ", 1)[0])

    if isinstance(qrec, dict):
        _add_mixed(qrec.get("claim_check_issues"))
        revise = qrec.get("revise")
        if isinstance(revise, dict):
            if revise.get("status") == "fixed":
                _add_mixed(revise.get("final_audit_issues"))
            elif revise.get("status") in ("discarded", "unchanged"):
                _add_mixed(revise.get("gate_issues"))
    return claims


def _claim_spans(text: str, claim: str) -> tuple[int, int] | None:
    """Locate ``claim`` in ``text``: exact, then case-insensitive, then
    whitespace-tolerant + straight/curly quote-tolerant. None if not found
    (the checker sometimes elides or paraphrases a long phrase)."""
    import re as _re

    idx = text.find(claim)
    if idx >= 0:
        return (idx, idx + len(claim))
    low = text.lower()
    idx = low.find(claim.lower())
    if idx >= 0:
        return (idx, idx + len(claim))
    pattern = _re.escape(claim)
    pattern = pattern.replace(r"\ ", r"\s+")
    for a, b in (("'", "['’]"), ('"', "[\"“”]")):
        pattern = pattern.replace(_re.escape(a), b)
    m = _re.search(pattern, text, _re.IGNORECASE)
    if m:
        return (m.start(), m.end())
    return None


def highlight_claims_html(text: str, claims: list[str]) -> tuple[str, int]:
    """Escaped HTML of ``text`` with each located claim wrapped in a
    ``<mark>``. Returns ``(html, located_count)``; claims that can't be
    located are simply not marked (the flag list below the card still shows
    them). Overlapping matches keep the first-located span."""
    import html as _html

    spans: list[tuple[int, int]] = []
    located = 0
    for claim in claims:
        span = _claim_spans(text, claim)
        if span is None:
            continue
        if any(not (span[1] <= s or span[0] >= e) for s, e in spans):
            located += 1  # overlaps an existing mark -- already visible
            continue
        spans.append(span)
        located += 1
    spans.sort()
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(_html.escape(text[cursor:start]))
        parts.append(
            f'<mark style="{_MARK_STYLE}">'
            + _html.escape(text[start:end])
            + "</mark>"
        )
        cursor = end
    parts.append(_html.escape(text[cursor:]))
    body = "".join(parts)
    return (
        '<div style="white-space:pre-wrap;line-height:1.55">' + body + "</div>",
        located,
    )


def hand_rows_to_csv(
    fieldnames: list[str],
    legs: list[dict[str, str]],
    reviews: dict[str, dict[str, str]],
) -> str:
    """One HAND's legs as a standalone CSV (July 2026, per-hand download).

    ``legs`` are the hand's rows in play (sequence) order; any reviewer-edited
    explanation in the sidecar overrides the generated one, same rule as the
    approved-pool download. Pure (browserless-tested); the grouped Review
    page's download button is a thin shell over this."""
    out: list[dict[str, str]] = []
    for row in legs:
        d = dict(row)
        override = reviews.get(str(d.get("No", "")), {}).get("explanation")
        if override:
            d["Answer Explanation"] = override
        out.append(d)
    return approved_rows_to_csv(fieldnames, out)
