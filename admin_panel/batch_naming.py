"""Batch file naming + creation-time ordering for the admin panel (July 2026).

Two user asks live here, both pure and browserless-testable:

* **Auto-descriptive filenames** (:func:`auto_plo_batch_name`): a PLO batch
  file names itself after its settings, starting with the TIME of day (no
  date, per the user: "just literally the time it was generated so I can
  organize by what time of day I made it"), e.g.
  ``21.47.32 · 9max · Hard · 12q · CO+BTN · vs 3-bet.csv``.
* **Stable newest-first ordering** (:func:`batch_creation_dt`): the Review
  picker used to sort by file MTIME, so editing/refreshing any old batch
  shoved it above a freshly generated one (the "my new batch doesn't show
  up first" glitch). Creation time never changes: parse the legacy
  ``_YYYYMMDD_HHMMSS`` filename stamp when present, else the filesystem
  birth time (macOS ``st_birthtime``; in-place rewrites like
  ``scripts/refresh_plo_batch.py`` truncate the same inode, so it
  survives refreshes), else mtime as the last resort.

INVARIANT: nothing in this module may import streamlit -- the Review/
Generate pages are thin shells over these functions (fix-durability rule).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# Legacy batch names end with the full creation stamp: <prefix>_YYYYMMDD_HHMMSS
_LEGACY_STAMP = re.compile(r"_(\d{8})_(\d{6})$")

# Compact filename forms of the PLO action-faced buckets.
_CONTEXT_SHORT = {
    "Opening": "open",
    "Facing single raise": "vs raise",
    "Facing 3-bet": "vs 3-bet",
    "Facing 4-bet+": "vs 4-bet+",
    "After one call": "after 1 call",
    "After multiple calls": "after calls",
}
# More selections than this and the segment just counts them -- a filename,
# not a manifest.
_MAX_LISTED = 3
_SEP = " · "


def _listed(values: list[str], noun: str) -> str:
    """``["CO","BTN"] -> "CO+BTN"``; too many -> ``"5 seats"``."""
    if len(values) > _MAX_LISTED:
        return f"{len(values)} {noun}"
    return "+".join(values)


def auto_plo_batch_name(
    *,
    now: datetime,
    table_size: int,
    difficulty_label: str,
    count: int,
    positions: list[str] | None = None,
    action_contexts: list[str] | None = None,
    player_counts: list[int] | None = None,
    custom_label: str = "",
) -> str:
    """The self-describing filename stem for a PLO batch.

    Segments, in order: time of day (``HH.MM.SS``, 24h so name-sorting is
    time-of-day order), the user's optional free-text label, pack table
    size, difficulty band, question count, then ONLY the filters that were
    actually set (blank multiselect = "any" = omitted). ``positions``
    should already be display seat names (the 6-max LJ->UTG remap happens
    at the caller, where the pack is known).
    """
    parts = [now.strftime("%H.%M.%S")]
    label = custom_label.strip()
    if label:
        parts.append(label)
    parts.append(f"{table_size}max")
    parts.append(difficulty_label)
    parts.append(f"{count}q")
    if positions:
        parts.append(_listed(list(positions), "seats"))
    if action_contexts:
        parts.append(
            _listed(
                [_CONTEXT_SHORT.get(c, c) for c in action_contexts], "contexts"
            )
        )
    if player_counts:
        parts.append(
            _listed(
                [
                    "open" if n == 1 else "HU" if n == 2 else f"{n}-way"  # noqa: PLR2004
                    for n in sorted(player_counts)
                ],
                "pot sizes",
            )
        )
    return _SEP.join(parts)


def dedupe_path(
    directory: Path,
    stem: str,
    suffix: str = ".csv",
    taken: frozenset[str] | set[str] = frozenset(),
) -> Path:
    """First non-colliding ``<stem>.csv`` / ``<stem> (2).csv`` / ... path.

    The time-only stamp means a same-second, same-settings batch on a
    DIFFERENT day would collide -- and a collision would silently
    overwrite a kept batch, so uniqueness is enforced here instead of by
    the name. ``taken`` adds filenames that are RESERVED but not yet on
    disk (July 2026: queued background batches haven't written their CSV
    yet -- two same-second Generate clicks must not share a name).
    """
    path = directory / f"{stem}{suffix}"
    n = 2
    while path.exists() or path.name in taken:
        path = directory / f"{stem} ({n}){suffix}"
        n += 1
    return path


def batch_creation_dt(path: Path) -> datetime:
    """When this batch was CREATED -- immune to edits, refreshes, regrades.

    Priority: the legacy ``_YYYYMMDD_HHMMSS`` name stamp (exact and
    portable), else filesystem birth time (macOS), else mtime. Missing
    files sort to the epoch rather than raising (the picker may race a
    deletion).
    """
    m = _LEGACY_STAMP.search(path.stem)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        stat = path.stat()
    except OSError:
        return datetime.fromtimestamp(0)
    birth = getattr(stat, "st_birthtime", None)
    return datetime.fromtimestamp(birth if birth else stat.st_mtime)


def plo_batch_display_label(path: Path) -> str:
    """Picker label: the stem plus enough date context to tell days apart.

    Legacy names keep their familiar ``prefix · YYYY-MM-DD HH:MM:SS`` form;
    new auto-names already start with the time, so only the DATE is
    appended (``21.47.32 · 9max · ... · Jul 16``).
    """
    m = _LEGACY_STAMP.search(path.stem)
    if m:
        try:
            when = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            prefix = path.stem[: m.start()] or path.stem
        except ValueError:
            pass
        else:
            return f"{prefix} · {when:%Y-%m-%d %H:%M:%S}"
    created = batch_creation_dt(path)
    return f"{path.stem} · {created:%b} {created.day}"


__all__ = [
    "auto_plo_batch_name",
    "batch_creation_dt",
    "dedupe_path",
    "plo_batch_display_label",
]
