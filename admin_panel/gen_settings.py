"""Persist a Generate page's last-used settings to disk.

Pure logic (no Streamlit import) so it is unit-testable, same split as
:mod:`admin_panel.review`. The page snapshots its widget state to a JSON
file when a batch is launched, and re-seeds widget session state from that
file on every render -- so after a batch (or a panel restart) the page
still shows exactly the setup that batch ran with, and regenerating is one
click. The file is hidden (dot-prefixed) so the batch-dir ``*.csv`` globs
never see it.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_settings(path: Path) -> dict[str, object]:
    """The saved settings dict, or ``{}`` when missing/corrupt/not a dict.

    A broken file must never break the page -- worst case the widgets fall
    back to their hardcoded defaults.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(path: Path, settings: dict[str, object]) -> None:
    """Write the settings snapshot (creates the parent dir if needed).

    Values must be JSON-serializable; tuples (range sliders) become lists,
    which the page's sanitizers normalize back on load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def worthiness_bounds(
    saved: dict[str, object],
    *,
    min_key: str,
    max_key: str,
    legacy_key: str,
    default: tuple[int, int] = (65, 99),
    lo: int = 50,
    hi: int = 100,
) -> tuple[int, int]:
    """Seed values for the worthiness min/max NUMBER INPUTS.

    July 2026: the worthiness range SLIDER was replaced by two number
    inputs -- with both slider thumbs dragged to 100 (a legitimate
    "pure spots only" window) they stack on the track's right edge and can
    barely be separated again; the user was locked at 100/100. This reads
    the new per-bound keys first, falls back to the legacy ``[lo, hi]``
    slider list (so the last-used window survives the widget swap), then
    the default. Out-of-range/garbage values clamp to the default bound;
    an inverted pair is swapped rather than rejected.
    """
    legacy = saved.get(legacy_key)
    legacy_pair = (
        (legacy[0], legacy[1])
        if isinstance(legacy, (list, tuple)) and len(legacy) == 2  # noqa: PLR2004
        else default
    )

    def _bound(value: object, fallback: int) -> int:
        try:
            x = int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return fallback
        return min(max(x, lo), hi)

    v_min = _bound(saved.get(min_key, legacy_pair[0]), default[0])
    v_max = _bound(saved.get(max_key, legacy_pair[1]), default[1])
    if v_min > v_max:
        v_min, v_max = v_max, v_min
    return v_min, v_max


__all__ = ["load_settings", "save_settings", "worthiness_bounds"]
