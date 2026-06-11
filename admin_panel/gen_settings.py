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


__all__ = ["load_settings", "save_settings"]
