"""Pure logic for the admin panel's Compare page (head-to-head prompt A/B).

Two batches are generated on the SAME spots (same RNG seed) with two
different prompts; this module joins their rows spot-by-spot and tracks a
win / lose / tie verdict per spot in a sidecar JSON next to batch A's CSV.

No Streamlit here -- unit-testable, same split as :mod:`admin_panel.review`.
Verdicts are keyed by a stable per-spot key (node_id + hand_class) rather
than row order, so a generation failure on one side can't misalign the join.
"""

from __future__ import annotations

import json
from pathlib import Path

# Verdict vocabulary: which prompt's output was better for a spot.
VERDICTS = ("A", "B", "tie")


def _node_id(solver_reference: str) -> str:
    """node_id = last '/'-segment of a solver_reference (<pack>/<actor>/<id>)."""
    return solver_reference.rsplit("/", 1)[-1] if solver_reference else ""


def spot_key(solver_reference: str, hand_class: str) -> str:
    """Stable per-spot key joining the two batches + keying verdicts.

    A spot is uniquely (node_id, hand_class); using that rather than the
    ``No`` column means the join survives row reordering / a missing row.
    """
    return f"{_node_id(solver_reference)}|{hand_class}"


def join_by_spot(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    """Pair batch-A and batch-B rows by spot, in A's order.

    Returns ``(spot_key, row_a, row_b)`` for every spot present in BOTH
    batches. Spots only one side produced (e.g. an LLM failure under one
    prompt) are skipped so the comparison only shows true head-to-heads.
    """
    b_by_key: dict[str, dict[str, str]] = {
        spot_key(r.get("solver_reference", ""), r.get("hand_class", "")): r
        for r in rows_b
    }
    out: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for row_a in rows_a:
        key = spot_key(
            row_a.get("solver_reference", ""), row_a.get("hand_class", "")
        )
        row_b = b_by_key.get(key)
        if row_b is not None:
            out.append((key, row_a, row_b))
    return out


def verdicts_path(base_csv: Path) -> Path:
    """Sidecar path holding the A/B/tie verdicts for a comparison."""
    return base_csv.with_suffix(".verdicts.json")


def load_verdicts(base_csv: Path) -> dict[str, str]:
    """Return ``{spot_key: 'A'|'B'|'tie'}``; ``{}`` if missing/malformed.

    Only well-formed verdicts survive, so a corrupt sidecar can't crash the
    page.
    """
    path = verdicts_path(base_csv)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if str(v) in VERDICTS}


def save_verdict(base_csv: Path, key: str, winner: str) -> None:
    """Upsert one spot's verdict.

    Raises:
        ValueError: if ``winner`` isn't one of :data:`VERDICTS`.
    """
    if winner not in VERDICTS:
        raise ValueError(f"winner must be one of {VERDICTS}, got {winner!r}")
    verdicts = load_verdicts(base_csv)
    verdicts[key] = winner
    path = verdicts_path(base_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdicts, indent=2), encoding="utf-8")


def tally(verdicts: dict[str, str]) -> dict[str, int]:
    """Count verdicts into ``{'A': n, 'B': n, 'tie': n}``."""
    out = {"A": 0, "B": 0, "tie": 0}
    for verdict in verdicts.values():
        if verdict in out:
            out[verdict] += 1
    return out


__all__ = [
    "VERDICTS",
    "join_by_spot",
    "load_verdicts",
    "save_verdict",
    "spot_key",
    "tally",
    "verdicts_path",
]
