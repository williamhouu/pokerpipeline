"""Pure helpers for rendering preflop ranges as a 13x13 grid.

No Streamlit / pandas here -- just the grid math, so it's unit-testable.
The admin panel turns :func:`grid_matrix` into a colored DataFrame.

Grid convention (rows + columns run A,K,Q,...,2): upper-right triangle =
suited, the diagonal = pairs, lower-left triangle = offsuit -- the standard
poker range chart.
"""

from __future__ import annotations

RANKS = "AKQJT98765432"


def hand_at(i: int, j: int) -> str:
    """The 169-class label for grid cell (row ``i``, col ``j``)."""
    hi, lo = RANKS[i], RANKS[j]
    if i == j:
        return hi + lo
    if i < j:
        return hi + lo + "s"  # upper-right = suited
    return lo + hi + "o"  # lower-left = offsuit


def combos(i: int, j: int) -> int:
    """Combinatoric weight of cell (i, j): 6 pairs, 4 suited, 12 offsuit."""
    if i == j:
        return 6
    return 4 if i < j else 12


def grid_matrix(weights: dict[str, float]) -> list[list[float]]:
    """13x13 matrix of PERCENTAGES (0-100) for a range, in ``RANKS`` order.

    ``weights`` maps a 169-class label to a frequency in [0, 1] (the pack's
    raw value); missing classes are treated as 0.
    """
    return [
        [weights.get(hand_at(i, j), 0.0) * 100.0 for j in range(13)]
        for i in range(13)
    ]


def range_pct(weights: dict[str, float]) -> float:
    """Combo-weighted size of a range, as a percentage of all 1326 combos."""
    total = 0.0
    for i in range(13):
        for j in range(13):
            total += weights.get(hand_at(i, j), 0.0) * combos(i, j)
    return total / 1326.0 * 100.0


def node_id_from_solver_reference(solver_reference: str) -> str:
    """The node id is the last ``/``-segment of a ``solver_reference``
    (``<pack>/<actor>/<node_id>``). Empty string for a blank reference."""
    return solver_reference.rsplit("/", 1)[-1] if solver_reference else ""


# GTO-Wizard-style action colours for the strategy grid.
COLOR_FOLD = "#5b8fb0"
COLOR_CALL = "#4a9e5c"
COLOR_RAISE = "#c2492f"
COLOR_ALLIN = "#7a1f1f"
COLOR_INRANGE = "#4a9e5c"


def grid_html(segments_by_hand: dict[str, list[tuple[float, str]]]) -> str:
    """Render a 13x13 range grid as a self-contained HTML table string.

    ``segments_by_hand`` maps each 169-class label to a list of
    ``(fraction_in_[0,1], css_colour)`` segments, drawn as proportional
    vertical bands across the cell (left to right) -- so one cell can show a
    full action mix (fold/call/raise/all-in) like a GTO Wizard chart.
    Whatever a cell's bands don't cover shows the dark "empty" background.
    Pure (no Streamlit) so it unit-tests; the caller wraps it in ``st.html``.
    """
    rows: list[str] = []
    for i in range(13):
        cells: list[str] = []
        for j in range(13):
            hand = hand_at(i, j)
            bars = "".join(
                f'<div style="width:{max(0.0, min(1.0, frac)) * 100:.1f}%;'
                f'background:{color};"></div>'
                for frac, color in segments_by_hand.get(hand, [])
            )
            cells.append(
                '<td style="border:1px solid #0006;height:32px;padding:0;'
                "position:relative;text-align:center;vertical-align:middle;"
                'background:#222831;">'
                f'<div style="position:absolute;inset:0;display:flex;">{bars}</div>'
                '<span style="position:relative;z-index:1;color:#fff;'
                'font:600 11px monospace;text-shadow:0 0 2px #000,0 0 3px #000;">'
                f"{hand}</span></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table style="border-collapse:collapse;width:100%;table-layout:fixed;">'
        + "".join(rows)
        + "</table>"
    )


def cell_css(value: float) -> str:
    """CSS for one range-grid cell: green intensity scales with the cell's
    frequency (a percentage in [0, 100]).

    Pure Python on purpose -- pandas' ``Styler.background_gradient`` requires
    matplotlib, which isn't a project dependency, so we colour the grid
    ourselves (used via ``Styler.map``).
    """
    try:
        frac = max(0.0, min(1.0, float(value) / 100.0))
    except (TypeError, ValueError):
        return ""
    text = "white" if frac > 0.55 else "#111"
    return f"background-color: rgba(38, 139, 38, {frac:.3f}); color: {text}"


__all__ = [
    "COLOR_ALLIN",
    "COLOR_CALL",
    "COLOR_FOLD",
    "COLOR_INRANGE",
    "COLOR_RAISE",
    "RANKS",
    "cell_css",
    "combos",
    "grid_html",
    "grid_matrix",
    "hand_at",
    "node_id_from_solver_reference",
    "range_pct",
]
