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
# Neutral slate for a "holdings" grid (a player IN the pot but NOT acting on this
# street -- e.g. the villain who already checked back). Deliberately distinct
# from every ACTION colour: it was previously identical to COLOR_CALL (#4a9e5c),
# so a holdings grid read as "this player is calling", which is misleading (the
# player has no action here). Now clearly "just the range, no action".
COLOR_INRANGE = "#8a8f98"


# Tooltip styles for the tap-a-cell frequency readout. Emitted once per grid
# (repeating the block is harmless -- identical rules). CSS-only interaction:
# cells with a tooltip carry tabindex="0", so a TAP/CLICK focuses the cell and
# :focus-within shows the tooltip (hover works too on desktop). No JS -- st.html
# strips scripts, and Streamlit can't round-trip a cell click to Python.
_TIP_CSS = (
    "<style>"
    ".rgt td.rgc{cursor:pointer;}"
    ".rgt td.rgc:focus{outline:2px solid #ffffff88;outline-offset:-2px;}"
    ".rgt .rgtip{display:none;position:absolute;z-index:40;"
    "background:#11151c;color:#eee;border:1px solid #555;border-radius:6px;"
    "padding:6px 10px;font:12px/1.6 monospace;white-space:nowrap;"
    "box-shadow:0 3px 12px #000c;text-align:left;text-shadow:none;}"
    ".rgt td.rgc:focus-within .rgtip,.rgt td.rgc:hover .rgtip{display:block;}"
    ".rgt .rgtip-dn{top:100%;}.rgt .rgtip-up{bottom:100%;}"
    ".rgt .rgtip-l{left:0;}.rgt .rgtip-r{right:0;}"
    ".rgt .rgtip-c{left:50%;transform:translateX(-50%);}"
    ".rgt .rgsw{display:inline-block;width:9px;height:9px;border-radius:2px;"
    "margin-right:6px;vertical-align:middle;}"
    "</style>"
)


def _tip_pct(frac: float) -> str:
    """A tooltip percentage: whole numbers stay clean, tiny slivers readable."""
    pct = frac * 100.0
    if pct >= 1.0 or pct == 0.0:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def grid_html(
    segments_by_hand: dict[str, list[tuple]],
) -> str:
    """Render a 13x13 range grid as a self-contained HTML table string.

    ``segments_by_hand`` maps each 169-class label to a list of segments,
    each ``(fraction_in_[0,1], css_colour)`` or ``(fraction, colour, label)``,
    drawn as proportional vertical bands across the cell (left to right) -- so
    one cell can show a full action mix (fold/call/raise/all-in) like a GTO
    Wizard chart. Whatever a cell's bands don't cover shows the dark "empty"
    background.

    When a cell has any LABELLED segment, the cell becomes tappable
    (``tabindex``) and shows a tooltip listing each labelled action's
    frequency -- the "tap a hand to see its numbers" readout. Tooltip
    placement flips below/above and left/centre/right by grid position so it
    never clips at the edges. Pure (no Streamlit) so it unit-tests; the
    caller wraps it in ``st.html``.
    """
    any_tips = False
    rows: list[str] = []
    for i in range(13):
        cells: list[str] = []
        for j in range(13):
            hand = hand_at(i, j)
            segs = segments_by_hand.get(hand, [])
            bars = "".join(
                f'<div style="width:{max(0.0, min(1.0, s[0])) * 100:.1f}%;'
                f'background:{s[1]};"></div>'
                for s in segs
            )
            labelled = [s for s in segs if len(s) >= 3 and s[2]]  # noqa: PLR2004
            tip = ""
            td_attrs = ""
            if labelled:
                any_tips = True
                vert = "rgtip-dn" if i < 7 else "rgtip-up"  # noqa: PLR2004
                horiz = "rgtip-l" if j < 2 else ("rgtip-r" if j > 10 else "rgtip-c")  # noqa: PLR2004
                lines = "<br>".join(
                    f'<span class="rgsw" style="background:{s[1]};"></span>'
                    f"{s[2]} {_tip_pct(s[0])}"
                    for s in labelled
                )
                tip = (
                    f'<div class="rgtip {vert} {horiz}">'
                    f"<b>{hand}</b><br>{lines}</div>"
                )
                td_attrs = ' class="rgc" tabindex="0"'
            cells.append(
                f"<td{td_attrs} "
                'style="border:1px solid #0006;height:32px;padding:0;'
                "position:relative;text-align:center;vertical-align:middle;"
                'background:#222831;">'
                f'<div style="position:absolute;inset:0;display:flex;">{bars}</div>'
                '<span style="position:relative;z-index:1;color:#fff;'
                'font:600 11px monospace;text-shadow:0 0 2px #000,0 0 3px #000;">'
                f"{hand}</span>{tip}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = (
        '<table class="rgt" '
        'style="border-collapse:collapse;width:100%;table-layout:fixed;">'
        + "".join(rows)
        + "</table>"
    )
    return (_TIP_CSS + table) if any_tips else table


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
