"""Tests for admin_panel.range_view -- the pure 13x13 grid helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from admin_panel import range_view  # noqa: E402


def test_hand_at_corners_and_triangles() -> None:
    assert range_view.hand_at(0, 0) == "AA"  # top-left diagonal
    assert range_view.hand_at(12, 12) == "22"  # bottom-right diagonal
    assert range_view.hand_at(0, 1) == "AKs"  # upper-right = suited
    assert range_view.hand_at(1, 0) == "AKo"  # lower-left = offsuit
    assert range_view.hand_at(2, 4) == "QTs"
    assert range_view.hand_at(4, 2) == "QTo"


def test_combos_by_cell_type() -> None:
    assert range_view.combos(0, 0) == 6  # pair
    assert range_view.combos(0, 1) == 4  # suited
    assert range_view.combos(1, 0) == 12  # offsuit


def test_grid_matrix_shape_and_scaling() -> None:
    mat = range_view.grid_matrix({"AA": 1.0, "AKs": 0.5})
    assert len(mat) == 13
    assert all(len(row) == 13 for row in mat)
    assert mat[0][0] == 100.0  # AA at 1.0 -> 100%
    assert mat[0][1] == 50.0  # AKs at 0.5 -> 50%
    assert mat[5][5] == 0.0  # 99 absent -> 0


def test_range_pct_full_range_is_100() -> None:
    full = {range_view.hand_at(i, j): 1.0 for i in range(13) for j in range(13)}
    assert round(range_view.range_pct(full), 6) == 100.0


def test_range_pct_weights_by_combos() -> None:
    # One pair (6 combos) fully in: 6/1326.
    assert round(range_view.range_pct({"AA": 1.0}), 4) == round(6 / 1326 * 100, 4)
    # One offsuit class (12 combos) fully in: 12/1326.
    assert round(range_view.range_pct({"AKo": 1.0}), 4) == round(12 / 1326 * 100, 4)


def test_node_id_from_solver_reference() -> None:
    ref = "ryan_preflop_tree/CO/UTG_Fold_HJ_60%_CO_decision"
    assert (
        range_view.node_id_from_solver_reference(ref) == "UTG_Fold_HJ_60%_CO_decision"
    )
    assert range_view.node_id_from_solver_reference("") == ""
    assert range_view.node_id_from_solver_reference("just_a_node") == "just_a_node"


def test_cell_css_scales_green_with_frequency() -> None:
    # 0% -> transparent green + dark text; 100% -> full green + white text.
    assert range_view.cell_css(0) == (
        "background-color: rgba(38, 139, 38, 0.000); color: #111"
    )
    full = range_view.cell_css(100)
    assert "rgba(38, 139, 38, 1.000)" in full
    assert "white" in full
    assert "0.500" in range_view.cell_css(50)


def test_cell_css_clamps_out_of_range() -> None:
    assert "1.000" in range_view.cell_css(150)
    assert "0.000" in range_view.cell_css(-10)


def test_grid_html_is_a_table_with_all_169_cells() -> None:
    html = range_view.grid_html({})
    assert html.startswith("<table")
    assert html.count("<td") == 169
    for hand in ("AA", "AKs", "AKo", "72o", "22"):
        assert hand in html


def test_grid_html_renders_segment_widths_and_colors() -> None:
    html = range_view.grid_html({"AA": [(0.7, "#abc123"), (0.3, "#def456")]})
    assert "width:70.0%;background:#abc123" in html
    assert "width:30.0%;background:#def456" in html


def test_grid_html_clamps_fraction_to_100() -> None:
    assert "width:100.0%" in range_view.grid_html({"AA": [(1.5, "#abc123")]})


# --- tap-a-cell frequency tooltip (July 2026) -------------------------------
def test_grid_html_labelled_segments_render_tooltip() -> None:
    """A cell with labelled segments becomes tappable (tabindex) and carries a
    tooltip listing each action's frequency; the style block is included."""
    html = range_view.grid_html(
        {"AA": [(0.7, "#c2492f", "Raise 3"), (0.3, "#5b8fb0", "Fold")]}
    )
    assert "<style>" in html and ".rgtip" in html
    assert 'tabindex="0"' in html
    assert "Raise 3 70%" in html and "Fold 30%" in html
    # The bars still render alongside the tooltip.
    assert "width:70.0%" in html and "width:30.0%" in html


def test_grid_html_unlabelled_segments_unchanged() -> None:
    """Plain (frac, color) segments keep the old behaviour: no tooltip, no
    tabindex, no style block -- full backward compatibility."""
    html = range_view.grid_html({"AA": [(0.7, "#abc123"), (0.3, "#def456")]})
    assert "<style>" not in html
    assert "tabindex" not in html
    assert "rgtip" not in html


def test_grid_html_tooltip_placement_flips_at_edges() -> None:
    """Top rows tip below (rgtip-dn), bottom rows above (rgtip-up); leftmost
    columns anchor left, rightmost anchor right, middle centre -- so a tooltip
    never clips outside the grid."""
    seg = [(1.0, "#c2492f", "Raise")]
    top_left = range_view.grid_html({"AA": seg})       # row 0, col 0
    assert "rgtip-dn" in top_left and "rgtip-l" in top_left
    bottom_right = range_view.grid_html({"22": seg})   # row 12, col 12
    assert "rgtip-up" in bottom_right and "rgtip-r" in bottom_right
    middle = range_view.grid_html({"87s": seg})        # row 6, col 7 -> dn + centre
    assert "rgtip-dn" in middle and "rgtip-c" in middle


def test_grid_html_tooltip_pct_formatting() -> None:
    """Whole percentages render clean; sub-1% slivers keep one decimal."""
    html = range_view.grid_html({"AA": [(0.845, "#c2492f", "Raise"), (0.005, "#5b8fb0", "Fold")]})
    assert "Raise 84%" in html or "Raise 85%" in html
    assert "Fold 0.5%" in html
