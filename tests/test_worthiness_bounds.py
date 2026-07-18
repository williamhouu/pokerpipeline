"""Tests for gen_settings.worthiness_bounds (July 16 2026).

The PLO worthiness range slider was replaced by two number inputs after the
user got locked at a 100/100 window (both slider thumbs stacked on the
track's right edge are nearly impossible to separate, and every page switch
re-seeded the stuck pair from the settings file). This helper seeds the new
inputs, migrating the legacy slider list -- pinned here browserless per the
fix-durability rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel.gen_settings import worthiness_bounds  # noqa: E402

_KW = {
    "min_key": "plo_worthy_min",
    "max_key": "plo_worthy_max",
    "legacy_key": "plo_worthiness_slider",
}


def test_migrates_the_legacy_slider_pair():
    # The exact stuck state from the user's settings file: last-used values
    # survive the widget swap (now trivially editable as typed numbers).
    assert worthiness_bounds({"plo_worthiness_slider": [100, 100]}, **_KW) == (100, 100)
    assert worthiness_bounds({"plo_worthiness_slider": [65, 99]}, **_KW) == (65, 99)


def test_new_keys_win_over_legacy():
    saved = {
        "plo_worthiness_slider": [100, 100],
        "plo_worthy_min": 70,
        "plo_worthy_max": 95,
    }
    assert worthiness_bounds(saved, **_KW) == (70, 95)


def test_defaults_when_nothing_saved():
    assert worthiness_bounds({}, **_KW) == (65, 99)


def test_garbage_clamps_and_falls_back():
    assert worthiness_bounds({"plo_worthy_min": "x", "plo_worthy_max": 400}, **_KW) == (65, 100)
    assert worthiness_bounds({"plo_worthiness_slider": "junk"}, **_KW) == (65, 99)
    assert worthiness_bounds({"plo_worthiness_slider": [10, 999]}, **_KW) == (50, 100)


def test_inverted_pair_is_swapped():
    assert worthiness_bounds({"plo_worthy_min": 95, "plo_worthy_max": 70}, **_KW) == (70, 95)


def test_worthiness_inputs_commit_in_the_real_page():
    """AppTest regression for the stuck-window bug: the two number inputs
    must exist on the PLO Generate page, seed from the saved snapshot, and
    a typed value must drive the "Numbers in effect" line. (The old range
    slider could get stuck at 100/100 with both thumbs stacked; a widget
    swap that silently broke the commit path would pass the pure tests
    above, so this drives the actual page.)"""
    import pytest

    repo = Path(__file__).resolve().parent.parent
    if not any((repo / d).is_dir() for d in ("plo9_ranges", "plo_ranges")):
        pytest.skip("no PLO pack on this machine; the page returns early")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(repo / "admin_panel" / "app.py"), default_timeout=120)
    at.session_state["nav_page"] = "PLO Generate"
    at.run()
    mins = [n for n in at.number_input if n.key == "plo_worthy_min"]
    maxs = [n for n in at.number_input if n.key == "plo_worthy_max"]
    assert mins and maxs, "worthiness number inputs missing from the page"

    mins[0].set_value(65).run()
    maxs = [n for n in at.number_input if n.key == "plo_worthy_max"]
    maxs[0].set_value(99).run()
    effect = [i.body for i in at.info if "Numbers in effect" in i.body]
    assert effect, "Numbers-in-effect line missing"
    flat = effect[0].replace("**", "")
    assert "worthiness 65–99%" in flat or "worthiness 65-99%" in flat
