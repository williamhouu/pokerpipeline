"""The preflop Generate page's Cash/Tournament format pre-filter (Aug 2026).

Drives the real app via AppTest (no browser): the format radio renders when
both cash and tournament packs exist on disk, Tournament narrows the pack
dropdown to the MTT bb-ante packs, and the page renders without the
stakes/venue widgets in tournament mode. Skips cleanly when the MTT pack
dirs (gitignored) aren't extracted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO = Path(__file__).resolve().parents[1]
_APP = _REPO / "admin_panel" / "app.py"

needs_mtt = pytest.mark.skipif(
    not (_REPO / "mtt8_15bb_ranges").is_dir(),
    reason="mtt8 pack files not extracted",
)


def _generate_page(AppTest):
    at = AppTest.from_file(str(_APP), default_timeout=180)
    at.session_state["nav_page"] = "Generate"
    at.session_state["generate_mode"] = "Preflop"  # page defaults to Postflop
    at.run()
    assert not at.exception, at.exception
    return at


def _radio(at, key: str):
    return next((r for r in at.radio if r.key == key), None)


@needs_mtt
def test_format_radio_renders_and_defaults_to_cash() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = _generate_page(AppTest)
    fmt = _radio(at, "preflop_gen_pack_format")
    assert fmt is not None, "format radio missing on the Generate page"
    assert fmt.value == "Cash"
    # Cash mode must not offer any MTT pack.
    pack_sel = next(s for s in at.selectbox if s.key == "preflop_gen_pack")
    assert all("mtt8" not in o for o in pack_sel.options)


@needs_mtt
def test_tournament_mode_lists_only_mtt_packs() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = _generate_page(AppTest)
    _radio(at, "preflop_gen_pack_format").set_value("Tournament (MTT)")
    at.run()
    assert not at.exception, at.exception
    pack_sel = next(s for s in at.selectbox if s.key == "preflop_gen_pack")
    assert pack_sel.options, "tournament mode lists no packs"
    assert all("mtt8" in o for o in pack_sel.options)
    # All seven extracted depths are offered.
    for depth in (10, 15, 20, 30, 50, 75, 300):
        assert any(f"mtt8_{depth}bb" in o for o in pack_sel.options)
    # Tournament mode hides the cash-only display widgets.
    stake_keys = [s.key for s in at.selectbox if s.key and "preflop_stakes" in s.key]
    venue_keys = [r.key for r in at.radio if r.key and "preflop_venue" in r.key]
    assert not stake_keys and not venue_keys
    # The currency radio is replaced by the tournament caption.
    assert _radio(at, "preflop_currency") is None


@needs_mtt
def test_switching_back_to_cash_resets_stale_selection() -> None:
    """INVARIANT (see _select_preflop_pack): a saved pack id from the other
    format must be reset, never crash the selectbox."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = _generate_page(AppTest)
    _radio(at, "preflop_gen_pack_format").set_value("Tournament (MTT)")
    at.run()
    assert not at.exception, at.exception
    # Pick an MTT pack, then flip back to Cash -- the stale MTT selection
    # must be replaced by a cash pack, not raise.
    pack_sel = next(s for s in at.selectbox if s.key == "preflop_gen_pack")
    pack_sel.set_value(pack_sel.options[-1])
    at.run()
    assert not at.exception, at.exception
    _radio(at, "preflop_gen_pack_format").set_value("Cash")
    at.run()
    assert not at.exception, at.exception
    pack_sel = next(s for s in at.selectbox if s.key == "preflop_gen_pack")
    assert "mtt8" not in pack_sel.value


@needs_mtt
def test_fully_balanced_checkbox_renders_and_defaults_on() -> None:
    """The 🎛️ Fully balanced checkbox (preflop twin of the PLO/full-hand
    buttons) renders in preflop Generate mode, defaults ON, and disables
    the plain diversify toggle while checked."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = _generate_page(AppTest)
    cb = next(
        (c for c in at.checkbox if c.key == "preflop_fully_balanced"), None
    )
    assert cb is not None, "fully-balanced checkbox missing"
    assert cb.value is True
    div = next(
        (c for c in at.checkbox if c.key == "preflop_diversify"), None
    )
    assert div is not None and div.disabled


@needs_mtt
def test_all_depths_toggle_repurposes_generate_button() -> None:
    """The 🏆 all-depths TOGGLE (bet-sizing-trainer pattern, Aug 2026)
    renders only in Tournament format; when ON, the main GENERATE button
    becomes the all-depths launcher while every other setting (prompt,
    style, count) stays user-adjustable."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = _generate_page(AppTest)
    # Cash mode: no toggle.
    assert not any(
        t.key == "preflop_all_depths_toggle" for t in at.toggle
    )
    _radio(at, "preflop_gen_pack_format").set_value("Tournament (MTT)")
    at.run()
    assert not at.exception, at.exception
    tog = next(
        (t for t in at.toggle if t.key == "preflop_all_depths_toggle"), None
    )
    assert tog is not None, "all-depths toggle missing in tournament mode"
    assert tog.value is False  # off by default
    tog.set_value(True)
    at.run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button if b.key and "preflop_generate_btn" in b.key]
    assert any("ALL TOURNAMENT DEPTHS" in label for label in labels), labels
    # The prompt picker and style radio remain available in this mode.
    assert any(r.key == "preflop_answer_style" for r in at.radio)
