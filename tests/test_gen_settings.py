"""Tests for admin_panel.gen_settings (Generate-page settings persistence)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel.gen_settings import load_settings, save_settings  # noqa: E402


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "batches" / ".plo_generate_settings.json"  # parent missing
    settings = {
        "plo_gen_count": 12,
        "plo_gen_amounts": "Big blinds",
        "plo_worthiness_slider": (60, 99),  # tuple -> list -> tuple via sanitizer
        "plo_gen_positions": ["SB", "BB"],
        "plo_gen_pin_seed": False,
    }
    save_settings(path, settings)
    loaded = load_settings(path)
    assert loaded["plo_gen_count"] == 12
    assert loaded["plo_worthiness_slider"] == [60, 99]  # JSON has no tuples
    assert loaded["plo_gen_positions"] == ["SB", "BB"]


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "nope.json") == {}


def test_corrupt_or_non_dict_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_settings(p) == {}
    p.write_text('["a", "list"]', encoding="utf-8")
    assert load_settings(p) == {}


# --- hero-seat auto-select per solve (Aug 2026, user ask) ---------------------
def test_seed_heroes_selects_all_on_new_solve() -> None:
    """First render of a solve: every seat selected, tag recorded."""
    from admin_panel.gen_settings import seed_heroes_for_solve

    state: dict = {}
    seed_heroes_for_solve(
        state, key="h", tag_key="t", solve_tag="utg_sb.db",
        options=["UTG", "SB"],
    )
    assert state["h"] == ["UTG", "SB"] and state["t"] == "utg_sb.db"


def test_seed_heroes_resets_on_solve_switch() -> None:
    """The observed bug: BTN/BB picks survive into a UTG/SB solve, where
    Streamlit silently drops them (empty picker). Switching solves must
    re-select ALL of the new solve's seats."""
    from admin_panel.gen_settings import seed_heroes_for_solve

    state = {"h": ["BTN", "BB"], "t": "btn_bb.db"}
    seed_heroes_for_solve(
        state, key="h", tag_key="t", solve_tag="utg_sb.db",
        options=["UTG", "SB"],
    )
    assert state["h"] == ["UTG", "SB"] and state["t"] == "utg_sb.db"


def test_seed_heroes_keeps_subset_within_same_solve() -> None:
    """A deliberate one-seat pick sticks across reruns of the SAME solve
    (including an emptied picker -- the launch button blocks on it)."""
    from admin_panel.gen_settings import seed_heroes_for_solve

    state = {"h": ["SB"], "t": "utg_sb.db"}
    seed_heroes_for_solve(
        state, key="h", tag_key="t", solve_tag="utg_sb.db",
        options=["UTG", "SB"],
    )
    assert state["h"] == ["SB"]

    state = {"h": [], "t": "utg_sb.db"}
    seed_heroes_for_solve(
        state, key="h", tag_key="t", solve_tag="utg_sb.db",
        options=["UTG", "SB"],
    )
    assert state["h"] == []


def test_seed_heroes_repairs_stale_seats_same_tag() -> None:
    """Defensive: stored seats that don't exist for this solve reset to
    all even when the tag matches (edited/renamed state)."""
    from admin_panel.gen_settings import seed_heroes_for_solve

    state = {"h": ["BTN"], "t": "utg_sb.db"}
    seed_heroes_for_solve(
        state, key="h", tag_key="t", solve_tag="utg_sb.db",
        options=["UTG", "SB"],
    )
    assert state["h"] == ["UTG", "SB"]


def test_seed_heroes_missing_key_seeds_all() -> None:
    """Tag present but the widget key vanished (cleared state): reseed."""
    from admin_panel.gen_settings import seed_heroes_for_solve

    state = {"t": "other.db"}
    seed_heroes_for_solve(
        state, key="h", tag_key="t", solve_tag="utg_sb.db",
        options=["UTG", "SB"],
    )
    assert state["h"] == ["UTG", "SB"]


# --- EV-crown tie tolerance (Aug 9 2026, pot-relative) ------------------------
def test_ev_tie_tolerance_pot_relative() -> None:
    """The observed confusion: Call +0.96bb crowned over the 62%-fold
    correct answer on a 116bb pot. 1% of pot (1.16bb) now reads that gap
    as tied; small pots keep the legacy 0.10bb floor exactly."""
    from admin_panel.review import ev_tie_tolerance

    assert ev_tie_tolerance("116BB") == 1.16
    assert 0.96 < ev_tie_tolerance("116BB")          # the observed row: tied
    assert ev_tie_tolerance("8BB") == 0.10           # small pot: legacy floor
    assert ev_tie_tolerance(None) == 0.10            # missing POT
    assert ev_tie_tolerance("$232") == 0.10          # dollar display: floor
