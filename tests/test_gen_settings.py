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
