"""Smoke test: the admin postflop Generate page renders without crashing.

Drives the real Streamlit app via ``AppTest`` (no browser): navigate to
Generate, switch to Postflop mode, and assert no exception. Catches UI-level
breakage (bad widget calls, missing imports, a crash in the solve picker) that
the pipeline unit tests don't see. Skips cleanly if Streamlit's test harness
isn't importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_APP = Path(__file__).resolve().parents[1] / "admin_panel" / "app.py"


def test_postflop_generate_page_renders() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.session_state["nav_page"] = "Generate"
    at.session_state["generate_mode"] = "Postflop"
    at.run()
    assert not at.exception, at.exception
    # The picker should have rendered the page heading (a solve found, or the
    # friendly "no solves" info) -- either way, no crash and some markdown.
    assert at.title  # "Generate questions"


def test_postflop_review_page_renders() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.session_state["nav_page"] = "Postflop Review"
    at.run()
    assert not at.exception, at.exception


def test_prompt_page_postflop_mode_renders() -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.session_state["nav_page"] = "Prompt"
    at.session_state["prompt_mode"] = "Postflop (editable)"
    at.run()
    assert not at.exception, at.exception
