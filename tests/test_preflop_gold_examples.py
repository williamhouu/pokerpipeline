"""Tests for pipeline.preflop.gold_examples."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.gold_examples import (  # noqa: E402
    _is_preflop,
    load_preflop_gold_examples,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- _is_preflop -------------------------------------------------------------
def test_is_preflop_lowercase():
    assert _is_preflop({"Hand Stage": "preflop"}) is True


def test_is_preflop_capitalised():
    assert _is_preflop({"Hand Stage": "Preflop"}) is True


def test_is_preflop_with_whitespace():
    assert _is_preflop({"Hand Stage": "  Preflop  "}) is True


def test_not_preflop_for_postflop_stages():
    for stage in ("flop", "Flop", "turn", "Turn", "river", "River"):
        assert _is_preflop({"Hand Stage": stage}) is False, stage


def test_not_preflop_for_missing_stage():
    assert _is_preflop({}) is False


def test_not_preflop_for_non_string_stage():
    assert _is_preflop({"Hand Stage": None}) is False
    assert _is_preflop({"Hand Stage": 42}) is False


# --- load_preflop_gold_examples ---------------------------------------------
def test_load_preflop_examples_returns_only_preflop():
    """Sanity check against the real xlsx: every returned entry is preflop."""
    xlsx = REPO_ROOT / "docs" / "output_format_examples.xlsx"
    if not xlsx.is_file():
        pytest.skip("xlsx not present locally")
    examples = load_preflop_gold_examples()
    assert len(examples) >= 1
    for ex in examples:
        stage = ex.get("Hand Stage", "")
        assert isinstance(stage, str)
        assert stage.strip().lower() == "preflop", (
            f"non-preflop slipped through: {stage!r}"
        )


def test_load_preflop_examples_have_required_fields():
    """Each preflop example carries the fields Layer 6 will quote."""
    xlsx = REPO_ROOT / "docs" / "output_format_examples.xlsx"
    if not xlsx.is_file():
        pytest.skip("xlsx not present locally")
    examples = load_preflop_gold_examples()
    required = (
        "Question",
        "option 1",
        "option 2",
        "Correct Answer",
        "Answer Explanation",
    )
    for ex in examples:
        for field in required:
            value = ex.get(field)
            assert value, (
                f"missing or empty {field!r} in example: "
                f"{(ex.get('Question') or '')[:60]!r}"
            )


def test_load_preflop_examples_returns_tuple():
    """Returns a tuple (immutable so lru_cache can hash it)."""
    xlsx = REPO_ROOT / "docs" / "output_format_examples.xlsx"
    if not xlsx.is_file():
        pytest.skip("xlsx not present locally")
    examples = load_preflop_gold_examples()
    assert isinstance(examples, tuple)


def test_load_preflop_examples_subset_of_full_pool():
    """The preflop count is a strict subset of the full pool."""
    xlsx = REPO_ROOT / "docs" / "output_format_examples.xlsx"
    if not xlsx.is_file():
        pytest.skip("xlsx not present locally")
    from pipeline.explanation_generator import load_gold_examples
    full = load_gold_examples()
    preflop = load_preflop_gold_examples()
    assert 0 < len(preflop) <= len(full)
