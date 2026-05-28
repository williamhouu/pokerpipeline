"""Tests for admin_panel.usage.

Covers cost computation against the rate table, edge cases (unknown
model, zero tokens, cache tokens), and the JSONL log round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from admin_panel import usage


# --- compute_cost_usd ----------------------------------------------------
def test_cost_sonnet_basic() -> None:
    """1M input + 1M output on Sonnet 4.6 = $3 + $15 = $18."""
    cost = usage.compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(18.0)


def test_cost_opus_basic() -> None:
    """1M input + 1M output on Opus 4.7 = $15 + $75 = $90."""
    cost = usage.compute_cost_usd(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(90.0)


def test_cost_with_cache_tokens_sonnet() -> None:
    """Cache write is 1.25x input; cache read is 0.1x input.
    For Sonnet ($3/M input):
      1M cache_creation = $3 * 1.25 = $3.75
      1M cache_read     = $3 * 0.10 = $0.30
    """
    cost = usage.compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    assert cost == pytest.approx(3.75 + 0.30)


def test_cost_zero_tokens() -> None:
    """No tokens = no cost regardless of model."""
    assert (
        usage.compute_cost_usd(
            model="claude-opus-4-7", input_tokens=0, output_tokens=0
        )
        == 0.0
    )


def test_cost_unknown_model_returns_zero() -> None:
    """Pricing table miss returns 0 (rather than raising) so the UI
    never crashes on a new/typo'd model id."""
    cost = usage.compute_cost_usd(
        model="claude-something-not-real",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == 0.0


def test_cost_realistic_small_batch() -> None:
    """5 questions on Sonnet, each ~2000 in / ~300 out (back-of-envelope):
    input = 10000 * $3/M = $0.030
    output = 1500 * $15/M = $0.0225
    total ~ $0.0525.
    Sanity-check that small batches produce reasonable numbers."""
    cost = usage.compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=10_000,
        output_tokens=1_500,
    )
    assert cost == pytest.approx(0.0525)


# --- JSONL log round-trip ------------------------------------------------
def test_log_append_and_read(tmp_path: Path) -> None:
    log = tmp_path / "usage_log.jsonl"
    usage.append_log_entry(
        log,
        model="claude-sonnet-4-6",
        input_tokens=12_400,
        output_tokens=3_100,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        cost_usd=0.0837,
        questions_written=5,
        output_filename="batch.csv",
    )
    entries = usage.read_log_entries(log)
    assert len(entries) == 1
    e = entries[0]
    assert e.model == "claude-sonnet-4-6"
    assert e.input_tokens == 12_400
    assert e.output_tokens == 3_100
    assert e.cost_usd == pytest.approx(0.0837)
    assert e.questions_written == 5
    assert e.output_filename == "batch.csv"
    assert e.timestamp  # auto-populated, just check non-empty


def test_log_append_creates_parent_dir(tmp_path: Path) -> None:
    """Log path under a not-yet-existing dir is created."""
    log = tmp_path / "nested" / "subdir" / "usage_log.jsonl"
    usage.append_log_entry(
        log,
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        cost_usd=0.01,
        questions_written=1,
        output_filename="x.csv",
    )
    assert log.is_file()


def test_log_append_skips_empty_model(tmp_path: Path) -> None:
    """Dry-run / no-LLM batches send model='' -- those don't get logged."""
    log = tmp_path / "usage_log.jsonl"
    usage.append_log_entry(
        log,
        model="",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        cost_usd=0.0,
        questions_written=5,
        output_filename="dry.csv",
    )
    assert not log.exists()
    assert usage.read_log_entries(log) == []


def test_log_read_missing_file_returns_empty(tmp_path: Path) -> None:
    log = tmp_path / "nonexistent.jsonl"
    assert usage.read_log_entries(log) == []


def test_log_read_skips_malformed_lines(tmp_path: Path) -> None:
    """A SIGKILL mid-write could leave a partial line. Don't poison
    the whole log -- skip the bad line and keep going."""
    log = tmp_path / "log.jsonl"
    good_entry = {
        "timestamp": "2026-05-27T10:00:00",
        "model": "claude-sonnet-4-6",
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.01,
        "questions_written": 1,
        "output_filename": "x.csv",
    }
    log.write_text(
        json.dumps(good_entry) + "\n"
        + "not valid json{\n"
        + json.dumps(good_entry) + "\n",
        encoding="utf-8",
    )
    entries = usage.read_log_entries(log)
    assert len(entries) == 2  # malformed middle line skipped


def test_lifetime_stats_sums_all_entries(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    for i in range(3):
        usage.append_log_entry(
            log,
            model="claude-sonnet-4-6" if i < 2 else "claude-opus-4-7",
            input_tokens=1_000 * (i + 1),
            output_tokens=100 * (i + 1),
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cost_usd=0.01 * (i + 1),
            questions_written=5 * (i + 1),
            output_filename=f"batch{i}.csv",
        )
    stats = usage.compute_lifetime_stats(log)
    assert stats.total_batches == 3
    assert stats.total_questions == 5 + 10 + 15
    assert stats.total_input_tokens == 1000 + 2000 + 3000
    assert stats.total_output_tokens == 100 + 200 + 300
    assert stats.total_cost_usd == pytest.approx(0.01 + 0.02 + 0.03)
    assert stats.models_used == ("claude-opus-4-7", "claude-sonnet-4-6")


def test_lifetime_stats_empty_log_returns_zeros(tmp_path: Path) -> None:
    stats = usage.compute_lifetime_stats(tmp_path / "no.jsonl")
    assert stats.total_cost_usd == 0.0
    assert stats.total_batches == 0
    assert stats.models_used == ()


# --- format_cost ----------------------------------------------------------
def test_format_cost_sub_penny() -> None:
    """Under $0.01 shows 4 decimals so a $0.0008 estimate isn't '$0.00'."""
    assert usage.format_cost(0.0008) == "$0.0008"


def test_format_cost_cents() -> None:
    """Between $0.01 and $1 shows 3 decimals."""
    assert usage.format_cost(0.275) == "$0.275"


def test_format_cost_dollars() -> None:
    """$1 and up uses standard 2 decimals."""
    assert usage.format_cost(14.83) == "$14.83"
