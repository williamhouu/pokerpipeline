"""Cost computation + persistence for Anthropic API usage.

Lives in the admin_panel layer (not the pipeline) because the pipeline
shouldn't know about pricing -- it only emits token counts. This module
applies a per-model rate table, formats a per-batch summary, and
appends each completed batch to a JSONL log so a lifetime total
survives admin-panel restarts.

Rates are list prices for Anthropic's Opus / Sonnet models. They can
and do change; bump :data:`PRICING_PER_M_TOKENS` when they shift. The
admin panel reads this table to render the per-batch cost in the
result UI and a "Lifetime spend" widget in the sidebar.

The log file is ``test_output/usage_log.jsonl`` -- one JSON object per
line per batch -- gitignored. Compact and append-only so a SIGKILL'd
admin panel can't lose a row mid-write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Per-million-token list prices, in USD. Cache write is 1.25x base
# input; cache read is 0.1x base input -- consistent across models.
# Update when Anthropic publishes new pricing.
PRICING_PER_M_TOKENS: dict[str, dict[str, float]] = {
    # Opus 4.x list price is $5/$25 -- the old $15/$75 here was the
    # Opus-4.1-era price (corrected June 2026; earlier Opus batches were
    # over-counted ~3x in the lifetime-spend metric).
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    # Backward-compat aliases in case anything in the project still
    # uses an older model id. Same Opus/Sonnet families, same prices.
    "claude-opus-4-5": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
}
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1


@dataclass(frozen=True)
class UsageTotals:
    """Token totals + computed cost for one batch (or summed across many).

    Cost is in USD, computed from the model + token counts via the
    list rates in :data:`PRICING_PER_M_TOKENS`. ``model`` may be empty
    (e.g. lifetime totals where multiple models contribute) -- the
    cost is still right because it was summed at per-batch time.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )


def compute_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Apply the per-model rate table to a token count and return USD.

    Unknown models score 0 (rather than raising) so the UI never blows
    up on a new model id that pre-dates a pricing-table update. The
    sidebar surfaces "unknown model" as a hint when this happens.
    """
    rates = PRICING_PER_M_TOKENS.get(model)
    if rates is None:
        return 0.0
    in_rate = rates["input"]
    out_rate = rates["output"]
    return (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_creation_tokens * in_rate * _CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * in_rate * _CACHE_READ_MULTIPLIER
    ) / 1_000_000


@dataclass(frozen=True)
class LogEntry:
    """One line of the usage log, deserialised."""

    timestamp: str  # ISO-8601
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float
    questions_written: int
    output_filename: str

    @classmethod
    def from_jsonl(cls, line: str) -> LogEntry:
        data = json.loads(line)
        return cls(
            timestamp=str(data.get("timestamp", "")),
            model=str(data.get("model", "")),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cache_creation_tokens=int(data.get("cache_creation_tokens", 0)),
            cache_read_tokens=int(data.get("cache_read_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            questions_written=int(data.get("questions_written", 0)),
            output_filename=str(data.get("output_filename", "")),
        )


@dataclass(frozen=True)
class LifetimeStats:
    """Summed totals across every log entry."""

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_questions: int = 0
    total_batches: int = 0
    models_used: tuple[str, ...] = field(default_factory=tuple)


def append_log_entry(
    log_path: Path,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    cost_usd: float,
    questions_written: int,
    output_filename: str,
) -> None:
    """Append one batch's usage to the JSONL log. Creates the file if missing.

    ``model=""`` (dry-runs) is dropped -- there's nothing useful to
    record. Callers that want a "dry-run-happened" trail can write it
    elsewhere.
    """
    if not model:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cost_usd": round(cost_usd, 6),
        "questions_written": questions_written,
        "output_filename": output_filename,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_log_entries(log_path: Path) -> list[LogEntry]:
    """Parse every line of the log into LogEntry. Missing file -> []."""
    if not log_path.is_file():
        return []
    entries: list[LogEntry] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(LogEntry.from_jsonl(line))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Skip malformed lines rather than crashing the UI -- a
            # half-written line from a SIGKILL'd write would otherwise
            # poison the whole sidebar widget.
            continue
    return entries


def compute_lifetime_stats(log_path: Path) -> LifetimeStats:
    """Sum every log entry. Used by the sidebar's lifetime widget."""
    entries = read_log_entries(log_path)
    if not entries:
        return LifetimeStats()
    models_seen: set[str] = set()
    total_cost = 0.0
    total_in = 0
    total_out = 0
    total_cc = 0
    total_cr = 0
    total_q = 0
    for entry in entries:
        models_seen.add(entry.model)
        total_cost += entry.cost_usd
        total_in += entry.input_tokens
        total_out += entry.output_tokens
        total_cc += entry.cache_creation_tokens
        total_cr += entry.cache_read_tokens
        total_q += entry.questions_written
    return LifetimeStats(
        total_cost_usd=total_cost,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cache_creation_tokens=total_cc,
        total_cache_read_tokens=total_cr,
        total_questions=total_q,
        total_batches=len(entries),
        models_used=tuple(sorted(models_seen)),
    )


def format_cost(cost_usd: float) -> str:
    """Render a USD amount for UI display. Always 4 decimals under $1."""
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    if cost_usd < 1:
        return f"${cost_usd:.3f}"
    return f"${cost_usd:.2f}"


__all__ = [
    "PRICING_PER_M_TOKENS",
    "LifetimeStats",
    "LogEntry",
    "UsageTotals",
    "append_log_entry",
    "compute_cost_usd",
    "compute_lifetime_stats",
    "format_cost",
    "read_log_entries",
]
