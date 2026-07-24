"""Approved-pool balance report (pure logic, browserless-testable).

The balance machinery reports on what a batch GENERATES; the app receives
what the reviewer APPROVES. Layer-7 flags do not fall evenly across
question types -- measured July 23 2026 on the two fresh 20bb batches:
all-in / four-bet-pot spots survived the audit unflagged at ~30% while
fold/call spots survived at ~75-80%, so an approve-only-clean workflow
silently un-balances the app pool (jam spots and Medium difficulty starve
even though every batch was generated balanced). This module turns the
cross-batch approved rows (``review.collect_approved_rows``) into a
per-axis composition report so the reviewer sees the APP-side mix drift
while approving, not after.

Pure functions over plain row dicts (the CSV columns) -- no Streamlit, no
I/O -- per the fix-durability rule; the Review-page renderer is a thin
shell over :func:`approved_balance_report`.
"""

from __future__ import annotations

from collections import Counter

from pipeline.provenance import parse_notes

# The admin Generate difficulty presets (Easy 400-1300, Medium 1300-2100,
# Hard 2100-3200) -- the report must bucket with the SAME edges the
# generation filters use, or the two surfaces would disagree about what
# counts as "Medium".
_BAND_EDGES: tuple[tuple[str, float], ...] = (("Easy", 1300.0), ("Medium", 2100.0))


def difficulty_band(rating: object) -> str:
    """Preset band for a difficulty rating; ``"?"`` for a blank/junk cell."""
    try:
        value = float(str(rating).strip())
    except (TypeError, ValueError):
        return "?"
    for name, upper in _BAND_EDGES:
        if value < upper:
            return name
    return "Hard"


def balance_axes(row: dict[str, str]) -> dict[str, str]:
    """The balance-report bucket of one approved row on every axis.

    Axes mirror the generation-side balance axes where a CSV column carries
    them: difficulty band, the answer verb, the preflop pot type (the
    situation), hero's seat, and the source chart/pack (so a 20bb-vs-9max-vs-
    MTT mix is visible too). Missing columns bucket as ``"?"`` rather than
    raising -- old batches must never crash the report.
    """
    matchup = (row.get("Position Matchup") or "").strip()
    chart = parse_notes(row.get("Notes") or "").chart
    return {
        "Difficulty": difficulty_band(row.get("Difficulty Rating")),
        "Correct answer": (row.get("Correct Answer") or "?").strip() or "?",
        "Pot type": (row.get("Preflop Pot Type") or "?").strip() or "?",
        "Hero seat": matchup.split("_")[0] if matchup else "?",
        "Chart": chart or "?",
    }


def approved_balance_report(
    rows: list[dict[str, str]],
) -> list[tuple[str, list[tuple[str, int, float]]]]:
    """``[(axis, [(bucket, count, share), ...]), ...]`` over the approved pool.

    Buckets are sorted most-common first within each axis; ``share`` is the
    bucket's fraction of the pool (0..1). Empty pool -> empty list.
    """
    if not rows:
        return []
    counters: dict[str, Counter[str]] = {}
    for row in rows:
        for axis, bucket in balance_axes(row).items():
            counters.setdefault(axis, Counter())[bucket] += 1
    total = len(rows)
    return [
        (
            axis,
            [
                (bucket, count, count / total)
                for bucket, count in counter.most_common()
            ],
        )
        for axis, counter in counters.items()
    ]


def balance_warnings(
    rows: list[dict[str, str]], *, min_pool: int = 12
) -> list[str]:
    """Plain-English drift warnings for the approved pool, or ``[]``.

    Deliberately opinionated on ONE axis pair with a known failure mode
    (July 23 2026): the audit-survivorship tilt away from aggressive
    answers and harder questions. Fires only on pools of at least
    ``min_pool`` rows (tiny pools are all noise):

    - an "aggressive-answer" share (All-in/3-bet/4-bet/5-bet/Raise) below
      20% of the pool;
    - a Hard+Medium share below 25% of the pool.
    """
    if len(rows) < min_pool:
        return []
    warnings: list[str] = []
    aggressive = {"All-in", "3-bet", "4-bet", "5-bet", "Raise"}
    n_aggr = sum(
        1 for r in rows if (r.get("Correct Answer") or "").strip() in aggressive
    )
    if n_aggr / len(rows) < 0.20:
        warnings.append(
            f"Only {n_aggr} of {len(rows)} approved questions "
            f"({100 * n_aggr / len(rows):.0f}%) have an aggressive correct "
            "answer (raise/3-bet/all-in). The Layer-7 audit flags these "
            "spots the most, so approving only clean questions starves "
            "them -- consider fixing flagged jam/raise spots instead of "
            "skipping them."
        )
    n_hardish = sum(
        1
        for r in rows
        if difficulty_band(r.get("Difficulty Rating")) in ("Medium", "Hard")
    )
    if n_hardish / len(rows) < 0.25:
        warnings.append(
            f"Only {n_hardish} of {len(rows)} approved questions "
            f"({100 * n_hardish / len(rows):.0f}%) rate Medium or Hard. "
            "Harder questions draw more audit flags, so the clean-only "
            "pool drifts easy -- review flagged Medium/Hard questions to "
            "keep the app's difficulty mix honest."
        )
    return warnings


__all__ = [
    "approved_balance_report",
    "balance_axes",
    "balance_warnings",
    "difficulty_band",
]
