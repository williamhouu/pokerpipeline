#!/usr/bin/env python3
"""Batch-generate 30 real training questions and surface their failure modes.

The brief's Phase-1 quality gate: run the full pipeline against ~30-50 spots,
read the output by hand, and let real failures drive what validators we need
in Layer 7. This is the script that produces the read-by-hand batch.

What it does:
  * loads the BTN-vs-BB test solve via Path Sampler;
  * buckets decision spots by street and walks each bucket in shuffled order;
  * generates a fixed `--per-street-target` (default 10) questions per street,
    so the batch is balanced across flop / turn / river -- one street's pool
    running short reports a clear warning and the run continues;
  * runs each survivor through Layer 5 -> Layer 6 -> Layer 8 with a SHARED
    Anthropic client, so the prompt cache hits on every call after the first;
  * writes test_output/batch_30_questions.csv (35 columns, all populated);
  * reports wall time, approximate API cost (from token usage + Sonnet 4.6
    pricing), per-street tally (evaluated/passed/generated/target), and
    distributions by street / hand class / concept tag.

This hits the real Anthropic API. It is NOT a test. CI never runs it. Skips
cleanly when ANTHROPIC_API_KEY is unset.

Usage:
    set ANTHROPIC_API_KEY=sk-...
    python scripts/batch_demo_30_questions.py [--pio-exe PATH] [--cfr PATH]
                                              [--per-street-target N] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.explanation_generator import (                             # noqa: E402
    ExplanationValidationError, explanation_to_row_overrides,
    generate_explanation, load_gold_examples, GOLD_EXAMPLE_COUNT,
)
from pipeline.fact_extractor import extract_facts                        # noqa: E402
from pipeline.format_writer import CSV_COLUMNS, build_row                # noqa: E402
from pipeline.path_sampler import PathSampler                            # noqa: E402
from pipeline.piosolver import PioSolverClient, find_piosolver           # noqa: E402
from pipeline.question_extractor import evaluate_spot                    # noqa: E402
from pipeline.scenario_config import get_scenario                        # noqa: E402

DEFAULT_CFR = REPO_ROOT / "test_solves" / "btn_vs_bb_srp_2cJs7s.cfr"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "test_output" / "batch_30_questions.csv"
RANDOM_SEED = 7
# Scenario metadata extract_facts attaches to every SpotData. The full scenario
# (stakes, table size, etc.) comes from pipeline.scenario_config.get_scenario;
# these are the smaller subset Layer 5 reads directly off the data block.
SCENARIO = {"preflop_raise_count": 1, "game_format": "cash",
            "active_players_on_flop": 2, "stack_depth_bb": 100}
STREETS = ("flop", "turn", "river")
# Sonnet 4.6 pricing (USD per 1M tokens). See Anthropic pricing page.
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
PRICE_CACHE_WRITE = 3.75              # 5-minute ephemeral cache write
PRICE_CACHE_READ = 0.30               # cache hit
# Backoff settings for transient API failures.
MAX_API_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0


@dataclass
class UsageTally:
    """Running token usage across every API call in the batch."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    api_calls: int = 0

    def add(self, usage) -> None:
        """Accumulate one Anthropic Messages response `usage` object."""
        self.api_calls += 1
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.cache_write_tokens += int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.cache_read_tokens += int(
            getattr(usage, "cache_read_input_tokens", 0) or 0)

    def cost_usd(self) -> float:
        """Dollar cost using published Sonnet 4.6 per-million-token rates."""
        return ((self.input_tokens * PRICE_INPUT
                 + self.output_tokens * PRICE_OUTPUT
                 + self.cache_write_tokens * PRICE_CACHE_WRITE
                 + self.cache_read_tokens * PRICE_CACHE_READ) / 1_000_000)


class _UsageRecordingClient:
    """Wraps an Anthropic client to record token usage + retry on transient errors.

    Layer 6 (`generate_explanation`) accepts any object with a `messages.create`
    method; we keep all the batch concerns (cost tracking, backoff) out of
    Layer 6 by injecting this wrapper instead.
    """

    def __init__(self, real_client, tally: UsageTally):
        self._client = real_client
        self._tally = tally
        # Anthropic SDK exposes the same nested attribute (`client.messages.create`);
        # we mimic that with an inner namespace object exposing create().
        from types import SimpleNamespace
        self.messages = SimpleNamespace(create=self._create_with_backoff)

    def _create_with_backoff(self, **kwargs):
        """Call messages.create with exponential backoff on transient errors."""
        import anthropic                       # imported lazily -- runtime dep
        attempt = 0
        last_exc: Exception | None = None
        while attempt < MAX_API_ATTEMPTS:
            attempt += 1
            try:
                response = self._client.messages.create(**kwargs)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self._tally.add(usage)
                return response
            except (anthropic.RateLimitError,
                    anthropic.APITimeoutError,
                    anthropic.APIConnectionError,
                    anthropic.InternalServerError) as exc:
                last_exc = exc
                delay = min(BACKOFF_CAP_SECONDS,
                            BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                print(f"    [retry {attempt}/{MAX_API_ATTEMPTS}] "
                      f"{type(exc).__name__}: {exc}. sleeping {delay:.1f}s",
                      file=sys.stderr)
                time.sleep(delay)
        # Out of retries -- re-raise the last exception for the caller to log.
        assert last_exc is not None
        raise last_exc


# --- stratified spot pool ---------------------------------------------------
def _bucket_by_street(nodes, seed: int) -> dict[str, list]:
    """Shuffle nodes (seeded) and bucket them by street.

    Returns a dict keyed by `flop`/`turn`/`river`; nodes from any other street
    are dropped (Tier 1 postflop solves only). Within each bucket order is
    randomised so successive runs sample different decision lines.
    """
    rng = random.Random(seed)
    buckets: dict[str, list] = {street: [] for street in STREETS}
    for node in nodes:
        if node.street in buckets:
            buckets[node.street].append(node)
    for street in STREETS:
        rng.shuffle(buckets[street])
    return buckets


# --- generation loop --------------------------------------------------------
@dataclass
class StreetTally:
    """Per-street counters for the run summary."""

    target: int = 0
    evaluated: int = 0                  # ran extract_facts on
    passed_layer4: int = 0              # is_question_worthy True
    generated: int = 0                  # Layer 6 produced a row
    pool_size: int = 0                  # total candidates available


@dataclass
class BatchResult:
    """Everything we collected during the run, for the summary print."""

    rows: list[dict] = field(default_factory=list)
    extract_skipped: int = 0
    extract_errors: list[str] = field(default_factory=list)
    layer6_validation_failures: list[str] = field(default_factory=list)
    layer6_api_failures: list[str] = field(default_factory=list)
    usage: UsageTally = field(default_factory=UsageTally)
    per_street: dict[str, StreetTally] = field(
        default_factory=lambda: {s: StreetTally() for s in STREETS})

    # Backwards-compat aggregates kept on the dataclass so the summary printer
    # can read both per-street and totals.
    @property
    def evaluated(self) -> int:
        return sum(t.evaluated for t in self.per_street.values())

    @property
    def filter_failed(self) -> int:
        return sum(t.evaluated - t.passed_layer4 for t in self.per_street.values())


def _generate_one(spot, client, scenario, result: BatchResult) -> bool:
    """Run Layer 6 once and append the resulting CSV row. Returns success."""
    try:
        explanation = generate_explanation(spot, client=client)
    except ExplanationValidationError as exc:
        result.layer6_validation_failures.append(str(exc)[:200])
        return False
    except Exception as exc:
        result.layer6_api_failures.append(f"{type(exc).__name__}: {exc}"[:200])
        return False

    row = build_row(spot, difficulty_score=evaluate_spot(spot).difficulty_score,
                    number=len(result.rows) + 1, scenario=scenario)
    row.update(explanation_to_row_overrides(explanation))
    result.rows.append(row)
    return True


def run_batch(sampler, big_blind: float, client, scenario, *,
              per_street_target: int) -> BatchResult:
    """Walk each street's bucket until `per_street_target` questions have been
    generated for that street (or the bucket runs out).

    Streets are processed flop -> turn -> river; total target is therefore
    3 * per_street_target. If a street's pool is too small to hit the target
    we keep going with the next street and the per-street tally records the
    shortfall. The Anthropic prompt cache hits regardless of street order
    (the system+gold-examples payload is identical across calls).
    """
    print("Enumerating decision spots ...")
    nodes = list(sampler.enumerate_decision_nodes(max_chance_children=2))
    buckets = _bucket_by_street(nodes, seed=RANDOM_SEED)
    total_target = per_street_target * len(STREETS)
    pool_sizes = {s: len(buckets[s]) for s in STREETS}
    print(f"  {len(nodes)} found "
          f"(flop={pool_sizes['flop']}, turn={pool_sizes['turn']}, "
          f"river={pool_sizes['river']}); "
          f"target {per_street_target}/street ({total_target} total).\n")

    result = BatchResult()
    for street in STREETS:
        tally = result.per_street[street]
        tally.target = per_street_target
        tally.pool_size = pool_sizes[street]
        for node in buckets[street]:
            if tally.generated >= per_street_target:
                break
            try:
                ctx = sampler.build_spot_context(node)
                spot = extract_facts(ctx, scenario=SCENARIO,
                                     big_blind=big_blind)
            except (ValueError, KeyError) as exc:
                result.extract_skipped += 1
                result.extract_errors.append(
                    f"{street} {node.node_id}: {exc}")
                continue
            tally.evaluated += 1

            verdict = evaluate_spot(spot)
            if not verdict.is_worthy:
                continue
            tally.passed_layer4 += 1

            ok = _generate_one(spot, client, scenario, result)
            if ok:
                tally.generated += 1
                print(f"  [{len(result.rows):>2d}/{total_target}] "
                      f"{street:<5s} ({tally.generated}/{per_street_target}) "
                      f"diff={verdict.difficulty_score:<4d} "
                      f"freq={verdict.top_action_frequency:.2f} "
                      f"ev_gap={verdict.ev_gap_bb:>5.2f}bb "
                      f"node={spot.spot_metadata.node_id}")
            else:
                print(f"  [skip] {spot.spot_metadata.node_id}: Layer 6 failed",
                      file=sys.stderr)

        if tally.generated < per_street_target:
            print(f"  [warn] {street}: only {tally.generated} of "
                  f"{per_street_target} target generated "
                  f"(evaluated {tally.evaluated}/{tally.pool_size} candidates, "
                  f"{tally.passed_layer4} passed Layer 4)", file=sys.stderr)

    return result


# --- summary printing -------------------------------------------------------
def _print_summary(result: BatchResult, elapsed: float) -> None:
    print()
    print("=" * 72)
    print(f"  spots evaluated         : {result.evaluated}")
    print(f"  spots skipped (extract) : {result.extract_skipped}")
    print(f"  spots failing filters   : {result.filter_failed}")
    print(f"  questions generated     : {len(result.rows)}")
    print(f"  Layer-6 validation fails: "
          f"{len(result.layer6_validation_failures)}")
    print(f"  Layer-6 API failures    : {len(result.layer6_api_failures)}")
    print()
    print("  Per-street tally (evaluated / passed Layer 4 / generated / target):")
    for street in STREETS:
        t = result.per_street[street]
        flag = "" if t.generated == t.target else "  <-- short of target"
        print(f"    {street:<6s} {t.evaluated:>4d} / {t.passed_layer4:>4d} / "
              f"{t.generated:>2d} / {t.target:>2d}   "
              f"(pool {t.pool_size}){flag}")
    print()
    u = result.usage
    print(f"  Anthropic calls         : {u.api_calls}")
    print(f"  input tokens            : {u.input_tokens:>9,}")
    print(f"  output tokens           : {u.output_tokens:>9,}")
    print(f"  cache-write tokens      : {u.cache_write_tokens:>9,}")
    print(f"  cache-read tokens       : {u.cache_read_tokens:>9,}  "
          f"(hits after request 1 should dominate)")
    print(f"  approx. cost            : ${u.cost_usd():.4f}")
    print(f"  wall-clock              : {elapsed:.1f}s "
          f"({elapsed / max(1, u.api_calls):.2f}s / API call)")
    print("=" * 72)

    if not result.rows:
        return

    streets = Counter(r["Hand Stage"] for r in result.rows)
    hand_classes = Counter(r["hand_class"] for r in result.rows)
    tag_counter: Counter = Counter()
    for r in result.rows:
        for tag in (r["concept_tags"] or "").split(", "):
            if tag:
                tag_counter[tag] += 1

    print("\n  By street:")
    for street, count in streets.most_common():
        print(f"    {street:<8s} {count}")
    print("\n  Top hand classes:")
    for hc, count in hand_classes.most_common(8):
        print(f"    {count:>2d}  {hc}")
    print("\n  Top concept tags:")
    for tag, count in tag_counter.most_common(12):
        print(f"    {count:>2d}  {tag}")

    if result.layer6_validation_failures:
        print("\n  Layer 6 validation failures (first 3):")
        for msg in result.layer6_validation_failures[:3]:
            print(f"    - {msg}")
    if result.layer6_api_failures:
        print("\n  Layer 6 API failures (first 3):")
        for msg in result.layer6_api_failures[:3]:
            print(f"    - {msg}")


# --- entry point ------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pio-exe", help="path to the PioSolver Edge executable")
    parser.add_argument("--cfr", type=Path, default=DEFAULT_CFR)
    parser.add_argument("--per-street-target", type=int, default=10,
                        help="questions to generate per street (flop/turn/river); "
                             "total = 3 * this (default 10 -> 30 total)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help="output CSV path (default test_output/batch_30_questions.csv)")
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- skipping the batch demo.")
        print("Set the key and rerun; this script never runs in CI.")
        return 0

    exe = find_piosolver(args.pio_exe)
    if exe is None:
        print("ERROR: PioSolver Edge not found (pass --pio-exe or set "
              "$PIOSOLVER_EXE).", file=sys.stderr)
        return 2
    if not args.cfr.is_file():
        print(f"ERROR: solve file not found: {args.cfr}", file=sys.stderr)
        return 2

    print(f"PioSolver : {exe}")
    print(f"Solve     : {args.cfr.name}")
    print(f"Target    : {args.per_street_target} questions/street "
          f"({args.per_street_target * len(STREETS)} total)\n")

    # Pre-load gold examples once so the cached payload is the same on every
    # Layer-6 call -- otherwise the cache key won't match call to call.
    gold_examples = list(load_gold_examples())[:GOLD_EXAMPLE_COUNT]
    print(f"Gold examples in-context: {len(gold_examples)}\n")

    # Build the Anthropic client once, wrap it for usage tracking + backoff.
    import anthropic
    base_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    usage = UsageTally()
    client = _UsageRecordingClient(base_client, usage)

    # Layer 6 will receive `client=client`; we also need to make
    # `generate_explanation` pass the same gold_examples on every call so the
    # cache hits. Patch via a closure rather than mutating the module.
    from pipeline import explanation_generator

    original = explanation_generator.generate_explanation

    def _generate(spot, **kwargs):
        kwargs.setdefault("client", client)
        kwargs.setdefault("gold_examples", gold_examples)
        return original(spot, **kwargs)

    explanation_generator.generate_explanation = _generate

    # Look up the scenario for this solve (raises a clear error if unregistered).
    scenario = get_scenario(args.cfr)
    print(f"Scenario  : {scenario.format} -- {scenario.stakes} -- "
          f"{scenario.preflop_action}\n")

    start = time.time()
    try:
        with PioSolverClient(exe) as pio:
            pio.load_tree(args.cfr)
            big_blind = (pio.show_effective_stack() or 100) / 100.0   # 100bb
            sampler = PathSampler(pio, oop_position=scenario.oop_position,
                                  ip_position=scenario.ip_position)
            result = run_batch(sampler, big_blind, client, scenario,
                               per_street_target=args.per_street_target)
    finally:
        explanation_generator.generate_explanation = original

    elapsed = time.time() - start
    result.usage = usage

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row)

    _print_summary(result, elapsed)
    try:
        out_display = args.out.relative_to(REPO_ROOT)
    except ValueError:
        out_display = args.out
    print(f"\n  output CSV: {out_display}")
    print("\nRead a sample of the generated rows by hand -- the brief's "
          "Phase-1 quality gate. Real failure modes here drive Layer 7's "
          "checker priorities.")
    return 0 if result.rows else 1


if __name__ == "__main__":
    sys.exit(main())
