#!/usr/bin/env python3
"""Batch-generate 30 real training questions and surface their failure modes.

The brief's Phase-1 quality gate: run the full pipeline against ~30-50 spots,
read the output by hand, and let real failures drive what validators we need
in Layer 7. This is the script that produces the read-by-hand batch.

What it does:
  * loads the BTN-vs-BB test solve via Path Sampler;
  * stratifies decision spots across flop / turn / river so the batch isn't
    100% flop;
  * walks the stratified pool until 30 spots survive Layer 4's filters;
  * runs each survivor through Layer 5 -> Layer 6 -> Layer 8 with a SHARED
    Anthropic client, so the prompt cache hits on every call after the first;
  * writes test_output/batch_30_questions.csv (35 columns, all populated);
  * reports wall time, approximate API cost (from token usage + Sonnet 4.6
    pricing), and distributions by street / hand class / concept tag.

This hits the real Anthropic API. It is NOT a test. CI never runs it. Skips
cleanly when ANTHROPIC_API_KEY is unset.

Usage:
    set ANTHROPIC_API_KEY=sk-...
    python scripts/batch_demo_30_questions.py [--pio-exe PATH] [--cfr PATH]
                                              [--target N] [--per-street N]
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
def _stratified_pool(nodes, per_street: int, seed: int) -> list:
    """A reordered node list that round-robins across flop/turn/river.

    Within each street, nodes are randomly shuffled (seeded). The interleaving
    means we hit roughly equal coverage of every street as we walk the pool,
    even if the run is cut short by an API budget.
    """
    rng = random.Random(seed)
    buckets: dict[str, list] = {street: [] for street in STREETS}
    other: list = []
    for node in nodes:
        (buckets.get(node.street) or other).append(node)
    for street in STREETS:
        rng.shuffle(buckets[street])

    # Round-robin across streets, taking up to `per_street` from each bucket
    # first, then continuing through the leftovers.
    head: list = []
    for street in STREETS:
        head.extend(buckets[street][:per_street])
    tail: list = []
    for street in STREETS:
        tail.extend(buckets[street][per_street:])
    rng.shuffle(head)               # mix streets within the priority window
    rng.shuffle(tail)
    return head + tail + other


# --- generation loop --------------------------------------------------------
@dataclass
class BatchResult:
    """Everything we collected during the run, for the summary print."""

    rows: list[dict] = field(default_factory=list)
    evaluated: int = 0
    extract_skipped: int = 0
    extract_errors: list[str] = field(default_factory=list)
    filter_failed: int = 0
    layer6_validation_failures: list[str] = field(default_factory=list)
    layer6_api_failures: list[str] = field(default_factory=list)
    usage: UsageTally = field(default_factory=UsageTally)


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
              target: int, per_street: int) -> BatchResult:
    """Walk the stratified pool until `target` questions have been generated."""
    print("Enumerating decision spots ...")
    nodes = list(sampler.enumerate_decision_nodes(max_chance_children=2))
    pool = _stratified_pool(nodes, per_street=per_street, seed=RANDOM_SEED)
    by_street = Counter(n.street for n in pool)
    print(f"  {len(nodes)} found "
          f"(flop={by_street['flop']}, turn={by_street['turn']}, "
          f"river={by_street['river']}); processing in stratified order.\n")

    result = BatchResult()
    for index, node in enumerate(pool, start=1):
        if len(result.rows) >= target:
            break
        try:
            ctx = sampler.build_spot_context(node)
            spot = extract_facts(ctx, scenario=SCENARIO, big_blind=big_blind)
        except (ValueError, KeyError) as exc:
            result.extract_skipped += 1
            result.extract_errors.append(f"{node.street} {node.node_id}: {exc}")
            continue
        result.evaluated += 1

        verdict = evaluate_spot(spot)
        if not verdict.is_worthy:
            result.filter_failed += 1
            continue

        ok = _generate_one(spot, client, scenario, result)
        if ok:
            print(f"  [{len(result.rows):>2d}/{target}] "
                  f"{spot.spot_metadata.street:<5s} "
                  f"diff={verdict.difficulty_score:<4d} "
                  f"freq={verdict.top_action_frequency:.2f} "
                  f"ev_gap={verdict.ev_gap_bb:>5.2f}bb "
                  f"node={spot.spot_metadata.node_id}")
        else:
            print(f"  [skip] {spot.spot_metadata.node_id}: Layer 6 failed",
                  file=sys.stderr)

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
    parser.add_argument("--target", type=int, default=30,
                        help="number of questions to generate (default 30)")
    parser.add_argument("--per-street", type=int, default=15,
                        help="priority pool size per street before round-robin "
                             "fills from the tail (default 15)")
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
    print(f"Target    : {args.target} questions\n")

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
                               target=args.target, per_street=args.per_street)
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
