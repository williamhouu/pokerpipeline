#!/usr/bin/env python3
"""V6: stratified multi-flop batch -- per-cfr cap on EVERY street.

V5 (`scripts/batch_demo_v5_multi_flop.py`) applied a per-cfr cap to flop
questions only; turn and river had `this_cfr_room = global_room`, i.e. no
per-cfr cap at all. As a result, the first `.cfr` in sorted order (which on
this directory is `4d4sKh.cfr` because digits sort before letters in ASCII)
got to fill every turn and every river slot from its own runouts. The V5
output -- 22 rows, every one stamped `flop_4d4sKh` or `turn_4d4sKh*` in
`solver_reference` -- was almost entirely the first solve's tree.

V6 fixes that defect: a single `--per-cfr-cap-per-street` knob applies to
flop, turn, AND river. With target_per_street=15 and per-cfr cap=2, at least
8 distinct .cfrs are needed to fill targets, and in practice (Layer-4 / Layer-6
rejections) we'd expect 10+. The script also prints the .cfrs discovered up
front and per-`.cfr` instrumentation (enumerated / extract-ok / Layer-4 pass /
sampled) so iteration health is visible in the run log, not inferred from the
final CSV.

Two other small changes from V5:

  * `--dry-run`: walk every .cfr and print per-file enumeration + Layer-4
    pass counts WITHOUT calling the LLM. Cheap way to verify iteration
    health before committing API budget on the real run.
  * CSV is written with `encoding="utf-8-sig"` (UTF-8 plus BOM). Excel on
    Windows opens UTF-8-without-BOM CSVs as cp1252 and re-decodes the suit
    emoji bytes (♥ ♦ ♣ ♠ + U+FE0F variation selector) as Latin-1 mojibake.
    The BOM makes Excel auto-detect UTF-8.

Everything else (scenario template lookup, shared Anthropic client, prompt
caching, usage tally, retry tracking, summary print) matches V5.

This hits the real Anthropic API. It is NOT a test. CI never runs it. Skips
cleanly when ANTHROPIC_API_KEY is unset.

Usage:
    set ANTHROPIC_API_KEY=sk-...
    python scripts/batch_demo_v6_stratified.py
    python scripts/batch_demo_v6_stratified.py --dry-run        # no API calls
    python scripts/batch_demo_v6_stratified.py \\
        --scenario Cash6max_100bb_BTN_open_BB_call \\
        --target-per-street 15 --per-cfr-cap-per-street 2
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
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
from pipeline.scenario_config import ScenarioConfig, SCENARIOS           # noqa: E402

DEFAULT_SCENARIO_DIR = "Cash6max_100bb_BTN_open_BB_call"
DEFAULT_SOLVE_ROOT = REPO_ROOT / "solves"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "test_output" / "batch_questions_v6.csv"
SKIP_CFR_STEMS = frozenset({"2cJs7s"})        # MINIMAL_DEBUG verification solve
RANDOM_SEED = 7

# Until the SolverSpec -> ScenarioConfig auto-derivation lands (CLAUDE.md
# "future refactor"), every multi-flop run needs an explicit pointer from
# `solves/<dir>/` to a registered ScenarioConfig that supplies the shared
# scenario metadata (stakes, table size, oop/ip seats, preflop line).
SCENARIO_DIR_TO_TEMPLATE_KEY = {
    "Cash6max_100bb_BTN_open_BB_call": "btn_vs_bb_srp_2cJs7s",
}

LAYER5_SCENARIO = {"preflop_raise_count": 1, "game_format": "cash",
                   "active_players_on_flop": 2, "stack_depth_bb": 100}
STREETS = ("flop", "turn", "river")

PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
PRICE_CACHE_WRITE = 3.75
PRICE_CACHE_READ = 0.30

MAX_API_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0


# --- usage / client wrapping (identical contract to V5) ---------------------
@dataclass
class UsageTally:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    api_calls: int = 0

    def add(self, usage) -> None:
        self.api_calls += 1
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.cache_write_tokens += int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.cache_read_tokens += int(
            getattr(usage, "cache_read_input_tokens", 0) or 0)

    def cost_usd(self) -> float:
        return ((self.input_tokens * PRICE_INPUT
                 + self.output_tokens * PRICE_OUTPUT
                 + self.cache_write_tokens * PRICE_CACHE_WRITE
                 + self.cache_read_tokens * PRICE_CACHE_READ) / 1_000_000)


class _UsageRecordingClient:
    def __init__(self, real_client, tally: UsageTally):
        self._client = real_client
        self._tally = tally
        from types import SimpleNamespace
        self.messages = SimpleNamespace(create=self._create_with_backoff)

    def _create_with_backoff(self, **kwargs):
        import anthropic
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
        assert last_exc is not None
        raise last_exc


# --- scenario lookup for the multi-cfr case ---------------------------------
def _resolve_template(scenario_dir_name: str) -> ScenarioConfig:
    template_key = SCENARIO_DIR_TO_TEMPLATE_KEY.get(scenario_dir_name)
    if template_key is None:
        raise KeyError(
            f"no template registered for scenario directory "
            f"{scenario_dir_name!r}. Add an entry to "
            f"SCENARIO_DIR_TO_TEMPLATE_KEY in batch_demo_v6_stratified.py.")
    try:
        return SCENARIOS[template_key]
    except KeyError as exc:
        raise KeyError(
            f"template {template_key!r} not registered in "
            f"pipeline.scenario_config.SCENARIOS. Known scenarios: "
            f"{sorted(SCENARIOS)}.") from exc


def _scenario_for_cfr(cfr_path: Path, template: ScenarioConfig) -> ScenarioConfig:
    return dataclasses.replace(template, cfr_key=cfr_path.stem)


# --- per-cfr accounting -----------------------------------------------------
@dataclass
class StreetTally:
    target: int = 0
    candidates: int = 0
    extract_skipped: int = 0
    layer4_passed: int = 0
    generated: int = 0


@dataclass
class CfrTally:
    """Per-.cfr instrumentation. Printed in the summary and inline as we walk
    so iteration health is visible without reading the final CSV."""
    stem: str
    enumerated_total: int = 0
    enumerated_per_street: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in STREETS})
    layer4_passed_per_street: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in STREETS})
    sampled_per_street: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in STREETS})

    @property
    def sampled_total(self) -> int:
        return sum(self.sampled_per_street.values())


@dataclass
class BatchResult:
    rows: list[dict] = field(default_factory=list)
    source_flops: list[str] = field(default_factory=list)
    extract_errors: list[str] = field(default_factory=list)
    layer6_validation_failures: list[str] = field(default_factory=list)
    layer6_api_failures: list[str] = field(default_factory=list)
    spot_api_calls: list[int] = field(default_factory=list)
    usage: UsageTally = field(default_factory=UsageTally)
    per_street: dict[str, StreetTally] = field(
        default_factory=lambda: {s: StreetTally() for s in STREETS})
    cfrs_walked: int = 0
    cfrs_contributing: set[str] = field(default_factory=set)
    per_cfr_tallies: list[CfrTally] = field(default_factory=list)


def _bucket_by_street(nodes, seed: int) -> dict[str, list]:
    rng = random.Random(seed)
    buckets: dict[str, list] = {street: [] for street in STREETS}
    for node in nodes:
        if node.street in buckets:
            buckets[node.street].append(node)
    for street in STREETS:
        rng.shuffle(buckets[street])
    return buckets


def _targets_met(result: BatchResult) -> bool:
    return all(result.per_street[s].generated >= result.per_street[s].target
               for s in STREETS)


def _process_one_cfr(cfr_path: Path, pio: PioSolverClient,
                     template: ScenarioConfig, *,
                     client, gold_examples,
                     per_cfr_cap_per_street: int,
                     dry_run: bool,
                     result: BatchResult) -> CfrTally:
    """Walk one .cfr's tree. The per-cfr cap is applied to EVERY street --
    this is the V6 fix. Returns the per-.cfr tally for the summary.

    `dry_run` short-circuits the LLM call (Layer 6). Layer 4 verdicts are still
    computed, so the printed counts reflect what V6 would sample with a real
    API key.
    """
    cfr_stem = cfr_path.stem
    scenario = _scenario_for_cfr(cfr_path, template)
    print(f"\n--- {cfr_stem} ---")
    pio.load_tree(cfr_path)
    big_blind = (pio.show_effective_stack() or 100) / 100.0
    sampler = PathSampler(pio,
                          oop_position=scenario.oop_position,
                          ip_position=scenario.ip_position)

    nodes = list(sampler.enumerate_decision_nodes(max_chance_children=2))
    buckets = _bucket_by_street(nodes, seed=RANDOM_SEED)
    pool = {s: len(buckets[s]) for s in STREETS}

    cfr_tally = CfrTally(stem=cfr_stem, enumerated_total=len(nodes),
                         enumerated_per_street=dict(pool))
    result.per_cfr_tallies.append(cfr_tally)

    print(f"  enumerated {len(nodes)} decision nodes "
          f"(flop={pool['flop']}, turn={pool['turn']}, river={pool['river']})")

    for street in STREETS:
        tally = result.per_street[street]
        tally.candidates += pool[street]
        global_room = max(0, tally.target - tally.generated)
        # V6 FIX: per-cfr cap applies to all streets, not just flop. This
        # prevents the "first .cfr fills every turn+river slot" failure mode
        # V5 exhibited.
        this_cfr_room = min(global_room, per_cfr_cap_per_street)
        if this_cfr_room == 0:
            continue

        for node in buckets[street]:
            if this_cfr_room == 0:
                break
            try:
                ctx = sampler.build_spot_context(node)
                spot = extract_facts(ctx, scenario=LAYER5_SCENARIO,
                                     big_blind=big_blind)
            except (ValueError, KeyError) as exc:
                tally.extract_skipped += 1
                result.extract_errors.append(
                    f"{cfr_stem}/{street}/{node.node_id}: {exc}")
                continue

            verdict = evaluate_spot(spot)
            if not verdict.is_worthy:
                continue
            tally.layer4_passed += 1
            cfr_tally.layer4_passed_per_street[street] += 1

            if dry_run:
                # Count what we WOULD sample without burning API budget.
                cfr_tally.sampled_per_street[street] += 1
                tally.generated += 1
                this_cfr_room -= 1
                result.source_flops.append(cfr_stem)
                result.cfrs_contributing.add(cfr_stem)
                # Append a placeholder row so the per-street counter logic
                # tracks correctly; not written to disk in dry-run mode.
                result.rows.append({"_dry_run_placeholder": True,
                                    "solver_reference": f"DRY/{cfr_stem}/{street}"})
                print(f"  [{len(result.rows):>2d}] {street:<5s} "
                      f"diff={verdict.difficulty_score:<4d} "
                      f"freq={verdict.top_action_frequency:.2f} "
                      f"ev_gap={verdict.ev_gap_bb:>5.2f}bb "
                      f"(DRY) node={spot.spot_metadata.node_id}")
                continue

            calls_before = result.usage.api_calls
            try:
                explanation = generate_explanation(spot, client=client,
                                                   gold_examples=gold_examples)
            except ExplanationValidationError as exc:
                result.layer6_validation_failures.append(
                    f"{cfr_stem}/{street}/{node.node_id}: {str(exc)[:160]}")
                result.spot_api_calls.append(
                    result.usage.api_calls - calls_before)
                continue
            except Exception as exc:
                result.layer6_api_failures.append(
                    f"{cfr_stem}/{street}/{node.node_id}: "
                    f"{type(exc).__name__}: {exc}"[:200])
                result.spot_api_calls.append(
                    result.usage.api_calls - calls_before)
                continue
            result.spot_api_calls.append(result.usage.api_calls - calls_before)

            row = build_row(spot,
                            difficulty_score=verdict.difficulty_score,
                            number=len(result.rows) + 1,
                            scenario=scenario)
            row.update(explanation_to_row_overrides(explanation))
            result.rows.append(row)
            result.source_flops.append(cfr_stem)
            result.cfrs_contributing.add(cfr_stem)
            tally.generated += 1
            cfr_tally.sampled_per_street[street] += 1
            this_cfr_room -= 1

            tag = "RETRY" if result.spot_api_calls[-1] > 1 else "ok"
            print(f"  [{len(result.rows):>2d}] {street:<5s} "
                  f"diff={verdict.difficulty_score:<4d} "
                  f"freq={verdict.top_action_frequency:.2f} "
                  f"ev_gap={verdict.ev_gap_bb:>5.2f}bb "
                  f"calls={result.spot_api_calls[-1]} ({tag}) "
                  f"node={spot.spot_metadata.node_id}")

    if cfr_tally.sampled_total == 0:
        print(f"  (no questions added from {cfr_stem} -- "
              f"targets full or no Layer-4 survivors)")
    return cfr_tally


def run_multi_flop_batch(scenario_dir: Path, *,
                         pio: PioSolverClient,
                         client, gold_examples,
                         target_per_street: int,
                         per_cfr_cap_per_street: int,
                         dry_run: bool) -> BatchResult:
    template = _resolve_template(scenario_dir.name)
    cfrs = sorted(p for p in scenario_dir.glob("*.cfr")
                  if p.stem not in SKIP_CFR_STEMS)
    if not cfrs:
        raise FileNotFoundError(
            f"no .cfr files in {scenario_dir} (after skipping "
            f"{sorted(SKIP_CFR_STEMS)})")

    print(f"Scenario  : {scenario_dir.name}")
    print(f"Template  : {template.cfr_key} -- {template.format}, "
          f"{template.stakes}, {template.preflop_action}")
    print(f"Walking   : {len(cfrs)} .cfr files (skipped "
          f"{sorted(SKIP_CFR_STEMS)})")
    print(f"  discovered .cfrs (sorted): "
          f"{', '.join(p.stem for p in cfrs)}")
    print(f"Targets   : flop={target_per_street}, turn={target_per_street}, "
          f"river={target_per_street}  "
          f"(per-cfr cap: {per_cfr_cap_per_street}/street)")
    if dry_run:
        print("DRY RUN   : no Layer-6 API calls; counts reflect what V6 "
              "WOULD sample with a real API key.")

    result = BatchResult()
    for street in STREETS:
        result.per_street[street].target = target_per_street

    for cfr_path in cfrs:
        if _targets_met(result):
            remaining = len(cfrs) - result.cfrs_walked
            print(f"\n--- all targets met; skipping remaining "
                  f"{remaining} .cfrs ---")
            break
        result.cfrs_walked += 1
        _process_one_cfr(cfr_path, pio, template,
                         client=client, gold_examples=gold_examples,
                         per_cfr_cap_per_street=per_cfr_cap_per_street,
                         dry_run=dry_run,
                         result=result)

    return result


# --- summary printing -------------------------------------------------------
def _print_summary(result: BatchResult, elapsed: float, scenario_dir: Path,
                   total_cfrs: int, dry_run: bool) -> None:
    print()
    print("=" * 78)
    print(f"V6 stratified batch summary{' (DRY RUN)' if dry_run else ''}")
    print("=" * 78)
    print(f"  scenario directory      : {scenario_dir.name}")
    print(f"  .cfrs in directory      : {total_cfrs} "
          f"({len(SKIP_CFR_STEMS)} skipped: {sorted(SKIP_CFR_STEMS)})")
    print(f"  .cfrs walked            : {result.cfrs_walked}")
    print(f"  .cfrs contributing      : {len(result.cfrs_contributing)}")
    print(f"  questions written       : {len(result.rows)}")
    if not dry_run:
        print(f"  Layer-6 validation fails: "
              f"{len(result.layer6_validation_failures)}")
        print(f"  Layer-6 API failures    : {len(result.layer6_api_failures)}")

    print()
    print("  Per-.cfr instrumentation (enum / layer4-pass / sampled, per street):")
    print(f"    {'cfr_stem':<12s} {'enum':>5s} "
          f"{'fl_e':>4s}/{'fl_p':>4s}/{'fl_s':>4s}  "
          f"{'tu_e':>4s}/{'tu_p':>4s}/{'tu_s':>4s}  "
          f"{'ri_e':>4s}/{'ri_p':>4s}/{'ri_s':>4s}")
    for t in result.per_cfr_tallies:
        print(f"    {t.stem:<12s} {t.enumerated_total:>5d} "
              f"{t.enumerated_per_street['flop']:>4d}/"
              f"{t.layer4_passed_per_street['flop']:>4d}/"
              f"{t.sampled_per_street['flop']:>4d}  "
              f"{t.enumerated_per_street['turn']:>4d}/"
              f"{t.layer4_passed_per_street['turn']:>4d}/"
              f"{t.sampled_per_street['turn']:>4d}  "
              f"{t.enumerated_per_street['river']:>4d}/"
              f"{t.layer4_passed_per_street['river']:>4d}/"
              f"{t.sampled_per_street['river']:>4d}")

    if not dry_run and result.spot_api_calls:
        print()
        print("  Per-spot API-call distribution:")
        call_counts = Counter(result.spot_api_calls)
        for n in sorted(call_counts):
            label = ("first-pass success" if n == 1
                     else "one corrective retry" if n == 2
                     else f"{n} calls (multiple retries)")
            print(f"    {n} call(s)  x{call_counts[n]:>3d}   ({label})")
        retried = sum(c for n, c in call_counts.items() if n > 1)
        accepted = len(result.rows)
        print(f"    -> {retried}/{retried + accepted} spots needed at least one "
              f"corrective retry "
              f"({retried / max(1, retried + accepted):.0%})")

    print()
    print("  Per-street tally (candidates / extract-ok / Layer 4 pass / "
          "generated / target):")
    for street in STREETS:
        t = result.per_street[street]
        flag = "" if t.generated >= t.target else "  <-- short of target"
        extract_ok = t.candidates - t.extract_skipped
        print(f"    {street:<6s} {t.candidates:>4d} / {extract_ok:>4d} / "
              f"{t.layer4_passed:>3d} / {t.generated:>2d} / "
              f"{t.target:>2d}{flag}")

    print()
    print("  Distribution by source flop:")
    source_counts = Counter(result.source_flops)
    for cfr_stem, count in sorted(source_counts.items(),
                                  key=lambda kv: (-kv[1], kv[0])):
        bar = "#" * count
        print(f"    {cfr_stem:<10s} x{count:<2d}  {bar}")

    if not dry_run:
        print()
        print("  Distribution by board_texture:")
        texture_counts = Counter(r.get("board_texture", "") for r in result.rows)
        for texture, count in sorted(texture_counts.items(),
                                     key=lambda kv: (-kv[1], kv[0])):
            bar = "#" * count
            print(f"    {texture:<48s} x{count:<2d}  {bar}")

        u = result.usage
        print()
        print(f"  Anthropic calls         : {u.api_calls}")
        print(f"  input tokens            : {u.input_tokens:>9,}")
        print(f"  output tokens           : {u.output_tokens:>9,}")
        print(f"  cache-write tokens      : {u.cache_write_tokens:>9,}")
        print(f"  cache-read tokens       : {u.cache_read_tokens:>9,}")
        print(f"  approx. cost            : ${u.cost_usd():.4f}")
        print(f"  wall-clock              : {elapsed:.1f}s "
              f"({elapsed / max(1, u.api_calls):.2f}s / API call, "
              f"{elapsed / max(1, len(result.rows)):.2f}s / question)")
    else:
        print()
        print(f"  wall-clock              : {elapsed:.1f}s (dry run)")
    print("=" * 78)

    if result.layer6_validation_failures:
        print("\n  Layer 6 validation failures (first 3):")
        for msg in result.layer6_validation_failures[:3]:
            print(f"    - {msg}")
    if result.layer6_api_failures:
        print("\n  Layer 6 API failures (first 3):")
        for msg in result.layer6_api_failures[:3]:
            print(f"    - {msg}")
    if result.extract_errors:
        print(f"\n  extract_facts skipped {len(result.extract_errors)} spots; "
              f"first 3:")
        for msg in result.extract_errors[:3]:
            print(f"    - {msg}")


# --- entry point ------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--solve-root", type=Path, default=DEFAULT_SOLVE_ROOT)
    parser.add_argument("--pio-exe")
    parser.add_argument("--target-per-street", type=int, default=15,
                        help="questions per street; total = 3 * this (default 15)")
    parser.add_argument("--per-cfr-cap-per-street", type=int, default=2,
                        help="max questions per .cfr per street (default 2 -- "
                             "guarantees >= 8 .cfrs to fill 15-per-street targets)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--dry-run", action="store_true",
                        help="walk every .cfr and print enumeration / Layer-4 "
                             "pass counts WITHOUT calling the LLM. Iteration-"
                             "health verification on $0 API budget.")
    args = parser.parse_args(argv)

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- run with --dry-run to verify "
              "iteration without API calls, or set the key and rerun.")
        return 0

    exe = find_piosolver(args.pio_exe)
    if exe is None:
        print("ERROR: PioSolver Edge not found (pass --pio-exe or set "
              "$PIOSOLVER_EXE).", file=sys.stderr)
        return 2

    scenario_dir = args.solve_root / args.scenario
    if not scenario_dir.is_dir():
        print(f"ERROR: scenario directory not found: {scenario_dir}",
              file=sys.stderr)
        return 2

    print(f"PioSolver : {exe}")

    if args.dry_run:
        client, gold_examples, usage = None, [], UsageTally()
    else:
        gold_examples = list(load_gold_examples())[:GOLD_EXAMPLE_COUNT]
        print(f"Gold examples in-context: {len(gold_examples)}")
        import anthropic
        base_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        usage = UsageTally()
        client = _UsageRecordingClient(base_client, usage)

    total_cfrs = len(list(scenario_dir.glob("*.cfr")))

    start = time.time()
    with PioSolverClient(exe) as pio:
        result = run_multi_flop_batch(
            scenario_dir, pio=pio, client=client,
            gold_examples=gold_examples,
            target_per_street=args.target_per_street,
            per_cfr_cap_per_street=args.per_cfr_cap_per_street,
            dry_run=args.dry_run,
        )
    elapsed = time.time() - start
    result.usage = usage

    if not args.dry_run:
        # utf-8-sig writes a BOM so Excel on Windows auto-detects UTF-8
        # instead of falling back to cp1252 and mojibake-ing the suit emojis.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in result.rows:
                writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    _print_summary(result, elapsed, scenario_dir, total_cfrs, args.dry_run)

    if not args.dry_run:
        try:
            out_display = args.out.relative_to(REPO_ROOT)
        except ValueError:
            out_display = args.out
        print(f"\n  output CSV: {out_display}")
    return 0 if result.rows else 1


if __name__ == "__main__":
    sys.exit(main())
