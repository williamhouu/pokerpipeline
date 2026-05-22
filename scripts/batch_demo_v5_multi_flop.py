#!/usr/bin/env python3
"""V5: batch-generate 45 questions across MANY solved flops, not just one.

V4 (`scripts/batch_demo_30_questions.py`) generated 30 questions off a single
`.cfr` -- enough to surface the structural failure modes that drove Layer 7's
audit validators, but every question shared the same flop board. V5 is the
first multi-flop batch: it walks every `.cfr` in a scenario directory and
distributes questions across distinct boards, so the read-by-hand review can
finally measure how the pipeline behaves with real board diversity.

What it does that V4 doesn't:

  * accepts `--scenario` (a `solves/` subdirectory) and walks every `.cfr`
    in it, with a `--solve-root` override to point at a different cache;
  * SKIPS `2cJs7s.cfr` (the MINIMAL_DEBUG verification solve from Phase 0)
    so it doesn't double-count against the STANDARD_25_FLOPS cache;
  * stratified target 15 flop + 15 turn + 15 river = 45 questions total,
    with a per-`.cfr` cap of 3 flop questions for flop-board diversity
    (turn/river are inherently diverse since each `.cfr` produces spots on
    many turn cards x river cards, so no per-`.cfr` cap there);
  * runs each candidate through Layers 4 -> 5 -> 6 (with Layer 7's
    `run_audit_validators` retry loop) -> 8 with a SHARED Anthropic client,
    so the prompt cache hits on every call after the first;
  * tracks per-spot API calls so the summary distinguishes first-pass
    successes (1 call) from validator-triggered retries (2 calls) from
    exhausted-retry failures (`ExplanationValidationError`);
  * reports distribution by source flop, distribution by board_texture
    category, per-street tallies, validator/retry counts, API cost, and
    wall-clock.

This hits the real Anthropic API. It is NOT a test. CI never runs it. Skips
cleanly when ANTHROPIC_API_KEY is unset.

Scenario lookup: `pipeline.scenario_config.get_scenario` is currently keyed
by `.cfr` stem and only has `btn_vs_bb_srp_2cJs7s` registered. CLAUDE.md
flags the auto-derive-from-SolverSpec refactor as future work; until then V5
clones the registered template per-`.cfr` via `dataclasses.replace`. The
scenario metadata (format, stakes, table size, oop/ip positions) is identical
across every flop in `Cash6max_100bb_BTN_open_BB_call/` -- only the cfr_key
changes -- so the clone is semantically correct.

Usage:
    set ANTHROPIC_API_KEY=sk-...
    python scripts/batch_demo_v5_multi_flop.py
    python scripts/batch_demo_v5_multi_flop.py \
        --scenario Cash6max_100bb_BTN_open_BB_call \
        --target-per-street 15 --flop-cap-per-cfr 3
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
DEFAULT_OUTPUT_CSV = REPO_ROOT / "test_output" / "batch_questions_v5.csv"
SKIP_CFR_STEMS = frozenset({"2cJs7s"})        # MINIMAL_DEBUG verification solve
RANDOM_SEED = 7

# Until the SolverSpec -> ScenarioConfig auto-derivation lands (CLAUDE.md
# "future refactor"), every multi-flop run needs an explicit pointer from
# `solves/<dir>/` to a registered ScenarioConfig that supplies the shared
# scenario metadata (stakes, table size, oop/ip seats, preflop line). When
# the refactor lands this dict goes away.
SCENARIO_DIR_TO_TEMPLATE_KEY = {
    "Cash6max_100bb_BTN_open_BB_call": "btn_vs_bb_srp_2cJs7s",
}

# Layer-5 scenario metadata that the .cfr doesn't carry. Identical across
# every flop in a SRP cash scenario -- the values come from the brief and
# match what V4 passes for the single-flop run.
LAYER5_SCENARIO = {"preflop_raise_count": 1, "game_format": "cash",
                   "active_players_on_flop": 2, "stack_depth_bb": 100}
STREETS = ("flop", "turn", "river")

# Sonnet 4.6 pricing (USD per 1M tokens). Same values V4 uses.
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
PRICE_CACHE_WRITE = 3.75
PRICE_CACHE_READ = 0.30

# Backoff for transient API failures.
MAX_API_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0


# --- usage / client wrapping (identical contract to V4) ---------------------
@dataclass
class UsageTally:
    """Running token usage across every API call in the batch."""

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
    """Wraps an Anthropic client to record token usage + retry transient errors.

    Layer 6 accepts any object with a `messages.create` method; we keep batch
    concerns (cost tracking, backoff) out of Layer 6 by injecting this wrapper.
    """

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
    """The registered ScenarioConfig that supplies metadata for every .cfr in
    `solves/<scenario_dir_name>/`. Raises with a clear pointer when missing.
    """
    template_key = SCENARIO_DIR_TO_TEMPLATE_KEY.get(scenario_dir_name)
    if template_key is None:
        raise KeyError(
            f"no template registered for scenario directory "
            f"{scenario_dir_name!r}. Add an entry to "
            f"SCENARIO_DIR_TO_TEMPLATE_KEY in batch_demo_v5_multi_flop.py, "
            f"pointing at the ScenarioConfig in pipeline/scenario_config.py "
            f"that shares this directory's scenario metadata.")
    try:
        return SCENARIOS[template_key]
    except KeyError as exc:
        raise KeyError(
            f"template {template_key!r} not registered in "
            f"pipeline.scenario_config.SCENARIOS. Known scenarios: "
            f"{sorted(SCENARIOS)}.") from exc


def _scenario_for_cfr(cfr_path: Path, template: ScenarioConfig) -> ScenarioConfig:
    """Clone the registered template with this `.cfr`'s stem as cfr_key.

    Every other field (format, stakes, oop/ip positions, preflop_actions, ...)
    is identical -- it's the same scenario, just played out on a different
    flop. `dataclasses.replace` re-runs `__post_init__`, so the derived
    `context` string is recomputed (a no-op here since the deriving inputs
    don't change, but correct in general).
    """
    return dataclasses.replace(template, cfr_key=cfr_path.stem)


# --- the multi-flop batch ----------------------------------------------------
@dataclass
class StreetTally:
    target: int = 0
    candidates: int = 0                  # decision nodes enumerated this street
    extract_skipped: int = 0             # extract_facts raised
    layer4_passed: int = 0               # is_question_worthy True
    generated: int = 0                   # row appended


@dataclass
class BatchResult:
    """Everything we collected during the run, for the summary print."""

    rows: list[dict] = field(default_factory=list)
    source_flops: list[str] = field(default_factory=list)
    extract_errors: list[str] = field(default_factory=list)
    layer6_validation_failures: list[str] = field(default_factory=list)
    layer6_api_failures: list[str] = field(default_factory=list)
    # Per-spot retry tracking: each entry is the number of API calls that
    # generate_explanation made for that spot. 1 = first-pass success;
    # >1 = at least one corrective retry triggered by a validator failure.
    spot_api_calls: list[int] = field(default_factory=list)
    usage: UsageTally = field(default_factory=UsageTally)
    per_street: dict[str, StreetTally] = field(
        default_factory=lambda: {s: StreetTally() for s in STREETS})
    # Per-cfr counts -- which solves contributed to the final batch and which
    # were skipped because targets were already met.
    cfrs_walked: int = 0
    cfrs_contributing: set[str] = field(default_factory=set)


def _bucket_by_street(nodes, seed: int) -> dict[str, list]:
    """Shuffle nodes (seeded) and bucket them by street."""
    rng = random.Random(seed)
    buckets: dict[str, list] = {street: [] for street in STREETS}
    for node in nodes:
        if node.street in buckets:
            buckets[node.street].append(node)
    for street in STREETS:
        rng.shuffle(buckets[street])
    return buckets


def _targets_met(result: BatchResult) -> bool:
    """All three street targets satisfied -- no point walking more .cfrs."""
    return all(result.per_street[s].generated >= result.per_street[s].target
               for s in STREETS)


def _process_one_cfr(cfr_path: Path, pio: PioSolverClient,
                     template: ScenarioConfig, *,
                     client, gold_examples,
                     flop_cap_per_cfr: int,
                     result: BatchResult) -> None:
    """Walk a single .cfr's tree, contribute up to the per-street targets.

    Per-street stop conditions:
        flop  -- this cfr contributes at most `flop_cap_per_cfr` AND no more
                 than the global flop target.
        turn  -- no per-cfr cap (turn cards already diversify within a single
                 .cfr); stops when global target met.
        river -- ditto.

    Mutates `result` in place.
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
    print(f"  enumerated {len(nodes)} decision nodes "
          f"(flop={pool['flop']}, turn={pool['turn']}, river={pool['river']})")

    contributed_this_cfr = 0
    for street in STREETS:
        tally = result.per_street[street]
        tally.candidates += pool[street]
        # Per-street + per-cfr cap. For flop, contribute at most flop_cap_per_cfr
        # additional rows from THIS .cfr; for turn/river there's no per-cfr cap.
        global_room = max(0, tally.target - tally.generated)
        if street == "flop":
            this_cfr_room = min(global_room, flop_cap_per_cfr)
        else:
            this_cfr_room = global_room
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

            # Snapshot API-call count so we can tell first-pass from retry.
            calls_before = result.usage.api_calls
            try:
                explanation = generate_explanation(spot, client=client,
                                                   gold_examples=gold_examples)
            except ExplanationValidationError as exc:
                result.layer6_validation_failures.append(
                    f"{cfr_stem}/{street}/{node.node_id}: {str(exc)[:160]}")
                # Record the calls this failed spot consumed (typically max
                # retries) so the retry-counts table includes them.
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
            this_cfr_room -= 1
            contributed_this_cfr += 1

            tag = "RETRY" if result.spot_api_calls[-1] > 1 else "ok"
            print(f"  [{len(result.rows):>2d}/45] {street:<5s} "
                  f"diff={verdict.difficulty_score:<4d} "
                  f"freq={verdict.top_action_frequency:.2f} "
                  f"ev_gap={verdict.ev_gap_bb:>5.2f}bb "
                  f"calls={result.spot_api_calls[-1]} ({tag}) "
                  f"node={spot.spot_metadata.node_id}")

    if contributed_this_cfr == 0:
        print(f"  (no questions added from {cfr_stem} -- "
              f"targets full or no Layer-4 survivors)")


def run_multi_flop_batch(scenario_dir: Path, *,
                         pio: PioSolverClient,
                         client, gold_examples,
                         target_per_street: int,
                         flop_cap_per_cfr: int) -> BatchResult:
    """Walk every .cfr in `scenario_dir` (except SKIP_CFR_STEMS) in sorted
    order, contributing to per-street targets until all are met or the
    directory is exhausted.

    Sorted order keeps the run reproducible; within each .cfr, the bucket
    shuffle uses the fixed RANDOM_SEED so a re-run hits the same nodes.
    """
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
    print(f"Targets   : flop={target_per_street} (<={flop_cap_per_cfr}/cfr), "
          f"turn={target_per_street}, river={target_per_street}")

    result = BatchResult()
    for street in STREETS:
        result.per_street[street].target = target_per_street

    for cfr_path in cfrs:
        if _targets_met(result):
            print(f"\n--- all targets met; skipping remaining "
                  f"{len(cfrs) - result.cfrs_walked} .cfrs ---")
            break
        result.cfrs_walked += 1
        _process_one_cfr(cfr_path, pio, template,
                         client=client, gold_examples=gold_examples,
                         flop_cap_per_cfr=flop_cap_per_cfr,
                         result=result)

    return result


# --- summary printing -------------------------------------------------------
def _print_summary(result: BatchResult, elapsed: float, scenario_dir: Path,
                   total_cfrs: int) -> None:
    print()
    print("=" * 78)
    print("V5 multi-flop batch summary")
    print("=" * 78)
    print(f"  scenario directory      : {scenario_dir.name}")
    print(f"  .cfrs in directory      : {total_cfrs} "
          f"({len(SKIP_CFR_STEMS)} skipped: {sorted(SKIP_CFR_STEMS)})")
    print(f"  .cfrs walked            : {result.cfrs_walked}")
    print(f"  .cfrs contributing      : {len(result.cfrs_contributing)}")
    print(f"  questions written       : {len(result.rows)}")
    print(f"  Layer-6 validation fails: "
          f"{len(result.layer6_validation_failures)} "
          f"(retries exhausted -- routed to human review)")
    print(f"  Layer-6 API failures    : {len(result.layer6_api_failures)}")

    # Retry distribution -- the V5 question the brief most wants answered:
    # how often does the validator catch something and a corrective retry
    # produces an acceptable explanation, vs first-pass success?
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
    print("  Per-street tally (candidates / extract-ok / Layer 4 pass / generated / target):")
    for street in STREETS:
        t = result.per_street[street]
        flag = "" if t.generated >= t.target else "  <-- short of target"
        extract_ok = t.candidates - t.extract_skipped
        print(f"    {street:<6s} {t.candidates:>4d} / {extract_ok:>4d} / "
              f"{t.layer4_passed:>3d} / {t.generated:>2d} / "
              f"{t.target:>2d}{flag}")

    # Per-source-flop distribution -- the diversity metric V5 exists to surface.
    print()
    print("  Distribution by source flop:")
    source_counts = Counter(result.source_flops)
    for cfr_stem, count in sorted(source_counts.items(),
                                  key=lambda kv: (-kv[1], kv[0])):
        bar = "#" * count
        print(f"    {cfr_stem:<10s} x{count:<2d}  {bar}")

    # Board-texture distribution (the categorical label Layer 5 attaches).
    print()
    print("  Distribution by board_texture:")
    texture_counts = Counter(r["board_texture"] for r in result.rows)
    for texture, count in sorted(texture_counts.items(),
                                 key=lambda kv: (-kv[1], kv[0])):
        bar = "#" * count
        print(f"    {texture:<48s} x{count:<2d}  {bar}")

    # API cost + wall.
    u = result.usage
    print()
    print(f"  Anthropic calls         : {u.api_calls}")
    print(f"  input tokens            : {u.input_tokens:>9,}")
    print(f"  output tokens           : {u.output_tokens:>9,}")
    print(f"  cache-write tokens      : {u.cache_write_tokens:>9,}")
    print(f"  cache-read tokens       : {u.cache_read_tokens:>9,}  "
          f"(hits dominate after call 1)")
    print(f"  approx. cost            : ${u.cost_usd():.4f}")
    print(f"  wall-clock              : {elapsed:.1f}s "
          f"({elapsed / max(1, u.api_calls):.2f}s / API call, "
          f"{elapsed / max(1, len(result.rows)):.2f}s / question)")
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
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO_DIR,
                        help=f"scenario directory under --solve-root "
                             f"(default {DEFAULT_SCENARIO_DIR})")
    parser.add_argument("--solve-root", type=Path, default=DEFAULT_SOLVE_ROOT,
                        help=f"root that contains scenario subdirectories "
                             f"(default {DEFAULT_SOLVE_ROOT})")
    parser.add_argument("--pio-exe", help="path to the PioSolver Edge executable")
    parser.add_argument("--target-per-street", type=int, default=15,
                        help="questions to generate per street (flop/turn/river); "
                             "total = 3 * this (default 15 -> 45 total)")
    parser.add_argument("--flop-cap-per-cfr", type=int, default=3,
                        help="max flop questions per single .cfr "
                             "(default 3, for flop-board diversity)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help=f"output CSV path (default {DEFAULT_OUTPUT_CSV})")
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- skipping the V5 batch demo.")
        print("Set the key and rerun; this script never runs in CI.")
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

    # Pre-load gold examples once so the cached system+exemplar payload is
    # identical on every Layer 6 call. (V4 also does this.)
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
            flop_cap_per_cfr=args.flop_cap_per_cfr,
        )
    elapsed = time.time() - start
    result.usage = usage

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row)

    _print_summary(result, elapsed, scenario_dir, total_cfrs)
    try:
        out_display = args.out.relative_to(REPO_ROOT)
    except ValueError:
        out_display = args.out
    print(f"\n  output CSV: {out_display}")
    print("\nRead a sample of the V5 rows by hand. The brief's Phase-1 quality "
          "gate is now multi-flop -- failures that only appear with board "
          "diversity (not the V4 single-flop run) drive the next round of "
          "Layer-7 checker priorities.")
    return 0 if result.rows else 1


if __name__ == "__main__":
    sys.exit(main())
