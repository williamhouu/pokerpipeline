#!/usr/bin/env python3
"""Layer 2 batch-solve CLI: produce one .cfr per flop for a scenario.

This is the entry point for scaling beyond the single hand-solved test .cfr.
Reads the registered solver spec (pipeline.scenario_spec) + the chosen flop
set (pipeline.flop_sets) and drives PioSolver Edge via UPI to produce one
solve per (spec, flop) at:

    solves/<spec.name>/<flop_stem>.cfr

Resume-safe: skips flops whose `.cfr` already exists. Failure-isolated: one
flop crashing writes a `.failed`/`.timeout` marker and the batch proceeds.

Two-phase recommended workflow:

    1. Dry run first to confirm the spec is well-formed without compute:
         python scripts/batch_solve.py \\
             --scenario Cash6max_100bb_BTN_open_BB_call \\
             --flop-set MINIMAL_DEBUG --dry-run

    2. Real run on MINIMAL_DEBUG (1 flop ~ 30-60 min) to verify Layer 2
       produces a structurally equivalent solve to the hand-solved file:
         python scripts/batch_solve.py \\
             --scenario Cash6max_100bb_BTN_open_BB_call \\
             --flop-set MINIMAL_DEBUG

    3. Once that lands clean, STANDARD_25_FLOPS overnight (~12-25 h total):
         python scripts/batch_solve.py \\
             --scenario Cash6max_100bb_BTN_open_BB_call \\
             --flop-set STANDARD_25_FLOPS

CI never runs this -- it requires a local PioSolver Edge installation and
significant compute. Use a dedicated long-lived terminal.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.batch_solver import (                                       # noqa: E402
    DEFAULT_SOLVE_ROOT, plan_batch, run_batch,
)
from pipeline.flop_sets import FLOP_SETS, select_flops                    # noqa: E402
from pipeline.piosolver import find_piosolver                             # noqa: E402
from pipeline.scenario_spec import SOLVER_SPECS, get_solver_spec          # noqa: E402


def _print_dry_run(plan: list[dict], spec_name: str, flop_set_name: str) -> int:
    """Show the planned execution without compute. Returns process exit code."""
    print(f"DRY RUN -- no UPI commands will be issued.\n")
    print(f"  scenario  : {spec_name}")
    print(f"  flop set  : {flop_set_name} ({len(plan)} flops)")
    print(f"  cache root: {(REPO_ROOT / DEFAULT_SOLVE_ROOT).resolve()}")
    print()
    existing = sum(1 for entry in plan if entry["already_exists"])
    print(f"  to solve  : {len(plan) - existing}")
    print(f"  to skip   : {existing}  (already on disk)")
    print()
    for i, entry in enumerate(plan, 1):
        flag = "[exists -> skip]" if entry["already_exists"] else "[to solve]"
        print(f"    [{i:>2d}/{len(plan)}] {flag:<18s} "
              f"flop {entry['flop_board']:<10s} "
              f"-> {entry['output_path']}")
    print()
    print("Re-run without --dry-run to actually solve.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True,
                        choices=sorted(SOLVER_SPECS),
                        help="solver spec name from pipeline/scenario_spec.py")
    parser.add_argument("--flop-set", required=True,
                        choices=sorted(FLOP_SETS),
                        help="flop set name from pipeline/flop_sets.py")
    parser.add_argument("--pio-exe", type=Path,
                        help="path to PioSolver Edge executable (auto-detect if omitted)")
    parser.add_argument("--solve-root", type=Path, default=REPO_ROOT / "solves",
                        help="cache root directory (default <repo>/solves)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the execution plan without solving")
    parser.add_argument("--force-resolve", action="store_true",
                        help="delete and re-solve any .cfr files already on "
                             "disk (default: skip and resume). Use this when "
                             "the spec changed (e.g. ranges swapped) and the "
                             "cached solves are stale.")
    args = parser.parse_args(argv)

    spec = get_solver_spec(args.scenario)
    flops = select_flops(args.flop_set)
    plan = plan_batch(spec, flops, solve_root=args.solve_root)

    if args.dry_run:
        return _print_dry_run(plan, args.scenario, args.flop_set)

    pio_exe = args.pio_exe or find_piosolver()
    if pio_exe is None or not Path(pio_exe).is_file():
        print("ERROR: PioSolver Edge not found. Pass --pio-exe or set "
              "$PIOSOLVER_EXE.", file=sys.stderr)
        return 2

    start = time.time()
    print(f"PioSolver : {pio_exe}\n")
    result = run_batch(spec, flops, pio_exe=pio_exe,
                       solve_root=args.solve_root,
                       flop_set_name=args.flop_set,
                       force=args.force_resolve)
    elapsed = time.time() - start

    # Summary.
    print()
    print("=" * 72)
    print(f"  scenario  : {result.spec_name}")
    print(f"  flop set  : {result.flop_set_name}")
    print(f"  total     : {len(result.solves)}")
    print(f"  solved    : {len(result.solved)}")
    print(f"  skipped   : {len(result.skipped)}  (existing .cfr)")
    print(f"  failed    : {len(result.failed)}")
    print(f"  elapsed   : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print("=" * 72)
    if result.failed:
        print("\n  Failed/timed-out flops:")
        for s in result.failed:
            print(f"    {s.flop_stem} ({s.status}): {s.error_message[:120]}")
    if result.solved:
        print("\n  Solved this run:")
        for s in result.solved:
            expl = (f"{s.final_exploitability_chips:.3f} chips"
                    if s.final_exploitability_chips is not None else "?")
            print(f"    {s.flop_stem}: {s.elapsed_seconds:.0f}s, "
                  f"{s.file_size_bytes:,} bytes, exploit={expl}")
    return 0 if not result.failed else 1


if __name__ == "__main__":
    sys.exit(main())
