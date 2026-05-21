#!/usr/bin/env python3
"""End-to-end pipeline demo: solve -> Path Sampler -> Layer 5 -> Layer 4 -> Layer 8.

Pulls 50 decision spots from the BTN-vs-BB test solve and runs each through the
full chain that now exists:

    Path Sampler        -> a SpotContext
    extract_facts (L5)  -> a populated SpotData
    Question Extractor (L4) -> keep the spot only if it is question-worthy
    Format Writer (L8)  -> one CSV row

The question-worthy spots are written to test_output/demo_questions.csv. The
rows still carry [TBD by Layer 6] placeholders for the question stem, options,
and explanation -- the LLM Explanation Generator is not built yet.

Usage:
    python scripts/end_to_end_demo.py [--pio-exe PATH] [--cfr PATH] [--count N]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.fact_extractor import extract_facts                       # noqa: E402
from pipeline.format_writer import write_csv                            # noqa: E402
from pipeline.path_sampler import PathSampler                           # noqa: E402
from pipeline.piosolver import PioSolverClient, find_piosolver           # noqa: E402
from pipeline.question_extractor import evaluate_spot                    # noqa: E402

DEFAULT_CFR = REPO_ROOT / "test_solves" / "btn_vs_bb_srp_2cJs7s.cfr"
OUTPUT_CSV = REPO_ROOT / "test_output" / "demo_questions.csv"
SAMPLE_SIZE = 50
RANDOM_SEED = 7
# The test solve is a 6-max 100bb cash BTN-vs-BB single-raised pot.
SCENARIO = {"preflop_raise_count": 1, "game_format": "cash",
            "active_players_on_flop": 2, "stack_depth_bb": 100}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pio-exe", help="path to the PioSolver Edge executable")
    parser.add_argument("--cfr", type=Path, default=DEFAULT_CFR)
    parser.add_argument("--count", type=int, default=SAMPLE_SIZE,
                        help="number of decision spots to sample")
    args = parser.parse_args(argv)

    exe = find_piosolver(args.pio_exe)
    if exe is None:
        print("ERROR: PioSolver Edge not found (pass --pio-exe or set "
              "$PIOSOLVER_EXE).", file=sys.stderr)
        return 2
    if not args.cfr.is_file():
        print(f"ERROR: solve file not found: {args.cfr}", file=sys.stderr)
        return 2

    print(f"PioSolver: {exe}")
    print(f"Solve:     {args.cfr.name}\n")

    with PioSolverClient(exe) as client:
        client.load_tree(args.cfr)
        big_blind = (client.show_effective_stack() or 100) / 100.0   # 100bb solve
        sampler = PathSampler(client, oop_position="BB", ip_position="BTN")

        print("Enumerating decision spots ...")
        nodes = list(sampler.enumerate_decision_nodes(max_chance_children=2))
        sample = random.Random(RANDOM_SEED).sample(nodes, min(args.count, len(nodes)))
        print(f"  {len(nodes)} found; running {len(sample)} through the pipeline.\n")

        evaluated = 0
        questions: list[tuple] = []          # (SpotData, difficulty) for worthy spots
        for index, node in enumerate(sample, start=1):
            try:
                ctx = sampler.build_spot_context(node)
                spot = extract_facts(ctx, scenario=SCENARIO, big_blind=big_blind)
            except (ValueError, KeyError) as exc:
                print(f"  spot {index}: skipped ({exc})")
                continue
            evaluated += 1
            verdict = evaluate_spot(spot)
            if verdict.is_worthy:
                questions.append((spot, verdict.difficulty_score))
            if index % 10 == 0:
                print(f"  ... {index}/{len(sample)} spots processed")

        written = write_csv(OUTPUT_CSV, questions)

    print()
    print("=" * 60)
    print(f"  spots evaluated : {evaluated}")
    print(f"  passed filters  : {len(questions)}")
    print(f"  rows written    : {written}")
    print(f"  output CSV      : {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print("=" * 60)
    if questions:
        difficulties = sorted(d for _, d in questions)
        print(f"\nDifficulty range of written questions: "
              f"{difficulties[0]} - {difficulties[-1]} "
              f"(500 easiest, 3000 hardest).")
    print("Rows carry [TBD by Layer 6] for the question text -- the LLM "
          "Explanation\nGenerator is the next layer to build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
