#!/usr/bin/env python3
"""End-to-end pipeline demo, now WITH Layer 6 (LLM explanations).

The first time the pipeline produces a complete training question -- option
strings, correct answer, and a coaching-voice explanation -- not just a CSV
shell with [TBD by Layer 6] placeholders.

Flow:
    PioSolver tree
      -> Path Sampler            (enumerate decision spots)
      -> Layer 5 (extract_facts) (SpotData)
      -> Layer 4 (passes_filters)?
      -> Layer 6 (generate_explanation)         <-- THIS LAYER (calls Claude)
      -> Layer 8 (build_row + overrides)
      -> test_output/demo_one_question.csv

Skips cleanly when ANTHROPIC_API_KEY is not set (the only script in the repo
that touches the real API; it is deliberately NOT a test).

Usage:
    set ANTHROPIC_API_KEY=sk-...
    python scripts/demo_layer6_real_api.py [--pio-exe PATH] [--cfr PATH]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.explanation_generator import (                             # noqa: E402
    ExplanationValidationError, explanation_to_row_overrides,
    generate_explanation,
)
from pipeline.fact_extractor import extract_facts                        # noqa: E402
from pipeline.format_writer import CSV_COLUMNS, build_row                # noqa: E402
from pipeline.path_sampler import PathSampler                            # noqa: E402
from pipeline.piosolver import PioSolverClient, find_piosolver           # noqa: E402
from pipeline.question_extractor import evaluate_spot                    # noqa: E402

DEFAULT_CFR = REPO_ROOT / "test_solves" / "btn_vs_bb_srp_2cJs7s.cfr"
OUTPUT_CSV = REPO_ROOT / "test_output" / "demo_one_question.csv"
RANDOM_SEED = 7
# The test solve is a 6-max 100bb cash BTN-vs-BB single-raised pot.
SCENARIO = {"preflop_raise_count": 1, "game_format": "cash",
            "active_players_on_flop": 2, "stack_depth_bb": 100}


def _first_question_worthy_spot(sampler, big_blind, max_attempts: int = 200):
    """Walk the tree until we find one spot that passes Layer 4 filters.

    Yields (SpotContext, SpotData, difficulty) for the first hit, or returns
    None if `max_attempts` spots fail.
    """
    nodes = list(sampler.enumerate_decision_nodes(max_chance_children=2))
    random.Random(RANDOM_SEED).shuffle(nodes)
    for index, node in enumerate(nodes[:max_attempts], start=1):
        try:
            ctx = sampler.build_spot_context(node)
            spot = extract_facts(ctx, scenario=SCENARIO, big_blind=big_blind)
        except (ValueError, KeyError) as exc:
            continue
        verdict = evaluate_spot(spot)
        if verdict.is_worthy:
            print(f"  found a question-worthy spot at attempt {index} "
                  f"(difficulty {verdict.difficulty_score}, "
                  f"ev_gap {verdict.ev_gap_bb:.2f} bb).")
            return spot, verdict.difficulty_score
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pio-exe", help="path to the PioSolver Edge executable")
    parser.add_argument("--cfr", type=Path, default=DEFAULT_CFR)
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- skipping the Layer 6 demo.")
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

    print(f"PioSolver: {exe}")
    print(f"Solve:     {args.cfr.name}\n")

    with PioSolverClient(exe) as client:
        client.load_tree(args.cfr)
        big_blind = (client.show_effective_stack() or 100) / 100.0   # 100bb solve
        sampler = PathSampler(client, oop_position="BB", ip_position="BTN")

        print("Hunting for a question-worthy spot ...")
        result = _first_question_worthy_spot(sampler, big_blind)
        if result is None:
            print("No question-worthy spot found in 200 attempts. "
                  "Try a larger sample size or a different solve.",
                  file=sys.stderr)
            return 1
        spot, difficulty = result

    print("\nCalling Layer 6 (Anthropic) ...")
    try:
        explanation = generate_explanation(spot)
    except ExplanationValidationError as exc:
        print(f"ERROR: Layer 6 failed validation -- spot routed to human review:\n"
              f"  {exc}", file=sys.stderr)
        return 1

    # Build the Layer 8 row and patch in Layer 6's six columns.
    row = build_row(spot, difficulty, number=1)
    row.update(explanation_to_row_overrides(explanation))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    print()
    print("=" * 64)
    print(f"  Wrote 1 complete question to "
          f"{OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print("=" * 64)
    print(f"  Hand stage      : {row['Hand Stage']}")
    print(f"  Position        : {row['Relative Position']}")
    print(f"  User cards      : {row['User Cards']}")
    print(f"  Cards on table  : {row['Cards on Table']}")
    print(f"  Options         : "
          f"{[explanation.option_1, explanation.option_2, explanation.option_3, explanation.option_4]}")
    print(f"  Correct answer  : {explanation.correct_answer}")
    print(f"  Concept tags    : {row['concept_tags']}")
    print(f"  Difficulty      : {row['Difficulty Rating']}")
    print(f"\n  --- Answer Explanation ---")
    print(f"  {explanation.answer_explanation}")
    print("\n  This is the pipeline's first complete end-to-end question. "
          "Layer 7 (validators)\n  is the next layer to build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
