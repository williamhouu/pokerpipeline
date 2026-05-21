#!/usr/bin/env python3
"""Demo: run the full Layer 5 fact extractor on real solver spots.

Layer 5 against real PioSolver data. This script:

  1. loads the BTN-vs-BB test solve,
  2. uses the Layer 3 Path Sampler to enumerate its decision spots,
  3. picks 5 at random,
  4. runs extract_facts() on each -- hand class, board texture, equity / range /
     blocker extraction, and the concept tagger,
  5. prints the resulting facts and the concept tags that fire.

With the equity/range/blocker extraction now built, equity_data and range_data
are populated from real solver ranges, so far more of the 42 concept tags fire
than when those sections were empty.

Usage:
    python scripts/demo_layer5_on_real_spots.py [--pio-exe PATH] [--cfr PATH]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.fact_extractor import extract_facts                       # noqa: E402
from pipeline.path_sampler import PathSampler                           # noqa: E402
from pipeline.piosolver import PioSolverClient, find_piosolver          # noqa: E402

DEFAULT_CFR = REPO_ROOT / "test_solves" / "btn_vs_bb_srp_2cJs7s.cfr"
PER_STREET = 3                        # spots sampled from each of flop/turn/river
RANDOM_SEED = 7                       # fixed so the demo is reproducible

# The test solve is a 6-max 100bb cash BTN-vs-BB single-raised pot. These facts
# are not in the .cfr (a postflop solve); a real run gets them from Layer 1/2.
SCENARIO = {"preflop_raise_count": 1, "game_format": "cash",
            "active_players_on_flop": 2}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pio-exe", help="path to the PioSolver Edge executable")
    parser.add_argument("--cfr", type=Path, default=DEFAULT_CFR)
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
        sampler = PathSampler(client, oop_position="BB", ip_position="BTN")

        print("Enumerating decision spots ...")
        nodes = list(sampler.enumerate_decision_nodes(max_chance_children=3))
        print(f"  {len(nodes)} decision spots found.\n")

        # Sample evenly across streets so the demo shows the full tag range.
        rng = random.Random(RANDOM_SEED)
        chosen = []
        for street in ("flop", "turn", "river"):
            pool = [n for n in nodes if n.street == street]
            chosen += rng.sample(pool, min(PER_STREET, len(pool)))
        tag_total = 0
        for index, node in enumerate(chosen, start=1):
            ctx = sampler.build_spot_context(node)
            spot = extract_facts(ctx, scenario=SCENARIO)
            tag_total += len(spot.concept_tags)

            equity = spot.equity_data
            rng = spot.range_data
            print("=" * 72)
            print(f"SPOT {index}:  {node.node_id}   ({node.node_type})")
            print(f"  street/board : {node.street}  {' '.join(node.board)}")
            print(f"  pot / stack  : {node.pot:.0f} / {node.effective_stack:.0f}"
                  f"  (SPR {spot.spot_metadata.spr:.1f})")
            print(f"  hero hand    : {spot.hand_class.label}"
                  f"   |  board: {spot.board_texture.composite}")
            print(f"  equity       : {equity.hero_raw_equity_vs_continuing:.1%} "
                  f"vs continuing range   pot odds {equity.pot_odds_required:.1%}"
                  f"   MDF {equity.mdf:.1%}")
            print(f"  ranges       : shape={rng.villain_range_shape or '-'}  "
                  f"hero/villain equity {rng.hero_total_equity:.0%}/"
                  f"{rng.villain_total_equity:.0%}  "
                  f"blockers v/b {rng.hero_blocks_value_pct:.0%}/"
                  f"{rng.hero_blocks_bluffs_pct:.0%}")
            print(f"  concept tags ({len(spot.concept_tags)}): "
                  f"{', '.join(spot.concept_tags) or '(none)'}")

        print("=" * 72)
        print(f"\n{tag_total} concept tags fired across {len(chosen)} spots "
              f"(avg {tag_total / len(chosen):.1f}/spot).")
        print("equity_data and range_data are now populated from the solver's "
              "ranges,\nso Sections A, D, E, F and much of B can fire -- not just "
              "the structural tags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
