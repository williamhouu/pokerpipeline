#!/usr/bin/env python3
"""Demo: run the Layer 5 concept tagger on real solver spots.

The first time Layer 5 meets real PioSolver data. This script:

  1. loads the BTN-vs-BB test solve,
  2. uses the Layer 3 Path Sampler to enumerate its decision spots,
  3. picks 5 at random,
  4. maps each Path Sampler SpotContext into a Layer 5 SpotData,
  5. runs compute_tags() and prints the concept tags that fire.

It is a sanity check, not a finished pipeline. A SpotData built here is only
partially populated -- the equity / range / blocker fact-extraction that fills
equity_data and range_data is still to be built, so tags in Sections A, E, F
and most of B cannot fire yet. What this demo proves is that the path from
solver output to compute_tags() works end to end on real data.

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

from pipeline.fact_extractor.concept_tags.registry import compute_tags  # noqa: E402
from pipeline.fact_extractor.spot_data import (                         # noqa: E402
    BoardTexture, DecisionData, HandClass, SpotData, SpotMetadata,
)
from pipeline.path_sampler import PathSampler                           # noqa: E402
from pipeline.piosolver import PioSolverClient, find_piosolver          # noqa: E402

DEFAULT_CFR = REPO_ROOT / "test_solves" / "btn_vs_bb_srp_2cJs7s.cfr"
SAMPLE_SIZE = 5
RANDOM_SEED = 7                       # fixed so the demo is reproducible


def _canonical(label: str) -> str:
    """Action label -> canonical verb: 'bet 36' -> 'bet', 'check' -> 'check'."""
    return label.split()[0]


def _street_segments(action_sequence):
    """Split an action sequence into per-street segments at the card deals."""
    segments = [[]]
    for actor, label in action_sequence:
        if actor == "deal":
            segments.append([])
        else:
            segments[-1].append((actor, label))
    return segments


def spot_data_from_context(ctx) -> SpotData:
    """Map a Path Sampler SpotContext into a (partial) Layer 5 SpotData.

    Populates what the solver directly provides -- street, board texture, hand
    class, strategy, action EVs, the action line. equity_data and range_data
    are left at defaults: that fact-extraction is the remaining Layer 5 work.
    """
    node = ctx.node
    hero_side = "OOP" if node.hero_is_oop else "IP"

    segments = _street_segments(node.action_sequence)
    convert = lambda entries: [("hero" if actor == hero_side else "villain",
                                _canonical(label)) for actor, label in entries]
    street_actions = convert(segments[-1])
    prior_actions = convert(segments[-2]) if len(segments) >= 2 else []

    # Hero's most likely specific combo, for the hand class.
    hero_combo = max(ctx.hero_range, key=ctx.hero_range.get)

    metadata = SpotMetadata(
        street=node.street,
        spr=node.effective_stack / node.pot if node.pot else 0.0,
        position_dynamic=f"{node.hero_position}_vs_{node.villain_position}",
        hero_position=node.hero_position,
        villain_position=node.villain_position,
        hero_in_position=not node.hero_is_oop,
        preflop_raise_count=1,        # the test solve is a single-raised pot
        active_players_on_flop=2,     # heads-up solve
    )
    decision = DecisionData(
        options=[a.label for a in ctx.actions],
        hero_combo_evs={_canonical(a.label): a.ev for a in ctx.actions},
        range_aggregate_strategy={_canonical(a.label): min(1.0, max(0.0, a.frequency))
                                  for a in ctx.actions},
        street_actions=street_actions,
        prior_street_actions=prior_actions,
        correct_action=_canonical(max(ctx.actions,
                                      key=lambda a: a.frequency).label),
    )
    return SpotData(
        spot_metadata=metadata,
        decision_data=decision,
        hand_class=HandClass.from_cards(hero_combo, node.board),
        board_texture=BoardTexture.from_board(node.board),
    ), hero_combo


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

        chosen = random.Random(RANDOM_SEED).sample(nodes, min(SAMPLE_SIZE, len(nodes)))
        for index, node in enumerate(chosen, start=1):
            ctx = sampler.build_spot_context(node)
            spot, hero_combo = spot_data_from_context(ctx)
            tags = compute_tags(spot)

            print("=" * 70)
            print(f"SPOT {index}:  {node.node_id}   ({node.node_type})")
            print(f"  street/board : {node.street}  {' '.join(node.board)}")
            print(f"  pot / stack  : {node.pot:.0f} / {node.effective_stack:.0f}"
                  f"  (SPR {spot.spot_metadata.spr:.1f})")
            print(f"  hero         : {node.hero_position} with {hero_combo}"
                  f"  ->  {spot.hand_class.label}")
            print(f"  board texture: {spot.board_texture.composite}")
            line = "  ".join(f"{actor}:{label}"
                              for actor, label in node.action_sequence) or "(first to act)"
            print(f"  line         : {line}")
            actions = "  ".join(f"{a.label} [{a.frequency * 100:.0f}%, "
                                f"EV {a.ev:.1f}]" for a in ctx.actions)
            print(f"  actions      : {actions}")
            print(f"  concept tags : {', '.join(tags) if tags else '(none)'}")

        print("=" * 70)
        print("\nNote: equity_data and range_data are left at defaults -- the "
              "equity/range/blocker\nfact-extraction that feeds Sections A, E, F "
              "and most of B is still to be built.\nThis demo confirms the "
              "solver -> Path Sampler -> SpotData -> compute_tags path works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
