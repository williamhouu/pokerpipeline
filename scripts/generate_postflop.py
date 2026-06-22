#!/usr/bin/env python
"""CLI for the postflop question pipeline.

Runs :func:`pipeline.postflop.batch.generate_postflop_batch` against a solve
and writes a CSV (+ a ``.meta.json`` sidecar). v1 sources the solve from the
synthetic fixture, so this runs with NO solver file and NO API key in dry-run
mode -- the deterministic end-to-end demo. A real-solve adapter (Pio ``.cfr``
/ the third-party ``.db``) plugs in at the marked spot once available.

Examples
--------
    # Deterministic dry run (no API key needed) -> a CSV you can open:
    python scripts/generate_postflop.py --dry-run --out test_output/postflop_demo.csv

    # Real run (needs ANTHROPIC_API_KEY):
    python scripts/generate_postflop.py --out test_output/postflop.csv -n 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.postflop.batch import generate_postflop_batch  # noqa: E402
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.spot_selection import diversify_spots  # noqa: E402


def _load_solve(name: str, *, streets: tuple[str, ...], max_nodes_per_street: int | None):
    """Resolve a solve by name or a vendor-file path.

    ``fixture`` -> the synthetic in-memory solve (``streets`` is ignored; it is a
    fixed flop solve). A path ending in ``.db`` -> the third-party SQLite adapter
    (``streets`` selects flop/turn/river; ``max_nodes_per_street`` caps each). A
    ``.cfr`` path is the documented next step (the PioSolver UPI client)."""
    if name in ("fixture", "btn_vs_bb_srp_2cJs7s"):
        return btn_vs_bb_srp_2cJs7s()
    if name.endswith(".db"):
        from pipeline.postflop.adapters.sqlite_db import load_postflop_db

        return load_postflop_db(
            name, streets=streets, max_nodes_per_street=max_nodes_per_street
        )
    raise SystemExit(
        f"unknown solve {name!r}. Pass 'fixture' or a vendor '.db' path; a "
        "'.cfr' adapter (PioSolver UPI) is the documented next step."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate postflop questions.")
    parser.add_argument("--solve", default="fixture", help="'fixture' or a vendor '.db' path")
    parser.add_argument("--out", default="test_output/postflop_demo.csv", help="output CSV path")
    parser.add_argument("-n", "--num", type=int, default=30, help="max questions")
    parser.add_argument("--dry-run", action="store_true", help="no API call; deterministic placeholder prose")
    parser.add_argument("--model", default=None, help="override the LLM model")
    parser.add_argument(
        "--streets", nargs="+", choices=["flop", "turn", "river"], default=["flop"],
        help="which streets to generate questions from (default: flop). '.db' "
             "solves only; the fixture is flop-only.",
    )
    parser.add_argument(
        "--max-nodes-per-street", type=int, default=600,
        help="cap on nodes built per street (turn/river sets are huge; "
             "deterministically down-sampled). 0 = no cap.",
    )
    parser.add_argument(
        "--diversify", action="store_true",
        help="round-robin the worthy spots across streets and decision types so "
             "a fill-to-N batch is not dominated by one archetype/street "
             "(recommended for real solves)",
    )
    parser.add_argument(
        "--min-ev-gap", type=float, default=None,
        help="OPTIONAL EV-gap quality filter in bb (off by default, like "
             "preflop): drop spots whose EV gap is below this",
    )
    args = parser.parse_args(argv)

    cap = args.max_nodes_per_street if args.max_nodes_per_street > 0 else None
    solve = _load_solve(args.solve, streets=tuple(args.streets), max_nodes_per_street=cap)
    client = None
    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set; falling back to --dry-run.", file=sys.stderr)
            args.dry_run = True
        else:
            from anthropic import Anthropic  # local import: only needed for a real run

            client = Anthropic()

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.diversify:
        kwargs["spot_selector"] = diversify_spots
    if args.min_ev_gap is not None:
        kwargs["min_ev_gap_bb"] = args.min_ev_gap
    result = generate_postflop_batch(
        solve=solve,
        output_path=args.out,
        total_questions=args.num,
        client=client,
        dry_run=args.dry_run,
        progress_callback=lambda msg, done, total: print(f"  {msg}", file=sys.stderr),
        **kwargs,
    )
    print(
        f"Wrote {result.questions_written}/{result.requested_questions} questions "
        f"({result.worthy_spots_available} worthy, {result.soft_flagged_rows} flagged, "
        f"{len(result.failures)} failed) -> {result.output_path}"
    )
    if result.meta_path:
        print(f"Meta sidecar: {result.meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
