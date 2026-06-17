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


def _load_solve(name: str):
    """Resolve a solve by name. Today only the synthetic fixture exists; a
    real-solve adapter would dispatch here on a path / format flag."""
    if name in ("fixture", "btn_vs_bb_srp_2cJs7s"):
        return btn_vs_bb_srp_2cJs7s()
    raise SystemExit(
        f"unknown solve {name!r}. Only the synthetic fixture is wired today; "
        "a real-solve adapter (.cfr/.db) is the documented next step."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate postflop questions.")
    parser.add_argument("--solve", default="fixture", help="solve to use (default: the synthetic fixture)")
    parser.add_argument("--out", default="test_output/postflop_demo.csv", help="output CSV path")
    parser.add_argument("-n", "--num", type=int, default=30, help="max questions")
    parser.add_argument("--dry-run", action="store_true", help="no API call; deterministic placeholder prose")
    parser.add_argument("--model", default=None, help="override the LLM model")
    args = parser.parse_args(argv)

    solve = _load_solve(args.solve)
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
