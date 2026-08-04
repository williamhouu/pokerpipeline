"""📏 Bet-sizing trainer CLI: one fully-balanced batch across ALL solves.

Pools sizing-viable spots (menu >= 2 open-bet sizes, dominant action IS a
sized bet) from every ``.db`` under ``solves/postflop/`` and balances the
batch across flops, streets, difficulty, correct size, situation, position,
and hand strength. See ``pipeline/postflop/sizing_batch.py``.

Usage::

    venv/bin/python scripts/generate_sizing_batch.py --dry-run           # free
    venv/bin/python scripts/generate_sizing_batch.py --count 50          # real
    venv/bin/python scripts/generate_sizing_batch.py --solve-dir <dir> --out X.csv

A real run needs ANTHROPIC_API_KEY (env or the repo-root .env) and defaults
to the full Layer-7 audit chain (claim gate + auto-fix + final audit).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.postflop.adapters.sqlite_db import discover_db_solves  # noqa: E402
from pipeline.postflop.run import (  # noqa: E402
    POSTFLOP_OUTPUT_DIR,
    generate_sizing_batch_from_paths,
)


def _load_env_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY") and "=" in line:
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["ANTHROPIC_API_KEY"] = key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve-dir", default=str(ROOT / "solves" / "postflop"))
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-layer7", action="store_true",
                    help="skip the claim gate / auto-fix / final audit")
    ap.add_argument("--llm-workers", type=int, default=1,
                    help="concurrent question LLM chains (1 = sequential; "
                         "~Nx faster, same cost, identical output)")
    args = ap.parse_args()

    if not args.dry_run:
        _load_env_key()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("No ANTHROPIC_API_KEY (env or .env) — use --dry-run or set it.")

    summaries = [s for s in discover_db_solves(args.solve_dir) if s.ok]
    if not summaries:
        sys.exit(f"no readable .db solves under {args.solve_dir}")
    db_paths = tuple(s.path for s in summaries)
    name = args.out or (
        f"SIZING TRAINER {args.count}q" + (" dry" if args.dry_run else "") + ".csv"
    )
    out = POSTFLOP_OUTPUT_DIR / name

    layer7 = not (args.dry_run or args.no_layer7)
    result = generate_sizing_batch_from_paths(
        db_paths=db_paths,
        output_path=out,
        total_questions=args.count,
        dry_run=args.dry_run,
        run_claim_checker=layer7,
        revise_pass=layer7,
        final_audit=layer7,
        llm_workers=args.llm_workers,
    )
    print(f"rows: {result.questions_written}/{result.requested_questions} -> {out}")
    print(f"pool scored: {result.pool_scored} | per-solve: {result.per_solve_written}")
    if result.solves_skipped:
        print("skipped solves:", result.solves_skipped)
    if result.total_input_tokens:
        print(f"tokens: {result.total_input_tokens} in / "
              f"{result.total_output_tokens} out")


if __name__ == "__main__":
    main()
