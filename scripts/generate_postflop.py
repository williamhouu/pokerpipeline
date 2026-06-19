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

_RANKS = "23456789TJQKA"


def _load_solve(name: str):
    """Resolve a solve by name or a vendor-file path.

    ``fixture`` -> the synthetic in-memory solve. A path ending in ``.db`` ->
    the third-party SQLite adapter (flop nodes, v1). A ``.cfr`` path is the
    documented next step (the PioSolver UPI client)."""
    if name in ("fixture", "btn_vs_bb_srp_2cJs7s"):
        return btn_vs_bb_srp_2cJs7s()
    if name.endswith(".db"):
        from pipeline.postflop.adapters.sqlite_db import load_postflop_db

        return load_postflop_db(name)
    raise SystemExit(
        f"unknown solve {name!r}. Pass 'fixture' or a vendor '.db' path; a "
        "'.cfr' adapter (PioSolver UPI) is the documented next step."
    )


def _combo_class(combo: str) -> str:
    """'AsKs' -> 'AKs', '3s3c' -> '33' (suit-isomorphic 169 class)."""
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if _RANKS.index(r1) > _RANKS.index(r2) else (r2, r1)
    return hi + lo + ("s" if s1 == s2 else "o")


def _node_kind(node_id: str) -> str:
    """Classify a flop node into a decision type for diversity grouping."""
    t = node_id.split(":")[2:]
    if not t:
        return "bb_lead"
    if t == ["c"]:
        return "btn_cbet"
    if len(t) == 1 and t[0].startswith("b"):
        return "btn_faces_donk"
    if len(t) == 2 and t[0] == "c" and t[1].startswith("b"):
        return "bb_faces_cbet"
    return "deeper"


def diversify_spots(worthy, *, per_class: int = 1, max_depth: int = 2):
    """Deterministically curate the worthy pool for a varied fill-to-N batch.

    Drops all-in / deep multi-raise lines (poor first questions), caps each
    (node, 169-class) to ``per_class`` combos (kills near-duplicate suits), then
    round-robins across the four decision types so the batch is not dominated by
    one archetype. Pure + sorted -> byte-identical output preserved.
    """
    from collections import defaultdict

    seen: dict[tuple, int] = defaultdict(int)
    buckets: dict[str, list] = defaultdict(list)
    for spot in sorted(worthy, key=lambda s: (s.node.node_id, s.hero_combo)):
        nid = spot.node.node_id
        if "b9697" in nid or (nid.count(":") - 1) > max_depth:
            continue
        kind = _node_kind(nid)
        if kind == "deeper":
            continue
        key = (kind, nid, _combo_class(spot.hero_combo))
        if seen[key] >= per_class:
            continue
        seen[key] += 1
        buckets[kind].append(spot)

    order = ["btn_cbet", "bb_faces_cbet", "btn_faces_donk", "bb_lead"]
    out, i = [], 0
    while any(i < len(buckets[k]) for k in order):
        for k in order:
            if i < len(buckets[k]):
                out.append(buckets[k][i])
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate postflop questions.")
    parser.add_argument("--solve", default="fixture", help="'fixture' or a vendor '.db' path")
    parser.add_argument("--out", default="test_output/postflop_demo.csv", help="output CSV path")
    parser.add_argument("-n", "--num", type=int, default=30, help="max questions")
    parser.add_argument("--dry-run", action="store_true", help="no API call; deterministic placeholder prose")
    parser.add_argument("--model", default=None, help="override the LLM model")
    parser.add_argument(
        "--diversify", action="store_true",
        help="round-robin the worthy spots across the four flop decision types "
             "so a fill-to-N batch is not dominated by one archetype "
             "(recommended for real solves)",
    )
    parser.add_argument(
        "--min-ev-gap", type=float, default=None,
        help="OPTIONAL EV-gap quality filter in bb (off by default, like "
             "preflop): drop spots whose EV gap is below this",
    )
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
