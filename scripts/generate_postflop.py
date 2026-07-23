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
from pipeline.postflop.facts import preflop_aggressor  # noqa: E402
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.premise import DEFAULT_MIN_PREMISE_FREQ  # noqa: E402
from pipeline.postflop.spot_selection import (  # noqa: E402
    DECISION_TYPES,
    STRENGTH_BUCKETS,
    make_spot_selector,
)


def _load_solve(
    name: str, *, streets: tuple[str, ...], max_nodes_per_street: int | None,
    include_ancestors: bool = False,
):
    """Resolve a solve by name or a vendor-file path.

    ``fixture`` -> the synthetic in-memory solve (``streets`` is ignored; it is a
    fixed flop solve). ``full-hand-fixture`` -> the connected single-hand line
    (for play-through demos). A path ending in ``.db`` -> the third-party SQLite
    adapter (``streets`` selects flop/turn/river; ``max_nodes_per_street`` caps
    each; ``include_ancestors`` line-closes for play-through assembly). A
    ``.cfr`` path is the documented next step (the PioSolver UPI client)."""
    if name in ("fixture", "btn_vs_bb_srp_2cJs7s"):
        return btn_vs_bb_srp_2cJs7s()
    if name in ("full-hand-fixture", "btn_vs_bb_full_hand_2cJs7s"):
        from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s

        return btn_vs_bb_full_hand_2cJs7s()
    if name.endswith(".db"):
        from pipeline.postflop.adapters.sqlite_db import load_postflop_db

        return load_postflop_db(
            name, streets=streets, max_nodes_per_street=max_nodes_per_street,
            include_ancestors=include_ancestors,
        )
    raise SystemExit(
        f"unknown solve {name!r}. Pass 'fixture' or a vendor '.db' path; a "
        "'.cfr' adapter (PioSolver UPI) is the documented next step."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate postflop questions.")
    parser.add_argument("--solve", default="fixture", help="'fixture' or a vendor '.db' path")
    parser.add_argument("--out", default="test_output/postflop_demo.csv", help="output CSV path")
    parser.add_argument("-n", "--num", type=int, default=30, help="max questions (or hands, in full-hand mode)")
    parser.add_argument(
        "--mode", choices=["spots", "full-hands", "preflop"], default="spots",
        help="'spots' = independent postflop questions (default); 'full-hands' = "
             "linked preflop->river play-through sequences (Option B hand_id/"
             "sequence_index); 'preflop' = standalone preflop-entry questions "
             "from the solve's flop-entry ranges. -n is the HAND count in "
             "full-hands mode.",
    )
    parser.add_argument(
        "--heroes", nargs="*", default=[],
        help="restrict to these hero seats (e.g. BB BTN); default = both",
    )
    parser.add_argument(
        "--include-villain", action="store_true",
        help="full-hands mode: also emit the villain's line on the same runout "
             "as a separate hand (the 'flip the frame' path)",
    )
    parser.add_argument(
        "--llm-workers", type=int, default=1,
        help="full-hands mode: generate a hand's legs with this many "
             "concurrent LLM calls (1 = sequential; ~Nx faster, same cost)",
    )
    parser.add_argument(
        "--no-action-heavy", action="store_true",
        help="full-hands mode: disable the 🎬 action-heavy policy (checkdown "
             "quota + trivial-fold-ender drop + density ordering + "
             "postflop-spine difficulty banding); on by default",
    )
    parser.add_argument(
        "--no-preflop-leg", action="store_true",
        help="full-hands mode: omit the preflop-entry leg from each hand",
    )
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
    parser.add_argument(
        "--strength", nargs="*", default=[], choices=STRENGTH_BUCKETS,
        help="keep only spots where hero's made-hand bucket is one of these "
             "(the postflop hand-strength filter); default = all",
    )
    parser.add_argument(
        "--decision", nargs="*", default=[], choices=DECISION_TYPES,
        help="keep only these decision situations (the 'action faced' analog); "
             "default = all",
    )
    parser.add_argument(
        "--no-quality-gate", action="store_true",
        help="disable the convergence/reach guard (by default low-quality / "
             "barely-reached nodes are skipped)",
    )
    parser.add_argument(
        "--min-premise-freq", type=float, default=DEFAULT_MIN_PREMISE_FREQ,
        help="premise-realism gate (0-1): drop a spot whose line includes a "
             "PRIOR action taken below this frequency (a question built on a "
             "line someone almost never takes). Default %(default)s; 0 disables.",
    )
    parser.add_argument(
        "--trap-difficulty", action="store_true",
        help="opt-in: floor counterintuitive PURE spots (solver folds despite "
             "equity >= price, etc.) to Hard, like preflop. Score only",
    )
    parser.add_argument(
        "--claim-checker", action="store_true",
        help="Layer-7: a 2nd LLM pass audits each explanation against the data "
             "block and FLAGS suspect claims (flag only; one extra call/question)",
    )
    parser.add_argument(
        "--revise", action="store_true",
        help="Layer-7 auto-fix: when the claim checker flags a question, a 3rd "
             "LLM pass rewrites the prose (re-validated; discarded if it breaks a "
             "rule). Implies --claim-checker as the gate",
    )
    parser.add_argument(
        "--final-audit", action="store_true",
        help="with --revise, re-run the claim checker on the rewrite (4th call, "
             "flag only)",
    )
    parser.add_argument(
        "--length-profile", choices=["river_heavy", "river_leaning", "equal"],
        default="river_heavy",
        help="with --balanced-lengths: hand-count share per ending street "
             "(river_heavy = 70%% river, the production default since July "
             "2026 -- river_leaning's 40%% rounds to a near-equal mix on "
             "small batches; equal = 25%% each)")
    parser.add_argument(
        "--balanced-lengths", action="store_true",
        help="full-hands mode: equal quarters of hands ending preflop / flop "
             "/ turn / river (early enders = correct-fold endings; the "
             "production anti-predictability mix); ignored with a "
             "difficulty band",
    )
    parser.add_argument(
        "--diversify-hands", action="store_true",
        help="full-hands mode: round-robin the batch across hero seat, line "
             "depth, and hand strength (assembles an oversized pool first); "
             "ignored with a difficulty band",
    )
    parser.add_argument(
        "--variety-seed", type=int, default=None, metavar="N",
        help="full-hands mode: shuffle which hands are picked (same seed = same "
             "hands; omit for the legacy fixed order, which serves the SAME "
             "hands every batch). The admin panel passes a fresh seed per run",
    )
    args = parser.parse_args(argv)

    cap = args.max_nodes_per_street if args.max_nodes_per_street > 0 else None
    # Full-hand mode needs every selected node's ancestors so a connected line
    # resolves; for the fixture it's already connected. Default full-hand streets
    # to all three so play-throughs can reach the river.
    full_hand = args.mode == "full-hands"
    streets = tuple(args.streets)
    if full_hand and streets == ("flop",):
        streets = ("flop", "turn", "river")
    solve = _load_solve(
        args.solve, streets=streets, max_nodes_per_street=cap,
        include_ancestors=full_hand,
    )
    # Provenance for a vendor-solve batch: what audit_postflop_batch.py needs
    # to REBUILD every row (db path + sampling) and to reproduce the display
    # framing exactly (stakes / venue). Without it a CLI batch re-verifies
    # with default framing and false-fails the Context columns. Mirrors the
    # dict pipeline/postflop/run.py records for admin batches.
    provenance = None
    if args.solve.endswith(".db"):
        provenance = {
            "db_path": str(Path(args.solve).resolve()),
            "streets": list(streets),
            "max_nodes_per_street": cap,
            "stakes": solve.stakes,
            "live_or_online": solve.live_or_online,
            "bb_in_dollars": solve.bb_in_dollars,
        }
    client = None
    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set; falling back to --dry-run.", file=sys.stderr)
            args.dry_run = True
        else:
            from anthropic import Anthropic  # local import: only needed for a real run

            client = Anthropic()

    # --- play-through / preflop-standalone modes (separate drivers) ---------
    if args.mode in ("full-hands", "preflop"):
        from pipeline.postflop.full_hand_batch import (
            generate_full_hand_batch,
            generate_preflop_entry_batch,
        )

        common = dict(
            solve=solve, output_path=args.out, client=client, dry_run=args.dry_run,
            heroes=tuple(args.heroes), answer_style="auto",
            progress_callback=lambda msg, done, total: print(f"  {msg}", file=sys.stderr),
        )
        if args.model:
            common["model"] = args.model
        if provenance is not None:
            common["provenance"] = provenance
        if args.mode == "full-hands":
            res = generate_full_hand_batch(
                total_hands=args.num, include_villain=args.include_villain,
                include_preflop=not args.no_preflop_leg,
                action_heavy=not args.no_action_heavy,
                llm_workers=args.llm_workers,
                # Pack-backed preflop legs from the repo's ranges/ -- the same
                # root the admin driver (run.py) uses. Without it the legs
                # degrade to entry-derived (SRP) or drop (3-bet pots).
                preflop_leg_pack_root=(
                    Path(__file__).resolve().parent.parent / "ranges"
                ),
                variety_seed=args.variety_seed,
                diversify_hands=args.diversify_hands,
                balanced_lengths=args.balanced_lengths,
                length_profile=args.length_profile,
                min_ev_gap_bb=args.min_ev_gap,
                quality_gate=not args.no_quality_gate,
                min_premise_freq=(args.min_premise_freq if args.min_premise_freq > 0 else None),
                trap_difficulty=args.trap_difficulty,
                # Layer-7 audit on the postflop legs (same flags as spots mode).
                run_claim_checker=args.claim_checker or args.revise,
                revise_pass=args.revise,
                final_audit=args.final_audit and args.revise,
                **common,
            )
        else:
            res = generate_preflop_entry_batch(total_questions=args.num, **common)
        print(
            f"Wrote {res.questions_written} questions "
            f"({res.worthy_spots_available} worthy, {len(res.failures)} failed) "
            f"-> {res.output_path}"
        )
        if res.meta_path:
            print(f"Meta sidecar: {res.meta_path}")
        return 0

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    # Build a curating selector when diversify / hand-strength / decision filters
    # are requested (the selector handles all three + needs the aggressor/IP for
    # the decision-type classifier).
    if args.diversify or args.strength or args.decision:
        kwargs["spot_selector"] = make_spot_selector(
            diversify=args.diversify,
            strength_buckets=tuple(args.strength) or None,
            decision_types=tuple(args.decision) or None,
            aggressor=preflop_aggressor(solve),
            ip_position=solve.ip_position,
        )
    if args.min_ev_gap is not None:
        kwargs["min_ev_gap_bb"] = args.min_ev_gap
    if provenance is not None:
        kwargs["provenance"] = provenance
    result = generate_postflop_batch(
        solve=solve,
        output_path=args.out,
        total_questions=args.num,
        client=client,
        dry_run=args.dry_run,
        quality_gate=not args.no_quality_gate,
        min_premise_freq=(args.min_premise_freq if args.min_premise_freq > 0 else None),
        trap_difficulty=args.trap_difficulty,
        run_claim_checker=args.claim_checker or args.revise,
        revise_pass=args.revise,
        final_audit=args.final_audit and args.revise,
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
