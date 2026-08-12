"""Audit the NLHE 6-max short-stack Monker packs (20bb / 30bb).

Phase-0-style verification for the short-stack 6-max integration (June
2026), run BEFORE the packs are registered for admin generation. These
packs share the Monker `.rng` format with the 9-max pack but use two
tokens the 9-max pack never does -- the min-raise open ``5`` (2bb) and
the BvB iso-over-a-limp ``14`` (2.5bb) -- so the headline section here is
the **token-size lock**, which re-derives both sizes from the solve's own
EV bookkeeping (independent of the pipeline's size walk).

Four sections:

1. **Tree sanity** -- node count, first-in (RFI) strategy per seat (opens
   are the ``5`` min-raise, so the raise column is the 2bb open), and an
   AA-always-continues canary on the BB-vs-open node.
2. **Token-size lock** -- the important one. The ``.rng`` EVs are in
   **milli-small-blinds** here (NOT the 9-max pack's milli-bb): a fold's
   EV equals minus the chips already committed, and the SB-blind fold
   reads 1.0 (= one small blind = 0.5bb), proving the unit. Given that, an
   opener who later folds reads 4.0 sb == the 2bb ``5`` open, and the BB
   who iso-raised ``14`` then folds reads 5.0 sb == 2.5bb. If a future
   short-stack pack reuses ``5``/``14`` at a different size this section
   fails loudly.
3. **EV-scale note** -- restates the milli-sb finding with the same
   indifference + fold-anchor proofs the 9-max audit uses.
4. **Render slice** -- run the real fact extractor + format writer on a
   min-raise open, a BvB limp-iso, and a 3-bet spot, and print the key CSV
   columns so the 6-seat prose, Seats tokens, pot math, archetypes, and
   skills can be eyeballed.

Usage::

    venv/bin/python scripts/audit_nlhe6_pack.py                  # both depths
    venv/bin/python scripts/audit_nlhe6_pack.py --depth 20
    venv/bin/python scripts/audit_nlhe6_pack.py --depth 30 --section sizes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.batch import _placeholder_explanation  # noqa: E402
from pipeline.preflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.preflop.ev_engine import compute_ev_gap_bb  # noqa: E402
from pipeline.preflop.fact_extractor import extract_facts  # noqa: E402
from pipeline.preflop.format_writer import build_preflop_row  # noqa: E402
from pipeline.preflop.grammars.monker_nlhe import parse  # noqa: E402
from pipeline.preflop.grammars.types import (  # noqa: E402
    MIN_RAISE_PCT,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.options import build_options  # noqa: E402
from pipeline.preflop.pack import PreflopPack  # noqa: E402
from pipeline.preflop.spot_sampler import sample_spot  # noqa: E402
from pipeline.preflop_ranges import parse_monker_rng_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SEAT_ORDER = ("UTG", "HJ", "CO", "BTN", "SB", "BB")

# Deterministic landmarks from the first walk per depth (20/30: June 14
# 2026; 40-200: the Aug 5 2026 ladder intake, 50bb counted WITH its token-15
# fixed-size registration).
EXPECTED_NODES = {
    20: 913,
    30: 2247,
    40: 2884,
    50: 2994,
    70: 2526,
    100: 9270,
    150: 14690,
    200: 21924,
}

_COMBOS = {"pair": 6, "suited": 4, "offsuit": 12}


def pack_root(depth: int) -> Path:
    return REPO_ROOT / f"nlhe6_ranges/ranges/Hold'em/6-way/{depth}bb(5p-0.5bb)"


def build_pack(depth: int) -> PreflopPack:
    return PreflopPack(
        pack_id=f"monker_nlhe_6max_{depth}bb",
        root_path=pack_root(depth),
        grammar_name="monker_nlhe",
        table_size=6,
        stack_depth_bb=depth,
        open_size_bb=(2.0 if depth <= 50 else 2.5),
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,  # milli-sb; the size lock below verifies it
        # 50bb only: the BvB iso token 15 = 3bb (EV-anchored; matches the
        # registered pack in pipeline/preflop/pack.py).
        fixed_raise_tokens_bb=((("15", 3.0),) if depth == 50 else None),
    )


def _combos(hand_class: str) -> int:
    if len(hand_class) == 2:
        return _COMBOS["pair"]
    return _COMBOS["suited"] if hand_class.endswith("s") else _COMBOS["offsuit"]


def _class_range_pct(weights: dict[str, float]) -> float:
    total = sum(w * _combos(h) for h, w in weights.items())
    return 100.0 * total / 1326.0


# --- section 1: tree sanity ---------------------------------------------------
def audit_tree(pack: PreflopPack, nodes: tuple[PreflopDecisionNode, ...]) -> None:
    depth = pack.stack_depth_bb
    print("== tree sanity ==")
    expected = EXPECTED_NODES.get(depth)
    print(f"nodes: {len(nodes):,} (expected {expected})")
    if expected is not None:
        assert len(nodes) == expected, "node count drifted from the June-14 walk"

    print("\nfirst-in strategy per seat (% of combos):")
    print(f"  {'seat':5s} {'fold%':>7s} {'limp%':>7s} {'raise%':>7s}  options")
    for n_folds, seat in enumerate(SEAT_ORDER[:-1]):  # BB never first-in
        node = next(
            n
            for n in nodes
            if n.actor == seat
            and len(n.history_before) == n_folds
            and all(a.action_type.value == "Fold" for a in n.history_before)
        )
        by_kind = {"fold": 0.0, "limp": 0.0, "raise": 0.0}
        for opt in node.actions:
            weights = {
                h: p
                for h, (p, _e) in parse_monker_rng_file(opt.range_file.path).items()
            }
            pct = _class_range_pct(weights)
            if opt.label == "Fold":
                by_kind["fold"] += pct
            elif opt.label == "Call":
                by_kind["limp"] += pct
            else:
                by_kind["raise"] += pct
        labels = [o.label for o in node.actions]
        print(
            f"  {seat:5s} {by_kind['fold']:6.1f}% {by_kind['limp']:6.1f}% "
            f"{by_kind['raise']:6.1f}%  {labels}"
        )

    # AA canary on the BB-vs-UTG-open node (UTG opens `5`, folds to BB).
    open_stem = ("5" if pack.stack_depth_bb <= 50 else "40060") + ".0" * 4
    bb_defend = next(
        n
        for n in nodes
        if n.actor == "BB"
        and len(n.history_before) == 5
        and n.actions[0].range_file.path.stem.startswith(open_stem)
    )
    aa_continue = sum(
        parse_monker_rng_file(opt.range_file.path)["AA"][0]
        for opt in bb_defend.actions
        if opt.label != "Fold"
    )
    print(f"\nBB vs UTG open: AA continues {aa_continue:.3f} (must be ~1.0)")
    assert aa_continue > 0.98


# --- section 2: token-size lock ----------------------------------------------
def _fold_commit_sb(path: Path) -> float | None:
    """Median ``-EV/1000`` over hands folded with weight>0.05.

    A fold's EV is minus the chips already committed; the EV unit here is
    milli-**small-blinds**, so this returns the committed amount in small
    blinds (= 2x the bb amount).
    """
    if not path.exists():
        return None
    data = parse_monker_rng_file(path)
    commits = sorted(-ev / 1000.0 for _h, (p, ev) in data.items() if p > 0.05)
    return commits[len(commits) // 2] if commits else None


def _find_opener_fold_anchor(
    pack: PreflopPack, open_tok: str = "5"
) -> Path | None:
    """Shallowest file where the seat that OPENED (``open_tok``) then folds.

    Its fold EVs read minus the open size, in small blinds -- the
    independent proof of the open token's decoded size. The opener is the
    first raise in the history (token-agnostic, so the `5` min-raise packs
    and the `40060` pot-relative packs both anchor).
    """
    best: tuple[int, Path] | None = None
    for f in pack.root_path.glob(f"{open_tok}.*.0.rng"):
        ah = parse(f, pack).action_history
        opener = next(
            (
                a.position
                for a in ah
                if a.action_type is PreflopActionType.RAISE
            ),
            None,
        )
        if ah[-1].position == opener and ah[-1].action_type is PreflopActionType.FOLD:
            if best is None or len(ah) < best[0]:
                best = (len(ah), f)
    return best[1] if best else None


def audit_token_sizes(pack: PreflopPack) -> None:
    print("\n== token-size lock (EV-derived, independent of the size walk) ==")
    root = pack.root_path
    failures = 0
    depth = pack.stack_depth_bb
    # Per-depth conventions (Aug 5 2026 intake of the 40-200bb ladder):
    # opens are the `5` min-raise (2bb) through 50bb, the `40060`
    # pot-relative token (2.5bb) from 70bb; the BvB iso over the SB limp is
    # `14` (2.5bb) through 40bb, the pack-registered `15` (3bb) at 50bb,
    # and pot-relative `40100` (3bb) from 70bb.
    open_tok = "5" if depth <= 50 else "40060"
    open_sb_expect = 4.0 if depth <= 50 else 5.0
    if depth <= 40:
        iso_anchor, iso_sb_expect = "0.0.0.0.1.14.3.0.rng", 5.0
    elif depth == 50:
        iso_anchor, iso_sb_expect = "0.0.0.0.1.15.40125.0.rng", 6.0
    else:
        iso_anchor, iso_sb_expect = "0.0.0.0.1.40100.40350.0.rng", 6.0

    # 1. Unit proof: SB folds first-in having posted only the 0.5bb blind.
    sb_blind = _fold_commit_sb(root / f"{open_tok}.0.0.0.0.rng")
    ok = sb_blind is not None and abs(sb_blind - 1.0) < 0.02
    failures += not ok
    print(
        f"  SB-blind fold reads {sb_blind!r} sb (expect 1.0 = 0.5bb -> unit is "
        f"milli-SB) {'OK' if ok else 'FAIL'}"
    )

    # 2. Open size lock: an opener who later folds is out the open size.
    anchor = _find_opener_fold_anchor(pack, open_tok)
    open_sb = _fold_commit_sb(anchor) if anchor else None
    ok = open_sb is not None and abs(open_sb - open_sb_expect) < 0.05
    failures += not ok
    name = anchor.name if anchor else "<none found>"
    print(
        f"  `{open_tok}` open via {name}: opener committed {open_sb!r} sb "
        f"(expect {open_sb_expect} = {open_sb_expect / 2:g}bb) "
        f"{'OK' if ok else 'FAIL'}"
    )

    # 3. BvB iso size lock: BB iso-raises over the SB limp, SB re-raises,
    #    BB folds -> BB committed exactly the iso size. The anchor file is
    #    discovered (the SB's re-raise token differs by depth).
    iso_tok = iso_anchor.split(".")[5]
    found = sorted(
        root.glob(f"0.0.0.0.1.{iso_tok}.*.0.rng"), key=lambda p: len(p.name)
    )
    iso_path = root / iso_anchor if (root / iso_anchor).exists() else (
        found[0] if found else None
    )
    iso_sb = _fold_commit_sb(iso_path) if iso_path else None
    ok = iso_sb is not None and abs(iso_sb - iso_sb_expect) < 0.05
    failures += not ok
    print(
        f"  `{iso_tok}` BB-iso via {iso_path.name if iso_path else '<none>'}: "
        f"BB committed {iso_sb!r} sb (expect {iso_sb_expect} = "
        f"{iso_sb_expect / 2:g}bb) {'OK' if ok else 'FAIL'}"
    )

    # 4. The pack's ev_units_per_bb converts file EVs to bb correctly: the
    #    BB-iso fold must read minus the iso size in bb through the divisor
    #    the format writer uses for the action_ev_bb column.
    units = pack.ev_units_per_bb
    iso_raw = None
    if iso_path is not None and iso_path.exists():
        data = parse_monker_rng_file(iso_path)
        evs = [ev for _h, (p, ev) in data.items() if p > 0.05]
        iso_raw = sorted(evs)[len(evs) // 2] if evs else None
    iso_bb = None if (iso_raw is None or not units) else iso_raw / units
    ok = iso_bb is not None and abs(iso_bb - (-iso_sb_expect / 2)) < 0.05
    failures += not ok
    print(
        f"  ev_units_per_bb={units}: BB-iso fold -> {iso_bb!r} bb "
        f"(expect {-iso_sb_expect / 2}) {'OK' if ok else 'FAIL'}"
    )

    print(f"  {'ALL TOKEN SIZES LOCKED' if failures == 0 else f'{failures} FAILURES'}")
    assert failures == 0, "token-size lock failed -- decode drifted"


# --- section 3: render slice ---------------------------------------------------
def render_spots(pack: PreflopPack, nodes: tuple[PreflopDecisionNode, ...]) -> None:
    print("\n== render slice (real fact extractor + format writer) ==")
    open_stem = ("5" if pack.stack_depth_bb <= 50 else "40060") + ".0" * 4

    def first(pred):
        return next(n for n in nodes if pred(n))

    picks = [
        (
            "BTN first-in open (min-raise, A5s)",
            first(
                lambda n: n.actor == "BTN"
                and len(n.history_before) == 3
                and all(a.action_type.value == "Fold" for a in n.history_before)
            ),
            "A5s",
        ),
        (
            "BB vs UTG open (defend, KQs)",
            first(
                lambda n: n.actor == "BB"
                and len(n.history_before) == 5
                and n.actions[0].range_file.path.stem.startswith(open_stem)
            ),
            "KQs",
        ),
        (
            "BB vs SB limp (iso candidate, A8s)",
            first(
                lambda n: n.actor == "BB"
                and len(n.history_before) == 5
                and all(a.action_type.value == "Fold" for a in n.history_before[:4])
                and n.history_before[-1].action_type.value == "Call"
            ),
            "A8s",
        ),
    ]
    for i, (title, node, hand) in enumerate(picks, start=1):
        spot = sample_spot(node, hand)
        facts = extract_facts(spot, pack, equity_runouts=80)
        ev_gap = compute_ev_gap_bb(facts, pack)
        options, correct = build_options(facts)
        row = build_preflop_row(
            facts,
            _placeholder_explanation(options, correct),
            pack=pack,
            difficulty=compute_difficulty(facts, ev_gap_bb=ev_gap),
            number=i,
        )
        print(f"\n--- {title} ---")
        print(f"node: {node.node_id}")
        print(
            f"freqs: { {k: round(v, 3) for k, v in spot.action_frequencies.items()} }"
        )
        print(f"archetype: {facts.archetype}")
        for col in (
            "Question",
            "User Seat",
            "Seats",
            "Table Size",
            "POT",
            "Relative Position",
            "Position Matchup",
            "skills",
            "concept_tags",
            "Notes",
            "Difficulty Rating",
            "ev_gap_bb",
        ):
            print(f"{col}: {row.get(col, '<missing>')!r}")


def run_depth(depth: int, section: str) -> None:
    root = pack_root(depth)
    if not root.is_dir():
        sys.exit(f"pack not found at {root}")
    print(f"\n################  {depth}bb 6-max  ################")
    pack = build_pack(depth)
    nodes: tuple[PreflopDecisionNode, ...] = ()
    if section in ("tree", "render", "all"):
        nodes = enumerate_nodes([pack])
    if section in ("tree", "all"):
        audit_tree(pack, nodes)
    if section in ("sizes", "all"):
        audit_token_sizes(pack)
    if section in ("render", "all"):
        render_spots(pack, nodes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, choices=[20, 30, 40, 50, 70, 100, 150, 200], default=None)
    ap.add_argument(
        "--section", choices=["tree", "sizes", "render", "all"], default="all"
    )
    args = ap.parse_args()
    depths = [args.depth] if args.depth else [20, 30]
    for d in depths:
        run_depth(d, args.section)


if __name__ == "__main__":
    main()
