"""Audit the NLHE 9-max Monker pack against known-GTO landmarks.

Phase-0-style verification for the 9-max integration (June 2026), run
BEFORE the pack is registered for admin generation. Three sections:

1. **Tree sanity** -- node count, root structure, RFI percentages per
   seat (must be tight-to-loose UTG->BTN and visibly tighter than a
   low-rake 6-max pack: this one is raked 10% with a 3bb cap), AA-always-
   continues canary on the high-traffic nodes.
2. **EV-scale calibration** -- the file EVs' unit is undocumented. Anchor
   on spots where the call EV is pure arithmetic (a closing call of an
   all-in: EV = equity x final_pot - cost) and measure the scale factor
   against analytic EV computed with the in-repo equity engine. Also
   check fold files (fold EV must be ~0 -- Monker measures from the
   status quo).
3. **Render slice** -- run the real fact extractor + format writer on a
   handful of spots (BB vs UTG open, SB first-in limp, BB vs SB limp)
   and print the key CSV columns, so 9-seat prose, Seats tokens, pot
   math, archetypes, and skills can be eyeballed.

Usage::

    venv/bin/python scripts/audit_nlhe9_pack.py            # all sections
    venv/bin/python scripts/audit_nlhe9_pack.py --section ev
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.batch import _placeholder_explanation  # noqa: E402
from pipeline.preflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.preflop.equity import preflop_equity_vs_range  # noqa: E402
from pipeline.preflop.ev_engine import compute_ev_gap_bb  # noqa: E402
from pipeline.preflop.fact_extractor import extract_facts  # noqa: E402
from pipeline.preflop.format_writer import build_preflop_row  # noqa: E402
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.options import build_options  # noqa: E402
from pipeline.preflop.pack import PreflopPack  # noqa: E402
from pipeline.preflop.spot_sampler import sample_spot  # noqa: E402
from pipeline.preflop_ranges import parse_monker_rng_file  # noqa: E402

PACK_ROOT = (
    Path(__file__).resolve().parent.parent
    / "nlhe9_ranges/ranges/Hold'em/9-way/100bb[10p-3bb]"
)
SEAT_ORDER = ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB")

# Combo multiplicity per hand-class kind, for range-percentage math.
_COMBOS = {"pair": 6, "suited": 4, "offsuit": 12}


def build_pack() -> PreflopPack:
    # Deliberately NO size_round_bb: the EV-anchor proofs in
    # audit_ev_scale compare the resolver's commitments against the
    # solve's own bookkeeping to the milli-bb, which requires EXACT
    # pot-percentage sizes (the registered pack quantizes to 0.5bb for
    # rendering; that is display-game policy, not solve truth).
    return PreflopPack(
        pack_id="monker_nlhe_9max_100bb",
        root_path=PACK_ROOT,
        grammar_name="monker_nlhe",
        table_size=9,
        stack_depth_bb=100,
        open_size_bb=4.0,
        file_glob="*.rng",
    )


def _combos(hand_class: str) -> int:
    if len(hand_class) == 2:
        return _COMBOS["pair"]
    return _COMBOS["suited"] if hand_class.endswith("s") else _COMBOS["offsuit"]


def _class_range_pct(weights: dict[str, float]) -> float:
    """Percent of all 1326 combos this weighted class range covers."""
    total = sum(w * _combos(h) for h, w in weights.items())
    return 100.0 * total / 1326.0


# --- section 1: tree sanity ---------------------------------------------------
def audit_tree(pack: PreflopPack, nodes: tuple[PreflopDecisionNode, ...]) -> None:
    print("== tree sanity ==")
    print(f"nodes: {len(nodes):,} (expected 44,058)")
    assert len(nodes) == 44058, "node count drifted from the June-11 audit"

    # First-in (RFI) node per seat: history is all folds.
    print("\nfirst-in strategy per seat (% of combos):")
    print(f"  {'seat':6s} {'fold%':>7s} {'limp%':>7s} {'raise%':>7s}  options")
    for n_folds, seat in enumerate(SEAT_ORDER[:-1]):  # BB is never first-in
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
            f"  {seat:6s} {by_kind['fold']:6.1f}% {by_kind['limp']:6.1f}% "
            f"{by_kind['raise']:6.1f}%  {labels}"
        )

    # AA canary on every first-in node + the BB-vs-open defends.
    open_stem = "40120" + ".0" * 7  # UTG opens, folds to BB
    bb_defend = next(
        n
        for n in nodes
        if n.actor == "BB"
        and len(n.history_before) == 8
        and n.actions[0].range_file.path.stem.startswith(open_stem)
    )
    aa_continue = 0.0
    for opt in bb_defend.actions:
        if opt.label == "Fold":
            continue
        data = parse_monker_rng_file(opt.range_file.path)
        aa_continue += data["AA"][0]
    print(f"\nBB vs UTG open: AA continues {aa_continue:.3f} (must be ~1.0)")
    assert aa_continue > 0.98


# --- section 2: EV calibration -------------------------------------------------
def _expand_class_range_to_combos(
    class_weights: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Expand {class: (p, ev)} to {combo: p} for the equity engine."""
    suits = "cdhs"
    out: dict[str, float] = {}
    for hand, (p, _ev) in class_weights.items():
        if p <= 0:
            continue
        if len(hand) == 2:
            combos = [
                f"{hand[0]}{a}{hand[1]}{b}"
                for i, a in enumerate(suits)
                for b in suits[i + 1 :]
            ]
        elif hand.endswith("s"):
            combos = [f"{hand[0]}{s}{hand[1]}{s}" for s in suits]
        else:
            combos = [
                f"{hand[0]}{a}{hand[1]}{b}" for a in suits for b in suits if a != b
            ]
        for c in combos:
            out[c] = p
    return out


def audit_ev_scale(pack: PreflopPack) -> None:
    """Verify the .rng EV unit/baseline on arithmetic-exact anchors.

    CONCLUSION (June 2026, see docs/nlhe9_pack_notes.md): raw EV is in
    **milli-big-blinds measured from the start of the hand** (pre-blind
    posting) -- NOT the PLO pack's milli-small-blinds, and NOT relative
    to the decision point. Proof is the indifference anchor: a hand that
    MIXES call/fold must have equal EV on both branches, and folding
    costs exactly the chips already committed. At the SB-open / BB-3bet
    (9bb) / SB-jam node, hand 66 mixes and reads -9001.9 / -9000.0 --
    exactly -9bb in milli-bb on both. At the root the mixed opens (QQ,
    A2s) read ~0.0 (UTG has nothing committed). Because the baseline is
    per-hand constant, EV DIFFERENCES between actions at a node are
    baseline-free: a future pack-EV gap = (ev_a - ev_b) / 1000, in bb.

    This function re-runs the proof (indifference + fold-file ==
    -commitment) plus an equity-engine cross-check on heads-up jam-call
    closing nodes (secondary, Monte-Carlo-noisy, rake-confounded).
    """
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        resolve_preflop_history,
    )
    from pipeline.preflop.grammars.monker_nlhe import parse  # noqa: E402,PLC0415
    from pipeline.preflop.grammars.types import (  # noqa: E402,PLC0415
        PreflopActionType,
    )

    print("\n== EV-scale verification (unit: milli-bb from hand start) ==")
    # Find heads-up jam-then-call closing nodes, shallowest first.
    candidates: list[tuple[int, Path]] = []
    for f in pack.root_path.glob("*.3.1.rng"):
        parsed = parse(f, pack)
        last_action = {}
        for a in parsed.action_history:
            last_action[a.position] = a.action_type
        active = [
            p for p, t in last_action.items() if t is not PreflopActionType.FOLD
        ]
        if len(active) == 2:  # jammer + caller only -> the call closes it
            candidates.append((len(parsed.action_history), f))
    candidates.sort(key=lambda kv: (kv[0], kv[1].name))
    if not candidates:
        print("no heads-up jam-call anchors found")
        return
    print(f"heads-up jam-call anchors available: {len(candidates)}")

    rng = random.Random(7)
    failures = 0
    for _depth, call_file in candidates[:3]:
        parsed = parse(call_file, pack)
        history = parsed.action_history[:-1]  # up to hero's decision
        caller = parsed.actor
        state = resolve_preflop_history(history, pack)
        cost = state.call_cost_bb(caller)
        final_pot = state.pot_bb + cost
        committed = state.committed_bb.get(caller, 0.0)
        jam_stem = ".".join(call_file.stem.split(".")[:-1])
        jam = parse_monker_rng_file(pack.root_path / f"{jam_stem}.rng")
        call = parse_monker_rng_file(call_file)
        fold = parse_monker_rng_file(pack.root_path / f"{jam_stem}.0.rng")

        print(f"\n  anchor {call_file.stem}  ({caller} committed "
              f"{committed:g}bb, pot {state.pot_bb:g}, cost {cost:g})")
        # 1. PRIMARY: fold EV == -commitment (milli-bb, from hand start).
        fold_evs = [ev for _h, (p, ev) in fold.items() if p > 0.05]
        expect = -committed * 1000.0
        off = max(abs(ev - expect) for ev in fold_evs)
        ok = off < 50.0  # 0.05bb tolerance
        failures += not ok
        print(f"    fold-file EVs == {expect:.0f} +/- {off:.1f} milli-bb "
              f"{'OK' if ok else 'FAIL'}")
        # 2. PRIMARY: mixed hands indifferent across the two branches.
        mixed = [
            h for h, (p, _e) in call.items()
            if 0.03 < p < 0.97 and 0.03 < fold[h][0] < 0.97
        ]
        for h in mixed[:2]:
            gap = abs(call[h][1] - fold[h][1])
            ok = gap < 50.0
            failures += not ok
            print(f"    mixed {h}: call {call[h][1]:.1f} vs fold "
                  f"{fold[h][1]:.1f} (gap {gap:.1f}) {'OK' if ok else 'FAIL'}")
        # 3. SECONDARY: equity-engine cross-check (MC-noisy, rake-confounded;
        #    file EV should sit a few bb BELOW the no-rake analytic value).
        jam_combos = _expand_class_range_to_combos(jam)
        for hand, combo in (("AA", ("Ah", "Ad")),):
            if call[hand][0] <= 0:
                continue
            eq = preflop_equity_vs_range(combo, jam_combos, n_samples=300, rng=rng)
            analytic_bb = eq * final_pot - committed - cost
            file_bb = call[hand][1] / 1000.0
            print(f"    {hand} cross-check: analytic {analytic_bb:.1f}bb "
                  f"(no rake) vs file {file_bb:.1f}bb")
    print(f"\n  {'ALL ANCHORS OK' if failures == 0 else f'{failures} FAILURES'}"
          " -- unit: milli-bb, baseline: hand start; gaps between actions"
          " are baseline-free ((ev_a - ev_b)/1000 = bb).")


# --- section 3: render slice ---------------------------------------------------
def render_spots(pack: PreflopPack, nodes: tuple[PreflopDecisionNode, ...]) -> None:
    print("\n== render slice (real fact extractor + format writer) ==")
    open_stem = "40120" + ".0" * 7

    def first(pred):
        return next(n for n in nodes if pred(n))

    picks = [
        (
            "BB defends vs UTG open (AJs)",
            first(
                lambda n: n.actor == "BB"
                and len(n.history_before) == 8
                and n.actions[0].range_file.path.stem.startswith(open_stem)
            ),
            "AJs",
        ),
        (
            "SB first-in (sb_complete candidate, 86s)",
            first(
                lambda n: n.actor == "SB"
                and len(n.history_before) == 7
                and all(a.action_type.value == "Fold" for a in n.history_before)
            ),
            "86s",
        ),
        (
            "BB vs SB limp (bb_check candidate, T5o)",
            first(
                lambda n: n.actor == "BB"
                and len(n.history_before) == 8
                and all(
                    a.action_type.value == "Fold" for a in n.history_before[:7]
                )
                and n.history_before[-1].action_type.value == "Call"
            ),
            "T5o",
        ),
        (
            "UTG+1 3-bets vs UTG open (QQ)",
            first(
                lambda n: n.actor == "UTG+1"
                and len(n.history_before) == 1
                and n.history_before[0].action_type.value == "Raise"
            ),
            "QQ",
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
        print(f"freqs: { {k: round(v, 3) for k, v in spot.action_frequencies.items()} }")
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
            "solver_reference",
            "Difficulty Rating",
            "ev_gap_bb",
        ):
            print(f"{col}: {row.get(col, '<missing>')!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--section",
        choices=["tree", "ev", "render", "all"],
        default="all",
    )
    args = ap.parse_args()
    if not PACK_ROOT.is_dir():
        sys.exit(f"pack not found at {PACK_ROOT}")
    pack = build_pack()
    nodes: tuple[PreflopDecisionNode, ...] = ()
    if args.section in ("tree", "render", "all"):
        nodes = enumerate_nodes([pack])
    if args.section in ("tree", "all"):
        audit_tree(pack, nodes)
    if args.section in ("ev", "all"):
        audit_ev_scale(pack)
    if args.section in ("render", "all"):
        render_spots(pack, nodes)


if __name__ == "__main__":
    main()
