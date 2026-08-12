"""Audit the NLH MTT 8-max BB-ante Monker packs (10/15/20/30/50/75/300bb).

Phase-0-style intake verification (Aug 2026), run BEFORE production use.
These packs share the ``monker_nlhe`` grammar with the 6-max short-stack
packs but add three things the audit must lock independently:

  * **The 1bb BB ante** -- IN the pot for Monker's pot-relative sizing
    (the 75/300bb ``40043`` open anchors at 2.505bb = the formula WITH the
    ante; 2.075 without), but SUNK in the EV baseline (BB folding to an
    open reads -1bb: the blind alone, never blind+ante).
  * **Fixed-size tokens** ``14``/``15``/``16``/``18`` (absolute raise-TO
    bb, registered per pack in ``fixed_raise_tokens_bb``) -- each re-derived
    here from the solve's own fold-EV bookkeeping (a raiser who later folds
    is out exactly the raise size).
  * **Completeness** -- the prior MTT preflop delivery (the CEV export) was
    REJECTED for missing every all-in file; this one is scanned so that
    failure mode can't ship silently.

Usage::

    venv/bin/python scripts/audit_mtt8_pack.py                 # all depths
    venv/bin/python scripts/audit_mtt8_pack.py --depth 15
    venv/bin/python scripts/audit_mtt8_pack.py --depth 300 --section sizes
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.action_history import resolve_preflop_history  # noqa: E402
from pipeline.preflop.batch import _placeholder_explanation  # noqa: E402
from pipeline.preflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.preflop.ev_engine import compute_ev_gap_bb  # noqa: E402
from pipeline.preflop.fact_extractor import extract_facts  # noqa: E402
from pipeline.preflop.format_writer import build_preflop_row  # noqa: E402
from pipeline.preflop.grammars.monker_nlhe import parse  # noqa: E402
from pipeline.preflop.grammars.types import PreflopActionType  # noqa: E402
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.options import build_options  # noqa: E402
from pipeline.preflop.pack import (  # noqa: E402
    KNOWN_PACK_SIGNATURES,
    PreflopPack,
    clear_registry,
    discover_packs,
    get_pack,
)
from pipeline.preflop.spot_sampler import sample_spot  # noqa: E402
from pipeline.preflop_ranges import parse_monker_rng_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPTHS = (10, 15, 20, 30, 40, 50, 75, 100, 200, 300)
SEATS_8MAX = ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB")

# Deterministic landmarks from the intake walk (Aug 3 2026). A drifted count
# means the extraction or the grammar changed -- investigate, don't update
# blindly.
EXPECTED_NODES: dict[int, int] = {
    10: 692,
    15: 3612,
    20: 3534,
    30: 8046,
    40: 11204,
    50: 12339,
    75: 10291,
    100: 42943,
    200: 89434,
    300: 89202,
}

# EV unit: milli-SMALL-blinds (2000 units = 1bb), proven by the SB
# open-fold anchor reading -1000 in every depth.
EV_UNITS_PER_BB = 2000.0

_COMBOS = {"pair": 6, "suited": 4, "offsuit": 12}


def _combos(hand_class: str) -> int:
    if len(hand_class) == 2:
        return _COMBOS["pair"]
    return _COMBOS["suited"] if hand_class.endswith("s") else _COMBOS["offsuit"]


def _class_range_pct(weights: dict[str, float]) -> float:
    total = sum(w * _combos(h) for h, w in weights.items())
    return 100.0 * total / 1326.0


def mtt_pack(depth: int) -> PreflopPack:
    clear_registry()
    discover_packs(REPO_ROOT / "ranges")
    return get_pack(f"monker_mtt8_{depth}bb")


def _fold_commit_sb(path: Path) -> float | None:
    """Median ``-EV/1000`` (in small blinds) over hands folded with weight
    > 0.05: a fold's EV is minus the chips already committed."""
    if not path.exists():
        return None
    data = parse_monker_rng_file(path)
    commits = sorted(-ev / 1000.0 for _h, (p, ev) in data.items() if p > 0.05)
    return commits[len(commits) // 2] if commits else None


def _walk_seats(stem: str) -> list[tuple[str, str]]:
    """(seat, token) pairs for a stem under the 8-max rotation."""
    queue = list(SEATS_8MAX)
    acts: list[tuple[str, str]] = []
    for t in stem.split("."):
        seat = queue.pop(0)
        acts.append((seat, t))
        if t not in ("0", "3"):
            queue.append(seat)
    return acts


def _raiser_fold_anchor(root: Path, token: str) -> Path | None:
    """Shallowest file where the seat that raised ``token`` later FOLDS
    (facing any re-raise). Its fold EVs read minus the raise-to size."""
    best: tuple[int, Path] | None = None
    for f in root.iterdir():
        if not f.name.endswith(".rng"):
            continue
        toks = f.stem.split(".")
        if toks[-1] != "0" or token not in toks or len(toks) > 12:
            continue
        acts = _walk_seats(f.stem)
        raiser = next((s for s, t in acts if t == token), None)
        if acts[-1][0] != raiser or acts[-1][1] != "0":
            continue
        i = [t for _s, t in acts].index(token)
        if any(t not in ("0", "1") for t in [t for _s, t in acts][i + 1 : -1]):
            if best is None or len(toks) < best[0]:
                best = (len(toks), f)
    return best[1] if best else None


# --- section 1: tree sanity ---------------------------------------------------
def audit_tree(pack: PreflopPack, nodes: tuple[PreflopDecisionNode, ...]) -> None:
    depth = pack.stack_depth_bb
    print("== tree sanity ==")
    print(f"nodes: {len(nodes):,}")
    expected = EXPECTED_NODES.get(depth)
    if expected is not None:
        assert len(nodes) == expected, (
            f"node count drifted: {len(nodes)} != recorded {expected}"
        )

    print("\nfirst-in strategy per seat (% of combos):")
    print(f"  {'seat':6s} {'fold%':>7s} {'limp%':>7s} {'raise%':>7s} {'jam%':>7s}  options")
    for n_folds, seat in enumerate(SEATS_8MAX[:-1]):  # BB never first-in
        node = next(
            (
                n
                for n in nodes
                if n.actor == seat
                and len(n.history_before) == n_folds
                and all(a.action_type.value == "Fold" for a in n.history_before)
            ),
            None,
        )
        if node is None:
            print(f"  {seat:6s} <no first-in node>")
            continue
        by_kind = {"fold": 0.0, "limp": 0.0, "raise": 0.0, "jam": 0.0}
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
            elif "AllIn" in opt.label or opt.label == "All-in":
                by_kind["jam"] += pct
            else:
                by_kind["raise"] += pct
        labels = [o.label for o in node.actions]
        print(
            f"  {seat:6s} {by_kind['fold']:6.1f}% {by_kind['limp']:6.1f}% "
            f"{by_kind['raise']:6.1f}% {by_kind['jam']:6.1f}%  {labels}"
        )

    # AA canary: BB facing a CLEAN single open (six folds + one raise/jam,
    # nobody else in) never folds. Multiway jam-called lines are excluded --
    # they are unconverged tree tail (six players flatting a jam), the same
    # class the PLO 9-max clean-line caps handle.
    bb_defend = next(
        (
            n
            for n in nodes
            if n.actor == "BB"
            and len(n.history_before) == 7
            and sum(
                1
                for a in n.history_before
                if a.action_type
                in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
            )
            == 1
            and sum(
                1
                for a in n.history_before
                if a.action_type is PreflopActionType.FOLD
            )
            == 6
        ),
        None,
    )
    if bb_defend is not None:
        aa_continue = sum(
            parse_monker_rng_file(opt.range_file.path)["AA"][0]
            for opt in bb_defend.actions
            if opt.label != "Fold"
        )
        print(f"\nBB vs open: AA continues {aa_continue:.3f} (must be ~1.0)")
        assert aa_continue > 0.98


# --- section 2: token-size + ante locks --------------------------------------
def audit_token_sizes(pack: PreflopPack) -> None:
    print("\n== token-size + ante locks (EV-derived, independent of the walk) ==")
    root = pack.root_path
    depth = pack.stack_depth_bb
    failures = 0

    def check(label: str, got: float | None, want: float, tol: float = 0.05) -> None:
        nonlocal failures
        ok = got is not None and abs(got - want) < tol
        failures += not ok
        print(f"  {label}: {got!r} (expect {want}) {'OK' if ok else 'FAIL'}")

    # 1. Unit proof: SB folds to an open having posted the 0.5bb blind.
    open_tok = {10: "3", 15: "5", 20: "5", 30: "5", 40: "40034", 50: "40034", 75: "40043", 100: "40034", 200: "40043", 300: "40043"}[depth]
    check(
        "SB-blind fold (unit = milli-sb; 1.0 sb = 0.5bb)",
        _fold_commit_sb(root / f"{open_tok}.0.0.0.0.0.0.rng"),
        1.0,
    )

    # 2. ANTE-SUNK proof: BB folding to the same open is out the 1bb blind
    #    ALONE (2.0 sb) -- the ante is sunk before the hand's EV baseline.
    check(
        "BB fold vs open (blind only -> ante SUNK in EV baseline)",
        _fold_commit_sb(root / f"{open_tok}.0.0.0.0.0.0.0.rng"),
        2.0,
    )

    # 3. Every registered fixed token anchors at its registered size.
    for token, bb in pack.fixed_raise_tokens_bb or ():
        anchor = _raiser_fold_anchor(root, token)
        got = _fold_commit_sb(anchor) if anchor else None
        check(
            f"fixed token `{token}` via {anchor.name if anchor else '<none>'}",
            got,
            bb * 2.0,  # sb units
        )

    # 4. ANTE-IN-POT proof: the pot-relative open resolves (through the
    #    pipeline walk, ante included) to the SAME size the EV anchor
    #    reads. Only meaningful for pot-% opens (50/75/300bb).
    if open_tok.startswith("40"):
        anchor = _raiser_fold_anchor(root, open_tok)
        anchored_bb = (_fold_commit_sb(anchor) or 0.0) / 2.0 if anchor else None
        open_file = root / f"{open_tok}.rng"
        parsed = parse(open_file, pack)
        state = resolve_preflop_history(parsed.action_history, pack)
        resolved = state.sizes_bb[-1]
        print(
            f"  `{open_tok}` open: EV-anchored {anchored_bb!r} bb, walk resolves "
            f"{resolved!r} bb (0.5bb display grid)"
        )
        assert anchored_bb is not None
        # The walk quantizes to the 0.5bb grid; the anchor is exact.
        assert resolved is not None and abs(resolved - round(anchored_bb * 2) / 2) < 0.26, (
            "ante-in-pot resolution drifted from the EV anchor"
        )

    # 5. min-raise open at 15-30bb = 2bb.
    if open_tok == "5":
        anchor = _raiser_fold_anchor(root, "5")
        if anchor is not None:
            check("`5` min-raise open", _fold_commit_sb(anchor), 4.0)
        else:
            print("  `5` open: no opener-later-folds anchor (openers never fold here) -- OK")

    print(f"  {'ALL LOCKS HOLD' if failures == 0 else f'{failures} FAILURES'}")
    assert failures == 0, "token-size/ante lock failed"


# --- section 3: completeness (the CEV-rejection failure mode) ----------------
def audit_completeness(pack: PreflopPack) -> None:
    print("\n== completeness (grammar accepts every file; menus are sane) ==")
    root = pack.root_path
    parse_fail = 0
    token_census: Counter[str] = Counter()
    menus: dict[str, set[str]] = {}
    n_files = 0
    for f in root.iterdir():
        if not f.name.endswith(".rng"):
            continue
        n_files += 1
        try:
            parse(f, pack)
        except ValueError as exc:
            parse_fail += 1
            if parse_fail <= 3:
                print(f"  PARSE FAIL: {exc}")
            continue
        toks = f.stem.split(".")
        for t in toks:
            token_census[t if len(t) < 4 else "40xxx"] += 1
        prefix = ".".join(toks[:-1])
        menus.setdefault(prefix, set()).add(toks[-1])
    print(f"  files: {n_files:,} | parse failures: {parse_fail}")
    assert parse_fail == 0, "grammar rejected files -- token decode incomplete"
    print(f"  token census: {dict(sorted(token_census.items()))}")

    # Every facing-a-raise menu must offer a Fold file (the CEV export shipped
    # nodes with missing branches). A menu = all siblings sharing a prefix.
    missing_fold = 0
    for prefix, tokens in menus.items():
        if prefix == "":
            continue
        last_tok = prefix.split(".")[-1]
        facing_raise = last_tok not in ("0", "1")
        if facing_raise and "0" not in tokens and "1" not in tokens:
            missing_fold += 1
            if missing_fold <= 3:
                print(f"  MENU MISSING FOLD/CALL: {prefix}.* -> {sorted(tokens)}")
    print(f"  decision menus: {len(menus):,} | facing-raise menus missing fold AND call: {missing_fold}")
    assert missing_fold == 0, "nodes with missing response branches"


# --- section 4: premium-inversion scan (the Monker QQ-fold bug) --------------
def audit_inversions(pack: PreflopPack, nodes: tuple[PreflopDecisionNode, ...]) -> None:
    print("\n== premium-inversion scan (QQ-fold-bug class) ==")
    hits = 0
    checked = 0
    for n in nodes:
        # facing exactly one raise, hero not a blind-vs-blind special
        n_raises = sum(
            1
            for a in n.history_before
            if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
        )
        if n_raises != 1:
            continue
        # CLEAN lines only (<=1 caller besides the raiser): the multiway
        # jam-called lines are unconverged tree tail -- five players flatting
        # a jam is initialization garbage, not solve output, and generation's
        # premise-realism gates exclude those lines anyway. The 9-max QQ-fold
        # bug this scan hunts lived on CLEAN facing-one-open nodes.
        n_callers = sum(
            1
            for a in n.history_before
            if a.action_type is PreflopActionType.CALL
        )
        if n_callers > 1:
            continue
        fold_opt = next((o for o in n.actions if o.label == "Fold"), None)
        if fold_opt is None:
            continue
        weights = {
            h: p
            for h, (p, _e) in parse_monker_rng_file(fold_opt.range_file.path).items()
        }
        checked += 1
        qq, jj, tt = (weights.get(h, 0.0) for h in ("QQ", "JJ", "TT"))
        # The 9-max bug shape: QQ folding overwhelmingly while JJ/TT continue.
        if qq > 0.9 and (jj < 0.5 or tt < 0.5):
            hits += 1
            if hits <= 5:
                print(f"  INVERSION: {n.node_id} QQ folds {qq:.2f} vs JJ {jj:.2f} / TT {tt:.2f}")
    print(f"  facing-one-raise nodes checked: {checked:,} | inversion hits: {hits}")
    assert hits == 0, "premium fold inversions found -- inspect before production"


# --- section 5: render slice --------------------------------------------------
def render_spots(pack: PreflopPack, nodes: tuple[PreflopDecisionNode, ...]) -> None:
    print("\n== render slice (real fact extractor + format writer) ==")
    picks: list[tuple[str, PreflopDecisionNode, str]] = []
    open_node = next(
        (
            n
            for n in nodes
            if n.actor in ("CO", "BTN")
            and all(a.action_type.value == "Fold" for a in n.history_before)
        ),
        None,
    )
    if open_node is not None:
        picks.append(("first-in open (A5s)", open_node, "A5s"))
    defend = next(
        (
            n
            for n in nodes
            if n.actor == "BB"
            and len(n.history_before) == 7
            and sum(
                1
                for a in n.history_before
                if a.action_type
                in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
            )
            == 1
            and sum(
                1
                for a in n.history_before
                if a.action_type is PreflopActionType.FOLD
            )
            == 6
        ),
        None,
    )
    if defend is not None:
        picks.append(("BB defend vs open/jam (KQs)", defend, "KQs"))
    limp_node = next(
        (
            n
            for n in nodes
            if n.actor == "BTN"
            and len(n.history_before) == 5
            and all(a.action_type.value == "Fold" for a in n.history_before)
        ),
        None,
    )
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
            game_format="tournament",
            display_in_bb=True,
            live_or_online="",
        )
        print(f"\n--- {title} ---")
        print(f"node: {node.node_id}")
        print(f"freqs: { {k: round(v, 3) for k, v in spot.action_frequencies.items()} }")
        for col in (
            "Question", "Context", "Cash/Tourney", "User Seat", "Seats",
            "POT", "Default Stack", "Correct Answer", "archetype",
            "Stack Depth", "Difficulty Rating", "skills",
        ):
            print(f"{col}: {row.get(col, '<missing>')!r}")


def run_depth(depth: int, section: str) -> None:
    print(f"\n################  {depth}bb MTT 8-max (BB ante)  ################")
    pack = mtt_pack(depth)
    nodes: tuple[PreflopDecisionNode, ...] = ()
    if section in ("tree", "inversions", "render", "all"):
        nodes = enumerate_nodes([pack])
    if section in ("tree", "all"):
        audit_tree(pack, nodes)
    if section in ("sizes", "all"):
        audit_token_sizes(pack)
    if section in ("complete", "all"):
        audit_completeness(pack)
    if section in ("inversions", "all"):
        audit_inversions(pack, nodes)
    if section in ("render", "all"):
        render_spots(pack, nodes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, choices=list(DEPTHS), default=None)
    ap.add_argument(
        "--section",
        choices=["tree", "sizes", "complete", "inversions", "render", "all"],
        default="all",
    )
    args = ap.parse_args()
    for d in [args.depth] if args.depth else DEPTHS:
        run_depth(d, args.section)


if __name__ == "__main__":
    main()
