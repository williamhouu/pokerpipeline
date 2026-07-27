"""🪤 Trap-aware difficulty for PLO (pipeline/plo/difficulty.plo_trap_margin).

The NLHE port RE-CALIBRATED for PLO's equity compression (ordinary folds sit
~9 points above the naive price -- see the calibration note in difficulty.py).
Covers: the fold-trap (premium/strong shape + equity far above the cushioned
price), the shape gate (a medium hand never fires), the continue-trap
(all-in price only), the heads-up / closes-action / no-equity gates, the
graded floor composition, and the batch threading (flag + counter +
run_settings mirror for the re-verifier).
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.batch import generate_plo_batch  # noqa: E402
from pipeline.plo.difficulty import (  # noqa: E402
    compute_plo_difficulty,
    plo_trap_margin,
)
from pipeline.plo.fact_extractor import extract_plo_facts  # noqa: E402
from pipeline.plo.hand_order import HAND_COUNT  # noqa: E402
from pipeline.plo.node_enumerator import enumerate_plo_nodes  # noqa: E402
from pipeline.plo.pack import PloPack, rake_pct_from_note  # noqa: E402
from pipeline.plo.spot_sampler import sample_plo_spot  # noqa: E402
from pipeline.trap_grading import graded_trap_floor  # noqa: E402


def _write_rng(path: Path, p: float) -> None:
    out: list[str] = []
    for _ in range(HAND_COUNT):
        out.append("????")
        out.append(f"{p};1000.0")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _bb_vs_sb_fold_pack(tmp_path: Path) -> PloPack:
    """Folds to the SB, SB pot-raises, BB decides: fold 80 / call 10 / 3b 10.

    Heads-up (SB + BB only), hero's call/fold closes the action -- the exact
    shape the trap detector's gates require.
    """
    root = tmp_path / "fold_pack"
    root.mkdir()
    stem = "0.0.0.0.40100"
    _write_rng(root / f"{stem}.rng", 0.5)  # the SB open range (villain)
    _write_rng(root / f"{stem}.0.rng", 0.8)
    _write_rng(root / f"{stem}.1.rng", 0.1)
    _write_rng(root / f"{stem}.40100.rng", 0.1)
    return PloPack(root=root, label="fold-trap-test")


def _bb_vs_sb_jam_pack(tmp_path: Path) -> PloPack:
    """Folds to the SB, SB open-JAMS (token 3), BB: fold 20 / call 80."""
    root = tmp_path / "jam_pack"
    root.mkdir()
    stem = "0.0.0.0.3"
    _write_rng(root / f"{stem}.rng", 0.5)  # the SB jam range (villain)
    _write_rng(root / f"{stem}.0.rng", 0.2)
    _write_rng(root / f"{stem}.1.rng", 0.8)
    return PloPack(root=root, label="jam-trap-test")


def _bu_vs_lj_pack(tmp_path: Path) -> PloPack:
    """LJ opens, folds to the BU: NOT heads-up (SB + BB still live behind)."""
    root = tmp_path / "multi_pack"
    root.mkdir()
    stem = "40100.0.0"
    _write_rng(root / "40100.rng", 0.5)  # the LJ open range (villain)
    _write_rng(root / f"{stem}.0.rng", 0.8)
    _write_rng(root / f"{stem}.1.rng", 0.1)
    _write_rng(root / f"{stem}.40100.rng", 0.1)
    return PloPack(root=root, label="multiway-test")


def _bb_node(pack: PloPack):
    nodes = [n for n in enumerate_plo_nodes(pack) if n.actor == "BB"]
    assert nodes, "expected a BB decision node"
    return nodes[0]


def _facts_by_strength(pack, node, wanted: str, *, equity: bool):
    """First hand whose hand-model strength bucket == ``wanted``."""
    for idx in range(0, HAND_COUNT, 37):
        spot = sample_plo_spot(node, idx)
        facts = extract_plo_facts(
            spot, pack, compute_equity=equity,
            equity_runouts=120, rng=random.Random(7),
        )
        if facts.hand_class.strength == wanted:
            return facts
    raise AssertionError(f"no {wanted} hand found")


def test_fold_trap_fires_only_with_shape_and_wide_margin(tmp_path) -> None:
    pack = _bb_vs_sb_fold_pack(tmp_path)
    node = _bb_node(pack)
    premium = _facts_by_strength(pack, node, "premium", equity=True)
    margin = plo_trap_margin(premium, stack_bb=100.0, rake_pct=0.05)
    assert margin is not None and margin > 0, (
        "a premium hand folding with equity far above the cushioned price "
        f"must fire (equity {premium.hero_equity_vs_villain})"
    )
    # The SHAPE GATE: identical facts relabelled medium must never fire --
    # this is what keeps PLO's equity compression from flagging normal folds.
    medium_class = _facts_by_strength(pack, node, "medium", equity=False).hand_class
    relabelled = dataclasses.replace(premium, hand_class=medium_class)
    assert plo_trap_margin(relabelled, stack_bb=100.0, rake_pct=0.05) is None


def test_no_equity_means_no_trap(tmp_path) -> None:
    pack = _bb_vs_sb_fold_pack(tmp_path)
    node = _bb_node(pack)
    premium = _facts_by_strength(pack, node, "premium", equity=False)
    assert premium.hero_equity_vs_villain is None
    assert plo_trap_margin(premium, stack_bb=100.0, rake_pct=0.05) is None


def test_multiway_spot_never_fires(tmp_path) -> None:
    pack = _bu_vs_lj_pack(tmp_path)
    nodes = [n for n in enumerate_plo_nodes(pack) if n.actor == "BU"]
    assert nodes
    for idx in range(0, HAND_COUNT, 211):
        spot = sample_plo_spot(nodes[0], idx)
        facts = extract_plo_facts(
            spot, pack, compute_equity=True,
            equity_runouts=60, rng=random.Random(7),
        )
        assert plo_trap_margin(facts, stack_bb=100.0, rake_pct=0.05) is None


def test_continue_trap_requires_allin_price(tmp_path) -> None:
    jam = _bb_vs_sb_jam_pack(tmp_path)
    node = _bb_node(jam)
    # Find a hand whose dominant action is CALL and equity sits clearly below
    # the ~0.49 jam price -- weak/trash hands vs a uniform range qualify.
    fired = None
    for idx in range(0, HAND_COUNT, 61):
        spot = sample_plo_spot(node, idx)
        facts = extract_plo_facts(
            spot, jam, compute_equity=True,
            equity_runouts=120, rng=random.Random(7),
        )
        if facts.spot.dominant_action.startswith("Fold"):
            continue
        m = plo_trap_margin(facts, stack_bb=100.0, rake_pct=0.05)
        if m is not None:
            fired = (facts, m)
            break
    assert fired is not None, "a clearly-below-price jam call must fire"
    # The same facts against a NON-all-in raise (the fold pack's node) must
    # not fire on the continue side: implied odds could justify the call.
    fold_pack = _bb_vs_sb_fold_pack(tmp_path)
    fnode = _bb_node(fold_pack)
    for idx in range(0, HAND_COUNT, 61):
        spot = sample_plo_spot(fnode, idx)
        facts = extract_plo_facts(
            spot, fold_pack, compute_equity=True,
            equity_runouts=60, rng=random.Random(7),
        )
        if facts.spot.dominant_action.startswith("Fold"):
            continue
        assert plo_trap_margin(facts, stack_bb=100.0, rake_pct=0.05) is None
        break


def test_graded_floor_composition_and_flag_off_identity(tmp_path) -> None:
    pack = _bb_vs_sb_fold_pack(tmp_path)
    node = _bb_node(pack)
    premium = _facts_by_strength(pack, node, "premium", equity=True)
    margin = plo_trap_margin(premium, stack_bb=100.0, rake_pct=0.05)
    assert margin is not None
    off = compute_plo_difficulty(premium)
    on = compute_plo_difficulty(
        premium, apply_trap_bump=True, stack_bb=100.0, rake_pct=0.05
    )
    assert not off.trap_bump_applied
    assert on.trap_bump_applied
    assert on.score == max(off.score, graded_trap_floor(margin))
    assert on.score >= 1800  # the graded floor's minimum
    # Default call (no flag) is byte-identical to the pre-port behaviour.
    assert compute_plo_difficulty(premium).score == off.score


def test_batch_threads_flag_counter_and_run_settings(tmp_path) -> None:
    # A pack where ONLY premium-shaped hands are worthy (fold 80% for them,
    # a 50/50 call/3-bet mix for everything else -> below the worthiness
    # floor), so every drawn spot is a premium fold and the trap counter
    # moves deterministically.
    from pipeline.plo.hand_model import classify_plo_hand  # noqa: PLC0415
    from pipeline.plo.hand_order import cards_at  # noqa: PLC0415

    premium = [
        classify_plo_hand(cards_at(i)).strength == "premium"
        for i in range(HAND_COUNT)
    ]
    assert any(premium)
    root = tmp_path / "premium_fold_pack"
    root.mkdir()
    stem = "0.0.0.0.40100"

    def _write(path: Path, p_premium: float, p_other: float) -> None:
        out: list[str] = []
        for i in range(HAND_COUNT):
            out.append("????")
            out.append(f"{p_premium if premium[i] else p_other};1000.0")
        path.write_text("\n".join(out) + "\n", encoding="utf-8")

    _write(root / f"{stem}.rng", 0.5, 0.5)  # the SB open range (villain)
    _write(root / f"{stem}.0.rng", 0.8, 0.0)
    _write(root / f"{stem}.1.rng", 0.1, 0.1)
    _write(root / f"{stem}.40100.rng", 0.1, 0.1)
    pack = PloPack(root=root, label="premium-fold-trap-test")

    out = tmp_path / "trap_batch.csv"
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=4,
        seed=0,
        trap_difficulty=True,
        compute_equity=True,
        max_prior_raises=None,
        max_active_players=None,
    )
    assert result.questions_written > 0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["run_settings"]["trap_difficulty"] is True
    assert meta["counters"]["trap_floored"] >= 1, (
        "a premium fold in this pack must be re-rated by the trap floor"
    )
    # Trap is score-only: every floored question still shipped normally.
    assert meta["counters"]["questions_written"] == result.questions_written


def test_rake_pct_from_note() -> None:
    assert rake_pct_from_note("5% up to 1bb") == 0.05
    assert rake_pct_from_note("5% up to 2bb") == 0.05
    assert rake_pct_from_note("no rake (MTT)") == 0.0
    assert rake_pct_from_note("") == 0.0
