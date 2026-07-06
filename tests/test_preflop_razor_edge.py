"""Tests for razor's-edge difficulty (July 2026).

Covers the pure neighbor geometry, boundary detection against a real
enumerated node (built from fixture range files), the graded floors, the
compute_difficulty wiring, the achievable-score ceiling, band
reachability, and the batch counter plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

import pytest  # noqa: E402

from pipeline.preflop.batch import generate_preflop_batch  # noqa: E402
from pipeline.preflop.difficulty import (  # noqa: E402
    classify_band_reachability,
    compute_difficulty,
    max_achievable_difficulty,
)
from pipeline.preflop.fact_extractor import (  # noqa: E402
    PreflopFacts,
    extract_facts,
)
from pipeline.preflop.node_enumerator import enumerate_nodes  # noqa: E402
from pipeline.preflop.pack import (  # noqa: E402
    PreflopPack,
    clear_registry,
    register_pack,
)
from pipeline.preflop.razor_edge import (  # noqa: E402
    RAZOR_FLOOR_BY_COUNT,
    RAZOR_FLOOR_MAX,
    find_opposite_neighbors,
    neighbor_classes,
    razor_floor_for_count,
)
from pipeline.preflop.spot_sampler import sample_spot  # noqa: E402
from pipeline.preflop_ranges import canonical_169_hand_classes  # noqa: E402


# --- fixtures ----------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_registry():
    """The pack registry is module-global; clean before + after every test
    so registrations from one test don't leak into the next."""
    clear_registry()
    yield
    clear_registry()


def _write_full_range(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = ",".join(
        f"{cls}:{weights.get(cls, 0.0)}" for cls in canonical_169_hand_classes()
    )
    path.write_text(line)


def _boundary_pack(tmp_path: Path) -> PreflopPack:
    """A UTG open node crafted with three boundary shapes:

    * ATo pure-folds while AJo (kicker up) AND ATs (twin) pure-raise ->
      2 opposing neighbors. A9o folds alongside ATo (no boundary there).
    * 77 pure-raises while 88 and 66 pure-fold -> 2 opposing neighbors
      (a pair island within its pair line).
    * T8s pure-raises while T9s, T7s, and T8o all pure-fold -> 3 opposing
      neighbors (a full island).
    * AA pure-raises with KK also raising -> 0 opposing neighbors
      (interior hand, no boundary).
    """
    pack_root = tmp_path / "boundary_pack"
    utg = pack_root / "UTG"
    classes = canonical_169_hand_classes()
    raise_weights = {c: 0.0 for c in classes}
    for c in ("AA", "KK", "AJo", "ATs", "77", "T8s"):
        raise_weights[c] = 1.0
    fold_weights = {c: (1.0 - raise_weights[c]) for c in classes}
    _write_full_range(utg / "UTG_60%.txt", raise_weights)
    _write_full_range(utg / "UTG_Fold.txt", fold_weights)
    pack = PreflopPack(
        pack_id="boundary_pack",
        root_path=pack_root,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
        description="razor's-edge fixture pack",
    )
    register_pack(pack)
    return pack


def _utg_node(pack: PreflopPack):
    nodes = enumerate_nodes([pack])
    return next(n for n in nodes if n.actor == "UTG")


# --- neighbor geometry --------------------------------------------------------
def test_neighbor_classes_offsuit_kicker_and_twin() -> None:
    assert set(neighbor_classes("ATo")) == {"AJo", "A9o", "ATs"}


def test_neighbor_classes_pair_adjacent_pairs_only() -> None:
    assert set(neighbor_classes("TT")) == {"JJ", "99"}
    assert set(neighbor_classes("AA")) == {"KK"}  # no rank above the ace
    assert set(neighbor_classes("22")) == {"33"}


def test_neighbor_classes_skips_collisions_with_high_card() -> None:
    # AKo's kicker-up would collide with the A -> only kicker-down + twin.
    assert set(neighbor_classes("AKo")) == {"AQo", "AKs"}
    # A2s has no kicker below the 2.
    assert set(neighbor_classes("A2s")) == {"A3s", "A2o"}


def test_razor_floor_grading_by_count() -> None:
    assert razor_floor_for_count(0) is None
    assert razor_floor_for_count(1) == RAZOR_FLOOR_BY_COUNT[1]
    assert razor_floor_for_count(2) == RAZOR_FLOOR_BY_COUNT[2]
    assert razor_floor_for_count(3) == RAZOR_FLOOR_MAX
    assert razor_floor_for_count(7) == RAZOR_FLOOR_MAX
    # Grading is monotonic: more opposing neighbors never rates easier.
    floors = [razor_floor_for_count(n) for n in (1, 2, 3)]
    assert floors == sorted(floors)


# --- boundary detection on a real node -----------------------------------------
def test_boundary_hand_detected_with_its_opposing_neighbors(tmp_path) -> None:
    pack = _boundary_pack(tmp_path)
    node = _utg_node(pack)
    opp = dict(find_opposite_neighbors(sample_spot(node, "ATo")))
    # ATo folds; AJo and the ATs twin raise (opposite); A9o folds (same).
    assert set(opp) == {"AJo", "ATs"}


def test_island_hand_has_three_opposing_neighbors(tmp_path) -> None:
    pack = _boundary_pack(tmp_path)
    node = _utg_node(pack)
    opp = find_opposite_neighbors(sample_spot(node, "T8s"))
    assert {c for c, _ in opp} == {"T9s", "T7s", "T8o"}


def test_interior_hand_has_no_boundary(tmp_path) -> None:
    pack = _boundary_pack(tmp_path)
    node = _utg_node(pack)
    # AA raises and its only neighbor (KK) raises too -> interior.
    assert find_opposite_neighbors(sample_spot(node, "AA")) == []
    # Deep-fold territory: 72o folds and so do all its neighbors.
    assert find_opposite_neighbors(sample_spot(node, "72o")) == []


# --- compute_difficulty wiring ---------------------------------------------------
def _facts_for(pack: PreflopPack, node, hand_class: str) -> PreflopFacts:
    return extract_facts(sample_spot(node, hand_class), pack, equity_runouts=40)


def test_razor_bump_floors_boundary_hand_and_flags_result(tmp_path) -> None:
    pack = _boundary_pack(tmp_path)
    node = _utg_node(pack)
    facts = _facts_for(pack, node, "ATo")  # pure fold, 2 opposing neighbors
    off = compute_difficulty(facts, ev_gap_bb=2.0, apply_razor_bump=False)
    on = compute_difficulty(facts, ev_gap_bb=2.0, apply_razor_bump=True)
    assert not off.razor_bump_applied
    assert on.razor_bump_applied
    assert on.score == RAZOR_FLOOR_BY_COUNT[2]
    assert off.score < on.score


def test_razor_bump_leaves_interior_hands_alone(tmp_path) -> None:
    pack = _boundary_pack(tmp_path)
    node = _utg_node(pack)
    facts = _facts_for(pack, node, "AA")
    off = compute_difficulty(facts, ev_gap_bb=2.0, apply_razor_bump=False)
    on = compute_difficulty(facts, ev_gap_bb=2.0, apply_razor_bump=True)
    assert not on.razor_bump_applied
    assert on.score == off.score


def test_island_hand_gets_the_max_floor(tmp_path) -> None:
    pack = _boundary_pack(tmp_path)
    node = _utg_node(pack)
    facts = _facts_for(pack, node, "T8s")  # 3 opposing neighbors
    on = compute_difficulty(facts, ev_gap_bb=2.0, apply_razor_bump=True)
    assert on.razor_bump_applied and on.score == RAZOR_FLOOR_MAX


# --- ceiling + band reachability ------------------------------------------------
def test_ceiling_includes_razor_max_when_enabled() -> None:
    assert (
        max_achievable_difficulty(1.0, razor_difficulty=True)
        == RAZOR_FLOOR_MAX
    )
    # And the razor ceiling composes with trap (trap max 2900 is higher).
    assert max_achievable_difficulty(
        1.0, trap_difficulty=True, razor_difficulty=True
    ) >= RAZOR_FLOOR_MAX


def test_band_reachability_razor_only() -> None:
    """1500+ at a 100% window: empty by default, special_only with razor on
    (graded floors 2000-2600 overlap), empty again when the band sits
    entirely above the razor range."""
    assert (
        classify_band_reachability(1500, 2750, 1.0, razor_difficulty=True)
        == "special_only"
    )
    assert (
        classify_band_reachability(2650, 3200, 1.0, razor_difficulty=True)
        == "empty"
    )


# --- batch plumbing ----------------------------------------------------------------
def test_batch_counts_razor_floored_and_records_flag(tmp_path) -> None:
    pack = _boundary_pack(tmp_path)
    out = tmp_path / "out.csv"
    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=50,
        min_frequency=1.0,  # pure-only: every worthy spot is 100%
        max_frequency=1.0,
        min_difficulty=1500,  # reachable ONLY via the razor floors
        razor_difficulty=True,
        dry_run=True,
        random_seed=7,
    )
    assert result.questions_written > 0
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["run_settings"]["razor_difficulty"] is True
    assert meta["counters"]["razor_floored"] == result.questions_written
    # Same batch with razor OFF is structurally empty (the old grind case).
    result_off = generate_preflop_batch(
        pack=pack,
        output_path=tmp_path / "off.csv",
        total_questions=50,
        min_frequency=1.0,
        max_frequency=1.0,
        min_difficulty=1500,
        razor_difficulty=False,
        dry_run=True,
        random_seed=7,
    )
    assert result_off.questions_written == 0
