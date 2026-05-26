"""Tests for pipeline.preflop.node_enumerator.

Most tests use synthetic packs (tmp_path fixtures with hand-written
filenames) so they're fast and hermetic. One integration test exercises
the real ryan_preflop_tree pack if it's present locally.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.grammars.types import PreflopActionType              # noqa: E402
from pipeline.preflop.node_enumerator import (                              # noqa: E402
    PreflopActionOption,
    PreflopDecisionNode,
    enumerate_nodes,
    enumerate_nodes_by_actor,
)
from pipeline.preflop.pack import (                                          # noqa: E402
    PreflopPack,
    clear_registry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_registry_between_tests():
    clear_registry()
    yield
    clear_registry()


# --- helpers ---------------------------------------------------------------
def _make_pack(root: Path, pack_id: str = "test_pack") -> PreflopPack:
    """Build a PreflopPack pointing at a synthetic on-disk layout."""
    return PreflopPack(
        pack_id=pack_id,
        root_path=root,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )


def _write(path: Path, content: str = "AA:1.0\n") -> None:
    """Create a file with minimal content (enumerator doesn't read content)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# --- empty / no-op cases ---------------------------------------------------
def test_enumerate_empty_pack_list_returns_empty():
    assert enumerate_nodes([]) == ()


def test_enumerate_empty_pack_dir_returns_empty(tmp_path):
    pack = _make_pack(tmp_path)
    assert enumerate_nodes([pack]) == ()


# --- single-node grouping --------------------------------------------------
def test_two_files_for_same_actor_become_one_node(tmp_path):
    """Two files where BB faces the same history -> one node with 2 actions."""
    bb_dir = tmp_path / "BB"
    _write(bb_dir / "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Call.txt")
    _write(bb_dir / "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Fold.txt")
    nodes = enumerate_nodes([_make_pack(tmp_path)])
    assert len(nodes) == 1
    node = nodes[0]
    assert node.actor == "BB"
    assert len(node.actions) == 2
    types = {opt.action_type for opt in node.actions}
    assert types == {PreflopActionType.CALL, PreflopActionType.FOLD}
    # history_before is the 5 actions PRECEDING BB's choice (BB's own action
    # is in `actions`, not history).
    assert len(node.history_before) == 5
    assert node.history_before[-1].position == "SB"


def test_three_files_for_btn_become_one_three_action_node(tmp_path):
    """BTN facing UTG open: can fold, call, or 3-bet -> one node, 3 actions."""
    btn_dir = tmp_path / "BTN"
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Call.txt")
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_77%.txt")
    nodes = enumerate_nodes([_make_pack(tmp_path)])
    assert len(nodes) == 1
    node = nodes[0]
    assert node.actor == "BTN"
    assert len(node.actions) == 3
    types = {opt.action_type for opt in node.actions}
    assert types == {
        PreflopActionType.FOLD,
        PreflopActionType.CALL,
        PreflopActionType.RAISE,
    }
    raise_opt = next(o for o in node.actions if o.action_type is PreflopActionType.RAISE)
    assert raise_opt.raise_size_pct == 77.0


def test_different_actors_become_different_nodes(tmp_path):
    """Two files for two different actors -> two nodes."""
    _write(tmp_path / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    _write(tmp_path / "BB" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_Fold.txt")
    nodes = enumerate_nodes([_make_pack(tmp_path)])
    assert len(nodes) == 2
    actors = {n.actor for n in nodes}
    assert actors == {"BTN", "BB"}


def test_different_histories_become_different_nodes(tmp_path):
    """Same actor, different histories -> different nodes."""
    btn_dir = tmp_path / "BTN"
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")     # facing UTG open
    _write(btn_dir / "UTG_Fold_HJ_60%_CO_Fold_BTN_Fold.txt")     # facing HJ open
    nodes = enumerate_nodes([_make_pack(tmp_path)])
    assert len(nodes) == 2
    assert all(n.actor == "BTN" for n in nodes)
    # Both histories start with UTG but with different action types --
    # one is a 60% open, the other is a fold.
    opener_position = {
        next(
            a for a in n.history_before
            if a.action_type is PreflopActionType.RAISE
        ).position
        for n in nodes
    }
    assert opener_position == {"UTG", "HJ"}


# --- multi-pack ---------------------------------------------------------------
def test_multi_pack_enumeration_tags_pack_id(tmp_path):
    """Two packs in the same enumeration -> each node tagged with source."""
    pack_a_root = tmp_path / "pack_a"
    pack_b_root = tmp_path / "pack_b"
    _write(pack_a_root / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    _write(pack_b_root / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    pack_a = _make_pack(pack_a_root, pack_id="pack_a")
    pack_b = _make_pack(pack_b_root, pack_id="pack_b")
    nodes = enumerate_nodes([pack_a, pack_b])
    assert len(nodes) == 2  # same logical node but tagged with different pack_ids
    assert {n.pack_id for n in nodes} == {"pack_a", "pack_b"}


# --- malformed-file resilience ----------------------------------------------
def test_malformed_filename_is_skipped_with_warning(tmp_path, caplog):
    """A bad filename produces a WARNING but doesn't abort enumeration."""
    btn_dir = tmp_path / "BTN"
    # Good file:
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    # Bad: parent folder doesn't match final action position
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BB_Fold.txt")
    # Bad: garbage action token
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Limp.txt")
    with caplog.at_level(logging.WARNING, logger="pipeline.preflop.node_enumerator"):
        nodes = enumerate_nodes([_make_pack(tmp_path)])
    # Only the good file made it into a node.
    assert len(nodes) == 1
    # Two skip warnings logged.
    skip_warnings = [r for r in caplog.records if "skipping" in r.message]
    assert len(skip_warnings) == 2


# --- node_id property ------------------------------------------------------
def test_node_id_includes_history_and_actor(tmp_path):
    _write(tmp_path / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    nodes = enumerate_nodes([_make_pack(tmp_path)])
    node = nodes[0]
    assert node.node_id == "UTG_60%_HJ_Fold_CO_Fold_BTN_decision"


def test_node_id_uses_decimal_g_for_sizes(tmp_path):
    """A 76.5% raise renders cleanly without trailing zeros."""
    _write(tmp_path / "BTN" / "UTG_76.5%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    nodes = enumerate_nodes([_make_pack(tmp_path)])
    assert nodes[0].node_id == "UTG_76.5%_HJ_Fold_CO_Fold_BTN_decision"


# --- has_action helper -----------------------------------------------------
def test_has_action_true_when_present(tmp_path):
    btn_dir = tmp_path / "BTN"
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Call.txt")
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    node = enumerate_nodes([_make_pack(tmp_path)])[0]
    assert node.has_action(PreflopActionType.FOLD)
    assert node.has_action(PreflopActionType.CALL)
    assert not node.has_action(PreflopActionType.RAISE)


# --- PreflopActionOption.label -------------------------------------------
def test_action_option_label_for_each_action_type(tmp_path):
    btn_dir = tmp_path / "BTN"
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_Call.txt")
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_77%.txt")
    _write(btn_dir / "UTG_60%_HJ_Fold_CO_Fold_BTN_AI.txt")
    node = enumerate_nodes([_make_pack(tmp_path)])[0]
    labels = {opt.label for opt in node.actions}
    assert labels == {"Fold", "Call", "Raise 77%", "AllIn"}


# --- enumerate_nodes_by_actor ---------------------------------------------
def test_enumerate_by_actor_groups_correctly(tmp_path):
    _write(tmp_path / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt")
    _write(tmp_path / "BTN" / "UTG_Fold_HJ_60%_CO_Fold_BTN_Fold.txt")
    _write(tmp_path / "BB" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_Fold.txt")
    by_actor = enumerate_nodes_by_actor([_make_pack(tmp_path)])
    assert set(by_actor) == {"BTN", "BB"}
    assert len(by_actor["BTN"]) == 2
    assert len(by_actor["BB"]) == 1


# --- ordering determinism --------------------------------------------------
def test_results_are_deterministically_ordered(tmp_path):
    """Multiple runs over the same pack return identically-ordered results."""
    btn_dir = tmp_path / "BTN"
    for size in ("60%", "Fold", "Call"):
        _write(btn_dir / f"UTG_Fold_HJ_Fold_CO_Fold_BTN_{size}.txt")
    first = enumerate_nodes([_make_pack(tmp_path)])
    second = enumerate_nodes([_make_pack(tmp_path)])
    assert [n.node_id for n in first] == [n.node_id for n in second]


# --- integration: real pack -------------------------------------------------
def test_enumerate_against_real_pack():
    """Sanity-check the enumerator on the real Ryan pack if present."""
    ranges = REPO_ROOT / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    from pipeline.preflop.pack import discover_packs
    packs = discover_packs(ranges)
    if not packs:
        pytest.skip("Ryan pack not present under ranges/")
    nodes = enumerate_nodes(packs)
    # The pack has ~20k files; node count should be a meaningful fraction.
    assert 1_000 <= len(nodes) <= 30_000, f"unexpected node count: {len(nodes)}"
    # Every node belongs to a known position.
    actors = {n.actor for n in nodes}
    assert actors.issubset({"UTG", "HJ", "CO", "BTN", "SB", "BB"})
    # Every node has at least 1 action.
    assert all(len(n.actions) >= 1 for n in nodes)
