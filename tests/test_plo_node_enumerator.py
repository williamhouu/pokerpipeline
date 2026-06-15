"""Tests for pipeline.plo.node_enumerator (sibling-file grouping into nodes)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.node_enumerator import (  # noqa: E402
    PLO_ACTION_CONTEXTS,
    PloDecisionNode,
    enumerate_plo_nodes,
    enumerate_plo_nodes_by_actor,
    plo_active_player_count,
    plo_node_action_context,
)
from pipeline.plo.pack import PloAction, PloActionType, PloPack  # noqa: E402

F = PloActionType.FOLD
C = PloActionType.CALL
R = PloActionType.RAISE
J = PloActionType.ALL_IN


def _touch(path: Path) -> None:
    # The enumerator parses filenames only -- file contents are irrelevant.
    path.write_text("x", encoding="utf-8")


def _pack_with(tmp_path: Path, *stems: str) -> PloPack:
    for stem in stems:
        _touch(tmp_path / f"{stem}.rng")
    return PloPack(root=tmp_path, label="test")


# An open node (LJ: fold or raise) plus an HJ node facing the open
# (fold / call / 3-bet). Five files -> two nodes.
def _two_node_pack(tmp_path: Path) -> PloPack:
    return _pack_with(
        tmp_path,
        "0",  # LJ fold
        "40100",  # LJ open-raise
        "40100.0",  # HJ fold to open
        "40100.1",  # HJ call open
        "40100.40100",  # HJ 3-bet
    )


def test_groups_siblings_into_two_nodes(tmp_path):
    nodes = enumerate_plo_nodes(_two_node_pack(tmp_path))
    assert len(nodes) == 2
    actors = {n.actor for n in nodes}
    assert actors == {"LJ", "HJ"}


def test_open_node_has_fold_and_raise(tmp_path):
    nodes = enumerate_plo_nodes(_two_node_pack(tmp_path))
    lj = next(n for n in nodes if n.actor == "LJ")
    assert lj.history_before == ()
    assert lj.history_stem == ""
    assert lj.node_id == "LJ_decision"
    assert lj.has_action(F)
    assert lj.has_action(R)
    assert not lj.has_action(C)
    assert {opt.label for opt in lj.actions} == {"Fold", "Raise 100%"}


def test_facing_open_node_history_and_actions(tmp_path):
    nodes = enumerate_plo_nodes(_two_node_pack(tmp_path))
    hj = next(n for n in nodes if n.actor == "HJ")
    # History is exactly LJ's open-raise.
    assert len(hj.history_before) == 1
    (prior,) = hj.history_before
    assert prior.seat == "LJ"
    assert prior.action is R
    assert hj.history_stem == "40100"
    assert hj.node_id == "LJ_raise100_HJ_decision"
    assert {opt.label for opt in hj.actions} == {"Fold", "Call", "Raise 100%"}


def test_actions_sorted_by_filename_for_determinism(tmp_path):
    nodes = enumerate_plo_nodes(_two_node_pack(tmp_path))
    hj = next(n for n in nodes if n.actor == "HJ")
    stems = [opt.stem for opt in hj.actions]
    assert stems == sorted(stems)
    # Each option's stem is the node's history_stem plus its own token.
    for opt in hj.actions:
        assert opt.stem.startswith(hj.history_stem)


def test_option_stem_and_path_point_at_the_file(tmp_path):
    nodes = enumerate_plo_nodes(_two_node_pack(tmp_path))
    lj = next(n for n in nodes if n.actor == "LJ")
    raise_opt = next(o for o in lj.actions if o.action.action is R)
    assert raise_opt.stem == "40100"
    assert raise_opt.path == tmp_path / "40100.rng"
    assert raise_opt.action.raise_pct == 100  # noqa: PLR2004


def test_malformed_filename_is_skipped(tmp_path):
    pack = _pack_with(tmp_path, "0", "40100")
    _touch(tmp_path / "zzz.rng")  # unrecognised action token -> skipped
    nodes = enumerate_plo_nodes(pack)
    # Only the one valid LJ node survives; the bad file is dropped, not fatal.
    assert len(nodes) == 1
    assert nodes[0].actor == "LJ"


def test_by_actor_grouping(tmp_path):
    by_actor = enumerate_plo_nodes_by_actor(_two_node_pack(tmp_path))
    assert set(by_actor) == {"LJ", "HJ"}
    assert len(by_actor["LJ"]) == 1
    assert len(by_actor["HJ"]) == 1


def test_squeeze_node_actor_is_last_raiser(tmp_path):
    # LJ open, all call around, BB squeezes -> BB is the actor.
    pack = _pack_with(tmp_path, "40100.1.1.1.1.40100")
    nodes = enumerate_plo_nodes(pack)
    assert len(nodes) == 1
    assert nodes[0].actor == "BB"
    assert nodes[0].history_stem == "40100.1.1.1.1"


# --- action-context + player-count categorization -------------------------
def _act(seat: str, atype: PloActionType, pct: int | None = None) -> PloAction:
    return PloAction(seat=seat, action=atype, raise_pct=pct)


def _node_with(actor: str, *history: PloAction) -> PloDecisionNode:
    return PloDecisionNode(
        actor=actor, history_before=tuple(history), actions=(), history_stem=""
    )


def test_action_context_opening():
    assert plo_node_action_context(_node_with("LJ")) == "Opening"


def test_action_context_facing_single_raise():
    node = _node_with("HJ", _act("LJ", R, 100))
    assert plo_node_action_context(node) == "Facing single raise"


def test_action_context_facing_3bet():
    node = _node_with("LJ", _act("LJ", R, 100), _act("BB", R, 100))
    assert plo_node_action_context(node) == "Facing 3-bet"


def test_action_context_facing_4bet_plus():
    node = _node_with(
        "BB", _act("LJ", R, 100), _act("BB", R, 100), _act("LJ", R, 100)
    )
    assert plo_node_action_context(node) == "Facing 4-bet+"


def test_action_context_all_in_counts_as_a_raise():
    node = _node_with("BB", _act("LJ", J))
    assert plo_node_action_context(node) == "Facing single raise"


def test_action_context_after_one_call():
    # A raise then a single live flat-caller before hero acts = a tame squeeze.
    node = _node_with("BB", _act("LJ", R, 100), _act("CO", C))
    assert plo_node_action_context(node) == "After one call"


def test_action_context_after_multiple_calls():
    # Two or more live flat-callers = the multiway bucket.
    node = _node_with("BB", _act("LJ", R, 100), _act("CO", C), _act("BU", C))
    assert plo_node_action_context(node) == "After multiple calls"


def test_action_context_folded_caller_is_not_live():
    # A caller who later folds doesn't count -- live callers = 1, so this stays
    # "After one call" rather than "multiple". (Mirrors the NLHE split test.)
    node = _node_with(
        "BB",
        _act("LJ", R, 100),
        _act("CO", C),
        _act("BU", C),
        _act("SB", R, 100),
        _act("CO", F),  # one of the two callers folds to the squeeze
    )
    assert plo_node_action_context(node) == "After one call"


def test_all_contexts_are_in_the_catalog(tmp_path):
    for node in enumerate_plo_nodes(_two_node_pack(tmp_path)):
        assert plo_node_action_context(node) in PLO_ACTION_CONTEXTS


def test_active_player_count_counts_non_folders_plus_hero():
    # LJ raises, HJ folds (excluded), CO calls; hero is BB -> {LJ, CO, BB}.
    node = _node_with("BB", _act("LJ", R, 100), _act("HJ", F), _act("CO", C))
    assert plo_active_player_count(node) == 3  # noqa: PLR2004


def test_active_player_count_open_node_is_heads_up_floor():
    # First in: only hero is "in" so far.
    assert plo_active_player_count(_node_with("LJ")) == 1


def test_active_player_count_excludes_entrant_who_later_folds():
    # The squeeze line: HJ opens, SB calls, BB 3-bets, HJ folds -> SB's
    # decision is heads-up vs BB (HJ's dead money doesn't keep him "in").
    node = _node_with(
        "SB", _act("HJ", R, 100), _act("SB", C), _act("BB", R, 100), _act("HJ", F)
    )
    assert plo_active_player_count(node) == 2  # noqa: PLR2004
    # If HJ calls the 3-bet instead, the pot is genuinely three-way.
    node = _node_with(
        "SB", _act("HJ", R, 100), _act("SB", C), _act("BB", R, 100), _act("HJ", C)
    )
    assert plo_active_player_count(node) == 3  # noqa: PLR2004
