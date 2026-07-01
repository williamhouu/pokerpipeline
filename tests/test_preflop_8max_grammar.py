"""Tests for the 8-max 200bb preflop pack adapter (gto_preflop_8max grammar).

Covers the pieces that make the materialised SQLite-derived pack readable by the
preflop pipeline, WITHOUT depending on the local (gitignored) pack data:

  * ``grammars.gto_preflop_8max.parse`` -- the ``SEAT-CODE`` filename decoder.
  * ``action_history`` -- the bb-native raise sizing branch (token == bb).
  * ``node_enumerator`` grouping ``.rng`` action files into a decision node.
  * ``fact_extractor.construct_villain_range_path`` -- slicing the option stem
    to the villain's own decision-node file (per-actor subfolder).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.action_history import resolve_preflop_history  # noqa: E402
from pipeline.preflop.fact_extractor import construct_villain_range_path  # noqa: E402
from pipeline.preflop.grammars.gto_preflop_8max import parse  # noqa: E402
from pipeline.preflop.grammars.types import PreflopActionType  # noqa: E402
from pipeline.preflop.node_enumerator import enumerate_nodes  # noqa: E402
from pipeline.preflop.pack import PreflopPack  # noqa: E402
from pipeline.preflop_ranges import canonical_169_hand_classes  # noqa: E402


def _pack(root: Path) -> PreflopPack:
    return PreflopPack(
        pack_id="preflop_8max_200bb",
        root_path=root,
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=200,
        open_size_bb=3.0,
        file_glob="*.rng",
        ev_units_per_bb=1.0,
    )


def _write_rng(path: Path) -> None:
    """A minimal valid 169-class .rng file (all zero weight + ev)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for hc in canonical_169_hand_classes():
        lines.append(hc)
        lines.append("0;0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# BTN opens to 3, SB folds, BB 3-bets to 17, BTN now decides (a vs-3bet node).
_HISTORY = "UTG-F_UTG+1-F_LJ-F_HJ-F_CO-F_BTN-R3_SB-F_BB-R17"


def test_grammar_parses_seats_actions_and_bb_sizes(tmp_path: Path) -> None:
    f = tmp_path / f"{_HISTORY}_BTN-C.rng"
    pr = parse(f, _pack(tmp_path))
    assert pr.actor == "BTN"
    assert pr.actor_action is PreflopActionType.CALL
    # 9 actions incl. the final call; UTG1 normalised to UTG+1.
    assert len(pr.action_history) == 9  # noqa: PLR2004
    assert pr.action_history[1].position == "UTG+1"
    btn_open = pr.action_history[5]
    assert btn_open.position == "BTN"
    assert btn_open.action_type is PreflopActionType.RAISE
    assert btn_open.raise_size_pct == 3.0  # noqa: PLR2004  # bb, not pct
    bb_3bet = pr.action_history[7]
    assert bb_3bet.raise_size_pct == 17.0  # noqa: PLR2004


def test_raise_sizes_resolve_bb_native(tmp_path: Path) -> None:
    pr = parse(tmp_path / f"{_HISTORY}_BTN-C.rng", _pack(tmp_path))
    st = resolve_preflop_history(pr.action_history[:-1], _pack(tmp_path))
    # Tokens ARE the bb amounts -- no pot-fraction conversion.
    assert st.sizes_bb[5] == 3.0  # noqa: PLR2004  # BTN open
    assert st.sizes_bb[7] == 17.0  # noqa: PLR2004  # BB 3-bet
    assert st.high_bet_bb == 17.0  # noqa: PLR2004
    assert st.raise_level == 2  # noqa: PLR2004


def test_enumerate_groups_actions_and_resolves_villain_file(tmp_path: Path) -> None:
    root = tmp_path / "preflop_8max_200bb"
    # The BTN vs-3bet node's options (fold / call / 4-bet) live in BTN/.
    for act in ("BTN-F", "BTN-C", "BTN-R38.5"):
        _write_rng(root / "BTN" / f"{_HISTORY}_{act}.rng")
    # The villain's (BB's) own 3-bet file lives in BB/ -- this is what
    # construct_villain_range_path must resolve to.
    villain_file = root / "BB" / f"{_HISTORY}.rng"
    _write_rng(villain_file)

    pack = _pack(root)
    nodes = enumerate_nodes([pack])
    btn = next(n for n in nodes if n.actor == "BTN")
    assert {o.action_type for o in btn.actions} == {
        PreflopActionType.FOLD, PreflopActionType.CALL, PreflopActionType.RAISE
    }
    # The BB's 3-bet ParsedAction from the history.
    bb_3bet = next(
        a for a in btn.history_before
        if a.position == "BB" and a.action_type is PreflopActionType.RAISE
    )
    resolved = construct_villain_range_path(btn, bb_3bet, pack)
    assert resolved == villain_file
    assert resolved.is_file()
