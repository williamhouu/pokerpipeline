"""Tests for pipeline.preflop.grammars.ryan_pack -- the Ryan-format parser."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.grammars.ryan_pack import parse                     # noqa: E402
from pipeline.preflop.grammars.types import (                             # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.pack import PreflopPack                             # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pack(table_size: int = 6) -> PreflopPack:
    """Build a fake pack object for the parser to tag results with."""
    return PreflopPack(
        pack_id="ryan_test", root_path=Path("/tmp/fake"),
        grammar_name="ryan_pack", table_size=table_size,
        stack_depth_bb=100, open_size_bb=2.5,
    )


# --- happy path -------------------------------------------------------------
def test_parse_srp_bb_call():
    """The Scenario-1 BB caller's range, simplest non-trivial file."""
    p = Path(
        "/x/PioViewer - NLH 6max 100bb 2.5x Open/BB/"
        "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Call.txt"
    )
    r = parse(p, _pack())
    assert r.actor == "BB"
    assert r.actor_action == PreflopActionType.CALL
    assert r.actor_raise_size_pct is None
    assert len(r.action_history) == 6
    # Spot-check: BTN opens 60%, BB calls.
    btn = r.action_history[3]
    assert btn.position == "BTN"
    assert btn.action_type == PreflopActionType.RAISE
    assert btn.raise_size_pct == 60.0
    bb = r.action_history[-1]
    assert bb.position == "BB"
    assert bb.action_type == PreflopActionType.CALL


def test_parse_btn_open():
    """The BTN opener's range itself."""
    p = Path(
        "/x/PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
        "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%.txt"
    )
    r = parse(p, _pack())
    assert r.actor == "BTN"
    assert r.actor_action == PreflopActionType.RAISE
    assert r.actor_raise_size_pct == 60.0
    assert len(r.action_history) == 4


def test_parse_3bet_pot():
    """3-bet pot: BTN opens, BB 3-bets to 182%, BTN calls."""
    p = Path(
        "/x/PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
        "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_Call.txt"
    )
    r = parse(p, _pack())
    assert r.actor == "BTN"
    assert r.actor_action == PreflopActionType.CALL
    bb_3bet = r.action_history[5]
    assert bb_3bet.position == "BB"
    assert bb_3bet.action_type == PreflopActionType.RAISE
    assert bb_3bet.raise_size_pct == 182.0


def test_parse_all_in_token():
    """All-in token (AI) is recognised."""
    p = Path(
        "/x/PioViewer - NLH 6max 100bb 2.5x Open/BB/"
        "UTG_60%_HJ_Fold_CO_Fold_BTN_AI_SB_Fold_BB_Fold.txt"
    )
    r = parse(p, _pack())
    btn = r.action_history[3]
    assert btn.action_type == PreflopActionType.ALL_IN
    assert btn.raise_size_pct is None


def test_parse_multi_round():
    """A multi-round file: 6 (R1) + 6 (R2) + 4 (R3) = 16 actions across 3 rounds.

    Round 1: UTG opens, HJ-CO-BTN-SB all call, BB squeezes.
    Round 2: UTG-HJ-CO-BTN call the squeeze, SB jams, BB folds.
    Round 3: (SB and BB out) UTG-HJ-CO-BTN each fold.
    """
    p = Path(
        "/x/PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
        "UTG_60%_HJ_Call_CO_Call_BTN_Call_SB_Call_BB_198%_"
        "UTG_Call_HJ_Call_CO_Call_BTN_Call_SB_AI_BB_Fold_"
        "UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold.txt"
    )
    r = parse(p, _pack())
    assert r.actor == "BTN"
    assert r.actor_action == PreflopActionType.FOLD
    assert len(r.action_history) == 16
    # Round 1 squeeze action: BB raises 198%.
    bb_squeeze = r.action_history[5]
    assert bb_squeeze.position == "BB"
    assert bb_squeeze.raise_size_pct == 198.0
    # SB's all-in in round 2.
    sb_ai = next(
        a for a in r.action_history
        if a.position == "SB" and a.action_type == PreflopActionType.ALL_IN
    )
    assert sb_ai.raise_size_pct is None


def test_parse_decimal_raise_size():
    """Sizing tokens with decimals (e.g. 76.5%) are accepted."""
    p = Path("/x/PioViewer - NLH 6max 100bb 2.5x Open/SB/UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_76.5%.txt")
    r = parse(p, _pack())
    sb = r.action_history[-1]
    assert sb.action_type == PreflopActionType.RAISE
    assert sb.raise_size_pct == 76.5


# --- error paths ------------------------------------------------------------
def test_parse_rejects_odd_token_count():
    """A trailing unpaired token (e.g. position with no action) is rejected."""
    p = Path("/x/PioViewer - NLH 6max 100bb 2.5x Open/BB/UTG_60%_BB.txt")
    with pytest.raises(ValueError, match="odd token count"):
        parse(p, _pack())


def test_parse_rejects_unknown_position():
    p = Path("/x/PioViewer - NLH 6max 100bb 2.5x Open/BB/XYZ_60%_BB_Call.txt")
    with pytest.raises(ValueError, match="unknown position 'XYZ'"):
        parse(p, _pack())


def test_parse_rejects_unknown_action_token():
    """A garbage action like 'Limp' is rejected (only Fold/Call/AI/<N>% allowed)."""
    p = Path(
        "/x/PioViewer - NLH 6max 100bb 2.5x Open/BB/"
        "UTG_60%_HJ_Fold_CO_Fold_BTN_Limp_SB_Fold_BB_Call.txt"
    )
    with pytest.raises(ValueError, match="unrecognised action token 'Limp'"):
        parse(p, _pack())


def test_parse_rejects_parent_folder_mismatch():
    """Filename ends in BB_Call but lives in BTN/ folder -> rejected."""
    p = Path(
        "/x/PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
        "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Call.txt"
    )
    with pytest.raises(ValueError, match="does not match parent folder"):
        parse(p, _pack())


# --- metadata propagation ---------------------------------------------------
def test_parse_tags_pack_id():
    """The returned ParsedRangeFile carries the source pack's id."""
    p = Path("/x/PioViewer - NLH 6max 100bb 2.5x Open/SB/UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold.txt")
    r = parse(p, _pack())
    assert r.pack_id == "ryan_test"


def test_parse_accepts_9max_position_tokens():
    """The parser knows about UTG1/UTG2/LJ for the future 9-max pack."""
    p = Path("/x/9max-pack/UTG/UTG_60%.txt")
    r = parse(p, _pack(table_size=9))
    assert r.actor == "UTG"
    p2 = Path("/x/9max-pack/UTG1/UTG_60%_UTG1_Fold.txt")
    r2 = parse(p2, _pack(table_size=9))
    assert r2.actor == "UTG1"


# --- ParsedAction equality helper ------------------------------------------
def test_parsed_action_equality():
    """Two ParsedActions with the same fields compare equal (frozen dataclass)."""
    a = ParsedAction(position="BTN", action_type=PreflopActionType.RAISE,
                     raise_size_pct=60.0)
    b = ParsedAction(position="BTN", action_type=PreflopActionType.RAISE,
                     raise_size_pct=60.0)
    assert a == b
    c = ParsedAction(position="BTN", action_type=PreflopActionType.CALL)
    assert a != c
