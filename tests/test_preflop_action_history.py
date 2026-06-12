"""Tests for pipeline.preflop.action_history helpers.

The size-resolution helper `_raise_size_bb` had a bug where the level-1
short-circuit returned `pack.open_size_bb` without consulting the
position-specific lookup table -- so SB BvB opens (3bb, token 76.0)
came back as the generic 2.5bb open size. This file covers the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.action_history import (  # noqa: E402
    _RYAN_PACK_RAISE_SIZES_BB,
    _raise_size_bb,
)
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.pack import PreflopPack  # noqa: E402


def _pack(open_size_bb: float = 2.5) -> PreflopPack:
    """Minimal pack fixture mirroring the Ryan 6-max 100bb 2.5x setup."""
    return PreflopPack(
        pack_id="t", root_path=Path("/tmp/test_pack"),
        grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100,
        open_size_bb=open_size_bb, sb_to_bb_ratio=0.5,
    )


# --- bug regression: SB BvB open ------------------------------------------
def test_raise_size_bb_resolves_sb_bvb_open_to_3bb() -> None:
    """SB opens in a BvB spot with token 76.0 -- the lookup table says
    3.0bb but the old code short-circuited to pack.open_size_bb (2.5).
    Verify the table now wins.
    """
    sb_open = ParsedAction("SB", PreflopActionType.RAISE, 76.0)
    assert _raise_size_bb(sb_open, raise_level=1, pack=_pack()) == 3.0


def test_raise_size_bb_resolves_standard_open_to_pack_default() -> None:
    """Token 60.0 (UTG/HJ/CO/BTN open) is in the table at 2.5bb AND
    matches pack.open_size_bb -- both paths give 2.5bb."""
    btn_open = ParsedAction("BTN", PreflopActionType.RAISE, 60.0)
    assert _raise_size_bb(btn_open, raise_level=1, pack=_pack()) == 2.5


def test_raise_size_bb_unknown_level_1_token_falls_back_to_pack() -> None:
    """If a token has no level-1 table entry, fall back to
    pack.open_size_bb. Preserves the "don't crash on unknown tokens"
    behavior the old short-circuit provided."""
    unknown_open = ParsedAction("UTG", PreflopActionType.RAISE, 999.0)
    assert _raise_size_bb(unknown_open, raise_level=1, pack=_pack()) == 2.5


def test_raise_size_bb_uses_table_for_3bet() -> None:
    """Sanity: non-level-1 lookups still work. BB 3-bets vs HJ open
    with token 182.0 -> 12bb per the table."""
    bb_3bet = ParsedAction("BB", PreflopActionType.RAISE, 182.0)
    assert _raise_size_bb(bb_3bet, raise_level=2, pack=_pack()) == 12.0


def test_raise_size_bb_fallback_for_unknown_3bet() -> None:
    """Unknown 3-bet token falls back to the multiplicative heuristic
    (3x open)."""
    unknown_3bet = ParsedAction("BB", PreflopActionType.RAISE, 999.0)
    # heuristic: open_size_bb * 3 = 7.5
    assert _raise_size_bb(unknown_3bet, raise_level=2, pack=_pack()) == 7.5


def test_lookup_table_has_distinct_sb_bvb_entry() -> None:
    """Sanity: the table actually distinguishes SB BvB (3bb) from
    standard opens (2.5bb). This locks the data so a later refactor
    doesn't silently delete the entry."""
    assert _RYAN_PACK_RAISE_SIZES_BB[(60.0, 1)] == 2.5
    assert _RYAN_PACK_RAISE_SIZES_BB[(76.0, 1)] == 3.0


# --- compute_action_pending (June 2026 multiway-awareness facts) -------------
def test_pending_open_spot_everyone_behind() -> None:
    """CO first-in at 6-max: BTN/SB/BB all still to act."""
    from pipeline.preflop.action_history import compute_action_pending
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType

    history = tuple(
        ParsedAction(p, PreflopActionType.FOLD) for p in ("UTG", "HJ")
    )
    others, pending, closes = compute_action_pending(history, "CO", 6)
    assert others == ["BTN", "SB", "BB"]
    assert pending == ["BTN", "SB", "BB"]
    assert closes is False


def test_pending_hero_closes_after_jam_when_all_folded() -> None:
    """The June-12 audit #1 spot shape: jam, everyone else folds, hero
    closes -- nobody can 'wake up behind you'."""
    from pipeline.preflop.action_history import compute_action_pending
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType

    PT = PreflopActionType
    history = (
        ParsedAction("UTG", PT.FOLD), ParsedAction("UTG+1", PT.FOLD),
        ParsedAction("UTG+2", PT.FOLD),
        ParsedAction("LJ", PT.RAISE, 120.0),
        ParsedAction("HJ", PT.CALL), ParsedAction("CO", PT.CALL),
        ParsedAction("BTN", PT.RAISE, 90.0),
        ParsedAction("SB", PT.FOLD), ParsedAction("BB", PT.CALL),
        ParsedAction("LJ", PT.CALL),
        ParsedAction("HJ", PT.ALL_IN),
        ParsedAction("CO", PT.FOLD), ParsedAction("BTN", PT.FOLD),
        ParsedAction("BB", PT.FOLD),
    )
    others, pending, closes = compute_action_pending(history, "LJ", 9)
    assert others == ["HJ (all-in)"]
    assert pending == []
    assert closes is True


def test_pending_caller_before_jam_still_to_act() -> None:
    """The audit #3 spot shape: LJ called BEFORE the jam so LJ still has a
    decision, but UTG+1 (folded) does not."""
    from pipeline.preflop.action_history import compute_action_pending
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType

    PT = PreflopActionType
    history = (
        ParsedAction("UTG", PT.FOLD),
        ParsedAction("UTG+1", PT.RAISE, 120.0),
        ParsedAction("UTG+2", PT.CALL),
        ParsedAction("LJ", PT.CALL),
        ParsedAction("HJ", PT.ALL_IN),
        ParsedAction("CO", PT.FOLD), ParsedAction("BTN", PT.FOLD),
        ParsedAction("SB", PT.FOLD), ParsedAction("BB", PT.FOLD),
        ParsedAction("UTG+1", PT.FOLD),
    )
    others, pending, closes = compute_action_pending(history, "UTG+2", 9)
    assert others == ["LJ", "HJ (all-in)"]
    assert pending == ["LJ"]
    assert closes is False


def test_pending_bb_option_in_limped_pot() -> None:
    """SB limps: the BB has matched the bet but never acted -> the BB has
    the option (pending) until they check/raise."""
    from pipeline.preflop.action_history import compute_action_pending
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType

    PT = PreflopActionType
    history = tuple(
        ParsedAction(p, PT.FOLD)
        for p in ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN")
    ) + (ParsedAction("SB", PT.CALL),)
    others, pending, closes = compute_action_pending(history, "SB", 9)
    # From the SB's own decision point AFTER limping isn't meaningful;
    # check the BB's view instead: SB limped, BB to act -> closes.
    others, pending, closes = compute_action_pending(history, "BB", 9)
    assert others == ["SB"]
    assert pending == []
    assert closes is True


def test_pending_jam_over_jam_is_call_off() -> None:
    """A second all-in over an existing all-in must NOT reopen the action
    for players who already called the first jam."""
    from pipeline.preflop.action_history import compute_action_pending
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType

    PT = PreflopActionType
    history = (
        ParsedAction("UTG", PT.FOLD), ParsedAction("UTG+1", PT.FOLD),
        ParsedAction("UTG+2", PT.ALL_IN),
        ParsedAction("LJ", PT.CALL),
        ParsedAction("HJ", PT.ALL_IN),
        ParsedAction("CO", PT.FOLD), ParsedAction("BTN", PT.FOLD),
        ParsedAction("SB", PT.FOLD),
    )
    others, pending, closes = compute_action_pending(history, "BB", 9)
    assert others == ["UTG+2 (all-in)", "LJ", "HJ (all-in)"]
    # LJ already matched the full stack calling jam #1; the equal-stack
    # second jam leaves it nothing to decide -> no pending action.
    assert pending == []
    assert closes is True
