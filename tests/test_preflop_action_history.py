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
