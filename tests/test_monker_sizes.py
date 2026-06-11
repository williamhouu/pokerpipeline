"""Tests for Monker pot-relative raise sizing in the shared history resolver.

``resolve_preflop_history`` is the single source of truth for preflop pot
math (Question prose, POT column, EV engine, ranges-column action mixes).
For Monker-grammar packs the raise token IS the size (percent of pot);
these tests pin the resolved bb amounts against hand-checked arithmetic,
and pin that the Ryan-pack lookup path is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.action_history import (  # noqa: E402
    raise_to_bb,
    resolve_preflop_history,
)
from pipeline.preflop.grammars.monker_nlhe import parse  # noqa: E402
from pipeline.preflop.pack import PreflopPack  # noqa: E402


def _monker_pack() -> PreflopPack:
    return PreflopPack(
        pack_id="monker_test",
        root_path=Path("/tmp/fake"),
        grammar_name="monker_nlhe",
        table_size=9,
        stack_depth_bb=100,
        open_size_bb=4.0,
        file_glob="*.rng",
    )


def _ryan_pack() -> PreflopPack:
    return PreflopPack(
        pack_id="ryan_test",
        root_path=Path("/tmp/fake"),
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )


def _history(stem: str, pack: PreflopPack):
    """Decode a Monker stem into its full action tuple (incl. final action)."""
    return parse(Path(f"/x/{stem}.rng"), pack).action_history


# --- Monker pot-relative sizing ----------------------------------------------
def test_monker_first_in_open_40120_is_4bb():
    """The pack's 120%-pot open: 1 + 1.2 x (1.5 + 1) = 4.0 -- the 4x open."""
    state = resolve_preflop_history(_history("40120", _monker_pack()), _monker_pack())
    assert state.sizes_bb == (4.0,)
    assert state.high_bet_bb == 4.0
    assert state.pot_bb == pytest.approx(0.5 + 1.0 + 4.0)
    assert state.raise_level == 1


def test_monker_sb_first_in_open_40100_is_3bb():
    """SB open after 7 folds: SB owes 0.5, so 1 + 1.0 x (1.5 + 0.5) = 3.0."""
    stem = "0.0.0.0.0.0.0.40100"
    state = resolve_preflop_history(_history(stem, _monker_pack()), _monker_pack())
    assert state.sizes_bb[-1] == 3.0
    assert state.committed_bb["SB"] == 3.0
    assert state.pot_bb == pytest.approx(3.0 + 1.0)


def test_monker_bb_iso_raise_over_sb_limp_is_3bb():
    """SB limps (completes to 1), BB raises 100%: 1 + 1.0 x (2 + 0) = 3.0."""
    stem = "0.0.0.0.0.0.0.1.40100"
    state = resolve_preflop_history(_history(stem, _monker_pack()), _monker_pack())
    assert state.sizes_bb[7] is None  # the limp is a call, no "to" size
    assert state.sizes_bb[8] == 3.0
    assert state.pot_bb == pytest.approx(1.0 + 3.0)


def test_monker_bb_3bet_vs_4x_open():
    """UTG opens 4bb, folds to BB, BB raises 84%: 4 + 0.84 x (5.5 + 3) = 11.14."""
    stem = "40120.0.0.0.0.0.0.0.40084"
    state = resolve_preflop_history(_history(stem, _monker_pack()), _monker_pack())
    assert state.sizes_bb[-1] == pytest.approx(11.14)
    assert state.raise_level == 2


def test_monker_limp_raise_war_with_jam():
    """SB limp, BB iso 3bb, SB 3-bets 100%, BB jams, SB calls -> 200bb pot."""
    stem = "0.0.0.0.0.0.0.1.40100.40100.3.1"
    pack = _monker_pack()
    state = resolve_preflop_history(_history(stem, pack), pack)
    # SB re-raise: 3 + 1.0 x ((1+3) + 2) = 9.0
    assert state.sizes_bb[9] == pytest.approx(9.0)
    assert state.sizes_bb[10] == 100.0  # BB jam commits the stack
    assert state.sizes_bb[11] is None  # SB's call
    assert state.committed_bb == {"SB": 100.0, "BB": 100.0}
    assert state.pot_bb == pytest.approx(200.0)
    assert state.call_cost_bb("SB") == 0.0  # already matched


def test_monker_call_cost_facing_open():
    """BB facing the 4bb open owes 3 more; pot is 5.5."""
    stem = "40120.0.0.0.0.0.0.0"
    pack = _monker_pack()
    state = resolve_preflop_history(_history(stem, pack), pack)
    assert state.pot_bb == pytest.approx(5.5)
    assert state.call_cost_bb("BB") == pytest.approx(3.0)
    assert state.call_cost_bb("SB") == pytest.approx(3.5)


def test_monker_raise_to_bb_hypothetical_option():
    """Sizing an option AT a node (not in history) uses the resolved state."""
    stem = "40120.0.0.0.0.0.0.0"  # BB to act vs the open
    pack = _monker_pack()
    state = resolve_preflop_history(_history(stem, pack), pack)
    assert raise_to_bb(
        state, position="BB", raise_size_pct=84.0, pack=pack
    ) == pytest.approx(11.14)


# --- Ryan-pack lookup path unchanged ------------------------------------------
def test_ryan_history_sizes_still_come_from_lookup():
    from pipeline.preflop.grammars.types import (  # noqa: PLC0415
        ParsedAction,
        PreflopActionType,
    )

    history = (
        ParsedAction("UTG", PreflopActionType.FOLD, None),
        ParsedAction("HJ", PreflopActionType.FOLD, None),
        ParsedAction("CO", PreflopActionType.FOLD, None),
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ParsedAction("SB", PreflopActionType.FOLD, None),
        ParsedAction("BB", PreflopActionType.RAISE, 182.0),
    )
    state = resolve_preflop_history(history, _ryan_pack())
    # Documented pack sizes: 60% open = 2.5bb, BB 182% 3-bet = 12bb.
    assert state.sizes_bb == (None, None, None, 2.5, None, 12.0)
    assert state.pot_bb == pytest.approx(0.5 + 2.5 + 12.0)
    assert state.call_cost_bb("BTN") == pytest.approx(12.0 - 2.5)
    assert raise_to_bb(
        state, position="BTN", raise_size_pct=50.0, pack=_ryan_pack()
    ) == pytest.approx(25.0)  # (50%, level 3) lookup entry
