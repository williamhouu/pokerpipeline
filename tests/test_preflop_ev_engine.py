"""Tests for pipeline.preflop.ev_engine.

The v1 engine handles call/fold EVs precisely and skips raise EVs
(documented limitation -- raise EVs need villain's response distribution,
which requires a game-tree solver). Tests cover:
  - EV(Fold) is always 0
  - EV(Call) follows the equity * pot - cost_to_call formula
  - compute_ev_gap_bb returns a number for call/fold spots
  - compute_ev_gap_bb returns None when raises are involved
  - the gap respects the call_cost/pot/equity math from PreflopFacts
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.ev_engine import (  # noqa: E402
    compute_ev_gap_bb,
    ev_call_bb,
    ev_fold_bb,
)
from pipeline.preflop.fact_extractor import (  # noqa: E402
    PreflopFacts,
    VillainRangeStats,
)
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.pack import PreflopPack  # noqa: E402
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


def _pack(stack_depth_bb: int = 100) -> PreflopPack:
    return PreflopPack(
        pack_id="t",
        root_path=Path("/tmp/test_pack"),
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=stack_depth_bb,
        open_size_bb=2.5,
        sb_to_bb_ratio=0.5,
    )


def _facts(
    *,
    actor: str = "BB",
    history: tuple[ParsedAction, ...] = (),
    action_frequencies: dict[str, float] | None = None,
    dominant_action: str = "Call",
    dominant_frequency: float = 0.66,
    hero_equity: float | None = 0.50,
) -> PreflopFacts:
    """Build a minimal PreflopFacts for EV testing. Default fixture:
    BB facing a BTN open with a 66/34 call/fold mix."""
    if action_frequencies is None:
        action_frequencies = {"Fold": 0.34, "Call": 0.66}
    if not history:
        history = (
            ParsedAction("UTG", PreflopActionType.FOLD),
            ParsedAction("HJ", PreflopActionType.FOLD),
            ParsedAction("CO", PreflopActionType.FOLD),
            ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
            ParsedAction("SB", PreflopActionType.FOLD),
        )
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t",
            actor=actor,
            history_before=history,
            actions=(),
        ),
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies=action_frequencies,
        dominant_action=dominant_action,
        dominant_frequency=dominant_frequency,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="BTN",
            action_label="Raise 60%",
            weighted_combo_count=600.0,
            pct_of_dealt_hands=45.0,
            top_combos=(),
        ),
        hero_equity_vs_villain=hero_equity,
        archetype="call_for_value",
    )


# --- ev_fold_bb -------------------------------------------------------------
def test_ev_fold_is_always_zero() -> None:
    """Folding gives up nothing more than already committed; we measure
    relative to status quo."""
    assert ev_fold_bb() == 0.0


# --- ev_call_bb -------------------------------------------------------------
def test_ev_call_at_50_pct_equity_bb_facing_btn_open() -> None:
    """BB facing 2.5bb BTN open:
    pot = SB 0.5 + BB 1.0 + BTN 2.5 = 4bb
    call_cost = 2.5 - 1.0 (BB already in) = 1.5bb
    EV(Call) = 0.50 * (4 + 1.5) - 1.5 = 2.75 - 1.5 = 1.25 bb.
    """
    facts = _facts(actor="BB", hero_equity=0.50)
    ev = ev_call_bb(facts, _pack())
    assert ev is not None
    assert ev == pytest.approx(1.25)


def test_ev_call_at_30_pct_equity_negative() -> None:
    """Same spot, 30% equity:
      EV(Call) = 0.30 * 5.5 - 1.5 = 1.65 - 1.5 = 0.15 bb.
    Just barely positive at 30% equity vs a 2.5bb open (pot odds = 1.5/4
    = 27.3%, so 30% equity is breakeven-positive)."""
    facts = _facts(actor="BB", hero_equity=0.30)
    ev = ev_call_bb(facts, _pack())
    assert ev is not None
    assert ev == pytest.approx(0.15)


def test_ev_call_at_25_pct_equity_negative() -> None:
    """Below pot-odds equity: EV(Call) < 0.
    EV(Call) = 0.25 * 5.5 - 1.5 = 1.375 - 1.5 = -0.125 bb.
    """
    facts = _facts(actor="BB", hero_equity=0.25)
    ev = ev_call_bb(facts, _pack())
    assert ev is not None
    assert ev == pytest.approx(-0.125)


def test_ev_call_returns_none_without_equity() -> None:
    """Open spot (no villain, no equity data) -> can't compute EV(Call)."""
    facts = _facts(hero_equity=None)
    assert ev_call_bb(facts, _pack()) is None


def test_ev_call_sb_facing_btn_open_has_different_call_cost() -> None:
    """SB facing BTN open:
      pot = SB 0.5 + BB 1.0 + BTN 2.5 = 4bb
      call_cost = 2.5 - 0.5 (SB already in) = 2.0bb
      EV(Call) at 50% = 0.5 * (4 + 2) - 2 = 3 - 2 = 1.0 bb.
    SB has to put in more than BB to call, so the EV is lower at the
    same equity."""
    facts = _facts(
        actor="SB",
        history=(
            ParsedAction("UTG", PreflopActionType.FOLD),
            ParsedAction("HJ", PreflopActionType.FOLD),
            ParsedAction("CO", PreflopActionType.FOLD),
            ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ),
        hero_equity=0.50,
    )
    ev = ev_call_bb(facts, _pack())
    assert ev is not None
    assert ev == pytest.approx(1.0)


# --- compute_ev_gap_bb ------------------------------------------------------
def test_ev_gap_call_dominant_over_fold() -> None:
    """Call dominant (66%), Fold secondary (34%).
    EV(Call) at 50% equity = 1.25 bb (computed above).
    EV(Fold) = 0.
    gap = 1.25 - 0 = 1.25 bb.
    """
    facts = _facts(
        actor="BB",
        action_frequencies={"Call": 0.66, "Fold": 0.34},
        dominant_action="Call",
        hero_equity=0.50,
    )
    gap = compute_ev_gap_bb(facts, _pack())
    assert gap is not None
    assert gap == pytest.approx(1.25)


def test_ev_gap_fold_dominant_over_call_below_pot_odds() -> None:
    """Fold dominant (60%), Call secondary (40%). Equity 25%, so calling
    is -EV:
      EV(Call) = -0.125 bb. EV(Fold) = 0.
      gap = |0 - (-0.125)| = 0.125 bb.
    The gap is the AMOUNT hero saves by folding."""
    facts = _facts(
        actor="BB",
        action_frequencies={"Fold": 0.60, "Call": 0.40},
        dominant_action="Fold",
        hero_equity=0.25,
    )
    gap = compute_ev_gap_bb(facts, _pack())
    assert gap is not None
    assert gap == pytest.approx(0.125)


def test_ev_gap_returns_none_when_dominant_is_raise() -> None:
    """v1 engine doesn't model raise EVs -- gap returns None when
    dominant or secondary is a Raise."""
    facts = _facts(
        actor="BB",
        action_frequencies={"Raise 308%": 0.60, "Call": 0.40},
        dominant_action="Raise 308%",
        hero_equity=0.55,
    )
    assert compute_ev_gap_bb(facts, _pack()) is None


def test_ev_gap_returns_none_when_secondary_is_raise() -> None:
    """Same restriction the other way: Call dominant but Raise secondary."""
    facts = _facts(
        actor="BB",
        action_frequencies={"Call": 0.60, "Raise 308%": 0.40},
        dominant_action="Call",
        hero_equity=0.55,
    )
    assert compute_ev_gap_bb(facts, _pack()) is None


def test_ev_gap_returns_none_for_single_action_strategy() -> None:
    """Pure strategy (one canonical action) -> can't compute gap."""
    facts = _facts(
        actor="BB",
        action_frequencies={"Call": 1.0},
        dominant_action="Call",
        hero_equity=0.50,
    )
    assert compute_ev_gap_bb(facts, _pack()) is None


def test_ev_gap_returns_none_without_equity_data() -> None:
    """No equity -> can't compute Call EV -> can't compute gap."""
    facts = _facts(
        actor="BB",
        action_frequencies={"Call": 0.66, "Fold": 0.34},
        dominant_action="Call",
        hero_equity=None,
    )
    assert compute_ev_gap_bb(facts, _pack()) is None


def test_ev_gap_is_always_non_negative() -> None:
    """The gap is the absolute difference between dominant and 2nd-best
    EVs -- always >= 0 by convention (the worthiness filter looks at
    magnitude)."""
    # Construct a scenario where Pio's "dominant" might have lower EV than
    # the alternative -- shouldn't happen in real data, but defensive:
    # Pio plays Fold 60% with 50% equity at $0.50/bb. Pio's choice would
    # be irrational here but the engine returns the magnitude either way.
    facts = _facts(
        actor="BB",
        action_frequencies={"Fold": 0.60, "Call": 0.40},
        dominant_action="Fold",
        hero_equity=0.50,
    )
    gap = compute_ev_gap_bb(facts, _pack())
    assert gap is not None
    assert gap >= 0
