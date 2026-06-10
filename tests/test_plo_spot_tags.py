"""Tests for pipeline.plo.spot_tags (facts-relative concept tags)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402
from pipeline.plo.spot_tags import compute_plo_concept_tags  # noqa: E402

F = PloActionType.FOLD
C = PloActionType.CALL
R = PloActionType.RAISE

ACE_SUITED = ("As", "Ks", "2h", "3d")  # ace suited to the king -> suited_ace
ACE_OFFSUIT = ("Ac", "Kd", "2h", "3s")  # has an ace, but not suited
NO_ACE = ("Kc", "Qd", "2h", "3s")
PREMIUM_DS = ("As", "Ks", "Ah", "Kh")  # double-suited AAKK


def _act(seat: str, atype: PloActionType, pct: int | None = None) -> PloAction:
    return PloAction(seat=seat, action=atype, raise_pct=pct)


def _tags(
    *,
    actor: str = "HJ",
    history: tuple[PloAction, ...] = (),
    freqs: dict[str, float] | None = None,
    hero_cards: tuple[str, str, str, str] = NO_ACE,
    eq: float | None = None,
    range_eq: float | None = None,
    villain: bool = False,
) -> list[str]:
    node = PloDecisionNode(
        actor=actor, history_before=history, actions=(), history_stem=""
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=hero_cards,
        action_frequencies=freqs or {"Call": 1.0},
        presence=1.0,
    )
    vstats = (
        PloVillainStats(
            seat="LJ",
            action_label="Raise 100%",
            weighted_combo_count=1.0,
            pct_of_dealt_hands=1.0,
        )
        if villain
        else None
    )
    facts = PloFacts(
        spot=spot,
        hand_class=classify_plo_hand(hero_cards),
        archetype="x",
        villain_stats=vstats,
        hero_equity_vs_villain=eq,
        hero_range_equity_vs_villain=range_eq,
    )
    return compute_plo_concept_tags(facts)


# --- position -------------------------------------------------------------
def test_position_buckets():
    assert "early_position" in _tags(actor="LJ")
    assert "middle_position" in _tags(actor="HJ")
    assert "late_position" in _tags(actor="CO")
    assert "late_position" in _tags(actor="BU")
    assert "small_blind" in _tags(actor="SB")
    assert "big_blind" in _tags(actor="BB")


# --- decision context -----------------------------------------------------
def test_open_decision():
    assert "open_decision" in _tags(actor="LJ", history=())


def test_facing_single_raise():
    tags = _tags(actor="HJ", history=(_act("LJ", R, 100),))
    assert "facing_single_raise" in tags
    assert "facing_3bet" not in tags


def test_facing_3bet_and_4bet_plus():
    assert "facing_3bet" in _tags(history=(_act("LJ", R, 100), _act("HJ", R, 100)))
    assert "facing_4bet_plus" in _tags(
        history=(_act("LJ", R, 100), _act("HJ", R, 100), _act("CO", R, 100))
    )


def test_squeeze_opportunity():
    tags = _tags(actor="CO", history=(_act("LJ", R, 100), _act("HJ", C)))
    assert "squeeze_opportunity" in tags
    assert "facing_single_raise" not in tags  # a caller is between the raise and hero


def test_bvb_spot():
    tags = _tags(
        actor="BB",
        history=(_act("LJ", F), _act("HJ", F), _act("CO", F), _act("BU", F), _act("SB", R, 100)),
    )
    assert "bvb_spot" in tags


def test_multiway_pot():
    tags = _tags(
        actor="BB", history=(_act("LJ", R, 100), _act("HJ", C), _act("CO", C))
    )
    assert "multiway_pot" in tags


def test_multiway_pot_excludes_entrant_who_later_folds():
    # HJ opened then folded to BB's 3-bet: SB's decision is heads-up vs BB,
    # not multiway -- HJ's dead money doesn't keep him in the pot.
    history = (_act("HJ", R, 100), _act("SB", C), _act("BB", R, 100), _act("HJ", F))
    assert "multiway_pot" not in _tags(actor="SB", history=history)
    # If HJ calls the 3-bet instead, the pot is genuinely three-way.
    history = (_act("HJ", R, 100), _act("SB", C), _act("BB", R, 100), _act("HJ", C))
    assert "multiway_pot" in _tags(actor="SB", history=history)


# --- strategy shape -------------------------------------------------------
def test_strategy_shape_tags():
    assert "mixed_strategy" in _tags(freqs={"Raise 100%": 0.7, "Fold": 0.3})
    assert "near_pure_strategy" in _tags(freqs={"Call": 0.98, "Fold": 0.02})
    assert "dominant_is_aggressive" in _tags(freqs={"Raise 100%": 1.0})
    assert "dominant_is_passive" in _tags(freqs={"Call": 1.0})
    assert "dominant_is_fold" in _tags(freqs={"Fold": 1.0})


# --- equity context -------------------------------------------------------
def test_equity_bands():
    assert "equity_dominant" in _tags(eq=0.70)
    assert "equity_favorite" in _tags(eq=0.58)
    assert "coinflip" in _tags(eq=0.50)
    assert "dominated" in _tags(eq=0.35)


def test_no_equity_tag_without_equity():
    tags = _tags(eq=None)
    for name in ("equity_dominant", "equity_favorite", "coinflip", "dominated"):
        assert name not in tags


# --- range dynamics -------------------------------------------------------
def test_range_dynamics_bands():
    assert "hero_range_advantage" in _tags(range_eq=0.56)
    assert "villain_range_advantage" in _tags(range_eq=0.44)
    assert "roughly_equal_ranges" in _tags(range_eq=0.50)


# --- blockers -------------------------------------------------------------
def test_blocker_tags_need_an_ace_and_a_villain():
    assert "blocks_villain_value" in _tags(hero_cards=ACE_OFFSUIT, villain=True)
    assert "blocks_villain_value" not in _tags(hero_cards=ACE_OFFSUIT, villain=False)
    assert "blocks_villain_value" not in _tags(hero_cards=NO_ACE, villain=True)


def test_nut_flush_blocker_needs_a_suited_ace():
    assert "blocks_villain_nut_flush" in _tags(hero_cards=ACE_SUITED, villain=True)
    assert "blocks_villain_nut_flush" not in _tags(hero_cards=ACE_OFFSUIT, villain=True)


# --- stack + aggregation --------------------------------------------------
def test_standard_stack_always_fires():
    tags = _tags()
    assert "standard_stack" in tags
    assert "short_stack" not in tags
    assert "deep_stack" not in tags


def test_aggregator_includes_hand_structure_tags():
    tags = _tags(hero_cards=PREMIUM_DS, villain=True, eq=0.63)
    # hand-structure tags (from compute_plo_hand_tags) are mixed in:
    assert "double_suited" in tags
    assert "pocket_aces" in tags
    # and the facts-relative tags:
    assert "equity_dominant" in tags  # 0.63 > 0.62
    assert "standard_stack" in tags
