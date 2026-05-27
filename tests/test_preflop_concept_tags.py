"""Tests for pipeline.preflop.concept_tags.

Each tag is a pure boolean Python function. We test:
  - each tag fires when its condition is met
  - each tag does NOT fire when its condition isn't met (negative test)
  - the aggregator compute_concept_tags returns the right set of names
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.concept_tags import (  # noqa: E402
    ace_blocker,
    big_blind,
    blocks_villain_top_value,
    bvb_spot,
    coinflip,
    compute_concept_tags,
    dominant_is_aggressive,
    dominant_is_fold,
    dominant_is_passive,
    dominated,
    early_position,
    equity_dominant,
    equity_favorite,
    facing_3bet,
    facing_4bet_plus,
    facing_single_raise,
    hero_range_advantage,
    king_blocker,
    late_position,
    medium_pair,
    middle_position,
    mixed_strategy,
    multiway_pot,
    near_pure_strategy,
    open_decision,
    premium_pair,
    premium_unpaired,
    roughly_equal_ranges,
    small_blind,
    small_pair,
    squeeze_opportunity,
    standard_stack,
    suited_ace,
    suited_broadway,
    suited_connector,
    unconnected_offsuit,
    villain_range_advantage,
)
from pipeline.preflop.fact_extractor import PreflopFacts, VillainRangeStats  # noqa: E402
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


# --- fixture builders -------------------------------------------------------
def _node(
    actor: str = "BTN",
    history: tuple[ParsedAction, ...] = (),
) -> PreflopDecisionNode:
    return PreflopDecisionNode(
        pack_id="t", actor=actor, history_before=history, actions=()
    )


def _facts(
    *,
    actor: str = "BTN",
    history: tuple[ParsedAction, ...] = (),
    hand_class: str = "AKo",
    combo: str = "AhKc",
    action_freqs: dict[str, float] | None = None,
    dominant_action: str | None = None,
    dominant_frequency: float = 1.0,
    hero_equity: float | None = None,
    hero_range_equity: float | None = None,
    blockers: dict[str, int] | None = None,
    archetype: str = "open_for_value",
    villain_stats: VillainRangeStats | None = None,
) -> PreflopFacts:
    """Build a fully-specified PreflopFacts for tag-function testing."""
    action_freqs = action_freqs or {"Fold": 0.0, "Raise 60%": 1.0}
    dom = (
        dominant_action
        if dominant_action is not None
        else max(action_freqs.items(), key=lambda kv: kv[1])[0]
    )
    spot = PreflopSpot(
        node=_node(actor=actor, history=history),
        hero_hand_class=hand_class,
        hero_card_combo=combo,
        action_frequencies=action_freqs,
        dominant_action=dom,
        dominant_frequency=dominant_frequency,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=villain_stats,
        hero_equity_vs_villain=hero_equity,
        hero_range_equity_vs_villain=hero_range_equity,
        blockers=blockers or {},
        archetype=archetype,
    )


# --- Position tags ----------------------------------------------------------
def test_early_position_fires_for_utg() -> None:
    assert early_position(_facts(actor="UTG")) is True


def test_early_position_does_not_fire_for_btn() -> None:
    assert early_position(_facts(actor="BTN")) is False


def test_middle_position_fires_for_hj() -> None:
    assert middle_position(_facts(actor="HJ")) is True


def test_middle_position_fires_for_lj() -> None:
    assert middle_position(_facts(actor="LJ")) is True


def test_late_position_fires_for_co_and_btn() -> None:
    assert late_position(_facts(actor="CO")) is True
    assert late_position(_facts(actor="BTN")) is True


def test_small_blind_and_big_blind() -> None:
    assert small_blind(_facts(actor="SB")) is True
    assert big_blind(_facts(actor="BB")) is True
    assert small_blind(_facts(actor="BB")) is False
    assert big_blind(_facts(actor="SB")) is False


# --- Decision context tags --------------------------------------------------
def test_open_decision_no_history() -> None:
    """First-to-act spot (UTG with empty history) -> open_decision."""
    assert open_decision(_facts(actor="UTG", history=())) is True


def test_open_decision_after_folds() -> None:
    """Folds before hero don't count as raises -- still an open spot."""
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
    )
    assert open_decision(_facts(actor="BTN", history=history)) is True


def test_facing_single_raise_fires() -> None:
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
    )
    facts = _facts(actor="BB", history=history)
    assert facing_single_raise(facts) is True
    assert open_decision(facts) is False


def test_facing_single_raise_does_not_fire_with_caller() -> None:
    """An open + a caller before hero is a squeeze opportunity, not a
    plain facing-single-raise."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.CALL),
    )
    facts = _facts(actor="CO", history=history)
    assert facing_single_raise(facts) is False
    assert squeeze_opportunity(facts) is True


def test_facing_3bet_fires() -> None:
    history = (
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ParsedAction("BB", PreflopActionType.RAISE, 182.0),
    )
    facts = _facts(actor="BTN", history=history)
    assert facing_3bet(facts) is True
    assert facing_single_raise(facts) is False


def test_facing_4bet_plus_fires_for_three_raises() -> None:
    history = (
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ParsedAction("BB", PreflopActionType.RAISE, 182.0),
        ParsedAction("BTN", PreflopActionType.RAISE, 50.0),
    )
    facts = _facts(actor="BB", history=history)
    assert facing_4bet_plus(facts) is True


def test_squeeze_opportunity_open_plus_call() -> None:
    history = (
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
        ParsedAction("CO", PreflopActionType.CALL),
    )
    assert squeeze_opportunity(_facts(actor="BTN", history=history)) is True


def test_bvb_spot_sb_open_only_blinds_left() -> None:
    """UTG/HJ/CO/BTN all folded; SB acting first -> BvB."""
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.FOLD),
    )
    facts = _facts(actor="SB", history=history)
    assert bvb_spot(facts) is True


def test_bvb_spot_does_not_fire_when_non_blind_called() -> None:
    """If anyone non-blind called (not folded), it's no longer BvB."""
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
    )
    facts = _facts(actor="BB", history=history)
    assert bvb_spot(facts) is False


def test_multiway_pot_three_non_folds() -> None:
    """UTG opens, HJ calls -> hero (CO) makes 3 active players."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.CALL),
    )
    facts = _facts(actor="CO", history=history)
    assert multiway_pot(facts) is True


def test_multiway_pot_does_not_fire_heads_up() -> None:
    """Only one non-fold actor (the opener) + hero = 2 players, not multiway."""
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
    )
    facts = _facts(actor="BB", history=history)
    assert multiway_pot(facts) is False


# --- Hand strength tags -----------------------------------------------------
def test_premium_pair() -> None:
    assert premium_pair(_facts(hand_class="AA")) is True
    assert premium_pair(_facts(hand_class="QQ")) is True
    assert premium_pair(_facts(hand_class="JJ")) is False


def test_medium_pair() -> None:
    assert medium_pair(_facts(hand_class="TT")) is True
    assert medium_pair(_facts(hand_class="JJ")) is True
    assert medium_pair(_facts(hand_class="88")) is False


def test_small_pair() -> None:
    assert small_pair(_facts(hand_class="22")) is True
    assert small_pair(_facts(hand_class="88")) is True
    assert small_pair(_facts(hand_class="99")) is False


def test_premium_unpaired() -> None:
    assert premium_unpaired(_facts(hand_class="AKs")) is True
    assert premium_unpaired(_facts(hand_class="AKo")) is True
    assert premium_unpaired(_facts(hand_class="AQs")) is True
    assert premium_unpaired(_facts(hand_class="AQo")) is True
    assert premium_unpaired(_facts(hand_class="AJs")) is False
    assert premium_unpaired(_facts(hand_class="AA")) is False


def test_suited_broadway() -> None:
    assert suited_broadway(_facts(hand_class="KQs")) is True
    assert suited_broadway(_facts(hand_class="JTs")) is True
    assert suited_broadway(_facts(hand_class="KQo")) is False
    assert suited_broadway(_facts(hand_class="T9s")) is False  # T9 is connector


def test_suited_connector() -> None:
    """Suited consecutive ranks, 98s and below (above is broadway)."""
    assert suited_connector(_facts(hand_class="98s")) is True
    assert suited_connector(_facts(hand_class="76s")) is True
    assert suited_connector(_facts(hand_class="54s")) is True
    assert suited_connector(_facts(hand_class="JTs")) is False  # broadway
    assert suited_connector(_facts(hand_class="97s")) is False  # gapped
    assert suited_connector(_facts(hand_class="76o")) is False  # offsuit


def test_suited_ace() -> None:
    """A2s-AJs, NOT AKs/AQs."""
    assert suited_ace(_facts(hand_class="A2s")) is True
    assert suited_ace(_facts(hand_class="A5s")) is True
    assert suited_ace(_facts(hand_class="AJs")) is True
    assert suited_ace(_facts(hand_class="AKs")) is False  # premium_unpaired
    assert suited_ace(_facts(hand_class="AQs")) is False  # premium_unpaired


def test_unconnected_offsuit() -> None:
    """Offsuit hand with rank gap >= 2, not premium."""
    assert unconnected_offsuit(_facts(hand_class="J7o")) is True
    assert unconnected_offsuit(_facts(hand_class="K3o")) is True
    assert unconnected_offsuit(_facts(hand_class="AKo")) is False  # premium
    assert unconnected_offsuit(_facts(hand_class="76o")) is False  # connected
    assert unconnected_offsuit(_facts(hand_class="A5s")) is False  # suited


# --- Strategy shape tags ----------------------------------------------------
def test_mixed_strategy_55_to_95_pct() -> None:
    assert mixed_strategy(_facts(dominant_frequency=0.55)) is True
    assert mixed_strategy(_facts(dominant_frequency=0.75)) is True
    assert mixed_strategy(_facts(dominant_frequency=0.94)) is True
    assert mixed_strategy(_facts(dominant_frequency=0.95)) is False
    assert mixed_strategy(_facts(dominant_frequency=0.40)) is False


def test_near_pure_strategy_at_or_above_95() -> None:
    assert near_pure_strategy(_facts(dominant_frequency=0.95)) is True
    assert near_pure_strategy(_facts(dominant_frequency=1.00)) is True
    assert near_pure_strategy(_facts(dominant_frequency=0.94)) is False


def test_dominant_is_aggressive_for_raises_and_allin() -> None:
    assert (
        dominant_is_aggressive(
            _facts(dominant_action="Raise 60%", action_freqs={"Raise 60%": 1.0})
        )
        is True
    )
    assert (
        dominant_is_aggressive(
            _facts(dominant_action="AllIn", action_freqs={"AllIn": 1.0})
        )
        is True
    )
    assert (
        dominant_is_aggressive(
            _facts(dominant_action="Call", action_freqs={"Call": 1.0})
        )
        is False
    )


def test_dominant_is_passive_for_call() -> None:
    assert (
        dominant_is_passive(_facts(dominant_action="Call", action_freqs={"Call": 1.0}))
        is True
    )
    assert (
        dominant_is_passive(_facts(dominant_action="Fold", action_freqs={"Fold": 1.0}))
        is False
    )


def test_dominant_is_fold() -> None:
    assert (
        dominant_is_fold(_facts(dominant_action="Fold", action_freqs={"Fold": 1.0}))
        is True
    )


# --- Equity tags ------------------------------------------------------------
def test_equity_dominant_above_70() -> None:
    assert equity_dominant(_facts(hero_equity=0.75)) is True
    assert equity_dominant(_facts(hero_equity=0.70)) is False
    assert equity_dominant(_facts(hero_equity=None)) is False  # no equity data


def test_equity_favorite_55_to_70() -> None:
    assert equity_favorite(_facts(hero_equity=0.60)) is True
    assert equity_favorite(_facts(hero_equity=0.70)) is True
    assert equity_favorite(_facts(hero_equity=0.71)) is False
    assert equity_favorite(_facts(hero_equity=0.50)) is False


def test_coinflip_45_to_55() -> None:
    assert coinflip(_facts(hero_equity=0.50)) is True
    assert coinflip(_facts(hero_equity=0.45)) is True
    assert coinflip(_facts(hero_equity=0.55)) is False  # in equity_favorite


def test_dominated_below_35() -> None:
    assert dominated(_facts(hero_equity=0.30)) is True
    assert dominated(_facts(hero_equity=0.34)) is True
    assert dominated(_facts(hero_equity=0.35)) is False


# --- Blocker tags -----------------------------------------------------------
def test_ace_blocker_fires_when_hero_has_ace() -> None:
    assert ace_blocker(_facts(hand_class="AKo", combo="AhKc")) is True
    assert ace_blocker(_facts(hand_class="AA", combo="AcAd")) is True
    assert ace_blocker(_facts(hand_class="KQs", combo="KsQs")) is False


def test_king_blocker_fires_when_hero_has_king() -> None:
    assert king_blocker(_facts(hand_class="KQs", combo="KsQs")) is True
    assert king_blocker(_facts(hand_class="KK", combo="KcKd")) is True
    assert king_blocker(_facts(hand_class="QJs", combo="QsJs")) is False


def test_blocks_villain_top_value_fires_when_blockers_present() -> None:
    facts = _facts(blockers={"AA": 1, "AKs": 3})
    assert blocks_villain_top_value(facts) is True


def test_blocks_villain_top_value_does_not_fire_with_empty_dict() -> None:
    assert blocks_villain_top_value(_facts(blockers={})) is False


def test_blocks_villain_top_value_does_not_fire_with_all_zeros() -> None:
    """Defensive: a dict with all-zero counts still doesn't count as blocking."""
    assert blocks_villain_top_value(_facts(blockers={"AA": 0, "AKs": 0})) is False


# --- Range dynamics tags ----------------------------------------------------
def test_hero_range_advantage_at_or_above_53() -> None:
    assert hero_range_advantage(_facts(hero_range_equity=0.55)) is True
    assert hero_range_advantage(_facts(hero_range_equity=0.53)) is True
    assert hero_range_advantage(_facts(hero_range_equity=0.52)) is False


def test_villain_range_advantage_at_or_below_47() -> None:
    assert villain_range_advantage(_facts(hero_range_equity=0.45)) is True
    assert villain_range_advantage(_facts(hero_range_equity=0.47)) is True
    assert villain_range_advantage(_facts(hero_range_equity=0.48)) is False


def test_roughly_equal_ranges_47_to_53() -> None:
    assert roughly_equal_ranges(_facts(hero_range_equity=0.50)) is True
    assert roughly_equal_ranges(_facts(hero_range_equity=0.48)) is True
    assert roughly_equal_ranges(_facts(hero_range_equity=0.47)) is False  # boundary


# --- compute_concept_tags aggregator ---------------------------------------
def test_aggregator_returns_list_of_strings() -> None:
    """The aggregator returns a list[str] of firing tag names."""
    facts = _facts(actor="BTN", hand_class="AKs", combo="AsKs")
    tags = compute_concept_tags(facts)
    assert isinstance(tags, list)
    assert all(isinstance(t, str) for t in tags)


def test_aggregator_btn_aks_open_typical() -> None:
    """A typical BTN-opens-AKs spot fires expected tags."""
    facts = _facts(
        actor="BTN",
        history=(
            ParsedAction("UTG", PreflopActionType.FOLD),
            ParsedAction("HJ", PreflopActionType.FOLD),
            ParsedAction("CO", PreflopActionType.FOLD),
        ),
        hand_class="AKs",
        combo="AsKs",
        action_freqs={"Fold": 0.0, "Raise 60%": 1.0},
        dominant_action="Raise 60%",
        dominant_frequency=1.0,
    )
    tags = compute_concept_tags(facts)
    # Position.
    assert "late_position" in tags
    # Decision context.
    assert "open_decision" in tags
    # Hand strength.
    assert "premium_unpaired" in tags
    # Strategy.
    assert "near_pure_strategy" in tags
    assert "dominant_is_aggressive" in tags
    # Blockers.
    assert "ace_blocker" in tags
    assert "king_blocker" in tags
    # Standard stack (Ryan-pack v1 default).
    assert "standard_stack" in tags
    # And things that should NOT fire.
    assert "early_position" not in tags
    assert "facing_3bet" not in tags
    assert "small_pair" not in tags
    assert "dominant_is_fold" not in tags


def test_aggregator_mixed_strategy_bb_facing_open() -> None:
    """BB facing a BTN open with a borderline hand -> mixed strategy tags."""
    facts = _facts(
        actor="BB",
        history=(
            ParsedAction("UTG", PreflopActionType.FOLD),
            ParsedAction("HJ", PreflopActionType.FOLD),
            ParsedAction("CO", PreflopActionType.FOLD),
            ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ),
        hand_class="A5s",
        combo="AsKs",  # actually A5s but the combo doesn't matter for tags
        action_freqs={"Fold": 0.0, "Call": 0.66, "Raise 182%": 0.34},
        dominant_action="Call",
        dominant_frequency=0.66,
        hero_equity=0.42,
        hero_range_equity=0.49,
    )
    tags = compute_concept_tags(facts)
    assert "big_blind" in tags
    assert "facing_single_raise" in tags
    assert "mixed_strategy" in tags
    assert "dominant_is_passive" in tags
    assert "ace_blocker" in tags
    # Equity bucket -- 0.42 is in (dominated < 0.35 false, coinflip 0.45-0.55 false)
    # so neither equity bucket should fire.
    assert "dominated" not in tags
    assert "coinflip" not in tags
    # Range dynamics: 0.49 is in roughly_equal_ranges window.
    assert "roughly_equal_ranges" in tags


def test_aggregator_squeeze_spot() -> None:
    """Multiway with prior open + call -> squeeze_opportunity fires."""
    facts = _facts(
        actor="BTN",
        history=(
            ParsedAction("UTG", PreflopActionType.FOLD),
            ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
            ParsedAction("CO", PreflopActionType.CALL),
        ),
        hand_class="AKs",
        combo="AsKs",
    )
    tags = compute_concept_tags(facts)
    assert "squeeze_opportunity" in tags
    assert "multiway_pot" in tags
    assert "facing_single_raise" not in tags  # caller after raise prevents this
    assert "facing_3bet" not in tags  # only one prior raise


def test_aggregator_deterministic_same_input_same_output() -> None:
    """Pure functions -> identical input always yields identical output."""
    facts = _facts(actor="BTN", hand_class="AKs", combo="AsKs")
    tags_1 = compute_concept_tags(facts)
    tags_2 = compute_concept_tags(facts)
    assert tags_1 == tags_2


def test_standard_stack_always_fires_in_v1() -> None:
    """Ryan pack is 100bb; standard_stack always returns True. Short/deep
    are TODO for multi-stack packs."""
    assert standard_stack(_facts()) is True
