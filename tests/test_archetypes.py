"""Tests for pipeline.fact_extractor.archetypes (Ryan-feedback Fix 5).

Pure unit tests on the archetype classifier and the aggression-history helper.
No PioSolver, no LLM. Coverage:

  * each of the 13 archetypes returns from its triggering combination of
    (correct_action, hand_strength_bucket, facing_bet, hero_was_aggressor);
  * aggression_history_from_action_sequence handles empty / single-street /
    multi-street paths and correctly resolves OOP/IP -> hero/villain;
  * the V7.1 failure case (set on later street, hero had the betting lead,
    recommended check) classifies as trap_check.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.archetypes import (                            # noqa: E402
    ARCHETYPE_GUIDANCE, RECOMMENDED_ACTION_ARCHETYPES,
    aggression_history_from_action_sequence, classify_recommended_archetype,
)
from pipeline.fact_extractor.spot_data import (                             # noqa: E402
    DecisionData, HandClass, SpotData, SpotMetadata,
)


def _spot(*, correct_action, bucket="", draws=None, facing_bet_frac=None,
          aggression_history=(), street="turn"):
    """Minimal SpotData for archetype classification tests."""
    meta = SpotMetadata(street=street, hero_position="BB",
                        villain_position="BTN",
                        position_dynamic="BB_vs_BTN",
                        hero_cards=("Ah", "Kh"), board=["2c", "Js", "7s"])
    object.__setattr__(meta, "aggression_history", list(aggression_history))
    decision = DecisionData(
        correct_action=correct_action,
        range_aggregate_strategy={correct_action: 1.0},
        facing_bet_pot_fraction=facing_bet_frac,
    )
    hand_class = HandClass(made_hand="x", draws=list(draws or []),
                           strength_bucket=bucket, label="x")
    return SpotData(spot_metadata=meta, decision_data=decision,
                    hand_class=hand_class)


# --- catalog sanity ---------------------------------------------------------
def test_thirteen_archetypes_in_catalog():
    assert len(RECOMMENDED_ACTION_ARCHETYPES) == 13
    # Every archetype has a guidance paragraph in the prompt-side catalog.
    for arch in RECOMMENDED_ACTION_ARCHETYPES:
        assert arch in ARCHETYPE_GUIDANCE
        assert ARCHETYPE_GUIDANCE[arch].strip()


# --- bet/raise classification ----------------------------------------------
def test_value_bet_when_strong_hand_bets():
    assert classify_recommended_archetype(
        _spot(correct_action="bet", bucket="strong")) == "value_bet"
    assert classify_recommended_archetype(
        _spot(correct_action="bet", bucket="premium")) == "value_bet"


def test_bluff_when_air_bets():
    assert classify_recommended_archetype(
        _spot(correct_action="bet", bucket="air")) == "bluff"
    # Semi-bluff (air + draws) still classifies as bluff for the frame.
    assert classify_recommended_archetype(
        _spot(correct_action="bet", bucket="air",
              draws=["flush_draw_weak"])) == "bluff"


def test_protection_bet_when_medium_bets():
    assert classify_recommended_archetype(
        _spot(correct_action="bet", bucket="medium")) == "protection_bet"
    assert classify_recommended_archetype(
        _spot(correct_action="bet", bucket="vulnerable")) == "protection_bet"


def test_value_raise_when_strong_hand_raises_facing_bet():
    assert classify_recommended_archetype(
        _spot(correct_action="raise", bucket="strong",
              facing_bet_frac=0.65)) == "value_raise"
    assert classify_recommended_archetype(
        _spot(correct_action="raise", bucket="premium",
              facing_bet_frac=1.0)) == "value_raise"


def test_bluff_raise_when_air_raises_facing_bet():
    assert classify_recommended_archetype(
        _spot(correct_action="raise", bucket="air",
              facing_bet_frac=0.5)) == "bluff_raise"


# --- check classification ---------------------------------------------------
def test_trap_check_strong_hand_with_aggressor_history():
    """The canonical V7.1 failure case: strong hand on a later street, hero
    led prior, recommended action is check. Must classify as trap_check
    (not pot_control_check, not defensive_check).
    """
    spot = _spot(correct_action="check", bucket="premium",
                 aggression_history=["hero"], street="turn")
    assert classify_recommended_archetype(spot) == "trap_check"
    # Strong-bucket also qualifies.
    spot2 = _spot(correct_action="check", bucket="strong",
                  aggression_history=["hero", "hero"], street="river")
    assert classify_recommended_archetype(spot2) == "trap_check"


def test_pot_control_check_strong_hand_no_aggressor_history():
    """Strong hand checking WITHOUT having had the betting lead -> pot
    control (not trap, since hero never initiated)."""
    spot = _spot(correct_action="check", bucket="strong",
                 aggression_history=["villain"], street="turn")
    assert classify_recommended_archetype(spot) == "pot_control_check"


def test_pot_control_check_medium_hand():
    assert classify_recommended_archetype(
        _spot(correct_action="check", bucket="medium")) == "pot_control_check"


def test_defensive_check_weak_hand():
    assert classify_recommended_archetype(
        _spot(correct_action="check", bucket="marginal")) == "defensive_check"
    assert classify_recommended_archetype(
        _spot(correct_action="check", bucket="air")) == "defensive_check"


# --- call classification ----------------------------------------------------
def test_bluff_catch_when_made_hand_calls_facing_bet():
    assert classify_recommended_archetype(
        _spot(correct_action="call", bucket="strong",
              facing_bet_frac=1.5)) == "bluff_catch"
    assert classify_recommended_archetype(
        _spot(correct_action="call", bucket="medium",
              facing_bet_frac=0.65)) == "bluff_catch"


def test_call_drawing_when_air_with_draws_calls():
    assert classify_recommended_archetype(
        _spot(correct_action="call", bucket="air",
              draws=["flush_draw_weak"],
              facing_bet_frac=0.5)) == "call_drawing"


def test_call_with_showdown_when_not_facing_bet():
    assert classify_recommended_archetype(
        _spot(correct_action="call", bucket="medium",
              facing_bet_frac=None)) == "call_with_showdown"


# --- fold classification ----------------------------------------------------
def test_fold_to_polar():
    assert classify_recommended_archetype(
        _spot(correct_action="fold", bucket="medium",
              facing_bet_frac=1.5)) == "fold_to_polar"


# --- aggression_history -----------------------------------------------------
def test_aggression_history_empty_when_only_one_segment():
    # No 'deal' markers -> just the current street segment, no priors.
    seq = [("OOP", "check"), ("IP", "bet 30")]
    assert aggression_history_from_action_sequence(seq, hero_is_oop=True) == []


def test_aggression_history_one_entry_per_prior_street():
    # Flop: OOP led; turn: IP raised over a bet; river: current decision.
    seq = [
        ("OOP", "bet 30"), ("IP", "call"),
        ("deal", "Th"),
        ("OOP", "check"), ("IP", "bet 50"), ("OOP", "raise 150"), ("IP", "call"),
        ("deal", "9c"),
        ("OOP", "check"),                    # current decision
    ]
    # hero_is_oop=True -> OOP=hero, IP=villain.
    history = aggression_history_from_action_sequence(seq, hero_is_oop=True)
    # 2 prior streets (flop + turn); current river excluded.
    assert len(history) == 2
    # Flop: OOP=hero bet, IP=villain just called -> hero was last aggressor.
    assert history[0] == "hero"
    # Turn: OOP=hero raised last (over IP's bet) -> hero was last aggressor.
    assert history[1] == "hero"


def test_aggression_history_check_when_no_bets_on_street():
    seq = [
        ("OOP", "check"), ("IP", "check"),
        ("deal", "Th"),
        ("OOP", "bet 30"),                   # current decision
    ]
    history = aggression_history_from_action_sequence(seq, hero_is_oop=True)
    assert history == ["check"]


def test_aggression_history_oop_ip_swap_when_hero_is_ip():
    # Same action sequence, but hero is IP. OOP bets, IP calls -> villain
    # is the aggressor (OOP labeled as villain because hero is IP).
    seq = [
        ("OOP", "bet 30"), ("IP", "call"),
        ("deal", "Th"),
        ("OOP", "check"),                    # current decision
    ]
    history = aggression_history_from_action_sequence(seq, hero_is_oop=False)
    assert history == ["villain"]
