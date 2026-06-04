"""Tests for pipeline.plo.skill_tagger (preflop PLO skill mapping)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.fact_extractor import PloFacts  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloActionOption, PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.skill_tagger import compute_plo_skills  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

F = PloActionType.FOLD
C = PloActionType.CALL
R = PloActionType.RAISE

AAKK_DS = ("As", "Ks", "Ah", "Kh")  # pocket aces, double-suited, suited ace
DS_RUNDOWN = ("Ts", "9s", "8h", "7h")  # double-suited unpaired rundown, no ace
BARE_ACE = ("Ac", "5d", "8h", "Js")  # an ace with no nut-flush / connection
LOW_CARDS = ("2c", "4d", "5h", "6s")  # all low -> non-nut everything


def _act(seat: str, atype: PloActionType, pct: int | None = None) -> PloAction:
    return PloAction(seat=seat, action=atype, raise_pct=pct)


def _facts(
    *,
    archetype: str,
    hero_cards: tuple[str, str, str, str],
    history: tuple[PloAction, ...] = (),
    actor: str = "HJ",
    freqs: dict[str, float] | None = None,
    actions: tuple[PloActionOption, ...] = (),
) -> PloFacts:
    node = PloDecisionNode(
        actor=actor, history_before=history, actions=actions, history_stem=""
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=hero_cards,
        action_frequencies=freqs or {"Raise 100%": 1.0},
        presence=1.0,
    )
    return PloFacts(
        spot=spot, hand_class=classify_plo_hand(hero_cards), archetype=archetype
    )


# --- carry-over decision skills -------------------------------------------
def test_open_maps_to_hand_selection():
    skills = compute_plo_skills(_facts(archetype="open_for_value", hero_cards=AAKK_DS, actor="LJ"))
    assert "Preflop Hand Selection" in skills


def test_3bet_and_facing_skills():
    threebet = compute_plo_skills(
        _facts(archetype="3bet_for_value", hero_cards=AAKK_DS, history=(_act("LJ", R, 100),))
    )
    assert "3-Betting" in threebet

    facing = compute_plo_skills(
        _facts(
            archetype="call_for_value",
            hero_cards=AAKK_DS,
            history=(_act("LJ", R, 100), _act("HJ", R, 100)),
            actor="LJ",
            freqs={"Call": 1.0},
        )
    )
    assert "Facing a 3-Bet" in facing


def test_squeeze_and_facing_squeeze():
    squeeze = compute_plo_skills(
        _facts(
            archetype="squeeze_for_value",
            hero_cards=AAKK_DS,
            history=(_act("LJ", R, 100), _act("HJ", C)),
            actor="CO",
        )
    )
    assert "Squeezing" in squeeze

    facing_sq = compute_plo_skills(
        _facts(
            archetype="call_for_value",
            hero_cards=AAKK_DS,
            history=(_act("LJ", R, 100), _act("HJ", C), _act("CO", R, 100)),
            actor="LJ",
            freqs={"Call": 1.0},
        )
    )
    assert "Facing a Squeeze" in facing_sq


def test_blind_and_position_skills():
    bb = compute_plo_skills(
        _facts(
            archetype="call_for_value",
            hero_cards=AAKK_DS,
            history=(_act("LJ", R, 100),),
            actor="BB",
            freqs={"Call": 1.0},
        )
    )
    assert "Blind Defense" in bb
    assert "Out of Position Play" in bb

    bu = compute_plo_skills(
        _facts(
            archetype="call_for_value",
            hero_cards=AAKK_DS,
            history=(_act("LJ", R, 100),),
            actor="BU",
            freqs={"Call": 1.0},
        )
    )
    assert "In Position Play" in bu


def test_pot_odds_on_call_and_fold():
    assert "Pot Odds" in compute_plo_skills(
        _facts(archetype="call_for_value", hero_cards=AAKK_DS, history=(_act("LJ", R, 100),), freqs={"Call": 1.0})
    )
    assert "Pot Odds" in compute_plo_skills(
        _facts(archetype="fold_dominated", hero_cards=LOW_CARDS, history=(_act("LJ", R, 100),), freqs={"Fold": 1.0})
    )


# --- PLO hand-reading skills (the edge) -----------------------------------
def test_big_pair_construction_not_suitedness():
    # AAxx double-suited: the lesson is Big-Pair Construction, not Suitedness.
    skills = compute_plo_skills(
        _facts(archetype="3bet_for_value", hero_cards=AAKK_DS, history=(_act("LJ", R, 100),))
    )
    assert "Big-Pair Construction" in skills
    assert "Suitedness" not in skills  # excluded for big pairs
    assert "Nut-Flush Awareness" in skills  # As-Ks share a suit -> nut-flush


def test_suitedness_and_rundowns_on_a_ds_rundown():
    skills = compute_plo_skills(
        _facts(archetype="3bet_for_value", hero_cards=DS_RUNDOWN, history=(_act("LJ", R, 100),))
    )
    assert "Suitedness" in skills  # double-suited, no pair
    assert "Rundowns & Connectivity" in skills


def test_nuttedness_on_bare_ace_and_low_cards():
    assert "Nuttedness & Non-Nut Traps" in compute_plo_skills(
        _facts(archetype="call_for_implied_odds", hero_cards=BARE_ACE, history=(_act("LJ", R, 100),), freqs={"Call": 1.0})
    )
    assert "Nuttedness & Non-Nut Traps" in compute_plo_skills(
        _facts(archetype="call_for_value", hero_cards=LOW_CARDS, history=(_act("LJ", R, 100),), freqs={"Call": 1.0})
    )


def test_reverse_implied_odds_on_speculative_nonnut_call():
    skills = compute_plo_skills(
        _facts(
            archetype="call_for_implied_odds",
            hero_cards=BARE_ACE,
            history=(_act("LJ", R, 100),),
            freqs={"Call": 1.0},
        )
    )
    assert "Reverse Implied Odds" in skills


# --- strictness -----------------------------------------------------------
def test_hand_reading_skills_gated_on_continuing():
    # A folded double-suited rundown is NOT testing suitedness / connectivity.
    skills = compute_plo_skills(
        _facts(
            archetype="fold_dominated",
            hero_cards=DS_RUNDOWN,
            history=(_act("LJ", R, 100), _act("HJ", R, 100)),
            freqs={"Fold": 1.0},
        )
    )
    assert "Suitedness" not in skills
    assert "Rundowns & Connectivity" not in skills


def test_typical_question_yields_a_handful_of_skills():
    skills = compute_plo_skills(
        _facts(archetype="3bet_for_value", hero_cards=DS_RUNDOWN, history=(_act("LJ", R, 100),))
    )
    assert 2 <= len(skills) <= 5  # noqa: PLR2004  # strict, not noisy


def test_pot_limit_bet_sizing_needs_multiple_sizes():
    # Single pot-size tree -> off; a multi-size node -> on.
    one_size = (PloActionOption(_act("HE", R, 100), Path("a.rng")),)
    two_sizes = (
        PloActionOption(_act("HE", R, 50), Path("a.rng")),
        PloActionOption(_act("HE", R, 100), Path("b.rng")),
    )
    assert "Pot-Limit Bet Sizing" not in compute_plo_skills(
        _facts(archetype="3bet_for_value", hero_cards=AAKK_DS, history=(_act("LJ", R, 100),), actions=one_size)
    )
    assert "Pot-Limit Bet Sizing" in compute_plo_skills(
        _facts(archetype="3bet_for_value", hero_cards=AAKK_DS, history=(_act("LJ", R, 100),), actions=two_sizes)
    )
