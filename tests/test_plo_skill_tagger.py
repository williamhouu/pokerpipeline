"""Tests for pipeline.plo.skill_tagger (preflop PLO skill mapping)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloActionOption, PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.skill_tagger import (  # noqa: E402
    SKILL_CATALOG,
    SKILL_CATEGORIES,
    SKILL_META,
    compute_plo_skills,
)
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
    with_villain: bool = False,
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
    villain = (
        PloVillainStats(
            seat="LJ",
            action_label="Raise 100%",
            weighted_combo_count=1.0,
            pct_of_dealt_hands=18.0,
        )
        if with_villain
        else None
    )
    return PloFacts(
        spot=spot,
        hand_class=classify_plo_hand(hero_cards),
        archetype=archetype,
        villain_stats=villain,
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


def test_implied_odds_fires_on_speculative_call_and_pairs_with_reverse():
    # A well-shaped speculative call fires Implied Odds, NOT Reverse.
    good = compute_plo_skills(
        _facts(
            archetype="call_for_implied_odds",
            hero_cards=DS_RUNDOWN,
            history=(_act("LJ", R, 100),),
            freqs={"Call": 1.0},
        )
    )
    assert "Implied Odds" in good
    assert "Reverse Implied Odds" not in good
    # A trappy (non-nut) speculative call fires BOTH lenses, like NLHE.
    trappy = compute_plo_skills(
        _facts(
            archetype="call_for_implied_odds",
            hero_cards=BARE_ACE,
            history=(_act("LJ", R, 100),),
            freqs={"Call": 1.0},
        )
    )
    assert "Implied Odds" in trappy
    assert "Reverse Implied Odds" in trappy


def test_nut_blockers_fire_on_aggressive_spot_with_a_blocker():
    # 3-betting AAxx (an ace blocks AA, a suited ace blocks the nut flush)
    # vs a villain -> Nut Blockers fires.
    assert "Nut Blockers & Card Removal" in compute_plo_skills(
        _facts(
            archetype="3bet_for_value",
            hero_cards=AAKK_DS,
            history=(_act("LJ", R, 100),),
            freqs={"Raise 100%": 1.0},
            with_villain=True,
        )
    )
    # An open (no villain to block) -> does NOT fire.
    assert "Nut Blockers & Card Removal" not in compute_plo_skills(
        _facts(archetype="open_for_value", hero_cards=AAKK_DS, actor="LJ")
    )
    # A passive call (not aggressive) with the same blocker -> does NOT fire.
    assert "Nut Blockers & Card Removal" not in compute_plo_skills(
        _facts(
            archetype="call_for_value",
            hero_cards=AAKK_DS,
            history=(_act("LJ", R, 100),),
            freqs={"Call": 1.0},
            with_villain=True,
        )
    )


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


def test_skill_meta_covers_catalog_with_valid_categories():
    # Every catalog skill has display metadata, and there are no orphans.
    assert set(SKILL_META) == set(SKILL_CATALOG)
    for name, m in SKILL_META.items():
        assert m.description, name
        assert m.category in SKILL_CATEGORIES, name
    # Pot-Limit Bet Sizing is the one dormant (single-raise-size pack) skill.
    assert SKILL_META["Pot-Limit Bet Sizing"].fires is False
    assert all(
        m.fires for name, m in SKILL_META.items() if name != "Pot-Limit Bet Sizing"
    )
