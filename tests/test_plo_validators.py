"""Tests for pipeline.plo.validators (the PLO deterministic audit stack)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import GeneratedExplanation  # noqa: E402
from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402
from pipeline.plo.validators import (  # noqa: E402
    run_plo_audit_validators,
    run_plo_soft_validators,
    soft_validate_position_words,
)

CARDS = ("As", "Ks", "Ah", "Kh")  # double-suited AAKK: no dangler


def _facts() -> PloFacts:
    node = PloDecisionNode(
        actor="HJ",
        history_before=(PloAction("LJ", PloActionType.RAISE, 100),),
        actions=(),
        history_stem="40100",
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=CARDS,
        action_frequencies={"Call": 0.7, "Raise 100%": 0.3},
        ev_by_action={"Call": 2.0, "Raise 100%": 1.0, "Fold": -7.0},
        presence=1.0,
    )
    return PloFacts(
        spot=spot,
        hand_class=classify_plo_hand(CARDS),
        archetype="3bet_for_value",
        villain_stats=PloVillainStats(
            seat="LJ", action_label="Raise 100%",
            weighted_combo_count=1.0, pct_of_dealt_hands=18.0,
        ),
    )


def _gen(prose: str) -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1="Always Call",
        option_2="Mostly Call",
        option_3="Mostly Fold",
        option_4="Always Fold",
        correct_answer="Mostly Call",
        answer_explanation=prose,
    )


# --- hard validators ---------------------------------------------------------
def test_clean_prose_passes():
    result = run_plo_audit_validators(
        _gen("Your double-suited hand plays well against the raise, so calling "
             "keeps the pot manageable while you keep every strong runout."),
        _facts(),
    )
    assert result.is_valid


def test_bulleted_list_is_rejected():
    result = run_plo_audit_validators(
        _gen("Calling is right. Here's why:\n- you dominate both suits\n- the "
             "price is good"),
        _facts(),
    )
    assert not result.is_valid
    assert "list" in result.error_message


def test_factor_list_prompts_sanction_bullets():
    """A factor-list system prompt legitimizes '- ' lines: the shared
    prompt_sanctions_lists helper detects it, and the audit runner skips the
    no-list rule when told so (July 2026 -- the rule briefly rejected the
    very format the factor-list prompts request, in both games)."""
    from pipeline.explanation_generator import prompt_sanctions_lists

    assert prompt_sanctions_lists("... Then the factor list: 2 to 5 lines ...")
    assert not prompt_sanctions_lists("Write flowing coaching prose.")
    assert not prompt_sanctions_lists(None)

    bulleted = _gen(
        "Mostly call. Here's why:\n- your double suits keep both flush "
        "draws live\n- the price is right\n\nCalling keeps the pot in range."
    )
    assert not run_plo_audit_validators(bulleted, _facts()).is_valid
    assert run_plo_audit_validators(
        bulleted, _facts(), allow_list_formatting=True
    ).is_valid


def test_inline_hyphens_and_numbers_are_not_lists():
    result = run_plo_audit_validators(
        _gen("You are getting 3-to-1 on the call and your AAKK-type hand "
             "plays on."),
        _facts(),
    )
    assert result.is_valid


def test_fabricated_card_is_rejected():
    # Hero holds As Ks Ah Kh -- a Q with any suit emoji is a fabrication.
    result = run_plo_audit_validators(
        _gen("Your Q♠ dominates the villain's suit, so call."), _facts()
    )
    assert not result.is_valid
    assert "not in your hand" in result.error_message


def test_made_set_claim_preflop_is_rejected():
    """The exact live miss (July 2026): 'Your hand is a set of threes' about
    a PAIR -- both gate passes missed it; now it's a deterministic reject."""
    result = run_plo_audit_validators(
        _gen("The best play is to 4-bet. Your hand is a set of threes with "
             "the nut diamond suit alongside."),
        _facts(),
    )
    assert not result.is_valid
    assert "set/trips stated preflop" in result.error_message


def test_set_mining_and_flop_a_set_language_stays_legal():
    result = run_plo_audit_validators(
        _gen("You are set-mining here, and flopping a set is your main way "
             "to win a big pot, so you keep your set outs alive by calling."),
        _facts(),
    )
    assert result.is_valid


def test_false_shape_claim_is_rejected():
    # AAKK double-suited has NO dangler; asserting one is a shape fabrication.
    result = run_plo_audit_validators(
        _gen("You should call even though your hand has a dangler dragging "
             "it down."),
        _facts(),
    )
    assert not result.is_valid
    assert "misstates the hand" in result.error_message


# --- soft position validator --------------------------------------------------
def test_hero_bound_position_reversal_is_flagged():
    # HJ acts after LJ postflop -> hero is IN position; the prose says the
    # opposite about hero.
    warnings = soft_validate_position_words(
        _gen("Calling is fine, but you will be out of position for the rest "
             "of the hand."),
        _facts(),
    )
    assert warnings and "In Position" in warnings[0]


def test_villain_bound_position_phrase_does_not_flag():
    warnings = soft_validate_position_words(
        _gen("The lojack opened and will be out of position against you "
             "postflop."),
        _facts(),
    )
    assert warnings == []


def test_correct_position_wording_does_not_flag():
    warnings = run_plo_soft_validators(
        _gen("You act after the opener on every street, so you are in "
             "position with a premium double-suited hand."),
        _facts(),
    )
    assert warnings == []
