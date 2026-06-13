"""Tests for pipeline.preflop.validators.

Per-validator positive (returns ok) and negative (returns fail with a
useful error message) cases for each of the v1 hard validators, plus a
runner test that confirms first-failure short-circuiting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import (  # noqa: E402
    GeneratedExplanation,
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
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402
from pipeline.preflop.validators import (  # noqa: E402
    PreflopValidationResult,
    run_preflop_audit_validators,
    run_preflop_soft_validators,
    soft_validate_position_words,
    validate_banned_phrases,
    validate_blocker_claims,
    validate_card_suit_consistency,
    validate_composite_label_frequencies,
    validate_no_postflop_on_allin,
    validate_no_standalone_sometimes,
    validate_option_set,
    validate_terminology,
)


# --- fixtures -------------------------------------------------------------
# Default history: folded to BTN who opens, SB folds, hero BB decides.
_DEFAULT_HISTORY = (
    ParsedAction("UTG", PreflopActionType.FOLD),
    ParsedAction("HJ", PreflopActionType.FOLD),
    ParsedAction("CO", PreflopActionType.FOLD),
    ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
    ParsedAction("SB", PreflopActionType.FOLD),
)

# 9-max-style cold 3-bet line: UTG opens, UTG+2 3-bets, folds to hero BB.
# No caller anywhere -- the squeeze/cold-call terminology tests' canvas.
_HIST_UTG_OPEN_UTG2_3BET = (
    ParsedAction("UTG", PreflopActionType.RAISE, 120.0),
    ParsedAction("UTG+1", PreflopActionType.FOLD),
    ParsedAction("UTG+2", PreflopActionType.RAISE, 84.0),
    ParsedAction("LJ", PreflopActionType.FOLD),
    ParsedAction("HJ", PreflopActionType.FOLD),
    ParsedAction("CO", PreflopActionType.FOLD),
    ParsedAction("BTN", PreflopActionType.FOLD),
    ParsedAction("SB", PreflopActionType.FOLD),
)


def _facts(
    *,
    action_frequencies: dict[str, float] | None = None,
    dominant_action: str = "Call",
    dominant_frequency: float = 0.66,
    actor: str = "BB",
    archetype: str = "call_for_value",
    history: tuple[ParsedAction, ...] | None = None,
    hero_card_combo: str = "AhKc",
    blockers: dict[str, int] | None = None,
    villain_position: str | None = "BTN",
    villain_top: tuple[tuple[str, float], ...] = (),
) -> PreflopFacts:
    """Minimal PreflopFacts. Default: BB facing a BTN open, 66/34 call/fold.

    The round-2 validator tests vary the history, hero combo, blockers
    fact, and villain identity; ``villain_position=None`` makes an open
    spot (no villain, empty blockers).
    """
    if action_frequencies is None:
        action_frequencies = {"Call": 0.66, "Fold": 0.34}
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor=actor,
            history_before=(
                history if history is not None else _DEFAULT_HISTORY
            ),
            actions=(),
        ),
        hero_hand_class="AKo",
        hero_card_combo=hero_card_combo,
        action_frequencies=action_frequencies,
        dominant_action=dominant_action,
        dominant_frequency=dominant_frequency,
    )
    villain = (
        VillainRangeStats(
            position=villain_position, action_label="Raise 60%",
            weighted_combo_count=600.0, pct_of_dealt_hands=45.0,
            top_combos=villain_top,
        )
        if villain_position is not None
        else None
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=villain,
        hero_equity_vs_villain=0.55,
        archetype=archetype,
        blockers=blockers or {},
    )


def _gen(
    *,
    options: tuple[str, str, str, str] = ("Fold", "Call", "", ""),
    correct: str = "Call",
    prose: str = "We have enough equity for the price.",
) -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1=options[0],
        option_2=options[1],
        option_3=options[2],
        option_4=options[3],
        correct_answer=correct,
        answer_explanation=prose,
    )


# --- validate_option_set --------------------------------------------------
def test_option_set_ok_on_call_fold() -> None:
    result = validate_option_set(_gen(), _facts())
    assert result.is_valid, result.error_message


def test_option_set_fails_on_invented_raise() -> None:
    """LLM proposes a Raise option on a pure call/fold spot."""
    generated = _gen(options=("Fold", "Call", "Raise 3x", ""), correct="Call")
    result = validate_option_set(generated, _facts())
    assert not result.is_valid
    assert "raise" in result.error_message.lower()


def test_option_set_skips_when_no_strategy() -> None:
    """Spots with empty action_frequencies (defensive) -> pass."""
    facts = _facts(action_frequencies={})
    result = validate_option_set(_gen(), facts)
    assert result.is_valid


def test_option_set_ok_with_frequency_prefix() -> None:
    generated = _gen(options=("Always fold", "", "", ""), correct="Always fold")
    facts = _facts(action_frequencies={"Fold": 1.0}, dominant_action="Fold",
                   dominant_frequency=1.0)
    result = validate_option_set(generated, facts)
    assert result.is_valid, result.error_message


# --- validate_no_standalone_sometimes -------------------------------------
def test_standalone_sometimes_fails() -> None:
    generated = _gen(options=("Sometimes call", "Sometimes fold", "", ""),
                     correct="Sometimes call")
    result = validate_no_standalone_sometimes(generated, _facts())
    assert not result.is_valid
    assert "sometimes" in result.error_message.lower()


def test_standalone_rarely_fails() -> None:
    generated = _gen(options=("Rarely raise", "Mostly call", "", ""),
                     correct="Mostly call")
    result = validate_no_standalone_sometimes(generated, _facts())
    assert not result.is_valid


def test_composite_label_passes_standalone_check() -> None:
    """'Mostly call, sometimes raise' has Mostly as the LEADING word --
    the standalone check should pass it."""
    generated = _gen(
        options=("Mostly call, sometimes raise", "Fold", "", ""),
        correct="Mostly call, sometimes raise",
    )
    result = validate_no_standalone_sometimes(generated, _facts())
    assert result.is_valid, result.error_message


# --- validate_composite_label_frequencies --------------------------------
def test_composite_frequencies_ok_when_mix_real() -> None:
    """75% call / 20% raise -> 'Mostly call, sometimes raise' is honest."""
    facts = _facts(
        action_frequencies={"Call": 0.75, "Raise 3x": 0.20, "Fold": 0.05},
        dominant_action="Call", dominant_frequency=0.75,
    )
    generated = _gen(
        options=("Fold", "Mostly call, sometimes raise", "", ""),
        correct="Mostly call, sometimes raise",
    )
    result = validate_composite_label_frequencies(generated, facts)
    assert result.is_valid, result.error_message


def test_composite_frequencies_fails_when_secondary_is_noise() -> None:
    """98% call / 2% raise -> labelling it 'Mostly call, sometimes raise'
    promotes a 2% mix-in to a teaching point. Fail."""
    facts = _facts(
        action_frequencies={"Call": 0.98, "Raise 3x": 0.02},
        dominant_action="Call", dominant_frequency=0.98,
    )
    generated = _gen(
        options=("Fold", "Mostly call, sometimes raise", "", ""),
        correct="Mostly call, sometimes raise",
    )
    result = validate_composite_label_frequencies(generated, facts)
    assert not result.is_valid
    assert "secondary verb" in result.error_message.lower() \
        or "raise" in result.error_message.lower()


def test_composite_frequencies_passes_4bet_at_4bet_spot() -> None:
    """Regression for the multi-raise alias bug. BB facing BTN open +
    SB 3-bet would 4-bet at raise_level=3. canonicalize_strategy
    relabels Pio's raise as '4-bet'. The LLM correctly says 'Mostly
    4-bet, sometimes Fold' -- the validator must resolve '4-bet' to
    the same internal 'raise' bucket the strategy is stored under,
    not look up '4-bet' literally and find 0%."""
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ParsedAction("SB", PreflopActionType.RAISE, 150.0),  # 3-bet by SB
    )
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor="BB",
            history_before=history, actions=(),
        ),
        hero_hand_class="99",
        hero_card_combo="9h9d",
        action_frequencies={"Raise 5x": 0.7, "Fold": 0.244, "Call": 0.056},
        dominant_action="Raise 5x",
        dominant_frequency=0.7,
    )
    facts = PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="SB", action_label="Raise 150%",
            weighted_combo_count=80.0, pct_of_dealt_hands=6.0,
            top_combos=(),
        ),
        hero_equity_vs_villain=0.42,
        archetype="4bet_for_value",
    )
    # Hero's would-be raise is the 3rd raise = "4-bet" via canonicalize.
    generated = _gen(
        options=("Fold", "Mostly 4-bet, sometimes Fold", "", ""),
        correct="Mostly 4-bet, sometimes Fold",
    )
    result = validate_composite_label_frequencies(generated, facts)
    assert result.is_valid, (
        f"validator should accept '4-bet' as the LLM verb when the "
        f"canonical strategy is stored under 'raise': {result.error_message}"
    )


def test_composite_frequencies_fails_when_primary_smaller() -> None:
    """Primary verb should have HIGHER freq than secondary."""
    facts = _facts(
        action_frequencies={"Call": 0.3, "Raise 3x": 0.6, "Fold": 0.1},
        dominant_action="Raise 3x", dominant_frequency=0.6,
    )
    generated = _gen(
        options=("Fold", "Mostly call, sometimes raise", "", ""),
        correct="Mostly call, sometimes raise",
    )
    result = validate_composite_label_frequencies(generated, facts)
    assert not result.is_valid


# --- validate_banned_phrases ----------------------------------------------
def test_banned_phrases_fails_on_em_dash() -> None:
    generated = _gen(prose="We have a decent hand — call here.")
    result = validate_banned_phrases(generated, _facts())
    assert not result.is_valid
    assert "em dash" in result.error_message.lower() \
        or "punctuation" in result.error_message.lower()


def test_banned_phrases_fails_on_semicolon() -> None:
    generated = _gen(prose="AK is dominant; we 3-bet for value.")
    result = validate_banned_phrases(generated, _facts())
    assert not result.is_valid


def test_banned_phrases_passes_clean_prose() -> None:
    generated = _gen(
        prose="AKo plays well as a 3-bet here. We have card removal "
              "blockers and dominate villain's weaker calls.",
    )
    result = validate_banned_phrases(generated, _facts())
    assert result.is_valid, result.error_message


# --- validate_no_postflop_on_allin ----------------------------------------
def test_no_postflop_on_allin_fails_on_implied_odds() -> None:
    """On an all-in spot, implied-odds / postflop framing is rejected."""
    facts = _facts(archetype="call_allin")
    generated = _gen(
        prose="The real reason to call is implied odds: you can chase flushes "
              "and stack a caller on later streets.")
    result = validate_no_postflop_on_allin(generated, facts)
    assert not result.is_valid
    assert "implied odds" in result.error_message.lower()


def test_no_postflop_on_allin_passes_pot_odds_prose() -> None:
    """An all-in spot framed around pot odds + showdown equity passes -- even
    mentioning flush/straight outs (legit showdown equity on the runout)."""
    facts = _facts(archetype="call_allin")
    generated = _gen(
        prose="You need about 23% equity and you have 40% against the shoving "
              "range, so the price is right. Your flush and straight outs add "
              "to your equity on the runout.")
    result = validate_no_postflop_on_allin(generated, facts)
    assert result.is_valid, result.error_message


def test_no_postflop_on_allin_skips_non_allin_spots() -> None:
    """On a normal (non-all-in) call, 'implied odds' is fine -> validator skips."""
    facts = _facts(archetype="call_for_implied_odds")  # facing a raise, not a jam
    generated = _gen(
        prose="The reason to call is implied odds with this speculative hand.")
    result = validate_no_postflop_on_allin(generated, facts)
    assert result.is_valid, result.error_message


# --- run_preflop_audit_validators ----------------------------------------
def test_runner_returns_ok_when_all_pass() -> None:
    """A clean explanation passes the full stack."""
    generated = _gen(
        options=("Fold", "Call", "", ""),
        correct="Call",
        prose="We have enough equity to defend with this suited connector.",
    )
    result = run_preflop_audit_validators(generated, _facts())
    assert result.is_valid, result.error_message


def test_runner_short_circuits_on_first_failure() -> None:
    """Order matters: option_set is first, so an invented option
    fails BEFORE any banned-phrase check would catch the em dash too."""
    generated = _gen(
        options=("Fold", "Call", "Raise 5x", ""),  # invented
        correct="Call",
        prose="Plenty of equity — we call.",  # also em dash
    )
    result = run_preflop_audit_validators(generated, _facts())
    assert not result.is_valid
    # The option-set message wins (first in the chain).
    assert "raise" in result.error_message.lower() or "doesn't offer" in result.error_message.lower()


def test_runner_allows_playability_talk() -> None:
    """The postflop-keyword check was removed: prose that mentions
    postflop playability ("on most flops", "guessing on runouts") is no
    longer rejected when the options + punctuation are clean."""
    generated = _gen(
        options=("Fold", "Call", "", ""),
        correct="Call",
        prose="With AK we have great equity and good playability on most flops.",
    )
    result = run_preflop_audit_validators(generated, _facts())
    assert result.is_valid, result.error_message


# --- ValidationResult sanity ---------------------------------------------
def test_validation_result_ok_factory() -> None:
    result = PreflopValidationResult.ok()
    assert result.is_valid
    assert result.error_message == ""


def test_validation_result_fail_factory() -> None:
    result = PreflopValidationResult.fail("oops")
    assert not result.is_valid
    assert result.error_message == "oops"


# --- validate_blocker_claims (June 2026 round-2 audit) ---------------------
def test_blocker_claim_outside_fact_fails() -> None:
    """The flagship round-2 defect: 'your queens block KK and AA' with no
    ace or king in hand. Both impossible hands must be named in the error."""
    facts = _facts(
        hero_card_combo="QcQd",
        blockers={"QQ": 5, "KQs": 2, "AQs": 2},
    )
    generated = _gen(
        prose="4-betting works because your queens block KK and AA combos."
    )
    result = validate_blocker_claims(generated, facts)
    assert not result.is_valid
    assert "AA" in result.error_message
    assert "KK" in result.error_message


def test_blocker_claim_licensed_passes() -> None:
    """Bare 'AK' is satisfied by the AKo/AKs entries in the fact."""
    facts = _facts(
        hero_card_combo="KsTs",
        blockers={"KK": 3, "AKo": 3, "AKs": 1},
    )
    generated = _gen(
        prose="Your K♠️ blocks some of villain's KK and AK combos."
    )
    assert validate_blocker_claims(generated, facts).is_valid


def test_blocker_negated_and_unblock_sentences_skipped() -> None:
    """Negative claims ('block almost nothing', 'unblocks AA') are often
    TRUE because the hand is absent from the fact -- never police them."""
    facts = _facts(hero_card_combo="TcTd", blockers={"TT": 5})
    generated = _gen(
        prose=(
            "You block almost nothing in villain's value range. "
            "Your hand unblocks the AA and AQs hands that continue."
        )
    )
    assert validate_blocker_claims(generated, facts).is_valid


def test_blocker_talk_with_no_blockers_fact_fails() -> None:
    """Open spots carry no blockers fact, so ANY blocker sentence is
    unlicensed (round-1 finding #6, now enforced)."""
    facts = _facts(villain_position=None, history=())
    generated = _gen(prose="Your ace is a strong blocker here.")
    result = validate_blocker_claims(generated, facts)
    assert not result.is_valid
    assert "blockers fact" in result.error_message


def test_no_blocker_talk_with_empty_fact_passes() -> None:
    facts = _facts(villain_position=None, history=())
    generated = _gen(prose="Open this hand for value and size up.")
    assert validate_blocker_claims(generated, facts).is_valid


# --- validate_terminology (June 2026 round-2 audit) -------------------------
def test_opener_cold_call_fails() -> None:
    """3 of 20 round-2 rows said the OPENER could 'cold-call' the 3-bet."""
    facts = _facts(
        actor="BB",
        history=_HIST_UTG_OPEN_UTG2_3BET,
        villain_position="UTG+2",
    )
    generated = _gen(prose="UTG can cold-call or 4-bet behind you.")
    result = validate_terminology(generated, facts)
    assert not result.is_valid
    assert "cold-call" in result.error_message
    assert "UTG" in result.error_message


def test_cold_call_sentence_naming_a_cold_seat_passes() -> None:
    """Mixed sentences (a raiser AND a genuinely cold seat named) pass --
    the validator must not cry wolf when the verb's subject is ambiguous."""
    facts = _facts()
    generated = _gen(prose="After BTN's raise, SB can cold-call in front of you.")
    assert validate_terminology(generated, facts).is_valid


def test_hero_cold_call_when_hero_is_cold_passes() -> None:
    facts = _facts()  # hero BB, no prior voluntary action
    generated = _gen(prose="You'd be cold-calling against a strong range.")
    assert validate_terminology(generated, facts).is_valid


def test_hero_cold_call_after_hero_raised_fails() -> None:
    """Hero opened earlier in the line -- hero can't cold-call the 3-bet."""
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("UTG+1", PreflopActionType.FOLD),
        ParsedAction("UTG+2", PreflopActionType.RAISE, 120.0),  # hero's open
        ParsedAction("LJ", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.RAISE, 84.0),
        ParsedAction("BTN", PreflopActionType.FOLD),
        ParsedAction("SB", PreflopActionType.FOLD),
        ParsedAction("BB", PreflopActionType.FOLD),
    )
    facts = _facts(actor="UTG+2", history=history, villain_position="CO")
    generated = _gen(prose="You can cold-call the 3-bet here.")
    result = validate_terminology(generated, facts)
    assert not result.is_valid


def test_unnamed_cold_caller_with_no_cold_pending_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 #14: 'a cold-caller behind' when every pending seat had
    already raised. Needs the pack's seat roster, so the registry lookup
    is monkeypatched to a 6-max stub."""
    import pipeline.preflop.pack as pack_mod

    monkeypatch.setattr(
        pack_mod, "get_pack", lambda pid: SimpleNamespace(table_size=6)
    )
    facts = _facts()  # default 6-max line: hero BB closes, only BTN raised
    generated = _gen(
        prose="You are stuck between the raiser and a cold-caller behind."
    )
    result = validate_terminology(generated, facts)
    assert not result.is_valid
    assert "cold-call" in result.error_message


def test_unnamed_cold_caller_unknown_pack_passes() -> None:
    """Bare fixtures have no registered pack: the fallback must stay
    silent rather than guess (mirrors the Layer-6 graceful degradation)."""
    facts = _facts()
    generated = _gen(prose="A cold-caller behind would change the math.")
    assert validate_terminology(generated, facts).is_valid


def test_squeeze_label_on_villains_3bet_without_caller_fails() -> None:
    facts = _facts(
        actor="BB",
        history=_HIST_UTG_OPEN_UTG2_3BET,
        villain_position="UTG+2",
    )
    generated = _gen(prose="UTG opened and UTG+2 squeezed over the top.")
    result = validate_terminology(generated, facts)
    assert not result.is_valid
    assert "squeeze" in result.error_message


def test_get_squeezed_off_passive_passes() -> None:
    """'You can get squeezed off the hand' is about HERO's future, not a
    mislabel of the recorded raise -- must not fire (round-2 #16 phrasing)."""
    facts = _facts(
        actor="BB",
        history=_HIST_UTG_OPEN_UTG2_3BET,
        villain_position="UTG+2",
    )
    generated = _gen(prose="If you call light you can get squeezed off the hand.")
    assert validate_terminology(generated, facts).is_valid


def test_squeeze_with_real_caller_passes() -> None:
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 120.0),
        ParsedAction("UTG+1", PreflopActionType.CALL),
        ParsedAction("UTG+2", PreflopActionType.RAISE, 84.0),
        ParsedAction("LJ", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.FOLD),
        ParsedAction("SB", PreflopActionType.FOLD),
    )
    facts = _facts(actor="BB", history=history, villain_position="UTG+2")
    generated = _gen(prose="UTG+2 squeezed the opener and the caller.")
    assert validate_terminology(generated, facts).is_valid


def test_open_fold_while_facing_a_raise_fails() -> None:
    facts = _facts()  # BTN open in history
    generated = _gen(prose="Against a tighter opener, just open-fold every time.")
    result = validate_terminology(generated, facts)
    assert not result.is_valid
    assert "open-fold" in result.error_message


def test_open_fold_first_in_passes() -> None:
    facts = _facts(villain_position=None, actor="UTG", history=())
    generated = _gen(prose="You can open-fold the bottom of your range.")
    assert validate_terminology(generated, facts).is_valid


def test_raise_ladder_overflow_fails() -> None:
    """Two raises in history: hero's raise is the 4-bet, the response the
    5-bet -- a 6-bet is unreachable and must fail."""
    facts = _facts(
        actor="BB",
        history=_HIST_UTG_OPEN_UTG2_3BET,
        villain_position="UTG+2",
    )
    generated = _gen(prose="He will never fold KK to a 6-bet.")
    result = validate_terminology(generated, facts)
    assert not result.is_valid
    assert "6-bet" in result.error_message


def test_raise_ladder_within_bound_passes() -> None:
    facts = _facts(
        actor="BB",
        history=_HIST_UTG_OPEN_UTG2_3BET,
        villain_position="UTG+2",
    )
    generated = _gen(prose="You can 4-bet, and calling a 5-bet jam is fine.")
    assert validate_terminology(generated, facts).is_valid


def test_capped_for_aa_heavy_villain_range_fails() -> None:
    facts = _facts(villain_top=(("AA", 1.0), ("KK", 1.0)))
    generated = _gen(prose="Villain's raising range is capped here.")
    result = validate_terminology(generated, facts)
    assert not result.is_valid
    assert "capped" in result.error_message


def test_capped_about_hero_range_passes() -> None:
    """Hero's own range CAN legitimately be capped -- only villain-referenced
    capped claims are checked against villain's top combos."""
    facts = _facts(villain_top=(("AA", 1.0), ("KK", 1.0)))
    generated = _gen(prose="Your range is capped after just calling.")
    assert validate_terminology(generated, facts).is_valid


# --- validate_card_suit_consistency (June 2026 round-2 audit) ----------------
def test_wrong_suit_in_prose_fails() -> None:
    """Round-2 #16: hero holds K♠5♠ but the prose said 'your K❤️' twice."""
    facts = _facts(hero_card_combo="Ks5s")
    generated = _gen(prose="Your K❤️ is the key card in this spot.")
    result = validate_card_suit_consistency(generated, facts)
    assert not result.is_valid
    assert "Kh" in result.error_message


def test_correct_suits_and_heart_variants_pass() -> None:
    """Both heart emojis (U+2665 and U+2764) normalise to hearts."""
    facts = _facts(hero_card_combo="AhQc")
    generated = _gen(prose="Your A♥️Q♣️ plays well, and the A❤️ blocks a lot.")
    assert validate_card_suit_consistency(generated, facts).is_valid


def test_hand_class_labels_without_emoji_ignored() -> None:
    facts = _facts(hero_card_combo="Ks5s")
    generated = _gen(prose="Hands like AKo and JTs prefer raising instead.")
    assert validate_card_suit_consistency(generated, facts).is_valid


# --- soft validators ---------------------------------------------------------
def test_soft_position_warns_on_oop_claim_when_ip() -> None:
    """Round-2 #15/#18: 'calling out of position' while the fact says In
    Position. Soft (flag-not-reject): legit multiway-realization phrasing
    exists, so it queues for review instead of failing the row."""
    facts = _facts(actor="BTN", villain_position="SB")
    generated = _gen(prose="You are out of position in this pot.")
    warnings = soft_validate_position_words(generated, facts)
    assert len(warnings) == 1
    assert "In Position" in warnings[0]


def test_soft_position_warns_on_ip_claim_when_oop() -> None:
    facts = _facts()  # BB vs BTN: hero OOP
    generated = _gen(prose="You get to play in position with the lead.")
    warnings = soft_validate_position_words(generated, facts)
    assert len(warnings) == 1


def test_soft_position_consistent_prose_is_silent() -> None:
    facts = _facts()  # BB vs BTN: hero OOP
    generated = _gen(prose="You are out of position against the raiser.")
    assert soft_validate_position_words(generated, facts) == []
    assert run_preflop_soft_validators(generated, facts) == []


def test_soft_position_villain_described_oop_not_flagged() -> None:
    """First-batch false positive (June 2026): hero IP, prose correctly says
    the VILLAIN (SB) 3-bets out of position. Must NOT flag -- the phrase's
    subject is the small blind, not hero."""
    facts = _facts(actor="BTN", villain_position="SB")  # hero IP
    generated = _gen(
        prose=(
            "When the small blind 3-bets out of position, they need a tight "
            "range to do it."
        )
    )
    assert soft_validate_position_words(generated, facts) == []


def test_soft_position_negated_claim_not_flagged() -> None:
    """'You are not in position' AGREES with an OOP hero -- the negation
    guard before the phrase must keep it quiet."""
    facts = _facts()  # BB vs BTN: hero OOP
    generated = _gen(prose="You are not in position here, so play it safe.")
    assert soft_validate_position_words(generated, facts) == []


# --- runner picks up the round-2 validators ----------------------------------
def test_runner_catches_round2_defects() -> None:
    facts = _facts(hero_card_combo="Ks5s")
    generated = _gen(prose="Your K❤️ likes this price.")
    result = run_preflop_audit_validators(generated, facts)
    assert not result.is_valid
    assert "Kh" in result.error_message
