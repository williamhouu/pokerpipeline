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
    soft_validate_named_combo_in_range,
    soft_validate_number_vs_data,
    soft_validate_position_words,
    soft_validate_verdict_vs_answer,
    validate_banned_phrases,
    validate_blocker_claims,
    validate_card_suit_consistency,
    validate_composite_label_frequencies,
    validate_equity_vs_price_direction,
    validate_no_postflop_on_allin,
    validate_no_standalone_sometimes,
    validate_option_set,
    validate_terminology,
)

# A squeeze line: UTG opens, two callers, SB squeezes (a 3-bet); hero BB folds.
# Two raises total -> the latest raise is a 3-bet, so naming it a "4-bet" is the
# round-2 error the villain-raise-level check targets.
_HIST_SQUEEZE = (
    ParsedAction("UTG", PreflopActionType.RAISE, 120.0),
    ParsedAction("LJ", PreflopActionType.CALL, 120.0),
    ParsedAction("CO", PreflopActionType.CALL, 120.0),
    ParsedAction("SB", PreflopActionType.RAISE, 500.0),
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
    hero_equity_vs_villain: float | None = 0.55,
    hero_range_equity_vs_villain: float | None = None,
    break_even_equity: float | None = None,
    showdown_opponents: tuple[str, ...] = (),
    villain_played_classes: frozenset[str] | None = None,
) -> PreflopFacts:
    """Minimal PreflopFacts. Default: BB facing a BTN open, 66/34 call/fold.

    The round-2 validator tests vary the history, hero combo, blockers
    fact, and villain identity; ``villain_position=None`` makes an open
    spot (no villain, empty blockers). The June 2026 soft-validator tests
    vary the equity / break-even / showdown / villain-range fields.
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
        hero_equity_vs_villain=hero_equity_vs_villain,
        hero_range_equity_vs_villain=hero_range_equity_vs_villain,
        break_even_equity=break_even_equity,
        showdown_opponents=showdown_opponents,
        archetype=archetype,
        blockers=blockers or {},
        villain_played_classes=(
            villain_played_classes
            if villain_played_classes is not None
            else frozenset()
        ),
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


def test_no_postflop_on_allin_allows_negated_phrases() -> None:
    """The flagged screenshot bug: prose that says there is NO postflop play /
    NO future streets is CORRECT showdown framing, not a violation. The
    negation guard must let it pass (it was a false rejection before)."""
    facts = _facts(archetype="call_allin")
    generated = _gen(
        prose="There's no postflop play to save you and there are no future "
              "streets to navigate; the chips go in now and the hand is "
              "decided at showdown, so all that matters is the price versus "
              "your raw equity.")
    result = validate_no_postflop_on_allin(generated, facts)
    assert result.is_valid, result.error_message


def test_no_postflop_on_allin_still_fails_mixed_negated_and_real() -> None:
    """A genuine violation isn't laundered by a negated phrase elsewhere:
    'no future streets' is fine, but 'you have implied odds' (un-negated) in
    the same prose still fails."""
    facts = _facts(archetype="call_allin")
    generated = _gen(
        prose="There are no future streets, but you still have implied odds "
              "to chase a flush here.")
    result = validate_no_postflop_on_allin(generated, facts)
    assert not result.is_valid
    assert "implied odds" in result.error_message.lower()


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


def test_blocker_blocks_zero_combos_is_negation_skipped() -> None:
    """'blocks ZERO combos of AA' asserts absence, like 'blocks nothing' --
    it must NOT be policed (June 13 diagnostic: a poker-correct QQ line was
    being hard-rejected because 'zero' wasn't a negation marker)."""
    facts = _facts(
        hero_card_combo="QcQd",
        blockers={"QQ": 5, "KQs": 2, "AQs": 2},
    )
    generated = _gen(
        prose=(
            "You hold two queens, which blocks zero combos of AA, KK, and "
            "AKs, the exact hands crushing you."
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


def test_soft_position_catches_button_out_of_position_misfire() -> None:
    """June 2026 screenshot: hero on the BUTTON facing a HJ open + CO 3-bet is
    IN position, but the fold_no_continue archetype frame (which used to bake
    in 'poor playability out of position') led the LLM to write 'continuing
    out of position'. The guidance was fixed to defer to the hero_position
    fact; this asserts the deterministic soft validator still catches the
    contradiction as the safety net."""
    facts = _facts(actor="BTN", villain_position="CO", archetype="fold_no_continue")
    generated = _gen(
        prose=(
            "The best play is to mostly fold. You'd be continuing out of "
            "position in a bloated pot against a 3-bettor, which means you "
            "realize your equity poorly."
        )
    )
    warnings = soft_validate_position_words(generated, facts)
    assert len(warnings) == 1
    assert "In Position" in warnings[0]


# --- soft_validate_number_vs_data (June 2026) --------------------------------
def test_soft_number_flags_wrong_equity() -> None:
    """Prose says 55% equity, data block has 40% -- a 15-pt miss flags."""
    facts = _facts(hero_equity_vs_villain=0.40)
    generated = _gen(prose="You have about 55% equity against the range here.")
    warnings = soft_validate_number_vs_data(generated, facts)
    assert len(warnings) == 1
    assert "55" in warnings[0] and "40%" in warnings[0]


def test_soft_number_flags_wrong_break_even() -> None:
    """'need 41% to call' vs a 25% break-even -- a 16-pt miss flags."""
    facts = _facts(hero_equity_vs_villain=0.55, break_even_equity=0.25)
    generated = _gen(prose="You need about 41% to call, so the price is fine.")
    warnings = soft_validate_number_vs_data(generated, facts)
    assert len(warnings) == 1
    assert "break-even" in warnings[0].lower()


def test_soft_number_passes_matching_equity() -> None:
    facts = _facts(hero_equity_vs_villain=0.55)
    generated = _gen(prose="You hold about 55% equity, so calling is clear.")
    assert soft_validate_number_vs_data(generated, facts) == []


def test_soft_number_passes_matching_break_even() -> None:
    facts = _facts(hero_equity_vs_villain=0.55, break_even_equity=0.30)
    generated = _gen(prose="You need roughly 30% to call and you beat that.")
    assert soft_validate_number_vs_data(generated, facts) == []


def test_soft_number_break_even_cue_wins_over_equity_word() -> None:
    """'need 41% equity to call' is a BREAK-EVEN threshold, not hero's actual
    equity -- the 'to call' cue must route it to break_even_equity (40%), not
    compare 41 against hero's 55% equity."""
    facts = _facts(hero_equity_vs_villain=0.55, break_even_equity=0.40)
    generated = _gen(prose="You need about 41% equity to call here.")
    assert soft_validate_number_vs_data(generated, facts) == []


def test_soft_number_skips_multiway() -> None:
    """Multi-way pots carry field-vs-per-opponent numbers; gate them out."""
    facts = _facts(
        hero_equity_vs_villain=0.40,
        showdown_opponents=("BTN", "HJ"),
    )
    generated = _gen(prose="You have about 55% equity against the field.")
    assert soft_validate_number_vs_data(generated, facts) == []


def test_soft_number_skips_range_equity_phrasing() -> None:
    """'your range has 53%' is RANGE equity (a different fact) -- never
    compare it to hero's HAND equity."""
    facts = _facts(hero_equity_vs_villain=0.40)
    generated = _gen(prose="Your range has about 53% equity in this spot.")
    assert soft_validate_number_vs_data(generated, facts) == []


def test_soft_number_skips_frequency_percent() -> None:
    """A frequency ('raise 70% of the time') has no equity/pot-odds cue, so
    it is not mapped to a fact and never flags."""
    facts = _facts(hero_equity_vs_villain=0.40)
    generated = _gen(prose="You raise 70% of the time and fold the rest.")
    assert soft_validate_number_vs_data(generated, facts) == []


def test_soft_number_matches_range_equity_candidate() -> None:
    """Defensive backstop: a number that matches the range-equity fact (53%)
    must not flag even though it differs from hand equity (40%)."""
    facts = _facts(
        hero_equity_vs_villain=0.40,
        hero_range_equity_vs_villain=0.53,
    )
    generated = _gen(prose="You have around 53% equity in this spot.")
    assert soft_validate_number_vs_data(generated, facts) == []


# --- soft_validate_verdict_vs_answer (June 2026) -----------------------------
def test_soft_verdict_flags_call_when_answer_is_fold() -> None:
    facts = _facts(dominant_action="Fold")
    generated = _gen(prose="This is a clear call with our equity.")
    warnings = soft_validate_verdict_vs_answer(generated, facts)
    assert len(warnings) == 1
    assert "Fold" in warnings[0]


def test_soft_verdict_flags_fold_when_answer_is_raise() -> None:
    facts = _facts(dominant_action="Raise 60%")
    generated = _gen(prose="The best play is to fold this hand here.")
    warnings = soft_validate_verdict_vs_answer(generated, facts)
    assert len(warnings) == 1
    assert "Raise 60%" in warnings[0]


def test_soft_verdict_flags_fold_when_answer_is_allin() -> None:
    facts = _facts(dominant_action="AllIn", archetype="all_in_for_value")
    generated = _gen(prose="Just fold and wait for a better spot.")
    warnings = soft_validate_verdict_vs_answer(generated, facts)
    assert len(warnings) == 1


def test_soft_verdict_passes_matching_action() -> None:
    facts = _facts(dominant_action="Call")
    generated = _gen(prose="This is a clear call given the price.")
    assert soft_validate_verdict_vs_answer(generated, facts) == []


def test_soft_verdict_passes_raise_family_match() -> None:
    """Dominant is a sized raise; the verdict says '3-bet' -- same aggressive
    bucket, so no flag (sizing words aren't a conflict)."""
    facts = _facts(dominant_action="Raise 60%", archetype="3bet_for_value")
    generated = _gen(prose="You should 3-bet this for value.")
    assert soft_validate_verdict_vs_answer(generated, facts) == []


def test_soft_verdict_passes_hedged_verdict_naming_answer() -> None:
    """A hedge that NAMES the answer ('folding looks tempting but calling is
    right') passes -- the answer's action appears in the first sentence."""
    facts = _facts(dominant_action="Call")
    generated = _gen(
        prose="Folding looks tempting, but calling is correct with these odds."
    )
    assert soft_validate_verdict_vs_answer(generated, facts) == []


def test_soft_verdict_check_compatible_with_call() -> None:
    """A BB check is filed under 'Call'; a 'check' verdict must not flag."""
    facts = _facts(actor="BB", dominant_action="Call", archetype="bb_check")
    generated = _gen(prose="Just check behind and see a free flop.")
    assert soft_validate_verdict_vs_answer(generated, facts) == []


def test_soft_verdict_silent_without_action_word() -> None:
    facts = _facts(dominant_action="Call")
    generated = _gen(prose="AKo is a premium holding in this position.")
    assert soft_validate_verdict_vs_answer(generated, facts) == []


# --- soft_validate_named_combo_in_range (June 2026) --------------------------
_VILLAIN_PLAYED = frozenset({"AA", "KK", "QQ", "AKs", "AKo", "AQs"})


def test_soft_named_combo_flags_absent_class() -> None:
    facts = _facts(villain_played_classes=_VILLAIN_PLAYED)
    generated = _gen(prose="Villain's range includes 72o and other trash.")
    warnings = soft_validate_named_combo_in_range(generated, facts)
    assert len(warnings) == 1
    assert "72o" in warnings[0]


def test_soft_named_combo_flags_absent_among_present() -> None:
    """A mixed claim ('AK and 32o') flags only the absent class."""
    facts = _facts(villain_played_classes=_VILLAIN_PLAYED)
    generated = _gen(prose="They have AK and 32o in their value range.")
    warnings = soft_validate_named_combo_in_range(generated, facts)
    assert len(warnings) == 1
    # Only the absent class is named in the flagged list (the message echoes
    # the full sentence after "Offending sentence:", which still has "AK").
    absent_list = warnings[0].split("but that hand class")[0]
    assert "32o" in absent_list
    assert "AK" not in absent_list


def test_soft_named_combo_passes_present_classes() -> None:
    facts = _facts(villain_played_classes=_VILLAIN_PLAYED)
    generated = _gen(prose="Villain's range includes AK and QQ for value.")
    assert soft_validate_named_combo_in_range(generated, facts) == []


def test_soft_named_combo_skips_negated_claim() -> None:
    """'villain doesn't have 72o' is a TRUE absence claim -- never police it."""
    facts = _facts(villain_played_classes=_VILLAIN_PLAYED)
    generated = _gen(prose="Villain doesn't have 72o in this spot.")
    assert soft_validate_named_combo_in_range(generated, facts) == []


def test_soft_named_combo_ignores_hero_hand_mention() -> None:
    """A hand named without a villain-possession cue (it's hero's hand) is not
    policed, even if absent from villain's range."""
    facts = _facts(villain_played_classes=_VILLAIN_PLAYED)
    generated = _gen(prose="72o would be a strong holding for us here.")
    assert soft_validate_named_combo_in_range(generated, facts) == []


def test_soft_named_combo_skips_multiway() -> None:
    facts = _facts(
        villain_played_classes=_VILLAIN_PLAYED,
        showdown_opponents=("BTN", "HJ"),
    )
    generated = _gen(prose="Villain's range includes 72o here.")
    assert soft_validate_named_combo_in_range(generated, facts) == []


def test_soft_named_combo_skips_when_range_unknown() -> None:
    """No villain range loaded (open spot / load failure) -> can't verify."""
    facts = _facts(villain_played_classes=frozenset())
    generated = _gen(prose="Villain's range includes 72o here.")
    assert soft_validate_named_combo_in_range(generated, facts) == []


# --- run_preflop_soft_validators wiring --------------------------------------
def test_runner_collects_new_soft_validators() -> None:
    """The runner picks up the verdict-vs-answer check (and friends)."""
    facts = _facts(dominant_action="Fold")
    generated = _gen(prose="This is a clear call with our equity.")
    warnings = run_preflop_soft_validators(generated, facts)
    assert any("Fold" in w for w in warnings)


def test_runner_soft_clean_prose_silent() -> None:
    facts = _facts(
        dominant_action="Call",
        hero_equity_vs_villain=0.55,
        villain_played_classes=_VILLAIN_PLAYED,
    )
    generated = _gen(
        prose="This is a clear call. You hold around 55% equity against the "
              "range, so the price is right."
    )
    assert run_preflop_soft_validators(generated, facts) == []


# --- runner picks up the round-2 validators ----------------------------------
def test_runner_catches_round2_defects() -> None:
    facts = _facts(hero_card_combo="Ks5s")
    generated = _gen(prose="Your K❤️ likes this price.")
    result = run_preflop_audit_validators(generated, facts)
    assert not result.is_valid
    assert "Kh" in result.error_message


# --- validate_equity_vs_price_direction (Finding 1) -----------------------
def test_equity_price_fires_on_price_not_met() -> None:
    facts = _facts(archetype="fold_pot_odds", hero_equity_vs_villain=0.28,
                   break_even_equity=0.26)
    gen = _gen(correct="Fold", prose=(
        "The best play is to fold. The price to continue is not being met "
        "given how often you're dominated."))
    res = validate_equity_vs_price_direction(gen, facts)
    assert not res.is_valid
    assert "28%" in res.error_message and "26%" in res.error_message


def test_equity_price_fires_on_collapses_below() -> None:
    facts = _facts(archetype="fold_dominated", hero_equity_vs_villain=0.34,
                   break_even_equity=0.265)
    gen = _gen(correct="Fold",
               prose="Fold. Your equity collapses below what the price needs.")
    assert not validate_equity_vs_price_direction(gen, facts).is_valid


def test_equity_price_uses_field_equity_when_present() -> None:
    import dataclasses
    facts = dataclasses.replace(
        _facts(archetype="fold_pot_odds", hero_equity_vs_villain=None,
               break_even_equity=0.265),
        hero_equity_vs_field=0.34,
    )
    gen = _gen(correct="Fold", prose="Fold. Your equity sits below the price here.")
    assert not validate_equity_vs_price_direction(gen, facts).is_valid


# False-positive guards (must PASS):
def test_equity_price_passes_genuine_price_fold() -> None:
    # equity BELOW break-even -> "price not met" is CORRECT.
    facts = _facts(archetype="fold_pot_odds", hero_equity_vs_villain=0.20,
                   break_even_equity=0.26)
    gen = _gen(correct="Fold",
               prose="Fold. The price isn't being met with this little equity.")
    assert validate_equity_vs_price_direction(gen, facts).is_valid


def test_equity_price_passes_realization_framing() -> None:
    # equity ABOVE the price, framed (correctly) around realization, no price claim.
    facts = _facts(archetype="fold_pot_odds", hero_equity_vs_villain=0.30,
                   break_even_equity=0.26)
    gen = _gen(correct="Fold", prose=(
        "Fold. Your raw equity beats the price, but out of position multiway "
        "you won't realize it."))
    assert validate_equity_vs_price_direction(gen, facts).is_valid


def test_equity_price_passes_within_noise_margin() -> None:
    # near-tie (gap < margin) -> don't fire even with a price phrase.
    facts = _facts(archetype="fold_pot_odds", hero_equity_vs_villain=0.265,
                   break_even_equity=0.26)
    gen = _gen(correct="Fold", prose="Fold. The price isn't quite being met here.")
    assert validate_equity_vs_price_direction(gen, facts).is_valid


def test_equity_price_passes_non_fold_archetype() -> None:
    facts = _facts(archetype="call_for_value", hero_equity_vs_villain=0.55,
                   break_even_equity=0.26)
    gen = _gen(correct="Call", prose="Call. The price is easily met here.")
    assert validate_equity_vs_price_direction(gen, facts).is_valid


def test_equity_price_passes_when_no_break_even() -> None:
    facts = _facts(archetype="fold_pot_odds", hero_equity_vs_villain=0.40,
                   break_even_equity=None)
    gen = _gen(correct="Fold", prose="Fold. The price is not met.")
    assert validate_equity_vs_price_direction(gen, facts).is_valid


def test_equity_price_passes_position_idiom() -> None:
    # "under the gun" must not read as "under ... the price".
    facts = _facts(archetype="fold_dominated", hero_equity_vs_villain=0.30,
                   break_even_equity=0.26)
    gen = _gen(correct="Fold", prose=(
        "Fold. From under the gun this hand is too dominated to continue."))
    assert validate_equity_vs_price_direction(gen, facts).is_valid


# --- validate_terminology: villain raise named too high (Finding 2) -------
def test_terminology_fires_on_squeeze_called_4bet() -> None:
    facts = _facts(actor="BB", archetype="fold_dominated", villain_position="SB",
                   history=_HIST_SQUEEZE)
    gen = _gen(correct="Fold", prose=(
        "The SB cold 4-bet against an open and two callers, so fold."))
    res = validate_terminology(gen, facts)
    assert not res.is_valid
    assert "4-bet" in res.error_message and "SB" in res.error_message


def test_terminology_passes_hypothetical_4bet() -> None:
    facts = _facts(actor="BB", archetype="fold_dominated", villain_position="SB",
                   history=_HIST_SQUEEZE)
    gen = _gen(correct="Fold", prose="If you 4-bet, SB might just call. Fold.")
    assert validate_terminology(gen, facts).is_valid


def test_terminology_passes_villain_responding_to_hero_raise() -> None:
    # "SB folds to a 4-bet" describes hero's hypothetical raise, not SB raising.
    facts = _facts(actor="BB", archetype="fold_dominated", villain_position="SB",
                   history=_HIST_SQUEEZE)
    gen = _gen(correct="Fold", prose="SB folds to a 4-bet a fair amount.")
    assert validate_terminology(gen, facts).is_valid


def test_terminology_passes_correct_completed_4bet() -> None:
    # A real 4-bet line: open, 3-bet, 4-bet (3 raises) -> "BTN's 4-bet" is right.
    hist = (
        ParsedAction("UTG", PreflopActionType.RAISE, 100.0),
        ParsedAction("CO", PreflopActionType.RAISE, 300.0),
        ParsedAction("BTN", PreflopActionType.RAISE, 800.0),
    )
    facts = _facts(actor="BB", archetype="fold_dominated", villain_position="BTN",
                   history=hist)
    gen = _gen(correct="Fold", prose="BTN's 4-bet is enormous, so fold.")
    assert validate_terminology(gen, facts).is_valid


def test_terminology_skips_sentence_with_second_person() -> None:
    # A "you" sentence is skipped (hero raises create hypothetical re-raises).
    facts = _facts(actor="BB", archetype="fold_dominated", villain_position="SB",
                   history=_HIST_SQUEEZE)
    gen = _gen(correct="Fold",
               prose="The SB cold 4-bet and you simply have to let it go.")
    assert validate_terminology(gen, facts).is_valid
