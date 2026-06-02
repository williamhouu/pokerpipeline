"""Tests for pipeline.preflop.validators.

Per-validator positive (returns ok) and negative (returns fail with a
useful error message) cases for each of the v1 hard validators, plus a
runner test that confirms first-failure short-circuiting.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    validate_banned_phrases,
    validate_composite_label_frequencies,
    validate_no_postflop_on_allin,
    validate_no_standalone_sometimes,
    validate_option_set,
)


# --- fixtures -------------------------------------------------------------
def _facts(
    *,
    action_frequencies: dict[str, float] | None = None,
    dominant_action: str = "Call",
    dominant_frequency: float = 0.66,
    actor: str = "BB",
    archetype: str = "call_for_value",
) -> PreflopFacts:
    """Minimal PreflopFacts. Default: BB facing a BTN open, 66/34 call/fold."""
    if action_frequencies is None:
        action_frequencies = {"Call": 0.66, "Fold": 0.34}
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor=actor,
            history_before=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.FOLD),
                ParsedAction("CO", PreflopActionType.FOLD),
                ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
                ParsedAction("SB", PreflopActionType.FOLD),
            ),
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
            position="BTN", action_label="Raise 60%",
            weighted_combo_count=600.0, pct_of_dealt_hands=45.0,
            top_combos=(),
        ),
        hero_equity_vs_villain=0.55,
        archetype=archetype,
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
