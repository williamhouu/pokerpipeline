"""Tests for pipeline.preflop.question_extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.question_extractor import (  # noqa: E402
    MAX_TOP_FREQUENCY,
    MIN_PRESENCE,
    MIN_TOP_FREQUENCY,
    PreflopQuestionEvaluation,
    difficulty_score,
    evaluate_spot,
    is_question_worthy,
    top_action_frequency,
    total_presence,
)
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


def _spot(
    frequencies: dict[str, float],
    *,
    dominant_action: str | None = None,
) -> PreflopSpot:
    """Build a minimal PreflopSpot with rigged frequencies."""
    dom = dominant_action or max(frequencies.items(), key=lambda kv: kv[1])[0]
    node = PreflopDecisionNode(
        pack_id="t",
        actor="BTN",
        history_before=(),
        actions=(),
    )
    return PreflopSpot(
        node=node,
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies=frequencies,
        dominant_action=dom,
        dominant_frequency=frequencies[dom],
    )


# --- top_action_frequency + total_presence -------------------------------
def test_top_action_frequency_returns_dominant():
    spot = _spot({"Raise": 0.7, "Call": 0.2, "Fold": 0.1})
    assert top_action_frequency(spot) == 0.7


def test_total_presence_sums_all():
    spot = _spot({"Raise": 0.7, "Call": 0.2, "Fold": 0.1})
    assert total_presence(spot) == pytest.approx(1.0)


def test_total_presence_zero_for_no_show():
    spot = _spot({"Raise": 0.0, "Call": 0.0, "Fold": 0.0}, dominant_action="Fold")
    assert total_presence(spot) == 0.0


# --- difficulty_score ------------------------------------------------------
def test_difficulty_at_min_frequency_is_ceiling():
    """A barely-dominant 55% spot is the hardest (3000)."""
    spot = _spot({"Raise": 0.55, "Fold": 0.45})
    assert difficulty_score(spot) == 3000


def test_difficulty_at_max_frequency_is_floor():
    """A near-pure 95% spot is the easiest (500)."""
    spot = _spot({"Raise": 0.95, "Fold": 0.05})
    assert difficulty_score(spot) == 500


def test_difficulty_mid_range():
    """75% -- halfway between 55% and 95% -- should be ~1750 (midpoint)."""
    spot = _spot({"Raise": 0.75, "Fold": 0.25})
    score = difficulty_score(spot)
    assert 1700 < score < 1800


def test_difficulty_above_max_clamps_to_floor():
    """A 100% pure spot is even easier than 95% -- clamped to 500."""
    spot = _spot({"Raise": 1.0, "Fold": 0.0})
    assert difficulty_score(spot) == 500


def test_difficulty_below_min_clamps_to_ceiling():
    """A 40% non-dominant spot wouldn't pass worthiness, but if asked
    its difficulty is clamped to 3000 (hardest)."""
    spot = _spot({"Raise": 0.4, "Call": 0.35, "Fold": 0.25})
    assert difficulty_score(spot) == 3000


# --- is_question_worthy ----------------------------------------------------
def test_worthy_spot_passes_all_gates():
    spot = _spot({"Raise": 0.7, "Fold": 0.3})
    assert is_question_worthy(spot) is True


def test_worthy_at_exactly_min_frequency():
    """Inclusive at the low end: 55% exactly passes."""
    spot = _spot({"Raise": 0.55, "Fold": 0.45})
    assert is_question_worthy(spot) is True


def test_worthy_at_exactly_max_frequency():
    """Inclusive at the high end: 95% exactly passes."""
    spot = _spot({"Raise": 0.95, "Fold": 0.05})
    assert is_question_worthy(spot) is True


def test_unworthy_just_below_min():
    spot = _spot({"Raise": 0.54, "Call": 0.46})
    assert is_question_worthy(spot) is False


def test_unworthy_just_above_max():
    spot = _spot({"Raise": 0.96, "Fold": 0.04})
    assert is_question_worthy(spot) is False


def test_unworthy_for_zero_presence():
    """Hand doesn't reach the node -> not worthy."""
    spot = _spot({"Raise": 0.0, "Call": 0.0, "Fold": 0.0}, dominant_action="Fold")
    assert is_question_worthy(spot) is False


def test_custom_frequency_window():
    """The admin panel's difficulty presets override the defaults."""
    spot = _spot({"Raise": 0.62, "Fold": 0.38})
    # Default window includes 62% -> worthy.
    assert is_question_worthy(spot) is True
    # "Easy" preset (85-95) doesn't include 62% -> not worthy.
    assert is_question_worthy(spot, min_frequency=0.85, max_frequency=0.95) is False
    # "Hard" preset (55-70) includes 62% -> worthy.
    assert is_question_worthy(spot, min_frequency=0.55, max_frequency=0.70) is True


def test_min_presence_override():
    """Tight presence threshold filters out marginal hands."""
    spot = _spot({"Raise": 0.50, "Call": 0.04, "Fold": 0.01}, dominant_action="Raise")
    # default 0.55 floor on frequency -> not worthy anyway
    spot2 = _spot({"Raise": 0.60, "Call": 0.05, "Fold": 0.0})
    # Presence is 0.65; with min_presence=0.7 it'd be excluded.
    assert is_question_worthy(spot2, min_presence=0.7) is False
    assert is_question_worthy(spot2, min_presence=0.5) is True


# --- evaluate_spot ---------------------------------------------------------
def test_evaluate_spot_returns_full_verdict():
    spot = _spot({"Raise": 0.7, "Call": 0.2, "Fold": 0.1})
    ev = evaluate_spot(spot)
    assert isinstance(ev, PreflopQuestionEvaluation)
    assert ev.is_worthy is True
    assert ev.top_action_frequency == 0.7
    assert ev.total_presence == pytest.approx(1.0)
    # 70% -> mid-difficulty (~2063 per formula)
    assert 1900 < ev.difficulty_score < 2200


def test_evaluate_spot_passes_overrides():
    """Custom thresholds in evaluate_spot reach the worthiness check."""
    spot = _spot({"Raise": 0.62, "Fold": 0.38})
    ev = evaluate_spot(spot, min_frequency=0.85, max_frequency=0.95)
    assert ev.is_worthy is False


# --- constants sanity ------------------------------------------------------
def test_default_constants_are_sensible():
    assert 0.0 < MIN_TOP_FREQUENCY < MAX_TOP_FREQUENCY < 1.0
    assert 0.0 < MIN_PRESENCE < 1.0
