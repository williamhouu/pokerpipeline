"""Tests for pipeline.preflop.question_extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.question_extractor import (  # noqa: E402
    AMBIGUOUS_BAND_CEILING,
    AMBIGUOUS_BAND_FLOOR,
    MAX_TOP_FREQUENCY,
    MIN_PRESENCE,
    MIN_TOP_FREQUENCY,
    NEAR_PURE_BAND_CEILING,
    NEAR_PURE_BAND_FLOOR,
    PreflopQuestionEvaluation,
    difficulty_score,
    evaluate_spot,
    in_ambiguous_band,
    in_near_pure_band,
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
    """For test fixtures that don't set presence explicitly, the sentinel
    default + __post_init__ in PreflopSpot auto-computes presence from
    action_frequencies sum. Frequencies summing to 1.0 -> presence 1.0."""
    spot = _spot({"Raise": 0.7, "Call": 0.2, "Fold": 0.1})
    assert total_presence(spot) == pytest.approx(1.0)


def test_total_presence_zero_for_no_show():
    """All-zero frequencies (hand never reaches node) -> presence 0."""
    spot = _spot({"Raise": 0.0, "Call": 0.0, "Fold": 0.0}, dominant_action="Fold")
    assert total_presence(spot) == 0.0


def test_total_presence_reads_explicit_presence_field():
    """total_presence reads spot.presence directly -- it's the
    UN-NORMALISED 'how often does hand reach node' signal. When the
    fixture sets action_frequencies to the CONDITIONAL strategy (sums
    to 1.0) but presence to a different value, total_presence returns
    that explicit value, not the sum."""
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor="BTN", history_before=(), actions=()
        ),
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies={"Call": 0.75, "Fold": 0.25},  # conditional, sums to 1
        dominant_action="Call",
        dominant_frequency=0.75,
        presence=0.40,  # un-normalised
    )
    assert total_presence(spot) == pytest.approx(0.40)
    # Sanity: action_frequencies still sum to ~1.0 (conditional strategy).
    assert sum(spot.action_frequencies.values()) == pytest.approx(1.0)


# --- difficulty_score ------------------------------------------------------
def test_difficulty_at_min_frequency_is_ceiling():
    """A barely-dominant 55% spot is the hardest (3000)."""
    spot = _spot({"Raise": 0.55, "Fold": 0.45})
    assert difficulty_score(spot) == 3000


def test_difficulty_at_max_frequency_is_floor():
    """A 100% pure spot is the easiest (500). Post-Ryan-feedback May-2026
    the linear region extends to 1.00 so 100% IS the floor (previously
    95% already clamped to 500)."""
    spot = _spot({"Raise": 1.0, "Fold": 0.0})
    assert difficulty_score(spot) == 500


def test_difficulty_mid_range():
    """75% -- partway between 55% and 100% -- should be ~1889. Linear
    interp: 3000 - ((0.75 - 0.55) / 0.45) * 2500 = 3000 - 1111 = 1889."""
    spot = _spot({"Raise": 0.75, "Fold": 0.25})
    score = difficulty_score(spot)
    assert 1850 < score < 1950


def test_difficulty_95_pct_gradates_above_floor():
    """At 95% frequency the spot is still easier than 75% but no longer
    clamps to the floor -- Ryan asked for gradation in 95-100% range so a
    near-pure 95% spot is ~778 rather than 500."""
    spot = _spot({"Raise": 0.95, "Fold": 0.05})
    score = difficulty_score(spot)
    # Linear interp at 95%: 3000 - ((0.95 - 0.55) / 0.45) * 2500 = 778.
    assert 750 < score < 800


def test_difficulty_98_pct_still_above_floor():
    """98% gradates between 95% (~778) and 100% (500)."""
    spot = _spot({"Raise": 0.98, "Fold": 0.02})
    score = difficulty_score(spot)
    # Linear interp at 98%: 3000 - ((0.98 - 0.55) / 0.45) * 2500 = 611.
    assert 580 < score < 640
    # And it's between the 95% value (778) and the 100% floor (500).
    assert 500 < score < 778


def test_difficulty_below_min_clamps_to_ceiling():
    """A 40% non-dominant spot wouldn't pass worthiness, but if asked
    its difficulty is clamped to 3000 (hardest)."""
    spot = _spot({"Raise": 0.4, "Call": 0.35, "Fold": 0.25})
    assert difficulty_score(spot) == 3000


# NOTE (May 2026 redesign): difficulty_score() here is the cheap
# pre-facts freq-only ESTIMATE used by the worthiness gate. The
# canonical full-axis Difficulty Rating that lands in the CSV is
# computed by pipeline.preflop.difficulty.compute_difficulty() and
# tested in tests/test_preflop_difficulty.py. The freq-only tests
# above (test_difficulty_at_min_frequency_is_ceiling etc.) cover
# this function fully -- there's no longer an EV-gap blend here.


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
    spot = _spot({"Raise": 0.60, "Call": 0.05, "Fold": 0.0})
    # Presence is 0.65; with min_presence=0.7 it'd be excluded.
    assert is_question_worthy(spot, min_presence=0.7) is False
    assert is_question_worthy(spot, min_presence=0.5) is True


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


# --- ambiguous-band exclusion (a 90-95% HOLE, not a ceiling cap) -----------
def test_in_ambiguous_band_boundaries():
    assert not in_ambiguous_band(AMBIGUOUS_BAND_FLOOR - 0.001)
    assert in_ambiguous_band(AMBIGUOUS_BAND_FLOOR)
    assert in_ambiguous_band(AMBIGUOUS_BAND_CEILING - 0.001)
    assert not in_ambiguous_band(AMBIGUOUS_BAND_CEILING)  # 95% is a clear "Always"


def test_excluding_band_makes_92pct_spot_unworthy():
    """A 92% spot is worthy by frequency, but dropped by the band exclusion."""
    spot = _spot({"Raise": 0.92, "Fold": 0.08})
    assert is_question_worthy(spot, max_frequency=0.95) is True
    assert (
        is_question_worthy(spot, max_frequency=0.95, exclude_ambiguous_band=True)
        is False
    )


def test_band_exclusion_is_a_hole_keeps_pure_spots():
    """The exclusion punches a hole at 90-95% but KEEPS pure 95-100% spots
    when the window's max reaches them -- not a ceiling cap at 90%."""
    pure = _spot({"Raise": 0.97, "Fold": 0.03})
    band = _spot({"Raise": 0.92, "Fold": 0.08})
    assert is_question_worthy(pure, max_frequency=1.0, exclude_ambiguous_band=True) is True
    assert is_question_worthy(band, max_frequency=1.0, exclude_ambiguous_band=True) is False


def test_in_near_pure_band_boundaries():
    assert not in_near_pure_band(NEAR_PURE_BAND_FLOOR - 0.001)  # 94.9% is not near-pure
    assert in_near_pure_band(NEAR_PURE_BAND_FLOOR)  # 95%
    assert in_near_pure_band(0.99)
    assert not in_near_pure_band(NEAR_PURE_BAND_CEILING)  # a literal 100% is genuine "Always"


def test_excluding_near_pure_band_drops_97pct_keeps_92pct_and_pure():
    """The near-pure exclusion drops 95-99% but keeps a 92% spot and a literal
    100% spot (the genuine 'Always X')."""
    near_pure = _spot({"Raise": 0.97, "Fold": 0.03})
    mixed = _spot({"Raise": 0.92, "Fold": 0.08})
    pure = _spot({"Raise": 1.0, "Fold": 0.0})
    assert is_question_worthy(near_pure, max_frequency=1.0) is True  # off by default
    assert is_question_worthy(
        near_pure, max_frequency=1.0, exclude_near_pure_band=True
    ) is False
    assert is_question_worthy(
        mixed, max_frequency=1.0, exclude_near_pure_band=True
    ) is True
    assert is_question_worthy(
        pure, max_frequency=1.0, exclude_near_pure_band=True
    ) is True  # genuine 100% survives


# --- constants sanity ------------------------------------------------------
def test_default_constants_are_sensible():
    assert 0.0 < MIN_TOP_FREQUENCY < MAX_TOP_FREQUENCY < 1.0
    assert 0.0 < MIN_PRESENCE < 1.0
    # The ambiguous band sits just under the 95% "always" label line.
    assert MIN_TOP_FREQUENCY < AMBIGUOUS_BAND_FLOOR < MAX_TOP_FREQUENCY
