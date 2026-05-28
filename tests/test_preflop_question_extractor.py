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


# --- difficulty_score with EV-gap blend ------------------------------------
# May-2026: difficulty_score now accepts an optional ev_gap_bb. When None
# (raise-involved spots in v1) the formula falls back to freq-only --
# tested by the freq-only cases above. The cases below cover the blend.


def test_difficulty_with_ev_gap_at_high_freq_easy_spot():
    """95% freq + 3bb gap = very easy spot. easy = 0.6*0.889 + 0.4*1.0
    = 0.933 -> score = 3000 - 0.933*2500 = 666."""
    spot = _spot({"Raise": 0.95, "Fold": 0.05})
    score = difficulty_score(spot, ev_gap_bb=3.0)
    assert 650 < score < 700


def test_difficulty_with_zero_ev_gap_drops_score():
    """High freq (95%) but EV gap of 0 = pure-ish action but mistake
    is costless. Blend pulls "easy" down: easy = 0.6*0.889 + 0.4*0 =
    0.533 -> score = 3000 - 1333 = 1667. Significantly harder than
    the freq-only 778."""
    spot = _spot({"Raise": 0.95, "Fold": 0.05})
    score = difficulty_score(spot, ev_gap_bb=0.0)
    assert 1640 < score < 1700


def test_difficulty_with_big_gap_at_low_freq_still_hard():
    """55% freq + 3bb gap. Freq says "hard"; gap says "easy". The
    weighted blend lands around 2000 -- still on the hard half because
    the dominant signal (freq) is at the floor."""
    spot = _spot({"Raise": 0.55, "Fold": 0.45})
    score = difficulty_score(spot, ev_gap_bb=3.0)
    # easy = 0.6 * 0 + 0.4 * 1 = 0.4 -> score = 3000 - 1000 = 2000.
    assert 1950 < score < 2050


def test_difficulty_at_users_real_spot_3():
    """User's batch No.3: BTN 3c3d facing BB 3-bet. freq=0.66, gap=1.37.
    easy = 0.6*0.244 + 0.4*0.457 = 0.330 -> score ~2175.
    This was 2389 freq-only; EV gap pulled it ~210 points easier."""
    spot = _spot({"Fold": 0.66, "Call": 0.34})
    score = difficulty_score(spot, ev_gap_bb=1.37)
    assert 2150 < score < 2200


def test_difficulty_at_users_real_spot_5():
    """User's batch No.5: HJ As8s facing CO 3-bet. freq=0.81, gap=1.38.
    easy = 0.6*0.578 + 0.4*0.46 = 0.531 -> score ~1672.
    This was 1556 freq-only; EV gap pulled it ~115 points HARDER --
    high freq alone overstated easiness given the modest gap."""
    spot = _spot({"Fold": 0.81, "Call": 0.11, "4-bet": 0.08})
    score = difficulty_score(spot, ev_gap_bb=1.38)
    assert 1640 < score < 1700


def test_difficulty_ev_gap_capped_at_3bb():
    """EV gap of 10bb gives the SAME contribution as 3bb (the cap).
    Stops a rare blown-out spot from pushing difficulty below the floor."""
    spot = _spot({"Raise": 0.95, "Fold": 0.05})
    score_at_cap = difficulty_score(spot, ev_gap_bb=3.0)
    score_well_above = difficulty_score(spot, ev_gap_bb=10.0)
    assert score_at_cap == score_well_above


def test_difficulty_negative_ev_gap_clamped_to_zero():
    """Defensive: a negative gap shouldn't break the formula. Clamps
    to 0bb contribution."""
    spot = _spot({"Raise": 0.75, "Fold": 0.25})
    score_zero = difficulty_score(spot, ev_gap_bb=0.0)
    score_negative = difficulty_score(spot, ev_gap_bb=-1.0)
    assert score_zero == score_negative


def test_difficulty_none_ev_gap_matches_freq_only():
    """Backward compat: difficulty_score(spot) == difficulty_score(spot,
    ev_gap_bb=None). Old callers (and raise-involved spots in v1) get
    the freq-only score."""
    spot = _spot({"Raise": 0.75, "Fold": 0.25})
    assert difficulty_score(spot) == difficulty_score(spot, ev_gap_bb=None)


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


# --- constants sanity ------------------------------------------------------
def test_default_constants_are_sensible():
    assert 0.0 < MIN_TOP_FREQUENCY < MAX_TOP_FREQUENCY < 1.0
    assert 0.0 < MIN_PRESENCE < 1.0
