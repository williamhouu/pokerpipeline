"""Tests for pipeline.plo.question_extractor (worthiness gate)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.question_extractor import (  # noqa: E402
    effective_max_frequency,
    evaluate_spot,
    is_question_worthy,
)
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402


def _spot(*, freqs: dict[str, float], presence: float = 1.0) -> PloSpot:
    node = PloDecisionNode(actor="HJ", history_before=(), actions=(), history_stem="")
    return PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=("As", "Ks", "Ah", "Kh"),
        action_frequencies=freqs,
        presence=presence,
    )


def test_worthy_inside_window():
    assert is_question_worthy(_spot(freqs={"Call": 0.7, "Fold": 0.3}))


def test_unworthy_below_window():
    assert not is_question_worthy(_spot(freqs={"Call": 0.5, "Fold": 0.5}))


def test_unworthy_above_window():
    assert not is_question_worthy(_spot(freqs={"Call": 0.97, "Fold": 0.03}))


def test_unworthy_when_no_presence():
    assert not is_question_worthy(_spot(freqs={"Call": 0.7, "Fold": 0.3}, presence=0.005))


def test_effective_max_frequency_band_toggle():
    assert effective_max_frequency(0.95, exclude_ambiguous_band=True) == pytest.approx(0.90)
    assert effective_max_frequency(0.95, exclude_ambiguous_band=False) == pytest.approx(0.95)
    # No-op when the ceiling is already below the band floor.
    assert effective_max_frequency(0.70, exclude_ambiguous_band=True) == pytest.approx(0.70)


def test_difficulty_estimate_spans_the_freq_axis():
    hard = evaluate_spot(_spot(freqs={"Call": 0.55, "Fold": 0.45}))
    easy = evaluate_spot(_spot(freqs={"Call": 1.0}))
    assert hard.difficulty_estimate > easy.difficulty_estimate
    assert hard.difficulty_estimate == 3000  # noqa: PLR2004  # 55% -> hardest
    assert easy.difficulty_estimate == 500  # noqa: PLR2004  # 100% -> easiest


def test_evaluate_spot_reports_signals():
    ev = evaluate_spot(_spot(freqs={"Call": 0.7, "Fold": 0.3}))
    assert ev.is_worthy
    assert ev.top_action_frequency == pytest.approx(0.7)
    assert ev.total_presence == pytest.approx(1.0)
