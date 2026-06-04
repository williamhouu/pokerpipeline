"""Tests for pipeline.plo.options (deterministic answer options)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.fact_extractor import PloFacts  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.options import (  # noqa: E402
    build_options,
    canonicalize_action_label,
)
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

R = PloActionType.RAISE
CARDS = ("As", "Ks", "Ah", "Kh")


def _facts(freqs: dict[str, float], history: tuple[PloAction, ...] = ()) -> PloFacts:
    node = PloDecisionNode(
        actor="HJ", history_before=history, actions=(), history_stem=""
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=CARDS,
        action_frequencies=freqs,
        presence=1.0,
    )
    return PloFacts(spot=spot, hand_class=classify_plo_hand(CARDS), archetype="")


def test_canonicalize_label_by_raise_level():
    assert canonicalize_action_label("Raise 100%", raise_level=1) == "Raise"
    assert canonicalize_action_label("Raise 100%", raise_level=2) == "3-bet"
    assert canonicalize_action_label("Raise 100%", raise_level=3) == "4-bet"
    assert canonicalize_action_label("All-in", raise_level=1) == "All-in"
    assert canonicalize_action_label("Call", raise_level=1) == "Call"


def test_basic_is_fold_first_with_canonical_dominant():
    opts, correct = build_options(
        _facts({"Call": 0.7, "Fold": 0.0, "Raise 100%": 0.3}), style="basic"
    )
    assert correct == "Call"
    assert opts[0] == "Fold"  # least-aggressive first
    assert set(opts) == {"Fold", "Call", "Raise"}


def test_gto_spectrum_orders_by_aggression():
    # Mixed Call/raise; one prior raise -> hero's raise is a 3-bet.
    facts = _facts({"Call": 0.6, "Raise 100%": 0.4}, history=(PloAction("LJ", R, 100),))
    opts, correct = build_options(facts, style="gto")
    assert opts == ["Always Call", "Mostly Call", "Mostly 3-bet", "Always 3-bet"]
    assert correct == "Mostly Call"


def test_auto_uses_basic_when_dominant_and_gto_when_mixed():
    dominant = build_options(_facts({"Raise 100%": 0.9, "Fold": 0.1}), style="auto")
    assert dominant[1] == "Raise"
    assert all("Always" not in o and "Mostly" not in o for o in dominant[0])

    mixed = build_options(_facts({"Call": 0.6, "Fold": 0.4}), style="auto")
    assert mixed[0][0].startswith("Always")


@pytest.mark.parametrize("style", ["basic", "gto", "auto"])
@pytest.mark.parametrize(
    "freqs",
    [
        {"Call": 0.7, "Fold": 0.3},
        {"Raise 100%": 0.6, "Call": 0.4},
        {"Fold": 1.0},
        {"Call": 0.5, "Raise 100%": 0.3, "Fold": 0.2},
    ],
)
def test_correct_answer_is_always_one_of_the_options(style, freqs):
    opts, correct = build_options(_facts(freqs), style=style)
    assert correct in opts
    assert 1 <= len(opts) <= 4  # noqa: PLR2004


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="unknown answer style"):
        build_options(_facts({"Call": 1.0}), style="bogus")
