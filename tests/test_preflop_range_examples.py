"""Tests for pipeline.preflop.range_examples (NLHE leaning-hand examples)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop import range_examples  # noqa: E402
from pipeline.preflop.fact_extractor import PreflopFacts  # noqa: E402
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.range_examples import (  # noqa: E402
    format_examples,
    leaning_examples_for_spot,
)
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


def test_format_examples_band_cap_and_wording():
    rows = [
        (0.9, 0.99, "AA"),  # too pure -> obvious, excluded
        (0.9, 0.70, "AJo"),  # in band, high presence -> "mostly"
        (0.02, 0.70, "K2s"),  # presence floor -> excluded
        (0.5, 0.50, "T9s"),  # in band -> "often"
        (0.3, 0.30, "72o"),  # leans the wrong way -> out
    ]
    assert format_examples(rows, "calls") == [
        "AJo (mostly calls)",
        "T9s (often calls)",
    ]


def test_format_examples_caps_at_three():
    rows = [(0.9 - i * 0.1, 0.8, h) for i, h in enumerate(["AA", "KK", "QQ", "JJ"])]
    assert len(format_examples(rows, "folds")) == 3


def _facts(freqs: dict[str, float]) -> PreflopFacts:
    node = PreflopDecisionNode(
        pack_id="t", actor="BTN", history_before=(), actions=()
    )
    spot = PreflopSpot(
        node=node,
        hero_hand_class="AJo",
        hero_card_combo="AhJc",
        action_frequencies=freqs,
        dominant_action=max(freqs, key=lambda k: freqs[k]),
        dominant_frequency=max(freqs.values()),
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=None,
        hero_equity_vs_villain=None,
        hero_range_equity_vs_villain=None,
        blockers={},
        archetype="open_for_value",
    )


def test_leaning_examples_picks_runner_up(monkeypatch):
    seen: dict[str, str] = {}

    def _fake(node, raw_label, verb):
        seen["raw"], seen["verb"] = raw_label, verb
        return ("KQs (mostly calls)",)

    monkeypatch.setattr(range_examples, "hands_leaning_to_option", _fake)
    out = leaning_examples_for_spot(
        _facts({"Fold": 0.93, "Call": 0.05, "Raise 76%": 0.02})
    )
    assert seen["raw"] == "Call"  # runner-up by hero's frequencies
    assert out == {"action": "Call", "hands": ["KQs (mostly calls)"]}


def test_leaning_examples_none_paths(monkeypatch):
    assert leaning_examples_for_spot(_facts({"Fold": 1.0})) is None
    monkeypatch.setattr(
        range_examples, "hands_leaning_to_option", lambda *a: ()
    )
    assert leaning_examples_for_spot(_facts({"Fold": 0.9, "Call": 0.1})) is None
