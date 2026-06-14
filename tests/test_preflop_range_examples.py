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
    leaning_groups_to_option,
)
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


def _mock_node(option_weights: dict[str, dict[str, float]]):
    """A node whose options' range files return ``option_weights[label]``."""
    from types import SimpleNamespace

    actions = [
        SimpleNamespace(label=label, range_file=SimpleNamespace(path=label))
        for label in option_weights
    ]
    return SimpleNamespace(actions=actions), option_weights


def test_leaning_groups_combo_weighted_with_guardrail(monkeypatch):
    # Call vs Fold. premium_broadways (AKo/AKs) split 50/50 -> a 2-member
    # group at exactly the floor; wheel aces lean fold (40% call) -> dropped;
    # QQ alone calls 80% -> single member, named as the hand (not "premium
    # pairs"). Combo weighting: AKo's 12 combos count 3x AKs's 4.
    node, weights = _mock_node({
        "Call": {"AKo": 0.5, "AKs": 0.5, "A5s": 0.4, "A4s": 0.4, "QQ": 0.8},
        "Fold": {"AKo": 0.5, "AKs": 0.5, "A5s": 0.6, "A4s": 0.6, "QQ": 0.2},
    })
    monkeypatch.setattr(
        range_examples, "_cached_parse_range_file", lambda path: weights[path]
    )
    out = leaning_groups_to_option(node, "Call", "call")
    assert "QQ (call 80%)" in out  # single present class -> named hand
    assert "premium broadways (call 50%)" in out  # 2 members -> group
    assert all("wheel aces" not in s for s in out)  # 40% < 50% floor -> dropped


def test_leaning_groups_skips_heros_own_bucket(monkeypatch):
    # Hero holds A5o (weak_offsuit_aces). A2o/A3o (same bucket) lean 3-bet, but
    # the bucket is skipped so the contrast never describes hero's own type;
    # small pairs (a different bucket) still shows.
    node, weights = _mock_node({
        "Call": {"A2o": 0.3, "A3o": 0.3, "22": 0.4, "33": 0.4},
        "3-bet": {"A2o": 0.7, "A3o": 0.7, "22": 0.6, "33": 0.6},
    })
    monkeypatch.setattr(
        range_examples, "_cached_parse_range_file", lambda path: weights[path]
    )
    out = leaning_groups_to_option(
        node, "3-bet", "3-bet", skip_bucket="weak_offsuit_aces"
    )
    assert all("offsuit aces" not in s for s in out)  # hero's bucket excluded
    assert any("small pairs" in s for s in out)  # other buckets still shown


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

    def _fake(node, raw_label, action_word, skip_bucket=None):
        seen["raw"], seen["word"] = raw_label, action_word
        return ("wheel aces (call 64%)",)

    monkeypatch.setattr(range_examples, "leaning_groups_to_option", _fake)
    out = leaning_examples_for_spot(
        _facts({"Fold": 0.93, "Call": 0.05, "Raise 76%": 0.02})
    )
    assert seen["raw"] == "Call"  # runner-up by hero's frequencies
    assert seen["word"] == "call"  # lowercased action for the group strings
    assert out == {"action": "Call", "hands": ["wheel aces (call 64%)"]}


def test_leaning_examples_none_paths(monkeypatch):
    assert leaning_examples_for_spot(_facts({"Fold": 1.0})) is None
    monkeypatch.setattr(
        range_examples, "leaning_groups_to_option", lambda *a, **k: ()
    )
    assert leaning_examples_for_spot(_facts({"Fold": 0.9, "Call": 0.1})) is None
