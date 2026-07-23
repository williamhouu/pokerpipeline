"""Tests for pipeline/plo/balanced_select.py -- the fully-balanced batch
selector (greedy marginal balance over difficulty / situation / answer verb /
position / hand shape).

USER RULES pinned here (July 2026):
* the correct-answer verb axis balances fold vs call vs raise, and reads the
  solver's dominant action so basic AND GTO option styles classify the same;
* scarcity ships honestly (take what exists, report the shortfall) instead of
  silently unbalancing or blocking the batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.balanced_select import (  # noqa: E402
    BalanceAttrs,
    answer_verb,
    balance_report,
    balanced_order,
    difficulty_band,
    format_balance_report,
    hand_shape_family,
)


def _attrs(band="Easy", ctx="Opening", verb="fold", pos="early", shape="unpaired rainbow", node=""):
    return BalanceAttrs(
        difficulty_band=band, action_context=ctx, answer_verb=verb,
        position=pos, hand_shape=shape, node_id=node,
    )


def test_difficulty_band_edges_match_admin_presets():
    assert difficulty_band(400) == "Easy"
    assert difficulty_band(1299) == "Easy"
    assert difficulty_band(1300) == "Medium"
    assert difficulty_band(2099) == "Medium"
    assert difficulty_band(2100) == "Hard"
    assert difficulty_band(3200) == "Hard"


def test_answer_verb_is_option_style_independent():
    # The verb reads the RAW dominant action, so a spot classifies the same
    # whether its options render basic ("Call") or GTO ("Mostly Call").
    assert answer_verb("Fold") == "fold"
    assert answer_verb("Call") == "call/check"
    assert answer_verb("Check") == "call/check"
    for aggressive in ("Raise 72%", "Raise 100%", "3-bet", "4-bet", "All-in", "Min-raise"):
        assert answer_verb(aggressive) == "raise"


def test_hand_shape_family_is_coarse_and_total():
    assert hand_shape_family("unpaired", "double_suited") == "unpaired double-suited"
    assert hand_shape_family("one_pair", "single_suited") == "paired suited"
    assert hand_shape_family("two_pair", "rainbow") == "paired rainbow"
    assert hand_shape_family("unpaired", "monotone") == "unpaired suited"
    assert hand_shape_family("trips", "three_suited") == "paired suited"


def test_balanced_order_flattens_a_skewed_pool():
    # Pool: 60 spots, heavily skewed -- 40 Easy/fold/Opening, 10 Medium/call,
    # 10 Hard/raise. A raw draw of 12 would be ~8 Easy folds; balanced must
    # come out ~4/4/4 on difficulty AND verb.
    pool = (
        [_attrs("Easy", "Opening", "fold") for _ in range(40)]
        + [_attrs("Medium", "Facing single raise", "call/check") for _ in range(10)]
        + [_attrs("Hard", "Facing 3-bet", "raise") for _ in range(10)]
    )
    order = balanced_order(pool, 12)
    assert sorted(order) == list(range(60))  # every index exactly once
    first = [pool[i] for i in order[:12]]
    bands = [a.difficulty_band for a in first]
    verbs = [a.answer_verb for a in first]
    assert bands.count("Easy") == 4
    assert bands.count("Medium") == 4
    assert bands.count("Hard") == 4
    assert verbs.count("fold") == 4
    assert verbs.count("call/check") == 4
    assert verbs.count("raise") == 4


def test_balanced_order_scarcity_ships_what_exists():
    # Only 2 Hard spots exist: both are picked (no silent skip), the batch
    # fills the rest from Easy/Medium, and the report shows the shortfall.
    pool = (
        [_attrs("Easy", verb="fold") for _ in range(20)]
        + [_attrs("Medium", verb="call/check") for _ in range(20)]
        + [_attrs("Hard", verb="raise") for _ in range(2)]
    )
    order = balanced_order(pool, 12)
    first = [pool[i] for i in order[:12]]
    bands = [a.difficulty_band for a in first]
    assert bands.count("Hard") == 2  # everything the pool had
    assert len(first) == 12  # batch still fills
    report = balance_report(first, pool)
    hard = next(
        v
        for ax in report["axes"]
        if ax["axis"] == "difficulty_band"
        for v in ax["values"]
        if v["value"] == "Hard"
    )
    assert hard == {"value": "Hard", "achieved": 2, "target": 4.0, "pool": 2}
    # The done-panel line says it plainly.
    line = next(l for l in format_balance_report(report) if l.startswith("Difficulty"))
    assert "Hard 2 (pool only had 2)" in line


def test_balanced_order_is_deterministic():
    pool = (
        [_attrs("Easy", "Opening", "fold", "early") for _ in range(15)]
        + [_attrs("Hard", "Facing 3-bet", "raise", "bb") for _ in range(15)]
    )
    assert balanced_order(pool, 10) == balanced_order(pool, 10)


def test_balanced_order_spreads_across_nodes():
    # Same attrs everywhere; the only signal is node reuse -- the selector
    # must rotate across distinct nodes before repeating one.
    pool = [_attrs(node=f"n{i % 4}") for i in range(16)]
    order = balanced_order(pool, 8)
    first_nodes = [pool[i].node_id for i in order[:4]]
    assert sorted(first_nodes) == ["n0", "n1", "n2", "n3"]


def test_balanced_order_remainder_keeps_balancing():
    # Backfill draws (after the intended count) continue the greedy rule:
    # the 13th pick still chases the lagging axis value, so failures that
    # trigger backfill do not unbalance the batch.
    pool = (
        [_attrs("Easy", verb="fold") for _ in range(30)]
        + [_attrs("Hard", verb="raise") for _ in range(30)]
    )
    order = balanced_order(pool, 12)
    first14 = [pool[i].difficulty_band for i in order[:14]]
    assert first14.count("Easy") == 7
    assert first14.count("Hard") == 7


def test_empty_pool_is_safe():
    assert balanced_order([], 10) == []
    report = balance_report([], [])
    assert report["selected"] == 0 and report["pool"] == 0
    lines = format_balance_report(report)
    assert len(lines) == 5  # one per axis, no crash on an empty pool
