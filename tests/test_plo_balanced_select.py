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


# --- Always/Mostly qualifier axis (Aug 2026, user ask) ------------------------
# 44% Always / 56% Mostly overall with strong per-verb skews let players
# meta-game the prefix; the axis evens it out. RULE pinned here: the
# qualifier value comes from the SOLVER's dominant-action frequency
# (pipeline.plo.options.answer_qualifier), never from option text, and the
# axis is only active for GTO-capable answer styles.

def _qattrs(qualifier: str, i: int) -> BalanceAttrs:
    """All non-qualifier axes constant, distinct node ids (no spread ties)."""
    return BalanceAttrs(
        difficulty_band="Easy", action_context="Opening", answer_verb="fold",
        position="early", hand_shape="unpaired rainbow", node_id=f"n{i}",
        qualifier=qualifier,
    )


def test_balance_axes_qualifier_slots_below_answer_verb():
    from pipeline.plo.balanced_select import (
        BALANCE_AXES,
        QUALIFIER_AXIS,
        balance_axes,
    )

    assert balance_axes(False) is BALANCE_AXES  # inactive = untouched schema
    active = balance_axes(True)
    keys = [k for k, _l, _w in active]
    assert keys == [
        "difficulty_band", "action_context", "answer_verb",
        "qualifier", "position", "hand_shape",
    ]
    weights = {k: w for k, _l, w in active}
    # Slotted between answer verb (0.80) and position (0.50).
    assert weights["answer_verb"] > weights["qualifier"] > weights["position"]
    assert QUALIFIER_AXIS == ("qualifier", "Always/Mostly", 0.70)


def test_qualifier_axis_flattens_a_skewed_pool_when_active():
    # Pool skewed 80/20 Mostly/Always; a raw front-of-pool draw of 8 would be
    # 8 Mostly. With the axis active the pick alternates to an even 4/4.
    pool = [_qattrs("Mostly", i) for i in range(16)] + [
        _qattrs("Always", 16 + i) for i in range(4)
    ]
    order = balanced_order(pool, 8, include_qualifier=True)
    first = [pool[i].qualifier for i in order[:8]]
    assert first.count("Always") == 4
    assert first.count("Mostly") == 4


def test_qualifier_axis_inactive_is_byte_identical_to_before():
    # Same skewed pool, axis OFF (basic answer style): the qualifier field is
    # carried but IGNORED, so with every active axis constant the ordering is
    # exactly the pool order -- the pre-change behaviour.
    pool = [_qattrs("Mostly", i) for i in range(16)] + [
        _qattrs("Always", 16 + i) for i in range(4)
    ]
    order = balanced_order(pool, 8, include_qualifier=False)
    assert order == list(range(20))
    first = [pool[i].qualifier for i in order[:8]]
    assert first.count("Always") == 0  # the skew ships untouched


def test_balance_report_includes_qualifier_axis_only_when_active():
    pool = [_qattrs("Mostly", i) for i in range(3)] + [_qattrs("Always", 3)]
    on = balance_report(pool[:2], pool, include_qualifier=True)
    off = balance_report(pool[:2], pool, include_qualifier=False)
    assert "qualifier" in [ax["axis"] for ax in on["axes"]]
    assert "qualifier" not in [ax["axis"] for ax in off["axes"]]
    # The generic done-panel renderer picks the axis up with no extra code.
    line = next(
        l for l in format_balance_report(on) if l.startswith("Always/Mostly")
    )
    assert "Mostly" in line
