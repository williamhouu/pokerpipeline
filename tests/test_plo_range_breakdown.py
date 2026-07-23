"""Tests for pipeline/plo/range_breakdown.py -- the PLO hand-type range
breakdown (hero + every still-active opponent) and its deterministic copy.

Pure-logic edge cases are tested with constructed nodes (no pack); the
end-to-end aggregation + copy is pinned against the real 9-max pack when it is
present on the machine (skipped otherwise, like the other pack-backed tests).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.range_breakdown import (  # noqa: E402
    _Bucket,
    _category_for_cards,
    _hero_copy,
    _lead_actions,
    _round_actions,
    _still_active_villains,
    _villain_copy,
    _villain_role,
)

R, C, F = PloActionType.RAISE, PloActionType.CALL, PloActionType.FOLD


def _node(actor: str, history: tuple[PloAction, ...]) -> PloDecisionNode:
    return PloDecisionNode(
        actor=actor, history_before=history, actions=(), history_stem="",
        table_size=9,
    )


# --- category vocabulary ----------------------------------------------------
def test_category_labels_match_the_axis_vocabulary():
    assert _category_for_cards(("Ac", "Kc", "Ad", "Kd")) == "Two Pair Double-Suited"
    assert _category_for_cards(("Ac", "Kc", "Qd", "Jh")) == "Unpaired Single-Suited"
    assert _category_for_cards(("Ac", "Kd", "Qh", "Js")) == "Unpaired Rainbow"
    assert _category_for_cards(("Ac", "Ad", "Kh", "Qs")) == "One Pair Rainbow"
    assert _category_for_cards(("Ac", "Kc", "Qc", "Jd")) == "Unpaired Three-Suited"
    assert _category_for_cards(("Ac", "Kc", "Qc", "Jc")) == "Unpaired Monotone"
    assert _category_for_cards(("Ac", "Ad", "Ah", "Ks")) == "Trips Rainbow"
    assert _category_for_cards(("Ac", "Ad", "Ah", "Kc")) == "Trips Single-Suited"


# --- still-active villain detection -----------------------------------------
def test_still_active_villains_excludes_hero_and_folded_seats():
    # LJ opens, HJ/CO/BTN call, SB calls (hero), BB 3-bets, field folds back.
    node = _node("SB", (
        PloAction("UTG", F), PloAction("UTG+1", F), PloAction("UTG+2", F),
        PloAction("LJ", R), PloAction("HJ", C), PloAction("CO", C),
        PloAction("BTN", C), PloAction("SB", C), PloAction("BB", R),
        PloAction("LJ", F), PloAction("HJ", F), PloAction("CO", F),
        PloAction("BTN", F),
    ))
    seats = [s for s, _i, _a in _still_active_villains(node)]
    # Only BB is still live (the callers folded back; SB is hero).
    assert seats == ["BB"]


def test_still_active_villains_multiple_live_opponents():
    # UTG opens, CO calls, BTN calls, hero (BB) to act. Three-way.
    node = _node("BB", (
        PloAction("UTG", R), PloAction("UTG+1", F), PloAction("UTG+2", F),
        PloAction("LJ", F), PloAction("HJ", F), PloAction("CO", C),
        PloAction("BTN", C), PloAction("SB", F),
    ))
    result = _still_active_villains(node)
    seats = [s for s, _i, _a in result]
    assert seats == ["UTG", "CO", "BTN"]  # ordered by when they last acted
    # roles: UTG opened, CO/BTN called
    assert _villain_role(node, result[0][1], result[0][2]) == "open"
    assert _villain_role(node, result[1][1], result[1][2]) == "call"


def test_open_node_has_no_villains():
    assert _still_active_villains(_node("UTG", ())) == []


def test_villain_role_names_raise_levels():
    # UTG opens (1 raise), BB 3-bets (2 raises).
    node = _node("UTG", (
        PloAction("UTG", R), PloAction("UTG+1", F), PloAction("UTG+2", F),
        PloAction("LJ", F), PloAction("HJ", F), PloAction("CO", F),
        PloAction("BTN", F), PloAction("SB", F), PloAction("BB", R),
    ))
    assert _villain_role(node, 0, R) == "open"
    assert _villain_role(node, 8, R) == "3-bet"


# --- action-split rounding --------------------------------------------------
def test_round_actions_sums_to_100():
    assert sum(_round_actions({"Fold": 0.333, "Call": 0.333, "Raise": 0.334}).values()) == 100
    assert _round_actions({"Raise": 1.0}) == {"Raise": 100}
    assert _round_actions({}) == {}
    # zero-frequency actions drop out
    assert "Call" not in _round_actions({"Fold": 0.9, "Call": 0.0, "Raise": 0.1})


# --- copy: house phrasing + edge cases --------------------------------------
def test_hero_copy_open_and_hand_placement():
    buckets = [
        _Bucket("Unpaired Single-Suited", 38.0, {"Fold": 95, "Raise": 5}),
        _Bucket("One Pair Double-Suited", 9.5, {"Fold": 74, "Raise": 26}),
    ]
    copy = _hero_copy(buckets, "One Pair Double-Suited", is_open=True)
    assert copy.startswith("Your opening range is 38% unpaired single-suited")
    assert "a shape this range folds 74%, raises 26% here" in copy


def test_hero_copy_rare_shape():
    buckets = [_Bucket("Unpaired Single-Suited", 38.0, {"Fold": 95, "Call": 5})]
    copy = _hero_copy(buckets, "Unpaired Monotone", is_open=False)
    assert copy.startswith("Your continuing range is")
    assert "a rare shape in this range" in copy


def test_hero_copy_empty_range():
    copy = _hero_copy([], "One Pair Rainbow", is_open=True)
    assert "not enough of a range" in copy
    assert "one pair rainbow" in copy


def test_villain_copy_uses_seat_reference_verbatim():
    buckets = [
        _Bucket("One Pair Single-Suited", 41.0, None),
        _Bucket("Unpaired Double-Suited", 16.0, None),
    ]
    # "UTG" takes no article; "The Big Blind" keeps its article. "made up
    # of" (July 22 2026, user wording): plainer than the bare "is 41% ...".
    assert _villain_copy("UTG", "open", buckets) == (
        "UTG's opening range is made up of 41% one pair single-suited, "
        "16% unpaired double-suited."
    )
    assert _villain_copy("The Big Blind", "3-bet", buckets).startswith(
        "The Big Blind's 3-betting range is made up of 41% one pair single-suited"
    )


def test_villain_copy_unresolvable_range():
    assert "could not be broken down" in _villain_copy("The Button", "open", [])


def test_lead_actions_orders_by_frequency():
    b = _Bucket("x", 10.0, {"Fold": 33, "Call": 38, "4-bet": 29})
    assert _lead_actions(b) == "calls 38%, folds 33%, 4-bets 29%"


# --- end-to-end against the real pack ---------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_PACK_DIR = next(
    (_REPO / d for d in ("plo9_ranges", "plo_ranges") if (_REPO / d).is_dir()),
    None,
)


@pytest.mark.skipif(_PACK_DIR is None, reason="no PLO pack on this machine")
def test_end_to_end_open_and_facing_nodes():
    from pipeline.plo.fact_extractor import extract_plo_facts
    from pipeline.plo.node_enumerator import enumerate_plo_nodes
    from pipeline.plo.pack import discover_plo_pack
    from pipeline.plo.range_breakdown import build_range_breakdown
    from pipeline.plo.spot_sampler import sample_plo_spot

    pack = discover_plo_pack(_PACK_DIR)
    nodes = {n.node_id: n for n in enumerate_plo_nodes(pack)}

    def _bd(node_id: str, hero_index: int) -> dict:
        node = nodes[node_id]
        facts = extract_plo_facts(
            sample_plo_spot(node, hero_index), pack,
            compute_equity=False, rng=random.Random(0),
        )
        return build_range_breakdown(facts, pack)

    # Open node: hero-only, action split present, no villains.
    if "UTG_decision" in nodes:
        bd = _bd("UTG_decision", 100)
        assert bd["version"] == 1
        assert bd["villains"] == []
        assert bd["hero"]["role"] == "decision"
        assert bd["hero"]["buckets"], "open node should have hero buckets"
        assert "actions" in bd["hero"]["buckets"][0]  # hero has an action split
        assert bd["hero"]["summary"].startswith("Your opening range is")
        # JSON round-trips (it's a CSV cell)
        assert json.loads(json.dumps(bd))
        # Determinism
        assert _bd("UTG_decision", 100) == bd

    # A facing node with a live raiser: villain composition present, no split.
    facing = next(
        (nid for nid in nodes
         if nid.endswith("_BB_decision") and "raise100" in nid
         and nid.count("raise100") == 1 and "call" not in nid),
        None,
    )
    if facing:
        bd = _bd(facing, 300)
        assert bd["villains"], "facing node should surface at least one villain"
        v = bd["villains"][0]
        assert v["buckets"] and "actions" not in v["buckets"][0]  # composition only
        assert v["summary"].endswith(".")
        # every bucket category uses the axis vocabulary
        for pl in [bd["hero"], *bd["villains"]]:
            for b in pl["buckets"]:
                assert " " in b["category"]  # "<Pair> <Suit>"


# --- admin panel formatter (review.range_breakdown_panel) -------------------
# Pure display formatting for the "🃏 Range breakdown by hand type" expander;
# the Streamlit card is a thin shell over this (fix-durability rule).
def test_panel_formatter_empty_and_invalid_return_none():
    from admin_panel.review import range_breakdown_panel

    assert range_breakdown_panel("") is None
    assert range_breakdown_panel("   ") is None
    assert range_breakdown_panel("not json") is None
    assert range_breakdown_panel(json.dumps({"no": "hero"})) is None
    # A structurally-valid blob with no buckets renders nothing.
    assert range_breakdown_panel(json.dumps(
        {"hero": {"buckets": []}, "villains": []}
    )) is None


def test_panel_formatter_hero_and_villain_sections():
    from admin_panel.review import range_breakdown_panel

    cell = json.dumps({
        "version": 1,
        "hero": {
            "position": "SB", "role": "decision",
            "hand_category": "One Pair Single-Suited",
            "buckets": [
                {"category": "One Pair Single-Suited", "pct": 36.0,
                 "actions": {"Call": 38, "Fold": 33, "4-bet": 29}},
                {"category": "Unpaired Double-Suited", "pct": 18.0,
                 "actions": {"Fold": 50, "Call": 50}},
            ],
            "summary": "Your continuing range is 36% one pair single-suited.",
        },
        "villains": [{
            "position": "BB", "role": "3-bet",
            "buckets": [{"category": "One Pair Single-Suited", "pct": 41.0}],
            "summary": "The Big Blind's 3-betting range is 41% ...",
        }],
        "copy": "Your hand is one pair single-suited, 36% of the range.",
    })
    panel = range_breakdown_panel(cell)
    assert panel is not None
    assert panel["copy"].startswith("Your hand is one pair single-suited")
    hero, villain = panel["players"]
    assert hero["hero"] is True
    assert hero["heading"] == "Your range · SB"
    # Hero rows carry the action split, ordered by frequency.
    assert hero["rows"][0]["detail"] == "calls 38%, folds 33%, 4-bets 29%"
    # Villain section: composition only, no action split.
    assert villain["hero"] is False
    assert villain["heading"] == "BB · 3-betting range"
    assert villain["rows"][0]["detail"] == ""
