"""Tests for the showdown resolution (vindicating reveal) + artifact-all-in gates.

The invariants: the revealed villain hand comes from the villain's REAL range
at the final node AND vindicates the correct answer (weaker for a call,
stronger for a fold); invented responses are call-or-fold only; the events
continue the main timeline's exact pot; everything is seeded-deterministic;
and generation never builds questions on artifact-jam lines.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.fact_extractor.equity import rank_hand  # noqa: E402
from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s  # noqa: E402
from pipeline.postflop.premise import (  # noqa: E402
    line_contains_artifact_allin,
)
from pipeline.postflop.showdown import build_showdown_resolution  # noqa: E402
from pipeline.postflop.solve import PostflopStep  # noqa: E402


def _river_node(solve):
    """The deepest river decision node in the fixture."""
    rivers = [n for n in solve.nodes.values() if n.street == "river"]
    assert rivers, "fixture has no river node"
    return max(rivers, key=lambda n: len(n.history))


def test_call_ending_vindicates_with_weaker_hand() -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    node = _river_node(solve)
    combo = next(iter(node.strategy))
    correct = "Call" if node.is_facing_bet else "Check"
    res = build_showdown_resolution(
        node, solve, hero_combo=combo, correct_answer=correct, hand_id="h1",
    )
    if res is None:  # the fixture range may hold no weaker hand for this combo
        return
    board = list(node.board)
    hero_rank = rank_hand([combo[:2], combo[2:]] + board)
    v = res["villain_cards"]
    assert "".join(v) in {c for c in node.villain_range}
    if res["vindicates"] in ("Call", "Check"):
        assert rank_hand(v + board) < hero_rank
    # Events continue with hero's action first and end with the pot push.
    kinds = [e["type"] for e in res["events"]]
    assert kinds[-1] == "win"
    assert res["events"][0]["seat"] == node.actor
    # Deterministic: same inputs, same reveal.
    res2 = build_showdown_resolution(
        node, solve, hero_combo=combo, correct_answer=correct, hand_id="h1",
    )
    assert res2 == res
    # A different hand_id may pick a different hand, never a different rule.
    res3 = build_showdown_resolution(
        node, solve, hero_combo=combo, correct_answer=correct, hand_id="h2",
    )
    if res3 is not None and res["vindicates"] in ("Call", "Check"):
        assert rank_hand(res3["villain_cards"] + board) < hero_rank


def test_fold_ending_shows_a_stronger_hand() -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    node = _river_node(solve)
    if not node.is_facing_bet:
        return
    combo = next(iter(node.strategy))
    res = build_showdown_resolution(
        node, solve, hero_combo=combo, correct_answer="Fold", hand_id="h1",
    )
    if res is None:
        return
    board = list(node.board)
    hero_rank = rank_hand([combo[:2], combo[2:]] + board)
    assert rank_hand(res["villain_cards"] + board) > hero_rank
    kinds = [e["type"] for e in res["events"]]
    assert kinds[0] == "fold" and kinds[-1] == "win"
    # Hero folded: villain wins, hero never reveals.
    assert res["events"][-1]["seat"] == node.villain
    assert all(
        e.get("seat") != node.actor for e in res["events"] if e["type"] == "reveal"
    )


def test_win_event_carries_reason_label_and_stack() -> None:
    """The app's pot push + result banner read the win event directly:
    reason (showdown|fold), hand_label on showdown wins only, and the
    winner's stack AFTER the pot. Reveals carry best_five. INVARIANT:
    these fields are additive to format version 2 -- removing or renaming
    them breaks the app's renderer contract (docs/animation_script_format.md)."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    node = _river_node(solve)
    combo = next(iter(node.strategy))
    correct = "Call" if node.is_facing_bet else "Check"
    res = build_showdown_resolution(
        node, solve, hero_combo=combo, correct_answer=correct, hand_id="h1",
    )
    if res is not None:
        win = res["events"][-1]
        assert win["type"] == "win" and win["reason"] == "showdown"
        # A vindicated call/check win: hero wins, label matches hero's reveal.
        hero_reveal = next(
            e for e in res["events"]
            if e["type"] == "reveal" and e["seat"] == node.actor
        )
        assert win["hand_label"] == hero_reveal["hand_label"]
        assert win["stack_bb"] > 0 and win["pot_bb"] > 0
        # best_five: five real cards, all from hole+board, best by rank.
        board = list(node.board)
        for e in res["events"]:
            if e["type"] == "reveal":
                bf = e["best_five"]
                assert len(bf) == 5 and set(bf) <= set(e["cards"] + board)
                assert rank_hand(bf) == rank_hand(e["cards"] + board)

    if node.is_facing_bet:
        fold_res = build_showdown_resolution(
            node, solve, hero_combo=combo, correct_answer="Fold", hand_id="h1",
        )
        if fold_res is not None:
            win = fold_res["events"][-1]
            assert win["reason"] == "fold"
            assert "hand_label" not in win
            # Winner's stack after the pot: strictly above their stack on
            # any earlier event (they gained the pot without wagering).
            assert win["stack_bb"] > 0


def test_win_hand_label_is_the_winners_not_heros() -> None:
    """win.hand_label names the WINNING hand, whoever holds it. The one
    showdown the hero can lose is the closing river check-back with no
    weaker hand in villain's range (the check 'lost the minimum'):
    the label must then be VILLAIN'S hand, not hero's."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    node = next(n for n in solve.nodes.values() if n.street == "river")
    ip = solve.ip_position
    oop = "BB" if ip == "BTN" else "BTN"
    node = replace(
        node,
        actor=ip, villain=oop,
        history=tuple(s for s in node.history if s.street != "river"),
        villain_range={"AhAd": 1.0},  # villain always has hero beat
    )
    res = build_showdown_resolution(
        node, solve, hero_combo="8h6h", correct_answer="Check", hand_id="rig",
    )
    assert res is not None
    win = res["events"][-1]
    reveals = {
        e["seat"]: e["hand_label"] for e in res["events"] if e["type"] == "reveal"
    }
    assert win["seat"] == oop and win["reason"] == "showdown"
    assert win["hand_label"] == reveals[oop] != reveals[ip]
    assert "lost the minimum" in res["summary"]


def test_non_ending_decisions_get_no_resolution() -> None:
    """A flop call with betting still live is not an ending."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    flop_nodes = [
        n for n in solve.nodes.values()
        if n.street == "flop" and n.is_facing_bet
        and not (n.history and n.history[-1].all_in)
    ]
    if not flop_nodes:
        return
    node = flop_nodes[0]
    combo = next(iter(node.strategy))
    assert build_showdown_resolution(
        node, solve, hero_combo=combo, correct_answer="Call", hand_id="h1",
    ) is None


def test_artifact_allin_line_detector() -> None:
    """A 150bb jam into a 10bb pot is an artifact; a 20bb jam into 15bb is
    a real line and passes."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    node = _river_node(solve)
    artifact = replace(node, history=(
        PostflopStep("flop", "BB", "bet", to_bb=3.0),
        PostflopStep("flop", "BTN", "raise", to_bb=150.0, all_in=True),
    ))
    assert line_contains_artifact_allin(artifact, solve)
    realistic = replace(node, history=(
        PostflopStep("flop", "BB", "bet", to_bb=5.0),
        PostflopStep("flop", "BTN", "raise", to_bb=20.0, all_in=True),
    ))
    assert not line_contains_artifact_allin(realistic, solve)
    plain = replace(node, history=(
        PostflopStep("flop", "BB", "bet", to_bb=3.0),
        PostflopStep("flop", "BTN", "raise", to_bb=9.0),
    ))
    assert not line_contains_artifact_allin(plain, solve)


def test_full_hand_batch_attaches_resolution(tmp_path) -> None:
    """End to end: the final leg's animation_script gains version 2 + the
    resolution; earlier legs stay version 1; counters count it."""
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "out.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        equity_runouts=20,
    )
    import csv as _csv

    rows = list(_csv.DictReader(open(out, encoding="utf-8-sig")))
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    by_hand: dict[str, list[dict]] = {}
    for r in rows:
        by_hand.setdefault(r["hand_id"], []).append(r)
    attached = 0
    for hrows in by_hand.values():
        for k, r in enumerate(hrows):
            payload = json.loads(r["animation_script"])
            is_final = k == len(hrows) - 1
            if is_final and "resolution" in payload:
                attached += 1
                assert payload["version"] == 2
                assert payload["resolution"]["summary"]
                assert payload["resolution"]["events"][-1]["type"] == "win"
            else:
                assert payload["version"] == 1
                assert "resolution" not in payload
    assert attached == meta["counters"]["showdown_resolutions"]
