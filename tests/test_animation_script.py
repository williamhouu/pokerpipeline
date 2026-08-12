"""Tests for the ``animation_script`` column (the app's animation timeline).

The invariants that matter to the renderer: events are ordered with stable
indices, every seat that folds gets its OWN fold event at the right point in
the rotation (the SB's fold comes AFTER the open, not with the early seats),
chip events carry pot+stack AFTER the event, streets are dealt as explicit
events, and the timeline ends at the question's decision seat. All pure and
deterministic.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.postflop.animation_script import (  # noqa: E402
    build_postflop_animation_script,
    build_preflop_animation_script,
)
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.solve import PreflopStep  # noqa: E402


def _events(script: str) -> tuple[dict, list[dict]]:
    payload = json.loads(script)
    return payload, payload["events"]


def _three_bet(solve):
    return replace(solve, preflop_summary=(
        PreflopStep("BTN", "open", to_bb=2.5),
        PreflopStep("BB", "3-bet", to_bb=11.0),
        PreflopStep("BTN", "call"),
    ), table_size=8, effective_stack_bb=200.0)


def test_srp_postflop_timeline_shape() -> None:
    solve = btn_vs_bb_srp_2cJs7s()  # 6-max SRP, BTN opens 2.5, BB calls
    node = solve.nodes["r0"] if "r0" in solve.nodes else next(iter(solve.nodes.values()))
    payload, events = _events(build_postflop_animation_script(node, solve))
    assert payload["version"] == 1
    assert payload["hero_seat"] == node.actor
    kinds = [e["type"] for e in events]
    # Blinds first, then per-seat folds, the open, SB's fold, BB's call,
    # the flop deal, and finally the decision.
    assert kinds[:2] == ["post", "post"]
    assert events[0]["seat"] == "SB" and events[0]["amount_bb"] == 0.5
    assert events[1]["seat"] == "BB" and events[1]["pot_bb"] == 1.5
    assert kinds[-1] == "decision"
    # Indices are stable and 1-based contiguous.
    assert [e["i"] for e in events] == list(range(1, len(events) + 1))
    # Every early seat folds INDIVIDUALLY (6-max: UTG, HJ, CO before BTN).
    fold_seats = [e["seat"] for e in events if e["type"] == "fold"]
    assert fold_seats[:3] == ["UTG", "HJ", "CO"]
    # The open: amount added == to amount (nothing invested before).
    open_ev = next(e for e in events if e["type"] == "raise")
    assert open_ev["seat"] == "BTN" and open_ev["to_bb"] == 2.5
    assert open_ev["pot_bb"] == 4.0  # 1.5 blinds + 2.5
    # SB folds AFTER the open (order preserved), then BB calls 1.5 more.
    sb_fold_i = next(e["i"] for e in events if e["type"] == "fold" and e["seat"] == "SB")
    assert sb_fold_i > open_ev["i"]
    call_ev = next(e for e in events if e["type"] == "call")
    assert call_ev["seat"] == "BB" and call_ev["amount_bb"] == 1.5
    # Pot after the preflop line: 0.5 dead SB + 2.5 + 2.5 -- and it MUST
    # equal the solve's own starting_pot_bb (the timeline can never disagree
    # with the pot the questions are built on).
    assert call_ev["pot_bb"] == 5.5 == solve.starting_pot_bb
    rerun = json.loads(build_postflop_animation_script(node, solve))
    assert rerun["events"][call_ev["i"] - 1]["pot_bb"] == 5.5  # deterministic
    # The flop deal is an explicit event with the three cards.
    deal = next(e for e in events if e["type"] == "deal")
    assert deal["street"] == "flop" and len(deal["cards"]) == 3


def test_three_bet_pot_two_orbits_and_stacks() -> None:
    solve = _three_bet(btn_vs_bb_srp_2cJs7s())
    node = next(iter(solve.nodes.values()))
    _payload, events = _events(build_postflop_animation_script(node, solve))
    raises = [e for e in events if e["type"] == "raise"]
    assert [(e["seat"], e["to_bb"]) for e in raises] == [("BTN", 2.5), ("BB", 11.0)]
    # BTN's call of the 3-bet adds 8.5 (11 - 2.5 already in).
    call = next(e for e in events if e["type"] == "call")
    assert call["seat"] == "BTN" and call["amount_bb"] == 8.5
    # Pot: 0.5 (dead SB) + 11 + 11 = 22.5; BTN stack 200 - 11 = 189.
    assert call["pot_bb"] == 22.5 and call["stack_bb"] == 189.0
    # 8-max: five seats fold before the open, SB folds after it.
    fold_seats = [e["seat"] for e in events if e["type"] == "fold"]
    assert fold_seats == ["UTG", "UTG+1", "LJ", "HJ", "CO", "SB"]


def test_preflop_leg_stops_at_the_right_decision() -> None:
    solve = _three_bet(btn_vs_bb_srp_2cJs7s())
    # BTN's FIRST decision (the open): folds to BTN, then pause.
    _p, ev = _events(build_preflop_animation_script(solve, "BTN", step_index=0))
    assert ev[-1] == {"i": ev[-1]["i"], "type": "decision", "seat": "BTN",
                      "street": "preflop"}
    assert [e["type"] for e in ev].count("raise") == 0
    # BB's decision (the 3-bet): the open happened, SB folded, then pause.
    _p, ev = _events(build_preflop_animation_script(solve, "BB", step_index=1))
    assert [e["seat"] for e in ev if e["type"] == "raise"] == ["BTN"]
    assert ev[-1]["seat"] == "BB"
    # BTN's SECOND decision (facing the 3-bet): both raises animated.
    _p, ev = _events(build_preflop_animation_script(solve, "BTN", step_index=2))
    assert [e["seat"] for e in ev if e["type"] == "raise"] == ["BTN", "BB"]
    assert ev[-1]["seat"] == "BTN"
    # SRP entry default (step_index=None): hero's first step.
    srp = btn_vs_bb_srp_2cJs7s()
    _p, ev = _events(build_preflop_animation_script(srp, "BB"))
    assert ev[-1]["seat"] == "BB" and ev[-1]["type"] == "decision"


def test_consecutive_leg_timelines_are_prefix_extensions() -> None:
    """The full-hand fast-forward contract, documented for the app team in
    docs/animation_script_format.md: drop a leg's terminal ``decision`` event
    and what remains is an EXACT prefix of the next leg's event list (the
    decision is replaced by the action the hand actually took), so a renderer
    may play only the new tail. INVARIANT: any change that breaks this must
    update the format doc and bump the renderer guidance."""
    solve = btn_vs_bb_srp_2cJs7s()
    legs = [
        _events(build_preflop_animation_script(solve, "BB"))[1],
        _events(build_postflop_animation_script(
            solve.nodes["flop_oop_lead"], solve))[1],
        _events(build_postflop_animation_script(
            solve.nodes["turn_oop"], solve))[1],
    ]
    for prev, nxt in zip(legs, legs[1:]):
        assert prev[-1]["type"] == "decision"
        body = prev[:-1]
        assert nxt[:len(body)] == body
        # The event replacing the decision is the hero acting at that spot.
        assert nxt[len(body)]["seat"] == prev[-1]["seat"]


def test_dollar_twins_only_on_cash() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    node = next(iter(solve.nodes.values()))
    payload, events = _events(build_postflop_animation_script(node, solve))
    open_ev = next(e for e in events if e["type"] == "raise")
    assert open_ev["to_dollars"] == round(2.5 * solve.bb_in_dollars, 2)
    assert payload["bb_dollars"] == solve.bb_in_dollars
    mtt = replace(solve, game_format="tournament")
    payload_t, events_t = _events(build_postflop_animation_script(node, mtt))
    assert "bb_dollars" not in payload_t
    assert all("to_dollars" not in e for e in events_t)


def test_preflop_writer_animation_from_pack_history(tmp_path: Path) -> None:
    """The standalone-preflop writer's timeline: pack histories carry the
    folds EXPLICITLY (no synthesis), sizes come from the same resolution as
    the prose, and the timeline pauses at hero's decision."""
    from types import SimpleNamespace

    from pipeline.preflop.format_writer import _preflop_animation_script
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from tests.test_full_hand_pack_legs import _matching_pack

    pack = _matching_pack(tmp_path)
    # The BB facing BTN's open (all others folded) -- the defend node.
    node = next(
        n for n in enumerate_nodes([pack])
        if n.actor == "BB" and any(
            a.position == "BTN" for a in n.history_before
            if a.action_type.name == "RAISE"
        )
    )
    facts = SimpleNamespace(spot=SimpleNamespace(node=node))
    script = _preflop_animation_script(
        facts, pack, stakes_bb_dollars=1.0, game_format="cash",
    )
    payload = json.loads(script)
    events = payload["events"]
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["post", "post"]
    assert kinds[-1] == "decision" and events[-1]["seat"] == "BB"
    # 6-max fold-to-BTN line: UTG, HJ, CO fold; BTN opens 2.5; SB folds.
    assert [e["seat"] for e in events if e["type"] == "fold"] == ["UTG", "HJ", "CO", "SB"]
    open_ev = next(e for e in events if e["type"] == "raise")
    assert open_ev["seat"] == "BTN" and open_ev["to_bb"] == 2.5
    assert open_ev["pot_bb"] == 4.0 and open_ev["to_dollars"] == 2.5


def test_plo_writer_animation_uses_display_seats() -> None:
    """The PLO writer's timeline: pot-limit-resolved sizes, Monker seat
    codes mapped to the player-facing names (BU -> BTN, LJ -> UTG)."""
    from types import SimpleNamespace

    from pipeline.plo.format_writer import _plo_animation_script
    from pipeline.plo.pack import PloAction, PloActionType

    history = (
        PloAction("LJ", PloActionType.FOLD),
        PloAction("HJ", PloActionType.FOLD),
        PloAction("CO", PloActionType.FOLD),
        PloAction("BU", PloActionType.RAISE, raise_pct=100),
        PloAction("SB", PloActionType.FOLD),
    )
    facts = SimpleNamespace(
        spot=SimpleNamespace(
            node=SimpleNamespace(
                history_before=history, actor="BB", table_size=6,
            )
        ),
    )
    script = _plo_animation_script(
        facts, stack_bb=100.0, stakes_bb_dollars=1.0, game_format="cash",
    )
    payload = json.loads(script)
    events = payload["events"]
    # Monker codes are translated: BU raises as BTN, LJ folds as UTG.
    raise_ev = next(e for e in events if e["type"] == "raise")
    assert raise_ev["seat"] == "BTN"
    assert raise_ev["to_bb"] == 3.5  # the pot-limit open size over the blinds
    assert [e["seat"] for e in events if e["type"] == "fold"] == ["UTG", "HJ", "CO", "SB"]
    assert events[-1] == {"i": events[-1]["i"], "type": "decision",
                          "seat": "BB", "street": "preflop"}


# --- chip-conservation clamp + vacuous-decision gate (Aug 2026 MTT audit) -----
def test_wager_clamps_to_remaining_stack_bb_ante_jam() -> None:
    """The B205 bug, exactly: 10bb bb-ante MTT, BB posts blind 1 + ante 1
    (8 behind), then 'jams to 10bb' per the pack label. The old walk drove
    BB's stack to -1 in 89 shipped events; the clamp caps the wager at the
    real chips (to 9bb), zeroes the stack, and marks it all-in."""
    from pipeline.animation import AnimationTable

    t = AnimationTable(table_size=8, starting_stack_bb=10.0, ante_bb=1.0)
    t.post_blinds()
    assert t.stacks["BB"] == 8.0            # 10 - 1 blind - 1 ante
    t.wager_to("BB", "raise", 10.0)         # pack says "to 10"; only 9 possible
    e = t.events[-1]
    assert e["to_bb"] == 9.0 and e["stack_bb"] == 0.0 and e["all_in"] is True
    assert all((ev.get("stack_bb") or 0) >= 0 for ev in t.events)


def test_call_clamps_and_marks_all_in() -> None:
    from pipeline.animation import AnimationTable

    t = AnimationTable(table_size=8, starting_stack_bb=10.0, ante_bb=1.0)
    t.post_blinds()
    t.wager_to("HJ", "raise", 10.0, all_in=True)
    t.call("SB")                             # SB has 9.5 behind vs 10 to match
    e = t.events[-1]
    assert e["amount_bb"] == 9.5 and e["stack_bb"] == 0.0 and e["all_in"] is True


def test_unclamped_events_byte_identical() -> None:
    """A normal deep-stack walk must not change at all (no clamp, no
    forced all_in): the guard only fires on physically impossible wagers
    or exactly-emptied stacks."""
    from pipeline.animation import AnimationTable

    t = AnimationTable(table_size=8, starting_stack_bb=200.0)
    t.post_blinds()
    t.wager_to("BTN", "raise", 3.0)
    t.call("BB")
    for e in t.events:
        assert "all_in" not in e


def test_hero_stack_at_preflop_decision_vacuous() -> None:
    """The B205 question shape: SB calls the HJ jam all-in, BB re-jams,
    'SB decision' -- hero has 0bb and the gate must say so."""
    from pipeline.animation import hero_stack_at_preflop_decision

    actions = [
        ("UTG", "fold", None, False), ("UTG+1", "fold", None, False),
        ("LJ", "fold", None, False), ("HJ", "raise", 10.0, True),
        ("CO", "fold", None, False), ("BTN", "fold", None, False),
        ("SB", "call", None, True), ("BB", "raise", 10.0, True),
    ]
    behind = hero_stack_at_preflop_decision(
        table_size=8, starting_stack_bb=10.0, ante_bb=1.0,
        actions=actions, hero_seat="SB",
    )
    assert behind == 0.0
    # ...and the same line at 40bb leaves SB a REAL decision (not gated).
    behind_deep = hero_stack_at_preflop_decision(
        table_size=8, starting_stack_bb=40.0, ante_bb=1.0,
        actions=actions, hero_seat="SB",
    )
    assert behind_deep > 0
