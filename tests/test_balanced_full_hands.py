"""Tests for balanced variable-length full hands (July 2026).

The production anti-predictability rules:
  * FOLD-TRUNCATION COHERENCE -- a hand ENDS at the first decision whose
    solver-correct action is Fold; no non-final leg may be fold-dominant.
  * EQUAL LENGTH QUOTAS -- a balanced batch carries equal quarters of hands
    ending preflop / flop / turn / river (back-filled when a bucket is
    short).
  * PREFLOP-ENDING hands come from the matched pack (terminal-fold gate)
    and, when folding TO a raise, carry a vindication reveal drawn from the
    raiser's real range (clear preflop favourite only).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s  # noqa: E402
from pipeline.postflop.play_through import (  # noqa: E402
    ENDING_STREETS,
    HandLeg,
    PlayThroughHand,
    assemble_hands,
    balanced_length_mix,
)
from pipeline.postflop.batch import _collect_worthy  # noqa: E402
from tests.test_full_hand_pack_legs import _matching_pack  # noqa: E402


def _worthy(solve):
    return _collect_worthy(
        solve, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None,
    )[0]


def test_no_mid_hand_leg_is_fold_dominant() -> None:
    """INVARIANT: the correct answer on any non-final leg is a CONTINUE
    action -- a fold-dominant node truncates the hand right there."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    hands = assemble_hands(solve, seeds=_worthy(solve))
    assert hands
    for hand in hands:
        for leg in hand.legs[:-1]:
            if leg.kind == "postflop":
                assert leg.spot.dominant_verb != "fold", (
                    f"mid-hand fold-dominant leg in {hand.hand_id}"
                )
                assert not leg.terminal_fold
        last = hand.legs[-1]
        if last.kind == "postflop" and last.spot.dominant_verb == "fold":
            assert last.terminal_fold
        # AS-PLAYED COHERENCE: every non-final postflop leg's dominant verb
        # equals the action the line actually took there (no "Mostly Bet"
        # followed by "you check").
        from pipeline.postflop.play_through import _as_played_verb
        post = [leg for leg in hand.legs if leg.kind == "postflop"]
        if len(post) >= 2:
            anchor = post[-1].spot.node
            for leg in post[:-1]:
                taken = _as_played_verb(leg.spot.node, anchor, hand.hero)
                if taken is not None:
                    assert leg.spot.dominant_verb == taken, hand.hand_id


def _stub_hand(ending: str, i: int) -> PlayThroughHand:
    return PlayThroughHand(
        hand_id=f"{ending}_{i}", hero="BTN", hero_combo="AsKs", frame="hero",
        legs=(HandLeg(kind="postflop", street=ending, node_id=f"n{i}"),),
    )


def test_balanced_length_mix_quotas_and_backfill() -> None:
    hands = (
        [_stub_hand("river", i) for i in range(10)]
        + [_stub_hand("turn", i) for i in range(10)]
        + [_stub_hand("flop", i) for i in range(10)]
        + [_stub_hand("preflop", i) for i in range(10)]
    )
    out = balanced_length_mix(hands, 8)
    counts = {s: sum(1 for h in out if h.ending_street == s)
              for s in ENDING_STREETS}
    assert counts == {"preflop": 2, "flop": 2, "turn": 2, "river": 2}
    # Interleaved, not clustered: the first four cover all four streets.
    assert {h.ending_street for h in out[:4]} == set(ENDING_STREETS)
    # Back-fill: an empty flop bucket is topped up from the others.
    no_flop = [h for h in hands if h.ending_street != "flop"]
    out = balanced_length_mix(no_flop, 8)
    assert len(out) == 8
    assert sum(1 for h in out if h.ending_street == "flop") == 0


def test_terminal_fold_pack_gate_and_fold_ender_classes(tmp_path: Path) -> None:
    from pipeline.postflop.preflop_leg_pack import (
        _build_pack_facts,
        combo_for_hand_class,
        find_pack_leg_source,
        fold_ender_hand_classes,
    )
    import random

    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None
    hero = solve.positions[-1] if solve.positions[-1] != "BTN" else solve.positions[0]
    # BB facing the open: fold-dominant classes exist in any real defend node.
    classes = fold_ender_hand_classes(
        src, "BB", min_frequency=0.5, max_frequency=1.0,
    )
    if not classes:  # the synthetic test pack may defend everything
        return
    hand_class = classes[0]
    combo = combo_for_hand_class(hand_class, random.Random(7))
    built = _build_pack_facts(
        src, "BB", combo, solve, equity_runouts=20, terminal_fold=True,
    )
    assert built is not None
    assert built.spot.dominant_action.startswith("Fold")
    # The same combo must FAIL the normal (continue) gate: terminal folds
    # can never leak into as-played legs.
    assert _build_pack_facts(
        src, "BB", combo, solve, equity_runouts=20, terminal_fold=False,
    ) is None


def test_preflop_fold_resolution_vindicates(tmp_path: Path) -> None:
    from pipeline.postflop.preflop_leg_pack import (
        build_preflop_fold_resolution,
        combo_for_hand_class,
        find_pack_leg_source,
        fold_ender_hand_classes,
    )
    import random

    solve = btn_vs_bb_full_hand_2cJs7s()
    src = find_pack_leg_source(solve, tmp_path, packs=[_matching_pack(tmp_path)])
    classes = fold_ender_hand_classes(
        src, "BB", min_frequency=0.5, max_frequency=1.0,
    )
    if not classes:
        return
    combo = combo_for_hand_class(classes[0], random.Random(7))
    res = build_preflop_fold_resolution(
        src, "BB", combo, solve, hand_id="h1",
    )
    if res is None:  # no clear favourite in the synthetic raise range
        return
    assert res["vindicates"] == "Fold"
    kinds = [e["type"] for e in res["events"]]
    assert kinds[0] == "fold" and kinds[-1] == "win"
    win = res["events"][-1]
    assert win["reason"] == "fold" and win["seat"] == res["villain_seat"]
    assert "Folding saved you money" in res["summary"]
    # Deterministic: same inputs -> byte-identical reveal.
    assert build_preflop_fold_resolution(
        src, "BB", combo, solve, hand_id="h1",
    ) == res


def test_aggregate_hand_difficulty_blend() -> None:
    """Peak-anchored blend: the user's example (one 2000 spike over four
    500 legs) rates ~1475 -- not 2000 (over-promise) and not 800 (ambush).
    Floor property: never below 65% of the peak."""
    from pipeline.postflop.difficulty import aggregate_hand_difficulty

    assert aggregate_hand_difficulty([2000, 500, 500, 500, 500]) == 1475
    assert aggregate_hand_difficulty([2000, 1900, 1900, 1800]) == 1953
    assert aggregate_hand_difficulty([2400]) == 2400  # single-leg hand
    assert aggregate_hand_difficulty([]) == 0
    # duplicate peaks: one counts as the anchor, the twin tempers upward
    assert aggregate_hand_difficulty([2000, 2000]) == 2000
    assert aggregate_hand_difficulty([2400, 400, 400]) >= int(0.65 * 2400)


def test_raise_ender_gate_and_resolution(tmp_path: Path) -> None:
    """Raise-enders: a combo whose dominant pack action is the sized
    re-raise passes the terminal_raise gate (and fails the continue gate),
    and its resolution shows the villain FOLDING a real hand from their
    range to hero's raise, pot pushed to hero. Deterministic."""
    from pipeline.postflop.preflop_leg_pack import (
        _build_pack_facts,
        build_preflop_raise_resolution,
        combo_for_hand_class,
        find_pack_leg_source,
        raise_ender_hand_classes,
    )
    import random

    solve = btn_vs_bb_full_hand_2cJs7s()
    src = find_pack_leg_source(solve, tmp_path, packs=[_matching_pack(tmp_path)])
    assert src is not None
    classes = raise_ender_hand_classes(
        src, "BB", min_frequency=0.5, max_frequency=1.0,
    )
    if not classes:  # the synthetic pack may never favour the 3-bet
        return
    combo = combo_for_hand_class(classes[0], random.Random(7))
    built = _build_pack_facts(
        src, "BB", combo, solve, equity_runouts=20, terminal_raise=True,
    )
    assert built is not None
    assert built.spot.dominant_action.startswith("Raise")
    res = build_preflop_raise_resolution(src, "BB", combo, solve, hand_id="h1")
    if res is None:  # pack tree may not extend past the raise
        return
    kinds = [e["type"] for e in res["events"]]
    assert kinds[0] == "raise" and "fold" in kinds and kinds[-1] == "win"
    win = res["events"][-1]
    assert win["seat"] == "BB" and win["reason"] == "fold"
    reveal = next(e for e in res["events"] if e["type"] == "reveal")
    assert reveal.get("folded") is True
    assert "takes the pot" in res["summary"]
    assert build_preflop_raise_resolution(
        src, "BB", combo, solve, hand_id="h1",
    ) == res


def test_pack_first_no_entry_fallback(tmp_path: Path, monkeypatch) -> None:
    """PACK-FIRST RULE: with a matching preflop chart, the entry fallback
    never ships (a coherence-refused hand gets NO preflop question --
    the fallback's as-played framing could contradict the solver, e.g.
    'Mostly Open' correct above 'Fold: 81%'). Without a pack, an entry
    leg ships only when the continue action is dominant (>= 50%)."""
    import csv as _csv
    import json
    import re

    import pipeline.postflop.full_hand_batch as fhb
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch
    from pipeline.postflop.preflop_leg_pack import find_pack_leg_source

    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None
    monkeypatch.setattr(fhb, "find_pack_leg_source", lambda *a, **k: src)

    out = tmp_path / "with_pack.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=4, dry_run=True,
        equity_runouts=20, preflop_leg_pack_root=tmp_path,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    c = meta["counters"]
    assert c["preflop_leg_entry_fallback"] == 0, (
        "entry fallback shipped despite a matching pack"
    )

    out2 = tmp_path / "no_pack.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out2, total_hands=4, dry_run=True,
        equity_runouts=20, preflop_leg_pack_root=None,
    )
    rows = list(_csv.DictReader(open(out2, encoding="utf-8-sig")))
    for r in rows:
        if r.get("Cards on Table"):
            continue  # postflop leg
        freqs = r.get("action_frequencies", "")
        m = re.search(r"Fold:\s*(\d+)%", freqs)
        if m:
            assert int(m.group(1)) <= 50, (
                f"entry leg shipped with fold-dominant frequencies: {freqs}"
            )


def test_river_leaning_profile_quotas() -> None:
    """The production default: 40% river / 20% each; largest-remainder
    ties break deepest-first (n=1 -> river; n=2 -> river+turn)."""
    from pipeline.postflop.play_through import LENGTH_PROFILES

    pool = (
        [_stub_hand("river", i) for i in range(10)]
        + [_stub_hand("turn", i) for i in range(10)]
        + [_stub_hand("flop", i) for i in range(10)]
        + [_stub_hand("preflop", i) for i in range(10)]
    )
    w = LENGTH_PROFILES["river_leaning"]

    def mix(n):
        out = balanced_length_mix(pool, n, weights=w)
        return {s: sum(1 for h in out if h.ending_street == s)
                for s in ENDING_STREETS if any(h.ending_street == s for h in out)}

    assert mix(1) == {"river": 1}
    assert mix(2) == {"turn": 1, "river": 1}
    assert mix(10) == {"preflop": 2, "flop": 2, "turn": 2, "river": 4}


def test_topup_replaces_a_dropped_hand(tmp_path: Path, monkeypatch) -> None:
    """WHOLE-HAND ATOMICITY TOP-UP: when a hand dies on a failed leg, a
    same-ending reserve hand takes its slot -- a small batch can no longer
    silently lose its river hand (the bug the user hit live: a 2-hand
    batch shipping 1 turn hand)."""
    import json

    import pipeline.postflop.full_hand_batch as fhb
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    solve = btn_vs_bb_full_hand_2cJs7s()
    real = fhb._postflop_leg_row
    killed = {}

    def sabotage(spot, solve_, **kwargs):
        # Fail every leg of the FIRST river-ending hand we see.
        hid = kwargs.get("hand_id", "")
        if not killed or killed.get("hid") == hid:
            if killed.get("hid") is None and kwargs.get("sequence_total"):
                pass
            if killed.get("hid") is None:
                killed["hid"] = hid
            if killed["hid"] == hid:
                return None, None, {"hand_id": hid, "error_message": "sabotaged"}, fhb._zero_counters()
        return real(spot, solve_, **kwargs)

    monkeypatch.setattr(fhb, "_postflop_leg_row", sabotage)
    # The fixture pool holds exactly TWO hands: request 1 so the other is
    # the reserve; sabotage kills the selected one; top-up must deliver.
    out = tmp_path / "topup.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=1, dry_run=True,
        balanced_lengths=True, equity_runouts=20,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    c = meta["counters"]
    assert c["hands_dropped_failed_leg"] == 1
    assert c["hands_topped_up"] == 1
    # The batch still delivers the requested count (reserve permitting).
    assert c["hands_written"] == 1
