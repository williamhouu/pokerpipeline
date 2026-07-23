"""Tests for the 🎬 action-heavy hand policy (July 2026, user ask).

Pure-policy tests use synthetic hands (no solve, no facts). Batch-level
tests run the in-memory fixture solve dry (no API). The problem being
solved: 37% of generated full hands had <=1 bet in the whole postflop line
(check-check-checkdown into a near-pure river fold), and "Hard" hands were
hard PREFLOP (marginal defends max the frequency+EV axes) while every
postflop leg was trivial -- the peak-anchored hand_difficulty let the
preflop spike qualify the hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.postflop.hand_quality import (  # noqa: E402
    apply_action_heavy_policy,
    educational_density,
    is_passive_line,
    is_trivial_fold_ender,
)


def _step(street: str, verb: str):
    return SimpleNamespace(street=street, verb=verb)


def _hand(
    *,
    ending_street: str = "river",
    history: tuple = (),
    ender_verb: str = "call",
    ender_freq: float = 0.75,
    ender_to_call: float = 2.0,
    preflop_only: bool = False,
    hand_id: str = "h",
):
    """A minimal synthetic PlayThroughHand for the pure policy functions."""
    if preflop_only:
        legs = (SimpleNamespace(kind="preflop_line", street="preflop",
                                spot=None, terminal_fold=True),)
        return SimpleNamespace(hand_id=hand_id, legs=legs)
    node = SimpleNamespace(history=tuple(history), to_call_bb=ender_to_call)
    spot = SimpleNamespace(
        node=node, dominant_verb=ender_verb, dominant_frequency=ender_freq,
    )
    legs = (
        SimpleNamespace(kind="preflop_entry", street="preflop", spot=None),
        SimpleNamespace(kind="postflop", street=ending_street, spot=spot),
    )
    return SimpleNamespace(hand_id=hand_id, legs=legs)


# --- passive (checkdown) lines ----------------------------------------------
def test_checkdown_to_river_fold_is_passive() -> None:
    # x/x, x/x, then a river stab hero FOLDS to: the user's 5d4d example
    # (air giving up). The late bet does not redeem the line -- only a real
    # bluff-catch would (see the carve-out test below).
    h = _hand(history=(
        _step("flop", "check"), _step("flop", "check"),
        _step("turn", "check"), _step("turn", "check"),
        _step("river", "check"), _step("river", "bet"),
    ), ender_verb="fold", ender_freq=0.85)
    assert is_passive_line(h) is True


def test_flop_bet_makes_the_line_active() -> None:
    h = _hand(history=(
        _step("flop", "check"), _step("flop", "bet"), _step("flop", "call"),
        _step("turn", "check"), _step("turn", "check"),
        _step("river", "bet"),
    ))
    assert is_passive_line(h) is False


def test_flop_and_preflop_enders_are_exempt() -> None:
    # No earlier postflop street exists for action to have happened on.
    flop_ender = _hand(ending_street="flop", history=(_step("flop", "check"),))
    assert is_passive_line(flop_ender) is False
    assert is_passive_line(_hand(preflop_only=True)) is False


# --- trivial fold enders ------------------------------------------------------
def test_near_pure_fold_ender_is_trivial() -> None:
    h = _hand(ender_verb="fold", ender_freq=0.95)
    assert is_trivial_fold_ender(h) is True


def test_mixed_fold_and_non_fold_enders_are_kept() -> None:
    assert is_trivial_fold_ender(_hand(ender_verb="fold", ender_freq=0.85)) is False
    assert is_trivial_fold_ender(_hand(ender_verb="call", ender_freq=0.95)) is False
    assert is_trivial_fold_ender(_hand(preflop_only=True)) is False


# --- educational density -------------------------------------------------------
def test_action_hand_outranks_checkdown() -> None:
    barreled = _hand(history=(
        _step("flop", "bet"), _step("flop", "call"),
        _step("turn", "bet"), _step("turn", "call"),
        _step("river", "bet"),
    ), ender_freq=0.78)
    checkdown = _hand(history=(
        _step("flop", "check"), _step("flop", "check"),
        _step("turn", "check"), _step("turn", "check"),
        _step("river", "bet"),
    ), ender_freq=0.78)
    assert educational_density(barreled) > educational_density(checkdown)


def test_raised_line_gets_a_bonus() -> None:
    raised = _hand(history=(_step("flop", "bet"), _step("flop", "raise")))
    flat = _hand(history=(_step("flop", "bet"), _step("flop", "call")))
    assert educational_density(raised) > educational_density(flat)


# --- the composed policy --------------------------------------------------------
def test_policy_drops_trivial_folds_and_caps_passives() -> None:
    active = [
        _hand(hand_id=f"active{i}", history=(
            _step("flop", "bet"), _step("flop", "call"), _step("river", "bet"),
        )) for i in range(4)
    ]
    passive = [
        _hand(hand_id=f"passive{i}", history=(
            _step("flop", "check"), _step("flop", "check"),
            _step("turn", "check"), _step("turn", "check"),
            _step("river", "bet"),
        ), ender_verb="fold", ender_freq=0.85) for i in range(3)
    ]
    trivial = [_hand(hand_id="triv", ender_verb="fold", ender_freq=0.97)]
    preflop = [_hand(hand_id="pf", preflop_only=True)]

    kept, counters = apply_action_heavy_policy(
        active + passive + trivial + preflop, total_hands=4,
    )
    ids = [h.hand_id for h in kept]
    assert "triv" not in ids
    assert counters["hands_excluded_trivial_fold_ender"] == 1
    # cap = ceil(0.15 * 4) = 1 passive kept, 2 excluded.
    assert sum(1 for i in ids if i.startswith("passive")) == 1
    assert counters["hands_excluded_passive_line"] == 2
    assert counters["passive_hands_kept"] == 1
    # Preflop-only hands pass through untouched.
    assert "pf" in ids
    # Density ordering: every active hand ranks before the kept passive one.
    kept_postflop = [i for i in ids if i != "pf"]
    assert kept_postflop[-1].startswith("passive")


# --- batch-level integration (fixture solve, dry run) -------------------------
def test_batch_records_policy_and_counters(tmp_path) -> None:
    from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "ah.csv"
    res = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=3, dry_run=True,
        answer_style="gto", equity_runouts=20, include_villain=True,
    )
    assert res.questions_written > 0  # the policy must not empty the fixture
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["run_settings"]["action_heavy"] is True
    for key in (
        "hands_excluded_trivial_fold_ender",
        "hands_excluded_passive_line",
        "passive_hands_kept",
    ):
        assert key in meta["counters"]


def test_batch_action_heavy_off_restores_legacy_selection(tmp_path) -> None:
    """action_heavy=False must reproduce the pre-policy batch byte-for-byte
    (the toggle only changes which hands are picked, and off = old path)."""
    from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "legacy.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=3, dry_run=True,
        answer_style="gto", equity_runouts=20, include_villain=True,
        action_heavy=False,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["run_settings"]["action_heavy"] is False
    assert "hands_excluded_passive_line" not in meta["counters"]


# --- ⚡ parallel legs (July 2026, user ask: NLHE preflop→river at volume) ------
def test_parallel_legs_match_sequential_output(tmp_path) -> None:
    """llm_workers > 1 runs a hand's legs concurrently; results are applied
    in leg order and hand-level control flow is untouched, so the batch must
    come out byte-identical to a sequential run."""
    from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    solve = btn_vs_bb_full_hand_2cJs7s()
    out_seq = tmp_path / "seq.csv"
    out_par = tmp_path / "par.csv"
    common = dict(
        solve=solve, total_hands=3, dry_run=True, answer_style="gto",
        equity_runouts=20, include_villain=True,
    )
    r1 = generate_full_hand_batch(output_path=out_seq, llm_workers=1, **common)
    r3 = generate_full_hand_batch(output_path=out_par, llm_workers=3, **common)
    assert r1.questions_written == r3.questions_written > 0
    assert out_seq.read_text(encoding="utf-8") == out_par.read_text(
        encoding="utf-8"
    )
    meta_seq = json.loads(out_seq.with_suffix(".meta.json").read_text())
    meta_par = json.loads(out_par.with_suffix(".meta.json").read_text())
    assert meta_seq["questions"] == meta_par["questions"]
    assert meta_seq["counters"] == meta_par["counters"]
    assert meta_par["run_settings"]["llm_workers"] == 3  # noqa: PLR2004


# --- no mid-hand endings (July 22 2026, user standing rule) -------------------
def test_no_hand_ends_early_without_a_fold(tmp_path) -> None:
    """A play-through may end BEFORE the river only on a fold: a hand whose
    last question is a flop/turn check/bet/call is a story cut off mid-hand
    (these lines exist where the down-sampled solve lacks the river
    continuation) and must be dropped at assembly, with a counter."""
    from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "enders.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=10, dry_run=True,
        answer_style="gto", equity_runouts=20, include_villain=True,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert "hands_dropped_nonfold_early_ender" in meta["counters"]
    by_hand: dict[str, list] = {}
    for q in meta["questions"]:
        if q.get("hand_id"):
            by_hand.setdefault(q["hand_id"], []).append(q)
    for hid, legs in by_hand.items():
        legs.sort(key=lambda q: int(q.get("sequence_index") or 0))
        last = legs[-1]
        street = last.get("street", "preflop")
        if street in ("flop", "turn"):
            assert last.get("correct_answer", "").lower().find("fold") >= 0, (
                f"hand {hid} ends on the {street} with "
                f"{last.get('correct_answer')!r} -- mid-hand ending shipped"
            )


def test_checkdown_into_bluff_catch_is_not_passive() -> None:
    """July 22 refinement: a checkdown line ending in a REAL bluff-catch
    (facing a bet, correct action call/raise) is a legitimate story and
    escapes the passive cap; the same line ending in a fold stays capped."""
    checkdown_steps = (
        _step("flop", "check"), _step("flop", "check"),
        _step("turn", "check"), _step("turn", "check"),
        _step("river", "check"), _step("river", "bet"),
    )
    bluff_catch = _hand(history=checkdown_steps, ender_verb="call",
                        ender_freq=0.75, ender_to_call=2.0)
    fold_ender = _hand(history=checkdown_steps, ender_verb="fold",
                       ender_freq=0.80, ender_to_call=2.0)
    stab_spot = _hand(history=(
        _step("flop", "check"), _step("flop", "check"),
        _step("turn", "check"), _step("turn", "check"),
    ), ender_verb="check", ender_freq=0.75, ender_to_call=0.0)
    assert is_passive_line(bluff_catch) is False
    assert is_passive_line(fold_ender) is True
    assert is_passive_line(stab_spot) is True
