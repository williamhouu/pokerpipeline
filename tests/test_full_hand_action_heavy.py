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
        "hands_excluded_bluffcatch_checkdown",
        "passive_hands_kept",
        "bluffcatch_checkdowns_kept",
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


def test_checkdown_into_bluff_catch_is_passive_but_its_own_class() -> None:
    """July 23 2026 (replaces the July 22 full exemption, which swallowed
    whole batches -- 6/7 hands shipped as x/x, x/x, river-bet): a checkdown
    line ending in a REAL bluff-catch (facing a bet, correct action
    call/raise) IS passive, but it is classified separately so the policy
    can give it a ~25-30% sub-quota of the river enders instead of the
    generic passive cap. The same line ending in a fold or a "do you stab?"
    spot stays in the generic passive class."""
    from pipeline.postflop.hand_quality import is_bluffcatch_checkdown

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
    assert is_passive_line(bluff_catch) is True
    assert is_passive_line(fold_ender) is True
    assert is_passive_line(stab_spot) is True
    assert is_bluffcatch_checkdown(bluff_catch) is True
    assert is_bluffcatch_checkdown(fold_ender) is False
    assert is_bluffcatch_checkdown(stab_spot) is False
    # A line with real earlier action is not passive, hence never in the class.
    active = _hand(history=(
        _step("flop", "bet"), _step("flop", "call"),
        _step("river", "check"), _step("river", "bet"),
    ), ender_verb="call", ender_to_call=2.0)
    assert is_bluffcatch_checkdown(active) is False


def test_bluffcatch_checkdowns_capped_at_river_share() -> None:
    """The sub-quota: bluff-catch checkdowns are capped at
    ceil(BLUFFCATCH_RIVER_SHARE x river_ender_target), separately from the
    generic passive cap, and the best-scoring ones are the ones kept."""
    checkdown_steps = (
        _step("flop", "check"), _step("flop", "check"),
        _step("turn", "check"), _step("turn", "check"),
        _step("river", "check"), _step("river", "bet"),
    )
    bluffcatches = [
        _hand(hand_id=f"bc{i}", history=checkdown_steps, ender_verb="call",
              ender_freq=0.75, ender_to_call=2.0)
        for i in range(5)
    ]
    active = [
        _hand(hand_id=f"active{i}", history=(
            _step("flop", "bet"), _step("flop", "call"), _step("river", "bet"),
        )) for i in range(4)
    ]
    # river_ender_target=6 (a river_heavy 8-hand batch) -> cap = ceil(0.3*6)=2.
    kept, counters = apply_action_heavy_policy(
        active + bluffcatches, total_hands=8, river_ender_target=6,
    )
    ids = [h.hand_id for h in kept]
    assert sum(1 for i in ids if i.startswith("bc")) == 2
    assert counters["bluffcatch_checkdowns_kept"] == 2
    assert counters["hands_excluded_bluffcatch_checkdown"] == 3
    # The generic passive counters are untouched by the bluff-catch class.
    assert counters["hands_excluded_passive_line"] == 0
    assert counters["passive_hands_kept"] == 0
    # Real-action hands all survive and outrank the kept checkdowns.
    assert sum(1 for i in ids if i.startswith("active")) == 4
    assert ids[:4] == [f"active{i}" for i in range(4)]


def test_policy_rotates_line_shapes_within_a_street() -> None:
    """July 23 2026 (from the first paid fix-wave batch): 7/7 postflop hands
    shipped as the SAME line shape (c-bet call, turn barrel, fold) with
    different combos — density-identical hands cluster at the top of the
    street bucket and the quota picks them all. The policy output must
    rotate distinct shapes to the top of each ending-street bucket."""
    from pipeline.postflop.hand_quality import line_shape_signature

    barrel_fold = [
        _hand(hand_id=f"barrel{i}", ending_street="turn", history=(
            _step("flop", "check"), _step("flop", "bet"), _step("flop", "call"),
            _step("turn", "check"), _step("turn", "bet"),
        ), ender_verb="fold", ender_freq=0.80) for i in range(5)
    ]
    raise_line = [
        _hand(hand_id=f"raised{i}", ending_street="turn", history=(
            _step("flop", "bet"), _step("flop", "raise"), _step("flop", "call"),
            _step("turn", "bet"),
        ), ender_verb="call", ender_freq=0.75) for i in range(2)
    ]
    donk_line = [
        _hand(hand_id=f"donk{i}", ending_street="turn", history=(
            _step("flop", "bet"), _step("flop", "call"), _step("turn", "bet"),
        ), ender_verb="fold", ender_freq=0.80) for i in range(2)
    ]
    assert line_shape_signature(barrel_fold[0]) != line_shape_signature(donk_line[0])
    kept, _ = apply_action_heavy_policy(
        barrel_fold + raise_line + donk_line, total_hands=9,
    )
    # The first three picks are three DISTINCT shapes (one per group), so a
    # 3-hand turn quota gets variety instead of three copies of one line.
    first_three = [line_shape_signature(h) for h in kept[:3]]
    assert len(set(first_three)) == 3
    # All hands survive (no cap applies here) and the rotation is complete.
    assert len(kept) == 9


def test_density_ignores_the_ending_streets_own_bet() -> None:
    """July 23 2026: aggressive_steps/educational_density count only
    PRE-ender-street bets, so a pure checkdown into a river stab scores no
    action content -- it can no longer float above genuinely bet lines."""
    from pipeline.postflop.hand_quality import aggressive_steps

    checkdown_stab = _hand(history=(
        _step("flop", "check"), _step("flop", "check"),
        _step("turn", "check"), _step("turn", "check"),
        _step("river", "check"), _step("river", "bet"),
    ), ender_verb="call", ender_to_call=2.0)
    assert aggressive_steps(checkdown_stab) == 0
    flop_bet_line = _hand(history=(
        _step("flop", "bet"), _step("flop", "call"),
        _step("turn", "check"), _step("turn", "check"),
        _step("river", "check"), _step("river", "bet"),
    ), ender_verb="call", ender_to_call=2.0)
    assert aggressive_steps(flop_bet_line) == 1
    assert educational_density(flop_bet_line) > educational_density(checkdown_stab)


def test_exciting_hands_toggle_filters_honestly(tmp_path) -> None:
    """🔥 Exciting-pots toggle (July 23 2026, user ask): with the toggle on,
    only hands whose FINAL decision is a big hand (premium/strong) on a
    heated line (raise, or two-plus bets) survive; the rest are counted,
    never silently diluted. The fixture's two assemblable hands both fail
    the bar (the river King demotes the top-pair hands), so the batch
    honestly ships empty with the exclusions counted. Off = untouched."""
    import json

    from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    out = tmp_path / "exciting.csv"
    res = generate_full_hand_batch(
        solve=btn_vs_bb_full_hand_2cJs7s(), output_path=out, total_hands=4,
        dry_run=True, answer_style="gto", equity_runouts=20,
        include_villain=True, exciting_hands=True,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["run_settings"]["exciting_hands"] is True
    assert meta["counters"]["hands_excluded_not_exciting"] == 2
    assert res.questions_written == 0  # honest shrink, no dilution

    off = tmp_path / "off.csv"
    res_off = generate_full_hand_batch(
        solve=btn_vs_bb_full_hand_2cJs7s(), output_path=off, total_hands=4,
        dry_run=True, answer_style="gto", equity_runouts=20,
        include_villain=True,
    )
    meta_off = json.loads(off.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta_off["run_settings"]["exciting_hands"] is False
    assert meta_off["counters"]["hands_excluded_not_exciting"] == 0
    assert res_off.questions_written > 0


# --- class-street twin demotion (Aug 2026, user ask) --------------------------
def _twin_hand(hero, combo, street, *, fold=False, hid=""):
    leg = SimpleNamespace(street=street, terminal_fold=fold)
    return SimpleNamespace(
        hero=hero, hero_combo=combo, hand_id=hid or f"{hero}_{combo}_{street}",
        legs=(leg,), ending_street=street, ends_with_fold=fold,
    )


def test_demote_class_street_twins_moves_twins_to_tail() -> None:
    """AcKc + AhKh river hands (the observed 5-hand-batch repeat) are twins:
    the first keeps its slot, the later one moves to the TAIL (never
    dropped), everything else keeps relative order."""
    from pipeline.postflop.play_through import demote_class_street_twins

    a = _twin_hand("UTG", "AcKc", "river")
    b = _twin_hand("UTG", "QsQh", "flop")
    c = _twin_hand("UTG", "AhKh", "river")   # twin of a (AKs, river, no fold)
    d = _twin_hand("SB", "8h7h", "preflop")
    counters: dict = {}
    out = demote_class_street_twins([a, b, c, d], counters=counters)
    assert out == [a, b, d, c]               # c demoted, order otherwise kept
    assert counters["hands_demoted_class_twins"] == 1


def test_demote_class_street_twins_key_boundaries() -> None:
    """NOT twins: different ending street, different hero seat, suited vs
    offsuit, and fold-ender vs continue-ender of the same class (opposite
    lessons -- the quotas treat correct-fold endings as their own bucket)."""
    from pipeline.postflop.play_through import demote_class_street_twins

    hands = [
        _twin_hand("UTG", "AcKc", "river"),
        _twin_hand("UTG", "AhKh", "turn"),            # different street
        _twin_hand("SB", "AdKd", "river"),            # different hero
        _twin_hand("UTG", "AsKd", "river"),           # AKo, not AKs
        _twin_hand("UTG", "AdKh", "river", fold=True),  # fold-ender AKo
    ]
    counters: dict = {}
    out = demote_class_street_twins(hands, counters=counters)
    assert out == hands
    assert "hands_demoted_class_twins" not in counters


def test_demote_never_shrinks_the_pool() -> None:
    """All-twins pool: one stays in place, the rest demote -- pool size
    unchanged so a class-starved batch can still fill from the tail."""
    from pipeline.postflop.play_through import demote_class_street_twins

    hands = [_twin_hand("UTG", c, "river", hid=c)
             for c in ("AcKc", "AhKh", "AdKd", "AsKs")]
    out = demote_class_street_twins(hands, counters=None)
    assert len(out) == 4 and out[0] is hands[0]
    assert out[1:] == hands[1:]              # demoted tail keeps order


# --- batch-level rank-class cap (Aug 8 2026, user ask) ------------------------
def test_cap_class_repeats_demotes_third_telling() -> None:
    """The observed failure: QQ x5 across streets/enders (each a distinct
    street-twin key). The cap keeps the first TWO tellings of a class in
    the head, demotes the rest to the tail in order -- pool size unchanged."""
    from pipeline.postflop.play_through import cap_class_repeats

    qq = [_twin_hand("UTG", "QsQh", "flop"),
          _twin_hand("UTG", "QsQc", "turn", fold=True),
          _twin_hand("SB", "QhQd", "river"),
          _twin_hand("UTG", "QsQd", "river", fold=True),
          _twin_hand("UTG", "QdQc", "river")]
    ak = _twin_hand("UTG", "AcKc", "river")
    counters: dict = {}
    out = cap_class_repeats(qq + [ak], counters=counters)
    assert out == [qq[0], qq[1], ak, qq[2], qq[3], qq[4]]
    assert counters["hands_demoted_class_cap"] == 3
    assert len(out) == 6  # never drops


def test_cap_class_repeats_two_or_fewer_untouched() -> None:
    from pipeline.postflop.play_through import cap_class_repeats

    hands = [_twin_hand("UTG", "QsQh", "flop"),
             _twin_hand("SB", "QdQc", "river"),
             _twin_hand("UTG", "AcKc", "river")]
    counters: dict = {}
    assert cap_class_repeats(hands, counters=counters) == hands
    assert "hands_demoted_class_cap" not in counters


# --- post-quota class-variety swap (Aug 8 2026, user ask) ---------------------
def test_swap_replaces_excess_class_from_other_street_reserve() -> None:
    """The observed failure: QQ-only flop/turn buckets tail-fill QQ past the
    cap. The two lowest-priority excess QQ swap for river-reserve hands of
    under-cap classes; batch size unchanged; reserve consumed."""
    from pipeline.postflop.play_through import swap_for_class_variety

    selected = [
        _twin_hand("UTG", "QsQh", "flop"),
        _twin_hand("UTG", "QsQc", "turn", fold=True),
        _twin_hand("SB", "QhQd", "turn", fold=True),
        _twin_hand("UTG", "QsQd", "river"),
        _twin_hand("UTG", "AcKc", "river"),
    ]
    reserve = {
        "river": [_twin_hand("SB", "AhAc", "river"),
                  _twin_hand("UTG", "KhKc", "river")],
    }
    counters: dict = {}
    out = swap_for_class_variety(selected, reserve, counters=counters)
    assert len(out) == 5
    classes = [h.hero_combo[0] + h.hero_combo[2] for h in out]
    assert classes.count("QQ") == 2
    assert counters["hands_swapped_for_class_variety"] == 2
    assert reserve["river"] == []              # consumed, no double-serve
    # highest-priority QQ tellings (earliest in mix order) are the keepers
    assert out[0] is selected[0] and out[1] is selected[1]


def test_swap_stops_honestly_when_reserve_lacks_variety() -> None:
    """Reserve holding only more of the SAME class cannot fix anything:
    ship over-cap honestly, count only real swaps."""
    from pipeline.postflop.play_through import swap_for_class_variety

    selected = [_twin_hand("UTG", c, "river") for c in
                ("QsQh", "QsQc", "QhQd")]
    reserve = {"river": [_twin_hand("SB", "QdQc", "river")]}
    counters: dict = {}
    out = swap_for_class_variety(selected, reserve, counters=counters)
    assert out == selected
    assert "hands_swapped_for_class_variety" not in counters


def test_swap_noop_when_under_cap() -> None:
    from pipeline.postflop.play_through import swap_for_class_variety

    selected = [_twin_hand("UTG", "QsQh", "flop"),
                _twin_hand("UTG", "AcKc", "river")]
    reserve = {"river": [_twin_hand("SB", "AhAc", "river")]}
    out = swap_for_class_variety(selected, reserve, counters=None)
    assert out == selected and len(reserve["river"]) == 1


# --- class-aware backfill pull (Aug 8 2026, THE monoculture root fix) ---------
def test_pull_prefers_under_cap_class_over_reserve_head() -> None:
    """The observed funnel: the coherence gate drops varied combos and the
    old head-pop backfill returned the densest COHERENT hand -- always a
    premium. With QQ already at cap, the pull must skip coherent QQ and
    return the under-cap TT even though QQ sits first."""
    from pipeline.postflop.play_through import pull_replacement_class_aware

    qq = _twin_hand("UTG", "QdQc", "river")
    tt = _twin_hand("UTG", "ThTc", "river")
    reserve = {"river": [qq, tt]}
    got = pull_replacement_class_aware(
        reserve, ("river", "turn", "flop", "preflop"),
        lambda h: h, {"QQ": 2},
    )
    assert got is tt
    assert reserve["river"] == [qq]          # over-cap stays for pass 2


def test_pull_falls_back_to_over_cap_for_fullness() -> None:
    from pipeline.postflop.play_through import pull_replacement_class_aware

    qq = _twin_hand("UTG", "QdQc", "river")
    reserve = {"river": [qq]}
    got = pull_replacement_class_aware(
        reserve, ("river",), lambda h: h, {"QQ": 2},
    )
    assert got is qq                          # fullness beats variety


def test_pull_consumes_refused_candidates_for_good() -> None:
    """A coherence-refused candidate can never pass later: it must be
    consumed even when skipped-over classes are in play, and the pull
    continues to the next viable hand."""
    from pipeline.postflop.play_through import pull_replacement_class_aware

    bad = _twin_hand("UTG", "6h5h", "river", hid="bad")
    good = _twin_hand("UTG", "ThTc", "river", hid="good")
    reserve = {"river": [bad, good]}
    got = pull_replacement_class_aware(
        reserve, ("river",), lambda h: None if h.hand_id == "bad" else h, {},
    )
    assert got is good
    assert reserve["river"] == []             # bad consumed, good returned


def test_pull_prefers_unseen_combo_when_repeat_forced() -> None:
    """Tier 1: when every reserve class is at cap, a suit-sibling repeat
    (KhKd) beats the literal same combo (KsKc) already in the batch."""
    from pipeline.postflop.play_through import pull_replacement_class_aware

    same = _twin_hand("UTG", "KsKc", "river")
    sibling = _twin_hand("UTG", "KhKd", "river")
    reserve = {"river": [same, sibling]}
    got = pull_replacement_class_aware(
        reserve, ("river",), lambda h: h, {"KK": 2},
        seen_combos={"KsKc"},
    )
    assert got is sibling
    # ...and the literal duplicate still ships when it is ALL that's left.
    got2 = pull_replacement_class_aware(
        reserve, ("river",), lambda h: h, {"KK": 2},
        seen_combos={"KsKc"},
    )
    assert got2 is same


def test_pull_matches_legacy_when_nothing_over_cap() -> None:
    """Drop-free batches must stay byte-identical: with no class at cap,
    pass 1 consumes the queue exactly like the old head-pop."""
    from pipeline.postflop.play_through import pull_replacement_class_aware

    first = _twin_hand("UTG", "QdQc", "river")
    reserve = {"river": [first, _twin_hand("UTG", "ThTc", "river")]}
    got = pull_replacement_class_aware(
        reserve, ("river",), lambda h: h, {"QQ": 1},
    )
    assert got is first


# --- plain-path total_hands cap (Aug 2026 regression) -------------------------
def test_plain_path_honors_total_hands(tmp_path) -> None:
    """REGRESSION (Aug 2026): with action_heavy on (the default) the pool is
    assembled oversized (20x) for the policy to rank, and the plain path --
    no balanced lengths, no diversify, no difficulty band -- used to ship
    the WHOLE gated pool instead of trimming back (-n 4 shipped 22 hands on
    a real solve). The fixture assembles 2 hands; total_hands=1 must ship
    exactly 1, and the density-desc order means it is the policy's top pick."""
    import json

    from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    out = tmp_path / "capped.csv"
    res = generate_full_hand_batch(
        solve=btn_vs_bb_full_hand_2cJs7s(), output_path=out, total_hands=1,
        dry_run=True, answer_style="gto", equity_runouts=20,
        include_villain=True,
    )
    assert res.questions_written > 0
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    hand_ids = {q["hand_id"] for q in meta["questions"] if q.get("hand_id")}
    assert len(hand_ids) == 1
