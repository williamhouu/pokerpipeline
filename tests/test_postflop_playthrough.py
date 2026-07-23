"""Tests for the play-through (full-hand) feature + preflop-entry questions.

Covers the three new pieces, all driven by the synthetic fixtures (no solver
file, no API key):

* ``pipeline.postflop.preflop_entry`` -- preflop-entry facts / options / prose /
  CSV rows from the solve's flop-entry frequencies.
* ``pipeline.postflop.play_through`` -- assembling linked, ordered hands from a
  connected line (Option B hand_id / sequence_index).
* ``pipeline.postflop.full_hand_batch`` -- the full-hand + standalone-preflop
  batch drivers (deterministic, byte-identical CSV).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.postflop.fixtures import (  # noqa: E402
    btn_vs_bb_full_hand_2cJs7s,
    btn_vs_bb_srp_2cJs7s,
)
from pipeline.postflop.format_writer import POSTFLOP_CSV_COLUMNS  # noqa: E402
from pipeline.postflop.full_hand_batch import (  # noqa: E402
    generate_full_hand_batch,
    generate_preflop_entry_batch,
)
from pipeline.postflop.play_through import assemble_hands  # noqa: E402
from pipeline.postflop.preflop_entry import (  # noqa: E402
    build_preflop_entry_facts,
    build_preflop_entry_options,
    build_preflop_entry_row,
    enumerate_preflop_entry_facts,
    placeholder_preflop_entry_explanation,
    preflop_entry_is_worthy,
)
from pipeline.postflop.question_extractor import evaluate_spot  # noqa: E402
from pipeline.postflop.spot_sampler import enumerate_spots  # noqa: E402


# --- schema -----------------------------------------------------------------
def test_sequence_columns_in_schema() -> None:
    for col in ("hand_id", "sequence_index", "sequence_total"):
        assert col in POSTFLOP_CSV_COLUMNS


def test_standalone_postflop_batch_leaves_sequence_blank(tmp_path: Path) -> None:
    """A normal per-spot batch writes the new columns but they stay blank."""
    from pipeline.postflop.batch import generate_postflop_batch

    out = tmp_path / "spots.csv"
    generate_postflop_batch(
        solve=btn_vs_bb_srp_2cJs7s(), output_path=out, total_questions=3, dry_run=True,
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert rows
    for r in rows:
        assert r["hand_id"] == ""
        assert r["sequence_index"] == ""
        assert r["sequence_total"] == ""


# --- preflop entry ----------------------------------------------------------
def test_preflop_entry_facts_read_weight_and_price() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    # AJo defends at the fixture weight 0.7; BB faces a 2.5bb open.
    combo = next(c for c in solve.preflop_entry_ranges["BB"]
                 if c[0] == "A" and c[2] == "J" and c[1] != c[3])
    facts = build_preflop_entry_facts(solve, "BB", combo)
    assert facts.entry_verb == "call"
    assert 0.0 < facts.continue_freq <= 1.0
    assert facts.break_even_equity is not None  # a real price facing the open
    assert facts.to_call_bb == pytest.approx(1.5)  # 2.5 open - 1 posted BB


def test_preflop_entry_worthiness_window() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    facts = enumerate_preflop_entry_facts(solve, heroes=("BB",))
    worthy = [f for f in facts
              if preflop_entry_is_worthy(f, min_frequency=0.65, max_frequency=0.99)]
    # The fixture's mixed defends (0.7-0.8) are worthy; pure 1.0 calls are not.
    assert worthy
    assert all(0.65 <= f.continue_freq <= 0.99 for f in worthy)


def test_preflop_entry_options_styles_and_as_played() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    combo = next(c for c in solve.preflop_entry_ranges["BB"]
                 if c[0] == "A" and c[2] == "J" and c[1] != c[3])
    facts = build_preflop_entry_facts(solve, "BB", combo)
    # gto -> spectrum, basic -> plain.
    gto_opts, gto_correct = build_preflop_entry_options(facts, style="gto")
    assert gto_opts == ["Always Fold", "Mostly Fold", "Mostly Call", "Always Call"]
    assert gto_correct in gto_opts
    basic_opts, basic_correct = build_preflop_entry_options(facts, style="basic")
    assert basic_opts == ["Call", "Fold"]
    # as_played: correct is ALWAYS the continue (Call) side -- never Fold -- but
    # it HONOURS the requested option shape (basic -> "Call"; gto -> the spectrum
    # "Mostly/Always Call") so a play-through leg matches the postflop legs' style.
    _b_opts, b_correct = build_preflop_entry_options(facts, style="basic", as_played=True)
    assert b_correct == "Call"
    _g_opts, g_correct = build_preflop_entry_options(facts, style="gto", as_played=True)
    assert g_correct in ("Mostly Call", "Always Call")
    assert "Fold" not in g_correct


def test_preflop_entry_row_schema_and_stage() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    facts = enumerate_preflop_entry_facts(solve, heroes=("BB",))[0]
    opts, correct = build_preflop_entry_options(facts)
    expl = placeholder_preflop_entry_explanation(facts, opts, correct)
    row = build_preflop_entry_row(facts, expl, 1, hand_id="h1", sequence_index=1,
                                  sequence_total=4)
    assert set(row) == set(POSTFLOP_CSV_COLUMNS)
    assert row["Hand Stage"] == "Preflop"
    assert row["Cards on Table"] == ""  # no board preflop
    assert row["hand_id"] == "h1"
    assert row["sequence_index"] == "1"
    assert correct == row["Correct Answer"]


def test_preflop_entry_question_mentions_open_and_no_board() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    facts = enumerate_preflop_entry_facts(solve, heroes=("BB",))[0]
    from pipeline.postflop.preflop_entry import format_preflop_entry_question

    q = format_preflop_entry_question(facts)
    assert "opens to" in q
    # No postflop streets named.
    assert "flop" not in q.lower() and "turn" not in q.lower()


# --- play-through assembly --------------------------------------------------
def _worthy_seeds(solve, *, lo=0.50, hi=0.99):
    seeds = []
    for n in solve.nodes.values():
        for sp in enumerate_spots(n):
            if evaluate_spot(sp, min_frequency=lo, max_frequency=hi,
                             min_ev_gap_bb=None).is_worthy:
                seeds.append(sp)
    return seeds


def test_forced_single_action_legs_are_skipped(tmp_path) -> None:
    """FORCED-MOVE GUARD (July 2026, seen live on a real v8 batch): a node
    offering only ONE action (deep lines truncate to check-only in real
    solve trees) must not become a question leg -- a one-option "question"
    has nothing to decide, and it costs a real LLM call. The next leg's
    prose narrates the forced action, so the play-through stays continuous.
    """
    import dataclasses

    from pipeline.postflop.play_through import _build_legs

    solve = btn_vs_bb_full_hand_2cJs7s()
    node = next(
        n for n in solve.nodes.values()
        if n.actor == "BB" and n.strategy and len(n.actions) >= 2
    )
    combo = next(iter(node.strategy))
    forced = dataclasses.replace(
        node, node_id=node.node_id + ":forced", actions=(node.actions[0],),
    )
    legs = _build_legs(
        solve, "BB", combo, [node, forced], include_preflop=False,
    )
    assert [leg.node_id for leg in legs] == [node.node_id]

    # Batch-level invariant: EVERY emitted full-hand question offers a real
    # choice (option 2 non-empty) -- forced moves never reach the CSV.
    out = tmp_path / "fh_forced.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=3, dry_run=True,
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert rows
    for r in rows:
        assert r["option 2"].strip(), (
            f"one-option question shipped: #{r['No']} ({r['Hand Stage']})"
        )


def test_assemble_one_connected_hero_hand() -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    hands = assemble_hands(solve, seeds=_worthy_seeds(solve), heroes=("BB",))
    # The connected fixture is one BB line; deepest seed absorbs the shallower
    # BB seeds into a single hand.
    assert len(hands) == 1
    hand = hands[0]
    assert hand.frame == "hero" and hand.hero == "BB"
    # Legs are ordered preflop -> flop -> turn -> river by street.
    streets = [leg.street for leg in hand.legs]
    assert streets[0] == "preflop"
    order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    ranks = [order[s] for s in streets]
    assert ranks == sorted(ranks)
    # Every postflop leg is the SAME hero combo on the line.
    combos = {leg.spot.hero_combo for leg in hand.legs if leg.spot is not None}
    assert combos == {hand.hero_combo}


def test_assemble_villain_frame_adds_second_hand() -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    seeds = _worthy_seeds(solve)
    hero_only = assemble_hands(solve, seeds=seeds, heroes=("BB",))
    with_villain = assemble_hands(solve, seeds=seeds, heroes=("BB",), include_villain=True)
    assert len(with_villain) > len(hero_only)
    frames = {h.frame for h in with_villain}
    assert frames == {"hero", "villain"}


def test_assemble_is_deterministic() -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    seeds = _worthy_seeds(solve)
    a = assemble_hands(solve, seeds=seeds, heroes=("BB",), include_villain=True)
    b = assemble_hands(solve, seeds=seeds, heroes=("BB",), include_villain=True)
    assert [h.hand_id for h in a] == [h.hand_id for h in b]
    # hand_id is readable + carries the combo + an 8-char hash.
    assert all("_" in h.hand_id for h in a)


def test_assemble_dedupes_overlapping_lines() -> None:
    """Many seeds on ONE connected line (the same combo) collapse to ONE hand --
    no near-duplicate hands that share the preflop+flop legs and differ only in
    the deep runout."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    # EVERY BB spot as a seed (not just the worthy ones), incl. the shallow flop
    # lead -- they all lie on the one line, so dedup must yield a single hand.
    seeds = [sp for n in solve.nodes.values() for sp in enumerate_spots(n)
             if n.actor == "BB"]
    hands = assemble_hands(solve, seeds=seeds, heroes=("BB",))
    assert len(hands) == 1
    # No (hero, combo) is emitted twice.
    keys = [(h.hero, h.hero_combo) for h in hands]
    assert len(keys) == len(set(keys))


def test_full_hand_batch_no_duplicate_hands(tmp_path: Path) -> None:
    """A full-hand batch never emits two hands with the same (seat, cards)."""
    from collections import Counter

    out = tmp_path / "fh.csv"
    generate_full_hand_batch(
        solve=btn_vs_bb_full_hand_2cJs7s(), output_path=out, total_hands=20,
        dry_run=True, heroes=("BB",),
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    by_hand: dict[str, list] = defaultdict(list)
    for r in rows:
        by_hand[r["hand_id"]].append(r)
    sigs = [
        (legs[0]["User Seat"].split("-")[0], legs[0]["User Cards"])
        for legs in by_hand.values()
    ]
    assert all(c == 1 for c in Counter(sigs).values())


# --- Layer-7 on full-hand postflop legs (content-aware mock) ----------------
class _LBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _LUsage:
    input_tokens = 10
    output_tokens = 20


class _LResp:
    def __init__(self, text: str) -> None:
        self.content = [_LBlock(text)]
        self.usage = _LUsage()


class _LifecycleClient:
    """Content-aware mock: a claim-check call (system has 'poker editor') flags
    the original + clears the revised; a revise call (user has 'AUDIT ISSUES TO
    FIX') returns the rewrite; everything else is generation."""

    class _Msgs:
        def __init__(self, gen: str, revised: str, flag: bool) -> None:
            self.gen, self.revised, self.flag = gen, revised, flag

        def create(self, **kw):
            system = kw.get("system", "")
            user = kw["messages"][0]["content"]
            if "poker editor" in system:
                if not self.flag or self.revised in user:
                    return _LResp('{"issues": []}')
                return _LResp('{"issues": [{"claim": "vague", "problem": "unclear"}]}')
            if "AUDIT ISSUES TO FIX" in user:
                return _LResp(self.revised)
            return _LResp(self.gen)

    def __init__(self, gen: str, revised: str, *, flag: bool) -> None:
        self.messages = self._Msgs(gen, revised, flag)


def test_full_hand_layer7_audits_postflop_legs_only(tmp_path: Path) -> None:
    """The opt-in Layer-7 audit/revise runs on the full-hand POSTFLOP legs (same
    QA as a standalone spot) and SKIPS the preflop-entry leg."""
    out = tmp_path / "fh.csv"
    client = _LifecycleClient(
        "Check to keep the pot small.",
        "Check to keep the pot small and realize equity cheaply.",
        flag=True,
    )
    generate_full_hand_batch(
        solve=btn_vs_bb_full_hand_2cJs7s(), output_path=out, total_hands=1,
        dry_run=False, client=client, heroes=("BB",), revise_pass=True,
        final_audit=True, equity_runouts=40,
    )
    import json

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    c = meta["counters"]
    assert c["revise_flagged"] >= 1
    assert c["revise_fixed"] == c["revise_flagged"]  # mock always fixes
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    pre = [r for r in rows if r["Hand Stage"] == "Preflop"]
    post = [r for r in rows if r["Hand Stage"] != "Preflop"]
    assert pre and all(r["claim_check"] == "" for r in pre)  # preflop NOT audited
    assert any(r["claim_check"] for r in post)               # postflop legs audited
    assert any("realize equity cheaply" in r["Answer Explanation"] for r in post)


def test_allin_rendered_in_action_history() -> None:
    """A bet/raise that committed the stack renders as 'moves all-in for X',
    not the raw (huge) size."""
    from pipeline.postflop.action_history import _postflop_verb_phrase
    from pipeline.postflop.solve import PostflopStep

    shove = PostflopStep("river", "BTN", "raise", to_bb=197.0, all_in=True)
    assert _postflop_verb_phrase(shove, is_hero=False) == "moves all-in for 197bb"
    assert _postflop_verb_phrase(shove, is_hero=True) == "move all-in for 197bb"
    # A normal raise still reads as "raises to X".
    normal = PostflopStep("river", "BTN", "raise", to_bb=12.0, all_in=False)
    assert _postflop_verb_phrase(normal, is_hero=False) == "raises to 12bb"


def test_hand_id_unique_per_frame() -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    hands = assemble_hands(solve, seeds=_worthy_seeds(solve), heroes=("BB",),
                           include_villain=True)
    ids = [h.hand_id for h in hands]
    assert len(ids) == len(set(ids))  # globally unique


# --- full-hand batch --------------------------------------------------------
def test_full_hand_batch_groups_and_orders(tmp_path: Path) -> None:
    out = tmp_path / "fh.csv"
    res = generate_full_hand_batch(
        solve=btn_vs_bb_full_hand_2cJs7s(), output_path=out, total_hands=3,
        dry_run=True, heroes=("BB",),
    )
    assert res.questions_written > 0
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    groups = defaultdict(list)
    for r in rows:
        assert r["hand_id"]  # every full-hand row is tagged
        groups[r["hand_id"]].append(r)
    for legs in groups.values():
        seqs = sorted(int(r["sequence_index"]) for r in legs)
        assert seqs == list(range(1, len(legs) + 1))  # contiguous 1..N
        # sequence_total was dropped from the full-hand CSV (July 2026): the
        # app derives the leg count from the hand_id group size.
        assert all("sequence_total" not in r for r in legs)
        # First leg is the preflop entry.
        first = min(legs, key=lambda r: int(r["sequence_index"]))
        assert first["Hand Stage"] == "Preflop"


def test_full_hand_preflop_leg_correct_is_continue_never_fold() -> None:
    """The play-through preflop leg's answer is the action that led here (the
    continue / Call side), never Fold -- even for a hand whose flat-call weight
    is low. The style still applies (basic -> 'Call', gto -> 'Mostly Call')."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    hands = assemble_hands(solve, seeds=_worthy_seeds(solve), heroes=("BB",))
    pre_leg = hands[0].legs[0]
    assert pre_leg.kind == "preflop_entry"
    _o, basic = build_preflop_entry_options(pre_leg.entry_facts, style="basic", as_played=True)
    assert basic == "Call"
    _o2, gto = build_preflop_entry_options(pre_leg.entry_facts, style="gto", as_played=True)
    assert "Call" in gto and "Fold" not in gto


def test_full_hand_batch_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    for out in (a, b):
        generate_full_hand_batch(
            solve=btn_vs_bb_full_hand_2cJs7s(), output_path=out, total_hands=3,
            dry_run=True, heroes=("BB",), include_villain=True, write_meta=False,
        )
    assert a.read_bytes() == b.read_bytes()


# --- #6B: premium hands excluded from STANDALONE preflop questions ----------
def _solve_with_entry_range(bb_entry: dict[str, float]):
    from pipeline.postflop.solve import (
        NodeAction,
        PostflopNode,
        PostflopSolve,
        PreflopStep,
    )

    node = PostflopNode(
        node_id="r:0", street="flop", board=("2c", "Js", "7s"), actor="BB",
        villain="BTN", pot_bb=5.5, effective_stack_bb=97.5,
        actions=(NodeAction("Check", "check", 1.0),),
        strategy={"2c2d": {"Check": 1.0}}, hero_range={"2c2d": 1.0},
        villain_range={"AcKc": 1.0},
    )
    return PostflopSolve(
        solve_id="t", positions=("BB", "BTN"), effective_stack_bb=100.0,
        starting_pot_bb=5.5, flop=("2c", "Js", "7s"),
        preflop_summary=(PreflopStep("BTN", "open", to_bb=2.5), PreflopStep("BB", "call")),
        nodes={"r:0": node}, stakes="$1/$2", bb_in_dollars=2.0, table_size=8,
        preflop_entry_ranges={"BB": bb_entry, "BTN": {"AcKc": 1.0}},
    )


def test_standalone_excludes_premium_3bet_hands(tmp_path: Path) -> None:
    """A premium defender hand at a worthy call frequency is dropped from a
    STANDALONE preflop batch (its non-call mass is a 3-bet, not a fold)."""
    from pipeline.postflop.preflop_entry import (
        build_preflop_entry_facts,
        standalone_entry_is_reliable,
    )

    # AA flat-calling 80% (worthy window) -- but AA never folds to one open, so
    # the call/fold framing is unreliable; 76s flat-calling 80% is a real defend.
    solve = _solve_with_entry_range({"AhAs": 0.80, "7c6c": 0.80, "Jh9c": 0.80})
    aa = build_preflop_entry_facts(solve, "BB", "AhAs")
    s76 = build_preflop_entry_facts(solve, "BB", "7c6c")
    assert standalone_entry_is_reliable(aa) is False
    assert standalone_entry_is_reliable(s76) is True

    out = tmp_path / "pf.csv"
    generate_preflop_entry_batch(
        solve=solve, output_path=out, total_questions=20, dry_run=True, heroes=("BB",),
    )
    import json

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["counters"]["premium_3bet_excluded"] >= 1
    cards = {r["User Cards"] for r in csv.DictReader(out.open(encoding="utf-8-sig"))}
    assert "A-spades, A-hearts" not in cards  # AA excluded
    assert "7-clubs, 6-clubs" in cards        # genuine defend kept


def test_playthrough_keeps_premium_preflop_leg() -> None:
    """A play-through KEEPS a premium preflop leg (as_played: the hand really did
    call to reach the flop) -- #6B is standalone-only."""
    solve = _solve_with_entry_range({"Jh9c": 0.65})  # connected fixture hero
    # The real check: standalone reliability is False for AA but play-through uses
    # as_played, which is unconditional. Assert the guard is caller-only + the
    # play-through path doesn't consult it.
    from pipeline.postflop.preflop_entry import (
        build_preflop_entry_facts,
        standalone_entry_is_reliable,
    )

    aa = build_preflop_entry_facts(solve, "BB", "AhAs")  # AA not in range -> freq 0
    assert standalone_entry_is_reliable(aa) is False  # would be excluded standalone
    # Opener framing is always reliable (open/fold, no hidden 3bet).
    opener = build_preflop_entry_facts(solve, "BTN", "AhAs")
    assert standalone_entry_is_reliable(opener) is True


# --- postflop ranges column (current-street ranges per active player) --------
def test_postflop_ranges_column(tmp_path: Path) -> None:
    """Postflop legs carry each active player's current-street range (hero range +
    strategy, villain range); the preflop-entry leg's ranges cell is empty."""
    import json

    out = tmp_path / "fh.csv"
    generate_full_hand_batch(
        solve=btn_vs_bb_full_hand_2cJs7s(), output_path=out, total_hands=1,
        dry_run=True, heroes=("BB",),
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert "ranges" in rows[0]
    pre = [r for r in rows if r["Hand Stage"] == "Preflop"]
    post = [r for r in rows if r["Hand Stage"] != "Preflop"]
    assert pre and all(r["ranges"] == "" for r in pre)  # no accurate preflop range
    for r in post:
        rg = json.loads(r["ranges"])
        assert len(rg) == 2  # both players
        actors = [p for p, v in rg.items() if v["acting"]]
        assert len(actors) == 1
        hero = rg[actors[0]]
        assert hero["range"] and hero["strategy"]      # hero: range + action mix
        villain = rg[[p for p in rg if p != actors[0]][0]]
        # Villain always carries their range (holdings). When they ALSO acted on
        # this street, the entry adds their action-mix strategy at their own
        # action point (acted_this_street=True); otherwise it is range-only.
        assert villain["range"]
        if villain.get("acted_this_street"):
            assert villain["strategy"]
        else:
            assert "strategy" not in villain
        # strategy frequencies are within a class's presence (segments <= weight+eps)
        for hand, acts in hero["strategy"].items():
            assert sum(acts.values()) <= hero["range"].get(hand, 0) + 0.02


def test_postflop_ranges_deterministic() -> None:
    """The ranges JSON is byte-stable (sorted keys) for the same node."""
    from pipeline.postflop.range_export import build_active_ranges_json

    solve = btn_vs_bb_full_hand_2cJs7s()
    node = solve.nodes["r:0:c:b180"]
    assert build_active_ranges_json(node) == build_active_ranges_json(node)


# --- preflop-entry prompt override (admin-editable) -------------------------
def test_preflop_entry_prompt_override(tmp_path: Path, monkeypatch) -> None:
    """load_preflop_entry_system_prompt returns the built-in by default and the
    admin override file when present."""
    import pipeline.postflop.preflop_entry as pe

    monkeypatch.setattr(pe, "_PREFLOP_ENTRY_PROMPT_OVERRIDE_PATH", tmp_path / "missing.txt")
    assert pe.load_preflop_entry_system_prompt() == pe.PREFLOP_ENTRY_SYSTEM_PROMPT
    override = tmp_path / "preflop_entry_system.txt"
    override.write_text("CUSTOM PREFLOP PROMPT", encoding="utf-8")
    monkeypatch.setattr(pe, "_PREFLOP_ENTRY_PROMPT_OVERRIDE_PATH", override)
    assert pe.load_preflop_entry_system_prompt() == "CUSTOM PREFLOP PROMPT"


# --- full-hand re-verifier (rebuild logic, no .db needed) -------------------
def test_full_hand_reverifier_rebuilds_every_leg(tmp_path: Path) -> None:
    """The audit's per-leg rebuild reproduces the EXACT deterministic columns of
    every leg (preflop + postflop) of a fixture full-hand batch."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import audit_full_hand_batch as afh  # noqa: PLC0415

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "fh.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True, heroes=("BB",),
    )
    import json

    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    qs = meta["questions"]
    assert len(rows) == len(qs)
    # Each hand's FINAL leg carries the showdown resolution; the audit
    # re-attaches it before comparing (same seeded inputs), and so must we.
    final_seq: dict[str, int] = {}
    for q in qs:
        hid = q.get("hand_id", "")
        if hid:
            final_seq[hid] = max(
                final_seq.get(hid, 0), int(q.get("sequence_index") or 0)
            )
    checked_pre = checked_post = 0
    for row, q in zip(rows, qs, strict=True):
        if (q.get("street") == "preflop") or not q.get("node_id"):
            rebuilt, opts, correct = afh._rebuild_preflop(
                row, q, solve, answer_style="auto", display_in_bb=True
            )
            checked_pre += 1
        else:
            rebuilt, opts, correct = afh._rebuild_postflop(
                row, q, solve, answer_style="auto", display_in_bb=True,
                equity_runouts=int(meta["run_settings"]["equity_runouts"]),
            )
            checked_post += 1
            hid = q.get("hand_id", "")
            if (
                rebuilt is not None and hid
                and int(q.get("sequence_index") or 0) == final_seq.get(hid)
            ):
                from pipeline.postflop.showdown import (  # noqa: PLC0415
                    attach_showdown_resolution,
                )

                attach_showdown_resolution(
                    rebuilt, node=solve.nodes[q["node_id"]], solve=solve,
                    hero_combo=q["hero_combo"], correct_answer=correct,
                    hand_id=hid,
                )
        assert rebuilt is not None
        for col in afh.EXACT_COLS:
            assert rebuilt.get(col, "") == row.get(col, ""), (col, row["No"])
    assert checked_pre >= 1 and checked_post >= 1  # both leg types exercised


# --- standalone preflop batch -----------------------------------------------
def test_preflop_entry_batch_blank_hand_id(tmp_path: Path) -> None:
    out = tmp_path / "pf.csv"
    res = generate_preflop_entry_batch(
        solve=btn_vs_bb_srp_2cJs7s(), output_path=out, total_questions=5,
        dry_run=True, heroes=("BB",),
    )
    assert res.questions_written > 0
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    for r in rows:
        assert r["hand_id"] == ""  # standalone: no play-through linkage
        assert r["Hand Stage"] == "Preflop"
        # worthiness-gated to a real defend (mixed call frequency).
        assert r["Correct Answer"] in ("Call", "Mostly Call", "Always Call")
