"""Tests for pack-backed preflop legs in full-hand play-throughs (July 2026).

A tmp ryan-grammar fixture pack is crafted to MATCH the in-memory full-hand
fixture solve (6-max, 100bb, 2.5bb open), so the whole path runs in CI with
no real pack and no .db: the matcher's three gates (geometry, line,
coherence), the leg upgrade itself (EVs/stat_notes/ranges/difficulty), the
entry-derived fallback, and the hand-difficulty column + band filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

import pytest  # noqa: E402

import pipeline.postflop.full_hand_batch as fhb  # noqa: E402
from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s  # noqa: E402
from pipeline.postflop.full_hand_batch import (  # noqa: E402
    generate_full_hand_batch,
)
from pipeline.postflop.preflop_leg_pack import (  # noqa: E402
    find_pack_leg_source,
)
from pipeline.preflop.pack import (  # noqa: E402
    PreflopPack,
    clear_registry,
)
from pipeline.preflop_ranges import canonical_169_hand_classes  # noqa: E402

_LINE = "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%"


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _write(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = canonical_169_hand_classes()
    path.write_text(",".join(f"{c}:{weights.get(c, 0.0)}" for c in classes))


def _matching_pack(
    tmp_path: Path, *, call_freq: float = 0.7, threebet_freq: float = 0.1,
    open_token: str = "60%", pack_id: str = "fixture_6max_100bb",
    stack: int = 100,
) -> PreflopPack:
    """A ryan-grammar pack with the BTN-open + BB-defend line. Default
    weights make CALL the dominant defend for every class (coherence holds
    for every BB hand); the BTN opens everything."""
    root = tmp_path / pack_id
    classes = canonical_169_hand_classes()
    line = _LINE.replace("60%", open_token)
    _write(root / "BTN" / f"{line}.txt", {c: 1.0 for c in classes})
    _write(root / "BTN" / f"{_LINE.rsplit('_', 1)[0]}_Fold.txt", {})
    fold_freq = max(0.0, 1.0 - call_freq - threebet_freq)
    _write(root / "BB" / f"{line}_SB_Fold_BB_Call.txt",
           {c: call_freq for c in classes})
    _write(root / "BB" / f"{line}_SB_Fold_BB_Fold.txt",
           {c: fold_freq for c in classes})
    _write(root / "BB" / f"{line}_SB_Fold_BB_182%.txt",
           {c: threebet_freq for c in classes})
    return PreflopPack(
        pack_id=pack_id, root_path=root, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=stack, open_size_bb=2.5,
        description="full-hand pack-leg fixture",
    )


# --- the matcher's gates -------------------------------------------------------
def test_matcher_accepts_the_matching_pack(tmp_path: Path) -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None and src.pack_id == pack.pack_id
    assert src.open_size_bb == pytest.approx(2.5)
    assert src.opener_node.actor == "BTN"
    assert src.defender_node.actor == "BB"


def test_matcher_rejects_wrong_stack(tmp_path: Path) -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()  # 100bb
    pack = _matching_pack(tmp_path, stack=200, pack_id="fixture_200bb")
    assert find_pack_leg_source(solve, tmp_path, packs=[pack]) is None


def test_matcher_rejects_wrong_open_size(tmp_path: Path) -> None:
    """The pot-geometry gate: a 76% open resolves to 3.0bb, not the solve's
    2.5bb -- a mismatched open would make the preflop leg's pot math
    contradict the postflop legs of the same hand."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path, open_token="76%", pack_id="fixture_3x")
    assert find_pack_leg_source(solve, tmp_path, packs=[pack]) is None


def test_matcher_prefers_improved_packs(tmp_path: Path) -> None:
    solve = btn_vs_bb_full_hand_2cJs7s()
    plain = _matching_pack(tmp_path, pack_id="fixture_pack")
    improved = _matching_pack(tmp_path, pack_id="fixture_pack_IMPROVED")
    src = find_pack_leg_source(solve, tmp_path, packs=[plain, improved])
    assert src is not None and src.pack_id == "fixture_pack_IMPROVED"


# --- the batch integration -------------------------------------------------------
def _run_batch(tmp_path: Path, monkeypatch, pack: PreflopPack | None, **kwargs):
    solve = btn_vs_bb_full_hand_2cJs7s()
    if pack is not None:
        src = find_pack_leg_source(solve, tmp_path, packs=[pack])
        assert src is not None
        monkeypatch.setattr(fhb, "find_pack_leg_source", lambda *a, **k: src)
    out = tmp_path / "out.csv"
    result = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=3, dry_run=True,
        answer_style="gto", equity_runouts=20,
        preflop_leg_pack_root=(tmp_path if pack is not None else None),
        **kwargs,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return result, meta


def test_pack_legs_upgrade_the_preflop_question(tmp_path, monkeypatch) -> None:
    """With a matching pack, every preflop leg is pack-backed: real 4-axis
    difficulty, stat_notes for the math panel, the ranges JSON, and a
    solver_reference into the PACK node. hand_difficulty stamps every leg."""
    import csv as _csv

    pack = _matching_pack(tmp_path)
    result, meta = _run_batch(tmp_path, monkeypatch, pack)
    assert result.questions_written > 0
    c = meta["counters"]
    assert c["preflop_leg_pack_used"] > 0
    assert c["preflop_leg_entry_fallback"] == 0
    assert meta["run_settings"]["preflop_leg_pack"] == pack.pack_id
    rows = list(_csv.DictReader(
        (tmp_path / "out.csv").open(encoding="utf-8-sig")
    ))
    pre = [r for r in rows if r["Hand Stage"] == "Preflop"]
    assert pre
    for r in pre:
        # Defender legs (facing the open) always carry decision math; an
        # OPENER leg has no villain/price, so its panel comes from the
        # per-action EVs -- which this ryan-grammar fixture pack doesn't
        # ship (the real 8-max packs do).
        if "_vs_" in r["Position Matchup"]:
            assert r["stat_notes"], "defender pack leg must carry the math"
        assert r["ranges"], "pack leg must carry the ranges JSON"
        # The node reference (pack-relative) lives in Notes' Node: field, and
        # the provenance sentence names the source pack. Both check pack_id.
        assert pack.pack_id in r["Notes"]
        from pipeline.provenance import node_reference_from_notes

        assert pack.pack_id in node_reference_from_notes(r["Notes"])
        assert r["Cards on Table"] == ""
    for r in rows:
        assert r["hand_difficulty"], "every full-hand row carries hand_difficulty"
    # hand_difficulty == the peak-anchored blend over the hand's legs
    # (0.65 x hardest + 0.35 x mean of the rest -- July 2026).
    from pipeline.postflop.difficulty import aggregate_hand_difficulty

    by_hand: dict[str, list[dict]] = {}
    for r in rows:
        by_hand.setdefault(r["hand_id"], []).append(r)
    for legs in by_hand.values():
        expected = aggregate_hand_difficulty(
            [int(r["Difficulty Rating"]) for r in legs]
        )
        assert {r["hand_difficulty"] for r in legs} == {str(expected)}


def test_coherence_gate_drops_the_whole_hand_no_flop_start(tmp_path, monkeypatch) -> None:
    """NO FLOP-START HANDS (July 15 2026, the user's call, tightening the
    July 13 pack-first rule): when the pack says the hand mostly 3-BETS,
    the as-played call contradicts the pack's correct answer -- the WHOLE
    hand is dropped (the old rule shipped it without a preflop question,
    so play-throughs started at the flop on a preflop premise the chart
    refuses, e.g. a K4o open the pack plays as Fold 64%). Every shipped
    hand must start with a preflop question."""
    import csv as _csv

    pack = _matching_pack(
        tmp_path, call_freq=0.2, threebet_freq=0.7, pack_id="fixture_3betty",
    )
    result, meta = _run_batch(tmp_path, monkeypatch, pack)
    assert result.questions_written > 0
    c = meta["counters"]
    # Every BB defend leg disagrees (dominant 3-bet) -> its HAND is dropped.
    assert c["preflop_entry_legs_dropped"] > 0
    assert c["hands_dropped_preflop_incoherent"] > 0
    assert c["preflop_leg_entry_fallback"] == 0
    # INVARIANT: no shipped hand starts mid-hand -- the first leg of every
    # hand is a Preflop question.
    rows = list(_csv.DictReader(
        (tmp_path / "out.csv").open(encoding="utf-8-sig")
    ))
    firsts = [r for r in rows if r["sequence_index"] == "1"]
    assert firsts
    for r in firsts:
        assert r["Hand Stage"] == "Preflop", (r["hand_id"], r["Hand Stage"])


def test_hand_difficulty_band_filter(tmp_path, monkeypatch) -> None:
    """The band filter drops whole hands by hand_difficulty BEFORE any
    generation; an impossible band writes zero hands and records the
    band-scan diagnostics the Review page reads to explain the empty CSV."""
    pack = _matching_pack(tmp_path)
    result, meta = _run_batch(
        tmp_path, monkeypatch, pack,
        min_hand_difficulty=3200, max_hand_difficulty=3200,
    )
    assert result.questions_written == 0
    c = meta["counters"]
    assert c["hands_difficulty_filtered"] == c["hands_assembled"] > 0
    # Diagnostics (July 2026): the hardest hand seen must be recorded (and,
    # being below the impossible band, must be under its floor).
    assert isinstance(c["hand_difficulty_observed_max"], int)
    assert c["hand_difficulty_observed_max"] < 3200  # noqa: PLR2004


def test_band_scan_looks_past_the_requested_count(tmp_path, monkeypatch) -> None:
    """BAND-SCAN INVARIANT (July 2026, seen live): total_hands caps the KEPT
    hands, never the scanned ones. The old flow assembled exactly N hands
    and then band-filtered, so "1 Hard hand" assembled one arbitrary hand
    and usually deleted it -- a silent empty batch. Requesting 1 hand with a
    band that only a NON-FIRST hand satisfies must still find it."""
    import csv as _csv

    solve = btn_vs_bb_full_hand_2cJs7s()

    # Unfiltered reference run: every hand + its difficulty, in kept order.
    # action_heavy=False on BOTH runs: this test exercises the band-scan
    # MECHANICS against the aggregate hand_difficulty; with the 🎬 policy on
    # (the default) the band judges the postflop spine instead (its own
    # tests live in test_full_hand_action_heavy.py).
    out_all = tmp_path / "all.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out_all, total_hands=50, dry_run=True,
        answer_style="gto", equity_runouts=20, include_villain=True,
        action_heavy=False,
    )
    meta_all = json.loads(
        out_all.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    hands = meta_all["hands"]
    if len({h["hand_difficulty"] for h in hands}) < 2:  # pragma: no cover
        import pytest as _pytest

        _pytest.skip("fixture hands all rate identically; cannot exercise")
    first = hands[0]["hand_difficulty"]
    target = next(
        h["hand_difficulty"] for h in hands if h["hand_difficulty"] != first
    )

    # Request ONE hand in a band only the non-first hand(s) satisfy.
    out = tmp_path / "banded.csv"
    result = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=1, dry_run=True,
        answer_style="gto", equity_runouts=20, include_villain=True,
        min_hand_difficulty=target, max_hand_difficulty=target,
        action_heavy=False,
    )
    assert result.questions_written > 0, "band scan failed to look past hand #1"
    rows = list(_csv.DictReader(out.open(encoding="utf-8-sig")))
    assert {r["hand_difficulty"] for r in rows} == {str(target)}
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["counters"]["hands_written"] == 1


def test_no_pack_root_keeps_entry_legs(tmp_path, monkeypatch) -> None:
    """preflop_leg_pack_root=None (the default) = the pre-existing
    entry-derived behaviour, byte for byte."""
    result, meta = _run_batch(tmp_path, monkeypatch, None)
    assert result.questions_written > 0
    c = meta["counters"]
    assert c["preflop_leg_pack_used"] == 0
    assert meta["run_settings"]["preflop_leg_pack"] is None


# --- trimmed full-hand CSV schema (July 2026) --------------------------------
def test_full_hand_csv_uses_trimmed_schema(tmp_path, monkeypatch) -> None:
    """The full-hand CSV drops the pot_odds..easy_hand diagnostic columns
    (their values still live inside stat_notes) and KEEPS the sequence tags
    + hand_difficulty. Standalone postflop batches keep the full schema."""
    import csv as _csv

    from pipeline.postflop.format_writer import (
        _FULL_HAND_DROPPED_COLUMNS,
        FULL_HAND_CSV_COLUMNS,
        POSTFLOP_CSV_COLUMNS,
    )

    assert _FULL_HAND_DROPPED_COLUMNS == frozenset({
        "pot_odds", "hero_equity", "range_equity", "spr",
        "easy_freq", "easy_ev", "easy_concept", "easy_hand",
        # July 2026 (team): the app groups on hand_id + orders by
        # sequence_index; a leg count is just the group size.
        "sequence_total",
    })
    assert set(POSTFLOP_CSV_COLUMNS) - set(FULL_HAND_CSV_COLUMNS) == set(
        _FULL_HAND_DROPPED_COLUMNS
    )
    # The trailing columns read: ...sequence tags, then the hand selector.
    # (animation_script sits in the shared prefix, like chat_context.)
    assert FULL_HAND_CSV_COLUMNS[-3:] == (
        "hand_id", "sequence_index", "hand_difficulty",
    )
    assert "animation_script" in FULL_HAND_CSV_COLUMNS

    pack = _matching_pack(tmp_path)
    result, _meta = _run_batch(tmp_path, monkeypatch, pack)
    assert result.questions_written > 0
    with (tmp_path / "out.csv").open(encoding="utf-8-sig", newline="") as fh:
        header = next(_csv.reader(fh))
    assert header == list(FULL_HAND_CSV_COLUMNS)


# --- prompt threading (July 2026) --------------------------------------------
def test_batch_threads_pack_prompt_to_leg_builder(tmp_path, monkeypatch) -> None:
    """generate_full_hand_batch forwards preflop_pack_system_prompt to every
    pack-leg build (the admin picker's value reaches the leg builder)."""
    pack = _matching_pack(tmp_path)
    seen: list = []
    real = fhb.build_pack_preflop_leg_row

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("system_prompt"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fhb, "build_pack_preflop_leg_row", _spy)
    _result, meta = _run_batch(
        tmp_path, monkeypatch, pack,
        preflop_pack_system_prompt="PACK-PROMPT-X",
    )
    assert meta["counters"]["preflop_leg_pack_used"] > 0
    assert seen and all(sp == "PACK-PROMPT-X" for sp in seen)


def test_pack_leg_forwards_system_prompt_to_generator(tmp_path, monkeypatch) -> None:
    """build_pack_preflop_leg_row passes its system_prompt override into the
    PREFLOP explanation generator (not the postflop or preflop-entry one)."""
    import pipeline.preflop.explanation_generator as pre_gen
    from pipeline.explanation_generator import GeneratedExplanation
    from pipeline.postflop.preflop_leg_pack import build_pack_preflop_leg_row

    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None
    # A coherent (position, combo) pair, taken from a dry-run batch's meta.
    _result, meta = _run_batch(tmp_path, monkeypatch, pack)
    q = next(
        q for q in meta["questions"] if q.get("preflop_leg_source") == "pack"
    )

    captured: dict = {}

    def _fake_gen(facts, options, correct, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt")
        padded = list(options) + ["", "", "", ""]
        return GeneratedExplanation(
            option_1=padded[0], option_2=padded[1], option_3=padded[2],
            option_4=padded[3], correct_answer=correct,
            answer_explanation="Test prose.",
        )

    monkeypatch.setattr(
        pre_gen, "generate_preflop_answer_explanation", _fake_gen
    )
    row, record, failure = build_pack_preflop_leg_row(
        src, q["hero_position"], q["hero_combo"], solve,
        number=1, hand_id="h1", sequence_index=1, sequence_total=2,
        use_placeholder=False, client=object(), model="test-model",
        temperature=0.2, max_tokens=100, answer_style="gto",
        display_in_bb=True, equity_runouts=20,
        system_prompt="CUSTOM PREFLOP PROMPT",
    )
    assert failure is None and row is not None and record is not None
    assert captured["system_prompt"] == "CUSTOM PREFLOP PROMPT"


def test_pack_leg_usage_callback_arity_is_adapted(tmp_path, monkeypatch) -> None:
    """REGRESSION (July 2026): the postflop batch's usage counter takes ONE
    usage object, but the PREFLOP generator reports five positionals
    (model, in, out, cache_c, cache_r) -- the pack-leg seam must adapt, or
    the FIRST real-API full-hand run dies with a TypeError no dry-run can
    see (the June-2026 usage_callback bug class). This test drives the REAL
    preflop generator through a mock CLIENT -- mocking the generator itself
    would bypass the callback arity entirely (which is how it was missed).
    """
    from types import SimpleNamespace

    from pipeline.postflop.preflop_leg_pack import build_pack_preflop_leg_row

    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None
    _result, meta = _run_batch(tmp_path, monkeypatch, pack)
    q = next(
        q for q in meta["questions"] if q.get("preflop_leg_source") == "pack"
    )

    def _create(**_kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(
                text='{"answer_explanation": "This is a standard play."}'
            )],
            usage=SimpleNamespace(input_tokens=111, output_tokens=22),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=_create))

    totals = {"in": 0, "out": 0}

    def _usage(usage) -> None:  # the postflop batch's ONE-object convention
        totals["in"] += int(getattr(usage, "input_tokens", 0) or 0)
        totals["out"] += int(getattr(usage, "output_tokens", 0) or 0)

    row, record, failure = build_pack_preflop_leg_row(
        src, q["hero_position"], q["hero_combo"], solve,
        number=1, hand_id="h1", sequence_index=1, sequence_total=2,
        use_placeholder=False, client=client, model="test-model",
        temperature=0.0, max_tokens=64, answer_style="gto",
        display_in_bb=True, equity_runouts=20, usage_cb=_usage,
    )
    assert failure is None and row is not None and record is not None
    assert totals == {"in": 111, "out": 22}


def test_prompt_names_recorded_in_run_settings(tmp_path, monkeypatch) -> None:
    """The admin's chosen prompt NAMES land in meta run_settings so a batch
    always says which named prompts wrote it."""
    names = {"postflop": "My postflop prompt", "preflop_pack": "My preflop prompt"}
    _result, meta = _run_batch(tmp_path, monkeypatch, None, prompt_names=names)
    assert meta["run_settings"]["prompt_names"] == names
    # Omitted -> recorded as an empty map (not missing), so consumers can
    # rely on the key existing on new batches.
    _result2, meta2 = _run_batch(tmp_path, monkeypatch, None)
    assert meta2["run_settings"]["prompt_names"] == {}


# --- 3-bet-pot line legs (July 2026) ------------------------------------------
def _threebet_pack(
    tmp_path: Path, *, threebet_freq: float = 0.8, call_freq: float = 0.1,
    pack_id: str = "fixture_6max_100bb_3bp",
) -> PreflopPack:
    """The SRP fixture pack EXTENDED with the 3-bet line: BB's sized 3-bet is
    dominant (default), and BTN's facing-the-3-bet node mostly calls."""
    pack = _matching_pack(
        tmp_path, call_freq=call_freq, threebet_freq=threebet_freq,
        pack_id=pack_id,
    )
    classes = canonical_169_hand_classes()
    line3 = f"{_LINE}_SB_Fold_BB_182%"
    _write(pack.root_path / "BTN" / f"{line3}_BTN_Call.txt",
           {c: 0.8 for c in classes})
    _write(pack.root_path / "BTN" / f"{line3}_BTN_Fold.txt",
           {c: 0.2 for c in classes})
    return pack


def _as_three_bet_pot(solve, *, threebet_to: float):
    """The fixture solve with a 3-bet-pot preflop line (BB is the 3-bettor).
    The open matches the fixture pack's 2.5bb; ``threebet_to`` must match
    the pack's resolved 3-bet size for the geometry gate to pass."""
    from dataclasses import replace

    from pipeline.postflop.solve import PreflopStep

    return replace(solve, preflop_summary=(
        PreflopStep("BTN", "open", to_bb=2.5),
        PreflopStep("BB", "3-bet", to_bb=threebet_to),
        PreflopStep("BTN", "call"),
    ))


def _pack_threebet_size_bb(pack) -> float:
    """The pack's resolved (bb) 3-bet size on the fixture line, discovered
    rather than hardcoded so the test tracks the grammar's size table."""
    from pipeline.preflop.action_history import resolve_preflop_history
    from pipeline.preflop.grammars.types import PreflopActionType
    from pipeline.preflop.node_enumerator import enumerate_nodes

    for node in enumerate_nodes([pack]):
        acted = [a for a in node.history_before
                 if a.action_type is not PreflopActionType.FOLD]
        if node.actor == "BTN" and len(acted) == 2:  # facing the 3-bet
            sizes = [s for s in
                     resolve_preflop_history(node.history_before, pack).sizes_bb
                     if s is not None]
            return float(sizes[-1])
    raise AssertionError("fixture pack lacks the facing-3-bet node")


def test_matcher_resolves_the_three_bet_line(tmp_path: Path) -> None:
    """A pack carrying the whole line matches with one node per decision;
    a wrong 3-bet size or a pack without the line does not."""
    pack = _threebet_pack(tmp_path)
    size = _pack_threebet_size_bb(pack)
    solve = _as_three_bet_pot(btn_vs_bb_full_hand_2cJs7s(), threebet_to=size)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None
    assert [(s.position, s.as_played_prefix) for s in src.steps] == [
        ("BTN", "Raise"), ("BB", "Raise"), ("BTN", "Call"),
    ]
    assert src.steps[1].size_bb == size
    assert len(src.steps_for("BTN")) == 2 and len(src.steps_for("BB")) == 1
    # Wrong 3-bet size -> geometry gate refuses.
    wrong = _as_three_bet_pot(btn_vs_bb_full_hand_2cJs7s(), threebet_to=size + 2)
    assert find_pack_leg_source(wrong, tmp_path, packs=[pack]) is None
    # A pack WITHOUT the facing-3-bet node (the plain SRP fixture) never
    # matches a 3-bet-pot line.
    srp_only = _matching_pack(tmp_path, pack_id="fixture_srp_only")
    assert find_pack_leg_source(solve, tmp_path, packs=[srp_only]) is None


def test_full_hand_three_bet_pot_builds_line_legs(tmp_path, monkeypatch) -> None:
    """A 3-bet-pot full-hand batch: the opener (BTN) gets TWO preflop legs
    (the open + the call of the 3-bet), the 3-bettor (BB) one, sequence
    numbering contiguous, all from the pack."""
    pack = _threebet_pack(tmp_path)
    size = _pack_threebet_size_bb(pack)
    solve = _as_three_bet_pot(btn_vs_bb_full_hand_2cJs7s(), threebet_to=size)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None
    monkeypatch.setattr(fhb, "find_pack_leg_source", lambda *a, **k: src)

    out = tmp_path / "out.csv"
    result = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=4, dry_run=True,
        answer_style="gto", equity_runouts=20, preflop_leg_pack_root=tmp_path,
    )
    assert result.questions_written > 0
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    by_hand: dict[str, list[dict]] = {}
    for q in meta["questions"]:
        by_hand.setdefault(q["hand_id"], []).append(q)
    saw_btn = saw_bb = False
    for _hand_id, qs in by_hand.items():
        pre = [q for q in qs if q["street"] == "preflop"]
        hero = qs[0]["hero_position"]
        assert all(q.get("preflop_leg_source") == "pack" for q in pre)
        if hero == "BTN" and pre:
            saw_btn = True
            assert [q["preflop_step_index"] for q in pre] == [0, 2]
            assert [q["sequence_index"] for q in pre] == [1, 2]
        elif hero == "BB" and pre:
            saw_bb = True
            assert [q["preflop_step_index"] for q in pre] == [1]
        # Sequence indices are contiguous across the whole hand.
        assert [q["sequence_index"] for q in qs] == list(range(1, len(qs) + 1))
    assert saw_btn and saw_bb
    assert meta["counters"]["preflop_line_legs_dropped"] == 0
    assert meta["counters"]["preflop_leg_pack_used"] >= 3


def test_three_bet_pot_drops_incoherent_or_packless_hands(tmp_path, monkeypatch) -> None:
    """No honest fallback exists for a multi-raise preflop leg, and since
    July 15 2026 (the user's no-flop-start rule) a refused leg drops the
    WHOLE hand: a hand whose pack strategy contradicts the as-played line
    is gone, and with no matching pack at all a 3-bet-pot full-hand batch
    writes ZERO hands (the counters explain the empty CSV) rather than
    play-throughs that start mid-hand."""
    # Pack where the BB mostly CALLS (3-bet freq 0.1) -> the BB 3-bettor's
    # leg fails coherence and its whole hand is dropped; BTN hands (both
    # legs coherent) survive.
    pack = _threebet_pack(
        tmp_path, threebet_freq=0.1, call_freq=0.7, pack_id="fixture_3bp_calls",
    )
    size = _pack_threebet_size_bb(pack)
    solve = _as_three_bet_pot(btn_vs_bb_full_hand_2cJs7s(), threebet_to=size)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None
    monkeypatch.setattr(fhb, "find_pack_leg_source", lambda *a, **k: src)
    out = tmp_path / "out.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=4, dry_run=True,
        equity_runouts=20, preflop_leg_pack_root=tmp_path,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    bb_pre = [q for q in meta["questions"]
              if q["street"] == "preflop" and q["hero_position"] == "BB"]
    assert bb_pre == []  # coherence-dropped, never faked
    assert meta["counters"]["preflop_line_legs_dropped"] >= 1
    assert meta["counters"]["hands_dropped_preflop_incoherent"] >= 1
    # Surviving hands all start preflop (the BTN opener's leg 1).
    hand_ids = {q["hand_id"] for q in meta["questions"]}
    for hid in hand_ids:
        first = min(
            (q for q in meta["questions"] if q["hand_id"] == hid),
            key=lambda q: q["sequence_index"],
        )
        assert first["street"] == "preflop", (hid, first["street"])

    # No pack at all -> every hand's preflop leg is unbuildable -> the batch
    # is EMPTY (never flop-start hands), with the counter explaining why.
    monkeypatch.setattr(fhb, "find_pack_leg_source", lambda *a, **k: None)
    out2 = tmp_path / "out2.csv"
    result = generate_full_hand_batch(
        solve=solve, output_path=out2, total_hands=2, dry_run=True,
        equity_runouts=20, preflop_leg_pack_root=tmp_path,
    )
    assert result.questions_written == 0
    meta2 = json.loads(out2.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta2["counters"]["hands_dropped_preflop_incoherent"] >= 1


def test_preflop_entry_refuses_a_three_bet_pot_solve(tmp_path: Path) -> None:
    """STANDALONE preflop-entry mode stays SRP-only: its continue-or-fold
    framing cannot express a raise-or-call-or-fold decision."""
    from pipeline.postflop.full_hand_batch import generate_preflop_entry_batch

    solve = _as_three_bet_pot(btn_vs_bb_full_hand_2cJs7s(), threebet_to=11.0)
    with pytest.raises(ValueError, match="single-raised-pot"):
        generate_preflop_entry_batch(
            solve=solve, output_path=tmp_path / "out.csv", total_questions=1,
            dry_run=True,
        )


# --- variety seed (July 2026): fresh hands per batch --------------------------
def test_variety_order_shuffles_within_depth_only() -> None:
    """INVARIANT: cross-depth deepest-first order must survive the shuffle
    (the dedup keeps a combo's LONGEST line only because it is processed
    first); within one depth the order is free, and that freedom is what
    stops every batch opening with the same digit-first combos."""
    from types import SimpleNamespace

    from pipeline.postflop.play_through import _variety_order

    def spot(board_len: int, hist_len: int, combo: str):
        node = SimpleNamespace(board=["x"] * board_len, history=["a"] * hist_len)
        return SimpleNamespace(node=node, hero_combo=combo)

    deep = [spot(5, 4, c) for c in ("4d4c", "5c4c", "AhKh", "QsQd", "7c6c")]
    shallow = [spot(3, 0, c) for c in ("2d2c", "JhTh")]
    seeds = deep + shallow  # already deepest-first, as assemble_hands sorts

    # None = the legacy fixed order, untouched.
    assert _variety_order(seeds, None) == seeds
    # Same seed = same order (a batch is reproducible from its meta).
    once = _variety_order(seeds, 42)
    assert _variety_order(seeds, 42) == once
    # Depth groups never interleave: all 5 deep seeds still precede both
    # shallow seeds, whatever the permutation inside each group.
    combos = lambda spots: {s.hero_combo for s in spots}  # noqa: E731
    assert combos(once[:5]) == combos(deep) and combos(once[5:]) == combos(shallow)
    # Different seeds produce different picks at the front (this is the fix:
    # the first N seeds decide the batch). 5! orderings make a collision
    # across 5 seeds astronomically unlikely; assert at least two differ.
    fronts = {tuple(s.hero_combo for s in _variety_order(seeds, k)[:3]) for k in range(5)}
    assert len(fronts) > 1


def test_variety_seed_changes_the_assembled_hands(tmp_path: Path) -> None:
    """End to end on the fixture: same seed = same hands; the seed lands in
    meta run_settings so the batch says how it was drawn."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "seeded.csv"
    result = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        equity_runouts=20, variety_seed=7,
    )
    assert result.questions_written > 0
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["run_settings"]["variety_seed"] == 7
    # Reproducible: the same seed assembles the same hands.
    out2 = tmp_path / "seeded2.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out2, total_hands=2, dry_run=True,
        equity_runouts=20, variety_seed=7,
    )
    meta2 = json.loads(out2.with_suffix(".meta.json").read_text(encoding="utf-8"))
    hands = [q["hero_combo"] for q in meta["questions"]]
    assert [q["hero_combo"] for q in meta2["questions"]] == hands


# --- whole-hand atomicity + QA wave (July 2026) --------------------------------
def test_failed_leg_drops_the_whole_hand(tmp_path, monkeypatch) -> None:
    """A play-through with a missing street is a broken story: any leg
    failure discards the WHOLE hand, numbering stays contiguous, and the
    drop is counted."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    real = fhb._postflop_leg_row
    failed_hands: set = set()

    def flaky(spot, solve_, **kwargs):
        row, record, failure, counters = real(spot, solve_, **kwargs)
        # Fail exactly one hand: the first hand's first postflop leg.
        if not failed_hands or kwargs["hand_id"] in failed_hands:
            failed_hands.add(kwargs["hand_id"])
            return None, None, {
                "node_id": spot.node.node_id, "hero_combo": spot.hero_combo,
                "hand_id": kwargs["hand_id"], "error_message": "boom",
            }, counters
        return row, record, failure, counters

    monkeypatch.setattr(fhb, "_postflop_leg_row", flaky)
    out = tmp_path / "out.csv"
    result = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        equity_runouts=20,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["counters"]["hands_dropped_failed_leg"] == 1
    assert len(result.failures) >= 1
    # No row from the failed hand survives, and numbering is contiguous.
    written_hands = {q["hand_id"] for q in meta["questions"]}
    assert failed_hands.isdisjoint(written_hands)
    import csv as _csv

    rows = list(_csv.DictReader(open(out, encoding="utf-8-sig")))
    assert [int(r["No"]) for r in rows] == list(range(1, len(rows) + 1))


def test_cross_check_runs_and_counts(tmp_path) -> None:
    """The deterministic cross-check re-reads every written row; a clean
    dry-run batch reports zero problems, and a doctored row is caught."""
    from pipeline.postflop.preflop_leg_pack import run_full_hand_cross_check

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "out.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        equity_runouts=20,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["counters"]["cross_check_problems"] == 0
    # Doctored: BTN vs BB flagged as Out of Position must be caught.
    bad_row = {
        "Position Matchup": "BTN_vs_BB", "Relative Position": "Out of Position",
        "Difficulty Rating": "1000",
    }
    found = run_full_hand_cross_check([bad_row], [{}])
    assert found and any("Relative Position" in i for i in found[0])


def test_pack_leg_layer7_flag_and_revise(tmp_path, monkeypatch) -> None:
    """The pack preflop legs run the PREFLOP checker/reviser: flag-only
    records issues; revise ships the rewrite and records the lifecycle."""
    from types import SimpleNamespace

    import pipeline.preflop.batch as preflop_batch
    import pipeline.preflop.reviser as preflop_reviser
    from pipeline.postflop.preflop_leg_pack import build_pack_preflop_leg_row

    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None

    fake_issue = SimpleNamespace(claim="the pot is 9bb", problem="it is 5.5bb")
    monkeypatch.setattr(
        preflop_batch, "_safe_claim_check",
        lambda *a, **k: SimpleNamespace(issues=[fake_issue]),
    )

    def fake_generate(facts, options, correct, **kwargs):
        from pipeline.explanation_generator import GeneratedExplanation

        return GeneratedExplanation(
            option_1=options[0], option_2=options[1],
            option_3=options[2] if len(options) > 2 else "",
            option_4=options[3] if len(options) > 3 else "",
            correct_answer=correct, answer_explanation="Original prose.",
        )

    import pipeline.preflop.explanation_generator as preflop_gen

    monkeypatch.setattr(
        preflop_gen, "generate_preflop_answer_explanation", fake_generate,
    )

    common = dict(
        number=1, hand_id="h1", sequence_index=1, sequence_total=3,
        use_placeholder=False, client=object(), model="test-model",
        temperature=0.0, max_tokens=64, answer_style="gto",
        display_in_bb=True, equity_runouts=20,
    )
    # Flag only: issues recorded, row flagged, prose untouched.
    row, record, failure = build_pack_preflop_leg_row(
        src, "BB", "7h6h", solve, run_claim_checker=True, **common,
    )
    assert failure is None and row is not None
    assert record["claim_check_issues"] == ["the pot is 9bb -- it is 5.5bb"]
    assert row["validation_status"] == "flagged"

    # Revise: the rewrite ships and the lifecycle is recorded.
    def fake_revise(explanation, facts, *, issues, **kwargs):
        from dataclasses import replace as _r

        return SimpleNamespace(
            changed=True,
            explanation=_r(explanation, answer_explanation="Fixed prose."),
            rejected_reason="",
        )

    monkeypatch.setattr(preflop_reviser, "revise_explanation", fake_revise)
    row2, record2, _ = build_pack_preflop_leg_row(
        src, "BB", "7h6h", solve, revise_pass=True, **common,
    )
    assert record2["revise"]["status"] == "fixed"
    assert "Fixed prose." in row2["Answer Explanation"]


def test_balanced_hand_mix_round_robins_buckets() -> None:
    """The mix visits (hero, depth, strength) buckets round-robin,
    deterministically, and honours the limit."""
    from types import SimpleNamespace

    from pipeline.postflop.play_through import balanced_hand_mix

    def hand(hero, street, combo):
        node = SimpleNamespace(board=("Ks", "7d", "2c"), actor=hero)
        spot = SimpleNamespace(
            node=node, hero_combo=combo,
            hero_cards=[combo[:2], combo[2:]],  # the real spot's property
        )
        leg = SimpleNamespace(street=street, spot=spot)
        return SimpleNamespace(hero=hero, legs=[leg], hero_combo=combo)

    # 4 BTN-river hands then 2 BB-flop hands: an unmixed take-3 would be
    # all BTN river; the mix must include a BB hand.
    hands = [hand("BTN", "river", c) for c in ("AdAc", "QdQc", "JdJc", "TdTc")]
    hands += [hand("BB", "flop", c) for c in ("7h6h", "8h7h")]
    mixed = balanced_hand_mix(hands, 3)
    assert len(mixed) == 3
    assert any(h.hero == "BB" for h in mixed)
    assert balanced_hand_mix(hands, 3) == mixed  # deterministic
    assert balanced_hand_mix(hands, 99) == hands[:99] or len(
        balanced_hand_mix(hands, 99)
    ) == len(hands)


def test_pack_leg_second_rewrite_round(tmp_path, monkeypatch) -> None:
    """Second rewrite round vs final-audit flags (July 2026, strict-clean):
    the pack preflop leg mirrors the postflop layer7 -- a rewrite the final
    audit still flags gets ONE more revise round, and the re-audited clean
    second rewrite ships with an empty final_audit_issues list."""
    from types import SimpleNamespace

    import pipeline.preflop.batch as preflop_batch
    import pipeline.preflop.reviser as preflop_reviser
    from pipeline.postflop.preflop_leg_pack import build_pack_preflop_leg_row

    solve = btn_vs_bb_full_hand_2cJs7s()
    pack = _matching_pack(tmp_path)
    src = find_pack_leg_source(solve, tmp_path, packs=[pack])
    assert src is not None

    fake_issue = SimpleNamespace(claim="the pot is 9bb", problem="it is 5.5bb")

    def fake_claim_check(prose, *a, **k):
        # Flags everything except the SECOND rewrite.
        if prose == "Second fix.":
            return SimpleNamespace(issues=[])
        return SimpleNamespace(issues=[fake_issue])

    monkeypatch.setattr(preflop_batch, "_safe_claim_check", fake_claim_check)

    def fake_generate(facts, options, correct, **kwargs):
        from pipeline.explanation_generator import GeneratedExplanation

        return GeneratedExplanation(
            option_1=options[0], option_2=options[1],
            option_3=options[2] if len(options) > 2 else "",
            option_4=options[3] if len(options) > 3 else "",
            correct_answer=correct, answer_explanation="Original prose.",
        )

    import pipeline.preflop.explanation_generator as preflop_gen

    monkeypatch.setattr(
        preflop_gen, "generate_preflop_answer_explanation", fake_generate,
    )

    rewrites = iter(["Fixed prose.", "Second fix."])

    def fake_revise(explanation, facts, *, issues, **kwargs):
        from dataclasses import replace as _r

        return SimpleNamespace(
            changed=True,
            explanation=_r(explanation, answer_explanation=next(rewrites)),
            rejected_reason="",
        )

    monkeypatch.setattr(preflop_reviser, "revise_explanation", fake_revise)

    row, record, failure = build_pack_preflop_leg_row(
        src, "BB", "7h6h", solve,
        number=1, hand_id="h1", sequence_index=1, sequence_total=3,
        use_placeholder=False, client=object(), model="test-model",
        temperature=0.0, max_tokens=64, answer_style="gto",
        display_in_bb=True, equity_runouts=20,
        revise_pass=True, final_audit=True, second_rewrite=True,
    )
    assert failure is None and row is not None
    rev = record["revise"]
    assert rev["status"] == "fixed"
    assert rev["second_rewrite"]["status"] == "fixed"
    assert rev["second_rewrite"]["issues_before"] == [
        "the pot is 9bb -- it is 5.5bb"
    ]
    assert rev["final_audit_issues"] == []  # re-audited clean
    assert "Second fix." in row["Answer Explanation"]
    # A clean second rewrite means the leg ships unflagged.
    assert row["validation_status"] != "flagged"
