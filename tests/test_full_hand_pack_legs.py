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
        assert pack.pack_id in r["solver_reference"]
        assert pack.pack_id in r["Notes"]
        assert r["Cards on Table"] == ""
    for r in rows:
        assert r["hand_difficulty"], "every full-hand row carries hand_difficulty"
    # hand_difficulty == max leg difficulty within each hand.
    by_hand: dict[str, list[dict]] = {}
    for r in rows:
        by_hand.setdefault(r["hand_id"], []).append(r)
    for legs in by_hand.values():
        assert {r["hand_difficulty"] for r in legs} == {
            str(max(int(r["Difficulty Rating"]) for r in legs))
        }


def test_coherence_gate_falls_back_to_entry_leg(tmp_path, monkeypatch) -> None:
    """When the pack says the hand mostly 3-BETS, the as-played call would
    contradict the pack's correct answer, so the leg falls back to the
    entry-derived question (which frames the as-played action)."""
    pack = _matching_pack(
        tmp_path, call_freq=0.2, threebet_freq=0.7, pack_id="fixture_3betty",
    )
    result, meta = _run_batch(tmp_path, monkeypatch, pack)
    assert result.questions_written > 0
    c = meta["counters"]
    # Every BB defend leg disagrees (dominant 3-bet) -> entry fallback; BTN
    # opener legs (if any) still agree.
    assert c["preflop_leg_entry_fallback"] > 0


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
    out_all = tmp_path / "all.csv"
    generate_full_hand_batch(
        solve=solve, output_path=out_all, total_hands=50, dry_run=True,
        answer_style="gto", equity_runouts=20, include_villain=True,
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
    assert FULL_HAND_CSV_COLUMNS[-3:] == (
        "hand_id", "sequence_index", "hand_difficulty",
    )

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
