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
    generation; an impossible band writes zero hands."""
    pack = _matching_pack(tmp_path)
    result, meta = _run_batch(
        tmp_path, monkeypatch, pack,
        min_hand_difficulty=3200, max_hand_difficulty=3200,
    )
    assert result.questions_written == 0
    c = meta["counters"]
    assert c["hands_difficulty_filtered"] == c["hands_assembled"] > 0


def test_no_pack_root_keeps_entry_legs(tmp_path, monkeypatch) -> None:
    """preflop_leg_pack_root=None (the default) = the pre-existing
    entry-derived behaviour, byte for byte."""
    result, meta = _run_batch(tmp_path, monkeypatch, None)
    assert result.questions_written > 0
    c = meta["counters"]
    assert c["preflop_leg_pack_used"] == 0
    assert meta["run_settings"]["preflop_leg_pack"] is None
