"""Tests for pipeline.preflop.multi_depth (merged all-depths batches, Aug 2026).

Builds TWO tiny synthetic preflop packs at different stack depths in
tmp_path, runs :func:`generate_all_depths_batch` in dry-run mode, and
asserts on the merged CSV + meta.json + BatchResult contract, the
failed-depth continuation path, and the audit script's multi-pack
re-verification (0 problems on a freshly generated dry merged batch).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop import multi_depth as multi_depth_module  # noqa: E402
from pipeline.preflop.batch import BatchResult  # noqa: E402
from pipeline.preflop.format_writer import PREFLOP_CSV_COLUMNS  # noqa: E402
from pipeline.preflop.multi_depth import generate_all_depths_batch  # noqa: E402
from pipeline.preflop.pack import (  # noqa: E402
    PreflopPack,
    clear_registry,
    register_pack,
)
from pipeline.preflop_ranges import canonical_169_hand_classes  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """The pack registry is module-global; clean before + after every test."""
    clear_registry()
    yield
    clear_registry()


# --- fixture builders (mirror tests/test_preflop_batch.py) ------------------
def _write_full_range(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = ",".join(
        f"{cls}:{weights.get(cls, 0.0)}" for cls in canonical_169_hand_classes()
    )
    path.write_text(line)


def _build_depth_pack(
    tmp_path: Path, *, pack_id: str, stack_depth_bb: float
) -> PreflopPack:
    """One UTG open node (Fold + Raise 60%) with two mixed worthy hands."""
    pack_root = tmp_path / pack_id
    utg = pack_root / "UTG"

    classes = canonical_169_hand_classes()
    pure_open = {"AA", "KK", "QQ", "AKs", "AKo", "AQs", "AQo"}
    raise_weights = {c: (1.0 if c in pure_open else 0.0) for c in classes}
    fold_weights = {c: (1.0 - raise_weights[c]) for c in classes}
    raise_weights["A5s"] = 0.60
    fold_weights["A5s"] = 0.40
    raise_weights["77"] = 0.70
    fold_weights["77"] = 0.30

    _write_full_range(utg / "UTG_60%.txt", raise_weights)
    _write_full_range(utg / "UTG_Fold.txt", fold_weights)

    pack = PreflopPack(
        pack_id=pack_id,
        root_path=pack_root,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=stack_depth_bb,
        open_size_bb=2.5,
        description=f"test depth pack {stack_depth_bb:g}bb",
    )
    register_pack(pack)
    return pack


def _two_packs(tmp_path: Path) -> tuple[PreflopPack, PreflopPack]:
    shallow = _build_depth_pack(tmp_path, pack_id="test_mtt_10bb", stack_depth_bb=10)
    deep = _build_depth_pack(tmp_path, pack_id="test_mtt_15bb", stack_depth_bb=15)
    return shallow, deep


# --- merged CSV + meta ------------------------------------------------------
def test_merged_csv_schema_renumbering_and_depth_order(tmp_path: Path) -> None:
    shallow, deep = _two_packs(tmp_path)
    out = tmp_path / "merged.csv"
    progress: list[tuple[str, int, int]] = []

    result = generate_all_depths_batch(
        # Deliberately out of order: the run must sort ascending by depth.
        packs=[deep, shallow],
        questions_per_depth=2,
        output_path=out,
        dry_run=True,
        random_seed=42,
        progress_callback=lambda m, c, t: progress.append((m, c, t)),
    )

    assert isinstance(result, BatchResult)
    assert result.output_path == out
    assert result.questions_written == 4
    assert result.requested_questions == 4
    assert result.meta_path == out.with_suffix(".meta.json")

    with open(out, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PREFLOP_CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 4
    # "No" renumbered sequentially across the whole merged batch.
    assert [r["No"] for r in rows] == ["1", "2", "3", "4"]
    # Ascending depth order: the 10bb pack's rows first (Notes carries the
    # source pack id in its Chart: field).
    assert all(shallow.pack_id in r["Notes"] for r in rows[:2])
    assert all(deep.pack_id in r["Notes"] for r in rows[2:])

    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    rs = meta["run_settings"]
    assert rs["all_depths"] is True
    assert rs["pack_ids"] == [shallow.pack_id, deep.pack_id]
    assert rs["questions_per_depth"] == 2
    # Graceful multi-pack top-level pack_id for the Review caption.
    assert meta["pack_id"] == "2 tournament depths"
    # Per-question stamping + row alignment.
    assert len(meta["questions"]) == 4
    assert [q["pack_id"] for q in meta["questions"]] == [
        shallow.pack_id, shallow.pack_id, deep.pack_id, deep.pack_id,
    ]
    assert all(q["table_size"] == 6 for q in meta["questions"])
    # Counters: summed totals + a per-depth breakdown.
    counters = meta["counters"]
    assert counters["questions_written"] == 4
    assert set(counters["per_depth"]) == {shallow.pack_id, deep.pack_id}
    assert counters["per_depth"][shallow.pack_id]["questions_written"] == 2

    # Temp per-depth files were cleaned up after the merge.
    assert not (out.parent / multi_depth_module._TMP_DIR_NAME).exists()
    leftovers = [p for p in out.parent.glob("*.csv") if p != out]
    assert leftovers == []

    # Progress spans the whole run with depth-tagged messages.
    assert any(m.startswith("Depth 1/2 (10bb):") for m, _c, _t in progress)
    assert any(m.startswith("Depth 2/2 (15bb):") for m, _c, _t in progress)
    assert all(t == 4 for _m, _c, t in progress)


def test_failed_depth_recorded_and_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shallow, deep = _two_packs(tmp_path)
    out = tmp_path / "merged.csv"

    real = multi_depth_module.generate_preflop_batch

    def _flaky(*, pack: PreflopPack, **kwargs: Any) -> Any:
        if pack.pack_id == shallow.pack_id:
            raise RuntimeError("synthetic depth failure")
        return real(pack=pack, **kwargs)

    monkeypatch.setattr(multi_depth_module, "generate_preflop_batch", _flaky)

    result = generate_all_depths_batch(
        packs=[shallow, deep],
        questions_per_depth=2,
        output_path=out,
        dry_run=True,
        random_seed=42,
    )

    # The surviving depth still ships.
    assert result.questions_written == 2
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["run_settings"]["pack_ids"] == [deep.pack_id]
    assert meta["failed_depths"] == [
        {
            "pack_id": shallow.pack_id,
            "error": "RuntimeError: synthetic depth failure",
        }
    ]
    assert all(q["pack_id"] == deep.pack_id for q in meta["questions"])


def test_every_depth_failing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shallow, deep = _two_packs(tmp_path)

    def _always_fail(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        multi_depth_module, "generate_preflop_batch", _always_fail
    )
    with pytest.raises(RuntimeError, match="every depth failed"):
        generate_all_depths_batch(
            packs=[shallow, deep],
            questions_per_depth=2,
            output_path=tmp_path / "merged.csv",
            dry_run=True,
        )


def test_reserved_kwargs_rejected(tmp_path: Path) -> None:
    shallow, _deep = _two_packs(tmp_path)
    with pytest.raises(ValueError, match="pack"):
        generate_all_depths_batch(
            packs=[shallow],
            questions_per_depth=1,
            output_path=tmp_path / "x.csv",
            pack=shallow,  # owned by the orchestrator
        )


# --- audit script: multi-pack re-verification -------------------------------
def _load_audit_module() -> Any:
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "audit_preflop_batch.py"
    )
    spec = importlib.util.spec_from_file_location("audit_preflop_batch", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_audit_reverifies_merged_multi_pack_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-verifier resolves each row's OWN pack and rebuilds 0/0."""
    shallow, deep = _two_packs(tmp_path)
    out = tmp_path / "merged.csv"
    generate_all_depths_batch(
        packs=[shallow, deep],
        questions_per_depth=2,
        output_path=out,
        dry_run=True,
        random_seed=7,
    )

    audit = _load_audit_module()
    # The fixture packs are already in the live registry; neutralize the
    # audit's own clear + repo-ranges discovery so get_pack resolves them.
    monkeypatch.setattr(audit, "clear_registry", lambda: None)
    monkeypatch.setattr(audit, "discover_packs", lambda _root: ())
    assert audit.audit_batch(out) == 0


# --- Review join disambiguation ---------------------------------------------
def test_meta_question_for_prefers_matching_pack_id() -> None:
    """Two depths can share a bare node_id + hero cards; the stamped
    pack_id must disambiguate (it prefixes the row's node reference)."""
    from admin_panel.review import meta_question_for

    q10 = {"node_id": "0.0", "user_cards": "A-spades, 5-spades",
           "pack_id": "test_mtt_10bb", "tag": "shallow"}
    q15 = {"node_id": "0.0", "user_cards": "A-spades, 5-spades",
           "pack_id": "test_mtt_15bb", "tag": "deep"}
    meta = {"questions": [q10, q15]}
    got = meta_question_for(
        meta,
        user_cards="A-spades, 5-spades",
        node_reference="test_mtt_15bb/UTG/0.0",
    )
    assert got is not None and got["tag"] == "deep"
    # Records WITHOUT a pack_id stamp (single-pack batches) keep the old
    # first-match behavior.
    meta_old = {"questions": [dict(q10, pack_id=""), q15]}
    got_old = meta_question_for(
        meta_old,
        user_cards="A-spades, 5-spades",
        node_reference="test_mtt_15bb/UTG/0.0",
    )
    assert got_old is not None and got_old["tag"] == "shallow"
