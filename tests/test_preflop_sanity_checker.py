"""Tests for the Layer-7 sanity audit (July 2026).

The one LLM pass allowed to use its own poker knowledge, aimed at the
SOLVER DATA facts. Covers parsing (fail-open), the call plumbing with a
mock client, and the batch wiring (flag-only: records + counters + the
flagged status, never a reject).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from pipeline.preflop.batch import generate_preflop_batch  # noqa: E402
from pipeline.preflop.pack import (  # noqa: E402
    PreflopPack,
    clear_registry,
    register_pack,
)
from pipeline.preflop.sanity_checker import (  # noqa: E402
    check_solver_data_sanity,
    parse_sanity_response,
)
from pipeline.preflop_ranges import canonical_169_hand_classes  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _mock_client(responses: list[str]) -> SimpleNamespace:
    queue = list(responses)
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=queue.pop(0))])

    return SimpleNamespace(messages=SimpleNamespace(create=create), _calls=calls)


# --- parsing (fail-open) -------------------------------------------------------
def test_parse_flags_issues() -> None:
    r = parse_sanity_response(
        '{"issues": [{"fact": "hero_position", "problem": "BvB SB is OOP"}]}'
    )
    assert not r.passed
    assert r.issues[0].fact == "hero_position"


def test_parse_clean_pass() -> None:
    assert parse_sanity_response('{"issues": []}').passed


def test_parse_tolerates_code_fence() -> None:
    r = parse_sanity_response('```json\n{"issues": []}\n```')
    assert r.passed


def test_parse_fails_open_on_garbage() -> None:
    """A malfunctioning checker must NEVER flag rows: garbage output is a
    clean pass with the raw kept for debugging."""
    r = parse_sanity_response("I am not JSON")
    assert r.passed and r.issues == ()
    assert r.raw == "I am not JSON"


# --- the call ------------------------------------------------------------------
def test_check_sends_only_the_data_block() -> None:
    client = _mock_client(['{"issues": []}'])
    data = {"hero_position": "SB", "relative_position": "In Position"}
    result = check_solver_data_sanity(data, client)
    assert result.passed
    sent = client._calls[0]["messages"][0]["content"]
    assert "In Position" in sent
    # The sanity audit judges FACTS, never prose: no explanation is sent.
    assert "ANSWER EXPLANATION" not in sent


# --- batch wiring ----------------------------------------------------------------
def _open_pack(tmp_path: Path) -> PreflopPack:
    pack_root = tmp_path / "pack"
    utg = pack_root / "UTG"
    classes = canonical_169_hand_classes()
    raise_weights = {c: 0.0 for c in classes}
    raise_weights.update({"AA": 1.0, "KK": 1.0, "A5s": 0.6, "77": 0.7})
    fold_weights = {c: 1.0 - raise_weights[c] for c in classes}
    utg.mkdir(parents=True)
    line = ",".join(f"{c}:{raise_weights[c]}" for c in classes)
    (utg / "UTG_60%.txt").write_text(line)
    line = ",".join(f"{c}:{fold_weights[c]}" for c in classes)
    (utg / "UTG_Fold.txt").write_text(line)
    pack = PreflopPack(
        pack_id="sanity_pack", root_path=pack_root, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
        description="sanity fixture",
    )
    register_pack(pack)
    return pack


_GEN = (
    '{"option_1": "Fold", "option_2": "Raise 60%", "option_3": "", '
    '"option_4": "", "correct_answer": "Raise 60%", '
    '"answer_explanation": "Open this hand for value."}'
)
_SANITY_FLAG = (
    '{"issues": [{"fact": "relative_position", '
    '"problem": "test contradiction"}]}'
)


def test_batch_sanity_audit_flags_rows_without_rejecting(tmp_path: Path) -> None:
    """Flag-only contract: flagged rows STILL ship, marked flagged, with the
    issues in the meta record and the counter in meta.counters. A flag
    requires TWO passes to agree (July 2026), so a flagged spot costs one
    generation call + two sanity calls."""
    pack = _open_pack(tmp_path)
    out = tmp_path / "out.csv"
    client = _mock_client(
        [_GEN, _SANITY_FLAG, _SANITY_FLAG, _GEN, _SANITY_FLAG, _SANITY_FLAG]
    )
    result = generate_preflop_batch(
        pack=pack, output_path=out, total_questions=10,
        run_sanity_audit=True, client=client, dry_run=False, random_seed=1,
    )
    assert result.questions_written == 2       # nothing rejected
    assert len(client._calls) == 6             # 2 gen + 2x2 sanity
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["run_settings"]["run_sanity_audit"] is True
    assert meta["counters"]["sanity_flagged_rows"] == 2
    issues = [q.get("sanity_check_issues") for q in meta["questions"]]
    assert all(i and "test contradiction" in i[0] for i in issues)
    with open(out, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert all(r["validation_status"] == "flagged" for r in rows)


def test_batch_sanity_audit_needs_two_pass_agreement(tmp_path: Path) -> None:
    """A flag the second pass does NOT repeat is dropped (the v1
    calibration failure mode: one-off hallucinated flags)."""
    pack = _open_pack(tmp_path)
    out = tmp_path / "agree.csv"
    clean = '{"issues": []}'
    other = ('{"issues": [{"fact": "different_field", '
             '"problem": "something else"}]}')
    client = _mock_client(
        # Spot 1: flag then clean -> dropped. Spot 2: flag then a
        # DIFFERENT fact -> also dropped (no consensus on the same fact).
        [_GEN, _SANITY_FLAG, clean, _GEN, _SANITY_FLAG, other]
    )
    result = generate_preflop_batch(
        pack=pack, output_path=out, total_questions=10,
        run_sanity_audit=True, client=client, dry_run=False, random_seed=1,
    )
    assert result.questions_written == 2
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["counters"]["sanity_flagged_rows"] == 0
    assert all("sanity_check_issues" not in q for q in meta["questions"])


def test_batch_sanity_audit_clean_rows_stay_unflagged(tmp_path: Path) -> None:
    pack = _open_pack(tmp_path)
    out = tmp_path / "clean.csv"
    clean = '{"issues": []}'
    client = _mock_client([_GEN, clean, _GEN, clean])
    result = generate_preflop_batch(
        pack=pack, output_path=out, total_questions=10,
        run_sanity_audit=True, client=client, dry_run=False, random_seed=1,
    )
    assert result.questions_written == 2
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["counters"]["sanity_flagged_rows"] == 0
    assert all("sanity_check_issues" not in q for q in meta["questions"])


def test_batch_sanity_audit_fails_open(tmp_path: Path) -> None:
    """A checker call that returns garbage (or errors) never flags and
    never drops the row."""
    pack = _open_pack(tmp_path)
    out = tmp_path / "open.csv"
    client = _mock_client([_GEN, "garbage not json", _GEN, "also garbage"])
    result = generate_preflop_batch(
        pack=pack, output_path=out, total_questions=10,
        run_sanity_audit=True, client=client, dry_run=False, random_seed=1,
    )
    assert result.questions_written == 2
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["counters"]["sanity_flagged_rows"] == 0


def test_batch_sanity_audit_off_makes_no_extra_calls(tmp_path: Path) -> None:
    pack = _open_pack(tmp_path)
    client = _mock_client([_GEN, _GEN])
    result = generate_preflop_batch(
        pack=pack, output_path=tmp_path / "off.csv", total_questions=10,
        run_sanity_audit=False, client=client, dry_run=False, random_seed=1,
    )
    assert result.questions_written == 2
    assert len(client._calls) == 2  # generation only


# --- consensus matching (two-pass agreement) ----------------------------------
def test_consensus_keeps_same_fact_despite_different_wording() -> None:
    from pipeline.preflop.sanity_checker import (
        SanityCheckResult,
        SanityIssue,
        consensus_issues,
    )

    a = SanityCheckResult(passed=False, issues=(
        SanityIssue("hero_position: In Position", "wrong for BvB"),
    ))
    b = SanityCheckResult(passed=False, issues=(
        SanityIssue("HERO_POSITION", "seat order says otherwise"),
    ))
    kept = consensus_issues(a, b)
    assert len(kept) == 1
    assert kept[0].problem == "wrong for BvB"  # first pass's wording ships


def test_consensus_drops_disagreements() -> None:
    from pipeline.preflop.sanity_checker import (
        SanityCheckResult,
        SanityIssue,
        consensus_issues,
    )

    a = SanityCheckResult(passed=False, issues=(
        SanityIssue("break_even_equity_to_call: 0.4", "looks high"),
    ))
    b = SanityCheckResult(passed=True, issues=())
    assert consensus_issues(a, b) == ()
