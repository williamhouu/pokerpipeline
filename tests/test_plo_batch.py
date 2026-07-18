"""Tests for pipeline.plo.batch (the PLO batch orchestrator)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.batch import generate_plo_batch  # noqa: E402
from pipeline.plo.format_writer import PLO_CSV_COLUMNS  # noqa: E402
from pipeline.plo.hand_order import HAND_COUNT  # noqa: E402
from pipeline.plo.pack import PloPack  # noqa: E402


def _write_rng(path: Path, p: float) -> None:
    out: list[str] = []
    for _ in range(HAND_COUNT):
        out.append("????")
        out.append(f"{p};1000.0")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _clean_hj_pack(tmp_path: Path) -> PloPack:
    # HJ facing an LJ open: every hand calls 70% (worthy) / 3-bets 30% / folds 0%.
    root = tmp_path / "pack"
    root.mkdir()
    _write_rng(root / "40100.0.rng", 0.0)
    _write_rng(root / "40100.1.rng", 0.7)
    _write_rng(root / "40100.40100.rng", 0.3)
    return PloPack(root=root, label="test")


def test_batch_writes_a_complete_csv(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False
    )
    assert result.questions_written == 1
    assert result.shortfall == 0
    assert out.exists()

    with out.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(PLO_CSV_COLUMNS)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["No"] == "1"
    assert rows[0]["Answer Explanation"] == ""  # Layer 6 not built
    assert rows[0]["Correct Answer"]  # deterministic options are present
    assert "ranges" not in rows[0]


def test_progress_callback_reports_each_question(tmp_path):
    # The admin PLO Generate page drives its live progress bar from this
    # callback (PLO generation runs inline, so this is how "12 / 20" ticks up).
    pack = _clean_hj_pack(tmp_path)
    calls: list[tuple[int, int]] = []
    result = generate_plo_batch(
        pack, output_path=tmp_path / "b.csv", total_questions=4, seed=0,
        compute_equity=False, progress_callback=lambda d, t: calls.append((d, t)),
    )
    assert result.questions_written == 4  # noqa: PLR2004
    # One call per committed question, monotonically increasing, total constant.
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_single_node_contributes_multiple_distinct_hands(tmp_path):
    # A node is no longer capped at one question per batch: when the batch is
    # bigger than the node pool, repeat passes draw NEW hands from the same
    # node (never the same hand twice).
    pack = _clean_hj_pack(tmp_path)  # one node, every hand worthy
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=5, seed=0, compute_equity=False
    )
    assert result.questions_written == 5  # noqa: PLR2004
    assert result.shortfall == 0
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hands = [r["User Cards"] for r in rows]
    assert len(set(hands)) == 5  # noqa: PLR2004  # all different hands


def test_round_robin_spreads_across_nodes_before_repeating(tmp_path):
    # Two nodes, two questions -> one from EACH node (situations spread
    # first), never two hands from one node while the other sits unused.
    root = tmp_path / "pack"
    root.mkdir()
    # Node A: HJ facing the LJ open.
    _write_rng(root / "40100.0.rng", 0.0)
    _write_rng(root / "40100.1.rng", 0.7)
    _write_rng(root / "40100.40100.rng", 0.3)
    # Node B: CO facing the LJ open after HJ folds.
    _write_rng(root / "40100.0.0.rng", 0.0)
    _write_rng(root / "40100.0.1.rng", 0.7)
    _write_rng(root / "40100.0.40100.rng", 0.3)
    pack = PloPack(root=root, label="test")
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=2, seed=0, compute_equity=False
    )
    assert result.questions_written == 2  # noqa: PLR2004
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    from pipeline.provenance import node_reference_from_notes

    node_refs = {
        node_reference_from_notes(r["Notes"]).rsplit("/", 1)[-1] for r in rows
    }
    assert len(node_refs) == 2  # noqa: PLR2004  # one question per node


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [type("C", (), {"text": text})()]
        self.usage = None


class _MockMessages:
    def create(self, **_kw: object) -> _Resp:
        return _Resp('{"answer_explanation": "Call here. It plays well in position."}')


class _MockClient:
    def __init__(self) -> None:
        self.messages = _MockMessages()


def test_batch_difficulty_band_filters_out_of_band_spots(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    # A band above the 3200 ceiling rejects every spot before any LLM call.
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        min_difficulty=3300,
        max_difficulty=3400,
    )
    assert result.questions_written == 0
    assert result.difficulty_filtered_out >= 1


def test_batch_action_context_filter(tmp_path):
    pack = _clean_hj_pack(tmp_path)  # the only node is HJ facing a single raise
    out = tmp_path / "batch.csv"
    # Asking for opens excludes it; the matching context keeps it.
    assert (
        generate_plo_batch(
            pack,
            output_path=out,
            total_questions=1,
            seed=0,
            compute_equity=False,
            action_contexts=["Opening"],
        ).questions_written
        == 0
    )
    assert (
        generate_plo_batch(
            pack,
            output_path=out,
            total_questions=1,
            seed=0,
            compute_equity=False,
            action_contexts=["Facing single raise"],
        ).questions_written
        == 1
    )


def test_batch_ev_gap_gate_filters_coinflips(tmp_path):
    pack = _clean_hj_pack(tmp_path)  # every action has equal EV -> a 0 EV gap
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        min_ev_gap_bb=0.5,
    )
    assert result.questions_written == 0
    assert result.ev_gap_filtered_out >= 1


def test_batch_frequency_window_threads_through(tmp_path):
    pack = _clean_hj_pack(tmp_path)  # the dominant action sits at 70%
    out = tmp_path / "batch.csv"
    # Require >= 95% dominance: the 70% spot no longer clears the window.
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        min_frequency=0.95,
    )
    assert result.questions_written == 0


def test_batch_fills_explanations_with_a_client(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        generate_explanations=True,
        explanation_client=_MockClient(),
    )
    assert result.explanations_written == 1
    assert result.explanations_failed == 0
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Answer Explanation"] == "Call here. It plays well in position."


class _JunkMessages:
    def create(self, **_kw: object) -> _Resp:
        return _Resp("this is not json at all")  # fails validation every attempt


class _JunkClient:
    def __init__(self) -> None:
        self.messages = _JunkMessages()


def test_failed_explanations_drop_the_row_and_trip_the_breaker(tmp_path):
    # A response that never validates DROPS the question (no blank-explanation
    # rows in the CSV), records each reason for the UI, and -- since every
    # attempt fails -- the consecutive-failure circuit breaker aborts the
    # batch instead of burning spend across the whole spot pool.
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        generate_explanations=True,
        explanation_client=_JunkClient(),
    )
    assert result.questions_written == 0
    assert result.explanations_failed == 5  # the _MAX_CONSECUTIVE_FAILURES cap
    assert len(result.explanation_failure_reasons) == 5  # noqa: PLR2004
    assert "ExplanationValidationError" in result.explanation_failure_reasons[0]
    with out.open(encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []  # no blank rows shipped


class _FlakyMessages:
    """Fails the first spot's attempts, then answers cleanly."""

    def __init__(self, failures: int) -> None:
        self._remaining = failures

    def create(self, **_kw: object) -> _Resp:
        if self._remaining > 0:
            self._remaining -= 1
            return _Resp("this is not json at all")
        return _Resp('{"answer_explanation": "Call here. It plays well in position."}')


class _FlakyClient:
    def __init__(self, failures: int) -> None:
        self.messages = _FlakyMessages(failures)


def test_one_failed_explanation_backfills_with_another_spot(tmp_path):
    # One spot fails (both its attempts) and is dropped; the round-robin draws
    # a different hand and still delivers the requested question count.
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        generate_explanations=True,
        explanation_client=_FlakyClient(failures=2),  # one spot = 2 attempts
    )
    assert result.explanations_failed == 1
    assert result.questions_written == 1
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["Answer Explanation"] == "Call here. It plays well in position."


class _CapturingMessages:
    def __init__(self) -> None:
        self.systems: list[str] = []

    def create(self, **kw: object) -> _Resp:
        self.systems.append(str(kw.get("system", "")))
        return _Resp('{"answer_explanation": "Call here. It plays well in position."}')


class _CapturingClient:
    def __init__(self) -> None:
        self.messages = _CapturingMessages()


def test_batch_threads_explanation_system_prompt(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    client = _CapturingClient()
    custom = "CUSTOM PLO PROMPT for this A/B run."
    generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        generate_explanations=True,
        explanation_client=client,
        explanation_system_prompt=custom,
    )
    # The edited prompt reached the LLM call verbatim.
    assert client.messages.systems == [custom]


def test_seed_none_draws_fresh_spots_without_crashing(tmp_path):
    # seed=None seeds the RNG from OS entropy (the Generate page's default,
    # so batches stop repeating the identical spots). Smoke: runs end to end.
    pack = _clean_hj_pack(tmp_path)
    result = generate_plo_batch(
        pack,
        output_path=tmp_path / "batch.csv",
        total_questions=1,
        seed=None,
        compute_equity=False,
    )
    assert result.questions_written == 1
