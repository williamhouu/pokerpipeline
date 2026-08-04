"""Tests for pipeline.plo.batch (the PLO batch orchestrator)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

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
        # The shared call seam wraps a string system into a cache-controlled
        # block list (July 2026); record the TEXT so the assertion stays
        # about "the prompt reached the call verbatim".
        sys = kw.get("system", "")
        if isinstance(sys, list):
            sys = "".join(b.get("text", "") for b in sys)
        self.systems.append(str(sys))
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


# --- usage totals ride the result (July 2026, background jobs) ---------------
# PLO generation runs as a subprocess job, so the admin's spend logger can no
# longer accumulate usage via an in-process callback: generate_plo_batch now
# accumulates internally and the totals cross the process boundary on the
# (picklable) PloBatchResult. A caller-supplied callback still sees every event.


class _UsageResp(_Resp):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.usage = type(
            "U",
            (),
            {
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 3,
            },
        )()


class _UsageClient:
    def __init__(self) -> None:
        self.messages = type(
            "M",
            (),
            {
                "create": lambda _self, **_kw: _UsageResp(
                    '{"answer_explanation": "Call here. It plays well in position."}'
                )
            },
        )()


def test_result_carries_usage_totals_and_forwards_callback(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    seen: list[tuple] = []
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=1,
        seed=0,
        compute_equity=False,
        generate_explanations=True,
        explanation_client=_UsageClient(),
        usage_callback=lambda *a: seen.append(a),
    )
    assert result.explanations_written == 1
    # One LLM call -> its usage lands on the result...
    assert result.model_used  # the generation model
    assert result.total_input_tokens == 100
    assert result.total_output_tokens == 40
    assert result.total_cache_creation_tokens == 7
    assert result.total_cache_read_tokens == 3
    # ...AND the caller's callback still saw the same event (unchanged contract).
    assert len(seen) == 1
    assert seen[0][1:] == (100, 40, 7, 3)


def test_result_usage_zero_without_llm(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    result = generate_plo_batch(
        pack,
        output_path=tmp_path / "b.csv",
        total_questions=1,
        seed=0,
        compute_equity=False,
    )
    assert result.model_used == ""
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0


def test_result_pickles_across_a_process_boundary(tmp_path):
    import pickle

    pack = _clean_hj_pack(tmp_path)
    result = generate_plo_batch(
        pack,
        output_path=tmp_path / "b.csv",
        total_questions=1,
        seed=0,
        compute_equity=False,
    )
    clone = pickle.loads(pickle.dumps(result))
    assert clone.questions_written == result.questions_written
    assert clone.output_path == result.output_path


def test_run_plo_generate_job_adapts_the_progress_protocol(tmp_path, monkeypatch):
    """The subprocess wrapper: job-worker (message, current, total) callbacks
    out; generate_plo_batch's (done, total) callback in; result passed back."""
    import pipeline.plo.run as plo_run

    captured: dict = {}

    def _fake_generate(*, progress_callback=None, **kwargs):
        captured["kwargs"] = kwargs
        progress_callback(2, 4)  # the batch's 2-arg convention
        return "RESULT"

    monkeypatch.setattr(plo_run, "generate_plo_batch", _fake_generate)
    events: list[tuple] = []
    out = plo_run.run_plo_generate_job(
        progress_callback=lambda m, c, t: events.append((m, c, t)),
        total_questions=4,
        seed=1,
    )
    assert out == "RESULT"
    # stop_check passes through (None when the parent doesn't support it).
    assert captured["kwargs"] == {
        "total_questions": 4, "seed": 1, "stop_check": None,
    }
    # Initial "loading" tick + the adapted per-question tick, both 3-arg.
    assert events[0] == ("Loading pack + sampling spots…", 0, 4)
    assert events[1] == ("Generated 2 / 4 questions…", 2, 4)


# --- graceful stop + incremental commit (July 2026) --------------------------
# The CSV/meta are (re)written after every committed question, so a crash
# loses at most the in-flight question; stop_check ends the batch cleanly
# between questions ("finish current question, keep everything").


def test_graceful_stop_keeps_committed_questions(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    committed: list[int] = []
    result = generate_plo_batch(
        pack,
        output_path=out,
        total_questions=3,
        seed=0,
        compute_equity=False,
        progress_callback=lambda done, total: committed.append(done),
        # True as soon as one question has committed -> stop before the next.
        stop_check=lambda: bool(committed),
    )
    assert result.stopped_early is True
    assert result.questions_written == 1
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    # A gracefully-stopped batch is COMPLETE (clean early finish)...
    assert meta["complete"] is True
    # ...and says so in the counters.
    assert meta["counters"]["stopped_early"] is True
    assert meta["counters"]["questions_written"] == 1


def test_stop_check_false_is_byte_identical_no_op(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    generate_plo_batch(pack, output_path=a, total_questions=2, seed=0,
                       compute_equity=False)
    generate_plo_batch(pack, output_path=b, total_questions=2, seed=0,
                       compute_equity=False, stop_check=lambda: False)
    assert a.read_bytes() == b.read_bytes()
    meta_a = json.loads(a.with_suffix(".meta.json").read_text())
    meta_b = json.loads(b.with_suffix(".meta.json").read_text())
    assert meta_a["questions"] == meta_b["questions"]
    assert meta_a["complete"] is True and meta_a["counters"]["stopped_early"] is False


class _CrashSecondCallClient:
    """First explanation call succeeds; the second raises hard (a crash the
    batch does NOT absorb) -- simulates dying mid-run."""

    def __init__(self) -> None:
        outer = self

        class _M:
            def create(self, **_kw: object) -> _Resp:
                outer.calls = getattr(outer, "calls", 0) + 1
                if outer.calls > 1:
                    raise RuntimeError("simulated mid-batch crash")
                return _Resp(
                    '{"answer_explanation": "Call here. It plays well in position."}'
                )

        self.messages = _M()


def test_crash_mid_batch_leaves_committed_rows_on_disk(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    with pytest.raises(RuntimeError, match="simulated mid-batch crash"):
        generate_plo_batch(
            pack,
            output_path=out,
            total_questions=2,
            seed=0,
            compute_equity=False,
            generate_explanations=True,
            explanation_client=_CrashSecondCallClient(),
        )
    # The first committed question survived the crash on disk...
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["Answer Explanation"]
    # ...and the meta marks the batch as NOT cleanly finished.
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["complete"] is False
    assert meta["counters"]["questions_written"] == 1


def test_balanced_mode_writes_balance_report_and_stays_deterministic(tmp_path):
    """🎛️ Fully balanced (July 2026): the batch generates, records
    run_settings.balanced, and writes a meta balance_report whose achieved
    counts sum to the questions written. Same seed -> byte-identical output
    (the pre-pass is part of the deterministic draw)."""
    import json

    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "balanced.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=4, seed=0,
        compute_equity=False, balanced=True,
    )
    assert result.questions_written == 4  # noqa: PLR2004
    meta = json.loads(result.meta_path.read_text())
    assert meta["run_settings"]["balanced"] is True
    report = meta["balance_report"]
    assert report["selected"] == 4  # noqa: PLR2004
    assert report["pool"] >= 4  # noqa: PLR2004
    for axis in report["axes"]:
        assert sum(v["achieved"] for v in axis["values"]) == 4  # noqa: PLR2004
    # Determinism: a rerun with the same seed is byte-identical.
    first_csv = out.read_bytes()
    generate_plo_batch(
        pack, output_path=out, total_questions=4, seed=0,
        compute_equity=False, balanced=True,
    )
    assert out.read_bytes() == first_csv


def test_balanced_mode_balances_the_answer_verb(tmp_path):
    """USER RULE: fold / call / raise answers must spread. Two nodes whose
    worthy answers differ (call-dominant vs fold-dominant) -> a balanced
    4-question batch takes 2 of each, never 4 of one."""
    import csv as _csv

    root = tmp_path / "pack2"
    root.mkdir()
    # Node A (HJ vs open): call 70 / 3-bet 30 -> dominant Call.
    _write_rng(root / "40100.0.rng", 0.0)
    _write_rng(root / "40100.1.rng", 0.7)
    _write_rng(root / "40100.40100.rng", 0.3)
    # Node B (CO vs open, next seat on): fold 70 / call 30 -> dominant Fold.
    _write_rng(root / "40100.0.0.rng", 0.7)
    _write_rng(root / "40100.0.1.rng", 0.3)
    _write_rng(root / "40100.0.40100.rng", 0.0)
    pack = PloPack(root=root, label="test2")

    out = tmp_path / "verbs.csv"
    generate_plo_batch(
        pack, output_path=out, total_questions=4, seed=0,
        compute_equity=False, balanced=True,
    )
    with out.open(encoding="utf-8") as handle:
        answers = [r["Correct Answer"] for r in _csv.DictReader(handle)]
    folds = sum("fold" in a.lower() for a in answers)
    calls = sum("call" in a.lower() for a in answers)
    assert folds == 2 and calls == 2, answers


# --- ⚡ parallel LLM workers (July 2026, user ask) -----------------------------
# llm_workers > 1 runs each question's LLM chain on a worker thread while ALL
# deterministic work (facts RNG, gates, row building, incremental commit)
# stays on the main thread in draw order, with strictly in-order commits --
# so the CSV must come out IDENTICAL to a sequential run, and the usage
# totals must stay exact under concurrency (THE USAGE RULE).

def test_parallel_workers_match_sequential_output(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out_seq = tmp_path / "seq.csv"
    out_par = tmp_path / "par.csv"
    r1 = generate_plo_batch(
        pack, output_path=out_seq, total_questions=4, seed=7,
        compute_equity=False, generate_explanations=True,
        explanation_client=_MockClient(), llm_workers=1,
    )
    r3 = generate_plo_batch(
        pack, output_path=out_par, total_questions=4, seed=7,
        compute_equity=False, generate_explanations=True,
        explanation_client=_MockClient(), llm_workers=3,
    )
    assert r1.questions_written == r3.questions_written == 4  # noqa: PLR2004
    assert out_seq.read_text(encoding="utf-8") == out_par.read_text(
        encoding="utf-8"
    )
    # Meta question records match too (order + content), bar nothing.
    meta_seq = json.loads(r1.meta_path.read_text(encoding="utf-8"))
    meta_par = json.loads(r3.meta_path.read_text(encoding="utf-8"))
    assert meta_seq["questions"] == meta_par["questions"]
    assert meta_par["run_settings"]["llm_workers"] == 3  # noqa: PLR2004


def test_parallel_usage_totals_stay_exact(tmp_path):
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=4, seed=7,
        compute_equity=False, generate_explanations=True,
        explanation_client=_UsageClient(), llm_workers=4,
    )
    n = result.explanations_written
    assert n == 4  # noqa: PLR2004
    # One generation call per question with the fixed 100/40/7/3 usage.
    assert result.total_input_tokens == 100 * n
    assert result.total_output_tokens == 40 * n
    assert result.total_cache_creation_tokens == 7 * n
    assert result.total_cache_read_tokens == 3 * n


def test_parallel_graceful_stop_keeps_in_flight_work(tmp_path):
    """A graceful stop with workers in flight finishes and KEEPS every chain
    already submitted (the plural of 'finish the current question')."""
    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    committed: list[int] = []

    def stop_after_first() -> bool:
        return bool(committed)

    result = generate_plo_batch(
        pack, output_path=out, total_questions=6, seed=7,
        compute_equity=False, generate_explanations=True,
        explanation_client=_MockClient(), llm_workers=3,
        stop_check=stop_after_first,
        progress_callback=lambda done, total: committed.append(done),
    )
    assert result.stopped_early is True
    # At least the first commit, plus whatever was in flight when the stop
    # landed -- never more than requested.
    assert 1 <= result.questions_written <= 6  # noqa: PLR2004
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["complete"] is True
    assert len(meta["questions"]) == result.questions_written
