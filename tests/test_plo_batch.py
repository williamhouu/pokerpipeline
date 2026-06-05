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


def test_batch_reports_shortfall(tmp_path):
    pack = _clean_hj_pack(tmp_path)  # only one node -> one question max
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=5, seed=0, compute_equity=False
    )
    assert result.questions_written == 1
    assert result.shortfall == 4  # noqa: PLR2004


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
