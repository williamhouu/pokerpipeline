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
