"""Tests for admin_panel.compare -- the Compare page's pure logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from admin_panel import compare  # noqa: E402


def _row(node: str, hand: str, expl: str) -> dict[str, str]:
    return {
        "solver_reference": f"pack/CO/{node}",
        "User Cards": hand,  # spot join keys on the hole cards (June 2026)
        "Answer Explanation": expl,
    }


def test_spot_key_uses_node_id_and_hand() -> None:
    assert compare.spot_key("pack/CO/NODE_A", "AKs") == "NODE_A|AKs"
    assert compare.spot_key("", "AKs") == "|AKs"


def test_join_by_spot_pairs_matching_and_keeps_a_order() -> None:
    rows_a = [_row("N1", "AKs", "A1"), _row("N2", "QQ", "A2")]
    rows_b = [_row("N2", "QQ", "B2"), _row("N1", "AKs", "B1")]  # different order
    paired = compare.join_by_spot(rows_a, rows_b)
    assert [k for k, _, _ in paired] == ["N1|AKs", "N2|QQ"]  # A's order
    assert paired[0][1]["Answer Explanation"] == "A1"
    assert paired[0][2]["Answer Explanation"] == "B1"


def test_join_by_spot_skips_spots_missing_on_one_side() -> None:
    rows_a = [_row("N1", "AKs", "A1"), _row("N2", "QQ", "A2")]
    rows_b = [_row("N1", "AKs", "B1")]  # B is missing N2/QQ (e.g. it failed)
    paired = compare.join_by_spot(rows_a, rows_b)
    assert len(paired) == 1
    assert paired[0][0] == "N1|AKs"


def test_verdict_round_trip_and_path(tmp_path: Path) -> None:
    csv = tmp_path / "compare_A.csv"
    assert compare.verdicts_path(csv) == tmp_path / "compare_A.verdicts.json"
    compare.save_verdict(csv, "N1|AKs", "A")
    compare.save_verdict(csv, "N2|QQ", "tie")
    assert compare.load_verdicts(csv) == {"N1|AKs": "A", "N2|QQ": "tie"}


def test_save_verdict_rejects_bad_winner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="winner must be one of"):
        compare.save_verdict(tmp_path / "c.csv", "k", "maybe")


def test_load_verdicts_missing_or_malformed_is_empty(tmp_path: Path) -> None:
    assert compare.load_verdicts(tmp_path / "nope.csv") == {}
    bad = tmp_path / "c.csv"
    compare.verdicts_path(bad).write_text("{not json", encoding="utf-8")
    assert compare.load_verdicts(bad) == {}


def test_tally_counts_each_verdict() -> None:
    verdicts = {"k1": "A", "k2": "A", "k3": "B", "k4": "tie", "k5": "bogus"}
    assert compare.tally(verdicts) == {"A": 2, "B": 1, "tie": 1}
