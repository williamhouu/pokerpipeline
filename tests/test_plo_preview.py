"""Tests for admin_panel.plo_preview (read-only PLO pipeline preview)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel.plo_preview import (  # noqa: E402
    build_preview_rows,
    format_cards,
)
from pipeline.plo.hand_order import HAND_COUNT  # noqa: E402
from pipeline.plo.node_enumerator import (  # noqa: E402
    PloDecisionNode,
    enumerate_plo_nodes,
)
from pipeline.plo.pack import PloPack  # noqa: E402


def test_format_cards_uses_suit_emoji():
    assert format_cards(("As", "Ks", "Ah", "Kh")) == "A♠️ K♠️ A❤️ K❤️"


# --- build_preview_rows over a synthetic pack -----------------------------
def _write_rng(path: Path, p: float) -> None:
    """Write a `.rng` where EVERY hand has weight p (and a small ev)."""
    out: list[str] = []
    for _ in range(HAND_COUNT):
        out.append("????")
        out.append(f"{p};1000.0")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _hj_facing_open_pack(
    tmp_path: Path,
) -> tuple[PloPack, tuple[PloDecisionNode, ...]]:
    """HJ facing an LJ open: every hand calls 70% / 3-bets 30% / folds 0%."""
    _write_rng(tmp_path / "40100.0.rng", 0.0)  # HJ fold
    _write_rng(tmp_path / "40100.1.rng", 0.7)  # HJ call
    _write_rng(tmp_path / "40100.40100.rng", 0.3)  # HJ 3-bet
    pack = PloPack(root=tmp_path, label="test")
    return pack, enumerate_plo_nodes(pack)


def test_build_preview_rows_returns_worthy_spot(tmp_path):
    pack, nodes = _hj_facing_open_pack(tmp_path)
    rows = build_preview_rows(pack, nodes, count=1, seed=0, hero_positions=["HJ"])
    assert len(rows) == 1
    r = rows[0]
    assert r.position == "HJ"
    assert 0.55 <= r.dominant_freq <= 0.95  # worthy window
    assert r.correct_answer in r.options
    assert 400 <= r.difficulty <= 3200  # noqa: PLR2004
    assert r.equity is None  # compute_equity defaults off
    assert r.ev_gap_bb is not None  # EV gap is free from the pack
    assert r.cards  # rendered with emoji


def test_preview_action_line_is_real_prose_without_em_dash(tmp_path):
    # The preview now shows the REAL Question prose, not the old compact
    # summary that carried an em dash.
    pack, nodes = _hj_facing_open_pack(tmp_path)
    line = build_preview_rows(pack, nodes, count=1, seed=0)[0].action_line
    assert line.startswith("You're")  # format_plo_action_history shape
    assert "Hijack" in line
    assert "—" not in line  # the em-dash bug is gone
    assert "–" not in line  # en-dash too


def test_preview_surfaces_action_frequencies(tmp_path):
    pack, nodes = _hj_facing_open_pack(tmp_path)
    freqs = dict(build_preview_rows(pack, nodes, count=1, seed=0)[0].action_frequencies)
    assert freqs  # populated
    assert all(f > 0 for f in freqs.values())  # only played actions listed
    assert abs(sum(freqs.values()) - 1.0) < 1e-6  # conditional strategy
    assert max(freqs.items(), key=lambda kv: kv[1])[0] == "Call"  # dominant


def test_preview_action_context_filter(tmp_path):
    # The only node here is HJ facing a single raise.
    pack, nodes = _hj_facing_open_pack(tmp_path)
    assert build_preview_rows(
        pack, nodes, count=1, seed=0, action_contexts=["Facing single raise"]
    )
    assert not build_preview_rows(
        pack, nodes, count=1, seed=0, action_contexts=["Opening"]
    )
