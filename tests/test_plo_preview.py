"""Tests for admin_panel.plo_preview (read-only PLO pipeline preview)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel.plo_preview import (  # noqa: E402
    action_line,
    build_preview_rows,
    format_cards,
)
from pipeline.plo.hand_order import HAND_COUNT  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode, enumerate_plo_nodes  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType, PloPack  # noqa: E402

F = PloActionType.FOLD
C = PloActionType.CALL
R = PloActionType.RAISE


def _act(seat: str, atype: PloActionType, pct: int | None = None) -> PloAction:
    return PloAction(seat=seat, action=atype, raise_pct=pct)


def _node(actor: str, history: tuple[PloAction, ...]) -> PloDecisionNode:
    return PloDecisionNode(actor=actor, history_before=history, actions=(), history_stem="")


def test_format_cards_uses_suit_emoji():
    assert format_cards(("As", "Ks", "Ah", "Kh")) == "A♠️ K♠️ A❤️ K❤️"


def test_action_line_open_node():
    assert action_line(_node("LJ", ())) == "You're first to act in the Lojack."


def test_action_line_drops_folds_and_levels_raises():
    # LJ opens, everyone folds, BB 3-bets, LJ faces it.
    history = (_act("LJ", R, 100), _act("HJ", F), _act("CO", F), _act("BU", F), _act("SB", F), _act("BB", R, 100))
    line = action_line(_node("LJ", history))
    assert line == "The Lojack opens, the Big Blind 3-bets — you're in the Lojack."


# --- build_preview_rows over a synthetic pack -----------------------------
def _write_rng(path: Path, p: float) -> None:
    """Write a `.rng` where EVERY hand has weight p (and a small ev)."""
    out: list[str] = []
    for _ in range(HAND_COUNT):
        out.append("????")
        out.append(f"{p};1000.0")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_build_preview_rows_returns_worthy_spot(tmp_path):
    # HJ facing an open: every hand calls 70% (worthy) / 3-bets 30% / folds 0%.
    _write_rng(tmp_path / "40100.0.rng", 0.0)  # HJ fold
    _write_rng(tmp_path / "40100.1.rng", 0.7)  # HJ call
    _write_rng(tmp_path / "40100.40100.rng", 0.3)  # HJ 3-bet
    pack = PloPack(root=tmp_path, label="test")
    nodes = enumerate_plo_nodes(pack)

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
    assert "Hijack" in r.action_line
