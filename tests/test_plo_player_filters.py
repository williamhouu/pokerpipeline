"""Tests for the July-16-2026 PLO player-filter fix.

THE BUG (user report, verified on the real "100 percento" batch): with
"Players in the pot: 1 (open), 2 (heads-up)" selected, batches still shipped
caller-heavy squeeze monsters -- LJ opens, four players call, BB squeezes,
everyone folds back to hero. The old filter counted CURRENTLY-LIVE seats
(``plo_active_player_count``), and after the field folds such a node is
"2 live", so a 6-entrant pot passed both the player filter and the
clean-lines cap.

THE FIX: the filters (batch + preview + the Generate page's meta count) use
``plo_pot_entrant_count`` -- everyone who voluntarily put chips in, folded
since or not. Prose facts about who is STILL in the hand keep the live
count on purpose (that is what they describe).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.hand_order import HAND_COUNT  # noqa: E402
from pipeline.plo.node_enumerator import (  # noqa: E402
    PloDecisionNode,
    plo_active_player_count,
    plo_pot_entrant_count,
)
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402

R, C, F = PloActionType.RAISE, PloActionType.CALL, PloActionType.FOLD


def _node(actor: str, history: tuple[PloAction, ...]) -> PloDecisionNode:
    return PloDecisionNode(
        actor=actor, history_before=history, actions=(), history_stem="",
        table_size=9,
    )


def test_collapsed_squeeze_pot_counts_all_entrants():
    """The EXACT shape from the user's batch: LJ opens, HJ/CO/BTN call, SB
    (hero) calls, BB squeezes, everyone folds back to SB. Six players put
    chips in; only two are still live. The filter must see 6."""
    node = _node(
        "SB",
        (
            PloAction("UTG", F), PloAction("UTG+1", F), PloAction("UTG+2", F),
            PloAction("LJ", R), PloAction("HJ", C), PloAction("CO", C),
            PloAction("BTN", C), PloAction("SB", C), PloAction("BB", R),
            PloAction("LJ", F), PloAction("HJ", F), PloAction("CO", F),
            PloAction("BTN", F),
        ),
    )
    assert plo_active_player_count(node) == 2  # live: SB + BB
    assert plo_pot_entrant_count(node) == 6  # LJ HJ CO BTN SB BB


def test_entrants_equal_live_on_clean_lines():
    """On the clean shapes the filter is meant to keep, entrant and live
    counting agree -- so the fix removes junk without touching good spots."""
    open_node = _node("UTG", ())
    assert plo_pot_entrant_count(open_node) == 1
    assert plo_active_player_count(open_node) == 1

    hu_3bet = _node(
        "UTG",
        (
            PloAction("UTG", R), PloAction("UTG+1", F), PloAction("UTG+2", F),
            PloAction("LJ", F), PloAction("HJ", F), PloAction("CO", F),
            PloAction("BTN", F), PloAction("SB", F), PloAction("BB", R),
        ),
    )
    assert plo_pot_entrant_count(hu_3bet) == 2
    assert plo_active_player_count(hu_3bet) == 2

    after_one_call = _node(
        "BTN",
        (
            PloAction("UTG", R), PloAction("UTG+1", F), PloAction("UTG+2", F),
            PloAction("LJ", F), PloAction("HJ", F), PloAction("CO", C),
        ),
    )
    assert plo_pot_entrant_count(after_one_call) == 3
    assert plo_active_player_count(after_one_call) == 3


def test_hero_counts_even_before_acting():
    """A BB defend decision: hero has posted but not ACTED -- still counts."""
    node = _node(
        "BB",
        (
            PloAction("UTG", R), PloAction("UTG+1", F), PloAction("UTG+2", F),
            PloAction("LJ", F), PloAction("HJ", F), PloAction("CO", F),
            PloAction("BTN", F), PloAction("SB", F),
        ),
    )
    assert plo_pot_entrant_count(node) == 2  # UTG + hero BB


def _write_rng(path: Path, p: float) -> None:
    out = []
    for _ in range(HAND_COUNT):
        out.append("x")
        out.append(f"{p};0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_batch_player_filter_excludes_caller_heavy_nodes(tmp_path):
    """generate_plo_batch with player_counts=[1, 2]: a 3-entrant node
    (open + call + hero) must never produce a question; with [3] it must."""
    from pipeline.plo.batch import generate_plo_batch
    from pipeline.plo.pack import discover_plo_pack

    root = tmp_path / "ranges" / "Omaha" / "9-way" / "100bb"
    # UTG opens (2), UTG+1 calls (1) -> UTG+2 decision node "2.1" with a
    # worthy 70/30 fold/call mix. Entrants at that node = 3.
    _write_rng(root / "2.1.0.rng", 0.7)
    _write_rng(root / "2.1.1.rng", 0.3)
    _write_rng(root / "2.1.2.rng", 0.0)
    pack = discover_plo_pack(tmp_path)

    def _run(counts: list[int], tag: str) -> int:
        result = generate_plo_batch(
            pack,
            output_path=tmp_path / f"b_{tag}.csv",
            total_questions=1,
            seed=1,
            player_counts=counts,
            generate_explanations=False,
        )
        return result.questions_written

    assert _run([1, 2], "hu") == 0  # the 3-entrant node is filtered out
    assert _run([3], "3way") == 1  # and reachable when 3-way is asked for