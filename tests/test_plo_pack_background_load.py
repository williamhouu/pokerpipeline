"""Tests for the July-2026 PLO Generate perf fix: the background pack
loader (page renders while the 15-25s node walk runs in a thread) and the
precomputed filter meta (the live node recount reads plain tuples instead
of re-walking 160k histories on every Streamlit rerun).

Browserless by design (the fix-durability rule): the loader state machine
and the meta equivalence are pure logic, tested with a monkeypatched
loader -- no Streamlit runtime, no pack on disk.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel import plo_preview  # noqa: E402
from pipeline.plo.node_enumerator import (  # noqa: E402
    PloDecisionNode,
    plo_filter_meta,
    plo_node_action_context,
    plo_pot_entrant_count,
)
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402

R, C, F = PloActionType.RAISE, PloActionType.CALL, PloActionType.FOLD


def _node(actor: str, history: tuple[PloAction, ...]) -> PloDecisionNode:
    return PloDecisionNode(
        actor=actor, history_before=history, actions=(), history_stem="",
        table_size=9,
    )


def _nodes() -> tuple[PloDecisionNode, ...]:
    return (
        _node("UTG", ()),  # Opening, 1 player
        _node("HJ", (PloAction("UTG", F), PloAction("UTG+1", R),
                     PloAction("UTG+2", C), PloAction("LJ", F))),
        _node("BB", (PloAction("UTG", R), PloAction("UTG+1", R),
                     PloAction("UTG+2", F), PloAction("LJ", F),
                     PloAction("HJ", F), PloAction("CO", F),
                     PloAction("BTN", F), PloAction("SB", F))),  # Facing 3-bet
    )


def test_filter_meta_matches_per_node_derivation():
    """INVARIANT: a filter over the meta MUST equal a filter over the nodes
    (the Generate page's count can never drift from what generation sees)."""
    nodes = _nodes()
    meta = plo_filter_meta(nodes)
    assert len(meta) == len(nodes)
    for node, (actor, context, players) in zip(nodes, meta, strict=True):
        assert actor == node.actor
        assert context == plo_node_action_context(node)
        # Entrant counting -- the July 16 filter semantics; the page count
        # and generate_plo_batch must read the SAME function.
        assert players == plo_pot_entrant_count(node)


def test_filter_meta_count_equals_node_count_for_any_filter():
    nodes = _nodes()
    meta = plo_filter_meta(nodes)
    for ctx_filter in (None, {"Opening"}, {"Facing 3-bet", "After one call"}):
        from_nodes = sum(
            1 for n in nodes
            if ctx_filter is None or plo_node_action_context(n) in ctx_filter
        )
        from_meta = sum(
            1 for _a, c, _p in meta if ctx_filter is None or c in ctx_filter
        )
        assert from_nodes == from_meta


@pytest.fixture()
def _clean_loader_state():
    """Isolate the module-level background-load registries per test."""
    with plo_preview._BG_LOCK:
        plo_preview._BG_RESULTS.clear()
        plo_preview._BG_THREADS.clear()
    yield
    with plo_preview._BG_LOCK:
        plo_preview._BG_RESULTS.clear()
        plo_preview._BG_THREADS.clear()


def test_request_pack_load_is_nonblocking_then_ready(
    monkeypatch, _clean_loader_state
):
    release = threading.Event()
    pack = object()

    def _slow_load(pack_dir):
        release.wait(timeout=5)
        return pack, _nodes()

    monkeypatch.setattr(plo_preview, "load_pack_and_nodes", _slow_load)

    t0 = time.perf_counter()
    assert plo_preview.request_pack_load("fake_pack") is None  # starts thread
    assert time.perf_counter() - t0 < 1.0  # never blocked on the walk
    assert plo_preview.request_pack_load("fake_pack") is None  # still loading

    release.set()
    for _ in range(100):
        got = plo_preview.request_pack_load("fake_pack")
        if got is not None:
            break
        time.sleep(0.02)
    assert got is not None
    got_pack, got_nodes, got_meta = got
    assert got_pack is pack
    assert got_meta == plo_filter_meta(got_nodes)


def test_request_pack_load_starts_only_one_thread(
    monkeypatch, _clean_loader_state
):
    release = threading.Event()
    starts = []

    def _slow_load(pack_dir):
        starts.append(pack_dir)
        release.wait(timeout=5)
        return object(), ()

    monkeypatch.setattr(plo_preview, "load_pack_and_nodes", _slow_load)
    for _ in range(5):
        assert plo_preview.request_pack_load("fake_pack") is None
    release.set()
    for _ in range(100):
        if plo_preview.request_pack_load("fake_pack") is not None:
            break
        time.sleep(0.02)
    assert len(starts) == 1


def test_request_pack_load_failure_raises_once_then_retries(
    monkeypatch, _clean_loader_state
):
    calls = []

    def _failing_load(pack_dir):
        calls.append(pack_dir)
        msg = "pack exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(plo_preview, "load_pack_and_nodes", _failing_load)
    assert plo_preview.request_pack_load("fake_pack") is None
    excs = 0
    for _ in range(100):
        try:
            if plo_preview.request_pack_load("fake_pack") is None:
                time.sleep(0.02)
                continue
        except RuntimeError:
            excs += 1
            break
    assert excs == 1
    # The failed slot was cleared: the next request starts a fresh attempt.
    assert plo_preview.request_pack_load("fake_pack") is None
    time.sleep(0.2)
    assert len(calls) >= 2
