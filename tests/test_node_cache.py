"""Tests for pipeline.preflop.node_cache.

Covers the descriptor round-trip (must be byte-identical to the parse-walk,
since the admin panel reconstructs full nodes from it), the lightweight
node, and the disk cache's build / read / invalidate / corrupt-fallback
behavior. The cache turns the ~6s 9-max pack walk into a ~90ms descriptor
load on every panel restart, so its correctness matters.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop import node_cache  # noqa: E402
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    ParsedRangeFile,
    PreflopActionType,
)
from pipeline.preflop.node_cache import (  # noqa: E402
    clear_cache,
    descriptor_to_node,
    lightweight_node,
    load_descriptors,
    node_to_descriptor,
)
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopActionOption,
    PreflopDecisionNode,
)
from pipeline.preflop.pack import PreflopPack  # noqa: E402

_FOLD = PreflopActionType.FOLD
_CALL = PreflopActionType.CALL
_RAISE = PreflopActionType.RAISE


# --- fixtures ---------------------------------------------------------------
def _node(
    actor: str,
    history: list[tuple[str, PreflopActionType, float | None]],
    actions: list[tuple[PreflopActionType, float | None, str]],
    pack_id: str = "t",
) -> PreflopDecisionNode:
    """Hand-build a node whose range files' action_history is exactly
    ``history_before + (the actor's own action,)`` -- the same shape the
    real enumerator produces, so descriptor_to_node can reproduce it."""
    hb = tuple(ParsedAction(p, t, s) for p, t, s in history)
    options = []
    for action_type, size, path in actions:
        rf = ParsedRangeFile(
            pack_id=pack_id,
            path=Path(path),
            actor=actor,
            actor_action=action_type,
            actor_raise_size_pct=size,
            action_history=hb + (ParsedAction(actor, action_type, size),),
        )
        options.append(
            PreflopActionOption(
                action_type=action_type, raise_size_pct=size, range_file=rf
            )
        )
    return PreflopDecisionNode(
        pack_id=pack_id, actor=actor, history_before=hb, actions=tuple(options)
    )


def _pack(root: Path, pack_id: str = "testpack") -> PreflopPack:
    return PreflopPack(
        pack_id=pack_id,
        root_path=root,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
        file_glob="*.txt",
    )


def _seed_pack_dir(tmp_path: Path) -> tuple[Path, tuple[PreflopDecisionNode, ...]]:
    """A tmp pack root with two range files (for the file-count rglob) and
    the two nodes a stubbed enumerator should return for them."""
    root = tmp_path / "pack"
    (root / "SB").mkdir(parents=True)
    (root / "SB" / "a.txt").write_text("x", encoding="utf-8")
    (root / "BB").mkdir()
    (root / "BB" / "b.txt").write_text("y", encoding="utf-8")
    nodes = (
        _node("SB", [], [(_RAISE, 60.0, str(root / "SB" / "a.txt"))], "testpack"),
        _node(
            "BB",
            [("SB", _RAISE, 60.0)],
            [(_FOLD, None, str(root / "BB" / "b.txt"))],
            "testpack",
        ),
    )
    return root, nodes


# --- descriptor round-trip --------------------------------------------------
def test_descriptor_round_trip_is_faithful() -> None:
    node = _node(
        "BB",
        [("BTN", _RAISE, 60.0)],
        [(_FOLD, None, "/x/Fold.txt"), (_CALL, None, "/x/Call.txt")],
    )
    assert descriptor_to_node(node_to_descriptor(node), "t") == node


def test_descriptor_round_trip_open_node() -> None:
    node = _node("UTG", [], [(_RAISE, 60.0, "/x/Raise.txt")])
    assert descriptor_to_node(node_to_descriptor(node), "t") == node


def test_descriptors_are_plain_data_and_pickle_small() -> None:
    """Descriptors must be plain strings/floats (no dataclass instances),
    which is what makes them fast to unpickle vs full nodes."""
    node = _node("BB", [("BTN", _RAISE, 60.0)], [(_FOLD, None, "/x/F.txt")])
    descr = node_to_descriptor(node)
    # round-trips through pickle unchanged (no custom classes inside)
    assert pickle.loads(pickle.dumps(descr)) == descr


# --- lightweight node -------------------------------------------------------
def test_lightweight_node_keeps_history_drops_actions() -> None:
    node = _node("BB", [("BTN", _RAISE, 60.0)], [(_FOLD, None, "/x/F.txt")])
    light = lightweight_node(node_to_descriptor(node), "t")
    assert light.actor == "BB"
    assert light.history_before == node.history_before
    assert light.actions == ()
    # node_id (the lookup key downstream) matches the full node's
    assert light.node_id == node.node_id


# --- disk cache: build / read / invalidate ----------------------------------
def test_load_descriptors_builds_then_reads_from_disk(tmp_path, monkeypatch) -> None:
    root, nodes = _seed_pack_dir(tmp_path)
    pack = _pack(root)
    calls = {"n": 0}

    def fake_enum(_packs):
        calls["n"] += 1
        return nodes

    monkeypatch.setattr(node_cache, "enumerate_nodes", fake_enum)
    cache = tmp_path / "cache"

    descrs, file_count = load_descriptors(pack, cache)
    assert calls["n"] == 1  # cold build walked once
    assert file_count == 2  # two .txt files
    assert len(descrs) == 2
    assert (cache / "testpack.nodes.pkl").is_file()
    # full nodes reconstructed from the cache equal the originals
    assert tuple(descriptor_to_node(d, "testpack") for d in descrs) == nodes

    descrs2, fc2 = load_descriptors(pack, cache)
    assert calls["n"] == 1  # warm read did NOT re-walk
    assert descrs2 == descrs
    assert fc2 == 2


def test_signature_change_rebuilds(tmp_path, monkeypatch) -> None:
    root, nodes = _seed_pack_dir(tmp_path)
    pack = _pack(root)
    calls = {"n": 0}
    monkeypatch.setattr(
        node_cache, "enumerate_nodes",
        lambda _p: (calls.__setitem__("n", calls["n"] + 1), nodes)[1],
    )
    cache = tmp_path / "cache"
    load_descriptors(pack, cache)
    assert calls["n"] == 1
    # Add a new immediate child of the pack root -> scandir signature moves.
    (root / "CO").mkdir()
    load_descriptors(pack, cache)
    assert calls["n"] == 2  # rebuilt on signature change


def test_corrupt_cache_falls_back_to_rebuild(tmp_path, monkeypatch) -> None:
    root, nodes = _seed_pack_dir(tmp_path)
    pack = _pack(root)
    calls = {"n": 0}
    monkeypatch.setattr(
        node_cache, "enumerate_nodes",
        lambda _p: (calls.__setitem__("n", calls["n"] + 1), nodes)[1],
    )
    cache = tmp_path / "cache"
    load_descriptors(pack, cache)
    assert calls["n"] == 1
    # Corrupt the pickle -> next load must rebuild, not raise.
    (cache / "testpack.nodes.pkl").write_bytes(b"not a pickle")
    descrs, fc = load_descriptors(pack, cache)
    assert calls["n"] == 2
    assert len(descrs) == 2 and fc == 2


def test_version_mismatch_rebuilds(tmp_path, monkeypatch) -> None:
    root, nodes = _seed_pack_dir(tmp_path)
    pack = _pack(root)
    calls = {"n": 0}
    monkeypatch.setattr(
        node_cache, "enumerate_nodes",
        lambda _p: (calls.__setitem__("n", calls["n"] + 1), nodes)[1],
    )
    cache = tmp_path / "cache"
    load_descriptors(pack, cache)
    assert calls["n"] == 1
    monkeypatch.setattr(node_cache, "CACHE_VERSION", node_cache.CACHE_VERSION + 1)
    load_descriptors(pack, cache)
    assert calls["n"] == 2  # old version discarded


def test_clear_cache_removes(tmp_path, monkeypatch) -> None:
    root, nodes = _seed_pack_dir(tmp_path)
    pack = _pack(root)
    monkeypatch.setattr(node_cache, "enumerate_nodes", lambda _p: nodes)
    cache = tmp_path / "cache"
    load_descriptors(pack, cache)
    assert (cache / "testpack.nodes.pkl").is_file()
    removed = clear_cache(cache, "testpack")
    assert removed == 1
    assert not (cache / "testpack.nodes.pkl").exists()
    # clearing a missing cache dir is a harmless no-op
    assert clear_cache(tmp_path / "nope") == 0
