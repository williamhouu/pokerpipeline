"""Disk-persistent cache of enumerated preflop nodes.

Walking the 9-max Monker pack -- 93,235 range files grouped into 44,058
``PreflopDecisionNode`` objects -- takes ~6 seconds, almost all of it
parsing every filename through the pack's grammar. The result is
*deterministic* for a given pack on disk, so re-deriving it on every fresh
process (and the admin panel restarts often) was the multi-second stall the
first time a session opened the Generate / Compare / Ranges page. It is
I/O-bound, which is why the panel sat near 0% CPU during the hang.

The naive fix -- pickling the full node objects -- does NOT help: the
pickle is 57 MB and unpickling 44k richly-nested frozen dataclasses is as
slow as re-parsing (materialising millions of objects is the real cost,
whichever way you do it). So this module caches a **compact descriptor**
per node instead: plain strings / floats only, no dataclass instances and
no ``Path`` objects. That blob is ~18 MB and loads in ~90ms. From the
descriptors the caller can:

  * compute lightweight metadata (context, player count, node id) without
    building full nodes -- this is what the list / filter / count views
    need, and it stays sub-second (:func:`lightweight_node`); or
  * reconstruct the FULL, byte-identical node when one is actually needed
    -- the Range viewer and the prompt-preview sampler
    (:func:`descriptor_to_node`). Reconstructing all 44k is ~2s, still ~3x
    faster than the parse-walk, and reconstructing a single node is
    instant.

Invalidation is automatic and cheap: the signature is the pack's immediate
child entries plus their mtimes (one ``scandir``), so re-extracting a pack
-- which replaces those entries -- changes the signature and forces a
rebuild. An in-place edit to an existing range file that neither adds nor
removes a directory entry will NOT be detected (dir mtimes don't move on
content edits); packs are extracted wholesale and never edited in place,
but :func:`clear_cache` is the escape hatch. Bumping :data:`CACHE_VERSION`
invalidates every cache (use it when the descriptor shape changes).

Reads are defensive: any unpickling error, version mismatch, or signature
mismatch falls back to a fresh walk + rewrite, so a corrupt or stale cache
degrades to "slow once", never to a crash.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from pipeline.preflop.grammars.types import (
    ParsedAction,
    ParsedRangeFile,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import (
    PreflopActionOption,
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.pack import PreflopPack

logger = logging.getLogger(__name__)

# Bump whenever the descriptor shape below changes, so old pickles are
# discarded rather than misread.
CACHE_VERSION = 2

# One node, as plain pickle-cheap data:
#   (actor, history, actions)
#   history : tuple of (position, action_type_value, raise_size_pct)
#   actions : tuple of (action_type_value, raise_size_pct, range_file_path)
NodeDescriptor = tuple[
    str,
    tuple[tuple[str, str, float | None], ...],
    tuple[tuple[str, float | None, str], ...],
]


@dataclass(frozen=True)
class CachedEnumeration:
    """One pack's node descriptors plus the metadata that validates them."""

    version: int
    pack_id: str
    file_glob: str
    signature: str
    file_count: int
    descriptors: tuple[NodeDescriptor, ...]


# --- descriptor <-> node ----------------------------------------------------
def node_to_descriptor(node: PreflopDecisionNode) -> NodeDescriptor:
    """Flatten a node to its pickle-cheap descriptor (strings + floats)."""
    history = tuple(
        (a.position, a.action_type.value, a.raise_size_pct)
        for a in node.history_before
    )
    actions = tuple(
        (opt.action_type.value, opt.raise_size_pct, str(opt.range_file.path))
        for opt in node.actions
    )
    return (node.actor, history, actions)


def _history_from_descriptor(
    history: tuple[tuple[str, str, float | None], ...],
) -> tuple[ParsedAction, ...]:
    return tuple(
        ParsedAction(pos, PreflopActionType(at), size)
        for pos, at, size in history
    )


def lightweight_node(descr: NodeDescriptor, pack_id: str) -> PreflopDecisionNode:
    """A node carrying only actor + history (``actions=()``).

    Enough for :func:`node_action_context`, ``active_player_count`` and
    ``node_id`` -- all of which read only the history -- without paying to
    rebuild every action's range-file object. Use for the list / filter /
    count views.
    """
    return PreflopDecisionNode(
        pack_id=pack_id,
        actor=descr[0],
        history_before=_history_from_descriptor(descr[1]),
        actions=(),
    )


def descriptor_to_node(descr: NodeDescriptor, pack_id: str) -> PreflopDecisionNode:
    """Reconstruct the FULL, byte-identical node from its descriptor.

    The range files' ``action_history`` is rebuilt as ``history_before +
    (the actor's own action,)`` -- reusing the already-built history tuple
    rather than re-storing it -- which is both faithful (value-equal to the
    parse-walk) and cheap.
    """
    actor, history_raw, actions_raw = descr
    history_before = _history_from_descriptor(history_raw)
    options = []
    for at, size, path in actions_raw:
        action_type = PreflopActionType(at)
        range_file = ParsedRangeFile(
            pack_id=pack_id,
            path=Path(path),
            actor=actor,
            actor_action=action_type,
            actor_raise_size_pct=size,
            action_history=history_before
            + (ParsedAction(actor, action_type, size),),
        )
        options.append(
            PreflopActionOption(
                action_type=action_type,
                raise_size_pct=size,
                range_file=range_file,
            )
        )
    return PreflopDecisionNode(
        pack_id=pack_id,
        actor=actor,
        history_before=history_before,
        actions=tuple(options),
    )


# --- disk cache -------------------------------------------------------------
def _pack_signature(pack: PreflopPack) -> str:
    """A cheap fingerprint: the pack root's immediate child entries + mtimes.

    One ``scandir`` (no recursion), so it costs the same for 9 files or
    93,000. Re-extracting a pack replaces its top-level entries, moving
    their mtimes and changing this string. ``""`` when the root is
    unreadable -- an empty signature never matches a stored one, so the
    cache rebuilds.
    """
    try:
        entries = sorted(
            (e.name, e.stat(follow_symlinks=False).st_mtime_ns)
            for e in os.scandir(pack.root_path)
        )
    except OSError:
        return ""
    body = ";".join(f"{name}:{mtime}" for name, mtime in entries)
    return f"{pack.file_glob}|{body}"


def _cache_path(cache_dir: Path | str, pack_id: str) -> Path:
    return Path(cache_dir) / f"{pack_id}.nodes.pkl"


def _try_read(path: Path) -> CachedEnumeration | None:
    """Unpickle a cache file, or ``None`` on any problem. Never raises."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
            ModuleNotFoundError, ValueError) as exc:
        logger.warning("node_cache: ignoring unreadable cache %s: %s", path, exc)
        return None
    return obj if isinstance(obj, CachedEnumeration) else None


def _atomic_write(path: Path, payload: CachedEnumeration) -> None:
    """Pickle to a temp file + rename, so a reader never sees a half write.
    Write failures are logged, not raised -- caching is an optimization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("node_cache: could not write cache %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def load_descriptors(
    pack: PreflopPack,
    cache_dir: Path | str,
) -> tuple[tuple[NodeDescriptor, ...], int]:
    """Return ``(descriptors, file_count)`` for one pack, from disk if fresh.

    On a cache hit both come straight off disk (~90ms for the 9-max pack).
    On a miss (no cache / version or signature change / corrupt file) the
    pack is walked once, flattened to descriptors, persisted, and returned.
    ``file_count`` is the number of range files matched -- the sidebar's
    "pack ready" indicator, cached so it never re-globs 93k files.
    """
    path = _cache_path(cache_dir, pack.pack_id)
    signature = _pack_signature(pack)
    cached = _try_read(path)
    if (
        cached is not None
        and cached.version == CACHE_VERSION
        and cached.pack_id == pack.pack_id
        and cached.file_glob == pack.file_glob
        and cached.signature == signature
        and signature != ""
    ):
        logger.info(
            "node_cache: hit for %s (%d nodes, %d files)",
            pack.pack_id, len(cached.descriptors), cached.file_count,
        )
        return cached.descriptors, cached.file_count

    logger.info("node_cache: building enumeration for %s", pack.pack_id)
    nodes = enumerate_nodes([pack])
    descriptors = tuple(node_to_descriptor(n) for n in nodes)
    file_count = sum(1 for _ in pack.root_path.rglob(pack.file_glob))
    _atomic_write(
        path,
        CachedEnumeration(
            version=CACHE_VERSION,
            pack_id=pack.pack_id,
            file_glob=pack.file_glob,
            signature=signature,
            file_count=file_count,
            descriptors=descriptors,
        ),
    )
    return descriptors, file_count


def clear_cache(cache_dir: Path | str, pack_id: str | None = None) -> int:
    """Delete cached enumerations; return how many files were removed.

    ``pack_id=None`` clears every pack's cache in ``cache_dir``. The escape
    hatch for the rare case of an in-place pack edit the signature can't see.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return 0
    pattern = f"{pack_id}.nodes.pkl" if pack_id is not None else "*.nodes.pkl"
    removed = 0
    for path in cache_dir.glob(pattern):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


__all__ = [
    "CACHE_VERSION",
    "CachedEnumeration",
    "NodeDescriptor",
    "clear_cache",
    "descriptor_to_node",
    "lightweight_node",
    "load_descriptors",
    "node_to_descriptor",
]
