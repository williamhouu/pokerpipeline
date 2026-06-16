"""Disk-persistent caches of enumerated preflop nodes.

Walking the 9-max Monker pack -- 93,235 range files grouped into 44,058
``PreflopDecisionNode`` objects -- takes ~6 seconds, almost all of it
parsing every filename through the pack's grammar. The result is
*deterministic* for a given pack on disk, so re-deriving it on every fresh
process (and the admin panel restarts often) was the multi-second stall the
first time a session opened the Generate / Compare / Ranges page. It is
I/O-bound, which is why the panel sat near 0% CPU during the hang.

Pickling the full node objects does NOT help: the blob is 57 MB and
unpickling 44k richly-nested frozen dataclasses is as slow as re-parsing
(materialising millions of objects is the real cost). So this module keeps
TWO independent, self-contained caches keyed by a cheap pack signature:

  * :func:`load_metadata` -> ``<pack>.meta.pkl`` (~1 MB): per-node
    precomputed metadata (whatever the caller's ``derive`` returns -- the
    admin stores action context + player count). This is what the list /
    filter / count views need; it loads in tens of ms, so the sidebar,
    Generate and Compare are instant.
  * :func:`load_descriptors` -> ``<pack>.nodes.pkl`` (~18 MB): compact node
    descriptors (plain strings/floats) from which the FULL, byte-identical
    node is reconstructed (:func:`descriptor_to_node`) only where one is
    actually needed -- the Range viewer and the prompt-preview sampler.

The two caches are independent (no cross-coordination), so each rebuilds on
its own miss; a brand-new pack therefore walks twice the first time, which
is a once-ever cost. Within a process each is loaded once and held by the
admin's ``st.cache_resource``.

Invalidation is automatic and cheap: the signature is the pack's immediate
child entries plus their mtimes (one ``scandir``), so re-extracting a pack
-- which replaces those entries -- changes the signature and forces a
rebuild. An in-place edit to an existing range file that neither adds nor
removes a directory entry will NOT be detected (dir mtimes don't move on
content edits); packs are extracted wholesale and never edited in place,
but :func:`clear_cache` is the escape hatch. Bumping a version constant
invalidates the corresponding cache.

Reads are defensive: any unpickling error, version mismatch, or signature
mismatch falls back to a fresh walk + rewrite, so a corrupt or stale cache
degrades to "slow once", never to a crash.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable
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

# Bump when the descriptor shape (below) changes, so old node pickles are
# discarded rather than misread.
CACHE_VERSION = 2

# Bump when the metadata ``derive`` the caller passes changes meaning
# (the admin derives node_action_context / active_player_count); the cached
# values would otherwise go stale. Blast radius is UI-only -- the
# generation path never reads this cache, only the list/filter/count views.
META_CACHE_VERSION = 3

# One node, as pickle-cheap data:
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


@dataclass(frozen=True)
class CachedMetadata:
    """One pack's per-node precomputed metadata (caller-defined ``rows``)
    plus the fields that validate it. Small and fast to load."""

    version: int
    pack_id: str
    file_glob: str
    signature: str
    file_count: int
    rows: tuple


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


def descriptor_to_node(descr: NodeDescriptor, pack_id: str) -> PreflopDecisionNode:
    """Reconstruct the FULL, byte-identical node from its descriptor.

    The range files' ``action_history`` is rebuilt as ``history_before +
    (the actor's own action,)`` -- reusing the already-built history tuple
    rather than re-storing it -- which is both faithful (value-equal to the
    parse-walk) and cheap.
    """
    actor, history_raw, actions_raw = descr
    history_before = tuple(
        ParsedAction(pos, PreflopActionType(at), size)
        for pos, at, size in history_raw
    )
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


# --- disk cache helpers -----------------------------------------------------
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


def _nodes_path(cache_dir: Path | str, pack_id: str) -> Path:
    return Path(cache_dir) / f"{pack_id}.nodes.pkl"


def _meta_path(cache_dir: Path | str, pack_id: str) -> Path:
    return Path(cache_dir) / f"{pack_id}.meta.pkl"


def _try_read(path: Path, expected_type: type) -> object | None:
    """Unpickle a cache file, or ``None`` on any problem (missing, corrupt,
    wrong type). Never raises -- a bad cache must degrade to a rebuild."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
            ModuleNotFoundError, ValueError) as exc:
        logger.warning("node_cache: ignoring unreadable cache %s: %s", path, exc)
        return None
    return obj if isinstance(obj, expected_type) else None


def _atomic_write(path: Path, payload: object) -> None:
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


def _file_count(pack: PreflopPack) -> int:
    return sum(1 for _ in pack.root_path.rglob(pack.file_glob))


# --- public loaders ---------------------------------------------------------
def load_descriptors(
    pack: PreflopPack,
    cache_dir: Path | str,
) -> tuple[tuple[NodeDescriptor, ...], int]:
    """Return ``(descriptors, file_count)`` for one pack, from disk if fresh.

    On a cache hit both come straight off disk (~90ms for the 9-max pack).
    On a miss (no cache / version or signature change / corrupt file) the
    pack is walked once, flattened to descriptors, persisted, and returned.
    Use when a FULL node is needed (reconstruct via :func:`descriptor_to_node`).
    """
    path = _nodes_path(cache_dir, pack.pack_id)
    signature = _pack_signature(pack)
    cached = _try_read(path, CachedEnumeration)
    if (
        isinstance(cached, CachedEnumeration)
        and cached.version == CACHE_VERSION
        and cached.pack_id == pack.pack_id
        and cached.file_glob == pack.file_glob
        and cached.signature == signature
        and signature != ""
    ):
        logger.info(
            "node_cache: descriptors hit for %s (%d nodes)",
            pack.pack_id, len(cached.descriptors),
        )
        return cached.descriptors, cached.file_count

    logger.info("node_cache: building descriptors for %s", pack.pack_id)
    descriptors = tuple(node_to_descriptor(n) for n in enumerate_nodes([pack]))
    file_count = _file_count(pack)
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


def load_metadata(
    pack: PreflopPack,
    cache_dir: Path | str,
    derive: Callable[[PreflopDecisionNode], object],
) -> tuple[tuple, int]:
    """Return ``(rows, file_count)`` where ``rows[i] = derive(node_i)``.

    ``derive`` runs ONCE per node at build time (on the walked full nodes)
    and the result is cached, so warm loads skip both the walk and the
    per-node computation -- the list / filter / count views never touch a
    full node. The admin derives ``(actor, action_context, player_count)``.
    A cache hit is tens of ms; a miss walks the pack once and persists.
    """
    path = _meta_path(cache_dir, pack.pack_id)
    signature = _pack_signature(pack)
    cached = _try_read(path, CachedMetadata)
    if (
        isinstance(cached, CachedMetadata)
        and cached.version == META_CACHE_VERSION
        and cached.pack_id == pack.pack_id
        and cached.file_glob == pack.file_glob
        and cached.signature == signature
        and signature != ""
    ):
        logger.info(
            "node_cache: metadata hit for %s (%d rows)",
            pack.pack_id, len(cached.rows),
        )
        return cached.rows, cached.file_count

    logger.info("node_cache: building metadata for %s", pack.pack_id)
    rows = tuple(derive(n) for n in enumerate_nodes([pack]))
    file_count = _file_count(pack)
    _atomic_write(
        path,
        CachedMetadata(
            version=META_CACHE_VERSION,
            pack_id=pack.pack_id,
            file_glob=pack.file_glob,
            signature=signature,
            file_count=file_count,
            rows=rows,
        ),
    )
    return rows, file_count


def clear_cache(cache_dir: Path | str, pack_id: str | None = None) -> int:
    """Delete cached enumerations (both the node and metadata files); return
    how many files were removed.

    ``pack_id=None`` clears every pack's caches in ``cache_dir``. The escape
    hatch for the rare case of an in-place pack edit the signature can't see.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return 0
    stem = pack_id if pack_id is not None else "*"
    removed = 0
    for suffix in (".nodes.pkl", ".meta.pkl"):
        for path in cache_dir.glob(f"{stem}{suffix}"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


__all__ = [
    "CACHE_VERSION",
    "META_CACHE_VERSION",
    "CachedEnumeration",
    "CachedMetadata",
    "NodeDescriptor",
    "clear_cache",
    "descriptor_to_node",
    "load_descriptors",
    "load_metadata",
    "node_to_descriptor",
]
