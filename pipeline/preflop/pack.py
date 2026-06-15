"""Preflop pack registry.

A "pack" is a folder of preflop range files produced by some vendor's solver
(e.g. the Ryan preflop pack, generated from MonkerSolver and exported in
PioViewer format). Each pack covers one combination of table size, stack
depth, and opening-size convention.

Packs are normalized into ``PreflopPack`` dataclasses, registered at startup
via :func:`discover_packs`, and looked up by ``pack_id`` thereafter. When a
new pack arrives (e.g. a future 9-max pack):

  1. Add a new ``PreflopPackSignature`` entry to ``KNOWN_PACK_SIGNATURES``
     describing its on-disk shape and metadata.
  2. If its filename grammar differs from existing packs, add a new parser
     under ``pipeline/preflop/grammars/`` and reference its name in the
     signature's ``grammar_name`` field.
  3. Drop the pack files into ``ranges/<pack_id>/...`` matching the
     signature's ``relative_pack_root``.

``discover_packs`` then picks it up automatically the next time it runs --
no code changes needed in the node enumerator, fact extractor, admin
panel, or downstream layers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreflopPack:
    """One uploaded preflop range pack.

    Carries the metadata that's not in the pack files themselves -- table
    size, stack depth, opening-size convention, grammar -- so downstream
    code (node enumerator, fact extractor) doesn't have to infer them.

    Fields:
        pack_id: Stable string identifier, e.g. "ryan_preflop_tree". Used
            as the key in :func:`get_pack` and to tag PreflopNodes with
            their source pack.
        root_path: Absolute filesystem path to the pack root -- the folder
            that directly contains the per-position subfolders
            (``BB/``, ``BTN/``, ``CO/``, ...).
        grammar_name: Name of the filename-grammar parser under
            ``pipeline/preflop/grammars/`` to use for this pack.
        table_size: 6 for 6-max, 9 for 9-max, etc.
        stack_depth_bb: Effective starting stack in big blinds (e.g. 100).
        open_size_bb: The "RFI" open size in big blinds (e.g. 2.5).
        sb_to_bb_ratio: Small-blind size as a fraction of big-blind size.
            0.5 in most cash games; tournaments sometimes use 0.4 or other
            ratios.
        file_glob: Glob pattern (relative to ``root_path``, recursive) the
            node enumerator walks to find this pack's range files.
            ``"*.txt"`` for the PioViewer-format Ryan pack; Monker-export
            packs use ``"*.rng"``.
        size_round_bb: Quantum (in bb) to snap resolved raise sizes to, or
            ``None`` for exact. Monker trees are specified in percent-of-pot,
            so exact sizes come out like 13.625bb; with ``0.5`` the rendered
            game plays "raise to 13.5bb" and the pot math follows the
            ROUNDED sizes, so prose, POT column, Seats tokens, and pot-odds
            all stay mutually consistent (June 2026, Zach's call). Rounding
            never touches all-ins (always the effective stack) and is
            display-game quantization only -- the solver's frequencies are
            from the exact tree.
        description: Short human-readable description for the admin panel.
    """

    pack_id: str
    root_path: Path
    grammar_name: str
    table_size: int
    stack_depth_bb: int
    open_size_bb: float
    sb_to_bb_ratio: float = 0.5
    file_glob: str = "*.txt"
    size_round_bb: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not 2 <= self.table_size <= 10:
            raise ValueError(f"table_size must be 2-10, got {self.table_size}")
        if self.stack_depth_bb <= 0:
            raise ValueError(f"stack_depth_bb must be > 0, got {self.stack_depth_bb}")
        if self.open_size_bb <= 0:
            raise ValueError(f"open_size_bb must be > 0, got {self.open_size_bb}")
        if not 0 < self.sb_to_bb_ratio <= 1:
            raise ValueError(
                f"sb_to_bb_ratio must be in (0, 1], got {self.sb_to_bb_ratio}"
            )


@dataclass(frozen=True)
class PreflopPackSignature:
    """Discoverable metadata for a known pack.

    :func:`discover_packs` matches a real folder under ``ranges/`` against
    each signature; if the ``relative_pack_root`` exists, the signature's
    metadata is used to build the corresponding ``PreflopPack``.
    """

    pack_id: str
    relative_pack_root: str  # path under `ranges/` to the pack's root folder
    grammar_name: str
    table_size: int
    stack_depth_bb: int
    open_size_bb: float
    sb_to_bb_ratio: float = 0.5
    file_glob: str = "*.txt"
    size_round_bb: float | None = None
    description: str = ""


# Known packs ship pre-registered here. Adding a new pack = appending a
# signature (and adding a grammar parser if the filename format differs).
KNOWN_PACK_SIGNATURES: tuple[PreflopPackSignature, ...] = (
    PreflopPackSignature(
        pack_id="ryan_preflop_tree_6max_100bb",
        relative_pack_root=("ryan_preflop_tree/PioViewer - NLH 6max 100bb 2.5x Open"),
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
        sb_to_bb_ratio=0.5,
        description="Ryan's 6-max 100bb 2.5x Open pack, PioViewer format.",
    ),
    # The 9-max Monker pack lives in its own gitignored sibling dir (where
    # the June-2026 extraction was verified), not under ranges/ -- the
    # ".." hop is deliberate; discover_packs resolves it away.
    PreflopPackSignature(
        pack_id="monker_nlhe_9max_100bb",
        relative_pack_root="../nlhe9_ranges/ranges/Hold'em/9-way/100bb[10p-3bb]",
        grammar_name="monker_nlhe",
        table_size=9,
        stack_depth_bb=100,
        open_size_bb=4.0,  # the root 40120 token = 120% pot = 4bb
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # pot-% tree -> snap rendered sizes to 0.5bb
        description=(
            "NLHE 9-max 100bb Monker pack -- 4x opens, rake 10%/3bb cap "
            "(see docs/nlhe9_pack_notes.md)."
        ),
    ),
    # NLHE 6-max short-stack Monker packs (gitignored sibling dir, like the
    # 9-max pack). Opens are the min-raise token `5` (2bb); the BvB iso over
    # a limp is token `14` (2.5bb). Both decoded + locked in
    # scripts/audit_nlhe6_pack.py; details in docs/nlhe6_pack_notes.md.
    PreflopPackSignature(
        pack_id="monker_nlhe_6max_20bb",
        relative_pack_root="../nlhe6_ranges/ranges/Hold'em/6-way/20bb(5p-0.5bb)",
        grammar_name="monker_nlhe",
        table_size=6,
        stack_depth_bb=20,
        open_size_bb=2.0,  # the `5` min-raise open = 2bb
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # pot-% 3-bets -> snap rendered sizes to 0.5bb
        description=(
            "NLHE 6-max 20bb Monker pack -- min-raise (2bb) opens, "
            "rake 5%/0.5bb cap (see docs/nlhe6_pack_notes.md)."
        ),
    ),
    PreflopPackSignature(
        pack_id="monker_nlhe_6max_30bb",
        relative_pack_root="../nlhe6_ranges/ranges/Hold'em/6-way/30bb(5p-0.5bb)",
        grammar_name="monker_nlhe",
        table_size=6,
        stack_depth_bb=30,
        open_size_bb=2.0,  # the `5` min-raise open = 2bb
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # pot-% 3-bets -> snap rendered sizes to 0.5bb
        description=(
            "NLHE 6-max 30bb Monker pack -- min-raise (2bb) opens, "
            "rake 5%/0.5bb cap (see docs/nlhe6_pack_notes.md)."
        ),
    ),
)


# --- registry ---------------------------------------------------------------
# Module-level dict, populated by register_pack / discover_packs. Reset via
# clear_registry (intended for tests; not part of the public API).
_PACKS: dict[str, PreflopPack] = {}


def register_pack(pack: PreflopPack) -> None:
    """Register a pack in the module-level registry.

    Raises ValueError if a pack with the same ``pack_id`` is already
    registered (prevents accidental shadowing).
    """
    if pack.pack_id in _PACKS:
        raise ValueError(
            f"pack_id {pack.pack_id!r} is already registered "
            f"(at {_PACKS[pack.pack_id].root_path!r}); "
            "call clear_registry() first if intentional."
        )
    _PACKS[pack.pack_id] = pack


def get_pack(pack_id: str) -> PreflopPack:
    """Look up a registered pack by id. Raises KeyError if missing."""
    try:
        return _PACKS[pack_id]
    except KeyError as exc:
        known = ", ".join(sorted(_PACKS)) or "(none registered)"
        raise KeyError(
            f"no preflop pack registered for {pack_id!r}. Known packs: {known}."
        ) from exc


def all_packs() -> tuple[PreflopPack, ...]:
    """Return all registered packs in insertion order."""
    return tuple(_PACKS.values())


def clear_registry() -> None:
    """Empty the registry. Intended for tests; safe to call anywhere."""
    _PACKS.clear()


def discover_packs(
    ranges_root: Path | str,
    signatures: Iterable[PreflopPackSignature] = KNOWN_PACK_SIGNATURES,
) -> tuple[PreflopPack, ...]:
    """Scan ``ranges_root`` and register every known pack that exists on disk.

    For each signature in ``signatures``, checks whether
    ``ranges_root / relative_pack_root`` is an existing directory. If yes,
    builds a ``PreflopPack`` from the signature's metadata + the resolved
    absolute path, registers it, and adds it to the returned tuple.

    Idempotent in the sense that you can call it once per process; calling
    it twice without :func:`clear_registry` in between will raise on the
    second call (each signature would try to re-register its pack_id).

    Args:
        ranges_root: Repo's ``ranges/`` directory (absolute or relative).
        signatures: Override the default ``KNOWN_PACK_SIGNATURES`` (tests).

    Returns:
        Tuple of all packs that were found and registered. Empty tuple if
        none of the known packs are present on disk.
    """
    root = Path(ranges_root).resolve()
    found: list[PreflopPack] = []
    for sig in signatures:
        # resolve() collapses any ".." hops (packs living in sibling dirs
        # of ranges/) so downstream paths/solver references stay clean.
        pack_root = (root / sig.relative_pack_root).resolve()
        if not pack_root.is_dir():
            continue
        pack = PreflopPack(
            pack_id=sig.pack_id,
            root_path=pack_root,
            grammar_name=sig.grammar_name,
            table_size=sig.table_size,
            stack_depth_bb=sig.stack_depth_bb,
            open_size_bb=sig.open_size_bb,
            sb_to_bb_ratio=sig.sb_to_bb_ratio,
            file_glob=sig.file_glob,
            size_round_bb=sig.size_round_bb,
            description=sig.description,
        )
        register_pack(pack)
        found.append(pack)
    return tuple(found)


# Helper: filter registered packs by table size (admin panel uses this).
def packs_by_table_size(table_size: int) -> tuple[PreflopPack, ...]:
    """Return all registered packs with the given table size, ordered."""
    return tuple(p for p in _PACKS.values() if p.table_size == table_size)
