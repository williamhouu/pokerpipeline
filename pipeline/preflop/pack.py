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
            all stay mutually consistent (June 2026, the team's call). Rounding
            never touches all-ins (always the effective stack) and is
            display-game quantization only -- the solver's frequencies are
            from the exact tree.
        ev_units_per_bb: How many raw file-EV units equal one big blind, or
            ``None`` when the pack carries no per-action EVs (the PioViewer
            Ryan pack stores weights only). Monker packs ship a per-hand EV
            per action, but the unit differs by pack: the 9-max pack is
            milli-big-blinds (1000 units = 1bb), the short-stack 6-max packs
            are milli-small-blinds (2000 units = 1bb, since 1sb = 0.5bb).
            Used by the preflop format writer to fill the ``action_ev_bb``
            column; verified per-pack in the audit scripts.
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
    ev_units_per_bb: float | None = None
    # Rake as a fraction of the pot (e.g. 0.10 for 10%). Used as an extra equity
    # cushion in the trap-aware difficulty detector so a hand the solver folds
    # despite raw equity clearing the NAIVE pot-odds price -- correct because
    # rake (and, at depth, realisation) make it -EV -- is not mislabelled a
    # "trap". None/0 = no cushion. Not (yet) applied to displayed pot odds.
    rake_pct: float | None = None
    # Big-blind ante in bb (MTT bb-ante formats, Aug 2026). Dead money: it
    # joins the pot (resolve_preflop_history's Monker pot-relative sizing and
    # every pot/price consumer include it) but never changes the bet level a
    # caller must match. 0.0 = no ante (all cash packs). Anchored per pack by
    # scripts/audit_mtt8_pack.py (the 40043 open resolves to 2.5bb ONLY with
    # the ante in the pot).
    ante_bb: float = 0.0
    # "cash" or "tournament" -- the pack's native framing. Batch callers may
    # still override the cosmetic game_format columns, but MTT packs default
    # the whole render (Context, Cash/Tourney, animation) to tournament.
    game_format: str = "cash"
    # Absolute raise-TO sizes for special Monker tokens, as a tuple of
    # (token, bb) pairs (tuple: the dataclass is frozen+hashable). The MTT
    # 8-max packs use small-int tokens (14/15/16/18) whose sizes were
    # anchored from fold-EV bookkeeping; the grammar encodes them via
    # ``grammars.types.encode_fixed_bb`` so the node cache round-trips them.
    fixed_raise_tokens_bb: tuple[tuple[str, float], ...] | None = None
    # Explicit override for pack_allins_realistic (None = the <=40bb default).
    # MTT packs at 50/75bb set True: a 5-bet jam at those depths is a real
    # tournament line, not a tree artifact.
    allins_realistic: bool | None = None

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
        if self.ev_units_per_bb is not None and self.ev_units_per_bb <= 0:
            raise ValueError(
                f"ev_units_per_bb must be > 0 or None, got {self.ev_units_per_bb}"
            )


def pack_allins_realistic(pack: "PreflopPack") -> bool:
    """Whether this pack's all-in actions are REAL poker lines (team standing
    rule, July 2026: no unrealistic all-ins in generated content).

    Realism is about the SITUATION: a jam at short stacks is a normal line;
    a 100-300bb preflop jam exists because the solver TREE offers it, not
    because anyone plays it. Default: realistic only at short stacks
    (<= 40bb effective); a pack can override via an ``allins_realistic``
    attribute if a future registration needs to."""
    override = getattr(pack, "allins_realistic", None)
    if override is not None:
        return bool(override)
    return float(pack.stack_depth_bb) <= 40.0  # noqa: PLR2004


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
    ev_units_per_bb: float | None = None
    rake_pct: float | None = None
    ante_bb: float = 0.0
    game_format: str = "cash"
    fixed_raise_tokens_bb: tuple[tuple[str, float], ...] | None = None
    allins_realistic: bool | None = None


# Known packs ship pre-registered here. Adding a new pack = appending a
# signature (and adding a grammar parser if the filename format differs).
KNOWN_PACK_SIGNATURES: tuple[PreflopPackSignature, ...] = (
    PreflopPackSignature(
        pack_id="ryan_preflop_tree_6max_100bb",
        rake_pct=0.04,
        relative_pack_root=("ryan_preflop_tree/PioViewer - NLH 6max 100bb 2.5x Open"),
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
        sb_to_bb_ratio=0.5,
        # Snap rendered raise sizes to the 0.5bb grid (parity with the Monker
        # packs + the postflop/PLO 0.5bb display rounding). A no-op if its
        # lookup-table sizes are already clean.
        size_round_bb=0.5,
        description="Ryan's 6-max 100bb 2.5x Open pack, PioViewer format.",
    ),
    # The 9-max Monker pack lives in its own gitignored sibling dir (where
    # the June-2026 extraction was verified), not under ranges/ -- the
    # ".." hop is deliberate; discover_packs resolves it away.
    PreflopPackSignature(
        pack_id="monker_nlhe_9max_100bb",
        rake_pct=0.1,
        relative_pack_root="../nlhe9_ranges/ranges/Hold'em/9-way/100bb[10p-3bb]",
        grammar_name="monker_nlhe",
        table_size=9,
        stack_depth_bb=100,
        open_size_bb=4.0,  # the root 40120 token = 120% pot = 4bb
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # pot-% tree -> snap rendered sizes to 0.5bb
        ev_units_per_bb=1000.0,  # file EVs are milli-bb (see nlhe9_pack_notes.md)
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
        rake_pct=0.05,
        relative_pack_root="../nlhe6_ranges/ranges/Hold'em/6-way/20bb(5p-0.5bb)",
        grammar_name="monker_nlhe",
        table_size=6,
        stack_depth_bb=20,
        open_size_bb=2.0,  # the `5` min-raise open = 2bb
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # pot-% 3-bets -> snap rendered sizes to 0.5bb
        ev_units_per_bb=2000.0,  # file EVs are milli-sb (see nlhe6_pack_notes.md)
        description=(
            "NLHE 6-max 20bb Monker pack -- min-raise (2bb) opens, "
            "rake 5%/0.5bb cap (see docs/nlhe6_pack_notes.md)."
        ),
    ),
    # 8-max LIVE cash packs (100/200/300bb), each materialised from a flat
    # SQLite solve by scripts/convert_preflop_db_to_pack.py into bb-native
    # .rng files (the `gto_preflop_8max` grammar). Full open->4-bet tree;
    # vs_5bet EXCLUDED at conversion (the only under-converged layer). EVs are
    # per-action in bb (ev_units_per_bb=1.0). Source provider deliberately not
    # named. All three share the same 91-node/358-file structure and the 3bb
    # open. Audited 2026-07-01 (100/300) mirroring the 200bb audit: premiums
    # clean, RFI widths depth-sensible, EVs distinct per depth; known same-class
    # artifact = single-pair fold islands on a few deep vs_4bet nodes (TT/88 at
    # 100bb, 55 at 300bb — same shape as 200bb's documented 44 flaw).
    PreflopPackSignature(
        pack_id="preflop_8max_100bb",
        rake_pct=0.1,
        relative_pack_root="preflop_8max_100bb",
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=100,
        open_size_bb=3.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # bb sizes already clean; parity no-op
        ev_units_per_bb=1.0,  # file EVs are already in bb
        description="8-max 100bb LIVE cash preflop pack (open->4-bet; vs-5bet excluded).",
    ),
    # IMPROVED variant of the 100bb pack (user-supplied; final refresh from
    # improved_2 on 2026-07-02): the adjustment pass (whole-percent frequency
    # grid, <=3% slivers snapped to 0, >=97% near-pures to 100) PLUS open-node
    # inversion repairs (HJ K7s/K8s equalized; softened 0%-vs-100% cliffs next
    # to wheel hands -- the A5>A6/K5s>K6s ordering itself is real GTO
    # structure, kept) PLUS the vs_4bet 88 fold-island repairs (88 fold 93% ->
    # 30% where 77/99 continue). Audited each round: sums exact, EV ripples
    # negligible, RFI stable, no premium inversions. Remaining known artifact:
    # 3 TT vs_4bet fold-islands (CO|LJ, UTG1|UTG, HJ|LJ). pack_id/description
    # say IMPROVED so the dropdown disambiguates.
    PreflopPackSignature(
        pack_id="preflop_8max_100bb_IMPROVED",
        rake_pct=0.1,
        relative_pack_root="preflop_8max_100bb_improved",
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=100,
        open_size_bb=3.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # bb sizes already clean; parity no-op
        ev_units_per_bb=1.0,  # file EVs are already in bb
        description=(
            "IMPROVED 8-max 100bb LIVE cash preflop pack (whole-percent grid, "
            "slivers snapped, open-node inversion repairs; open->4-bet; "
            "vs-5bet excluded)."
        ),
    ),
    PreflopPackSignature(
        pack_id="preflop_8max_200bb",
        rake_pct=0.1,
        relative_pack_root="preflop_8max_200bb",
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=200,
        open_size_bb=3.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # bb sizes already clean; parity no-op
        ev_units_per_bb=1.0,  # file EVs are already in bb
        description="8-max 200bb LIVE cash preflop pack (open->4-bet; vs-5bet excluded).",
    ),
    # IMPROVED variant of the 200bb pack (user-supplied 2026-07-03): the same
    # adjustment pass as the 100/300bb IMPROVED packs -- whole-percent grid,
    # sliver/near-pure snapping, open-node cliff softening (LJ K6s 0->38%,
    # HJ A6o 100->71%), AJs vs-open/vs-3bet smoothing, and the vs_4bet 88
    # fold-island repair (CO|HJ 92->33%). Audited vs the original: sums exact,
    # EV ripples <=0.24bb, RFI identical, no premium inversions. Remaining
    # known artifacts: the documented 44 vs_3bet island (BTN vs BB -- the real
    # one; the SB-node ones are phantom, reach-gated out) + one TT vs_4bet
    # island (SB|LJ).
    PreflopPackSignature(
        pack_id="preflop_8max_200bb_IMPROVED",
        rake_pct=0.1,
        relative_pack_root="preflop_8max_200bb_improved",
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=200,
        open_size_bb=3.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # bb sizes already clean; parity no-op
        ev_units_per_bb=1.0,  # file EVs are already in bb
        description=(
            "IMPROVED 8-max 200bb LIVE cash preflop pack (whole-percent grid, "
            "slivers snapped, open-node inversion repairs; open->4-bet; "
            "vs-5bet excluded)."
        ),
    ),
    PreflopPackSignature(
        pack_id="preflop_8max_300bb",
        rake_pct=0.1,
        relative_pack_root="preflop_8max_300bb",
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=300,
        open_size_bb=3.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # bb sizes already clean; parity no-op
        ev_units_per_bb=1.0,  # file EVs are already in bb
        description="8-max 300bb LIVE cash preflop pack (open->4-bet; vs-5bet excluded).",
    ),
    # IMPROVED variant of the 300bb pack (user-supplied 2026-07-02): the same
    # adjustment pass as the 100bb IMPROVED -- whole-percent frequency grid,
    # sliver/near-pure snapping, open-node inversion repairs (LJ K8s/K7s
    # equalized, HJ A6o cliff softened), plus KJs/AJs vs-open/vs-3bet
    # smoothing. Audited vs the original 300bb: sums exact, EV ripples
    # <=0.29bb, RFI stable, no premium inversions. Remaining known artifact:
    # the 55 vs_4bet fold-islands (55 folds ~100% while 44/66 continue) --
    # inherited, not this pass's target.
    PreflopPackSignature(
        pack_id="preflop_8max_300bb_IMPROVED",
        rake_pct=0.1,
        relative_pack_root="preflop_8max_300bb_improved",
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=300,
        open_size_bb=3.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # bb sizes already clean; parity no-op
        ev_units_per_bb=1.0,  # file EVs are already in bb
        description=(
            "IMPROVED 8-max 300bb LIVE cash preflop pack (whole-percent grid, "
            "slivers snapped, open-node inversion repairs; open->4-bet; "
            "vs-5bet excluded)."
        ),
    ),
    PreflopPackSignature(
        pack_id="monker_nlhe_6max_30bb",
        rake_pct=0.05,
        relative_pack_root="../nlhe6_ranges/ranges/Hold'em/6-way/30bb(5p-0.5bb)",
        grammar_name="monker_nlhe",
        table_size=6,
        stack_depth_bb=30,
        open_size_bb=2.0,  # the `5` min-raise open = 2bb
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,  # pot-% 3-bets -> snap rendered sizes to 0.5bb
        ev_units_per_bb=2000.0,  # file EVs are milli-sb (see nlhe6_pack_notes.md)
        description=(
            "NLHE 6-max 30bb Monker pack -- min-raise (2bb) opens, "
            "rake 5%/0.5bb cap (see docs/nlhe6_pack_notes.md)."
        ),
    ),
    # NLH MTT 8-max BB-ante Monker packs (Aug 2026, MonkerGuy "NLH MTT 8max
    # (BB ANTE)" exports; gitignored sibling dirs mtt8_<depth>_ranges/). Shared
    # conventions, all anchored by scripts/audit_mtt8_pack.py from the files'
    # own EV bookkeeping: EVs milli-SMALL-blind (2000/bb; SB open-fold reads
    # -1000 = -0.5bb); the 1bb BB ante is IN the pot for Monker's pot-relative
    # sizing (the 75/300bb `40043` open anchors at 2.505bb = formula WITH ante;
    # 2.075 without) but SUNK in the EV baseline (BB fold vs an open reads
    # -1bb, the blind alone). Chip-EV, no ICM, no rake -- honest scope is
    # deep-field tournament play. Special small-int tokens are absolute
    # raise-TO sizes (fixed_raise_tokens_bb), each anchored by an
    # opener-later-folds EV probe. The excluded sibling depths (40/100/200bb)
    # were a deliberate scope cut (user call, Aug 2026).
    PreflopPackSignature(
        pack_id="monker_mtt8_10bb",
        relative_pack_root="../mtt8_10bb_ranges",
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=10,
        # Jam-or-fold tree (tokens 0/1/3 only): the "open" is the 10bb jam.
        open_size_bb=10.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,
        ante_bb=1.0,
        game_format="tournament",
        description=(
            "NLH MTT 8-max 10bb (BB ante 1bb) Monker pack -- jam-or-fold "
            "tree; vendor folder '10bb(vs50bb)', verified 10bb effective."
        ),
    ),
    PreflopPackSignature(
        pack_id="monker_mtt8_15bb",
        relative_pack_root="../mtt8_15bb_ranges",
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=15,
        open_size_bb=2.0,  # the `5` min-raise open
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,
        ante_bb=1.0,
        game_format="tournament",
        # 14 = the SB first-in raise AND the BB iso over an SB limp (both
        # anchor at 2.5bb); 15 = the SB/BB iso over a BTN limp (3bb).
        fixed_raise_tokens_bb=(("14", 2.5), ("15", 3.0)),
        description="NLH MTT 8-max 15bb (BB ante 1bb) Monker pack -- 2bb opens.",
    ),
    PreflopPackSignature(
        pack_id="monker_mtt8_20bb",
        relative_pack_root="../mtt8_20bb_ranges",
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=20,
        open_size_bb=2.0,  # the `5` min-raise open
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,
        ante_bb=1.0,
        game_format="tournament",
        # 16 = the SB iso over a BTN limp (3.5bb); 18 = the BB iso over a
        # BTN limp after the SB folds (4.5bb).
        fixed_raise_tokens_bb=(("16", 3.5), ("18", 4.5)),
        description="NLH MTT 8-max 20bb (BB ante 1bb) Monker pack -- 2bb opens.",
    ),
    PreflopPackSignature(
        pack_id="monker_mtt8_30bb",
        relative_pack_root="../mtt8_30bb_ranges",
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=30,
        open_size_bb=2.0,  # the `5` min-raise open
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,
        ante_bb=1.0,
        game_format="tournament",
        description="NLH MTT 8-max 30bb (BB ante 1bb) Monker pack -- 2bb opens.",
    ),
    PreflopPackSignature(
        pack_id="monker_mtt8_50bb",
        relative_pack_root="../mtt8_50bb_ranges",
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=50,
        # The `40034` open resolves to 2.19bb with the ante in the pot and
        # renders 2bb on the 0.5bb display grid (the pot math follows the
        # rounded game, per the pack-wide convention).
        open_size_bb=2.0,
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,
        ante_bb=1.0,
        game_format="tournament",
        allins_realistic=True,  # a 5-bet jam at 50bb is a real MTT line
        description="NLH MTT 8-max 50bb (BB ante 1bb) Monker pack.",
    ),
    PreflopPackSignature(
        pack_id="monker_mtt8_75bb",
        relative_pack_root="../mtt8_75bb_ranges",
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=75,
        open_size_bb=2.5,  # the `40043` open = 2.505bb -> 2.5bb on the grid
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,
        ante_bb=1.0,
        game_format="tournament",
        allins_realistic=True,  # stacking off pre at 75bb is still a real line
        description="NLH MTT 8-max 75bb (BB ante 1bb) Monker pack.",
    ),
    PreflopPackSignature(
        pack_id="monker_mtt8_300bb",
        relative_pack_root="../mtt8_300bb_ranges",
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=300,
        open_size_bb=2.5,  # the `40043` open = 2.505bb -> 2.5bb on the grid
        sb_to_bb_ratio=0.5,
        file_glob="*.rng",
        size_round_bb=0.5,
        ev_units_per_bb=2000.0,
        ante_bb=1.0,
        game_format="tournament",
        # 300bb preflop jams are tree artifacts (default <=40bb rule applies).
        description="NLH MTT 8-max 300bb (BB ante 1bb) Monker pack.",
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
            ev_units_per_bb=sig.ev_units_per_bb,
            rake_pct=sig.rake_pct,
            ante_bb=sig.ante_bb,
            game_format=sig.game_format,
            fixed_raise_tokens_bb=sig.fixed_raise_tokens_bb,
            allins_realistic=sig.allins_realistic,
        )
        register_pack(pack)
        found.append(pack)
    return tuple(found)


# Helper: filter registered packs by table size (admin panel uses this).
def packs_by_table_size(table_size: int) -> tuple[PreflopPack, ...]:
    """Return all registered packs with the given table size, ordered."""
    return tuple(p for p in _PACKS.values() if p.table_size == table_size)
