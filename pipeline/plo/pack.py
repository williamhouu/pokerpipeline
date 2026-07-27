"""PLO pack access — read a MonkerViewer `.rng` node and decode node paths.

Two responsibilities, both now fully specified (see ``docs/plo_rng_format.md``):

1. :func:`read_rng` — read one `.rng` file into the range it represents: the
   hands present (strategy weight ``p`` > 0) with their ``ev`` (small blinds).
   The i-th payload line is hand ``hand_order()[i]`` (the authoritative order
   baked in by :mod:`pipeline.plo.hand_order`); the pattern lines are ignored
   exactly as MonkerViewer ignores them.

2. :func:`parse_node_path` — decode a node's filename (``40100.0.1.rng`` ->
   action sequence) using the seat order and action tokens. Each `.rng` is the
   range with which the *last* actor takes the *last* action; the actions
   before it are the history that reached the decision.

This pack (PLO 6max 100bb) uses four tokens: ``0`` fold, ``1`` call, ``3``
all-in, ``40100`` the single pot-sized raise (``[2:]`` = 100% of pot). The
``5`` = min-raise token and other raise sizes are handled for generality but
do not occur here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pipeline.plo.hand_order import HAND_COUNT, hand_order

# Preflop acting order, 6-max. LJ (lojack) acts first; SB/BB have posted.
# (Kept as the module-level default so 6-max callers/tests are untouched;
# multi-pack code paths use each pack's ``spec.seats`` instead.)
SEATS: tuple[str, ...] = ("LJ", "HJ", "CO", "BU", "SB", "BB")

# Preflop acting order, 9-max (the July-2026 pack). Seat names follow the
# NLHE 9-max convention (UTG, UTG+1, UTG+2, LJ, HJ, CO, BTN, SB, BB) so both
# games read the same in the app; they are already display names, so the
# 6-max LJ->UTG / BU->BTN remap must NOT apply to them (display_seat is
# table-size aware).
SEATS_9MAX: tuple[str, ...] = (
    "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB",
)


# --- reading a node's range ----------------------------------------------
@dataclass(frozen=True)
class RngEntry:
    """One hand present in a node's range."""

    index: int  # position in the .rng file == hand_order() index
    label: str  # Monker hand string, e.g. "(AK)(AK)"
    p: float    # strategy weight in [0, 1] (how often the hand is here)
    ev: float   # expected value, in small blinds


def read_rng_values(path: Path) -> list[tuple[float, float]]:
    """Every hand's ``(p, ev)`` at the node stored in ``path``, by index.

    Returns a list of length :data:`HAND_COUNT`; element ``i`` is the
    ``(p, ev_sb)`` pair for ``hand_order()[i]``. ``ev`` is in small blinds.

    Unlike :func:`read_rng`, hands with ``p == 0`` are **kept**: Monker stores,
    per hand, the EV of *every* action including the ones the hand never takes
    (its counterfactual value), and that is exactly what is needed to measure
    how costly the alternative actions are (the EV gap). Raises ``ValueError``
    if the file does not have the expected ``2 x 16,432`` lines.
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
    if len(lines) != 2 * HAND_COUNT:
        msg = (
            f"{path.name}: expected {2 * HAND_COUNT} lines, got {len(lines)} "
            "-- not a 16,432-hand Omaha .rng"
        )
        raise ValueError(msg)

    values: list[tuple[float, float]] = []
    for i in range(HAND_COUNT):
        p_str, _, ev_str = lines[2 * i + 1].partition(";")
        ev = float(ev_str) / 1000.0 if ev_str else 0.0
        values.append((float(p_str), ev))
    return values


def read_rng(path: Path) -> list[RngEntry]:
    """The hands present (``p`` > 0) at the node stored in ``path``.

    Returns them in hand-order index order. Raises ``ValueError`` if the file
    does not have the expected ``2 x 16,432`` lines. For the full per-index
    ``(p, ev)`` table (including the zero-weight hands' counterfactual EVs) use
    :func:`read_rng_values`.
    """
    order = hand_order()
    return [
        RngEntry(index=i, label=order[i], p=p, ev=ev)
        for i, (p, ev) in enumerate(read_rng_values(path))
        if p > 0.0
    ]


# --- decoding a node's filename ------------------------------------------
class PloActionType(Enum):
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"
    MIN_RAISE = "min_raise"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class PloAction:
    seat: str
    action: PloActionType
    raise_pct: int | None = None  # % of pot for RAISE; None otherwise


# A raised seat keeps acting (it can face a re-raise); a folded or all-in seat
# is out of the action.
_CONTINUES = {PloActionType.CALL, PloActionType.RAISE, PloActionType.MIN_RAISE}
_SPECIAL_TOKENS: dict[str, PloActionType] = {
    "0": PloActionType.FOLD,
    "1": PloActionType.CALL,
    "3": PloActionType.ALL_IN,
    "5": PloActionType.MIN_RAISE,
}


def _decode_token(token: str) -> tuple[PloActionType, int | None]:
    special = _SPECIAL_TOKENS.get(token)
    if special is not None:
        return special, None
    # The 9-max export's generic raise token: pot-limit's one raise is the
    # pot, so bare "2" = a 100%-pot raise (the 6-max pack encodes the same
    # action as "40100"; "2" never occurs in it, so this mapping is safe
    # globally). Confirmed at intake (scripts/audit_plo9_pack.py): the pack's
    # only tokens are 0/1/2/3 and its EV geometry matches pot-sized opens.
    if token == "2":
        return PloActionType.RAISE, 100
    # Otherwise a sized raise; Monker encodes it as <2-char code><pct>.
    try:
        return PloActionType.RAISE, int(token[2:])
    except ValueError as exc:
        msg = f"unrecognised action token: {token!r}"
        raise ValueError(msg) from exc


def parse_node_path(
    stem: str, seats: tuple[str, ...] = SEATS
) -> tuple[PloAction, ...]:
    """Decode a `.rng` filename stem (no extension) into its action sequence.

    ``"40100.0"`` -> (LJ raise 100%, HJ fold). Seats act in ``seats`` order
    (default: the 6-max :data:`SEATS`; pass a pack's ``spec.seats`` for other
    tables); a caller/raiser rotates to the back (acts again on a re-raise), a
    folder/all-in leaves the action.
    """
    queue = list(seats)
    actions: list[PloAction] = []
    for token in stem.split("."):
        if not queue:
            msg = f"action {token!r} but no seat left to act in {stem!r}"
            raise ValueError(msg)
        seat = queue.pop(0)
        action_type, raise_pct = _decode_token(token)
        actions.append(PloAction(seat=seat, action=action_type, raise_pct=raise_pct))
        if action_type in _CONTINUES:
            queue.append(seat)
    if not actions:
        msg = f"empty node path: {stem!r}"
        raise ValueError(msg)
    return tuple(actions)


def node_actor(actions: tuple[PloAction, ...]) -> str:
    """The seat whose action range a node holds (the last to act)."""
    return actions[-1].seat


# --- pack registry ---------------------------------------------------------
@dataclass(frozen=True)
class PloPackSpec:
    """Static description of one known PLO pack (the NLHE
    ``KNOWN_PACK_SIGNATURES`` idea, sized for PLO's two-pack reality).

    ``dir_signature`` is the path fragment that identifies the pack inside an
    extracted Monker export (``.../ranges/Omaha/<N>-way/...``); discovery
    matches it against the `.rng` root's path.
    """

    pack_id: str            # stable id, recorded in batch meta (re-verifier key)
    display_label: str      # human label for the admin pack selector
    seats: tuple[str, ...]  # preflop acting order (pack-internal seat codes)
    table_size: int
    stack_bb: float
    rake_note: str          # e.g. "5% up to 2bb" -- documentation only
    dir_signature: str      # e.g. "Omaha/9-way"
    default_base: str       # conventional extraction folder in the repo root
    # MTT bb-ante packs (July 2026). ante_bb: the BB's dead ante, IN the pot
    # from the first action (the pot-limit sizes only resolve correctly with
    # it) but NOT part of the BB's callable commitment, and SUNK in the
    # pack's EV baseline (a BB fold's forced EV is the blind alone).
    ante_bb: float = 0.0
    # "cash" | "tournament" -- drives the Context line, the Cash/Tourney
    # column, and the bb-display default at generation time.
    game_format: str = "cash"
    # EV units in the .rng files: False = milli-SMALL-blind (the cash packs),
    # True = milli-BIG-blind (the MTT exports; SB open-fold EV reads -500).
    # The spot sampler normalises to sb-units at read time so every
    # downstream consumer (EV gap, difficulty, action_ev_bb) is unit-safe.
    ev_in_bb: bool = False
    # No Live/Online framing anywhere (July 22 2026, team ask): the Context
    # line says only the effective stacks and the CSV "Live or Online"
    # column stays blank. Set on the short-stack cash packs, whose venue
    # is deliberately unspecified.
    venue_neutral: bool = False


PLO_PACK_6MAX_100BB = PloPackSpec(
    pack_id="plo_6max_100bb",
    display_label="PLO 6-max · 100bb · rake 5%/1bb",
    seats=SEATS,
    table_size=6,
    stack_bb=100.0,
    rake_note="5% up to 1bb",
    dir_signature="Omaha/6-way",
    default_base="plo_ranges",
)

PLO_PACK_9MAX_100BB = PloPackSpec(
    pack_id="plo_9max_100bb",
    display_label="PLO 9-max · 100bb · rake 5%/2bb",
    seats=SEATS_9MAX,
    table_size=9,
    stack_bb=100.0,
    rake_note="5% up to 2bb",
    dir_signature="Omaha/9-way",
    default_base="plo9_ranges",
)

# Short-stack 6-max packs (July 2026): same provider/grammar/rake as the 100bb
# 6-max pack, but shallow trees for short-stack (tournament-flavoured) play.
# Their dir_signatures include the STACK folder so a 6-way root resolves to the
# right depth -- and they are listed BEFORE the 100bb spec (whose signature is
# the bare "Omaha/6-way") so the more-specific match wins. Intake-audited via
# scripts/audit_plo6_pack.py: 12bb fully clean; 20bb clean heads-up, with a
# tiny 3-way strategy/EV residual (0.04% of hands, <2.2sb) that is normal
# 3-way-PLO solve noise. NOTE: solved as cEV CASH (with rake), not ICM -- see
# the plo-shortstack-packs memory before framing these as "tournament".
PLO_PACK_6MAX_12BB = PloPackSpec(
    pack_id="plo_6max_12bb",
    display_label="PLO 6-max · 12bb · rake 5%/1bb",
    seats=SEATS,
    table_size=6,
    stack_bb=12.0,
    rake_note="5% up to 1bb",
    dir_signature="Omaha/6-way/12bb[5p-1bb]",
    default_base="plo12_ranges",
    venue_neutral=True,
)

PLO_PACK_6MAX_20BB = PloPackSpec(
    pack_id="plo_6max_20bb",
    display_label="PLO 6-max · 20bb · rake 5%/1bb",
    seats=SEATS,
    table_size=6,
    stack_bb=20.0,
    rake_note="5% up to 1bb",
    dir_signature="Omaha/6-way/20bb[5p-1bb]",
    default_base="plo20_ranges",
    venue_neutral=True,
)

# MTT bb-ante 6-max packs (July 2026): the "PLO MTT (BB ANTE) 3.5x Open"
# MonkerViewer export, one spec per extracted depth. No rake (tournament),
# 1bb BB ante, fixed ~3.5bb opens encoded as node-relative pot-% tokens
# (40072 first-in, 40083 SB first-in, ...), limp branches included. EV units
# are milli-BB with the ante sunk in the baseline -- both verified by
# scripts/audit_plo6_pack.py section [5b]. Chip-EV, NOT ICM: honest scope is
# early/deep-field tournament play. Depths 10-40 + 75 are extracted; the
# 50bb and 100bb depths stay unextracted (team call: not needed).
def _mtt_spec(stack: float, dir_name: str, base: str) -> PloPackSpec:
    return PloPackSpec(
        pack_id=f"plo_mtt_6max_{stack:g}bb",
        display_label=f"PLO MTT 6-max · {stack:g}bb · bb ante · chip-EV",
        seats=SEATS,
        table_size=6,
        stack_bb=stack,
        rake_note="no rake (MTT)",
        dir_signature=f"Omaha/6-way/{dir_name}",
        default_base=base,
        ante_bb=1.0,
        game_format="tournament",
        ev_in_bb=True,
    )


PLO_MTT_PACKS: tuple[PloPackSpec, ...] = (
    _mtt_spec(10, "10bb(bb-ante)3.5x", "plo_mtt10_ranges"),
    _mtt_spec(15, "15b(bb-ante)3.5x", "plo_mtt15_ranges"),
    _mtt_spec(20, "20bb(bb-ante)3.5x", "plo_mtt20_ranges"),
    _mtt_spec(25, "25bb(bb-ante)3.5x", "plo_mtt25_ranges"),
    _mtt_spec(30, "30b(bb-ante)3.5x", "plo_mtt30_ranges"),
    _mtt_spec(40, "40(bb-ante)3.5x", "plo_mtt40_ranges"),
    # Deep depth (July 21 PM, team ask): 75bb only -- the team wanted ~80bb;
    # the archive has no 80, so 75 is the nearest. 50bb and the deep MTT
    # 100bb deliberately skipped (disk; near-cash play).
    _mtt_spec(75, "75b(bb-ante)3.5x", "plo_mtt75_ranges"),
)

#: Every integrated PLO pack. Audit a new export first
#: (``scripts/audit_plo9_pack.py`` / ``scripts/audit_plo6_pack.py`` are the
#: templates), then register it here. ORDER MATTERS: specs whose dir_signature
#: is a prefix of another's (the bare ``Omaha/6-way`` 100bb spec) must come
#: LAST so the more-specific stack signatures match first in _spec_for_root.
KNOWN_PLO_PACKS: tuple[PloPackSpec, ...] = (
    PLO_PACK_6MAX_12BB,
    PLO_PACK_6MAX_20BB,
    *PLO_MTT_PACKS,
    PLO_PACK_6MAX_100BB,
    PLO_PACK_9MAX_100BB,
)


# --- pack discovery ------------------------------------------------------
@dataclass(frozen=True)
class PloPack:
    """A directory of `.rng` node files for one scenario."""

    root: Path  # the directory directly containing the .rng files
    label: str  # e.g. "Omaha/6-way/100bb(5p-1bb)"
    # Which registered pack this is. Defaults to the 6-max spec so every
    # pre-registry construction site (tests build PloPack(root, label))
    # behaves exactly as before.
    spec: PloPackSpec = field(default=PLO_PACK_6MAX_100BB)

    @property
    def seats(self) -> tuple[str, ...]:
        return self.spec.seats

    @property
    def table_size(self) -> int:
        return self.spec.table_size

    @property
    def pack_id(self) -> str:
        return self.spec.pack_id


def _spec_for_root(root: Path) -> PloPackSpec:
    """Match an `.rng` root directory to a registered spec by path fragment.

    Unknown layouts fall back to the 6-max spec -- the pre-registry behavior,
    so a renamed 6-max extraction keeps working. A NEW table size must be
    registered (and audited) explicitly to be recognized.
    """
    posix = root.as_posix()
    for spec in KNOWN_PLO_PACKS:
        if spec.dir_signature in posix:
            return spec
    return PLO_PACK_6MAX_100BB


def _rng_root(base: Path) -> Path | None:
    """The first directory (depth-first, subdirectories in name order)
    containing a `.rng` file. Early-exit on purpose: the old
    ``sorted(base.rglob(...))`` materialized and sorted the ENTIRE listing --
    ~3.3s on the 327k-file 9-max pack, and the admin calls discovery several
    times per render. Files are NOT sorted (any `.rng` hit identifies its
    directory as the root; which file it is doesn't matter), only the
    directory traversal order is deterministic."""
    import os  # noqa: PLC0415

    stack = [base]
    while stack:
        current = stack.pop()
        subdirs: list[Path] = []
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".rng"):
                        return current
                    if entry.is_dir():
                        subdirs.append(Path(entry.path))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        stack.extend(sorted(subdirs, reverse=True))  # depth-first, name order
    return None


@lru_cache(maxsize=8)
def discover_plo_pack(base: Path) -> PloPack:
    """Find the `.rng` directory under ``base`` (e.g. ``plo_ranges/``).

    Memoized per base path (July 2026): the admin selector labels + pack
    loaders call this on every render. Failures (no pack yet) are NOT cached,
    so dropping a pack in is picked up immediately; replacing a pack's
    CONTENTS in place needs a process restart (same caveat as the node
    caches).
    """
    root = _rng_root(base)
    if root is None:
        msg = f"no .rng files found under {base}"
        raise FileNotFoundError(msg)
    label = str(root.relative_to(base)) if root != base else root.name
    return PloPack(root=root, label=label, spec=_spec_for_root(root))


def discover_plo_packs(bases: tuple[Path, ...]) -> tuple[PloPack, ...]:
    """Every pack found across candidate base folders (missing/empty folders
    are skipped). The admin pack selector's data source."""
    packs: list[PloPack] = []
    for base in bases:
        try:
            packs.append(discover_plo_pack(base))
        except FileNotFoundError:
            continue
    return tuple(packs)


def rake_pct_from_note(rake_note: str) -> float:
    """The rake percentage in a spec's ``rake_note`` ("5% up to 1bb" -> 0.05).

    The notes are documentation strings, so this parses defensively: the
    first ``<number>%`` found wins; no percentage (e.g. "no rake (MTT)") is
    0.0. Used by the 🪤 trap detector's fold-side rake cushion -- an
    under-read (0.0) only makes fold-traps slightly easier to fire, never
    breaks a batch.
    """
    import re  # noqa: PLC0415

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", rake_note or "")
    return float(m.group(1)) / 100.0 if m else 0.0


def range_at(pack: PloPack, stem: str) -> list[RngEntry]:
    """Read the range for a node identified by its action-path stem."""
    return read_rng(pack.root / f"{stem}.rng")
