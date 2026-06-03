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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pipeline.plo.hand_order import HAND_COUNT, hand_order

# Preflop acting order, 6-max. LJ (lojack) acts first; SB/BB have posted.
SEATS: tuple[str, ...] = ("LJ", "HJ", "CO", "BU", "SB", "BB")


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
    # Otherwise a raise; Monker encodes the size as <2-char code><pct>.
    try:
        return PloActionType.RAISE, int(token[2:])
    except ValueError as exc:
        msg = f"unrecognised action token: {token!r}"
        raise ValueError(msg) from exc


def parse_node_path(stem: str) -> tuple[PloAction, ...]:
    """Decode a `.rng` filename stem (no extension) into its action sequence.

    ``"40100.0"`` -> (LJ raise 100%, HJ fold). Seats act in :data:`SEATS`
    order; a caller/raiser rotates to the back (acts again on a re-raise), a
    folder/all-in leaves the action.
    """
    queue = list(SEATS)
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


# --- pack discovery ------------------------------------------------------
@dataclass(frozen=True)
class PloPack:
    """A directory of `.rng` node files for one scenario."""

    root: Path  # the directory directly containing the .rng files
    label: str  # e.g. "Omaha/6-way/100bb(5p-1bb)"


def discover_plo_pack(base: Path) -> PloPack:
    """Find the `.rng` directory under ``base`` (e.g. ``plo_ranges/``)."""
    for rng in sorted(base.rglob("*.rng")):
        root = rng.parent
        label = str(root.relative_to(base)) if root != base else root.name
        return PloPack(root=root, label=label)
    msg = f"no .rng files found under {base}"
    raise FileNotFoundError(msg)


def range_at(pack: PloPack, stem: str) -> list[RngEntry]:
    """Read the range for a node identified by its action-path stem."""
    return read_rng(pack.root / f"{stem}.rng")
