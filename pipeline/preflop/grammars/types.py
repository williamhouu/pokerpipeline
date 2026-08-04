"""Shared output types for all grammar parsers.

Living in their own module so ``grammars/__init__.py`` can import them
without circular dependencies on individual grammar implementations
(which may want to import these types too).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Sentinel ``raise_size_pct`` meaning "minimum legal raise" rather than a
# percent-of-pot. Monker's ``5`` token (a min-raise, used by the
# short-stack 6-max packs) carries this: there is no pot fraction to apply
# -- the bb amount is the minimum legal raise, sized relative to the
# running bet in ``pipeline.preflop.action_history.resolve_preflop_history``.
# A negative value is impossible for a real pot-% token, so every size and
# label site can test for it unambiguously without a separate field (which
# would force a node-cache schema bump).
MIN_RAISE_PCT: float = -1.0

# Base for the fixed-bb sentinel family (MTT 8-max packs, Aug 2026): some
# Monker tokens (``14``/``15``/``16``/``18``) are absolute raise-TO sizes in
# bb, anchored per pack from the solve's own fold-EV bookkeeping (a raiser
# who later folds is out exactly the raise size). They are encoded into
# ``raise_size_pct`` as ``FIXED_BB_BASE - bb`` (e.g. 2.5bb -> -1002.5) so the
# value survives the node cache's plain-float descriptors unchanged -- same
# reasoning as MIN_RAISE_PCT, and unambiguous because real pot-% tokens are
# positive and MIN_RAISE_PCT is -1.0.
FIXED_BB_BASE: float = -1000.0


def encode_fixed_bb(raise_to_bb: float) -> float:
    """Encode an absolute raise-TO size (bb) into the sentinel float."""
    if raise_to_bb <= 0:
        raise ValueError(f"raise_to_bb must be > 0, got {raise_to_bb}")
    return FIXED_BB_BASE - raise_to_bb


def decode_fixed_bb(raise_size_pct: float | None) -> float | None:
    """The absolute bb size carried by a fixed-bb sentinel, else ``None``."""
    if raise_size_pct is not None and raise_size_pct <= FIXED_BB_BASE:
        return FIXED_BB_BASE - raise_size_pct
    return None


def render_raise_size_token(raise_size_pct: float | None) -> str:
    """Human/debug rendering of a raise's size token.

    Returns ``"min"`` for the min-raise sentinel (:data:`MIN_RAISE_PCT`),
    ``"<n>bb"`` for a fixed-bb sentinel (:func:`decode_fixed_bb`), otherwise
    the percent-of-pot form (``"60%"``). Used everywhere a raise
    is labelled -- option labels, node ids, villain action labels -- so the
    sentinel never leaks as ``"-1%"`` and the labels stay mutually
    consistent. (The premise-realism gate in ``pipeline.preflop.batch``
    matches option labels by exact string, so the option label and that
    gate's lookup label must render the same way.)
    """
    if raise_size_pct == MIN_RAISE_PCT:
        return "min"
    fixed = decode_fixed_bb(raise_size_pct)
    if fixed is not None:
        return f"{fixed:g}bb"
    return f"{raise_size_pct:g}%"


class PreflopActionType(StrEnum):
    """The four kinds of preflop actions a player can take.

    StrEnum (PEP 663, stdlib since 3.11) gives free JSON serialization
    and equality with the underlying string forms.
    """

    FOLD = "Fold"
    CALL = "Call"
    ALL_IN = "AllIn"
    RAISE = "Raise"  # any sized raise; size lives in ParsedAction


@dataclass(frozen=True)
class ParsedAction:
    """One preflop action inside the action history.

    For ``Fold`` / ``Call`` / ``AllIn``, ``raise_size_pct`` is ``None``.
    For ``Raise``, ``raise_size_pct`` is the vendor's percent-of-pot
    token (e.g. ``60.0`` for Ryan's ``60%`` open). Sizes-in-bb are
    derived downstream from the pack's stack/blind metadata, not here.
    """

    position: str  # "UTG" | "HJ" | "CO" | "BTN" | "SB" | "BB" (or 9-max extensions)
    action_type: PreflopActionType
    raise_size_pct: float | None = None  # only set for RAISE


@dataclass(frozen=True)
class ParsedRangeFile:
    """One parsed range-file's metadata.

    The range weights themselves (169 hand_class -> float entries) are
    loaded separately via ``pipeline.preflop_ranges.parse_range_file``;
    this struct is purely the path + action history + actor.

    The ``actor`` field identifies whose range this file represents --
    always equal to the position of the file's *last* action and to the
    file's parent folder name. The ``actor_action`` field is that final
    action (what the actor did to close the path encoded in the filename).
    """

    pack_id: str  # source PreflopPack.pack_id
    path: Path  # absolute filesystem path to the .txt
    actor: str  # whose range this file represents
    actor_action: PreflopActionType  # what actor did to close this path
    actor_raise_size_pct: float | None  # if RAISE, else None
    action_history: tuple[ParsedAction, ...]  # full sequence inc. actor's final action
