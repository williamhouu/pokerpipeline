"""Shared output types for all grammar parsers.

Living in their own module so ``grammars/__init__.py`` can import them
without circular dependencies on individual grammar implementations
(which may want to import these types too).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


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
