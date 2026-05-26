"""Filename-grammar parser for Ryan's PioViewer-format preflop range pack.

The grammar (from ``docs/ryan_range_pack_index.md``):

  * Files live in a per-position folder: ``<Pack>/BB/...txt``, ``<Pack>/BTN/...txt``
    etc. The folder name = the position whose range the file represents.
  * The filename itself is a chain of ``<Position>_<Action>`` token pairs
    separated by underscores, ending in a ``.txt`` extension.
  * The chain's FINAL ``<Position>`` must equal the parent folder name --
    i.e. the file is "<actor>'s range when <actor> takes <final action>
    after [all prior actions]".

Action tokens recognised:

  * ``Fold`` -- folds
  * ``Call`` -- calls the current bet
  * ``AI``   -- all-in shove
  * ``<digits>%`` -- sized raise; digits = literal percent-of-pot per the
    pack's docs. Stored verbatim as ``raise_size_pct`` (a float); the
    raise-to-bb conversion is downstream.

Example (Ryan's longest BTN file at 134 chars):

  ``UTG_60%_HJ_Call_CO_Call_BTN_Call_SB_Call_BB_198%_UTG_Call_HJ_Call_CO_Call_BTN_Call_SB_AI_BB_Fold_UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold``

  -> actor=BTN (matches parent folder), actor_action=Fold,
     action_history = 24 ParsedActions in order.

Round-order validation (UTG->HJ->CO->BTN->SB->BB cycle, skipping folded
and all-in players) is intentionally NOT enforced here; we trust the
pack is well-formed and defer that defense-in-depth to a later
``validate_pack`` step that could check the whole pack at once.
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.preflop.grammars.types import (
    ParsedAction,
    ParsedRangeFile,
    PreflopActionType,
)
from pipeline.preflop.pack import PreflopPack

GRAMMAR_NAME = "ryan_pack"

# The six positions Ryan's 6-max pack uses, and (for future 9-max) the
# three additional positions a 9-max pack would add. The parser accepts
# any of these; node enumerator filters by pack.table_size downstream.
_KNOWN_POSITIONS_6MAX = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
_KNOWN_POSITIONS_9MAX_EXTRA = ("UTG1", "UTG2", "LJ")
_ALL_KNOWN_POSITIONS = frozenset(_KNOWN_POSITIONS_6MAX + _KNOWN_POSITIONS_9MAX_EXTRA)

# Match a sizing token: integer or decimal followed by `%`. e.g. "60%",
# "76.5%", "182%".
_SIZE_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")


def parse(filename: Path, pack: PreflopPack) -> ParsedRangeFile:
    """Parse one Ryan-pack filename into a normalized ``ParsedRangeFile``.

    Args:
        filename: Path to a ``.txt`` file inside the pack. May be absolute
            or relative; only the filename and parent folder are used.
        pack: The pack this file belongs to (for ``pack_id`` tagging and
            future per-pack sanity checks).

    Returns:
        Normalized ``ParsedRangeFile``.

    Raises:
        ValueError: if the filename is malformed (odd token count, unknown
            position, unrecognised action token, parent-folder mismatch).
    """
    path = filename
    parent = path.parent.name
    stem = path.stem  # filename without extension

    tokens = stem.split("_")
    if len(tokens) < 2 or len(tokens) % 2 != 0:
        raise ValueError(
            f"ryan_pack filename has odd token count {len(tokens)} "
            f"(expected even pairs of <Position>_<Action>): {path.name!r}"
        )

    actions: list[ParsedAction] = []
    for i in range(0, len(tokens), 2):
        position = tokens[i]
        action_token = tokens[i + 1]
        if position not in _ALL_KNOWN_POSITIONS:
            raise ValueError(
                f"ryan_pack: unknown position {position!r} at token "
                f"index {i} in {path.name!r}"
            )
        action_type, raise_pct = _decode_action_token(action_token, path.name)
        actions.append(
            ParsedAction(
                position=position,
                action_type=action_type,
                raise_size_pct=raise_pct,
            )
        )

    final = actions[-1]
    if final.position != parent:
        raise ValueError(
            f"ryan_pack: file's final action position {final.position!r} "
            f"does not match parent folder {parent!r}: {path.name!r}. "
            "Per pack grammar, the chain must end in the folder's position."
        )

    return ParsedRangeFile(
        pack_id=pack.pack_id,
        path=path,
        actor=final.position,
        actor_action=final.action_type,
        actor_raise_size_pct=final.raise_size_pct,
        action_history=tuple(actions),
    )


def _decode_action_token(
    token: str, filename: str
) -> tuple[PreflopActionType, float | None]:
    """Map a single action token to (ActionType, optional raise size %)."""
    if token == "Fold":
        return (PreflopActionType.FOLD, None)
    if token == "Call":
        return (PreflopActionType.CALL, None)
    if token == "AI":
        return (PreflopActionType.ALL_IN, None)
    m = _SIZE_TOKEN_RE.match(token)
    if m:
        return (PreflopActionType.RAISE, float(m.group(1)))
    raise ValueError(
        f"ryan_pack: unrecognised action token {token!r} in {filename!r}. "
        "Expected Fold, Call, AI, or <digits>%."
    )
