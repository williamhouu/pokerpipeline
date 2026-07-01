"""Filename-grammar parser for the 8-max 200bb preflop pack.

This pack is materialised from a flat SQLite solve by
``scripts/convert_preflop_db_to_pack.py`` into ``.rng`` files whose NAME encodes
the full preflop action line. The grammar here turns that name back into a
``ParsedRangeFile``.

Filename format (``.rng`` extension; content is the Monker weight;ev format read
by ``pipeline.preflop_ranges.parse_range_file``):

  * Tokens ``<SEAT>-<CODE>`` joined by underscores. The FINAL token is the
    actor's action (one action option at the node); the rest are the history.
  * ``SEAT`` is one of UTG, UTG+1, LJ, HJ, CO, BTN, SB, BB.
  * ``CODE`` is ``F`` (fold), ``C`` (call), ``A`` (all-in), or ``R<bb>``
    (raise TO ``<bb>`` big blinds, e.g. ``R3`` / ``R38.5``).

Example::

  UTG-F_UTG+1-F_LJ-F_HJ-F_CO-F_BTN-R3_SB-F_BB-R17_BTN-C.rng

  -> actor=BTN, actor_action=Call, action_history = 9 ParsedActions
     (BTN opened to 3, BB 3-bet to 17, BTN now calls = a vs-3bet node).

Unlike the Monker grammar, raise sizes are stored in BIG BLINDS, not
percent-of-pot. They ride in ``raise_size_pct`` (the shared field) verbatim;
``pipeline.preflop.action_history`` has a bb-native branch for this grammar that
uses the value directly as bb (no pot-fraction conversion).
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

GRAMMAR_NAME = "gto_preflop_8max"

_KNOWN_POSITIONS = frozenset(
    ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB")
)
# A raise code: "R" followed by an integer or decimal bb size (R3, R38.5).
_RAISE_RE = re.compile(r"^R(\d+(?:\.\d+)?)$")


def _decode(token: str, filename: str) -> tuple[str, PreflopActionType, float | None]:
    """`<SEAT>-<CODE>` -> (seat, action_type, raise_size_bb_or_None)."""
    seat, _, code = token.partition("-")
    if not code:
        raise ValueError(f"gto_preflop_8max: malformed token {token!r} in {filename!r}")
    if seat not in _KNOWN_POSITIONS:
        raise ValueError(f"gto_preflop_8max: unknown seat {seat!r} in {filename!r}")
    if code == "F":
        return (seat, PreflopActionType.FOLD, None)
    if code == "C":
        return (seat, PreflopActionType.CALL, None)
    if code == "A":
        return (seat, PreflopActionType.ALL_IN, None)
    m = _RAISE_RE.match(code)
    if m:
        return (seat, PreflopActionType.RAISE, float(m.group(1)))
    raise ValueError(f"gto_preflop_8max: bad action code {code!r} in {filename!r}")


def parse(filename: Path, pack: PreflopPack) -> ParsedRangeFile:
    """Parse one ``.rng`` filename into a normalized ``ParsedRangeFile``."""
    path = filename
    tokens = path.stem.split("_")
    if not tokens:
        raise ValueError(f"gto_preflop_8max: empty filename {path.name!r}")
    actions = [
        ParsedAction(position=seat, action_type=at, raise_size_pct=sz)
        for seat, at, sz in (_decode(tok, path.name) for tok in tokens)
    ]
    final = actions[-1]
    return ParsedRangeFile(
        pack_id=pack.pack_id,
        path=path,
        actor=final.position,
        actor_action=final.action_type,
        actor_raise_size_pct=final.raise_size_pct,
        action_history=tuple(actions),
    )
