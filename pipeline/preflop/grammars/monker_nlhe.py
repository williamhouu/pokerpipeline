"""Filename-grammar parser for MonkerSolver-export NLHE range packs.

Monker packs (e.g. the 9-max 100bb pack under ``nlhe9_ranges/``) name each
`.rng` file by its action sequence alone -- dot-separated action tokens,
no seat names::

    40120.rng                  UTG opens (raise 120% pot = 4bb)
    40120.0.0.0.0.0.0.0.1.rng  UTG opens, folds to BB, BB calls
    0.0.0.0.0.0.0.1.rng        folds to SB, SB limps (completes)

Seats are implied by Monker's rotation convention: positions act in
preflop order (:func:`pipeline.action_history.preflop_order` for the
pack's table size); a caller/raiser rotates to the back of the queue (it
can act again on a re-raise), a folder or all-in player leaves the
action. Same convention as the PLO pack (`pipeline.plo.pack`), but this
parser emits the pipeline's canonical seat names (``UTG+1``, ``BTN``)
directly, so nothing downstream ever sees Monker's ``UTG1``/``BU``
dialect.

Action tokens:

  * ``0`` -- fold
  * ``1`` -- call (a first-in call from the SB is a limp; the BB
    "calling" when there is no raise outstanding is a check -- both are
    relabelled downstream, the grammar just reports CALL)
  * ``3`` -- all-in jam
  * ``5`` -- min-raise. Carries the :data:`~pipeline.preflop.grammars.
    types.MIN_RAISE_PCT` sentinel rather than a pot fraction: there is no
    percent to apply, the bb size is the minimum legal raise (twice the
    current bet), resolved in ``pipeline.preflop.action_history``. The
    short-stack 6-max packs open with this (verified: ``5`` is *only* ever
    the open, never a re-raise, so in practice it is the 2bb open).
  * ``14`` -- the BB iso-raise over a single SB limp. It appears *only* in
    the ``0.0.0.0.1.14...`` BvB-limped node (12 files per pack, identical
    in the 20bb and 30bb packs). Decoded via the fold-EV anchor: BB commits
    2.5bb into the 2bb limped pot = 75% pot under Monker's pot-relative
    rule, so it is emitted as a 75%-pot raise and resolves to 2.5bb through
    the normal size walk. ``scripts/audit_nlhe6_pack.py`` locks this size.
  * ``40<pct>`` -- a raise sized as ``<pct>``% of pot per Monker's
    pot-relative rule (the raise-to-bb conversion is downstream, in
    ``pipeline.preflop.action_history``). e.g. ``40120`` = 120% pot.

Any other bare-integer token (``2``, ``15``, ...) is still rejected: the
known packs don't use it, and silently mis-sizing a raise would be worse
than failing loudly until a future pack's token is decoded and added here.

Each `.rng` file is the range with which the *last* actor takes the
*last* action; the actions before it are the history that reached the
decision. That matches the Ryan-pack convention, so the node enumerator
groups Monker files into decision nodes with no special casing.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.action_history import preflop_order
from pipeline.preflop.grammars.types import (
    MIN_RAISE_PCT,
    ParsedAction,
    ParsedRangeFile,
    PreflopActionType,
)
from pipeline.preflop.pack import PreflopPack

# The BB iso-raise token (``14``), decoded as 75%-pot. See module docstring;
# its size is verified in scripts/audit_nlhe6_pack.py.
_BB_ISO_OVER_LIMP_PCT = 75.0

GRAMMAR_NAME = "monker_nlhe"

_SPECIAL_TOKENS: dict[str, PreflopActionType] = {
    "0": PreflopActionType.FOLD,
    "1": PreflopActionType.CALL,
    "3": PreflopActionType.ALL_IN,
}

# A raised/called seat keeps acting (it can face a re-raise); a folded or
# all-in seat is out of the action.
_CONTINUES = (PreflopActionType.CALL, PreflopActionType.RAISE)


def _decode_token(token: str, filename: str) -> tuple[PreflopActionType, float | None]:
    """Map one Monker action token to ``(ActionType, optional raise %)``."""
    special = _SPECIAL_TOKENS.get(token)
    if special is not None:
        return special, None
    if token == "5":
        # Min-raise (twice the current bet). No pot fraction -- the size is
        # resolved relative to the running bet in resolve_preflop_history.
        return PreflopActionType.RAISE, MIN_RAISE_PCT
    if token == "14":
        # BB iso-raise over a single SB limp == 75% pot (see module docstring).
        return PreflopActionType.RAISE, _BB_ISO_OVER_LIMP_PCT
    if token.startswith("40") and token[2:].isdigit() and int(token[2:]) > 0:
        return PreflopActionType.RAISE, float(int(token[2:]))
    raise ValueError(
        f"monker_nlhe: unrecognised action token {token!r} in {filename!r}. "
        "Expected 0 (fold), 1 (call), 3 (all-in), 5 (min-raise), 14 (BB "
        "iso over a limp), or 40<pct> (raise)."
    )


def parse(filename: Path, pack: PreflopPack) -> ParsedRangeFile:
    """Parse one Monker `.rng` filename into a normalized ``ParsedRangeFile``.

    Args:
        filename: Path to a ``.rng`` file inside the pack. Only the stem
            is used (Monker packs are flat -- no per-position folders).
        pack: The pack this file belongs to. ``pack.table_size`` decides
            the seat rotation (9-max: UTG, UTG+1, UTG+2, LJ, HJ, CO,
            BTN, SB, BB).

    Returns:
        Normalized ``ParsedRangeFile``; ``action_history`` is the full
        token chain including the actor's own final action, matching the
        ryan_pack convention.

    Raises:
        ValueError: if the stem is empty, contains an unrecognised
            token, or has more actions than seats left to act.
    """
    stem = filename.stem
    queue = list(preflop_order(pack.table_size))
    actions: list[ParsedAction] = []
    for token in stem.split("."):
        if not queue:
            raise ValueError(
                f"monker_nlhe: action token {token!r} but no seat left "
                f"to act in {filename.name!r}"
            )
        position = queue.pop(0)
        action_type, raise_pct = _decode_token(token, filename.name)
        actions.append(
            ParsedAction(
                position=position,
                action_type=action_type,
                raise_size_pct=raise_pct,
            )
        )
        if action_type in _CONTINUES:
            queue.append(position)
    if not actions:
        raise ValueError(f"monker_nlhe: empty node path: {filename.name!r}")

    final = actions[-1]
    return ParsedRangeFile(
        pack_id=pack.pack_id,
        path=filename,
        actor=final.position,
        actor_action=final.action_type,
        actor_raise_size_pct=final.raise_size_pct,
        action_history=tuple(actions),
    )
