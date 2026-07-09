"""Deterministic "decision math" notes for the postflop answer panel.

The app shows a collapsible "Show the math" strip under each answer: one compact
row per stat (pot odds, your equity, what you're currently ahead of, blockers,
SPR), each with a short plain-English phrase. This module produces those rows
for the POSTFLOP path so the panel renders on postflop questions and on the
preflop-entry / play-through legs (which the postflop writer emits) -- exactly
the same `stat_notes` JSON the preflop writer already produces, so the app
parses both identically.

This is the postflop ANALOGUE of :mod:`pipeline.preflop.stat_notes`, kept here
(not imported) so the postflop package stays self-contained per the package
contract -- it shares only the JSON SHAPE (`key`/`label`/`value`/`note`), which
is the app's actual interface. As there, the phrases are written by PLAIN PYTHON
from the Layer-5 facts, never the LLM: the number fully determines its framing,
so the copy can't be wrong and is identical every run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from pipeline.postflop.facts import PostflopFacts

# Hand-vs-range band: this-hand equity this far above/below the actor's RANGE
# equity reads as "stronger / weaker than your average hand here".
HAND_VS_RANGE_BAND = 0.03
# Blocker removal this small is noise -- mirror facts.compute_blocker_decomposition.
_BLOCKER_MIN_PCT = 5.0


@dataclass(frozen=True)
class StatNote:
    """One row of the "Show the math" panel (same shape as the preflop note)."""

    key: str   # stable id the app keys render order on, e.g. "pot_odds"
    label: str  # row label, e.g. "Pot odds"
    value: str  # the terse number, e.g. "47%"
    note: str   # the deterministic one-line phrase


def _pct(x: float) -> str:
    """A [0,1] fraction as a whole-number percent, e.g. 0.466 -> '47%'."""
    return f"{x:.0%}"


def _villain_poss(facts: PostflopFacts) -> str:
    """Possessive for the opponent, e.g. "BB's"; "their" if no seat recorded."""
    pos = facts.villain_position or ""
    return f"{pos}'s" if pos else "their"


def _pot_odds_note(
    be: float,
    pot_bb: float | None = None,
    call_bb: float | None = None,
    fmt=None,
) -> StatNote:
    # The subtext is the WRITTEN-OUT equation with each number labeled (team,
    # July 2026, "Format B"): "call 4bb ÷ (pot 15bb + call 4bb) = 21%".
    # Amounts are EXACT (never the 0.5bb display grid) so the arithmetic
    # always reproduces the printed percentage. Whether THIS hand should call
    # is context the answer explanation owns (implied odds can justify a
    # sub-threshold call). No em dashes -- the team bans them in copy.
    pct = _pct(be)
    if pot_bb is not None and call_bb is not None and fmt is not None:
        note = (
            f"call {fmt(call_bb)} ÷ (pot {fmt(pot_bb)} + "
            f"call {fmt(call_bb)}) = {pct}."
        )
    else:
        note = f"Your pot odds here are {pct}."
    return StatNote("pot_odds", "Pot odds", pct, note)


def _hero_equity_note(
    eq: float, range_eq: float, villain: str, hand_label: str = ""
) -> StatNote:
    pct = _pct(eq)
    # E1 (team, July 2026): name the specific cards so the number reads as a
    # fact about THIS holding.
    subject = f"Your hand, {hand_label}," if hand_label else "Your hand"
    if eq - range_eq >= HAND_VS_RANGE_BAND:
        note = (
            f"{subject} has about {pct} equity against {villain} range, a bit "
            f"stronger than your average hand here (your range is around "
            f"{_pct(range_eq)})."
        )
    elif eq - range_eq <= -HAND_VS_RANGE_BAND:
        note = (
            f"{subject} has about {pct} equity against {villain} range, a bit "
            f"weaker than your average hand here (your range is around "
            f"{_pct(range_eq)})."
        )
    else:
        note = (
            f"{subject} has about {pct} equity against {villain} range, about "
            f"average for your range here ({_pct(range_eq)})."
        )
    return StatNote("hero_equity", "Your equity", pct, note)


def _currently_ahead_note(
    ahead: float, facing_bet: bool, villain: str, tied: float = 0.0
) -> StatNote:
    # Whose hands hero beats RIGHT NOW (showdown equity, not draw equity). On a
    # facing-bet node the comparison set is the betting range -- mirror the data
    # block's wording so the panel and the SOLVER DATA never disagree.
    pct = _pct(ahead)
    whom = f"the hands {villain} betting" if facing_bet else f"{villain} range"
    note = f"Right now your hand beats about {pct} of {whom} at showdown."
    if tied >= 0.05:  # noqa: PLR2004 -- mirrors the data block's tie threshold
        note += (
            f" You also tie (chop the pot) with about {_pct(tied)} more, and "
            "a chop only gets your money back, it does not win the pot."
        )
    return StatNote("currently_ahead", "Currently ahead of", pct, note)


def _blockers_note(facts: PostflopFacts, villain: str) -> StatNote | None:
    # ALIGNMENT INVARIANT (team, July 2026): this row fires on EXACTLY the
    # condition that puts the BLOCKERS line in the LLM's SOLVER DATA block
    # (facts.blocker_effect resolved non-neutral) -- so whenever the prose is
    # allowed to mention blockers, the panel shows the numbers behind it, and
    # a reviewer never has to take a blocker claim on faith. (The old gate
    # compared the FRACTION blocked_value_pct to 5.0, a percent, so this row
    # never rendered at all -- a dead row since birth.)
    if facts.blocker_effect == "value":
        pct = f"{facts.blocked_value_pct:.0%}"
        note = (
            f"Your cards remove about {pct} of the value hands from {villain} "
            "range, so they hold fewer of the hands that beat you."
        )
        return StatNote("blockers", "Blockers", pct, note)
    if facts.blocker_effect == "bluffs":
        pct = f"{facts.blocked_bluff_pct:.0%}"
        note = (
            f"Your cards remove about {pct} of the bluffs from {villain} range, "
            "so there are fewer hands you beat that they would bet."
        )
        return StatNote("blockers", "Blockers", pct, note)
    return None


def _spr_note(
    spr: float,
    stack_bb: float | None = None,
    pot_bb: float | None = None,
    fmt=None,
) -> StatNote:
    # The subtext is the WRITTEN-OUT equation, like pot odds (team, July
    # 2026): "stack 121bb ÷ pot 6.7bb = 18.1". The amounts are the SAME
    # numbers node.spr divides (effective stack / pot including any bet
    # faced), exact and never display-grid-rounded, so the arithmetic always
    # reproduces the printed ratio.
    if stack_bb is not None and pot_bb is not None and pot_bb > 0 and fmt is not None:
        note = (
            f"stack {fmt(stack_bb)} ÷ pot {fmt(pot_bb)} = {spr:.1f}. The "
            "stack number is whichever player has less money left, because "
            "that is the most either of you can still put in."
        )
    else:  # fakes / missing geometry
        note = f"The stack-to-pot ratio here is about {spr:.1f}."
    return StatNote("spr", "SPR", f"{spr:.1f}", note)


def build_stat_notes(
    facts: PostflopFacts,
    *,
    display_in_bb: bool = True,
    bb_in_dollars: float | None = None,
) -> list[StatNote]:
    """The "Show the math" rows for one postflop spot, in display order.

    Only includes a row when its underlying fact exists, so the app's panel
    hides any row it has no data for (same contract as the preflop builder).
    ``display_in_bb`` / ``bb_in_dollars`` pick the unit for the pot-odds
    equation's amounts so the panel matches the Question prose's unit.
    """
    from pipeline.action_history import format_card  # noqa: PLC0415
    from pipeline.bb_display import exact_amount_str  # noqa: PLC0415

    def _amt(x: float) -> str:
        return exact_amount_str(
            x, display_in_bb=display_in_bb, bb_in_dollars=bb_in_dollars
        )

    # The specific cards, e.g. "K<spade>J<spade>" (duck-typed so test fakes
    # without a spot simply omit the name).
    combo = str(getattr(getattr(facts, "spot", None), "hero_combo", "") or "")
    hand_label = (
        format_card(combo[:2]) + format_card(combo[2:]) if len(combo) == 4 else ""  # noqa: PLR2004
    )
    notes: list[StatNote] = []
    villain = _villain_poss(facts)
    facing_bet = facts.break_even_equity is not None
    if facing_bet:
        notes.append(_pot_odds_note(
            facts.break_even_equity,  # type: ignore[arg-type]
            facts.pot_bb,
            facts.to_call_bb,
            _amt,
        ))
    notes.append(
        _hero_equity_note(
            facts.hero_equity_vs_villain, facts.hero_range_equity, villain,
            hand_label,
        )
    )
    if facts.currently_ahead_pct > 0:
        notes.append(_currently_ahead_note(
            facts.currently_ahead_pct, facing_bet, villain,
            getattr(facts, "currently_tied_pct", 0.0) or 0.0,
        ))
    blockers_note = _blockers_note(facts, villain)
    if blockers_note is not None:
        notes.append(blockers_note)
    _node = getattr(getattr(facts, "spot", None), "node", None)
    notes.append(_spr_note(
        facts.spr,
        getattr(_node, "effective_stack_bb", None),
        facts.pot_bb,
        _amt,
    ))
    return notes


def stat_notes_to_json(notes: list[StatNote]) -> str:
    """Serialize panel rows for the ``stat_notes`` CSV column (compact JSON).

    Empty list -> ``''`` (no panel) rather than ``'[]'`` -- same as preflop.
    """
    if not notes:
        return ""
    return json.dumps([asdict(n) for n in notes], separators=(",", ":"))


__all__ = ["StatNote", "build_stat_notes", "stat_notes_to_json"]
