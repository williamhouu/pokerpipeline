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


def _pot_odds_note(be: float) -> StatNote:
    # State the price only; whether THIS hand should call is context the answer
    # explanation owns (implied odds can justify a sub-threshold call). No em
    # dashes -- the team bans them in copy.
    pct = _pct(be)
    return StatNote("pot_odds", "Pot odds", pct, f"Your pot odds here are {pct}.")


def _hero_equity_note(eq: float, range_eq: float, villain: str) -> StatNote:
    pct = _pct(eq)
    if eq - range_eq >= HAND_VS_RANGE_BAND:
        note = (
            f"Your hand has about {pct} equity against {villain} range, a bit "
            f"stronger than your average hand here (your range is around "
            f"{_pct(range_eq)})."
        )
    elif eq - range_eq <= -HAND_VS_RANGE_BAND:
        note = (
            f"Your hand has about {pct} equity against {villain} range, a bit "
            f"weaker than your average hand here (your range is around "
            f"{_pct(range_eq)})."
        )
    else:
        note = (
            f"Your hand has about {pct} equity against {villain} range, about "
            f"average for your range here ({_pct(range_eq)})."
        )
    return StatNote("hero_equity", "Your equity", pct, note)


def _currently_ahead_note(ahead: float, facing_bet: bool, villain: str) -> StatNote:
    # Whose hands hero beats RIGHT NOW (showdown equity, not draw equity). On a
    # facing-bet node the comparison set is the betting range -- mirror the data
    # block's wording so the panel and the SOLVER DATA never disagree.
    pct = _pct(ahead)
    whom = f"the hands {villain} betting" if facing_bet else f"{villain} range"
    note = f"Right now your hand beats about {pct} of {whom} at showdown."
    return StatNote("currently_ahead", "Currently ahead of", pct, note)


def _blockers_note(facts: PostflopFacts, villain: str) -> StatNote | None:
    # Only when hero's cards meaningfully remove villain's value OR bluffs (the
    # resolved verdict, never the LLM). Direction follows facts.blocker_effect.
    if facts.blocker_effect == "value" and facts.blocked_value_pct >= _BLOCKER_MIN_PCT:
        pct = f"{facts.blocked_value_pct:.0f}%"
        note = (
            f"Your cards remove about {pct} of the value hands from {villain} "
            "range, so they hold fewer of the hands that beat you."
        )
        return StatNote("blockers", "Blockers", pct, note)
    if facts.blocker_effect == "bluffs" and facts.blocked_bluff_pct >= _BLOCKER_MIN_PCT:
        pct = f"{facts.blocked_bluff_pct:.0f}%"
        note = (
            f"Your cards remove about {pct} of the bluffs from {villain} range, "
            "so there are fewer hands you beat that they would bet."
        )
        return StatNote("blockers", "Blockers", pct, note)
    return None


def _spr_note(spr: float) -> StatNote:
    return StatNote(
        "spr", "SPR", f"{spr:.1f}",
        f"The stack-to-pot ratio here is about {spr:.1f}.",
    )


def build_stat_notes(facts: PostflopFacts) -> list[StatNote]:
    """The "Show the math" rows for one postflop spot, in display order.

    Only includes a row when its underlying fact exists, so the app's panel
    hides any row it has no data for (same contract as the preflop builder).
    """
    notes: list[StatNote] = []
    villain = _villain_poss(facts)
    facing_bet = facts.break_even_equity is not None
    if facing_bet:
        notes.append(_pot_odds_note(facts.break_even_equity))  # type: ignore[arg-type]
    notes.append(
        _hero_equity_note(facts.hero_equity_vs_villain, facts.hero_range_equity, villain)
    )
    if facts.currently_ahead_pct > 0:
        notes.append(_currently_ahead_note(facts.currently_ahead_pct, facing_bet, villain))
    blockers_note = _blockers_note(facts, villain)
    if blockers_note is not None:
        notes.append(blockers_note)
    notes.append(_spr_note(facts.spr))
    return notes


def stat_notes_to_json(notes: list[StatNote]) -> str:
    """Serialize panel rows for the ``stat_notes`` CSV column (compact JSON).

    Empty list -> ``''`` (no panel) rather than ``'[]'`` -- same as preflop.
    """
    if not notes:
        return ""
    return json.dumps([asdict(n) for n in notes], separators=(",", ":"))


__all__ = ["StatNote", "build_stat_notes", "stat_notes_to_json"]
