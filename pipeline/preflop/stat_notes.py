"""Deterministic "decision math" notes for the answer-explanation panel.

The admin panel shows a collapsible "Show the math" strip under each answer
explanation: one compact row per stat (pot odds, your equity, range
advantage, blockers, what you're up against), each with a short plain-English
phrase. This module produces those rows.

The phrases are written by PLAIN PYTHON from the Layer-5 facts -- never by the
LLM. That is the whole point: the number ("you have 47% vs 44%") fully
determines its framing ("a range disadvantage"), so the framing is a lookup,
not a judgement. Letting the model write these would re-open the exact door
the pipeline exists to keep shut (reversed blocker logic, hallucinated
numbers). Here the numbers come straight from the facts and the copy can't be
wrong -- and it's free (no API call) and identical every time.

Each :class:`StatNote` carries a ``value`` (the number, terse) and a ``note``
(the one-line phrase). ``build_stat_notes`` returns the panel rows; the
individual ``format_*`` helpers produce the matching standalone CSV columns
(so the column and the panel never disagree -- one computation).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from pipeline.preflop.fact_extractor import PreflopFacts

# --- framing thresholds (v1 -- tune against reviewer feedback) --------------
# Range advantage: hero range equity at/above ADVANTAGE => "advantage", at/
# below DISADVANTAGE => "disadvantage", between => "roughly even".
RANGE_ADVANTAGE = 0.53
RANGE_DISADVANTAGE = 0.47
# Hand-vs-range-average band: this-hand equity this much above/below hero's
# RANGE equity reads as "stronger / weaker than your average hand".
HAND_VS_RANGE_BAND = 0.03
# Villain range-width buckets (pct of dealt hands) for the "what you face" note.
RANGE_WIDTH_TIGHT = 6.0
RANGE_WIDTH_MODERATE = 15.0


@dataclass(frozen=True)
class StatNote:
    """One row of the "Show the math" panel."""

    key: str  # stable id, e.g. "pot_odds" (the app keys render order on this)
    label: str  # the row label, e.g. "Pot odds"
    value: str  # the terse number, e.g. "need 31%"
    note: str  # the deterministic one-line phrase


def _pct(x: float) -> str:
    """A [0,1] equity as a whole-number percent, e.g. 0.466 -> '47%'."""
    return f"{x:.0%}"


# --- standalone CSV-column formatters (one value each) ----------------------
def format_pct_or_blank(x: float | None) -> str:
    """``hero_equity`` / ``range_equity`` / ``pot_odds`` cell: '47%' or ''."""
    return _pct(x) if x is not None else ""


def format_blockers(blockers: dict[str, int]) -> str:
    """The ``blocker_combos`` cell: 'AKo:6, AKs:4, AA:3' (most blocked first).

    Empty string when hero blocks nothing (open spots, or no shared cards).
    """
    nonzero = [(c, n) for c, n in blockers.items() if n > 0]
    if not nonzero:
        return ""
    nonzero.sort(key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{c}:{n}" for c, n in nonzero)


def format_top_villain_combos(stats: object) -> str:
    """The ``top_villain_combos`` cell, e.g. 'AA, KK, AKs, QQ (~70% of 4.2%)'.

    ``stats`` is a ``VillainRangeStats`` (or None). Empty when there is no
    villain or no covering set.
    """
    if stats is None:
        return ""
    covering = getattr(stats, "top_combos_covering", ())
    if not covering:
        return ""
    cov = getattr(stats, "top_combos_coverage_pct", 0.0)
    width = getattr(stats, "pct_of_dealt_hands", 0.0)
    return f"{', '.join(covering)} (~{cov:.0f}% of {width:.1f}%)"


# --- the panel rows ---------------------------------------------------------
def _pot_odds_note(be: float) -> StatNote:
    # Just state the pot odds. Whether THIS hand should call is context the
    # answer explanation owns -- implied odds can make a sub-threshold call
    # correct -- so the panel never frames it as "equity needed to call".
    # No em dashes in any note: the team bans them in copy. (Zach, 6/26.)
    pct = _pct(be)
    return StatNote("pot_odds", "Pot odds", pct, f"Your pot odds here are {pct}.")


def _hero_equity_note(eq: float, range_eq: float | None) -> StatNote:
    pct = _pct(eq)
    if range_eq is None:
        note = f"Your hand has about {pct} equity against their range."
    elif eq - range_eq >= HAND_VS_RANGE_BAND:
        note = (
            f"Your hand has about {pct} equity against their range, a bit "
            f"stronger than your average hand here (your range is around "
            f"{_pct(range_eq)})."
        )
    elif eq - range_eq <= -HAND_VS_RANGE_BAND:
        note = (
            f"Your hand has about {pct} equity against their range, a bit "
            f"weaker than your average hand here (your range is around "
            f"{_pct(range_eq)})."
        )
    else:
        note = (
            f"Your hand has about {pct} equity against their range, about "
            f"average for your range here ({_pct(range_eq)})."
        )
    return StatNote("hero_equity", "Your equity", pct, note)


def _range_advantage_note(range_eq: float) -> StatNote:
    rpct, vpct = _pct(range_eq), _pct(1.0 - range_eq)
    if range_eq >= RANGE_ADVANTAGE:
        note = (
            f"As a whole, your hands are ahead of theirs here ({rpct} vs "
            f"{vpct}), a range advantage."
        )
    elif range_eq <= RANGE_DISADVANTAGE:
        note = (
            f"As a whole, your hands are behind theirs here ({rpct} vs "
            f"{vpct}), a range disadvantage."
        )
    else:
        note = (
            f"As a whole, your hands are about even with theirs here ({rpct} "
            f"vs {vpct})."
        )
    return StatNote(
        "range_advantage", "Range advantage", f"your range {rpct} vs theirs {vpct}", note
    )


def _blockers_note(blockers: dict[str, int]) -> StatNote | None:
    total = sum(n for n in blockers.values() if n > 0)
    if total <= 0:
        return None
    # Show every blocked class with its combo count -- the per-combo detail
    # is the point. format_blockers sorts most-blocked first.
    breakdown = format_blockers(blockers)
    note = (
        f"Your cards remove {total} combos from their range ({breakdown}), "
        "so they're a little less likely to hold those."
    )
    return StatNote("blockers", "Blockers", f"{total} combos", note)


def _villain_range_note(stats: object) -> StatNote | None:
    covering = getattr(stats, "top_combos_covering", ())
    if not covering:
        return None
    width = getattr(stats, "pct_of_dealt_hands", 0.0)
    hands = ", ".join(covering)
    if width < RANGE_WIDTH_TIGHT:
        shape = "tight"
    elif width < RANGE_WIDTH_MODERATE:
        shape = "fairly wide"
    else:
        shape = "wide"
    note = (
        f"Most of their range is these hands, a {shape} range "
        f"({width:.1f}% of all hands)."
    )
    return StatNote("villain_range", "You're up against", hands, note)


def build_stat_notes(facts: PreflopFacts) -> list[StatNote]:
    """The "Show the math" rows for one spot, in display order.

    Only includes a row when its underlying fact exists -- open/first-in
    spots (no villain) yield an empty list, and the panel hides itself.
    """
    notes: list[StatNote] = []
    if facts.break_even_equity is not None:
        notes.append(_pot_odds_note(facts.break_even_equity))
    if facts.hero_equity_vs_villain is not None:
        notes.append(
            _hero_equity_note(
                facts.hero_equity_vs_villain, facts.hero_range_equity_vs_villain
            )
        )
    if facts.hero_range_equity_vs_villain is not None:
        notes.append(_range_advantage_note(facts.hero_range_equity_vs_villain))
    blockers_note = _blockers_note(facts.blockers)
    if blockers_note is not None:
        notes.append(blockers_note)
    if facts.villain_stats is not None:
        villain_note = _villain_range_note(facts.villain_stats)
        if villain_note is not None:
            notes.append(villain_note)
    return notes


def stat_notes_to_json(notes: list[StatNote]) -> str:
    """Serialize panel rows for the ``stat_notes`` CSV column (compact JSON).

    Empty list -> ``''`` (no column noise / the panel renders nothing) rather
    than ``'[]'``.
    """
    if not notes:
        return ""
    return json.dumps([asdict(n) for n in notes], separators=(",", ":"))


def parse_stat_notes(cell: str) -> list[dict[str, str]]:
    """Read a ``stat_notes`` cell back into render-ready dicts (app side).

    Tolerant: blank or malformed cells yield an empty list, so a row without
    the math (or an older batch) simply shows no panel.
    """
    if not cell or not cell.strip():
        return []
    try:
        data = json.loads(cell)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]
