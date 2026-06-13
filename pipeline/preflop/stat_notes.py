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
# RANGE equity reads as "above / below your range's average".
HAND_VS_RANGE_BAND = 0.03
# Pot-odds edge band: |hero_equity - break_even| under this is "breakeven".
POT_ODDS_BAND = 0.02
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
def _pot_odds_note(be: float, eq: float | None) -> StatNote:
    value = f"need {_pct(be)}"
    if eq is None:
        note = f"You need {_pct(be)} equity to call profitably."
        return StatNote("pot_odds", "Pot odds", value, note)
    edge = eq - be
    if edge >= POT_ODDS_BAND:
        note = (
            f"You have {_pct(eq)} vs their range -- {_pct(edge)} over the "
            f"{_pct(be)} you need, a profitable call on raw equity (before "
            "position and realization)."
        )
    elif edge <= -POT_ODDS_BAND:
        note = (
            f"You have {_pct(eq)} vs their range -- {_pct(-edge)} short of the "
            f"{_pct(be)} you need, a losing call on raw equity."
        )
    else:
        note = (
            f"Your {_pct(eq)} is right at the {_pct(be)} you need -- a "
            "breakeven, marginal call."
        )
    return StatNote("pot_odds", "Pot odds", value, note)


def _hero_equity_note(eq: float, range_eq: float | None) -> StatNote:
    value = _pct(eq)
    if range_eq is None:
        note = f"Your hand has {_pct(eq)} equity against their range."
        return StatNote("hero_equity", "Your equity", value, note)
    diff = eq - range_eq
    if diff >= HAND_VS_RANGE_BAND:
        note = (
            f"Your hand ({_pct(eq)}) is above your range's {_pct(range_eq)} "
            "average here -- one of your stronger holdings in this spot."
        )
    elif diff <= -HAND_VS_RANGE_BAND:
        note = (
            f"Your hand ({_pct(eq)}) is below your range's {_pct(range_eq)} "
            "average here -- toward the weaker end of what you continue with."
        )
    else:
        note = (
            f"Your hand ({_pct(eq)}) sits about at your range's "
            f"{_pct(range_eq)} average here."
        )
    return StatNote("hero_equity", "Your equity", value, note)


def _range_advantage_note(range_eq: float) -> StatNote:
    villain = 1.0 - range_eq
    value = f"your range {_pct(range_eq)} vs theirs {_pct(villain)}"
    if range_eq >= RANGE_ADVANTAGE:
        note = (
            f"Your whole range is ahead ({_pct(range_eq)} vs {_pct(villain)}) "
            "-- a range advantage, so you can apply pressure."
        )
    elif range_eq <= RANGE_DISADVANTAGE:
        note = (
            f"Your whole range is behind ({_pct(range_eq)} vs {_pct(villain)}) "
            "-- a range disadvantage, which is why you fold and call more "
            "than you raise."
        )
    else:
        note = (
            f"The ranges are roughly even ({_pct(range_eq)} vs "
            f"{_pct(villain)}) -- neither side has a clear edge."
        )
    return StatNote("range_advantage", "Range advantage", value, note)


def _blockers_note(blockers: dict[str, int]) -> StatNote | None:
    total = sum(n for n in blockers.values() if n > 0)
    if total <= 0:
        return None
    breakdown = format_blockers(blockers)
    note = (
        f"Your cards remove {total} combos from their range ({breakdown}) -- "
        "fewer of those hands left for them to have."
    )
    return StatNote("blockers", "Blockers", f"remove {total} combos", note)


def _villain_range_note(stats: object) -> StatNote | None:
    covering = getattr(stats, "top_combos_covering", ())
    if not covering:
        return None
    cov = getattr(stats, "top_combos_coverage_pct", 0.0)
    width = getattr(stats, "pct_of_dealt_hands", 0.0)
    hands = ", ".join(covering)
    if width < RANGE_WIDTH_TIGHT:
        shape = "a tight, value-heavy range"
    elif width < RANGE_WIDTH_MODERATE:
        shape = "a moderately wide range"
    else:
        shape = "a wide range"
    note = (
        f"You're mainly up against {hands} (~{cov:.0f}% of a "
        f"{width:.1f}%-of-hands range) -- {shape}."
    )
    return StatNote("villain_range", "You're up against", hands, note)


def build_stat_notes(facts: PreflopFacts) -> list[StatNote]:
    """The "Show the math" rows for one spot, in display order.

    Only includes a row when its underlying fact exists -- open/first-in
    spots (no villain) yield an empty list, and the panel hides itself.
    """
    notes: list[StatNote] = []
    if facts.break_even_equity is not None:
        notes.append(
            _pot_odds_note(facts.break_even_equity, facts.hero_equity_vs_villain)
        )
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
