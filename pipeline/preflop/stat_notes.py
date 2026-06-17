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
# Hand-vs-range-average band: this-hand equity this much above/below hero's
# RANGE equity reads as "stronger / weaker than your average hand".
HAND_VS_RANGE_BAND = 0.03
# Villain range-width buckets (pct of dealt hands) for the "what you face" note.
# Calibrated to preflop reality: a 3-bet range is ~5%, UTG opens ~8%, CO ~20%,
# BTN/blinds ~30-45%. So < 15% reads as tight, 15-30% fairly wide, 30%+ wide.
# (Earlier 6/15 cutoffs called an 8% UTG open "fairly wide", which is wrong.)
RANGE_WIDTH_TIGHT = 15.0
RANGE_WIDTH_MODERATE = 30.0


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


def _villain_poss(stats: object) -> str:
    """Possessive name for the opponent we're comparing against, e.g.
    "UTG+1's". The villain is the most-recent raiser, so in a multiway pot
    (UTG opens, UTG+1 3-bets, hero on the button) this makes explicit that the
    equity and range numbers are measured against UTG+1, not UTG. Falls back to
    "their" when no position is recorded.
    """
    pos = getattr(stats, "position", "") or ""
    return f"{pos}'s" if pos else "their"


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
    # No em dashes in any note: the team bans them in copy. (the team, 6/26.)
    pct = _pct(be)
    return StatNote("pot_odds", "Pot odds", pct, f"Your pot odds here are {pct}.")


def _hero_equity_note(eq: float, range_eq: float | None, villain: str = "their") -> StatNote:
    pct = _pct(eq)
    if range_eq is None:
        note = f"Your hand has about {pct} equity against {villain} range."
    elif eq - range_eq >= HAND_VS_RANGE_BAND:
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


def _hero_field_equity_note(
    field_eq: float,
    per_opponent: dict[str, float],
    opponents: tuple[str, ...],
    is_all_in: bool,
) -> StatNote:
    """Multi-way equity row: spell out equity vs WHOM and how it breaks down --
    your equity against each player alone, then against the whole field.

    The framing depends on whether the money is all-in. ALL-IN: the field
    number IS the decision (you must beat everyone at showdown). NOT all-in
    (a squeeze / 3-bet with a caller): the field number is just "if it all
    goes in" -- your edge is fold equity and your equity vs whoever continues,
    so the field-at-showdown figure is context, not the whole story. Plain
    English so a reviewer sees why a 'mixed' multi-way spot isn't a coin-flip.
    """
    n = len(opponents)
    pct = _pct(field_eq)
    seats = ", ".join(opponents)
    breakdown = ", ".join(
        f"{pos} {_pct(per_opponent.get(pos, 0.0))}" for pos in opponents
    )
    head = (
        f"You're up against {n} players in this pot ({seats}). Against each "
        f"one alone your equity is {breakdown}."
    )
    if is_all_in:
        tail = (
            f" To win you have to beat all {n} on the same board, so your real "
            f"equity is about {pct} -- lower than against any one of them, "
            "because every extra player is another hand that can beat you."
        )
    else:
        tail = (
            f" If all the money went in you'd have about {pct} to win the pot, "
            "but the pot isn't all-in here -- your edge is also fold equity and "
            "how the hand plays out, so the number that matters most is your "
            "equity against whoever continues (usually the original raiser), "
            "not beating everyone at showdown."
        )
    return StatNote("hero_equity", "Your equity (multi-way)", pct, head + tail)


def _blockers_note(blockers: dict[str, int], villain: str = "their") -> StatNote | None:
    total = sum(n for n in blockers.values() if n > 0)
    if total <= 0:
        return None
    # Show every blocked class with its combo count -- the per-combo detail
    # is the point. format_blockers sorts most-blocked first.
    breakdown = format_blockers(blockers)
    note = (
        f"Your cards remove {total} combos from {villain} range ({breakdown}), "
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
    # Name the seat we're up against (the most-recent raiser) so a multiway
    # spot makes clear WHICH opponent this range belongs to.
    pos = getattr(stats, "position", "") or ""
    poss = _villain_poss(stats)
    label = f"You're up against {pos}" if pos else "You're up against"
    note = (
        f"Most of {poss} range is these hands, a {shape} range "
        f"({width:.1f}% of all hands)."
    )
    return StatNote("villain_range", label, hands, note)


def build_stat_notes(facts: PreflopFacts) -> list[StatNote]:
    """The "Show the math" rows for one spot, in display order.

    Only includes a row when its underlying fact exists -- open/first-in
    spots (no villain) yield an empty list, and the panel hides itself.
    """
    notes: list[StatNote] = []
    # Possessive name for the opponent (the most-recent raiser) so multiway
    # spots make clear which seat the numbers are measured against.
    villain = _villain_poss(facts.villain_stats) if facts.villain_stats else "their"
    if facts.break_even_equity is not None:
        notes.append(_pot_odds_note(facts.break_even_equity))
    # Multi-way all-in: show the FIELD equity (beat everyone) as the headline,
    # since the heads-up number overstates hero's real chance. Otherwise the
    # standard heads-up equity row.
    if facts.hero_equity_vs_field is not None and len(facts.showdown_opponents) >= 2:
        # All-in or not changes the framing (decision vs context). Read it
        # defensively from the action history (duck-typed so test fakes that
        # carry no spot simply read as not-all-in).
        history = getattr(
            getattr(getattr(facts, "spot", None), "node", None),
            "history_before",
            (),
        )
        is_all_in = any(
            getattr(a.action_type, "name", "") == "ALL_IN" for a in (history or ())
        )
        notes.append(
            _hero_field_equity_note(
                facts.hero_equity_vs_field,
                facts.per_opponent_equity,
                facts.showdown_opponents,
                is_all_in,
            )
        )
    elif facts.hero_equity_vs_villain is not None:
        notes.append(
            _hero_equity_note(
                facts.hero_equity_vs_villain,
                facts.hero_range_equity_vs_villain,
                villain,
            )
        )
    # NOTE: the standalone range-vs-range "Range advantage" row was dropped
    # from the player panel (June 2026) -- it answers a RANGE-level question
    # next to hand-level stats, which conflates "my hand's equity" with "my
    # range's equity" (a premium hand can sit in a range that's behind). The
    # hand-relative framing in _hero_equity_note ("stronger than your average
    # hand here") keeps the useful part; the `range_equity` CSV column still
    # carries the number for QA/analytics.
    # (The standalone "EV" row was removed June 2026 -- the team asked to take
    # the single EV-gap number out of the panel + the CSV stat_notes JSON. The
    # per-action EV column action_ev_bb, charted on the Review page, is the EV
    # signal now; the gap is still computed internally for difficulty + the
    # worthiness gate.)
    blockers_note = _blockers_note(facts.blockers, villain)
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
