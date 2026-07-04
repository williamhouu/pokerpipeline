"""Tests for pipeline.preflop.stat_notes -- the deterministic decision-math
copy. The phrases are pure functions of the facts, so every threshold has a
pinned expected framing here. Facts are stubbed (SimpleNamespace) because the
module only reads a handful of scalar attributes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.preflop import stat_notes as sn  # noqa: E402


def _facts(**kw: object) -> object:
    base = dict(
        break_even_equity=None,
        hero_equity_vs_villain=None,
        hero_equity_vs_field=None,
        showdown_opponents=(),
        per_opponent_equity={},
        hero_range_equity_vs_villain=None,
        ev_gap_bb=None,
        blockers={},
        villain_stats=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _villain(covering=(), coverage_pct=0.0, pct_of_dealt_hands=0.0) -> object:
    return SimpleNamespace(
        top_combos_covering=tuple(covering),
        top_combos_coverage_pct=coverage_pct,
        pct_of_dealt_hands=pct_of_dealt_hands,
    )


def _note(facts: object, key: str) -> sn.StatNote | None:
    return next((n for n in sn.build_stat_notes(facts) if n.key == key), None)


# --- multi-way all-in equity note -------------------------------------------
def test_multiway_equity_note_shows_field_and_per_opponent_breakdown() -> None:
    """2+ players in the pot -> the equity row names WHO else is in, breaks
    down equity vs each one, and shows the field number as the headline. The
    fake carries no all-in in its (absent) history, so it reads as the
    not-all-in (squeeze-style) framing -> fold equity."""
    facts = _facts(
        hero_equity_vs_field=0.38,     # vs the whole field (the truth)
        per_opponent_equity={"BTN": 0.42, "HJ": 0.55},
        showdown_opponents=("BTN", "HJ"),
    )
    note = _note(facts, "hero_equity")
    assert note is not None
    assert "multi-way" in note.label.lower()
    assert note.value == "38%"             # field equity is the headline value
    assert "BTN, HJ" in note.note          # names who else is in the pot
    assert "BTN 42%" in note.note          # per-opponent breakdown
    assert "HJ 55%" in note.note
    assert "fold equity" in note.note.lower()   # not-all-in -> fold-equity frame


def test_field_equity_note_all_in_frames_beat_everyone() -> None:
    """All-in multi-way: the field number IS the decision -- beat everyone."""
    note = sn._hero_field_equity_note(
        0.33, {"BTN": 0.44, "HJ": 0.64}, ("BTN", "HJ"), is_all_in=True
    )
    assert note.value == "33%"
    assert "BTN 44%" in note.note and "HJ 64%" in note.note
    assert "beat all 2" in note.note.lower()
    assert "fold equity" not in note.note.lower()


def test_field_equity_note_not_all_in_frames_fold_equity() -> None:
    """Squeeze / non-all-in multi-way: field is context, edge is fold equity
    and equity vs whoever continues -- not beating everyone at showdown."""
    note = sn._hero_field_equity_note(
        0.33, {"BTN": 0.44, "HJ": 0.64}, ("BTN", "HJ"), is_all_in=False
    )
    assert note.value == "33%"
    assert "BTN 44%" in note.note
    assert "fold equity" in note.note.lower()
    assert "continues" in note.note.lower()


def test_heads_up_allin_uses_plain_equity_note() -> None:
    """One opponent -> not multi-way -> the standard heads-up equity row."""
    facts = _facts(
        hero_equity_vs_villain=0.44,
        hero_equity_vs_field=0.44,
        showdown_opponents=("BTN",),       # only one -> heads-up
    )
    note = _note(facts, "hero_equity")
    assert note is not None
    assert "multi-way" not in note.label.lower()
    assert note.value == "44%"


# --- column formatters ------------------------------------------------------
def test_format_pct_blank_and_value() -> None:
    assert sn.format_pct_or_blank(None) == ""
    assert sn.format_pct_or_blank(0.466) == "47%"  # rounds
    assert sn.format_pct_or_blank(0.0) == "0%"


def test_format_blockers_sorts_by_count_then_class() -> None:
    assert sn.format_blockers({}) == ""
    assert sn.format_blockers({"AA": 0}) == ""  # zero-count dropped
    assert sn.format_blockers({"AA": 2, "AKs": 3, "AKo": 3}) == "AKo:3, AKs:3, AA:2"


def test_format_top_villain_combos() -> None:
    assert sn.format_top_villain_combos(None) == ""
    assert sn.format_top_villain_combos(_villain()) == ""  # no covering set
    s = sn.format_top_villain_combos(
        _villain(covering=("AA", "KK", "AKs"), coverage_pct=70.4, pct_of_dealt_hands=4.18)
    )
    assert s == "AA, KK, AKs (~70% of 4.2%)"


# --- pot odds (just the price -- no call-quality verdict, no "need to call") -
def test_pot_odds_states_price_only() -> None:
    # Same note no matter the hand's equity. It never frames "equity needed
    # to call" -- implied odds can make a sub-threshold call correct.
    note = _note(
        _facts(
            break_even_equity=0.41,
            hero_equity_vs_villain=0.40,
            villain_stats=_villain(),
        ),
        "pot_odds",
    )
    assert note is not None and note.value == "41%"
    assert note.note == "Your pot odds here are 41%."
    for banned in ("profitable", "losing", "marginal", "breakeven", "to call", "need"):
        assert banned not in note.note.lower()


def test_pot_odds_same_note_without_equity() -> None:
    note = _note(
        _facts(break_even_equity=0.33, villain_stats=_villain()), "pot_odds"
    )
    assert note is not None and note.note == "Your pot odds here are 33%."


def test_pot_odds_suppressed_on_first_in_open() -> None:
    # A first-in open faces no bet: the EV engine still computes a break-even
    # number (the open's risk-vs-blinds price), but showing it as "Pot odds"
    # invents a calling price. villain_stats is None on opens -> no row.
    # (QC 2026-07-01: "Pot odds = 40%" on an A2s first-in open.)
    assert _note(_facts(break_even_equity=0.40), "pot_odds") is None


# --- your equity vs range average -------------------------------------------
def test_hero_equity_above_range_average() -> None:
    note = _note(
        _facts(hero_equity_vs_villain=0.55, hero_range_equity_vs_villain=0.48),
        "hero_equity",
    )
    assert note is not None and note.value == "55%" and "stronger" in note.note


def test_hero_equity_below_range_average() -> None:
    note = _note(
        _facts(hero_equity_vs_villain=0.40, hero_range_equity_vs_villain=0.50),
        "hero_equity",
    )
    assert note is not None and "weaker" in note.note


def test_hero_equity_at_range_average_and_no_range() -> None:
    at = _note(
        _facts(hero_equity_vs_villain=0.50, hero_range_equity_vs_villain=0.49),
        "hero_equity",
    )
    assert at is not None and "about average" in at.note
    bare = _note(_facts(hero_equity_vs_villain=0.50), "hero_equity")
    assert bare is not None
    assert bare.note == "Your hand has about 50% equity against their range."


# --- range advantage --------------------------------------------------------
def test_range_advantage_row_is_not_shown_to_players() -> None:
    """The standalone range-vs-range row was dropped from the panel (June
    2026): it conflated range equity with this hand's equity. The hand-level
    'Your equity' row stays; nothing carries a 'range advantage' verdict."""
    facts = _facts(hero_equity_vs_villain=0.45, hero_range_equity_vs_villain=0.40)
    notes = sn.build_stat_notes(facts)
    assert _note(facts, "range_advantage") is None
    assert not any("range advantage" in n.note.lower() for n in notes)
    # The hand-relative framing is retained on the hero_equity row.
    assert any(n.key == "hero_equity" for n in notes)


# --- blockers ---------------------------------------------------------------
def test_blockers_present_and_absent() -> None:
    present = _note(_facts(blockers={"AA": 2, "AKs": 3}), "blockers")
    assert present is not None and present.value == "5 combos"
    # The full per-class breakdown with counts, most-blocked first.
    assert "AKs:3, AA:2" in present.note
    assert _note(_facts(blockers={}), "blockers") is None
    assert _note(_facts(blockers={"AA": 0}), "blockers") is None  # zero -> skip


# --- what you're up against -------------------------------------------------
def test_villain_range_width_buckets() -> None:
    # Calibrated to preflop reality: an 8% UTG open is tight, ~20% (CO) is
    # fairly wide, 30%+ (BTN / blind battles / defense) is wide.
    tight = _note(_facts(villain_stats=_villain(("AA", "KK"), 80.0, 8.0)), "villain_range")
    assert tight is not None and "a tight range" in tight.note
    moderate = _note(_facts(villain_stats=_villain(("AA", "KK"), 60.0, 20.0)), "villain_range")
    assert moderate is not None and "a fairly wide range" in moderate.note
    wide = _note(_facts(villain_stats=_villain(("AA", "KK"), 60.0, 40.0)), "villain_range")
    assert wide is not None and "a wide range" in wide.note


def test_villain_range_absent_when_no_covering() -> None:
    assert _note(_facts(villain_stats=_villain()), "villain_range") is None


# --- assembly + serialization -----------------------------------------------
def test_notes_use_no_em_or_en_dashes() -> None:
    # The team bans em dashes in copy; stat notes follow the same rule
    # (covers em dash, en dash, and the "--" ASCII stand-in).
    facts = _facts(
        break_even_equity=0.41,
        hero_equity_vs_villain=0.40,
        hero_range_equity_vs_villain=0.32,
        blockers={"AA": 3, "AKo": 3},
        villain_stats=_villain(("AA", "KK", "AKs"), 70.0, 3.5),
    )
    notes = sn.build_stat_notes(facts)
    assert len(notes) == 4
    for n in notes:
        for dash in ("—", "–", "--"):
            assert dash not in n.note, f"{n.key} note has a dash: {n.note!r}"


def test_open_spot_yields_no_notes() -> None:
    assert sn.build_stat_notes(_facts()) == []


def test_no_ev_gap_note_in_panel() -> None:
    """The standalone EV row was removed June 2026 (panel + CSV). Even with an
    EV gap present, build_stat_notes never emits an 'ev_gap' row."""
    assert _note(_facts(ev_gap_bb=1.7), "ev_gap") is None
    assert _note(_facts(ev_gap_bb=0.0), "ev_gap") is None


def test_full_spot_order_and_round_trip() -> None:
    facts = _facts(
        break_even_equity=0.31,
        hero_equity_vs_villain=0.47,
        hero_range_equity_vs_villain=0.44,
        ev_gap_bb=0.8,
        blockers={"AA": 2},
        villain_stats=_villain(("AA", "KK", "AKs"), 70.0, 4.2),
    )
    notes = sn.build_stat_notes(facts)
    # 'ev_gap' is intentionally absent (removed June 2026).
    assert [n.key for n in notes] == [
        "pot_odds", "hero_equity", "blockers", "villain_range"
    ]
    blob = sn.stat_notes_to_json(notes)
    parsed = sn.parse_stat_notes(blob)
    assert [d["key"] for d in parsed] == [n.key for n in notes]
    assert all({"key", "label", "value", "note"} <= set(d) for d in parsed)


def test_serialization_empty_and_malformed() -> None:
    assert sn.stat_notes_to_json([]) == ""
    assert sn.parse_stat_notes("") == []
    assert sn.parse_stat_notes("   ") == []
    assert sn.parse_stat_notes("{not json") == []
    assert sn.parse_stat_notes('{"a":1}') == []  # not a list
