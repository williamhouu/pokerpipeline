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
        hero_range_equity_vs_villain=None,
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


# --- pot odds ---------------------------------------------------------------
def test_pot_odds_profitable() -> None:
    note = _note(_facts(break_even_equity=0.31, hero_equity_vs_villain=0.47), "pot_odds")
    assert note is not None and note.value == "need 31%"
    assert "profitable" in note.note and "16%" in note.note


def test_pot_odds_losing() -> None:
    note = _note(_facts(break_even_equity=0.40, hero_equity_vs_villain=0.30), "pot_odds")
    assert note is not None and "short" in note.note and "losing" in note.note


def test_pot_odds_breakeven_band() -> None:
    note = _note(_facts(break_even_equity=0.40, hero_equity_vs_villain=0.41), "pot_odds")
    assert note is not None and "breakeven" in note.note


def test_pot_odds_without_equity() -> None:
    note = _note(_facts(break_even_equity=0.33), "pot_odds")
    assert note is not None and note.note == "You need 33% equity to call profitably."


# --- your equity vs range average -------------------------------------------
def test_hero_equity_above_range_average() -> None:
    note = _note(
        _facts(hero_equity_vs_villain=0.55, hero_range_equity_vs_villain=0.48),
        "hero_equity",
    )
    assert note is not None and note.value == "55%" and "above" in note.note


def test_hero_equity_below_range_average() -> None:
    note = _note(
        _facts(hero_equity_vs_villain=0.40, hero_range_equity_vs_villain=0.50),
        "hero_equity",
    )
    assert note is not None and "below" in note.note


def test_hero_equity_at_range_average_and_no_range() -> None:
    at = _note(
        _facts(hero_equity_vs_villain=0.50, hero_range_equity_vs_villain=0.49),
        "hero_equity",
    )
    assert at is not None and "about at" in at.note
    bare = _note(_facts(hero_equity_vs_villain=0.50), "hero_equity")
    assert bare is not None and bare.note == "Your hand has 50% equity against their range."


# --- range advantage --------------------------------------------------------
def test_range_advantage_disadvantage_even() -> None:
    adv = _note(_facts(hero_range_equity_vs_villain=0.58), "range_advantage")
    assert adv is not None and "advantage" in adv.note and "ahead" in adv.note
    dis = _note(_facts(hero_range_equity_vs_villain=0.40), "range_advantage")
    assert dis is not None and "disadvantage" in dis.note
    even = _note(_facts(hero_range_equity_vs_villain=0.50), "range_advantage")
    assert even is not None and "roughly even" in even.note


# --- blockers ---------------------------------------------------------------
def test_blockers_present_and_absent() -> None:
    present = _note(_facts(blockers={"AA": 2, "AKs": 3}), "blockers")
    assert present is not None and present.value == "remove 5 combos"
    assert "AKs:3, AA:2" in present.note
    assert _note(_facts(blockers={}), "blockers") is None
    assert _note(_facts(blockers={"AA": 0}), "blockers") is None  # zero -> skip


# --- what you're up against -------------------------------------------------
def test_villain_range_width_buckets() -> None:
    tight = _note(_facts(villain_stats=_villain(("AA", "KK"), 80.0, 3.0)), "villain_range")
    assert tight is not None and "tight, value-heavy" in tight.note
    moderate = _note(_facts(villain_stats=_villain(("AA", "KK"), 60.0, 10.0)), "villain_range")
    assert moderate is not None and "moderately wide" in moderate.note
    wide = _note(_facts(villain_stats=_villain(("AA", "KK"), 60.0, 22.0)), "villain_range")
    assert wide is not None and "a wide range" in wide.note


def test_villain_range_absent_when_no_covering() -> None:
    assert _note(_facts(villain_stats=_villain()), "villain_range") is None


# --- assembly + serialization -----------------------------------------------
def test_open_spot_yields_no_notes() -> None:
    assert sn.build_stat_notes(_facts()) == []


def test_full_spot_order_and_round_trip() -> None:
    facts = _facts(
        break_even_equity=0.31,
        hero_equity_vs_villain=0.47,
        hero_range_equity_vs_villain=0.44,
        blockers={"AA": 2},
        villain_stats=_villain(("AA", "KK", "AKs"), 70.0, 4.2),
    )
    notes = sn.build_stat_notes(facts)
    assert [n.key for n in notes] == [
        "pot_odds", "hero_equity", "range_advantage", "blockers", "villain_range"
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
