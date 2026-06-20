"""Tests for the deterministic neutral-credit rule (pipeline.neutral_credit).

The 20-point rule: an option earns credit when the hand's real solver
frequency is within 20 points of what the option claims -- "Always X" needs
X >= 80%, "Mostly X" / bare "X" needs X >= 20%. The full-credit option is
excluded; every other creditable option is neutral. These cases are the ones
worked through with the team, so the test file doubles as the spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.format_writer import CSV_COLUMNS  # noqa: E402
from pipeline.neutral_credit import (  # noqa: E402
    format_neutral_credit,
    neutral_credit_options,
)
from pipeline.plo.format_writer import PLO_CSV_COLUMNS  # noqa: E402
from pipeline.postflop.format_writer import POSTFLOP_CSV_COLUMNS  # noqa: E402
from pipeline.preflop.format_writer import PREFLOP_CSV_COLUMNS  # noqa: E402

# The GTO 4-rung spectrum for a Call/Raise spot (less-aggressive first), as
# pipeline.*.options.build_options_gto emits it.
_GTO_CALL_RAISE = ["Always Call", "Mostly Call", "Mostly Raise", "Always Raise"]


# --- GTO style: the symmetric within-20-points rule --------------------------
def test_gto_tossup_credits_reverse_lean_not_the_always_rung():
    # Call 65 / Raise 35: "Mostly Raise" is fine (35 >= 20); "Always Call" is a
    # MISTAKE because 65 < 80 -- "always" overstates a clearly mixed spot.
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Mostly Call", {"Call": 0.65, "Raise": 0.35}
    )
    assert neutral == ["Mostly Raise"]


def test_gto_near_pure_credits_always_rung_not_the_minority_lean():
    # Call 85 / Raise 15: "Always Call" is fine (85 >= 80); both raise rungs are
    # mistakes (15 < 20). The 15% line gets no credit -- the team's anchor case.
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Mostly Call", {"Call": 0.85, "Raise": 0.15}
    )
    assert neutral == ["Always Call"]


def test_gto_eighty_twenty_boundary_credits_both():
    # Call 80 / Raise 20: both "Always Call" (80 >= 80) and "Mostly Raise"
    # (20 >= 20) clear their floors -> two neutral options.
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Mostly Call", {"Call": 0.80, "Raise": 0.20}
    )
    assert neutral == ["Always Call", "Mostly Raise"]


def test_gto_pure_correct_is_always_credits_the_mostly_rung():
    # Call 97 / Raise 3: correct flips to "Always Call"; "Mostly Call" is the
    # forgiven understatement (Call 97 >= 20); raise rungs are mistakes.
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Always Call", {"Call": 0.97, "Raise": 0.03}
    )
    assert neutral == ["Mostly Call"]


def test_gto_raise_dominant_is_symmetric():
    # Raise 62 / Call 38: correct "Mostly Raise"; "Mostly Call" neutral (38>=20);
    # "Always Raise" is a MISTAKE (62 < 80) -- the same gate as the call side.
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Mostly Raise", {"Call": 0.38, "Raise": 0.62}
    )
    assert neutral == ["Mostly Call"]


def test_gto_lopsided_has_no_neutral_on_the_minority_side():
    # Call 92 / Raise 8: only "Always Call" survives (92 >= 80); raise rungs out.
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Mostly Call", {"Call": 0.92, "Raise": 0.08}
    )
    assert neutral == ["Always Call"]


# --- Basic style: bare action labels, the 20% floor only ---------------------
def test_basic_near_pure_minority_gets_no_credit():
    # Call 85 / Raise 15, basic options -> Raise 15 < 20 -> no neutral.
    neutral = neutral_credit_options(
        ["Call", "Raise"], "Call", {"Call": 0.85, "Raise": 0.15}
    )
    assert neutral == []


def test_basic_twenty_percent_minority_is_credited():
    # Raise 80 / Fold 20 -> Fold 20 >= 20 -> neutral.
    neutral = neutral_credit_options(
        ["Fold", "Raise"], "Raise", {"Raise": 0.80, "Fold": 0.20}
    )
    assert neutral == ["Fold"]


def test_basic_three_way_credits_only_actions_over_twenty():
    # Raise 60 / Call 22 / Fold 18 -> Call neutral (22>=20), Fold mistake (18<20).
    neutral = neutral_credit_options(
        ["Fold", "Raise", "Call"],
        "Raise",
        {"Raise": 0.60, "Call": 0.22, "Fold": 0.18},
    )
    assert neutral == ["Call"]


def test_basic_very_mixed_three_way_gives_two_neutral():
    # Raise 56 / Call 24 / Fold 20 -> both Call and Fold clear 20% -> two neutral
    # (every option creditable). The worthiness floor keeps such spots out of
    # production, but the rule handles them.
    neutral = neutral_credit_options(
        ["Fold", "Raise", "Call"],
        "Raise",
        {"Raise": 0.56, "Call": 0.24, "Fold": 0.20},
    )
    # Result is in DISPLAY order (Fold-first, as the basic builder emits).
    assert neutral == ["Fold", "Call"]


# --- Postflop labels (sizing) round-trip through the prefix stripper ----------
def test_sizing_labels_are_parsed():
    # "Mostly Bet 75%" -> action "Bet 75%"; the freq map is keyed by label.
    options = ["Always Check", "Mostly Check", "Mostly Bet 75%", "Always Bet 75%"]
    neutral = neutral_credit_options(
        options, "Mostly Check", {"Check": 0.70, "Bet 75%": 0.30}
    )
    assert neutral == ["Mostly Bet 75%"]


# --- Mechanics ---------------------------------------------------------------
def test_correct_answer_is_never_neutral():
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Mostly Call", {"Call": 0.55, "Raise": 0.45}
    )
    assert "Mostly Call" not in neutral


def test_empty_options_are_skipped():
    neutral = neutral_credit_options(
        ["Call", "", "Raise"], "Call", {"Call": 0.70, "Raise": 0.30}
    )
    assert neutral == ["Raise"]


def test_input_order_is_preserved():
    neutral = neutral_credit_options(
        _GTO_CALL_RAISE, "Mostly Call", {"Call": 0.80, "Raise": 0.20}
    )
    assert neutral == ["Always Call", "Mostly Raise"]  # spectrum order, not sorted


def test_unknown_action_defaults_to_zero_frequency():
    # An option whose action isn't in the map is treated as 0% -> a mistake.
    neutral = neutral_credit_options(
        ["Call", "Raise"], "Call", {"Call": 1.0}
    )
    assert neutral == []


def test_format_neutral_credit_joins_and_blanks():
    assert format_neutral_credit(["Always Call", "Mostly Raise"]) == (
        "Always Call, Mostly Raise"
    )
    assert format_neutral_credit([]) == ""


# --- Schema placement: right after Correct Answer in every pipeline -----------
def test_column_sits_right_after_correct_answer_everywhere():
    for columns in (CSV_COLUMNS, PREFLOP_CSV_COLUMNS, PLO_CSV_COLUMNS, POSTFLOP_CSV_COLUMNS):
        cols = list(columns)
        assert "neutral_credit" in cols
        assert cols[cols.index("Correct Answer") + 1] == "neutral_credit"
