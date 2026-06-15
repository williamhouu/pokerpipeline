"""Tests for pipeline.preflop.why_factors (deterministic "why this action").

Pure rule-engine tests on synthetic ``WhyInputs`` -- no solver, no files,
no API. Verifies the lean resolution (aggressive -> the more aggressive of
the spot's two actions; the archetype always favors the answer), the
summary/render shape, and the CSV-row builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.why_factors import (  # noqa: E402
    WhyInputs,
    compute_breakdown,
    render_why_text,
    why_factors_from_row,
)


def _inp(**kw) -> WhyInputs:
    base = dict(
        dominant_action="Raise",
        alternative_action="Fold",
        tags=frozenset(),
        archetype="",
        hero_equity=None,
        break_even_equity=None,
        in_position=None,
    )
    base.update(kw)
    return WhyInputs(**base)  # type: ignore[arg-type]


def test_archetype_always_favors_the_answer_even_when_passive() -> None:
    """call_for_implied_odds names the CALL, so its factor favors Call (the
    answer) -- not the less-aggressive Fold."""
    bd = compute_breakdown(
        _inp(dominant_action="Call", alternative_action="Fold",
             archetype="call_for_implied_odds")
    )
    solver = next(f for f in bd.factors if f.label == "Solver's read")
    assert solver.favors == "Call"
    assert solver.strength == 3


def test_premium_leans_to_the_more_aggressive_action() -> None:
    """A premium hand argues for the more aggressive of the two actions."""
    bd = compute_breakdown(
        _inp(dominant_action="Raise", alternative_action="Fold",
             tags=frozenset({"premium_pair"}))
    )
    premium = next(f for f in bd.factors if f.label == "Premium hand")
    assert premium.favors == "Raise"  # more aggressive than Fold
    # Same factor, decision axis Call-vs-Raise -> still the more aggressive.
    bd2 = compute_breakdown(
        _inp(dominant_action="Call", alternative_action="Raise",
             tags=frozenset({"premium_pair"}))
    )
    assert next(f for f in bd2.factors if f.label == "Premium hand").favors == "Raise"


def test_dominated_leans_to_the_less_aggressive_action() -> None:
    bd = compute_breakdown(
        _inp(dominant_action="Fold", alternative_action="Call",
             tags=frozenset({"dominated"}))
    )
    dom = next(f for f in bd.factors if f.label == "Dominated")
    assert dom.favors == "Fold"  # less aggressive than Call


def test_pot_odds_factor_uses_equity_vs_break_even() -> None:
    good = compute_breakdown(
        _inp(dominant_action="Call", alternative_action="Fold",
             hero_equity=0.50, break_even_equity=0.33)
    )
    price = next(f for f in good.factors if f.label == "Good raw price")
    assert price.favors == "Call"  # a good price argues for continuing, not folding
    bad = compute_breakdown(
        _inp(dominant_action="Fold", alternative_action="Call",
             hero_equity=0.30, break_even_equity=0.40)
    )
    assert next(f for f in bad.factors if f.label == "Bad price").favors == "Fold"
    # Within 4 points either way -> no pot-odds factor (too close to call).
    none = compute_breakdown(
        _inp(hero_equity=0.34, break_even_equity=0.33,
             dominant_action="Call", alternative_action="Fold")
    )
    assert not any(f.label in ("Good raw price", "Bad price") for f in none.factors)


def test_continue_and_fold_leans_drop_when_action_absent() -> None:
    """A good-raw-price factor needs a call on the axis; a bad-price factor
    needs a fold. On a raise-or-fold axis the good-price factor drops."""
    # Raise-vs-Fold: no call action, so "Good raw price" (continue) is dropped.
    bd = compute_breakdown(
        _inp(dominant_action="Raise", alternative_action="Fold",
             hero_equity=0.50, break_even_equity=0.33)
    )
    assert not any(f.label == "Good raw price" for f in bd.factors)
    # Call-vs-Raise: no fold action, so "Bad price" (fold) is dropped.
    bd2 = compute_breakdown(
        _inp(dominant_action="Call", alternative_action="Raise",
             hero_equity=0.30, break_even_equity=0.40)
    )
    assert not any(f.label == "Bad price" for f in bd2.factors)


def test_summary_names_answer_and_main_opposing_pull() -> None:
    bd = compute_breakdown(
        _inp(dominant_action="Fold", alternative_action="Call",
             archetype="fold_dominated",
             tags=frozenset({"dominated", "blocks_villain_top_value"}))
    )
    # dominated + solver's read favor Fold; blockers favor Call.
    assert bd.summary.endswith("The main pull toward Call is key blockers.")
    assert "push toward Fold" in bd.summary


def test_render_text_groups_and_carries_the_caveat() -> None:
    bd = compute_breakdown(
        _inp(dominant_action="Raise", alternative_action="Fold",
             archetype="open_for_value", tags=frozenset({"premium_pair"}),
             in_position=True)
    )
    text = render_why_text(bd)
    assert "Why Raise (vs Fold)" in text
    assert "Arguing for Raise:" in text
    assert "Directional heuristic" in text  # the honesty caveat is always there


def test_no_factors_degrades_gracefully() -> None:
    bd = compute_breakdown(_inp(dominant_action="Call", alternative_action="Fold"))
    assert bd.factors == ()
    assert bd.summary == "The solver's answer is Call."


def test_from_row_reads_the_csv_columns() -> None:
    """The admin-panel path reconstructs the same inputs from a written row."""
    row = {
        "action_frequencies": "Fold: 70%, Call: 30%",
        "concept_tags": "dominated, big_blind, villain_range_advantage",
        "archetype": "fold_dominated",
        "hero_equity": "31%",
        "pot_odds": "40%",
        "Relative Position": "Out of Position",
    }
    bd = why_factors_from_row(row)
    assert bd.answer == "Fold"  # first (highest-freq) action
    assert bd.alternative == "Call"
    assert any(f.label == "Dominated" for f in bd.factors)
    assert any(f.label == "Bad price" for f in bd.factors)  # 31% < 40%
    assert any(f.label == "Out of position" for f in bd.factors)
    # The solver's read favors the answer.
    assert next(f for f in bd.factors if f.label == "Solver's read").favors == "Fold"


def test_from_row_tolerates_blanks() -> None:
    bd = why_factors_from_row({})
    assert bd.answer == ""
    assert isinstance(bd.summary, str)
