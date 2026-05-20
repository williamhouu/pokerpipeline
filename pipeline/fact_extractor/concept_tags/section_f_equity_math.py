"""Concept tags -- Section F: Equity & Math Concepts (5 tags).

From docs/engineering_brief.docx, "Concept Tag Library Specification",
Section F. Each tag is a pure function: SpotData -> bool.

Numeric thresholds are the brief's starting values; they will be tuned against
the ~800-explanation gold pool before the tagger goes to production.
"""
from __future__ import annotations

from pipeline.fact_extractor.spot_data import SpotData

# "Solver's defence frequency within 5% of MDF" -- read as within 0.05
# absolute (5 percentage points).
_MDF_MARGIN = 0.05


def _ev(spot: SpotData, action: str):
    """Hero's combo EV for an action, or None when the action isn't offered."""
    return spot.decision_data.hero_combo_evs.get(action)


def equity_under_realized(spot: SpotData) -> bool:
    """Hero will not realise most of their raw equity.

    Brief definition: hero has decent raw equity but won't realise most of it
    because of position, stack depth, or reverse implied odds.

    Brief rule: equity_realization_ratio (EQR) < 0.85.
    """
    return spot.equity_data.equity_realization_ratio < 0.85


def equity_over_realized(spot: SpotData) -> bool:
    """Hero captures more EV than raw equity predicts.

    Brief definition: hero captures more EV than raw equity predicts.

    Brief rule: EQR > 1.05.
    """
    return spot.equity_data.equity_realization_ratio > 1.05


def implied_odds_call(spot: SpotData) -> bool:
    """Hero's call is profitable on later-street value, not immediate odds.

    Brief definition: hero's call is profitable not because of immediate pot
    odds, but because of expected value on later streets when hero hits their
    draw.

    Brief rule: hero_raw_equity < pot_odds_required (direct math says fold) AND
    solver_ev_call > 0.3bb AND hand class is a drawing hand.

    v1: "a drawing hand" is read as hero's hand_class carrying at least one
    draw. The brief's preflop set-mining example (calling with 22) is not
    captured -- the hand_class module is postflop-only.
    """
    call_ev = _ev(spot, "call")
    if call_ev is None:
        return False
    if spot.hand_class is None or not spot.hand_class.draws:
        return False
    equity = spot.equity_data
    return (equity.hero_raw_equity_vs_continuing < equity.pot_odds_required
            and call_ev > 0.3)


def reverse_implied_odds_call(spot: SpotData) -> bool:
    """Raw odds say call, but the hand pays off when it 'improves'.

    Brief definition: raw equity says the call is fine, but the solver hesitates
    because hero's hand class is prone to being dominated or paying off when
    "improving".

    Brief rule: hero_raw_equity > pot_odds_required x 1.15 (raw math says easy
    call) AND solver_ev_call < 0.3bb (solver disagrees) AND
    equity_realization_ratio < 0.85 (hero realises poorly).
    """
    call_ev = _ev(spot, "call")
    if call_ev is None:
        return False
    equity = spot.equity_data
    return (equity.hero_raw_equity_vs_continuing > equity.pot_odds_required * 1.15
            and call_ev < 0.3
            and equity.equity_realization_ratio < 0.85)


def mdf_defense_threshold(spot: SpotData) -> bool:
    """The decision is whether to defend to meet minimum defence frequency.

    Brief definition: the decision is fundamentally about whether to defend
    (call) to meet minimum defence frequency against villain's bet; hero's
    specific hand strength is less important than the population-defence math.

    Brief rule: hero is facing a bet. Solver's defence frequency for hero's
    range is within 5% of MDF. The decision is at the edge of defendable.
    """
    decision = spot.decision_data
    equity = spot.equity_data
    if decision.facing_bet_pot_fraction is None:        # not facing a bet
        return False
    if equity.mdf <= 0 or not decision.range_aggregate_strategy:
        return False                                    # MDF / strategy unknown
    defense_frequency = 1.0 - decision.range_aggregate_strategy.get("fold", 0.0)
    return abs(defense_frequency - equity.mdf) <= _MDF_MARGIN
