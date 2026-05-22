"""Concept tags -- Section B: Decision Class (9 tags).

From docs/engineering_brief.docx, "Concept Tag Library Specification",
Section B. Each tag is a pure function: SpotData -> bool.

Section B rules reference hand_class strength buckets; those map directly to
the buckets emitted by pipeline.fact_extractor.hand_class (premium, strong,
medium, vulnerable, marginal, air).

Numeric thresholds are the brief's starting values; they will be tuned against
the ~800-explanation gold pool before the tagger goes to production.
"""
from __future__ import annotations

from pipeline.fact_extractor.concept_tags.section_a_range import villain_polarized
from pipeline.fact_extractor.spot_data import SpotData

# The brief gives no SPR figure for pot_control's "SPR makes future barrels
# committing" clause; this ceiling is an added v1 starting value to tune.
_SPR_COMMIT_CEILING = 3.0


def _ev(spot: SpotData, action: str):
    """Range-mean EV for an action (bb), or None when the action isn't offered."""
    return spot.decision_data.range_mean_evs_per_action.get(action)


def thin_value_spot(spot: SpotData) -> bool:
    """Hero value-bets a hand that beats only marginal made hands.

    Brief definition: hero is betting for value with a hand that beats only
    marginal made hands in villain's calling range; EV gain is small but
    positive.

    Brief rule: bet_EV > check_EV by between 0.3 and 1.0bb, AND villain's
    calling range is dominated (>70%) by hands worse than hero, AND hero's hand
    is in the 50-70% equity bucket vs the calling range.
    """
    bet, check = _ev(spot, "bet"), _ev(spot, "check")
    if bet is None or check is None:
        return False
    ev_gain = bet - check
    return (0.3 <= ev_gain <= 1.0
            and spot.range_data.villain_calling_range_worse_pct > 0.70
            and 0.50 <= spot.equity_data.hero_raw_equity_vs_calling <= 0.70)


def bluffcatch_spot(spot: SpotData) -> bool:
    """Hero faces a bet with a hand that beats only bluffs.

    Brief definition: hero is facing a bet or raise with a hand that beats only
    bluffs in villain's range.

    Brief rule: hero_equity_vs_continuing_range is between 0.30 and 0.55, AND
    hero's hand has near-zero value-betting potential on later streets, AND
    villain's range is polarized.
    """
    equity = spot.equity_data.hero_raw_equity_vs_continuing
    return (0.30 <= equity <= 0.55
            and not spot.decision_data.hero_has_later_street_value
            and villain_polarized(spot))


def protection_bet_spot(spot: SpotData) -> bool:
    """Hero bets a vulnerable made hand to deny equity.

    Brief definition: hero bets a vulnerable made hand to deny equity to
    villain's marginal hands and draws.

    Brief rule: bet_EV > check_EV, hero's hand class is in the vulnerable
    bucket (top pair on wet board, weak pair, weak overpair), AND villain's
    range has significant equity from draws and overcards (>30%).
    """
    bet, check = _ev(spot, "bet"), _ev(spot, "check")
    if bet is None or check is None or spot.hand_class is None:
        return False
    return (bet > check
            and spot.hand_class.strength_bucket == "vulnerable"
            and spot.range_data.villain_draw_equity_pct > 0.30)


def merged_value_spot(spot: SpotData) -> bool:
    """Hero value-bets small to be called by a wider range.

    Brief definition: hero bets for value with a hand that's clearly ahead but
    not the nuts, using a smaller sizing to get called by a wider range.

    Brief rule: bet_EV > check_EV, villain's calling range includes both worse
    made hands AND draws, AND solver bet sizing is small (less than 60% pot).
    """
    bet, check = _ev(spot, "bet"), _ev(spot, "check")
    if bet is None or check is None:
        return False
    # "Solver bet sizing" -- the pot fraction of hero's bet option.
    bet_fraction = spot.decision_data.option_pot_fractions.get("bet")
    if bet_fraction is None:
        return False
    return (bet > check
            and spot.range_data.villain_calling_range_has_worse_made_hands
            and spot.range_data.villain_calling_range_has_draws
            and bet_fraction < 0.60)


def pot_control_spot(spot: SpotData) -> bool:
    """Hero checks a medium-strength hand to keep the pot small.

    Brief definition: hero checks with a marginal made hand rather than
    betting, to avoid building an awkward pot.

    Brief rule: check_EV > bet_EV, hero's hand class is medium-strength, AND
    hero is OOP or SPR makes future barrels committing.

    "Medium-strength" maps to the hand_class `medium` bucket. The brief's
    parenthetical examples (weak top pair, second pair) also span the
    `vulnerable` bucket -- flagged for tuning against gold.
    """
    bet, check = _ev(spot, "bet"), _ev(spot, "check")
    if bet is None or check is None or spot.hand_class is None:
        return False
    if spot.hand_class.strength_bucket != "medium":
        return False
    spr = spot.spot_metadata.spr
    commits = (not spot.spot_metadata.hero_in_position
               or 0 < spr < _SPR_COMMIT_CEILING)
    return check > bet and commits


def equity_denial_spot(spot: SpotData) -> bool:
    """Hero bets mainly to fold out equity, not for value.

    Brief definition: hero bets primarily to fold out hands with significant
    equity vs hero's made hand, not for value.

    Brief rule: bet_EV > check_EV, AND check_EV vs villain's overall range >
    bet_EV vs villain's calling-only range (better hands call, weaker hands
    fold). Hero's hand class is marginal (weak pair or worse).
    """
    bet, check = _ev(spot, "bet"), _ev(spot, "check")
    if bet is None or check is None or spot.hand_class is None:
        return False
    return (bet > check
            and spot.hand_class.strength_bucket == "marginal"
            and (spot.decision_data.check_ev_vs_villain_range
                 > spot.decision_data.bet_ev_vs_calling_range))


def bluff_spot(spot: SpotData) -> bool:
    """Hero bets a low-showdown-value hand for fold equity.

    Brief definition: hero bets with a hand that has little showdown value but
    has equity if called and good fold equity.

    Brief rule: bet_EV > check_EV, hero's hand has less than 30% equity vs
    villain's continuing range, AND fold equity is the dominant value source
    (estimated fold equity > estimated showdown value when called).
    """
    bet, check = _ev(spot, "bet"), _ev(spot, "check")
    if bet is None or check is None:
        return False
    return (bet > check
            and spot.equity_data.hero_raw_equity_vs_continuing < 0.30
            and (spot.decision_data.estimated_fold_equity
                 > spot.decision_data.estimated_showdown_value))


def commitment_threshold_decision(spot: SpotData) -> bool:
    """Hero is at the stack depth where continuing commits the stack.

    Brief definition: hero is at or near the stack depth where calling commits
    them to playing the rest of the hand.

    Brief rule: post-decision SPR < 1.5, OR hero's stack is committed >50% with
    the call.
    """
    decision = spot.decision_data
    spr_commits = 0 < decision.post_decision_spr < 1.5
    stack_commits = decision.stack_committed_fraction > 0.50
    return spr_commits or stack_commits


def float_call_spot(spot: SpotData) -> bool:
    """Hero calls weak, planning to take the pot away on a later street.

    Brief definition: hero calls a bet with a weak hand intending to take the
    pot away on a later street if villain shows weakness; the call's EV comes
    from future-street fold equity, not showdown value.

    Brief rule: call EV > fold EV, hero's equity vs villain's continuing range
    is <40%, AND hero has positive EV-by-line if villain checks the next street
    and hero bets (requires multi-street analysis from the solver).
    """
    call, fold = _ev(spot, "call"), _ev(spot, "fold")
    if call is None or fold is None:
        return False
    return (call > fold
            and spot.equity_data.hero_raw_equity_vs_continuing < 0.40
            and spot.decision_data.float_line_is_positive)
