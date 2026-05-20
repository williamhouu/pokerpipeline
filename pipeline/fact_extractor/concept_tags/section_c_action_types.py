"""Concept tags -- Section C: Postflop Action-Type Spots (8 tags).

From docs/engineering_brief.docx, "Concept Tag Library Specification",
Section C. Each tag is a pure function: SpotData -> bool.

These tags identify the postflop action structure of a spot (check-raise,
donk, probe, overbet, and their facing counterparts), independent of the
strategic concepts that apply. They read the action line from
DecisionData.street_actions and .prior_street_actions -- ordered (actor,
action) pairs whose actor is "hero" or "villain".

Numeric thresholds are the brief's starting values; they will be tuned against
the ~800-explanation gold pool before the tagger goes to production.
"""
from __future__ import annotations

from pipeline.fact_extractor.spot_data import SpotData

# "EV-best or near-EV-best": an option within this margin of the best EV.
_NEAR_BEST_BB = 0.5


def _first_index(line, actor: str, action: str):
    """Index of the first (actor, action) entry in the line, or None."""
    for index, entry in enumerate(line):
        if entry == (actor, action):
            return index
    return None


def _did(line, actor: str, action: str) -> bool:
    """Whether (actor, action) appears anywhere in the line."""
    return (actor, action) in line


def _near_best(spot: SpotData, option: str) -> bool:
    """Whether an offered option's EV is within _NEAR_BEST_BB of the best EV."""
    evs = spot.decision_data.hero_combo_evs
    if option not in evs:
        return False
    return max(evs.values()) - evs[option] <= _NEAR_BEST_BB


def check_raise_spot(spot: SpotData) -> bool:
    """Hero is considering check-raising.

    Brief definition: the decision involves a check-raise as one of the
    available actions, and it's the EV-best or near-EV-best play.

    Brief rule: action sequence shows hero checked, villain bet, and the
    decision now includes a raise option whose EV is within 0.5bb of the best
    action's EV.
    """
    line = spot.decision_data.street_actions
    hero_check = _first_index(line, "hero", "check")
    villain_bet = _first_index(line, "villain", "bet")
    line_ok = (hero_check is not None and villain_bet is not None
               and hero_check < villain_bet)
    return (line_ok and "raise" in spot.decision_data.options
            and _near_best(spot, "raise"))


def facing_check_raise_spot(spot: SpotData) -> bool:
    """Hero bet, villain check-raised, hero is responding.

    Brief definition: hero bet, villain check-raised, hero is responding.

    Brief rule: action sequence shows hero bet, villain raised (where villain
    had checked first). Hero is now responding.
    """
    line = spot.decision_data.street_actions
    villain_check = _first_index(line, "villain", "check")
    hero_bet = _first_index(line, "hero", "bet")
    villain_raise = _first_index(line, "villain", "raise")
    if villain_check is None or hero_bet is None or villain_raise is None:
        return False
    return villain_check < hero_bet < villain_raise


def donk_bet_spot(spot: SpotData) -> bool:
    """Hero is considering leading into the preflop raiser when out of position.

    Brief definition: hero is considering leading into the preflop raiser when
    out of position, rather than checking; the lead is one of the available
    actions and is the EV-best or near-EV-best play.

    Brief rule: hero is OOP. Hero was NOT the preflop raiser. No
    check-to-raiser action yet. Bet option has EV within 0.5bb of the best
    action.
    """
    meta = spot.spot_metadata
    if meta.hero_in_position or meta.hero_is_preflop_raiser:
        return False
    # No check-to-raiser action yet: hero has not checked on this street.
    if _first_index(spot.decision_data.street_actions, "hero", "check") is not None:
        return False
    return "bet" in spot.decision_data.options and _near_best(spot, "bet")


def facing_donk_spot(spot: SpotData) -> bool:
    """Hero is the preflop raiser; villain led into hero.

    Brief definition: hero is the preflop raiser, villain led into hero
    (instead of checking as expected).

    Brief rule: hero was the preflop raiser. Villain is OOP. Villain bet before
    hero acted on the current street.
    """
    meta = spot.spot_metadata
    if not meta.hero_is_preflop_raiser or not meta.hero_in_position:
        return False                            # villain OOP <=> hero IP
    line = spot.decision_data.street_actions
    villain_bet = _first_index(line, "villain", "bet")
    if villain_bet is None:
        return False
    hero_acted_first = any(actor == "hero" for actor, _ in line[:villain_bet])
    return not hero_acted_first


def probe_bet_spot(spot: SpotData) -> bool:
    """Hero is considering betting into an opponent who checked back.

    Brief definition: hero is considering betting into an opponent who checked
    back the previous street; the bet exploits villain's revealed weakness.

    Brief rule: previous street, villain had the option to bet and checked.
    Hero is now considering a bet whose EV is within 0.5bb of the best action.
    """
    if not _did(spot.decision_data.prior_street_actions, "villain", "check"):
        return False
    return "bet" in spot.decision_data.options and _near_best(spot, "bet")


def facing_probe_spot(spot: SpotData) -> bool:
    """Hero checked back last street; villain bet into hero this street.

    Brief definition: hero checked back the previous street, villain bet into
    hero on the current street.

    Brief rule: hero checked back on the previous street. Villain bet on the
    current street.
    """
    return (_did(spot.decision_data.prior_street_actions, "hero", "check")
            and _did(spot.decision_data.street_actions, "villain", "bet"))


def overbet_spot(spot: SpotData) -> bool:
    """Hero is considering betting more than the pot.

    Brief definition: hero is considering betting more than the pot; the
    overbet is one of the available actions and is the EV-best or near-EV-best
    play.

    Brief rule: decision options include a bet > pot. That overbet's EV is
    within 0.5bb of the best action's EV.
    """
    fractions = spot.decision_data.option_pot_fractions
    overbets = [option for option, fraction in fractions.items() if fraction > 1.0]
    return any(_near_best(spot, option) for option in overbets)


def facing_overbet_spot(spot: SpotData) -> bool:
    """Villain has bet more than the pot, hero is responding.

    Brief definition: villain has bet more than the pot, hero is responding.

    Brief rule: villain's bet amount > pot size before villain's bet.
    """
    fraction = spot.decision_data.facing_bet_pot_fraction
    return fraction is not None and fraction > 1.0
