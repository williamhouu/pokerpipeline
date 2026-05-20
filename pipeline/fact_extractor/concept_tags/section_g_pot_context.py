"""Concept tags -- Section G: Pot and Action Context (8 tags).

From docs/engineering_brief.docx, "Concept Tag Library Specification",
Section G. Each tag is a pure function: SpotData -> bool.

These describe the structural context of the pot. Two tags emit names that are
not valid Python identifiers -- the functions three_bet_pot and four_bet_pot
emit the tag strings "3bet_pot" and "4bet_pot"; the registry maps them.

Numeric thresholds are the brief's starting values; they will be tuned against
the ~800-explanation gold pool before the tagger goes to production.
"""
from __future__ import annotations

from pipeline.fact_extractor.spot_data import SpotData

# Tournament effective-stack ceiling (bb) below which push/fold math dominates.
_SHORT_STACK_BB = 25
_BLINDS = ("SB", "BB")


def three_bet_pot(spot: SpotData) -> bool:
    """Pot saw two preflop raises (open + 3-bet). Emits the tag "3bet_pot".

    Brief definition: pot saw two raises preflop (open + 3-bet).

    Brief rule: count of preflop raises = 2.
    """
    return spot.spot_metadata.preflop_raise_count == 2


def four_bet_pot(spot: SpotData) -> bool:
    """Pot saw three preflop raises (open + 3-bet + 4-bet). Emits "4bet_pot".

    Brief definition: pot saw three raises preflop (open + 3-bet + 4-bet).
    Tight ranges, very low SPR, often near commitment.

    Brief rule: count of preflop raises = 3.
    """
    return spot.spot_metadata.preflop_raise_count == 3


def single_raised_pot(spot: SpotData) -> bool:
    """Standard one-raise preflop pot.

    Brief definition: standard one-raise preflop pot. Wider ranges, higher SPR.

    Brief rule: count of preflop raises = 1.
    """
    return spot.spot_metadata.preflop_raise_count == 1


def squeeze_pot(spot: SpotData) -> bool:
    """A 3-bet pot where the 3-bet came over a cold-caller.

    Brief definition: 3-bet pot where the 3-bet came over a cold-caller, not
    just an opener. Tighter than standard 3-bet pots.

    Brief rule: 3-bet pot AND at least one player cold-called the opener before
    the 3-bet.
    """
    meta = spot.spot_metadata
    return meta.preflop_raise_count == 2 and meta.had_cold_caller


def multiway_pot(spot: SpotData) -> bool:
    """Three or more players see the flop.

    Brief definition: three or more players see the flop.

    Brief rule: active_players_on_flop >= 3.
    """
    return spot.spot_metadata.active_players_on_flop >= 3


def blind_vs_blind_pot(spot: SpotData) -> bool:
    """Preflop action involved only the two blinds.

    Brief definition: preflop action involved only SB and BB. Wider ranges,
    more aggression possible, different positional dynamics than standard
    raised pots.

    Brief rule: hero and the only opponent in the pot are SB and BB. No other
    players took non-fold actions preflop.
    """
    meta = spot.spot_metadata
    return (meta.hero_position in _BLINDS
            and meta.villain_position in _BLINDS
            and meta.hero_position != meta.villain_position)


def blind_defense_spot(spot: SpotData) -> bool:
    """Hero in a blind is defending preflop against an open.

    Brief definition: hero in SB or BB is defending preflop against an open
    raise.

    Brief rule: hero is SB or BB. Hero is facing a preflop open. The current
    decision is hero's preflop continue decision.
    """
    meta = spot.spot_metadata
    return (meta.hero_position in _BLINDS
            and meta.street == "preflop"
            and meta.preflop_raise_count == 1            # exactly an open
            and not meta.hero_is_preflop_raiser)         # hero defends, didn't open


def short_stack_tournament(spot: SpotData) -> bool:
    """Tournament spot shallow enough for push/fold math to dominate.

    Brief definition: tournament spot where effective stack is shallow enough
    that preflop push-fold and reshove math dominates.

    Brief rule: format == "tournament" AND effective_stack_bb < 25.
    """
    meta = spot.spot_metadata
    return (meta.game_format == "tournament"
            and meta.effective_stack_bb < _SHORT_STACK_BB)
