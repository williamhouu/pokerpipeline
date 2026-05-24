"""Recommended-action archetype classifier (Ryan-feedback Fix 5, May 2026).

Maps (correct_action, hand_strength, aggression_history, facing_bet, draws) ->
one of 13 strategic archetypes. The archetype is the STRATEGIC FRAME the LLM's
explanation should be built around -- not the action itself.

The V7.1 failure case the user named in the prompt: a strong hand on a later
street where hero had the betting lead and the recommended action is check.
Pre-Fix-5, the LLM matched the literal concept tag `nut_advantage_villain` and
wrote 'villain has the nut advantage' -- wrong frame. Post-Fix-5, the
classifier marks the spot as `trap_check`, and Layer 6's per-archetype voice
guidance instructs the LLM to frame the explanation as 'hero checks to induce
continued villain aggression' (the actual strategic reason).

The 13 archetypes (per user spec, May 2026):

    Bet/raise initiators
      value_bet           premium/strong hand bet for value
      bluff               air bet to fold out better hands (fold equity)
      protection_bet      medium/vulnerable made hand bet to deny equity
      induce_bet          small bet sized to induce a villain raise

    Check archetypes
      pot_control_check   medium/strong hand check to avoid bloating the pot
      trap_check          strong hand check after hero had the betting lead
      defensive_check     weak/marginal hand check (no profitable bet)

    Call archetypes
      call_with_showdown  call (not facing a bet) to close action with SDV
      call_drawing        air with draws calling on pot odds + implied
      bluff_catch         medium-strong made hand calling vs polarized villain

    Fold archetype
      fold_to_polar       villain's value beats hero's equity at offered price

    Raise archetypes (when facing a bet)
      value_raise         premium/strong hand raising for value
      bluff_raise         air with fold equity raising as a (semi-)bluff

A note on `induce_bet`: it's the least common postflop archetype and the
hardest to detect from solver output alone (the small-size signal that
distinguishes induce from value would need bet-size context this classifier
doesn't have yet). v1: classifier never returns `induce_bet`; medium-hand
bets fall into `protection_bet`. Adding `induce_bet` detection is a future
tune-against-gold improvement.
"""
from __future__ import annotations

# Canonical archetype names. Layer 6 prompt + Layer 7 validator both import
# this tuple; adding a new archetype here is the single point of edit.
RECOMMENDED_ACTION_ARCHETYPES = (
    "value_bet", "bluff", "protection_bet", "induce_bet",
    "pot_control_check", "trap_check", "defensive_check",
    "call_with_showdown", "call_drawing", "bluff_catch",
    "fold_to_polar",
    "value_raise", "bluff_raise",
)

# Verbs that bet money INTO the pot (open the action on a street).
_BET_VERBS = frozenset({"bet", "donk", "overbet", "shove", "jam", "all-in"})
# Verbs that raise OVER an existing bet.
_RAISE_VERBS = frozenset({"raise"})

# Made-hand strength buckets (from pipeline.fact_extractor.hand_class):
# premium > strong > medium > vulnerable > marginal > air.
_STRONG_BUCKETS = frozenset({"premium", "strong"})
_MEDIUM_BUCKETS = frozenset({"medium", "vulnerable"})
_WEAK_BUCKETS = frozenset({"marginal", "air"})


def _is_facing_bet(spot) -> bool:
    """Whether the decision is facing a non-trivial bet."""
    f = spot.decision_data.facing_bet_pot_fraction
    return f is not None and f > 0.0


def _hero_was_aggressor_last_street(spot) -> bool:
    """Whether the last entry in `aggression_history` is hero.

    Hero-was-aggressor is the trigger for trap_check vs pot_control_check:
    a strong hand checking AFTER betting is inducing; a strong hand checking
    when villain led prior is pot-controlling.
    """
    hist = spot.spot_metadata.aggression_history
    return bool(hist) and hist[-1] == "hero"


def _hand_bucket(spot) -> str:
    return spot.hand_class.strength_bucket if spot.hand_class else ""


def _hand_has_draws(spot) -> bool:
    return bool(spot.hand_class and spot.hand_class.draws)


def classify_recommended_archetype(spot) -> str:
    """Map a SpotData to one of the 13 strategic archetypes.

    Empty string when no `correct_action` is set (e.g. test fixtures with
    minimal DecisionData). Defaults are documented inline; the v1 heuristics
    are intentionally simple and tunable against the gold explanations.
    """
    correct = spot.decision_data.correct_action
    if not correct:
        return ""

    bucket = _hand_bucket(spot)
    facing_bet = _is_facing_bet(spot)
    has_draws = _hand_has_draws(spot)
    hero_was_aggressor = _hero_was_aggressor_last_street(spot)

    # ---- BET / DONK / OVERBET / SHOVE / JAM / ALL-IN ------------------------
    if correct in _BET_VERBS:
        if bucket in _STRONG_BUCKETS:
            return "value_bet"
        if bucket in _MEDIUM_BUCKETS:
            return "protection_bet"
        # _WEAK_BUCKETS or unknown -> bluff (with or without draws).
        return "bluff"

    # ---- RAISE (over a bet hero is facing) ----------------------------------
    if correct in _RAISE_VERBS:
        # If we're NOT facing a bet, raise is being used as the "bet" verb;
        # fall back to bet rules.
        if not facing_bet:
            if bucket in _STRONG_BUCKETS:
                return "value_bet"
            if bucket in _MEDIUM_BUCKETS:
                return "protection_bet"
            return "bluff"
        if bucket in _STRONG_BUCKETS:
            return "value_raise"
        # Medium-hand raises facing a bet are rare; default to value_raise so
        # the prompt frames them as merge-value rather than as a bluff.
        if bucket in _MEDIUM_BUCKETS:
            return "value_raise"
        return "bluff_raise"

    # ---- CHECK -------------------------------------------------------------
    if correct == "check":
        if bucket in _STRONG_BUCKETS:
            if hero_was_aggressor:
                return "trap_check"           # had lead -> induce continued bet
            return "pot_control_check"        # no lead -> control pot size
        if bucket in _MEDIUM_BUCKETS:
            return "pot_control_check"
        # _WEAK_BUCKETS or unknown -> defensive (no profitable bet line).
        return "defensive_check"

    # ---- CALL --------------------------------------------------------------
    if correct == "call":
        # Calling without facing a bet means closing the action (e.g. BB
        # completing) or unusual node shapes -- rare postflop.
        if not facing_bet:
            return "call_with_showdown"
        # Air WITH draws calling on pot odds -- pure draw call.
        if bucket in _WEAK_BUCKETS and has_draws:
            return "call_drawing"
        # Made hand calling vs villain's bet -- bluff catch (the catch covers
        # premium/strong / medium / vulnerable / marginal; the distinction is
        # frame, not value strength).
        if bucket in _STRONG_BUCKETS or bucket in _MEDIUM_BUCKETS:
            return "bluff_catch"
        if bucket == "marginal":
            return "bluff_catch"
        # Air without draws calling -> still a bluff catch on equity, but the
        # frame is closer to "the price is too good to fold" than "made hand
        # catching"; v1 collapses to call_with_showdown.
        return "call_with_showdown"

    # ---- FOLD --------------------------------------------------------------
    if correct == "fold":
        return "fold_to_polar"

    return ""


# Per-archetype voice guidance for the Layer 6 system prompt. Each entry is
# ONE short paragraph the LLM reads alongside the SOLVER DATA block: the
# strategic frame the explanation must convey when that archetype is set.
ARCHETYPE_GUIDANCE = {
    "value_bet":
        "The recommended action is a VALUE BET. Frame the explanation as "
        "'hero bets to extract value from worse hands that will continue.' "
        "Name 2-3 specific worse-hand combos in villain's continuing range "
        "(from `range_data.villain_top_value_combos`) that hero is targeting.",
    "bluff":
        "The recommended action is a BLUFF (or semi-bluff with draws). Frame "
        "as 'hero bets to fold out better hands; the bet has fold equity that "
        "the hand cannot realise by checking.' Name 2-3 specific better-hand "
        "combos in villain's range that fold to the bet.",
    "protection_bet":
        "The recommended action is a PROTECTION BET with a medium/vulnerable "
        "made hand. Frame as 'hero bets to deny villain's draw equity and "
        "make worse hands pay to chase.' DO NOT frame as 'thin value' -- the "
        "primary reason is equity denial, not extraction.",
    "induce_bet":
        "The recommended action is an INDUCE BET (a small or block bet meant "
        "to look weak so villain raises with a wider range than they would "
        "vs a check). This archetype is rare; v1 of the classifier never "
        "emits it, so you should not normally see this guidance fire.",
    "pot_control_check":
        "The recommended action is a POT CONTROL CHECK with a medium-to-strong "
        "hand. Frame as 'hero checks because betting builds a pot hero does "
        "not want to play big -- villain's worse hands will fold to a bet and "
        "villain's better hands will continue. Checking realises showdown "
        "value at small pot size.' DO NOT frame as 'villain has the nut "
        "advantage' -- that anti-pattern was caught in V7.1 and is wrong.",
    "trap_check":
        "The recommended action is a TRAP CHECK with a strong/premium hand. "
        "Hero HAD THE BETTING LEAD on prior streets and is now checking to "
        "INDUCE villain to continue betting (especially when villain holds "
        "bluffs or worse made hands that would fold to a bet but will fire "
        "as a bluff into a check). Frame as 'hero's hand is too strong to "
        "fold villain out -- let them barrel.' Cite 2-3 villain combos that "
        "would bluff into a check but fold to a bet. DO NOT frame as 'pot "
        "control' (hero is NOT trying to control the pot -- hero wants the "
        "pot big with this hand) and DO NOT frame as 'villain has the nut "
        "advantage' (hero has the strong hand).",
    "defensive_check":
        "The recommended action is a DEFENSIVE CHECK with a weak/marginal "
        "hand. Frame as 'hero checks because there is no profitable bet line "
        "-- worse hands continue, better hands don't fold, and hero has too "
        "little equity to barrel.' Letting villain check back lets hero "
        "realise the tiny equity hero does have.",
    "call_with_showdown":
        "The recommended action is a CALL (not facing a bet, or to close "
        "action with showdown value). Frame as 'hero calls because the hand "
        "has enough showdown value to see the next card / get to showdown "
        "without committing more chips.'",
    "call_drawing":
        "The recommended action is a CALL with a DRAWING HAND. Frame as "
        "'hero calls because the draw equity meets the required pot odds, "
        "and implied odds on improvement justify the call.' Name the draw "
        "type (flush draw, open-ended straight draw, combo draw) explicitly.",
    "bluff_catch":
        "The recommended action is a BLUFF CATCH. Frame as 'hero calls "
        "because villain's bluff combos outweigh value relative to the "
        "required equity at this price -- the hand is calling because the "
        "PRICE IS RIGHT, not because it beats value.' DO NOT frame as 'hand "
        "is strong enough' or 'for value' -- bluff_catch is NOT a value call.",
    "fold_to_polar":
        "The recommended action is a FOLD. Frame as 'villain's value combos "
        "outweigh bluffs at the offered price; hero's hand class does not "
        "have enough equity vs villain's continuing range.' Name 2-3 specific "
        "villain value combos hero loses to.",
    "value_raise":
        "The recommended action is a VALUE RAISE (facing a bet). Frame as "
        "'hero raises to build the pot vs villain's worse-hand continuing "
        "range.' Distinguish from a bluff raise: name the value combos hero "
        "is targeting.",
    "bluff_raise":
        "The recommended action is a BLUFF RAISE (or check-raise). Frame as "
        "'hero raises to fold out villain's bluff-catchers and pick up dead "
        "money -- the raise's fold equity vs villain's medium range justifies "
        "the line.' Name the equity hero has when called.",
}


def aggression_history_from_action_sequence(action_sequence, hero_is_oop: bool) -> list[str]:
    """Compute the per-street aggressor list from a Path Sampler action_sequence.

    Input: list of (actor, label) tuples; actor in {"OOP", "IP", "deal"};
    `deal` entries are street boundaries. `hero_is_oop` is from
    `SpotMetadata.hero_in_position` (= not hero_in_position) so we can label
    actors as hero / villain.

    Output: one entry per COMPLETED prior street (current street excluded).
    Each entry is "hero", "villain", or "check" (no bets that street).

    For a postflop solve the action_sequence does not carry preflop, so a
    flop-decision spot has an empty history; a turn-decision spot has one
    entry (who led / how flop ended); a river-decision spot has two entries.
    """
    if not action_sequence:
        return []
    segments: list[list] = [[]]
    for actor, label in action_sequence:
        if actor == "deal":
            segments.append([])
        else:
            segments[-1].append((actor, label))
    if len(segments) <= 1:
        return []
    history: list[str] = []
    for seg in segments[:-1]:                  # current street excluded
        last_aggressor: str | None = None
        for actor, label in seg:
            verb = label.split()[0] if label else ""
            if verb in _BET_VERBS or verb in _RAISE_VERBS:
                if actor == "OOP":
                    last_aggressor = "hero" if hero_is_oop else "villain"
                elif actor == "IP":
                    last_aggressor = "villain" if hero_is_oop else "hero"
        history.append(last_aggressor or "check")
    return history


__all__ = [
    "ARCHETYPE_GUIDANCE",
    "RECOMMENDED_ACTION_ARCHETYPES",
    "aggression_history_from_action_sequence",
    "classify_recommended_archetype",
]
