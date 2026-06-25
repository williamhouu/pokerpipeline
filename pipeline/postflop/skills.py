"""Postflop user-facing skill tagger (Phase 3, postflop-native).

Maps a :class:`~pipeline.postflop.facts.PostflopFacts` block onto the app's
skill catalog -- the labels the app surfaces for "study X" drills and progress
tracking ("skills" and "tags" are the same thing in the app's wording).

This is the postflop analogue of :mod:`pipeline.skill_tagger`. It is kept HERE,
in the self-contained postflop package, rather than extending the shared tagger,
because the shared tagger's postflop rules read the OLD
``pipeline.fact_extractor`` concept-tag vocabulary, while this pipeline emits its
own (``pipeline.postflop.concept_tags``). The skill NAMES are the app's
canonical names (identical to the shared tagger's catalog keys -- a parity test
guards against drift), so the ``skills`` CSV column the app consumes is the same
whether a row came from the preflop or postflop path.

Every rule is a pure, deterministic predicate over the facts (never the LLM).
Strict by design: a skill fires only when the spot clearly tests it, so a
typical postflop question gets ~2-5 skills, not a dozen. False negatives beat
noise. Each rule has a plain-English explainer in
:data:`POSTFLOP_SKILL_EXPLAINERS`, surfaced in the admin panel so a reviewer can
see exactly why each skill is (or isn't) tagged.
"""

from __future__ import annotations

from collections.abc import Callable

from pipeline.postflop.facts import PostflopFacts

SkillRule = Callable[[PostflopFacts], bool]


# --- helpers ----------------------------------------------------------------
def _dominant_action(facts: PostflopFacts):
    return facts.spot.node.action_by_label(facts.dominant_action)


def _is_overbet(facts: PostflopFacts) -> bool:
    """Hero's chosen bet/raise is larger than the pot (pot_fraction > 1)."""
    if facts.dominant_verb not in ("bet", "raise"):
        return False
    a = _dominant_action(facts)
    return a is not None and a.pot_fraction is not None and a.pot_fraction > 1.0


def _facing_overbet(facts: PostflopFacts) -> bool:
    """The bet hero faces is larger than the pot it was bet into.

    Heads-up: ``pot_bb`` already includes villain's bet ``B`` (= ``to_call_bb``),
    so the pot before the bet was ``pot_bb - B``; an overbet is ``B > pot_bb - B``.
    """
    b = facts.to_call_bb
    return b > 0 and b > (facts.pot_bb - b) + 1e-9


def _multiple_bet_sizes(facts: PostflopFacts) -> bool:
    """The node offers >=2 distinct bet/raise sizes -- a real sizing choice."""
    sizes = {a.label for a in facts.spot.node.actions if a.verb in ("bet", "raise")}
    return len(sizes) >= 2


# --- the catalog (postflop-relevant skills; names == the app's catalog) ------
# Only skills a postflop spot can clearly test are listed. Preflop-only skills
# (3-Betting, Squeezing, ...) and signals we can't derive yet (Facing a
# Check-Raise, MDF, Combinatorics, Hand Reading, Reverse Implied Odds, Equity
# Realization, ICM) are intentionally absent -- a strict-tagging choice.
POSTFLOP_SKILL_RULES: dict[str, SkillRule] = {
    # --- Section 2: Betting & Aggression ---
    "C-Betting": lambda f: "c_bet_spot" in f.concept_tags,
    "Facing a C-Bet": lambda f: (
        "facing_bet_spot" in f.concept_tags
        and not f.hero_is_preflop_aggressor
        and f.street == "flop"
    ),
    # OOP raising a bet is a check-raise (hero checked, villain bet, hero raises).
    "Check-Raising": lambda f: (
        f.dominant_verb == "raise"
        and "facing_bet_spot" in f.concept_tags
        and not f.hero_in_position
    ),
    "Donk Betting": lambda f: "donk_bet_spot" in f.concept_tags,
    # Hero (the preflop raiser) faces a flop bet => villain led into him (a donk).
    "Facing a Donk Bet": lambda f: (
        "facing_bet_spot" in f.concept_tags
        and f.hero_is_preflop_aggressor
        and f.street == "flop"
    ),
    "Probe Betting": lambda f: "probe_bet" in f.concept_tags,
    "Overbetting": _is_overbet,
    "Facing an Overbet": _facing_overbet,
    "Bet Sizing": lambda f: (
        f.dominant_verb in ("bet", "raise") and _multiple_bet_sizes(f)
    ),
    "Value Betting": lambda f: (
        "value_bet_spot" in f.concept_tags
        or f.archetype in ("value_bet", "value_raise")
    ),
    "Bluffing": lambda f: (
        "bluff_spot" in f.concept_tags or f.archetype in ("bluff", "bluff_raise")
    ),
    # --- Section 3: Defense & Response ---
    "Bluff Catching": lambda f: (
        "bluff_catch_spot" in f.concept_tags or f.archetype == "bluff_catch"
    ),
    # Floating: call IN POSITION with a weak, non-drawing hand to take it later
    # (a real draw would make it a semibluff/draw-call, not a pure float).
    "Floating": lambda f: (
        f.dominant_verb == "call"
        and f.hero_in_position
        and f.street == "flop"
        and f.strength_bucket in ("air", "marginal")
        and not f.has_strong_draw
    ),
    "Pot Control": lambda f: (
        "pot_control_spot" in f.concept_tags or f.archetype == "pot_control_check"
    ),
    # --- Section 4: Math & Theory ---
    "Pot Odds": lambda f: (
        "facing_bet_spot" in f.concept_tags and f.dominant_verb in ("call", "fold")
    ),
    "Implied Odds": lambda f: f.archetype == "call_drawing",
    # Low SPR turns a made hand into a commitment decision -- where SPR IS the
    # lesson. (High-SPR playability is real too but harder to isolate cleanly.)
    "Stack-to-Pot Ratio (SPR)": lambda f: (
        f.spr <= _LOW_SPR
        and f.strength_bucket in ("premium", "strong", "medium")
    ),
    # --- Section 5: Hand Analysis & Decision Making ---
    # An overbet (by hero or villain) is THE polarized-range situation.
    "Range Polarization": lambda f: _is_overbet(f) or _facing_overbet(f),
    # --- Section 6: Positional & Situational ---
    # Using position: checking back / calling in position to control + realize.
    "In Position Play": lambda f: (
        f.hero_in_position and f.dominant_verb in ("check", "call")
    ),
    # Navigating OOP: defending a bet, check-raising, or ceding initiative.
    "Out of Position Play": lambda f: (
        not f.hero_in_position
        and ("facing_bet_spot" in f.concept_tags or f.dominant_verb in ("check", "raise"))
    ),
    "Multiway Pot Strategy": lambda f: "multiway_pot" in f.concept_tags,
    # A genuine drawing hand: a real draw (not a backdoor / bare gutshot) that is
    # the hand's main value (not already a strong+ made hand).
    "Drawing Hand Strategy": lambda f: (
        f.has_strong_draw
        and f.strength_bucket in ("air", "marginal", "vulnerable", "medium")
    ),
}

# SPR threshold below which the spot is a commitment decision (SPR is the lesson).
_LOW_SPR = 3.5

# Tournament skills need the game format, which is on the solve, not the facts --
# computed separately in compute_postflop_skills.
_SHORT_STACK_BB = 25.0


# Plain-English "exactly how this skill is tagged", for the admin panel. One
# line per rule above, in catalog order.
POSTFLOP_SKILL_EXPLAINERS: dict[str, str] = {
    "C-Betting": "Hero is the preflop raiser and bets the flop (the `c_bet_spot` tag).",
    "Facing a C-Bet": "Hero is NOT the preflop raiser and faces a bet on the flop "
    "(the bettor is the raiser, so it's a c-bet).",
    "Check-Raising": "Hero is out of position and raises a bet (checked, villain "
    "bet, hero raises).",
    "Donk Betting": "Hero leads into the preflop raiser from out of position (the "
    "`donk_bet_spot` tag).",
    "Facing a Donk Bet": "Hero IS the preflop raiser and faces a flop bet, so "
    "villain has led (donked) into him.",
    "Probe Betting": "Hero bets a later street after the aggressor declined to "
    "c-bet the previous street (the `probe_bet` tag).",
    "Overbetting": "Hero's chosen bet or raise is larger than the pot (size > "
    "100% pot).",
    "Facing an Overbet": "The bet hero faces is larger than the pot it was bet into.",
    "Bet Sizing": "Hero bets or raises AND the solver offered two or more distinct "
    "sizes here, so the sizing choice is a real decision.",
    "Value Betting": "Hero bets/raises a strong made hand for value (the "
    "`value_bet_spot` tag or a value_bet / value_raise frame).",
    "Bluffing": "Hero bets/raises a hand that needs folds to win (the `bluff_spot` "
    "tag or a bluff / bluff_raise frame).",
    "Bluff Catching": "Hero calls with a hand that mainly beats bluffs (the "
    "`bluff_catch_spot` tag or a bluff_catch frame).",
    "Floating": "Hero calls a flop bet IN POSITION with a weak hand and NO real "
    "draw, planning to take the pot later.",
    "Pot Control": "Hero checks a medium hand to keep the pot small (the "
    "`pot_control_spot` tag or a pot_control_check frame).",
    "Pot Odds": "Hero faces a bet and the decision is call-or-fold, so the price "
    "is the core math.",
    "Implied Odds": "Hero calls a draw getting the right price plus future-street "
    "value (a call_drawing frame).",
    "Stack-to-Pot Ratio (SPR)": f"SPR is low (<= {_LOW_SPR}) with a made hand "
    "(premium/strong/medium), so commitment is driven by the stack-to-pot ratio.",
    "Range Polarization": "An overbet is in play (hero's or villain's) -- the "
    "textbook polarized-range situation.",
    "In Position Play": "Hero acts last and checks back or calls, using position "
    "to control the pot and realize equity.",
    "Out of Position Play": "Hero acts first and is defending a bet, check-raising, "
    "or ceding the betting lead.",
    "Multiway Pot Strategy": "Three or more players are in the pot (the "
    "`multiway_pot` tag).",
    "Drawing Hand Strategy": "Hero holds a real draw (flush draw, open-ended "
    "straight draw, or combo draw -- not a backdoor or bare gutshot) as the "
    "hand's main value.",
    "Short Stack Tournament Strategy": "Tournament format AND effective stack "
    f"<= {_SHORT_STACK_BB:g}bb -- push/fold-flavoured short-stack play.",
    "Tournament Blind vs. Blind": "Tournament format AND the pot is contested "
    "only between the small blind and big blind.",
}

# Skills the postflop path deliberately does NOT tag yet (no clean signal),
# shown in the admin explainer so the absence is documented, not a mystery.
POSTFLOP_SKILLS_NOT_TAGGED: dict[str, str] = {
    "Facing a Check-Raise": "Needs hero-bet-then-villain-raised detection from "
    "the street history (not yet exposed cleanly).",
    "Facing a Probe Bet": "Needs prior-street check-back detection on hero's side.",
    "Reverse Implied Odds": "No deterministic postflop signal yet (dominated-draw "
    "detection).",
    "Minimum Defense Frequency (MDF)": "No defense-threshold signal computed "
    "postflop yet.",
    "Combinatorics": "Nearly every spot involves combo counting; needs a narrower "
    "trigger.",
    "Equity Realization": "Too broad without a realization-gap signal.",
    "Hand Reading": "Universal in poker; would fire on every spot.",
    "Blockers & Card Removal": "Postflop spots ship NO blocker data, so blocker "
    "claims are never tagged (or made).",
    "ICM & Tournament Pressure": "Needs tournament-structure metadata (payouts, "
    "stacks) the solves don't carry.",
}


def compute_postflop_skills(
    facts: PostflopFacts, *, game_format: str = "cash"
) -> list[str]:
    """Every app skill that fires for this postflop spot, in catalog order.

    ``game_format`` (from the solve) gates the tournament skills; cash solves
    never tag them. Deterministic and pure -- safe for the byte-identical-CSV
    guarantee.
    """
    skills = [name for name, rule in POSTFLOP_SKILL_RULES.items() if rule(facts)]
    if game_format == "tournament":
        if facts.spot.node.effective_stack_bb <= _SHORT_STACK_BB:
            skills.append("Short Stack Tournament Strategy")
        if {facts.hero_position, facts.villain_position} == {"SB", "BB"}:
            skills.append("Tournament Blind vs. Blind")
    return skills


__all__ = [
    "POSTFLOP_SKILLS_NOT_TAGGED",
    "POSTFLOP_SKILL_EXPLAINERS",
    "POSTFLOP_SKILL_RULES",
    "compute_postflop_skills",
]
