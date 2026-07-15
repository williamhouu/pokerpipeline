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

import re
from collections.abc import Callable

from pipeline.postflop.facts import PostflopFacts

SkillRule = Callable[[PostflopFacts], bool]

# A dealt chance card in a node-id line (e.g. ``2c``) -- marks a street boundary.
_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")


# --- helpers ----------------------------------------------------------------
def _dominant_action(facts: PostflopFacts):
    return facts.spot.node.action_by_label(facts.dominant_action)


def _current_street_tokens(node_id: str) -> list[str]:
    """The action tokens on the CURRENT (last) street of a node-id line.

    A node id is ``r:0:<flop tokens>[:<turn card>:<turn tokens>...]``; a dealt
    chance card (``2c``) marks a street boundary, so the current street is the
    run of tokens after the last card. ``[]`` for the synthetic fixtures (their
    ids carry no ``:`` tokens). NB this solve encodes BOTH a bet and a raise as a
    ``b<size>`` token (a raise is just a second ``b`` -- there is no ``r``
    token), so two ``b`` tokens on one street == bet-then-raise."""
    seg: list[str] = []
    for token in node_id.split(":")[2:]:
        if _CARD_RE.match(token):
            seg = []
        else:
            seg.append(token)
    return seg


def _faces_check_raise_line(node_id: str) -> bool:
    """True iff hero faces a CHECK-RAISE: the current street ran check, bet,
    raise (``[c, b, b]``) and hero is now to act on the raise.

    Villain checked (a street can't open with a call, so the leading ``c`` is a
    check), hero bet, villain raised -- so hero is the original bettor (always in
    position here) facing the check-raise. Exactly three tokens excludes both a
    donk-lead-then-raise (starts with ``b``, not a check-raise) and a deeper
    re-raise war (4+ tokens)."""
    seg = _current_street_tokens(node_id)
    return (
        len(seg) == 3  # noqa: PLR2004 -- check, bet, raise
        and seg[0] == "c"
        and seg[1].startswith("b")
        and seg[2].startswith("b")
    )


# Dominated made hands prone to reverse implied odds: a second-best top pair
# (out-kicked), a non-nut made flush, or the ignorant end of a straight -- hands
# that win small pots and lose big ones. (A non-nut flush DRAW is handled
# separately via the draw list.)
_RIO_DOMINATED_MADE = frozenset(
    {"top_pair_weak_kicker", "flush_weak", "straight_weak"}
)


def _reverse_implied_odds(f: PostflopFacts) -> bool:
    """Hero holds a hand prone to reverse implied odds -- wins small, loses big.

    Two cases: a non-nut (weak) flush DRAW carries reverse implied odds in
    itself (make it, still lose to a higher flush) regardless of the action; a
    dominated MADE hand (second-best top pair / non-nut flush / ignorant-end
    straight) only has RIO exposure when there is a bet to pay off -- so it must
    be FACING a bet (betting a weak top pair thinly is not the RIO lesson).

    Excluded when the spot is a clean drawing call (``call_drawing`` = Implied
    Odds, the opposite concept), so the two stay disjoint -- mirroring the
    preflop tagger's fold-gating of its RIO skill. Also excluded when the
    wager hero faces is an ALL-IN: with no further betting possible there
    are no implied odds in either direction (the July-2026 preflop rule,
    ported after the full-hand cross-check flagged a weak flush facing a
    river jam tagged RIO). The batch cross-check enforces the same rule,
    so tagger and verifier must stay in lockstep."""
    if f.archetype == "call_drawing":
        return False
    history = f.spot.node.history
    if history and history[-1].all_in:
        return False  # facing an all-in: no future betting either way
    if "flush_draw_weak" in f.draws:
        return True
    return (
        f.made_hand in _RIO_DOMINATED_MADE
        and "facing_bet_spot" in f.concept_tags
    )


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
    """The node offers >=2 distinct bet/raise sizes -- a real sizing choice.

    Reads ``live_actions`` (artifact-strip): a stripped tree-artifact jam is
    not a real size the player is choosing between."""
    sizes = {a.label for a in facts.spot.live_actions if a.verb in ("bet", "raise")}
    return len(sizes) >= 2


# --- MDF (minimum defense frequency) ----------------------------------------
# MDF = pot / (pot + bet): the share of your range you must continue against a
# bet, or villain can profitably bluff any two cards. Pure formula, deterministic.
_MDF_MIN = 0.5             # MDF >= 50% (bet <= pot): a "defend WIDE" spot
_MDF_BUBBLE_MARGIN = 0.08  # hero's equity within 8 pts of the calling price


def _mdf_threshold(facts: PostflopFacts) -> float:
    """MDF = pot_before / (pot_before + bet). ``pot_bb`` already includes the bet,
    so pot_before = ``pot_bb - to_call_bb`` and bet = ``to_call_bb``."""
    bet = facts.to_call_bb
    pot_before = facts.pot_bb - bet
    denom = pot_before + bet
    return pot_before / denom if denom > 0 else 0.0


def _mdf_is_the_lesson(facts: PostflopFacts) -> bool:
    """True when minimum-defense-frequency is the operative concept (done properly,
    not a strength-bucket proxy): the bet is small/medium so MDF is high (you must
    defend WIDE -- over-folding is the leak), AND this hand is a BORDERLINE
    defender -- its equity sits within ``_MDF_BUBBLE_MARGIN`` of the calling price,
    so whether you defend IT is the decision that determines if you meet MDF.
    Against an overbet (MDF < ``_MDF_MIN``) it does not fire -- that is a pot-odds
    / strength decision, not a defend-wide one."""
    if facts.break_even_equity is None or facts.to_call_bb <= 0:
        return False
    if _mdf_threshold(facts) < _MDF_MIN:
        return False
    return abs(facts.hero_equity_vs_villain - facts.break_even_equity) <= _MDF_BUBBLE_MARGIN


# --- the catalog (postflop-relevant skills; names == the app's catalog) ------
# Only skills a postflop spot can clearly test are listed. Preflop-only skills
# (3-Betting, Squeezing, ...) and signals we still can't derive cleanly
# (Combinatorics, Hand Reading, Equity Realization, ICM) are intentionally
# absent -- a strict-tagging choice. (Facing a Check-Raise, MDF, Reverse Implied
# Odds, Blockers & Card Removal, and Facing a Probe Bet were added June 2026 once
# each had a clean deterministic signal -- Blockers once the value/bluff
# decomposition gave postflop real blocker data.)
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
    # The mirror: hero bet and villain check-raised, so hero (the bettor, always
    # IP here) now faces the raise. Detected from the betting line, not hero's
    # action, so it can't leak the answer. Disjoint from "Check-Raising" (that
    # needs hero OOP; a check-raise hero faces always has hero IP).
    "Facing a Check-Raise": lambda f: (
        f.spot.node.is_facing_bet
        and _faces_check_raise_line(f.spot.node.node_id)
    ),
    "Donk Betting": lambda f: "donk_bet_spot" in f.concept_tags,
    # Hero (the preflop raiser) faces a flop bet => villain led into him (a donk).
    "Facing a Donk Bet": lambda f: (
        "facing_bet_spot" in f.concept_tags
        and f.hero_is_preflop_aggressor
        and f.street == "flop"
    ),
    "Probe Betting": lambda f: "probe_bet" in f.concept_tags,
    # The mirror: hero (the aggressor who checked back) faces villain's later-
    # street lead -- a probe bet into the player who showed weakness.
    "Facing a Probe Bet": lambda f: "facing_probe_spot" in f.concept_tags,
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
    # MDF, done PROPERLY (deterministic, no LLM): fire on the BUBBLE defenders
    # against a bet you must defend WIDE -- a call/fold decision where the bet is
    # small enough that MDF is high AND this hand sits right at the calling-price
    # threshold (see _mdf_is_the_lesson). Computes the real MDF from the bet size
    # instead of the old strength-bucket proxy.
    "Minimum Defense Frequency (MDF)": lambda f: (
        "facing_bet_spot" in f.concept_tags
        and f.dominant_verb in ("call", "fold")
        and _mdf_is_the_lesson(f)
    ),
    "Implied Odds": lambda f: f.archetype == "call_drawing",
    "Reverse Implied Odds": _reverse_implied_odds,
    # Low SPR turns a made hand into a commitment decision -- where SPR IS the
    # lesson. (High-SPR playability is real too but harder to isolate cleanly.)
    "Stack-to-Pot Ratio (SPR)": lambda f: (
        f.spr <= _LOW_SPR
        and f.strength_bucket in ("premium", "strong", "medium")
    ),
    # --- Section 5: Hand Analysis & Decision Making ---
    # An overbet (by hero or villain) is THE polarized-range situation.
    "Range Polarization": lambda f: _is_overbet(f) or _facing_overbet(f),
    # Hero's cards meaningfully remove villain's VALUE or BLUFF combos AND the spot
    # is one where that removal DRIVES the decision: facing a bet (a bluff-catch --
    # "you block their value, so call") OR bluffing (you block their continues /
    # value, so the bet gets through -- e.g. the nut-flush blocker that lets you
    # keep barrelling air). NOT facing-bet-only (an earlier version was, which
    # missed the bluff case). Excludes value bets / pure checks where the blocker
    # is incidental, so it stays a useful filter rather than firing on every
    # non-neutral spot. Postflop DOES carry blocker data now (the value/bluff
    # decomposition); the old "no blocker data" note was stale.
    "Blockers & Card Removal": lambda f: (
        # Facing a bet (bluff-catch): blocking EITHER their value (-> call) or
        # their bluffs (-> fold) drives the call/fold.
        ("facing_bet_spot" in f.concept_tags and f.blocker_effect in ("value", "bluffs"))
        # Bluffing: only blocking their VALUE enables the bluff (you remove their
        # strong calls, so the bet gets through -- the nut-flush-blocker barrel);
        # blocking their bluffs is irrelevant to a bluff (those hands fold anyway).
        or (
            f.blocker_effect == "value"
            and ("bluff_spot" in f.concept_tags or f.archetype in ("bluff", "bluff_raise"))
        )
    ),
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
    "Facing a Check-Raise": "The betting line ran check, bet, raise on this street, "
    "so hero bet and villain check-raised and hero now faces the raise (read from "
    "the line, not hero's action).",
    "Donk Betting": "Hero leads into the preflop raiser from out of position (the "
    "`donk_bet_spot` tag).",
    "Facing a Donk Bet": "Hero IS the preflop raiser and faces a flop bet, so "
    "villain has led (donked) into him.",
    "Probe Betting": "Hero bets a later street after the aggressor declined to "
    "c-bet the previous street (the `probe_bet` tag).",
    "Facing a Probe Bet": "Hero is the preflop aggressor who checked back the "
    "prior street and now faces the out-of-position player's lead on the "
    "turn/river (the `facing_probe_spot` tag).",
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
    "Minimum Defense Frequency (MDF)": "Hero faces a small/medium bet (one you "
    "must defend WIDE, MDF >= 50%) with a borderline hand whose equity sits right "
    "at the calling price, so whether to defend it is the minimum-defense-frequency "
    "decision -- defend enough to deny villain a profitable any-two bluff.",
    "Implied Odds": "Hero calls a draw getting the right price plus future-street "
    "value (a call_drawing frame).",
    "Reverse Implied Odds": "Hero has a non-nut flush draw, or a dominated made "
    "hand (second-best top pair / non-nut flush or straight) while FACING A BET -- "
    "hands that win small and lose big (excluded on a clean drawing call).",
    "Stack-to-Pot Ratio (SPR)": f"SPR is low (<= {_LOW_SPR}) with a made hand "
    "(premium/strong/medium), so commitment is driven by the stack-to-pot ratio.",
    "Range Polarization": "An overbet is in play (hero's or villain's) -- the "
    "textbook polarized-range situation.",
    "Blockers & Card Removal": "Hero's cards meaningfully remove villain's value "
    "or bluff combos in a spot where it matters: facing a bet (a bluff-catch) OR "
    "bluffing (your blocker -- e.g. the nut-flush blocker -- removes their "
    "continues so the bet works).",
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
    "Combinatorics": "Nearly every spot involves combo counting; needs a narrower "
    "trigger.",
    "Equity Realization": "Too broad without a realization-gap signal.",
    "Hand Reading": "Universal in poker; would fire on every spot.",
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
