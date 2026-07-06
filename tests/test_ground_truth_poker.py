"""GROUND-TRUTH tier: poker facts asserted from OUTSIDE the pipeline.

Every other test in this suite verifies the code against itself (a rebuild,
a fixture derived from the same logic). These assertions encode facts a
strong player states on sight -- table mechanics, domination direction,
equity landmarks, implied-odds logic -- so a bug that is INTERNALLY
consistent still fails here. This tier exists because three shipped bugs
(the blind-vs-blind SB labeled in position, an empty dominated_by list for
A8o vs a button open, reverse implied odds "against" an all-in) were each
perfectly self-consistent and passed every re-verification audit.

RULES FOR THIS FILE:
  * A failing test here means THE CODE IS WRONG. Never "fix" a test to
    match the code without independently confirming the poker fact.
  * Facts only -- no strategy opinions, no solver-frequency assertions.
  * Each test says the fact in plain English in its docstring.

Equity landmarks use the Monte-Carlo engine with a fixed seed and wide
tolerances (well outside sampling noise), so they pin the ENGINE's
correctness, not exact decimals.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random  # noqa: E402
from dataclasses import replace  # noqa: E402
from itertools import combinations  # noqa: E402

import pytest  # noqa: E402

from pipeline.preflop.concept_tags import (  # noqa: E402
    reverse_implied_odds,
)
from pipeline.preflop.domination import (  # noqa: E402
    classify_matchup,
    dominating_map,
)
from pipeline.preflop.fact_extractor import (  # noqa: E402
    compute_hero_equity_vs_range,
)
from pipeline.preflop.position import ip_oop_positions  # noqa: E402
from pipeline.preflop.position import (  # noqa: E402
    hero_relative_position as nlhe_relative_position,
)

_SUITS = "shdc"
_ALL_RANKS = "23456789TJQKA"


def _combos_for_class(hand_class: str) -> dict[str, float]:
    """Expand a 169-class to its concrete combos at weight 1.0."""
    out: dict[str, float] = {}
    if len(hand_class) == 2:  # pair
        r = hand_class[0]
        for a, b in combinations(_SUITS, 2):
            out[f"{r}{a}{r}{b}"] = 1.0
        return out
    hi, lo, suffix = hand_class[0], hand_class[1], hand_class[2]
    for a in _SUITS:
        for b in _SUITS:
            if (suffix == "s" and a == b) or (suffix == "o" and a != b):
                out[f"{hi}{a}{lo}{b}"] = 1.0
    return out


def _equity(hero_combo: str, villain_class: str) -> float:
    """Seeded MC equity of a hero combo vs every combo of one class
    (card-conflicting combos removed, as at a real table)."""
    hero_cards = {hero_combo[:2], hero_combo[2:]}
    villain = {
        c: w for c, w in _combos_for_class(villain_class).items()
        if not ({c[:2], c[2:]} & hero_cards)
    }
    return compute_hero_equity_vs_range(
        hero_combo, villain, max_runouts=3000, rng=random.Random(42)
    )


# =============================================================================
# 1. Table mechanics / position
# =============================================================================
def test_blind_vs_blind_bb_has_position() -> None:
    """At a ring table, postflop action starts left of the button: the SB
    acts first on every street, so in a blind-vs-blind pot the BB is the
    player in position. (The SB has the button only at a 2-player table.)"""
    assert ip_oop_positions("SB", "BB") == ("BB", "SB")


def test_button_has_position_on_every_seat() -> None:
    """The button acts last postflop against any other seat."""
    for seat in ("SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"):
        assert ip_oop_positions("BTN", seat) == ("BTN", seat)


def test_blinds_are_out_of_position_against_non_blind_seats() -> None:
    """Both blinds act before every other seat postflop."""
    for blind in ("SB", "BB"):
        for seat in ("UTG", "HJ", "CO", "BTN"):
            ip, oop = ip_oop_positions(blind, seat)
            assert ip == seat and oop == blind


def test_cutoff_has_position_on_hijack() -> None:
    """Later seats act later postflop: CO covers HJ, HJ covers LJ."""
    assert ip_oop_positions("CO", "HJ") == ("CO", "HJ")
    assert ip_oop_positions("HJ", "LJ") == ("HJ", "LJ")


# =============================================================================
# 2. Domination direction (structural, no equity needed)
# =============================================================================
def test_bigger_kicker_dominates_shared_high_card() -> None:
    """AK dominates AQ; A9 dominates A8. Sharing the high card with a
    worse kicker is the definition of being dominated."""
    assert classify_matchup("AQo", "AKo") == "dominates_you"
    assert classify_matchup("A8o", "A9o") == "dominates_you"
    assert classify_matchup("AKo", "AQo") == "you_dominate"


def test_pair_of_your_kicker_dominates_you() -> None:
    """KTo against TT is crushed: the pair holds one of your cards."""
    assert classify_matchup("KTo", "TT") == "dominates_you"


def test_bigger_pair_dominates_smaller_pair() -> None:
    """QQ against KK is the classic cooler, not a live-cards race."""
    assert classify_matchup("QQ", "KK") == "dominates_you"
    assert classify_matchup("KK", "QQ") == "you_dominate"


def test_underpair_vs_two_overcards_is_a_flip_not_domination() -> None:
    """22 vs AKo is the coinflip; neither hand is dominated."""
    assert classify_matchup("22", "AKo") == "flip"


def test_live_ace_behind_a_pair_is_behind_not_dominated() -> None:
    """A9 against QQ is behind but LIVE (the ace outdraws the pair
    directly); that is different from domination and the distinction is
    the lesson."""
    assert classify_matchup("A9o", "QQ") == "behind"


def test_a8o_vs_a_button_open_lists_the_better_aces_as_dominators() -> None:
    """The bug your Review QC caught: against a range holding A9o-AKo and
    AA, A8o's dominated_by list must contain them -- an empty list here is
    always wrong."""
    btn_open = [
        "AKo", "AQo", "AJo", "ATo", "A9o", "A7o", "A2o",
        "KQo", "QJs", "T9s", "AA", "66",
    ]
    dom = dominating_map("A8o", btn_open)
    assert set(dom["dominated_by"]) >= {"AKo", "AQo", "AJo", "ATo", "A9o"}
    assert "A7o" in dom["you_dominate"]
    assert "A2o" in dom["you_dominate"]


# =============================================================================
# 3. Equity landmarks (the numbers a player quotes from memory)
# =============================================================================
def test_aces_vs_kings_is_roughly_four_to_one() -> None:
    """AA vs KK preflop is ~81-82% for the aces."""
    assert _equity("AhAd", "KK") == pytest.approx(0.815, abs=0.03)


def test_pair_vs_two_overcards_is_a_slight_favorite() -> None:
    """The classic race: 22 vs AKo is close to even, small edge to the
    pair (~52-53%)."""
    assert _equity("2c2d", "AKo") == pytest.approx(0.525, abs=0.035)


def test_ak_suited_vs_queens_is_a_slight_underdog() -> None:
    """AKs vs QQ: the pair leads ~54/46."""
    assert _equity("AsKs", "QQ") == pytest.approx(0.46, abs=0.035)


def test_dominated_ace_is_a_big_underdog() -> None:
    """A8o vs AKo (dominated, sharing the ace) wins only ~26-31%."""
    eq = _equity("Ah8c", "AKo")
    assert 0.22 <= eq <= 0.34


def test_dominating_hand_is_a_big_favorite() -> None:
    """The mirror: AKo vs A8-type hands is ~69-74%."""
    eq = _equity("AhKc", "A8o")
    assert 0.66 <= eq <= 0.78


def test_suited_adds_a_few_points_not_many() -> None:
    """Suitedness is worth ~2-4 points of equity, not more (a common
    human overestimate; the engine must not share it)."""
    suited = _equity("AsKs", "QQ")
    offsuit = _equity("AsKh", "QQ")
    assert 0.005 <= suited - offsuit <= 0.06


# =============================================================================
# 4. Implied-odds logic
# =============================================================================
def _rio_facts(**kwargs):
    """Reuse the concept-tag test fixture shapes without importing the
    other test module: minimal inline facts builder."""
    from pipeline.preflop.fact_extractor import (
        PreflopFacts,
        VillainRangeStats,
    )
    from pipeline.preflop.grammars.types import (
        ParsedAction,
        PreflopActionType,
    )
    from pipeline.preflop.node_enumerator import PreflopDecisionNode
    from pipeline.preflop.spot_sampler import PreflopSpot

    history = kwargs.get("history") or (
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
    )
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor=kwargs.get("actor", "BB"),
            history_before=history, actions=(),
        ),
        hero_hand_class=kwargs.get("hand_class", "A4s"),
        hero_card_combo=kwargs.get("combo", "As4s"),
        action_frequencies={"Fold": 1.0},
        dominant_action="Fold",
        dominant_frequency=1.0,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="HJ", action_label=kwargs.get("action_label", "Raise 60%"),
            weighted_combo_count=kwargs.get("pct", 8.0) / 100.0 * 1326.0,
            pct_of_dealt_hands=kwargs.get("pct", 8.0),
        ),
        hero_equity_vs_villain=0.35,
        archetype=kwargs.get("archetype", "fold_pot_odds"),
    )


def test_no_implied_odds_of_any_kind_against_an_all_in() -> None:
    """Implied odds (and reverse implied odds) are about money won or lost
    on LATER streets. Facing an all-in there are no later betting
    decisions, so no hand can be a reverse-implied-odds fold there."""
    from pipeline.preflop.grammars.types import (
        ParsedAction,
        PreflopActionType,
    )

    jam = (ParsedAction("HJ", PreflopActionType.ALL_IN),)
    assert reverse_implied_odds(_rio_facts(history=jam)) is False


def test_a_hand_cannot_have_positive_and_reverse_implied_odds_at_once() -> None:
    """If the correct call is FOR implied odds, the same hand cannot also
    be tagged as a reverse-implied-odds hand in that spot."""
    assert (
        reverse_implied_odds(_rio_facts(archetype="call_for_implied_odds"))
        is False
    )


def test_weak_ace_vs_tight_range_is_the_textbook_rio_spot() -> None:
    """A4s against a tight (~8%) early open: flops top pair second-best.
    This is THE reverse-implied-odds example every coach uses."""
    assert reverse_implied_odds(_rio_facts(pct=8.0)) is True


# =============================================================================
# 5. Relative-position facts end to end
# =============================================================================
def test_sb_open_spot_is_out_of_position() -> None:
    """An SB first-in open plays the whole hand out of position: the BB
    is behind preflop AND acts later on every postflop street."""
    facts = replace(_rio_facts(actor="SB", history=()), villain_stats=None)
    assert nlhe_relative_position(facts) == "Out of Position"


def test_button_open_spot_is_in_position() -> None:
    """A BTN open is the only first-in open guaranteed position postflop."""
    facts = replace(_rio_facts(actor="BTN", history=()), villain_stats=None)
    assert nlhe_relative_position(facts) == "In Position"
