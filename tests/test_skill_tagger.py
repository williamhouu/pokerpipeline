"""Tests for pipeline.skill_tagger.

Per skill, the test pattern is:
  1. construct a SkillContext that should fire the rule (positive)
  2. construct one that should NOT fire it (negative -- usually flipping
     one input)
  3. assert ``compute_skills`` includes / excludes the skill name

For ``# TODO Phase 4`` rules (off until further work), there's a single
test asserting the rule always returns False, so when the work lands and
the rule starts firing, the test fails loudly as a reminder to update
both the rule and its test.

The from_preflop_facts adapter has integration tests against minimal
real PreflopFacts at the bottom.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.skill_tagger import (  # noqa: E402
    SKILL_CATALOG,
    SkillContext,
    compute_skills,
)


# --- tiny helper ----------------------------------------------------------
def _ctx(**overrides: object) -> SkillContext:
    """Build a SkillContext with sensible preflop defaults + overrides.

    Default is an open-fold spot (BTN, no prior raises). Override
    whatever's needed for the specific test.
    """
    defaults: dict[str, object] = {
        "path": "preflop",
        "street": "Preflop",
        "concept_tags": frozenset(),
        "archetype": "",
        "hand_class": "AKo",
        "hero_position": "BTN",
        "game_format": "cash",
        "stack_depth_bb": 100,
        "n_prior_raises": 0,
        "n_calls_after_open": 0,
    }
    defaults.update(overrides)
    return SkillContext(**defaults)  # type: ignore[arg-type]


def _has(skill: str, ctx: SkillContext) -> bool:
    return skill in compute_skills(ctx)


# --- Section 1: Preflop (9) -----------------------------------------------
def test_preflop_hand_selection_fires_on_opens() -> None:
    assert _has("Preflop Hand Selection", _ctx(archetype="open_for_value"))
    assert _has("Preflop Hand Selection", _ctx(archetype="fold_outranged"))


def test_preflop_hand_selection_negative_on_3bet_spot() -> None:
    """3-bet spots get '3-Betting', not 'Preflop Hand Selection'."""
    assert not _has(
        "Preflop Hand Selection",
        _ctx(archetype="3bet_for_value", concept_tags=frozenset({"facing_single_raise"})),
    )


def test_3_betting_fires_on_3bet_archetypes() -> None:
    assert _has("3-Betting", _ctx(archetype="3bet_for_value"))
    assert _has("3-Betting", _ctx(archetype="3bet_as_bluff"))


def test_facing_3bet_fires_no_squeeze() -> None:
    assert _has(
        "Facing a 3-Bet",
        _ctx(concept_tags=frozenset({"facing_3bet"}), n_prior_raises=2, n_calls_after_open=0),
    )


def test_facing_3bet_excluded_when_squeeze() -> None:
    """When there was a caller between open and 3-bet, this is a squeeze
    response -- Facing a Squeeze fires instead."""
    ctx = _ctx(
        concept_tags=frozenset({"facing_3bet"}),
        n_prior_raises=2,
        n_calls_after_open=1,
    )
    assert not _has("Facing a 3-Bet", ctx)
    assert _has("Facing a Squeeze", ctx)


def test_4_betting() -> None:
    assert _has("4-Betting", _ctx(archetype="4bet_for_value"))
    assert _has("4-Betting", _ctx(archetype="4bet_as_bluff"))


def test_facing_4bet() -> None:
    assert _has("Facing a 4-Bet", _ctx(concept_tags=frozenset({"facing_4bet_plus"})))


def test_squeezing_fires_on_squeeze_archetypes() -> None:
    assert _has("Squeezing", _ctx(archetype="squeeze_for_value"))
    assert _has("Squeezing", _ctx(archetype="squeeze_as_bluff"))


def test_blind_defense_fires_in_blinds_facing_raise() -> None:
    assert _has(
        "Blind Defense",
        _ctx(concept_tags=frozenset({"big_blind", "facing_single_raise"})),
    )
    assert _has(
        "Blind Defense",
        _ctx(concept_tags=frozenset({"small_blind", "facing_3bet"})),
    )


def test_blind_defense_excluded_for_bvb() -> None:
    """BvB spots get their own skill, not Blind Defense."""
    ctx = _ctx(
        concept_tags=frozenset({"small_blind", "facing_single_raise", "bvb_spot"}),
    )
    assert not _has("Blind Defense", ctx)
    assert _has("Blind vs. Blind Play", ctx)


def test_blind_defense_excluded_for_non_blind() -> None:
    """BTN facing a raise = 3-betting / Facing a 3-Bet territory, not
    Blind Defense."""
    assert not _has(
        "Blind Defense",
        _ctx(concept_tags=frozenset({"facing_single_raise"})),
    )


def test_blind_vs_blind() -> None:
    assert _has("Blind vs. Blind Play", _ctx(concept_tags=frozenset({"bvb_spot"})))


# --- Section 2: Betting & Aggression -- postflop placeholders ------------
def test_check_raise_fires_on_postflop_tag() -> None:
    assert _has(
        "Check-Raising",
        _ctx(path="postflop", street="Flop",
             concept_tags=frozenset({"check_raise_spot"})),
    )


def test_donk_and_overbet() -> None:
    assert _has(
        "Donk Betting",
        _ctx(path="postflop", concept_tags=frozenset({"donk_bet_spot"})),
    )
    assert _has(
        "Facing an Overbet",
        _ctx(path="postflop", concept_tags=frozenset({"facing_overbet_spot"})),
    )


def test_value_betting_fires_on_thin_value() -> None:
    assert _has(
        "Value Betting",
        _ctx(path="postflop", concept_tags=frozenset({"thin_value_spot"})),
    )
    assert _has(
        "Value Betting",
        _ctx(path="postflop", concept_tags=frozenset({"merged_value_spot"})),
    )


def test_bluffing_postflop_tag() -> None:
    """Original postflop trigger -- bluff_spot tag fires the skill."""
    assert _has(
        "Bluffing",
        _ctx(path="postflop", concept_tags=frozenset({"bluff_spot"})),
    )


def test_bluffing_preflop_3bet_as_bluff() -> None:
    """Preflop archetype `3bet_as_bluff` is an explicit bluff."""
    assert _has("Bluffing", _ctx(archetype="3bet_as_bluff"))


def test_bluffing_preflop_all_5_bluff_archetypes() -> None:
    """Strict trigger -- only the EXPLICIT _as_bluff archetypes qualify."""
    for archetype in (
        "3bet_as_bluff", "4bet_as_bluff", "5bet_as_bluff",
        "squeeze_as_bluff", "all_in_as_bluff",
    ):
        assert _has(
            "Bluffing", _ctx(archetype=archetype)
        ), f"bluff archetype {archetype!r} did not fire Bluffing"


def test_bluffing_does_not_fire_on_value_archetype_with_low_equity() -> None:
    """The point of strict tagging: a call_for_implied_odds spot with
    low equity is NOT a bluff. The value-frame archetype rules it out."""
    assert not _has(
        "Bluffing",
        _ctx(
            archetype="call_for_implied_odds",
            concept_tags=frozenset({"dominated"}),  # low equity
        ),
    )


def test_bluffing_does_not_fire_on_3bet_for_value() -> None:
    """3bet_for_value is a value 3-bet, NOT a bluff. Must not tag."""
    assert not _has("Bluffing", _ctx(archetype="3bet_for_value"))


def test_bluffing_does_not_fire_on_fold_archetypes() -> None:
    """Folding is never a bluff."""
    for archetype in ("fold_pot_odds", "fold_dominated", "fold_outranged"):
        assert not _has(
            "Bluffing", _ctx(archetype=archetype)
        ), f"fold archetype {archetype!r} should NOT fire Bluffing"


def test_bet_sizing_off_until_phase_4() -> None:
    """TODO marker: should always be False until we have a sizing-axis
    signal. When this test starts failing, add the real rule."""
    for path in ("preflop", "postflop"):
        for tags in (frozenset(), frozenset({"overbet_spot"}), frozenset({"thin_value_spot"})):
            assert not _has("Bet Sizing", _ctx(path=path, concept_tags=tags))


# --- Section 3: Defense & Response ---------------------------------------
def test_bluff_catching() -> None:
    assert _has(
        "Bluff Catching",
        _ctx(path="postflop", concept_tags=frozenset({"bluffcatch_spot"})),
    )


def test_floating_and_pot_control() -> None:
    assert _has(
        "Floating", _ctx(path="postflop", concept_tags=frozenset({"float_call_spot"}))
    )
    assert _has(
        "Pot Control", _ctx(path="postflop", concept_tags=frozenset({"pot_control_spot"}))
    )


# --- Section 4: Math & Theory --------------------------------------------
def test_pot_odds_fires_on_call_fold_archetypes() -> None:
    assert _has("Pot Odds", _ctx(archetype="call_for_value"))
    assert _has("Pot Odds", _ctx(archetype="fold_pot_odds"))
    assert _has("Pot Odds", _ctx(archetype="fold_dominated"))


def test_pot_odds_negative_on_3bet() -> None:
    assert not _has("Pot Odds", _ctx(archetype="3bet_for_value"))


def test_implied_odds_fires_on_archetype_or_postflop_tag() -> None:
    assert _has("Implied Odds", _ctx(archetype="call_for_implied_odds"))
    assert _has(
        "Implied Odds",
        _ctx(path="postflop", concept_tags=frozenset({"implied_odds_call"})),
    )


def test_reverse_implied_odds() -> None:
    # Postflop: the dedicated tag.
    assert _has(
        "Reverse Implied Odds",
        _ctx(path="postflop", concept_tags=frozenset({"reverse_implied_odds_call"})),
    )
    # Preflop: FOLDING a dominated, weak-OFFSUIT hand fires (the insight
    # drives the decision).
    assert _has(
        "Reverse Implied Odds",
        _ctx(
            archetype="fold_dominated",
            concept_tags=frozenset({"dominated", "unconnected_offsuit"}),
        ),
    )
    assert _has(
        "Reverse Implied Odds",
        _ctx(
            archetype="fold_pot_odds",
            concept_tags=frozenset({"dominated", "unconnected_offsuit"}),
        ),
    )
    # A correct dominated CALL does NOT fire (RIO fear points at the wrong
    # answer there; it keeps Implied Odds instead -- disjoint by construction).
    call_ctx = _ctx(
        archetype="call_for_implied_odds",
        concept_tags=frozenset({"dominated", "unconnected_offsuit"}),
    )
    assert not _has("Reverse Implied Odds", call_ctx)
    assert _has("Implied Odds", call_ctx)
    # A dominated PAIR is a set-miner (good implied odds) -> does NOT fire.
    assert not _has(
        "Reverse Implied Odds",
        _ctx(
            archetype="fold_dominated",
            concept_tags=frozenset({"dominated", "small_pair"}),
        ),
    )
    # A dominated SUITED hand has nut-draw potential -> does NOT fire.
    assert not _has(
        "Reverse Implied Odds",
        _ctx(
            archetype="fold_dominated",
            concept_tags=frozenset({"dominated", "suited_ace"}),
        ),
    )
    # Weak offsuit but NOT dominated (decent equity) -> does NOT fire.
    assert not _has(
        "Reverse Implied Odds",
        _ctx(
            archetype="fold_dominated",
            concept_tags=frozenset({"unconnected_offsuit"}),
        ),
    )


def test_mdf() -> None:
    assert _has(
        "Minimum Defense Frequency (MDF)",
        _ctx(path="postflop", concept_tags=frozenset({"mdf_defense_threshold"})),
    )


def test_combinatorics_off_until_phase_4() -> None:
    assert not _has("Combinatorics", _ctx(concept_tags=frozenset({"facing_3bet"})))


def test_equity_realization() -> None:
    assert _has(
        "Equity Realization",
        _ctx(path="postflop", concept_tags=frozenset({"equity_under_realized"})),
    )
    assert _has(
        "Equity Realization",
        _ctx(path="postflop", concept_tags=frozenset({"equity_over_realized"})),
    )


def test_spr_off_until_phase_4() -> None:
    assert not _has("Stack-to-Pot Ratio (SPR)", _ctx())


# --- Section 5: Hand Analysis --------------------------------------------
def test_hand_reading_off_until_phase_4() -> None:
    assert not _has("Hand Reading", _ctx(concept_tags=frozenset({"facing_3bet"})))


def test_blockers_fires_for_high_impact_tags_only() -> None:
    """Fires on ace/king (preflop high-impact) and on the directional
    postflop tags. Does NOT fire on generic blocks_villain_top_value
    or postflop blocks_value (too permissive -- those fire on most hands)."""
    for tag in ("ace_blocker", "king_blocker",
                "blocks_value_unblocks_bluffs", "blocks_bluffs_unblocks_value"):
        assert _has(
            "Blockers & Card Removal",
            _ctx(concept_tags=frozenset({tag})),
        ), f"blocker tag '{tag}' did not fire skill"
    # These low-impact tags should NOT fire the skill alone.
    for tag in ("blocks_villain_top_value", "blocks_value"):
        assert not _has(
            "Blockers & Card Removal",
            _ctx(concept_tags=frozenset({tag})),
        ), f"low-impact tag '{tag}' should not fire skill alone"


def test_blockers_negative_without_blocker_tag() -> None:
    assert not _has(
        "Blockers & Card Removal",
        _ctx(concept_tags=frozenset({"facing_3bet", "small_pair"})),
    )


def test_range_polarization() -> None:
    assert _has(
        "Range Polarization",
        _ctx(path="postflop", concept_tags=frozenset({"villain_polarized"})),
    )


# --- Section 6: Positional & Situational ---------------------------------
def test_in_position_play_late_pos_facing_raise() -> None:
    assert _has(
        "In Position Play",
        _ctx(hero_position="BTN", n_prior_raises=1,
             concept_tags=frozenset({"facing_single_raise"})),
    )


def test_in_position_play_negative_opening() -> None:
    """BTN opening = no IP play to teach (no villain in pot yet)."""
    assert not _has("In Position Play", _ctx(hero_position="BTN", n_prior_raises=0))


def test_out_of_position_play_blinds_facing_raise() -> None:
    assert _has(
        "Out of Position Play",
        _ctx(
            hero_position="BB",
            n_prior_raises=1,
            concept_tags=frozenset({"big_blind", "facing_single_raise"}),
        ),
    )


def test_bvb_small_blind_is_out_of_position() -> None:
    """In blind-vs-blind at a ring table the SB acts FIRST on every postflop
    street, so it is OUT of position. (July 2026 bugfix -- an earlier version
    asserted the opposite via an 'SB is the dealer' exception that holds only
    at a literal 2-player table.) Mirrors pipeline.preflop.position."""
    ctx = _ctx(
        hero_position="SB",
        n_prior_raises=2,
        concept_tags=frozenset({"small_blind", "facing_3bet", "bvb_spot"}),
    )
    assert _has("Out of Position Play", ctx)
    assert not _has("In Position Play", ctx)


def test_bvb_big_blind_is_in_position() -> None:
    """The other seat of the same fix: the BB in a BvB pot acts last
    postflop, so it is IN position -- the blind-defending OOP heuristic
    must not fire on it."""
    ctx = _ctx(
        hero_position="BB",
        n_prior_raises=1,
        concept_tags=frozenset({"big_blind", "facing_single_raise", "bvb_spot"}),
    )
    assert _has("In Position Play", ctx)
    assert not _has("Out of Position Play", ctx)


def test_small_blind_vs_nonblind_open_is_out_of_position() -> None:
    """SB defending a BTN/CO open (not BvB) is genuinely OOP -- the BvB
    exception must not leak to non-BvB blind spots."""
    ctx = _ctx(
        hero_position="SB",
        n_prior_raises=1,
        concept_tags=frozenset({"small_blind", "facing_single_raise"}),
    )
    assert _has("Out of Position Play", ctx)
    assert not _has("In Position Play", ctx)


def test_multiway_pot() -> None:
    assert _has(
        "Multiway Pot Strategy",
        _ctx(concept_tags=frozenset({"multiway_pot"})),
    )


def test_drawing_hand_strategy_postflop_only() -> None:
    """Drawing strategy fires only when path=postflop AND hand_class
    contains 'draw' (e.g. flush_draw, straight_draw)."""
    assert _has(
        "Drawing Hand Strategy",
        _ctx(path="postflop", hand_class="open_ended_straight_draw"),
    )
    # Preflop hand classes never describe draws.
    assert not _has(
        "Drawing Hand Strategy",
        _ctx(path="preflop", hand_class="AKs"),
    )


# --- Section 7: Tournament ------------------------------------------------
def test_short_stack_tournament() -> None:
    assert _has(
        "Short Stack Tournament Strategy",
        _ctx(
            game_format="tournament",
            concept_tags=frozenset({"short_stack"}),
        ),
    )
    # Cash with short_stack tag (rare but possible) does NOT fire.
    assert not _has(
        "Short Stack Tournament Strategy",
        _ctx(game_format="cash", concept_tags=frozenset({"short_stack"})),
    )


def test_tournament_bvb_requires_tournament_format() -> None:
    assert _has(
        "Tournament Blind vs. Blind",
        _ctx(game_format="tournament", concept_tags=frozenset({"bvb_spot"})),
    )
    assert not _has(
        "Tournament Blind vs. Blind",
        _ctx(game_format="cash", concept_tags=frozenset({"bvb_spot"})),
    )


def test_icm_off_until_phase_4() -> None:
    assert not _has(
        "ICM & Tournament Pressure",
        _ctx(game_format="tournament", concept_tags=frozenset({"bvb_spot"})),
    )


# --- Compute_skills wiring ------------------------------------------------
def test_compute_skills_preserves_catalog_order() -> None:
    """Output skill list is in SKILL_CATALOG insertion order, not arbitrary."""
    # Force many to fire so we can inspect the order.
    ctx = _ctx(
        archetype="3bet_for_value",
        concept_tags=frozenset({
            "ace_blocker", "big_blind", "facing_single_raise"
        }),
    )
    skills = compute_skills(ctx)
    catalog_order = list(SKILL_CATALOG.keys())
    last_idx = -1
    for skill in skills:
        idx = catalog_order.index(skill)
        assert idx > last_idx, f"{skill} out of catalog order"
        last_idx = idx


def test_strict_tagging_open_spot_returns_two_or_three_skills() -> None:
    """A vanilla BTN open fires Preflop Hand Selection + maybe IP-related
    skills. Should NOT spam every skill."""
    ctx = _ctx(archetype="open_for_value", hero_position="BTN")
    skills = compute_skills(ctx)
    assert "Preflop Hand Selection" in skills
    assert len(skills) <= 3, f"too many skills fired on a clean open: {skills}"


def test_strict_tagging_pure_strategy_returns_empty_unrelated() -> None:
    """A SkillContext with no signals fires nothing."""
    assert compute_skills(_ctx()) == []


# --- Adapter integration --------------------------------------------------
def test_from_preflop_facts_round_trip() -> None:
    """Sanity check the adapter builds a usable SkillContext."""
    from dataclasses import dataclass

    from pipeline.preflop.fact_extractor import PreflopFacts, VillainRangeStats
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType
    from pipeline.preflop.node_enumerator import PreflopDecisionNode
    from pipeline.preflop.spot_sampler import PreflopSpot
    from pipeline.skill_tagger import compute_skills, from_preflop_facts

    @dataclass
    class _Pack:
        stack_depth_bb: int = 100

    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ParsedAction("SB", PreflopActionType.FOLD),
    )
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor="BB",
            history_before=history, actions=(),
        ),
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies={"Call": 0.66, "Fold": 0.34},
        dominant_action="Call",
        dominant_frequency=0.66,
    )
    facts = PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="BTN", action_label="Raise 60%",
            weighted_combo_count=600.0, pct_of_dealt_hands=45.0,
            top_combos=(),
        ),
        hero_equity_vs_villain=0.55,
        archetype="call_for_value",
    )
    ctx = from_preflop_facts(facts)
    assert ctx.path == "preflop"
    assert ctx.archetype == "call_for_value"
    assert ctx.hero_position == "BB"
    assert ctx.hand_class == "AKo"
    assert ctx.n_prior_raises == 1
    # Should fire Pot Odds, Blind Defense, plus call-for-value adjacent.
    skills = compute_skills(ctx)
    assert "Pot Odds" in skills
    assert "Blind Defense" in skills


def test_from_preflop_facts_detects_facing_squeeze() -> None:
    """Squeeze sequence: raise -> call -> raise -> hero. From the
    original raiser's perspective, that's Facing a Squeeze."""
    from pipeline.preflop.fact_extractor import PreflopFacts, VillainRangeStats
    from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType
    from pipeline.preflop.node_enumerator import PreflopDecisionNode
    from pipeline.preflop.spot_sampler import PreflopSpot
    from pipeline.skill_tagger import compute_skills, from_preflop_facts

    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),  # opener
        ParsedAction("CO", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.CALL),  # cold caller
        ParsedAction("SB", PreflopActionType.FOLD),
        ParsedAction("BB", PreflopActionType.RAISE, 308.0),  # squeezer
    )
    # Hero = the original raiser (HJ) is now facing the squeeze.
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor="HJ",
            history_before=history, actions=(),
        ),
        hero_hand_class="AQo",
        hero_card_combo="AhQc",
        action_frequencies={"Fold": 0.55, "Call": 0.35, "Raise 5x": 0.10},
        dominant_action="Fold",
        dominant_frequency=0.55,
    )
    facts = PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="BB", action_label="Raise 308%",
            weighted_combo_count=80.0, pct_of_dealt_hands=6.0,
            top_combos=(),
        ),
        hero_equity_vs_villain=0.45,
        archetype="fold_dominated",
    )
    ctx = from_preflop_facts(facts)
    skills = compute_skills(ctx)
    assert "Facing a Squeeze" in skills, f"expected Facing a Squeeze in {skills}"
    assert "Facing a 3-Bet" not in skills, (
        "Squeeze responses should NOT also tag as vanilla Facing a 3-Bet"
    )


# --- Catalog completeness sanity -----------------------------------------
def test_catalog_has_42_skills() -> None:
    """Locks the catalog at 42 skills (the user's source list). If a
    skill is added, update this assertion + add a test."""
    assert len(SKILL_CATALOG) == 42


def test_meta_covers_every_catalog_entry() -> None:
    """SKILL_META and SKILL_CATALOG must agree on the set of skills --
    every catalog rule needs a description + status + section."""
    from pipeline.skill_tagger import SKILL_META

    assert set(SKILL_META.keys()) == set(SKILL_CATALOG.keys()), (
        f"meta vs catalog mismatch: "
        f"only-in-meta={set(SKILL_META) - set(SKILL_CATALOG)}, "
        f"only-in-catalog={set(SKILL_CATALOG) - set(SKILL_META)}"
    )


def test_meta_status_values_are_valid() -> None:
    """Every meta status is one of the three known buckets."""
    from pipeline.skill_tagger import SKILL_META

    valid = {"preflop_fires", "postflop_fires", "todo"}
    for name, meta in SKILL_META.items():
        assert meta.status in valid, (
            f"{name}: invalid status {meta.status!r}; expected one of {valid}"
        )


def test_meta_descriptions_are_meaningful() -> None:
    """No empty / placeholder descriptions."""
    from pipeline.skill_tagger import SKILL_META

    for name, meta in SKILL_META.items():
        assert len(meta.description) > 40, (
            f"{name}: description too short ({len(meta.description)} chars)"
        )


@pytest.mark.parametrize("skill", list(SKILL_CATALOG.keys()))
def test_every_skill_predicate_callable(skill: str) -> None:
    """Sanity: every rule executes without raising on a default context."""
    SKILL_CATALOG[skill](_ctx())
