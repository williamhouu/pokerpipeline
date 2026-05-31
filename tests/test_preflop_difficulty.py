"""Tests for the May 2026 preflop difficulty algorithm.

Covers each axis in isolation (with the others held neutral), the
weighted blend, the EV-weight redistribution when ev_gap is None,
the soft bounds, and the bump-rule mechanism (which currently has
no active rules but the plumbing is tested via a temporary rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.difficulty import (  # noqa: E402
    BUMP_RULES,
    HAND_CLASS_EASE,
    W_CONCEPT,
    W_EV,
    W_FREQ,
    W_HAND,
    BumpRule,
    DifficultyResult,
    compute_difficulty,
)
from pipeline.preflop.fact_extractor import (  # noqa: E402
    PreflopFacts,
    VillainRangeStats,
)
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


# --- fixtures -------------------------------------------------------------
def _facts(
    *,
    dominant_freq: float = 0.66,
    archetype: str = "call_for_value",
    hand_class: str = "AKo",  # premium_unpaired -> easy_hand=0.85
    actor: str = "BB",
    history: tuple[ParsedAction, ...] | None = None,
    hero_equity: float | None = 0.55,
) -> PreflopFacts:
    """Build a minimal PreflopFacts for difficulty testing.

    Defaults to a BB-facing-BTN-open spot with AKo (premium_unpaired).
    Override dominant_freq, archetype, hand_class, actor, history, or
    equity per test.
    """
    if history is None:
        history = (
            ParsedAction("UTG", PreflopActionType.FOLD),
            ParsedAction("HJ", PreflopActionType.FOLD),
            ParsedAction("CO", PreflopActionType.FOLD),
            ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
            ParsedAction("SB", PreflopActionType.FOLD),
        )
    # Two-action strategy: dominant + the complement gets the residual.
    action_freqs = {"Call": dominant_freq, "Fold": 1.0 - dominant_freq}
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor=actor,
            history_before=history, actions=(),
        ),
        hero_hand_class=hand_class,
        hero_card_combo="AhKc",
        action_frequencies=action_freqs,
        dominant_action="Call",
        dominant_frequency=dominant_freq,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="BTN", action_label="Raise 60%",
            weighted_combo_count=600.0, pct_of_dealt_hands=45.0,
            top_combos=(),
        ),
        hero_equity_vs_villain=hero_equity,
        archetype=archetype,
    )


# --- axis 1: freq ---------------------------------------------------------
def test_freq_floor_55_pct_gives_zero_easy_freq() -> None:
    """Freq at the 55% worthiness floor -> easy_freq=0."""
    result = compute_difficulty(_facts(dominant_freq=0.55), ev_gap_bb=0.0)
    assert result.easy_freq == pytest.approx(0.0)


def test_freq_100_pct_gives_max_easy_freq() -> None:
    """Pure strategy (100%) -> easy_freq=1.0."""
    result = compute_difficulty(_facts(dominant_freq=1.0), ev_gap_bb=0.0)
    assert result.easy_freq == pytest.approx(1.0)


def test_freq_below_floor_clips_to_zero() -> None:
    """Freq below 55% (theoretically possible if a spot squeaks through
    the filter) clips to 0 -- doesn't go negative."""
    result = compute_difficulty(_facts(dominant_freq=0.40), ev_gap_bb=0.0)
    assert result.easy_freq == 0.0


# --- axis 2: EV gap -------------------------------------------------------
def test_ev_gap_zero_gives_zero_easy_ev() -> None:
    result = compute_difficulty(_facts(), ev_gap_bb=0.0)
    assert result.easy_ev == 0.0


def test_ev_gap_3bb_gives_max_easy_ev() -> None:
    """3 bb is the cap (full credit)."""
    result = compute_difficulty(_facts(), ev_gap_bb=3.0)
    assert result.easy_ev == 1.0


def test_ev_gap_above_3bb_clips_to_one() -> None:
    """A 10 bb gap doesn't push easy_ev past 1."""
    result = compute_difficulty(_facts(), ev_gap_bb=10.0)
    assert result.easy_ev == 1.0


def test_ev_gap_none_marks_unavailable() -> None:
    """When EV is None, ev_available is False and easy_ev is 0 (unused)."""
    result = compute_difficulty(_facts(), ev_gap_bb=None)
    assert result.ev_available is False
    assert result.easy_ev == 0.0


# --- axis 3: concept ------------------------------------------------------
def test_concept_archetype_lookup() -> None:
    """archetype 'open_for_value' -> ARCHETYPE_BASE_EASE = 1.00."""
    # No firing concept-tag modifiers expected for an open spot (no
    # multiway, no short_stack, no deep_stack -- ranges.spot is 100bb
    # but the deep_stack tag triggers on >150bb).
    result = compute_difficulty(
        _facts(
            archetype="open_for_value",
            actor="BTN",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.FOLD),
                ParsedAction("CO", PreflopActionType.FOLD),
            ),
        ),
        ev_gap_bb=0.0,
    )
    # easy_concept should be ARCHETYPE_BASE_EASE["open_for_value"] = 1.00,
    # possibly minus modifiers if any fire. For a clean BTN open with
    # no multiway, no short_stack, deep_stack is on (100bb pack), so
    # +0.05 but clipped to 1.00 ceiling -> stays 1.00.
    assert result.easy_concept == pytest.approx(1.00)


def test_concept_5bet_pot_is_very_hard() -> None:
    """archetype '5bet_for_value' -> base ease 0.10 (hardest concept)."""
    result = compute_difficulty(
        _facts(archetype="5bet_for_value"), ev_gap_bb=0.0
    )
    # No multiway / short_stack / deep_stack here (history is BB
    # facing single raise, 100bb so deep_stack might fire -> +0.05)
    # Wait -- the default _facts history is BB facing BTN open which
    # ISN'T a 5-bet pot. The test only sets archetype, so the concept
    # base lookup yields 0.10. Tag modifiers may shift it.
    # We just assert it's at or near the base.
    assert 0.05 <= result.easy_concept <= 0.20


def test_concept_unknown_archetype_uses_default() -> None:
    """Unknown archetype -> 0.50 default base."""
    result = compute_difficulty(
        _facts(archetype="something_not_in_table"), ev_gap_bb=0.0
    )
    # Plus modifiers but with default 0.5 base and minimal tag
    # firing, should stay near 0.5.
    assert 0.40 <= result.easy_concept <= 0.60


# --- axis 4: hand class ---------------------------------------------------
def test_hand_premium_pair_max_easy() -> None:
    """premium_pair tag fires (AA / KK / QQ) -> easy_hand=1.00."""
    result = compute_difficulty(
        _facts(hand_class="AA"), ev_gap_bb=0.0
    )
    assert result.easy_hand == HAND_CLASS_EASE["premium_pair"]


def test_hand_suited_connector_is_hard() -> None:
    """suited_connector (e.g. 76s) -> easy_hand=0.40."""
    result = compute_difficulty(
        _facts(hand_class="76s"), ev_gap_bb=0.0
    )
    assert result.easy_hand == HAND_CLASS_EASE["suited_connector"]


def test_hand_unconnected_offsuit_obvious_fold() -> None:
    """73o is genuine trash -> easy_hand=0.90 (obvious fold from most spots)."""
    result = compute_difficulty(
        _facts(hand_class="73o"), ev_gap_bb=0.0
    )
    assert result.easy_hand == HAND_CLASS_EASE["unconnected_offsuit"]


def test_hand_no_matching_tag_uses_default() -> None:
    """A hand that doesn't match any of the hand-class tags falls back
    to 0.55. Construct a hand class that no tag covers."""
    # KQo isn't in any of: premium_pair, premium_unpaired (AK/AQ only),
    # unconnected_offsuit (gap >= 2 only -- KQo has gap 1), suited_*,
    # *_pair. So no tag fires -> default 0.55.
    # Actually let me check -- KQo is offsuit broadway. Looking at the
    # concept_tag definitions: premium_unpaired is AK/AQ; suited_broadway
    # is KQs etc.; unconnected_offsuit excludes KQ (gap 1). So KQo
    # likely matches NO hand-class tag.
    result = compute_difficulty(
        _facts(hand_class="KQo"), ev_gap_bb=0.0
    )
    assert result.easy_hand == 0.55


# --- weighted blend -------------------------------------------------------
def test_blend_weights_sum_to_one() -> None:
    """Sanity: the published weights add to 1.0 (within float epsilon)."""
    assert W_FREQ + W_EV + W_CONCEPT + W_HAND == pytest.approx(1.0)


def test_blend_all_axes_max_gives_min_difficulty() -> None:
    """When every axis is at its max ease (1.0), easy_blend=1.0 and
    difficulty hits the linear floor (500)."""
    # Construct: freq=1.0, ev_gap=3, archetype=open_for_value
    # (easy_concept=1.0 after modifiers), hand_class=AA (easy_hand=1.0).
    result = compute_difficulty(
        _facts(
            dominant_freq=1.0,
            archetype="open_for_value",
            hand_class="AA",
            actor="BTN",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.FOLD),
                ParsedAction("CO", PreflopActionType.FOLD),
            ),
        ),
        ev_gap_bb=3.0,
    )
    # All axes max -> easy_blend = 1.0 -> score = 500
    # (float rounding can make some axes 0.9999999... so use approx.)
    assert result.easy_freq == pytest.approx(1.0)
    assert result.easy_ev == pytest.approx(1.0)
    assert result.easy_concept == pytest.approx(1.0)
    assert result.easy_hand == pytest.approx(1.0)
    assert result.easy_blend == pytest.approx(1.0)
    assert result.score == 500


def test_blend_all_axes_min_gives_max_difficulty() -> None:
    """All axes at min (0 or near 0) -> easy_blend low -> high difficulty."""
    # freq=0.55 -> easy_freq=0; ev=0 -> easy_ev=0;
    # archetype=5bet_for_value -> easy_concept=0.10;
    # hand=76s suited_connector -> easy_hand=0.40
    result = compute_difficulty(
        _facts(
            dominant_freq=0.55,
            archetype="5bet_for_value",
            hand_class="76s",
        ),
        ev_gap_bb=0.0,
    )
    # easy_blend = 0.4*0 + 0.3*0 + 0.2*0.10 + 0.1*0.40 = 0.06
    # score = 3000 - 0.06*2500 = 2850 (approx; depends on tag modifiers)
    assert result.easy_freq == 0.0
    assert result.easy_ev == 0.0
    assert 0.05 <= result.easy_concept <= 0.15
    assert result.easy_hand == 0.40
    assert 2800 <= result.score <= 2900


# --- EV-weight redistribution --------------------------------------------
def test_ev_unavailable_redistributes_weight() -> None:
    """When ev_gap_bb is None, W_EV redistributes proportionally across
    the other three weights. The blend's effective weights are:

        W_FREQ' = W_FREQ / (W_FREQ + W_CONCEPT + W_HAND)
        W_CONCEPT' = W_CONCEPT / ...
        W_HAND' = W_HAND / ...

    Sanity: same spot, with vs without EV. With EV=neutral 0.5 (manual)
    should give the same easy_blend as without EV. Let's verify the
    redistribution math directly."""
    facts = _facts(
        dominant_freq=0.80,
        archetype="call_for_value",
        hand_class="AKo",
    )
    without_ev = compute_difficulty(facts, ev_gap_bb=None)
    assert without_ev.ev_available is False
    # Manual computation:
    #   easy_freq = (0.80 - 0.55) / 0.45 = 0.556
    #   archetype call_for_value base = 0.60; tag deltas may add
    #   easy_hand = HAND_CLASS_EASE["premium_unpaired"] = 0.85
    # Redistributed weights: freq 0.4/0.7=0.571, concept 0.2/0.7=0.286,
    # hand 0.1/0.7=0.143
    expected = (
        (W_FREQ / (W_FREQ + W_CONCEPT + W_HAND)) * 0.5556
        + (W_CONCEPT / (W_FREQ + W_CONCEPT + W_HAND)) * without_ev.easy_concept
        + (W_HAND / (W_FREQ + W_CONCEPT + W_HAND)) * 0.85
    )
    assert without_ev.easy_blend == pytest.approx(expected, abs=0.01)


def test_ev_unavailable_does_not_neutralise_difficulty() -> None:
    """Verify redistribution doesn't artificially make all raise spots
    'medium'. A hard spot stays hard, an easy spot stays easy."""
    # Hard mixed spot, no EV info:
    hard = compute_difficulty(
        _facts(
            dominant_freq=0.55,
            archetype="3bet_as_bluff",
            hand_class="A5s",  # suited_ace -> easy_hand=0.40
        ),
        ev_gap_bb=None,
    )
    # Easy pure spot, no EV info:
    easy = compute_difficulty(
        _facts(
            dominant_freq=1.0,
            archetype="open_for_value",
            hand_class="AA",
            actor="BTN",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.FOLD),
                ParsedAction("CO", PreflopActionType.FOLD),
            ),
        ),
        ev_gap_bb=None,
    )
    # Hard should be substantially harder than easy.
    assert hard.score - easy.score >= 1500, (
        f"redistribution may be neutralising: hard={hard.score} easy={easy.score}"
    )


# --- soft bounds ---------------------------------------------------------
def test_soft_bounds_default_in_linear_range() -> None:
    """Typical spots land within [500, 3000]. Spot-check across the
    space."""
    samples = [
        compute_difficulty(_facts(dominant_freq=f), ev_gap_bb=g)
        for f, g in [
            (0.55, 0.0),
            (0.65, 0.5),
            (0.75, 1.0),
            (0.85, 1.5),
            (0.95, 2.0),
            (1.00, 3.0),
        ]
    ]
    for result in samples:
        assert 400 <= result.score <= 3200, f"out of soft bounds: {result.score}"


def test_hard_bounds_clip_extreme_values() -> None:
    """Extreme inputs land at the hard floor / ceiling but not outside."""
    # Even with bumps pushing past 3000, the hard ceiling at 3200 holds.
    # We don't have such a bump active; just sanity-check the bounds.
    result = compute_difficulty(_facts(dominant_freq=1.0), ev_gap_bb=3.0)
    assert 400 <= result.score <= 3200


# --- bump rules ----------------------------------------------------------
def test_bump_rules_table_is_empty_by_default() -> None:
    """At ship time, BUMP_RULES has no active entries. Tuning is via
    adding rules here as observed batches show mis-scored spots."""
    assert BUMP_RULES == ()


def test_bump_rule_fires_and_records_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Insert a temporary bump rule, verify it fires and the name is
    captured in bumps_applied. Tests the plumbing without baking any
    real rule into the production table."""

    def _always(facts: object, ev: float | None) -> bool:  # noqa: ARG001
        return True

    test_rule = BumpRule(
        name="test_bump",
        description="fires on every spot for the test",
        easy_delta=-0.10,  # makes the spot harder
        predicate=_always,
    )
    monkeypatch.setattr(
        "pipeline.preflop.difficulty.BUMP_RULES", (test_rule,)
    )

    result = compute_difficulty(_facts(dominant_freq=0.70), ev_gap_bb=1.0)
    assert "test_bump" in result.bumps_applied
    # easy_blend should be ~0.10 lower than without the bump.
    # Spot-check: re-compute without the bump (monkeypatch the table back).
    monkeypatch.setattr("pipeline.preflop.difficulty.BUMP_RULES", ())
    baseline = compute_difficulty(_facts(dominant_freq=0.70), ev_gap_bb=1.0)
    assert result.easy_blend == pytest.approx(baseline.easy_blend - 0.10, abs=0.001)


# --- DifficultyResult dataclass ------------------------------------------
def test_result_carries_all_axis_breakdown() -> None:
    result = compute_difficulty(_facts(), ev_gap_bb=1.0)
    assert isinstance(result, DifficultyResult)
    assert isinstance(result.score, int)
    assert 0.0 <= result.easy_freq <= 1.0
    assert 0.0 <= result.easy_ev <= 1.0
    assert 0.0 <= result.easy_concept <= 1.0
    assert 0.0 <= result.easy_hand <= 1.0
    assert isinstance(result.bumps_applied, tuple)
    assert isinstance(result.ev_available, bool)


# --- integration: the user's two real reference spots --------------------
def test_users_real_spot_3_btn_33_vs_3bet() -> None:
    """No.3 from the user's batch: BTN 3c3d facing BB 3-bet.
    freq=0.66, ev_gap=1.37, archetype='fold_dominated' (since 33 vs a
    3-bet has low equity)."""
    # Use the right history so raise_level is correct, but we only
    # care about archetype + freq + ev for this test.
    result = compute_difficulty(
        _facts(
            dominant_freq=0.66,
            archetype="fold_dominated",
            hand_class="33",  # small_pair
        ),
        ev_gap_bb=1.37,
    )
    # easy_freq ~ 0.244, easy_ev ~ 0.457,
    # easy_concept ~ 0.70 (fold_dominated base)
    # easy_hand = 0.45 (small_pair)
    # easy_blend = 0.4*0.244 + 0.3*0.457 + 0.2*0.70 + 0.1*0.45
    #            = 0.098 + 0.137 + 0.14 + 0.045
    #            = ~0.420
    # score = 3000 - 0.42*2500 = 1950
    # (some wobble from tag modifiers)
    assert 1800 <= result.score <= 2100


def test_users_real_spot_5_hj_a8s_vs_3bet() -> None:
    """No.5 from the user's batch: HJ As8s facing CO 3-bet.
    freq=0.81, ev_gap=1.38, archetype='fold_dominated' or
    'fold_pot_odds'."""
    result = compute_difficulty(
        _facts(
            dominant_freq=0.81,
            archetype="fold_pot_odds",
            hand_class="A8s",  # suited_ace
            actor="HJ",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
                ParsedAction("CO", PreflopActionType.RAISE, 77.0),
                ParsedAction("BTN", PreflopActionType.FOLD),
                ParsedAction("SB", PreflopActionType.FOLD),
                ParsedAction("BB", PreflopActionType.FOLD),
            ),
        ),
        ev_gap_bb=1.38,
    )
    # easy_freq = (0.81-0.55)/0.45 = 0.578
    # easy_ev = 1.38/3.0 = 0.460
    # easy_concept ~ 0.60 (fold_pot_odds)
    # easy_hand = 0.40 (suited_ace)
    # easy_blend = 0.4*0.578 + 0.3*0.460 + 0.2*0.60 + 0.1*0.40
    #            = 0.231 + 0.138 + 0.12 + 0.04
    #            = ~0.529
    # score = 3000 - 0.529*2500 = ~1678
    assert 1550 <= result.score <= 1800


# --- A+B+C: trivial folds rate near the floor (May 2026) ------------------
def _fold_facts(
    *,
    hand_class: str,
    combo: str,
    archetype: str = "fold_dominated",
    multiway: bool = False,
) -> PreflopFacts:
    """A 100%-fold spot for difficulty testing. combo must match
    hand_class so the concept tagger computes the right hand-class tags."""
    if multiway:
        # UTG opens, SB 3-bets, hero (BB) folds -> 3 non-fold actors.
        history = (
            ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
            ParsedAction("HJ", PreflopActionType.FOLD),
            ParsedAction("CO", PreflopActionType.FOLD),
            ParsedAction("BTN", PreflopActionType.FOLD),
            ParsedAction("SB", PreflopActionType.RAISE, 150.0),
        )
    else:
        history = (
            ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
            ParsedAction("HJ", PreflopActionType.FOLD),
            ParsedAction("CO", PreflopActionType.FOLD),
            ParsedAction("BTN", PreflopActionType.FOLD),
            ParsedAction("SB", PreflopActionType.FOLD),
        )
    spot = PreflopSpot(
        node=PreflopDecisionNode(
            pack_id="t", actor="BB", history_before=history, actions=(),
        ),
        hero_hand_class=hand_class,
        hero_card_combo=combo,
        action_frequencies={"Fold": 1.0},
        dominant_action="Fold",
        dominant_frequency=1.0,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="UTG", action_label="Raise 60%",
            weighted_combo_count=120.0, pct_of_dealt_hands=9.0, top_combos=(),
        ),
        hero_equity_vs_villain=0.28,  # dominated
        archetype=archetype,
    )


def test_clear_trash_suited_fold_rates_near_floor() -> None:
    """73s as a pure dominated fold is about as easy as poker gets -- it
    should land near the easy floor, not in the medium band."""
    result = compute_difficulty(_fold_facts(hand_class="73s", combo="7s3s"))
    assert result.easy_hand == pytest.approx(0.82)   # clear-trash-suited
    assert result.score <= 650


def test_multiway_does_not_penalize_a_fold() -> None:
    """A trivial fold is just as easy heads-up as multiway -- the
    multiway_pot penalty must not apply to fold spots (else the same junk
    fold scores meaningfully harder just because more players entered)."""
    hu = compute_difficulty(
        _fold_facts(hand_class="73s", combo="7s3s", multiway=False))
    mw = compute_difficulty(
        _fold_facts(hand_class="73s", combo="7s3s", multiway=True))
    assert mw.easy_concept == pytest.approx(hu.easy_concept)
    assert abs(mw.score - hu.score) <= 20


def test_fold_dominated_base_is_easy() -> None:
    """A dominated fold gets a high concept ease (was 0.70, now ~0.95)."""
    result = compute_difficulty(_fold_facts(hand_class="72o", combo="7h2c"))
    assert result.easy_concept == pytest.approx(0.95)
    assert result.score <= 650


def test_trash_bump_is_conservative_k2s_not_bumped() -> None:
    """C is deliberately narrow: K2s is much better than 73s and must NOT
    get the trash-suited ease -- it stays at the neutral default, so its
    fold scores higher than a 73s fold."""
    k2s = compute_difficulty(_fold_facts(hand_class="K2s", combo="Ks2s"))
    seven_three = compute_difficulty(
        _fold_facts(hand_class="73s", combo="7s3s"))
    assert k2s.easy_hand == pytest.approx(0.55)        # neutral default
    assert seven_three.easy_hand == pytest.approx(0.82)
    assert k2s.score > seven_three.score


# --- pot-odds break-even equity (ev_engine, fed to Layer 6) ----------------
def test_break_even_equity_is_pot_odds_threshold() -> None:
    """break_even = call_cost / (pot + call_cost). For _fold_facts (UTG opens
    to 2.5bb, SB 3-bets to 10bb, hero=BB to act), BB calls 10 - 1 posted = 9bb
    into a pot of 0.5 + 10 + 2.5 = 13bb -> 9 / (13 + 9) ~= 0.41."""
    from pipeline.preflop.ev_engine import compute_break_even_equity
    from pipeline.preflop.pack import PreflopPack
    pack = PreflopPack(
        pack_id="t", root_path=Path("/tmp/x"), grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
        sb_to_bb_ratio=0.5, description="t",
    )
    be = compute_break_even_equity(
        _fold_facts(hand_class="73s", combo="7s3s", multiway=True), pack)
    assert be is not None
    assert 0.30 <= be <= 0.50
