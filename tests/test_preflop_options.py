"""Tests for pipeline.preflop.options (deterministic option selection).

Replaces the LLM's responsibility for picking option strings + correct_answer
with pure Python algorithms. Every option string and the correct_answer are
computed deterministically from PreflopFacts -- no LLM, no retry loop, no
ambiguity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.fact_extractor import PreflopFacts  # noqa: E402
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.options import (  # noqa: E402
    ANSWER_STYLE_FROM_RADIO_LABEL,
    ANSWER_STYLES,
    build_options,
    build_options_auto,
    build_options_basic,
    build_options_gto,
    canonicalize_action_label,
    canonicalize_strategy,
    is_check_spot,
)
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


# --- fixture builders -------------------------------------------------------
def _node(
    actor: str = "BTN",
    history: tuple[ParsedAction, ...] = (),
) -> PreflopDecisionNode:
    return PreflopDecisionNode(
        pack_id="t",
        actor=actor,
        history_before=history,
        actions=(),
    )


def _facts_with_strategy(
    action_frequencies: dict[str, float],
    *,
    actor: str = "BTN",
    history: tuple[ParsedAction, ...] = (),
    hand_class: str = "AKo",
    combo: str = "AhKc",
) -> PreflopFacts:
    """Build a minimal PreflopFacts for the strategy under test.

    Default fixture: BTN actor with empty history (raise_level=1 ->
    Pio 'Raise X%' labels canonicalise to 'Raise'). Pass a non-empty
    `history` to exercise 3-bet / 4-bet level canonicalisation.
    """
    dominant_label, dominant_freq = max(
        action_frequencies.items(), key=lambda kv: kv[1]
    )
    spot = PreflopSpot(
        node=_node(actor=actor, history=history),
        hero_hand_class=hand_class,
        hero_card_combo=combo,
        action_frequencies=action_frequencies,
        dominant_action=dominant_label,
        dominant_frequency=dominant_freq,
    )
    return PreflopFacts(spot=spot, archetype="open_for_value")


# --- canonicalize_action_label ---------------------------------------------
def test_canonicalize_raise_level_1_is_raise() -> None:
    """1st raise of the hand: Pio 'Raise X%' label -> 'Raise'."""
    assert canonicalize_action_label("Raise 60%", raise_level=1) == "Raise"
    assert canonicalize_action_label("Raise 76%", raise_level=1) == "Raise"


def test_canonicalize_raise_level_2_is_3bet() -> None:
    """2nd raise of the hand: Pio 'Raise X%' label -> '3-bet'."""
    assert canonicalize_action_label("Raise 182%", raise_level=2) == "3-bet"
    assert canonicalize_action_label("Raise 77%", raise_level=2) == "3-bet"


def test_canonicalize_raise_level_3_is_4bet() -> None:
    assert canonicalize_action_label("Raise 50%", raise_level=3) == "4-bet"


def test_canonicalize_raise_level_4_is_5bet() -> None:
    assert canonicalize_action_label("Raise 100%", raise_level=4) == "5-bet"


def test_canonicalize_raise_level_5_falls_back_to_raise() -> None:
    """5+ raises (very rare): falls back to 'Raise'."""
    assert canonicalize_action_label("Raise 100%", raise_level=5) == "Raise"


def test_canonicalize_non_raise_passthrough() -> None:
    """Fold / Call don't depend on raise level."""
    assert canonicalize_action_label("Fold", raise_level=1) == "Fold"
    assert canonicalize_action_label("Call", raise_level=3) == "Call"


def test_canonicalize_allin_uses_hyphen() -> None:
    """Pio's 'AllIn' becomes 'All-in' for player display."""
    assert canonicalize_action_label("AllIn", raise_level=1) == "All-in"


# --- canonicalize_strategy --------------------------------------------------
def test_canonicalize_strategy_merges_duplicate_raise_sizes() -> None:
    """When a node has two raise sizes (rare in Ryan pack), they collapse
    into a single 'Raise' entry whose frequency is the sum."""
    facts = _facts_with_strategy({"Fold": 0.20, "Raise 60%": 0.50, "Raise 100%": 0.30})
    canonical = canonicalize_strategy(facts)
    # The two raise sizes merge into a single 'Raise' with summed freq.
    assert canonical == {"Fold": 0.20, "Raise": 0.80}


def test_canonicalize_strategy_uses_3bet_when_history_has_open() -> None:
    """Hero facing an open: hero's raise canonicalises to '3-bet'."""
    facts = _facts_with_strategy(
        {"Fold": 0.30, "Call": 0.40, "Raise 182%": 0.30},
        actor="BB",
        history=(ParsedAction("BTN", PreflopActionType.RAISE, 60.0),),
    )
    canonical = canonicalize_strategy(facts)
    assert "3-bet" in canonical
    assert canonical["3-bet"] == 0.30
    assert "Raise 182%" not in canonical


# --- build_options_basic ----------------------------------------------------
def test_basic_two_action_mix() -> None:
    """Two actions both included; Fold appears first when present."""
    facts = _facts_with_strategy({"Fold": 0.30, "Raise 60%": 0.70})
    options, correct = build_options_basic(facts)
    # Fold first, then Raise (the dominant action).
    assert options == ["Fold", "Raise"]
    assert correct == "Raise"
    assert correct in options


def test_basic_includes_low_freq_actions_when_room() -> None:
    """No frequency filter -- every Pio-offered canonical action shows
    up as an option as long as we have room (<=4 total). Even a 0.5%
    action is a real legal choice the player should be allowed to
    consider."""
    facts = _facts_with_strategy(
        {"Fold": 0.30, "Raise 60%": 0.65, "Call": 0.005, "AllIn": 0.045}
    )
    options, correct = build_options_basic(facts)
    # All 4 canonical actions present (no frequency filter), displayed up
    # the aggression ladder (July 2026 standing rule).
    assert options == ["Fold", "Call", "Raise", "All-in"]
    assert correct == "Raise"


def test_basic_pure_action_still_shows_alternatives() -> None:
    """A pure-fold strategy still shows Raise as an option -- the player
    needs the chance to consider the alternative and learn why it's
    wrong. Fold remains first, correct = 'Fold'."""
    facts = _facts_with_strategy({"Fold": 1.0, "Raise 60%": 0.0})
    options, correct = build_options_basic(facts)
    assert options == ["Fold", "Raise"]
    assert correct == "Fold"


def test_basic_canonicalisation_collapses_then_orders() -> None:
    """5 raw raise sizes -> 1 canonical 'Raise' (sum of frequencies).
    Result displays up the aggression ladder (Fold, Call, Raise, All-in)
    regardless of the frequencies -- July 2026 standing rule."""
    facts = _facts_with_strategy(
        {
            "Fold": 0.10,
            "Call": 0.15,
            "Raise 60%": 0.40,
            "Raise 100%": 0.20,
            "AllIn": 0.15,
        }
    )
    options, correct = build_options_basic(facts)
    # 4 canonical actions after collapsing the two raise sizes -- all fit.
    assert options == ["Fold", "Call", "Raise", "All-in"]
    # Dominant raw label was "Raise 60%" -> canonical "Raise".
    assert correct == "Raise"


def test_basic_option_order_is_always_the_aggression_ladder() -> None:
    """STANDING RULE (July 2026, user): the option row always reads least ->
    most aggressive. A dominant raise must NEVER sit between Fold and Call
    (the old frequency order shipped "Fold · 4-bet · Call")."""
    facts = _facts_with_strategy(
        {"Fold": 0.0, "Call": 0.07, "Raise 100%": 0.93},
        # Facing a 3-bet (two raises before hero) -> hero's raise is a 4-bet.
        history=(
            ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
            ParsedAction("SB", PreflopActionType.RAISE, 100.0),
        ),
    )
    options, correct = build_options_basic(facts)
    assert options == ["Fold", "Call", "4-bet"]
    assert correct == "4-bet"


def test_basic_drops_fold_only_when_zero_and_crowded() -> None:
    """5+ canonical actions and Fold=0%: drop Fold and take top 4.

    5+ canonical actions with Fold non-zero: keep Fold + top 3 non-Fold.
    """
    # Case A: Fold = 0%, 5 canonical actions -> drop Fold.
    # Use distinct verbs to avoid canonicalisation collapse.
    # Build a custom fixture with raise_level fixed so we can construct
    # 5 truly distinct canonical labels without the test depending on
    # how raises collapse.
    # We'll skip the truly 5+ case here (the Ryan pack doesn't produce
    # 5+ canonical actions in practice) and just test the rule via the
    # public canonical-strategy interface.
    facts = _facts_with_strategy(
        # 5 distinct canonical actions: Fold, Call, Raise, All-in,
        # and one extra by using a synthetic "Check" entry that the
        # canonicaliser passes through unchanged.
        {
            "Fold": 0.00,
            "Call": 0.40,
            "Raise 60%": 0.30,
            "AllIn": 0.20,
            "Check": 0.10,  # passthrough label
        }
    )
    options, correct = build_options_basic(facts)
    # Fold dropped (0% + crowded).
    assert "Fold" not in options
    assert len(options) == 4
    # Top 4 by frequency.
    assert set(options) == {"Call", "Raise", "All-in", "Check"}
    assert correct == "Call"

    # Case B: same 5 actions but Fold at 5% (non-zero) -- Fold protected.
    facts2 = _facts_with_strategy(
        {
            "Fold": 0.05,
            "Call": 0.40,
            "Raise 60%": 0.25,
            "AllIn": 0.20,
            "Check": 0.10,
        }
    )
    options2, correct2 = build_options_basic(facts2)
    assert "Fold" in options2
    assert options2[0] == "Fold"  # Fold-first ordering
    assert len(options2) == 4
    # Lowest-frequency non-Fold action gets dropped.
    # Sorted: Call 40, Raise 25, AllIn 20, Check 10, Fold 5.
    # Keep Fold + top 3 non-Fold = Fold, Call, Raise, All-in.
    assert set(options2) == {"Fold", "Call", "Raise", "All-in"}
    assert "Check" not in options2
    assert correct2 == "Call"


# --- build_options_gto ------------------------------------------------------
def test_gto_two_action_mix_uses_full_template() -> None:
    """Classic 2-action mix -> 4 options in Always/Mostly template,
    correct_answer uses the Mostly prefix (freq < 95%). With B = Fold
    and A != Fold, the standalone Always Fold / Mostly Fold options
    come first (Fold-first reordering)."""
    facts = _facts_with_strategy({"Call": 0.66, "Fold": 0.34})
    options, correct = build_options_gto(facts)
    assert options == [
        "Always Fold",
        "Mostly Fold",
        "Mostly Call",
        "Always Call",
    ]
    assert correct == "Mostly Call"
    assert correct in options


def test_gto_two_action_mix_no_fold_keeps_dominant_first() -> None:
    """When the secondary is not Fold, no Fold-first reordering -- the
    template stays Always-dominant / Mostly-dominant / Mostly-secondary /
    Always-secondary."""
    # Call dominant, Raise as secondary, no Fold in the mix.
    facts = _facts_with_strategy({"Call": 0.60, "Raise 60%": 0.40})
    options, correct = build_options_gto(facts)
    assert options == [
        "Always Call",
        "Mostly Call",
        "Mostly Raise",
        "Always Raise",
    ]
    assert correct == "Mostly Call"


def test_gto_fold_dominant_naturally_fold_first() -> None:
    """When Fold IS the dominant action, the template is already Fold-first
    naturally -- no reordering applied (A == Fold, not B)."""
    facts = _facts_with_strategy({"Fold": 0.60, "Call": 0.40})
    options, correct = build_options_gto(facts)
    assert options == [
        "Always Fold",
        "Mostly Fold",
        "Mostly Call",
        "Always Call",
    ]
    assert correct == "Mostly Fold"


def test_gto_uses_mostly_prefix_at_95_pct() -> None:
    """A near-pure 95% spot is still a MIX, so the correct answer is "Mostly
    Raise" -- "Always" is reserved for a literally-pure 100% action (June 2026).
    "Always Raise" becomes a neutral-credit near-miss, not the answer."""
    facts = _facts_with_strategy({"Raise 60%": 0.95, "Fold": 0.05})
    _options, correct = build_options_gto(facts)
    assert correct == "Mostly Raise"


def test_gto_uses_always_prefix_only_at_pure_100() -> None:
    """A literally-pure (100%) action keeps the "Always" correct answer (the
    0%-Fold gives the spectrum its secondary)."""
    facts = _facts_with_strategy({"Raise 60%": 1.0, "Fold": 0.0})
    _options, correct = build_options_gto(facts)
    assert correct == "Always Raise"


def test_gto_uses_mostly_prefix_at_94_pct() -> None:
    """One percentage point below the Always threshold -> Mostly prefix."""
    facts = _facts_with_strategy({"Raise 60%": 0.94, "Fold": 0.06})
    _options, correct = build_options_gto(facts)
    assert correct == "Mostly Raise"


def test_gto_three_action_mix_uses_unified_spectrum() -> None:
    """May 2026: 3+ action spots now use the SAME 4-option spectrum
    template as 2-action spots. The third+ actions don't get their
    own option slots -- they fold into the LLM's explanation prose
    via the SOLVER DATA block. The OPTIONS only test 'which direction
    dominates and is it pure or mixed?'.

    Replaces the older composite template ('Mostly X, sometimes Y')
    which was hard to evaluate without a meta-elimination shortcut.
    """
    # Call 60% (dominant), Fold 25% (most-frequent secondary), Raise 15%
    # -> the spectrum picks Call + Fold as top 2. Raise drops out of
    # the options but stays in the SOLVER DATA for the LLM's prose.
    facts = _facts_with_strategy({"Call": 0.60, "Fold": 0.25, "Raise 308%": 0.15})
    options, correct = build_options_gto(facts)
    # Aggression order: Fold (less aggressive) first.
    assert options == [
        "Always Fold",
        "Mostly Fold",
        "Mostly Call",
        "Always Call",
    ]
    # dominant freq 60% -> "Mostly" prefix
    assert correct == "Mostly Call"
    assert correct in options
    # No composite labels anywhere -- the unified template uses only
    # "Always X" and "Mostly X" forms.
    for option in options:
        assert ", sometimes " not in option
        assert not option.startswith("Sometimes")
        assert not option.startswith("Rarely")


def test_gto_near_pure_spot_still_emits_full_template() -> None:
    """If only one action is meaningfully played (Fold=100%, Raise=0%),
    GTO still emits the 4-option template using a B picked from the
    non-dominant Pio actions. The player can rule out the wrong
    alternatives rather than be handed a one-option non-question."""
    facts = _facts_with_strategy({"Fold": 1.0, "Raise 60%": 0.0})
    options, correct = build_options_gto(facts)
    # Fold dominant -> template Fold-first naturally; B = Raise (the
    # other Pio action at 0%).
    assert options == [
        "Always Fold",
        "Mostly Fold",
        "Mostly Raise",
        "Always Raise",
    ]
    assert correct == "Always Fold"
    assert correct in options


def test_gto_near_pure_with_fold_mixin_uses_fold_first() -> None:
    """Call 96% / Fold 4%: only Call is meaningful (>=5%), but Pio plays
    Fold 4%, so Fold becomes B. Standalone Fold-prefixed options come
    first (Fold is least aggressive)."""
    facts = _facts_with_strategy({"Call": 0.96, "Fold": 0.04})
    options, correct = build_options_gto(facts)
    assert options == [
        "Always Fold",
        "Mostly Fold",
        "Mostly Call",
        "Always Call",
    ]
    # 96% is still a mix -> "Mostly Call" correct ("Always Call" is neutral).
    assert correct == "Mostly Call"
    assert correct in options


def test_gto_2_action_template_orders_by_aggression() -> None:
    """Regression: when secondary is a smaller action than dominant
    (e.g. dominant=Raise/4-bet, secondary=Call), the lesser action
    (Call) comes FIRST -- generalises the Fold-first special case to
    'least-aggressive-first'. Otherwise the 4-option block reads as
    'most committed -> most conservative', which doesn't match how
    poker decisions are usually scanned.

    Uses a hero=BB-facing-3bet history (2 prior raises -> raise_level=3)
    so 'Raise X%' canonicalises to '4-bet'.
    """
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
        ParsedAction("CO", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.RAISE, 77.0),
        ParsedAction("SB", PreflopActionType.FOLD),
    )
    facts = _facts_with_strategy(
        {"Raise 50%": 0.76, "Call": 0.24},
        actor="BB",
        history=history,
    )
    options, correct = build_options_gto(facts)
    # Call (less aggressive) FIRST, 4-bet (more aggressive) LAST.
    assert options == [
        "Always Call",
        "Mostly Call",
        "Mostly 4-bet",
        "Always 4-bet",
    ]
    assert correct == "Mostly 4-bet"
    assert correct in options


def test_gto_pick_secondary_prefers_fold_on_tie() -> None:
    """When multiple non-dominant actions are tied at the same (lowest)
    frequency, Fold is preferred as the secondary. Pure Call (100%) with
    Fold and Raise both at 0% -> B = Fold."""
    facts = _facts_with_strategy({"Call": 1.0, "Fold": 0.0, "Raise 60%": 0.0})
    options, correct = build_options_gto(facts)
    assert options == [
        "Always Fold",
        "Mostly Fold",
        "Mostly Call",
        "Always Call",
    ]
    assert correct == "Always Call"


def test_gto_pick_secondary_takes_higher_freq_over_fold() -> None:
    """A non-tied higher-frequency non-dominant action wins over Fold.
    Call 60% / Raise 40% / Fold 0%: B = Raise (40% > 0%, not a tie)."""
    facts = _facts_with_strategy({"Call": 0.60, "Raise 60%": 0.40, "Fold": 0.0})
    options, correct = build_options_gto(facts)
    # B = Raise; no Fold-first reorder since B != Fold.
    assert options == [
        "Always Call",
        "Mostly Call",
        "Mostly Raise",
        "Always Raise",
    ]
    assert correct == "Mostly Call"


# --- build_options_auto -----------------------------------------------------
def test_auto_picks_basic_for_dominant_action() -> None:
    """Dominant freq >= 80% -> basic style (clean, no Always/Mostly noise).
    Raise label canonicalised."""
    facts = _facts_with_strategy({"Raise 60%": 0.90, "Fold": 0.10})
    options, correct = build_options_auto(facts)
    # Bare canonical labels, no prefix.
    assert "Always Raise" not in options
    assert "Mostly Raise" not in options
    assert "Raise" in options
    assert correct == "Raise"


def test_auto_picks_gto_for_mixed_strategy() -> None:
    """Dominant freq < 80% -> GTO style with Always/Mostly framing."""
    facts = _facts_with_strategy({"Call": 0.66, "Fold": 0.34})
    options, correct = build_options_auto(facts)
    assert "Mostly Call" in options
    assert correct == "Mostly Call"


def test_auto_threshold_at_80_pct_exact() -> None:
    """At exactly 80%, auto picks basic (>= comparison). Raise canonical."""
    facts = _facts_with_strategy({"Raise 60%": 0.80, "Fold": 0.20})
    options, _correct = build_options_auto(facts)
    assert "Raise" in options
    assert "Always Raise" not in options


# --- dispatcher -------------------------------------------------------------
def test_dispatcher_routes_to_each_style() -> None:
    """The build_options dispatcher routes correctly for each style id."""
    facts = _facts_with_strategy({"Call": 0.66, "Fold": 0.34})
    basic_opts, _ = build_options(facts, style="basic")
    gto_opts, _ = build_options(facts, style="gto")
    auto_opts, _ = build_options(facts, style="auto")
    assert basic_opts == build_options_basic(facts)[0]
    assert gto_opts == build_options_gto(facts)[0]
    assert auto_opts == build_options_auto(facts)[0]


def test_dispatcher_default_is_auto() -> None:
    """No style kwarg -> auto."""
    facts = _facts_with_strategy({"Call": 0.66, "Fold": 0.34})
    default_opts, default_correct = build_options(facts)
    auto_opts, auto_correct = build_options_auto(facts)
    assert default_opts == auto_opts
    assert default_correct == auto_correct


def test_dispatcher_rejects_unknown_style() -> None:
    facts = _facts_with_strategy({"Call": 1.0})
    with pytest.raises(ValueError, match="unknown answer style"):
        build_options(facts, style="bogus")


def test_answer_styles_constant_matches_radio_labels() -> None:
    """The admin panel's radio labels all map into ANSWER_STYLES."""
    for radio_label, canonical in ANSWER_STYLE_FROM_RADIO_LABEL.items():
        assert canonical in ANSWER_STYLES, (
            f"radio label {radio_label!r} maps to unknown style {canonical!r}"
        )


# --- invariants -------------------------------------------------------------
def test_correct_answer_always_appears_in_options() -> None:
    """Invariant: for any style, correct_answer is a member of options."""
    strategies = [
        {"Call": 1.0, "Fold": 0.0},  # pure
        {"Call": 0.95, "Fold": 0.05},  # Always boundary
        {"Call": 0.80, "Fold": 0.20},  # auto boundary
        {"Call": 0.66, "Fold": 0.34},  # mixed
        {"Call": 0.50, "Fold": 0.30, "Raise": 0.20},  # 3-way
    ]
    for strategy in strategies:
        facts = _facts_with_strategy(strategy)
        for style in ANSWER_STYLES:
            options, correct = build_options(facts, style=style)
            assert correct in options, (style, strategy, options, correct)


def test_options_never_empty() -> None:
    """Invariant: every (facts, style) produces at least one option."""
    strategies = [
        {"Call": 1.0},
        {"Call": 0.50, "Fold": 0.50},
        {"Fold": 0.005, "Raise 60%": 0.995},
    ]
    for strategy in strategies:
        facts = _facts_with_strategy(strategy)
        for style in ANSWER_STYLES:
            options, _correct = build_options(facts, style=style)
            assert len(options) >= 1, (style, strategy)
            assert len(options) <= 4, (style, strategy, options)


# --- check spots (BB closing a limped pot: "Call" must read "Check") --------
def _limped_bb(action_frequencies: dict[str, float]) -> PreflopFacts:
    """BB facing a limp (SB completed, no raise) -- a check spot."""
    return _facts_with_strategy(
        action_frequencies,
        actor="BB",
        history=(ParsedAction("SB", PreflopActionType.CALL, None),),
    )


def test_canonicalize_action_label_check_spot_flag() -> None:
    """The check_spot flag relabels Call -> Check; nothing else changes."""
    assert canonicalize_action_label("Call", raise_level=1, check_spot=True) == "Check"
    assert canonicalize_action_label("Call", raise_level=1, check_spot=False) == "Call"
    assert canonicalize_action_label("Fold", raise_level=1, check_spot=True) == "Fold"
    assert (
        canonicalize_action_label("Raise 60%", raise_level=1, check_spot=True) == "Raise"
    )


def test_is_check_spot_only_for_bb_with_no_raise() -> None:
    # BB facing a limp -> check spot.
    assert is_check_spot(_limped_bb({"Call": 0.95, "Raise 60%": 0.05}))
    # BB facing a raise -> a real call, not a check.
    assert not is_check_spot(
        _facts_with_strategy(
            {"Call": 0.6, "Fold": 0.4},
            actor="BB",
            history=(ParsedAction("BTN", PreflopActionType.RAISE, 60.0),),
        )
    )
    # Non-BB with no raise -> not a check spot (only the BB checks preflop).
    assert not is_check_spot(
        _facts_with_strategy({"Call": 0.9, "Raise 60%": 0.1}, actor="SB")
    )


def test_check_spot_canonicalizes_call_to_check() -> None:
    canon = canonicalize_strategy(_limped_bb({"Call": 0.95, "Raise 60%": 0.05}))
    assert canon == {"Check": 0.95, "Raise": 0.05}
    assert "Call" not in canon


def test_check_spot_basic_answer_is_check() -> None:
    options, correct = build_options(
        _limped_bb({"Call": 0.95, "Raise 60%": 0.05}), style="basic"
    )
    assert correct == "Check"
    assert "Check" in options
    assert "Call" not in options


def test_check_spot_gto_mostly_check_not_call() -> None:
    # Near-pure check -> "Mostly Check" correct (was "Always Call" bug, then
    # "Always Check"; now "Mostly" since 97% is a mix). The label is Check, not
    # Call, which is the point of this test.
    options, correct = build_options(
        _limped_bb({"Call": 0.97, "Raise 60%": 0.03}), style="gto"
    )
    assert correct == "Mostly Check"
    assert "Always Check" in options  # the rung still exists (as a neutral)
    assert not any("Call" in opt for opt in options)


def test_bb_facing_raise_still_calls() -> None:
    facts = _facts_with_strategy(
        {"Call": 0.6, "Fold": 0.4},
        actor="BB",
        history=(ParsedAction("BTN", PreflopActionType.RAISE, 60.0),),
    )
    canon = canonicalize_strategy(facts)
    assert "Call" in canon
    assert "Check" not in canon


def test_non_bb_no_raise_still_calls() -> None:
    canon = canonicalize_strategy(
        _facts_with_strategy({"Call": 0.9, "Raise 60%": 0.1}, actor="SB")
    )
    assert "Call" in canon
    assert "Check" not in canon


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --- GTO secondary: EV precedence on pure spots (July 2026) -------------------
def test_gto_pick_secondary_uses_ev_on_pure_spot(monkeypatch) -> None:
    """THE AQs-vs-3-bet Review catch: at a pure spot every alternative ties
    at 0%, and the secondary must be the SECOND-BEST action by solver EV
    (4-bet at +1.95bb), not Fold (0bb). The old ordering preferred Fold
    before consulting EV, so 'top two options by EV' silently applied only
    to non-fold ties."""
    import pipeline.preflop.format_writer as fw

    facts = _facts_with_strategy({"Call": 1.0, "Fold": 0.0, "Raise 60%": 0.0})
    monkeypatch.setattr(
        fw, "action_evs_bb",
        lambda _facts, _pack: {"Call": 2.87, "Fold": 0.0, "Raise": 1.95},
    )
    options, correct = build_options_gto(facts, pack=object())
    assert options == [
        "Always Call",
        "Mostly Call",
        "Mostly Raise",
        "Always Raise",
    ]
    assert correct == "Always Call"


def test_gto_pick_secondary_ev_can_still_pick_fold(monkeypatch) -> None:
    """EV precedence is symmetric: when folding really is the second-best
    EV (raising is -EV), the secondary stays Fold."""
    import pipeline.preflop.format_writer as fw

    facts = _facts_with_strategy({"Call": 1.0, "Fold": 0.0, "Raise 60%": 0.0})
    monkeypatch.setattr(
        fw, "action_evs_bb",
        lambda _facts, _pack: {"Call": 1.10, "Fold": 0.0, "Raise": -0.55},
    )
    options, correct = build_options_gto(facts, pack=object())
    assert options == [
        "Always Fold",
        "Mostly Fold",
        "Mostly Call",
        "Always Call",
    ]
    assert correct == "Always Call"


def test_gto_pick_secondary_never_picks_a_hidden_all_in(monkeypatch) -> None:
    """The EV ranking uses the SAME unreasonable-all-in filter as the
    action_ev_bb CSV cell: a deep-stack jam the solver never takes (0%
    and clearly dominated) can't become the secondary, even when its EV
    tops the other alternatives -- otherwise the options would show an
    'Always All-in' the EV panel hides."""
    import pipeline.preflop.format_writer as fw

    facts = _facts_with_strategy(
        {"Raise 60%": 1.0, "Call": 0.0, "Fold": 0.0, "AllIn": 0.0}
    )
    monkeypatch.setattr(
        fw, "action_evs_bb",
        lambda _facts, _pack: {
            "Raise": 4.0, "All-in": 1.0, "Call": 0.2, "Fold": 0.0,
        },
    )
    options, correct = build_options_gto(facts, pack=object())
    # All-in (freq 0, 3bb below best) is filtered; Call (+0.2) wins the
    # remaining ranking.
    assert correct == "Always Raise"
    assert "Mostly Call" in options
    assert not any("All-in" in o for o in options)


# --- Always/Mostly qualifier helper (Aug 2026 balanced-batch axis) -----------
def test_answer_qualifier_matches_the_gto_rendered_prefix() -> None:
    """PARITY PIN: the balanced-batch qualifier axis reads
    answer_qualifier(spot.dominant_frequency) -- it must equal the prefix
    build_options_gto actually renders on the correct answer, so the two
    can never drift (single source of truth:
    pipeline.explanation_generator.frequency_to_verb_prefix)."""
    from pipeline.preflop.options import answer_qualifier

    for freqs in (
        {"Call": 0.6, "Fold": 0.4},
        {"Raise 60%": 0.72, "Fold": 0.28},
        {"Fold": 0.94, "Call": 0.06},
    ):
        facts = _facts_with_strategy(freqs)
        _opts, correct = build_options_gto(facts)
        prefix = correct.split()[0]
        assert prefix in ("Always", "Mostly")
        assert prefix == answer_qualifier(facts.spot.dominant_frequency)


def test_answer_qualifier_delegates_to_frequency_to_verb_prefix() -> None:
    from pipeline.explanation_generator import frequency_to_verb_prefix
    from pipeline.preflop.options import answer_qualifier

    for freq in (1.0, 0.9999, 0.999, 0.95, 0.65, 0.05, 0.01):
        assert answer_qualifier(freq) == frequency_to_verb_prefix(freq)
    assert answer_qualifier(1.0) == "Always"
    assert answer_qualifier(0.99) == "Mostly"


def test_qualifier_axis_active_only_for_gto_capable_styles() -> None:
    from pipeline.preflop.options import qualifier_axis_active

    assert qualifier_axis_active("gto")
    assert qualifier_axis_active("auto")  # may fall to the GTO spectrum
    assert not qualifier_axis_active("basic")
