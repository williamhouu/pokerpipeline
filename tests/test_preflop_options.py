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
    ANSWER_STYLES,
    ANSWER_STYLE_FROM_RADIO_LABEL,
    build_options,
    build_options_auto,
    build_options_basic,
    build_options_gto,
    canonicalize_action_label,
    canonicalize_strategy,
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
    facts = _facts_with_strategy(
        {"Fold": 0.20, "Raise 60%": 0.50, "Raise 100%": 0.30}
    )
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
    """Two actions at meaningful frequencies -> 2 options, dominant first.
    Raise labels are canonicalised: 'Raise 60%' -> 'Raise' (raise_level=1
    since the BTN fixture has no prior raises in history)."""
    facts = _facts_with_strategy({"Fold": 0.30, "Raise 60%": 0.70})
    options, correct = build_options_basic(facts)
    assert options == ["Raise", "Fold"]
    assert correct == "Raise"
    assert correct in options


def test_basic_drops_tiny_frequencies() -> None:
    """Actions played at < 5% are filtered out -- noise, not strategic mix."""
    facts = _facts_with_strategy(
        {"Fold": 0.30, "Raise 60%": 0.65, "Call": 0.005, "AllIn": 0.045}
    )
    options, correct = build_options_basic(facts)
    assert options == ["Raise", "Fold"]
    assert "Call" not in options
    assert "All-in" not in options
    assert correct == "Raise"


def test_basic_pure_action_one_option() -> None:
    """Single meaningful action -> 1 option."""
    facts = _facts_with_strategy({"Fold": 1.0, "Raise 60%": 0.0})
    options, correct = build_options_basic(facts)
    assert options == ["Fold"]
    assert correct == "Fold"


def test_basic_caps_at_four_options() -> None:
    """When 5+ raw actions are meaningfully played, canonicalisation may
    collapse multiple raise-size labels into a single 'Raise' entry --
    so the number of post-canonical options can shrink. Top-frequency
    canonical action is first."""
    facts = _facts_with_strategy(
        {
            "Fold": 0.10,
            "Call": 0.15,
            "Raise 60%": 0.40,
            "Raise 100%": 0.20,
            "AllIn": 0.15,
        }
    )
    options, _correct = build_options_basic(facts)
    # Two raise sizes collapse into one 'Raise' (40% + 20% = 60%).
    # Result: ['Raise' (60%), 'Call' (15%), 'All-in' (15%), 'Fold' (10%)].
    assert len(options) <= 4
    assert options[0] == "Raise"


# --- build_options_gto ------------------------------------------------------
def test_gto_two_action_mix_uses_full_template() -> None:
    """Classic 2-action mix -> 4 options in Always/Mostly template,
    correct_answer uses the Mostly prefix (freq < 95%)."""
    facts = _facts_with_strategy({"Call": 0.66, "Fold": 0.34})
    options, correct = build_options_gto(facts)
    assert options == [
        "Always Call",
        "Mostly Call",
        "Mostly Fold",
        "Always Fold",
    ]
    assert correct == "Mostly Call"
    assert correct in options


def test_gto_uses_always_prefix_at_95_pct() -> None:
    """At 95% frequency the prefix flips to Always (per the deterministic
    frequency_to_verb_prefix table shared with postflop). Raise label
    canonicalised."""
    facts = _facts_with_strategy({"Raise 60%": 0.95, "Fold": 0.05})
    _options, correct = build_options_gto(facts)
    assert correct == "Always Raise"


def test_gto_uses_mostly_prefix_at_94_pct() -> None:
    """One percentage point below the Always threshold -> Mostly prefix."""
    facts = _facts_with_strategy({"Raise 60%": 0.94, "Fold": 0.06})
    _options, correct = build_options_gto(facts)
    assert correct == "Mostly Raise"


def test_gto_three_action_mix_uses_composite_labels() -> None:
    """A 3-action mix uses 'Mostly X, sometimes Y' composite labels rather
    than standalone 'Sometimes Y' options (which Ryan banned in Apr 2026
    as ambiguous). correct_answer is the composite pairing the dominant
    with the SECOND-most-frequent action. Raise labels canonicalised."""
    facts = _facts_with_strategy(
        {"Call": 0.60, "Fold": 0.25, "Raise 308%": 0.15}
    )
    options, correct = build_options_gto(facts)
    assert options[0] == "Always Call"
    # Composite labels for each secondary action -- raise canonicalised
    # to "Raise" since the BTN fixture has empty history (raise_level=1).
    assert "Mostly Call, sometimes Fold" in options
    assert "Mostly Call, sometimes Raise" in options
    # correct_answer pairs dominant with the second-most-frequent action
    # (Fold here, since Fold > Raise in this mix).
    assert correct == "Mostly Call, sometimes Fold"
    assert correct in options
    # No standalone "Sometimes X" labels.
    for option in options:
        assert not option.startswith("Sometimes")
        assert not option.startswith("Rarely")


def test_gto_single_action_one_option() -> None:
    """If only one action is meaningfully played, GTO returns a single
    'Always X' option."""
    facts = _facts_with_strategy({"Fold": 1.0, "Raise 60%": 0.0})
    options, correct = build_options_gto(facts)
    assert options == ["Always Fold"]
    assert correct == "Always Fold"


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
        {"Call": 1.0, "Fold": 0.0},                  # pure
        {"Call": 0.95, "Fold": 0.05},                # Always boundary
        {"Call": 0.80, "Fold": 0.20},                # auto boundary
        {"Call": 0.66, "Fold": 0.34},                # mixed
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
