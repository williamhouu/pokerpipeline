"""Tests for pipeline.plo.options (deterministic answer options)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.fact_extractor import PloFacts  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.options import (  # noqa: E402
    build_options,
    canonicalize_action_label,
    canonicalize_strategy,
    is_check_spot,
)
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

R = PloActionType.RAISE
C = PloActionType.CALL
CARDS = ("As", "Ks", "Ah", "Kh")


def _facts(
    freqs: dict[str, float],
    history: tuple[PloAction, ...] = (),
    *,
    actor: str = "HJ",
) -> PloFacts:
    node = PloDecisionNode(
        actor=actor, history_before=history, actions=(), history_stem=""
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=CARDS,
        action_frequencies=freqs,
        presence=1.0,
    )
    return PloFacts(spot=spot, hand_class=classify_plo_hand(CARDS), archetype="")


def test_canonicalize_label_by_raise_level():
    assert canonicalize_action_label("Raise 100%", raise_level=1) == "Raise"
    assert canonicalize_action_label("Raise 100%", raise_level=2) == "3-bet"
    assert canonicalize_action_label("Raise 100%", raise_level=3) == "4-bet"
    assert canonicalize_action_label("All-in", raise_level=1) == "All-in"
    assert canonicalize_action_label("Call", raise_level=1) == "Call"


def test_basic_is_fold_first_with_canonical_dominant():
    opts, correct = build_options(
        _facts({"Call": 0.7, "Fold": 0.0, "Raise 100%": 0.3}), style="basic"
    )
    assert correct == "Call"
    assert opts == ["Fold", "Call", "Raise"]  # the aggression ladder


def test_basic_option_order_is_always_the_aggression_ladder():
    """STANDING RULE (July 2026, user): the option row always reads least ->
    most aggressive, never frequency order. The exact live catch: UTG+1's
    AA92 facing a 3-bet showed "Fold · 4-bet · Call" because the 93% 4-bet
    outranked the 7% call by frequency."""
    facts = _facts(
        {"Raise 100%": 0.93, "Call": 0.07, "Fold": 0.0},
        # Two raises before hero -> hero's raise is a 4-bet.
        history=(PloAction("LJ", R, 100), PloAction("SB", R, 100)),
    )
    opts, correct = build_options(facts, style="basic")
    assert opts == ["Fold", "Call", "4-bet"]
    assert correct == "4-bet"


def test_gto_spectrum_orders_by_aggression():
    # Mixed Call/raise; one prior raise -> hero's raise is a 3-bet.
    facts = _facts({"Call": 0.6, "Raise 100%": 0.4}, history=(PloAction("LJ", R, 100),))
    opts, correct = build_options(facts, style="gto")
    assert opts == ["Always Call", "Mostly Call", "Mostly 3-bet", "Always 3-bet"]
    assert correct == "Mostly Call"


def test_auto_uses_basic_when_dominant_and_gto_when_mixed():
    dominant = build_options(_facts({"Raise 100%": 0.9, "Fold": 0.1}), style="auto")
    assert dominant[1] == "Raise"
    assert all("Always" not in o and "Mostly" not in o for o in dominant[0])

    mixed = build_options(_facts({"Call": 0.6, "Fold": 0.4}), style="auto")
    assert mixed[0][0].startswith("Always")


@pytest.mark.parametrize("style", ["basic", "gto", "auto"])
@pytest.mark.parametrize(
    "freqs",
    [
        {"Call": 0.7, "Fold": 0.3},
        {"Raise 100%": 0.6, "Call": 0.4},
        {"Fold": 1.0},
        {"Call": 0.5, "Raise 100%": 0.3, "Fold": 0.2},
    ],
)
def test_correct_answer_is_always_one_of_the_options(style, freqs):
    opts, correct = build_options(_facts(freqs), style=style)
    assert correct in opts
    assert 1 <= len(opts) <= 4  # noqa: PLR2004


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="unknown answer style"):
        build_options(_facts({"Call": 1.0}), style="bogus")


# --- check spots (BB closing a limped pot: "Call" must read "Check") --------
def _limped_bb(freqs: dict[str, float]) -> PloFacts:
    """BB facing a limp (SB completed, no raise) -- a check spot."""
    return _facts(freqs, history=(PloAction("SB", C),), actor="BB")


def test_canonicalize_action_label_check_spot_flag():
    """The check_spot flag relabels Call -> Check; nothing else changes."""
    assert canonicalize_action_label("Call", raise_level=1, check_spot=True) == "Check"
    assert canonicalize_action_label("Call", raise_level=1, check_spot=False) == "Call"
    assert canonicalize_action_label("Fold", raise_level=1, check_spot=True) == "Fold"
    assert canonicalize_action_label("Raise 100%", raise_level=1, check_spot=True) == "Raise"


def test_is_check_spot_only_for_bb_with_no_raise():
    # BB facing a limp -> check spot.
    assert is_check_spot(_limped_bb({"Call": 0.95, "Raise 100%": 0.05}))
    # BB facing a raise -> a real call, not a check.
    assert not is_check_spot(
        _facts({"Call": 0.6, "Fold": 0.4}, history=(PloAction("LJ", R, 100),), actor="BB")
    )
    # Non-BB with no raise -> not a check spot (only the BB checks preflop).
    assert not is_check_spot(_facts({"Call": 0.9, "Raise 100%": 0.1}, actor="SB"))


def test_check_spot_canonicalizes_call_to_check():
    canon = canonicalize_strategy(_limped_bb({"Call": 0.95, "Raise 100%": 0.05}))
    assert canon == {"Check": 0.95, "Raise": 0.05}
    assert "Call" not in canon


def test_check_spot_basic_answer_is_check():
    opts, correct = build_options(
        _limped_bb({"Call": 0.95, "Raise 100%": 0.05}), style="basic"
    )
    assert correct == "Check"
    assert "Check" in opts
    assert "Call" not in opts


def test_check_spot_gto_mostly_check_not_call():
    # Near-pure check -> "Mostly Check" correct (97% is a mix; "Always" is only
    # for a literally-pure action). The label is Check, not Call -- the point.
    opts, correct = build_options(
        _limped_bb({"Call": 0.97, "Raise 100%": 0.03}), style="gto"
    )
    assert correct == "Mostly Check"
    assert "Always Check" in opts  # the rung still exists (as a neutral)
    assert not any("Call" in o for o in opts)


def test_bb_facing_raise_still_calls():
    facts = _facts(
        {"Call": 0.6, "Fold": 0.4}, history=(PloAction("LJ", R, 100),), actor="BB"
    )
    canon = canonicalize_strategy(facts)
    assert "Call" in canon
    assert "Check" not in canon


def test_non_bb_no_raise_still_calls():
    canon = canonicalize_strategy(_facts({"Call": 0.9, "Raise 100%": 0.1}, actor="SB"))
    assert "Call" in canon
    assert "Check" not in canon


def test_integer_percentages_shared_by_csv_and_solver_data():
    """One allocation for every percentage surface (July 2026): the CSV
    action_frequencies column and the SOLVER DATA action_strategy previously
    used different rounding (largest-remainder vs naive round), which
    disagree on exact-.5 boundaries -- the player saw "Call: 99%" while the
    LLM read "Call: 98%" for the same node."""
    from pipeline.plo.options import integer_percentages

    # The observed boundary case: naive round() gives 98/2 (banker's), the
    # column's largest-remainder gives 99/1. Both surfaces must say 99/1.
    ints = integer_percentages({"Call": 0.985, "Fold": 0.015})
    assert ints == {"Call": 99, "Fold": 1}
    assert sum(ints.values()) == 100
    # Sums to exactly 100 on an awkward three-way split too.
    ints3 = integer_percentages({"Call": 1 / 3, "Fold": 1 / 3, "3-bet": 1 / 3})
    assert sum(ints3.values()) == 100
    # Labels come back highest-frequency first (the column's display order).
    assert list(integer_percentages({"Fold": 0.2, "Call": 0.8})) == ["Call", "Fold"]


def test_integer_percentages_never_hide_a_real_mix_as_100_0():
    """HONESTY CLAMP (July 2026, user rule): a genuinely-mixed strategy must
    never display 100%/0% -- the 10bb MTT spot Check 99.5/Raise 0.5 rendered
    "Check: 100%, Raise: 0%" next to a correct answer of "Mostly Check" (the
    Always qualifier requires a literally-pure action). Display and qualifier
    share one purity test so they can never contradict each other."""
    from pipeline.plo.options import _PURE_STRATEGY_PREFIX, integer_percentages

    # The observed spot: 99.5/0.5 shows 99/1, not 100/0.
    assert integer_percentages({"Check": 0.995, "Raise": 0.005}) == {
        "Check": 99,
        "Raise": 1,
    }
    # Two slivers: each shows 1, dominant gives up the difference; sum 100.
    ints = integer_percentages({"Check": 0.994, "Raise": 0.003, "All-in": 0.003})
    assert ints == {"Check": 98, "Raise": 1, "All-in": 1}
    # Literally pure stays 100.
    assert integer_percentages({"Check": 1.0}) == {"Check": 100}
    # Dust below the purity epsilon: the dominant IS "Always" to the
    # qualifier, so the display may say 100/0 -- still consistent.
    assert integer_percentages({"Check": 0.99995, "Raise": 0.00005}) == {
        "Check": 100,
        "Raise": 0,
    }
    assert 0.99995 >= _PURE_STRATEGY_PREFIX
    # A zero-frequency listed option stays 0 (options list zero-freq actions).
    ints0 = integer_percentages({"Check": 0.995, "Raise": 0.005, "Fold": 0.0})
    assert ints0["Fold"] == 0 and sum(ints0.values()) == 100
