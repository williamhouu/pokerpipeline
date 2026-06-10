"""Tests for pipeline.plo.fact_extractor (villain, range stats, archetype)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.fact_extractor import (  # noqa: E402
    classify_plo_archetype,
    extract_plo_facts,
    identify_villain,
    villain_range_stem,
)
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.hand_order import HAND_COUNT, cards_at, combo_multiplicity  # noqa: E402
from pipeline.plo.node_enumerator import (  # noqa: E402
    PloActionOption,
    PloDecisionNode,
    action_label,
    enumerate_plo_nodes,
)
from pipeline.plo.pack import PloAction, PloActionType, PloPack  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot, sample_plo_spot  # noqa: E402

F = PloActionType.FOLD
C = PloActionType.CALL
R = PloActionType.RAISE
J = PloActionType.ALL_IN

VALUE_CARDS = ("As", "Ks", "Ah", "Kh")  # double-suited AAKK -> premium
SPEC_CARDS = ("2c", "7d", "3h", "9s")  # rainbow nine-high -> marginal (not value)
TRASH_CARDS = ("2c", "7d", "3h", "8s")  # -> trash


def _act(seat: str, atype: PloActionType, pct: int | None = None) -> PloAction:
    return PloAction(seat=seat, action=atype, raise_pct=pct)


def _spot(
    *,
    history: tuple[PloAction, ...],
    options: tuple[tuple[PloActionType, int | None], ...],
    dominant: tuple[PloActionType, int | None],
    hero_cards: tuple[str, str, str, str],
    presence: float = 1.0,
    actor: str = "HE",
) -> PloSpot:
    """A minimal PloSpot for archetype unit tests (no pack files needed)."""
    opts = tuple(
        PloActionOption(action=_act("HE", at, pct), path=Path(f"{at.value}.rng"))
        for at, pct in options
    )
    node = PloDecisionNode(
        actor=actor, history_before=history, actions=opts, history_stem=""
    )
    dom_label = action_label(_act("HE", *dominant))
    freqs = {
        action_label(_act("HE", at, pct)): (1.0 if action_label(_act("HE", at, pct)) == dom_label else 0.0)
        for at, pct in options
    }
    return PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=hero_cards,
        action_frequencies=freqs,
        presence=presence,
    )


def _classify(spot: PloSpot, villain: PloAction | None) -> str:
    return classify_plo_archetype(
        spot, classify_plo_hand(spot.hero_cards), villain_action=villain
    )


# --- identify_villain -----------------------------------------------------
def test_identify_villain_returns_last_raiser():
    history = (_act("LJ", R, 100), _act("HJ", C), _act("CO", R, 100))
    node = PloDecisionNode(actor="BU", history_before=history, actions=(), history_stem="")
    result = identify_villain(node)
    assert result is not None
    idx, villain = result
    assert idx == 2  # noqa: PLR2004  # the CO 3-bet, not the LJ open
    assert villain.seat == "CO"


def test_identify_villain_none_when_first_in():
    node = PloDecisionNode(
        actor="LJ",
        history_before=(_act("LJ", F), _act("HJ", F)),
        actions=(),
        history_stem="",
    )
    assert identify_villain(node) is None


# --- villain_range_stem ---------------------------------------------------
def test_villain_range_stem_reconstructs_path():
    node = PloDecisionNode(
        actor="HJ",
        history_before=(_act("LJ", R, 100),),
        actions=(),
        history_stem="40100",
    )
    assert villain_range_stem(node, 0) == "40100"


def test_villain_range_stem_mid_history():
    node = PloDecisionNode(
        actor="SB",
        history_before=(_act("LJ", R, 100), _act("HJ", C), _act("CO", R, 100)),
        actions=(),
        history_stem="40100.1.40100",
    )
    assert villain_range_stem(node, 2) == "40100.1.40100"  # up to the CO 3-bet


# --- classify_plo_archetype -----------------------------------------------
def test_open_for_value_and_open_fold():
    opts = ((F, None), (R, 100))
    raise_spot = _spot(history=(), options=opts, dominant=(R, 100), hero_cards=VALUE_CARDS)
    fold_spot = _spot(history=(), options=opts, dominant=(F, None), hero_cards=TRASH_CARDS)
    assert _classify(raise_spot, None) == "open_for_value"
    assert _classify(fold_spot, None) == "open_fold"


def test_bb_check_in_limped_pot():
    # BB facing a limp (SB completed, no raise): the no-raise action is a
    # CHECK, not an open-fold. No aggression in history -> villain is None.
    spot = _spot(
        history=(_act("SB", C),),
        options=((C, None), (R, 100)),
        dominant=(C, None),
        hero_cards=TRASH_CARDS,
        actor="BB",
    )
    assert _classify(spot, None) == "bb_check"


def test_bb_raise_over_limp_is_not_bb_check():
    # Raising over a limp is not a check; only a dominant no-raise action is.
    spot = _spot(
        history=(_act("SB", C),),
        options=((C, None), (R, 100)),
        dominant=(R, 100),
        hero_cards=VALUE_CARDS,
        actor="BB",
    )
    assert _classify(spot, None) != "bb_check"


def test_non_bb_no_raise_is_not_bb_check():
    # bb_check is BB-only; a non-BB first-in never gets it.
    spot = _spot(
        history=(),
        options=((F, None), (R, 100)),
        dominant=(F, None),
        hero_cards=TRASH_CARDS,
        actor="CO",
    )
    assert _classify(spot, None) != "bb_check"


def test_sb_completing_first_in_is_sb_complete_not_open_fold():
    # The SB first-in calling is COMPLETING the half bet (a limp), neither an
    # open nor a fold -- "open_fold" handed the LLM a fold frame for a Call
    # answer (the EASYYYYY batch row 1 bug).
    spot = _spot(
        history=(),
        options=((F, None), (C, None), (R, 100)),
        dominant=(C, None),
        hero_cards=TRASH_CARDS,
        actor="SB",
    )
    assert _classify(spot, None) == "sb_complete"
    # SB folding or raising first-in keeps the normal open labels.
    fold_spot = _spot(
        history=(),
        options=((F, None), (C, None), (R, 100)),
        dominant=(F, None),
        hero_cards=TRASH_CARDS,
        actor="SB",
    )
    assert _classify(fold_spot, None) == "open_fold"
    raise_spot = _spot(
        history=(),
        options=((F, None), (C, None), (R, 100)),
        dominant=(R, 100),
        hero_cards=VALUE_CARDS,
        actor="SB",
    )
    assert _classify(raise_spot, None) == "open_for_value"


def test_3bet_value_vs_bluff():
    history = (_act("LJ", R, 100),)
    opts = ((F, None), (C, None), (R, 100))
    villain = _act("LJ", R, 100)
    value = _spot(history=history, options=opts, dominant=(R, 100), hero_cards=VALUE_CARDS)
    bluff = _spot(history=history, options=opts, dominant=(R, 100), hero_cards=TRASH_CARDS)
    assert _classify(value, villain) == "3bet_for_value"
    assert _classify(bluff, villain) == "3bet_as_bluff"


def test_call_value_vs_implied_odds():
    history = (_act("LJ", R, 100),)
    opts = ((F, None), (C, None), (R, 100))
    villain = _act("LJ", R, 100)
    value = _spot(history=history, options=opts, dominant=(C, None), hero_cards=VALUE_CARDS)
    spec = _spot(history=history, options=opts, dominant=(C, None), hero_cards=SPEC_CARDS)
    assert _classify(value, villain) == "call_for_value"
    assert _classify(spec, villain) == "call_for_implied_odds"


def test_call_allin_when_jam_in_history():
    history = (_act("LJ", R, 100), _act("BB", J))
    opts = ((F, None), (C, None))
    villain = _act("BB", J)
    spot = _spot(history=history, options=opts, dominant=(C, None), hero_cards=VALUE_CARDS)
    # A call of a jam is a pot-odds call, not implied odds -- even with a premium.
    assert _classify(spot, villain) == "call_allin"


def test_squeeze_detected_with_caller_between_raise_and_hero():
    history = (_act("LJ", R, 100), _act("HJ", C))
    opts = ((F, None), (C, None), (R, 100))
    villain = _act("LJ", R, 100)
    spot = _spot(history=history, options=opts, dominant=(R, 100), hero_cards=VALUE_CARDS)
    assert _classify(spot, villain) == "squeeze_for_value"


def test_4bet_over_two_raises():
    history = (_act("LJ", R, 100), _act("HJ", R, 100))
    opts = ((F, None), (C, None), (R, 100))
    villain = _act("HJ", R, 100)
    spot = _spot(history=history, options=opts, dominant=(R, 100), hero_cards=VALUE_CARDS)
    assert _classify(spot, villain) == "4bet_for_value"


def test_fold_dominated_vs_pot_odds():
    history = (_act("LJ", R, 100),)
    opts = ((F, None), (C, None), (R, 100))
    villain = _act("LJ", R, 100)
    trash = _spot(history=history, options=opts, dominant=(F, None), hero_cards=TRASH_CARDS)
    playable = _spot(history=history, options=opts, dominant=(F, None), hero_cards=VALUE_CARDS)
    assert _classify(trash, villain) == "fold_dominated"
    assert _classify(playable, villain) == "fold_pot_odds"


def test_unclassified_when_no_presence():
    history = (_act("LJ", R, 100),)
    opts = ((F, None), (C, None), (R, 100))
    spot = _spot(
        history=history, options=opts, dominant=(C, None), hero_cards=VALUE_CARDS, presence=0.0
    )
    assert _classify(spot, _act("LJ", R, 100)) == "unclassified"


# --- extract_plo_facts (integration, synthetic pack) ----------------------
def _write_rng(path: Path, weights: dict[int, tuple[float, float]]) -> None:
    out: list[str] = []
    for i in range(HAND_COUNT):
        out.append("????")
        p, ev = weights.get(i, (0.0, 0.0))
        out.append(f"{p};{ev * 1000}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


HERO = 100


def test_extract_facts_open_spot_has_no_villain(tmp_path):
    _write_rng(tmp_path / "0.rng", {HERO: (0.0, -1.0)})
    _write_rng(tmp_path / "40100.rng", {HERO: (1.0, 5.0)})
    pack = PloPack(root=tmp_path, label="t")
    node = next(n for n in enumerate_plo_nodes(pack) if n.actor == "LJ")
    facts = extract_plo_facts(sample_plo_spot(node, HERO), pack)
    assert facts.villain_stats is None
    assert facts.archetype == "open_for_value"
    assert facts.hand_class == classify_plo_hand(cards_at(HERO))


def test_extract_facts_facing_open_has_combo_weighted_villain_stats(tmp_path):
    # LJ's open range: AAAA (mult 1), HERO (full), KAAA (mult 4, half weight).
    _write_rng(tmp_path / "40100.rng", {0: (1.0, 5.0), HERO: (1.0, 3.0), 24: (0.5, 2.0)})
    _write_rng(tmp_path / "40100.0.rng", {HERO: (0.0, -7.0)})  # HJ fold
    _write_rng(tmp_path / "40100.1.rng", {HERO: (1.0, 1.0)})  # HJ call
    _write_rng(tmp_path / "40100.40100.rng", {HERO: (0.0, -2.0)})  # HJ 3-bet
    pack = PloPack(root=tmp_path, label="t")
    node = next(n for n in enumerate_plo_nodes(pack) if n.actor == "HJ")
    facts = extract_plo_facts(sample_plo_spot(node, HERO), pack)

    vs = facts.villain_stats
    assert vs is not None
    assert vs.seat == "LJ"
    assert vs.action_label == "Raise 100%"
    expected = (
        1.0 * combo_multiplicity(0)
        + 1.0 * combo_multiplicity(HERO)
        + 0.5 * combo_multiplicity(24)
    )
    assert vs.weighted_combo_count == pytest.approx(expected)
    assert vs.pct_of_dealt_hands == pytest.approx(expected / 270725 * 100)
    assert vs.top_hands[0] == ("AAAA", 1.0)  # highest weight, lowest index

    # HJ calls the open; ev gap = best(call 1.0) - second(3bet -2.0) = 3 sb = 1.5 bb.
    assert facts.archetype in {"call_for_value", "call_for_implied_odds"}
    assert facts.ev_gap_bb == pytest.approx(1.5)


def test_extract_facts_missing_villain_file_degrades_gracefully(tmp_path):
    # An HJ node whose villain (LJ open) range file is absent -> no stats, no raise.
    _write_rng(tmp_path / "40100.1.rng", {HERO: (1.0, 1.0)})
    _write_rng(tmp_path / "40100.0.rng", {HERO: (0.0, -7.0)})
    (tmp_path / "40100.rng").unlink(missing_ok=True)  # ensure villain file absent
    pack = PloPack(root=tmp_path, label="t")
    node = next(n for n in enumerate_plo_nodes(pack) if n.actor == "HJ")
    facts = extract_plo_facts(sample_plo_spot(node, HERO), pack)
    assert facts.villain_stats is None  # file missing -> graceful
    assert facts.archetype  # archetype still classified


# --- equity chunk ---------------------------------------------------------
def _facing_open_pack(tmp_path: Path) -> PloPack:
    # LJ opens a spread of hands; HJ faces with fold / call(hero) / 3-bet files.
    villain = {i: (1.0, 2.0) for i in (0, 200, 400, 800, 1600, 3200, 6400, 9000)}
    _write_rng(tmp_path / "40100.rng", villain)
    _write_rng(tmp_path / "40100.0.rng", {500: (1.0, -7.0), 600: (1.0, -7.0)})
    _write_rng(tmp_path / "40100.1.rng", {HERO: (1.0, 1.0), 700: (1.0, 0.5)})
    _write_rng(tmp_path / "40100.40100.rng", {900: (1.0, 2.0)})
    return PloPack(root=tmp_path, label="t")


def _hj_node(pack: PloPack):
    return next(n for n in enumerate_plo_nodes(pack) if n.actor == "HJ")


def test_equity_computed_for_facing_open(tmp_path):
    pack = _facing_open_pack(tmp_path)
    facts = extract_plo_facts(
        sample_plo_spot(_hj_node(pack), HERO),
        pack,
        equity_runouts=8,
        rng=random.Random(1),
    )
    assert facts.hero_equity_vs_villain is not None
    assert 0.0 <= facts.hero_equity_vs_villain <= 1.0
    assert facts.hero_equity_runouts_used == 8  # noqa: PLR2004
    assert facts.hero_range_equity_vs_villain is not None
    assert 0.0 <= facts.hero_range_equity_vs_villain <= 1.0


def test_compute_equity_false_skips_monte_carlo(tmp_path):
    pack = _facing_open_pack(tmp_path)
    facts = extract_plo_facts(
        sample_plo_spot(_hj_node(pack), HERO), pack, compute_equity=False
    )
    assert facts.hero_equity_vs_villain is None
    assert facts.hero_range_equity_vs_villain is None
    assert facts.hero_equity_runouts_used == 0
    assert facts.villain_stats is not None  # structural facts still computed


def test_equity_is_deterministic_with_a_seed(tmp_path):
    pack = _facing_open_pack(tmp_path)
    spot = sample_plo_spot(_hj_node(pack), HERO)
    a = extract_plo_facts(spot, pack, equity_runouts=8, rng=random.Random(7))
    b = extract_plo_facts(spot, pack, equity_runouts=8, rng=random.Random(7))
    assert a.hero_equity_vs_villain == b.hero_equity_vs_villain
    assert a.hero_range_equity_vs_villain == b.hero_range_equity_vs_villain


def test_open_spot_has_no_equity(tmp_path):
    _write_rng(tmp_path / "0.rng", {HERO: (0.0, -1.0)})
    _write_rng(tmp_path / "40100.rng", {HERO: (1.0, 5.0)})
    pack = PloPack(root=tmp_path, label="t")
    node = next(n for n in enumerate_plo_nodes(pack) if n.actor == "LJ")
    facts = extract_plo_facts(
        sample_plo_spot(node, HERO), pack, rng=random.Random(1)
    )
    assert facts.hero_equity_vs_villain is None  # no villain -> no equity
