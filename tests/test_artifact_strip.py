"""ARTIFACT-STRIP (July 2026, team standing rule) -- the question surfaces
never show an unrealistic tree-artifact all-in.

The shared realism test (>40bb AND >1.5x the pot it fires into,
``pipeline.postflop.premise``) already gated LINES and invented ANSWERS; this
feature covers the remaining surface -- the node's OWN action menu and the
frequency qualifiers. Per spot (node + combo):

* trace jam frequency (< ARTIFACT_MATERIALITY = 0.05, convergence sliver):
  STRIPPED and the rest renormalised -- the stripped strategy drives options,
  Always/Mostly qualifiers, worthiness, EVs; a 99/1 call/jam reads
  "Always Call".
* material jam frequency (>= 0.05): mixing is EV-parity, so the solver
  genuinely wants the jam -- the spot is NEVER asked anywhere (standalone,
  seed, final leg, or mid-hand question; narrated only), counted in
  ``artifact_material_spots_skipped``.
* realistic short-stack jams: not artifacts, untouched as option AND answer.

These are the user's named acceptance scenarios; the dry-run batch sweep in
``test_no_artifact_allin_label_ever_ships`` is the option-surface invariant.
"""

from __future__ import annotations

from dataclasses import replace

from pipeline.postflop.batch import _collect_worthy
from pipeline.postflop.options import build_options
from pipeline.postflop.play_through import _build_legs
from pipeline.postflop.premise import (
    ARTIFACT_MATERIALITY,
    artifact_allin_action_labels,
)
from pipeline.postflop.question_extractor import evaluate_spot
from pipeline.postflop.solve import (
    NodeAction,
    PostflopNode,
    PostflopSolve,
    PostflopStep,
)
from pipeline.postflop.spot_sampler import sample_spot, spot_action_evs_bb

FLOP = ("2c", "Js", "7s")


def _river_facing_bet_node(strategy: dict, *, allin_to_bb: float = 190.0,
                           pot_bb: float = 12.0, to_call_bb: float = 5.5,
                           effective_stack_bb: float = 190.0,
                           combo_evs: dict | None = None) -> PostflopNode:
    """Hero (BTN) faces a river bet; the node's only raise is an all-in.

    Defaults make the jam an ARTIFACT (190bb into a 12bb pot); pass
    ``allin_to_bb=20, pot_bb=15, effective_stack_bb=20`` for a realistic
    short-stack jam."""
    return PostflopNode(
        node_id="r:0:c:c:8d:c:c:Kh:b55",
        street="river",
        board=(*FLOP, "8d", "Kh"),
        actor="BTN",
        villain="BB",
        pot_bb=pot_bb,
        effective_stack_bb=effective_stack_bb,
        actions=(
            NodeAction(label="Fold", verb="fold", freq=0.2),
            NodeAction(label="Call", verb="call", freq=0.7, to_bb=to_call_bb),
            NodeAction(label="All-in", verb="raise", freq=0.1, to_bb=allin_to_bb),
        ),
        strategy=strategy,
        hero_range={c: 1.0 for c in strategy},
        villain_range={"AhKs": 1.0, "Qd9d": 1.0},
        history=(
            PostflopStep("flop", "BB", "check"),
            PostflopStep("flop", "BTN", "check"),
            PostflopStep("turn", "BB", "check"),
            PostflopStep("turn", "BTN", "check"),
            PostflopStep("river", "BB", "bet", to_bb=to_call_bb),
        ),
        to_call_bb=to_call_bb,
        combo_evs=combo_evs or {},
    )


def _solve_for(node: PostflopNode) -> PostflopSolve:
    return PostflopSolve(
        solve_id="artifact-strip-fixture",
        positions=("BB", "BTN"),
        effective_stack_bb=200.0,
        starting_pot_bb=6.5,
        flop=FLOP,
        preflop_summary=(),
        nodes={node.node_id: node},
    )


# --- the shared artifact test on the node's own action menu ------------------
def test_artifact_labels_shared_thresholds_and_raise_investment() -> None:
    # 190bb jam into a 12bb pot: artifact (>40bb AND >1.5x pot).
    node = _river_facing_bet_node({"4c4d": {"Call": 1.0}})
    assert artifact_allin_action_labels(node) == frozenset({"All-in"})
    # A real ~20bb jam into a 15bb pot fails BOTH prongs: never an artifact.
    real = _river_facing_bet_node(
        {"4c4d": {"Call": 1.0}}, allin_to_bb=20.0, pot_bb=15.0,
        effective_stack_bb=20.0,
    )
    assert artifact_allin_action_labels(real) == frozenset()
    # A raise "to" counts only the INCREMENTAL wager: hero already bet 30
    # this street, so a jam TO 60 wagers 30 more into a 100bb pot -- huge in
    # neither sense, not an artifact.
    invested = PostflopNode(
        node_id="r:0:b30:b45", street="flop", board=FLOP, actor="BB",
        villain="BTN", pot_bb=100.0, effective_stack_bb=30.0,
        actions=(
            NodeAction(label="Fold", verb="fold", freq=0.5),
            NodeAction(label="All-in", verb="raise", freq=0.5, to_bb=60.0),
        ),
        strategy={"AsAd": {"Fold": 0.5, "All-in": 0.5}},
        hero_range={"AsAd": 1.0}, villain_range={"KcKd": 1.0},
        history=(
            PostflopStep("flop", "BB", "bet", to_bb=30.0),
            PostflopStep("flop", "BTN", "raise", to_bb=45.0),
        ),
        to_call_bb=15.0,
    )
    assert artifact_allin_action_labels(invested) == frozenset()


# --- (a) the 99/1 trace case -------------------------------------------------
def test_trace_jam_stripped_renormalized_reads_always_call() -> None:
    """99% call / 1% artifact jam -> the jam is convergence dust: stripped,
    renormalised to a literal 100% Call, so the question reads "Always Call"
    (the user's named case) and the spot EXITS the featured window."""
    node = _river_facing_bet_node(
        {"4c4d": {"Call": 0.99, "All-in": 0.01}},
        combo_evs={"4c4d": {"Fold": 0.0, "Call": 0.4, "All-in": 0.35}},
    )
    spot = sample_spot(node, "4c4d")
    assert not spot.artifact_material
    assert spot.stripped_artifact_freq == 0.01
    assert "All-in" not in spot.action_frequencies
    assert spot.dominant_action == "Call"
    assert spot.dominant_frequency == 1.0  # literal 100% POST-strip
    # The Always/Mostly qualifier follows the renormalised strategy.
    options, correct = build_options(spot, style="gto")
    assert correct == "Always Call"
    assert all("All-in" not in o for o in options)
    # Renormalised to pure -> outside the 65-99% featured window (the
    # meta-solvability property: a surviving featured call/fold question is
    # a real bluff-catcher, not a monster with its raise hidden).
    assert not evaluate_spot(spot).is_worthy
    # The stripped action never resurfaces via the EV surfaces either.
    assert spot_action_evs_bb(spot) == {"Fold": 0.0, "Call": 0.4}


def test_trace_strip_below_literal_pure_reads_mostly() -> None:
    # 97/2/1: post-strip Call is ~98% -- NOT literally pure, so "Mostly Call".
    node = _river_facing_bet_node(
        {"4c4d": {"Call": 0.97, "Fold": 0.02, "All-in": 0.01}}
    )
    spot = sample_spot(node, "4c4d")
    _, correct = build_options(spot, style="gto")
    assert correct == "Mostly Call"


# --- (b) the 70/30 and 90/10 material cases ----------------------------------
def test_material_jam_spot_is_never_asked_and_counted() -> None:
    """At/above ARTIFACT_MATERIALITY the mix is real (mixing is EV-parity):
    the spot's true strategy needs a line we refuse to show, so it is
    silently skipped and counted -- never a featured question."""
    node = _river_facing_bet_node({
        "4c4d": {"Call": 0.70, "All-in": 0.30},
        "5c5d": {"Call": 0.90, "All-in": 0.10},
    })
    for combo in ("4c4d", "5c5d"):
        spot = sample_spot(node, combo)
        assert spot.artifact_material
        # Frequencies stay HONEST (unstripped) on a material spot.
        assert spot.action_frequencies["All-in"] > 0
        ev = evaluate_spot(spot)
        assert not ev.is_worthy and ev.reason.startswith("artifact-material")
    # The batch collector skips both and counts them.
    worthy, _lq, _premise, material = _collect_worthy(
        _solve_for(node), min_frequency=0.65, max_frequency=0.99,
        min_ev_gap_bb=None, quality_gate=False,
    )
    assert worthy == [] and material == 2


def test_materiality_threshold_is_five_percent() -> None:
    assert ARTIFACT_MATERIALITY == 0.05
    node = _river_facing_bet_node({
        "just_under": {"Call": 0.951, "All-in": 0.049},
        "at_threshold": {"Call": 0.95, "All-in": 0.05},
    })
    assert not sample_spot(node, "just_under").artifact_material
    assert sample_spot(node, "at_threshold").artifact_material


# --- (c) a realistic short-stack jam is untouched -----------------------------
def test_realistic_short_stack_jam_untouched_as_option_and_answer() -> None:
    """A ~20bb jam into a 15bb pot is a REAL line: never stripped, never
    material, shippable as both an option and the correct answer."""
    node = _river_facing_bet_node(
        {"AhAd": {"All-in": 0.80, "Call": 0.18, "Fold": 0.02}},
        allin_to_bb=20.0, pot_bb=15.0, to_call_bb=8.0,
        effective_stack_bb=20.0,
    )
    spot = sample_spot(node, "AhAd")
    assert spot.artifact_labels == frozenset()
    assert not spot.artifact_material and spot.stripped_artifact_freq == 0.0
    assert spot.action_frequencies["All-in"] == 0.80  # noqa: PLR2004
    assert evaluate_spot(spot).is_worthy
    options, correct = build_options(spot, style="basic")
    assert "All-in" in options
    assert correct == "All-in"  # the jam IS the answer
    _, gto_correct = build_options(spot, style="gto")
    assert gto_correct == "Mostly All-in"  # and survives on the spectrum


# --- (d) no artifact all-in label ever ships as an option ---------------------
def test_no_artifact_allin_label_in_any_option_style() -> None:
    node = _river_facing_bet_node({
        "trace": {"Call": 0.99, "All-in": 0.01},
        "sliver3way": {"Call": 0.60, "Fold": 0.38, "All-in": 0.02},
        "pure": {"Call": 1.0},
    })
    for combo in node.strategy:
        spot = sample_spot(node, combo)
        assert not spot.artifact_material  # all trace/clean -> askable
        for style in ("basic", "gto", "auto"):
            options, correct = build_options(spot, style=style)
            assert all("All-in" not in o for o in options), (combo, style, options)
            assert "All-in" not in correct


# --- full hands: material legs are narrated, never asked ----------------------
def _flop_cbet_node(strategy: dict, *, allin_to_bb: float = 190.0) -> PostflopNode:
    """Hero (BTN) after BB checks the flop; menu includes an artifact jam."""
    return PostflopNode(
        node_id="r:0:c", street="flop", board=FLOP, actor="BTN", villain="BB",
        pot_bb=6.5, effective_stack_bb=197.0,
        actions=(
            NodeAction(label="Check", verb="check", freq=0.3),
            NodeAction(label="Bet 60%", verb="bet", freq=0.6, to_bb=4.0,
                       pot_fraction=0.6),
            NodeAction(label="All-in", verb="raise", freq=0.1, to_bb=allin_to_bb),
        ),
        strategy=strategy,
        hero_range={c: 1.0 for c in strategy},
        villain_range={"Qd9d": 1.0},
        history=(PostflopStep("flop", "BB", "check"),),
    )


def _turn_anchor_node(strategy: dict) -> PostflopNode:
    return PostflopNode(
        node_id="r:0:c:b4:c:8d:b10", street="turn", board=(*FLOP, "8d"),
        actor="BTN", villain="BB", pot_bb=24.5, effective_stack_bb=193.0,
        actions=(
            NodeAction(label="Fold", verb="fold", freq=0.3),
            NodeAction(label="Call", verb="call", freq=0.6, to_bb=10.0),
            NodeAction(label="Raise to 30bb", verb="raise", freq=0.1, to_bb=30.0),
        ),
        strategy=strategy,
        hero_range={c: 1.0 for c in strategy},
        villain_range={"Qd9d": 1.0},
        history=(
            PostflopStep("flop", "BB", "check"),
            PostflopStep("flop", "BTN", "bet", to_bb=4.0),
            PostflopStep("flop", "BB", "call"),
            PostflopStep("turn", "BB", "bet", to_bb=10.0),
        ),
        to_call_bb=10.0,
    )


def test_mid_hand_trace_leg_is_asked_material_leg_is_narrated() -> None:
    anchor = _turn_anchor_node({"AsAd": {"Call": 0.7, "Fold": 0.3}})

    # TRACE mid leg (99/1): stripped, still asked, reads "Always Bet".
    flop_trace = _flop_cbet_node({"AsAd": {"Bet 60%": 0.99, "All-in": 0.01}})
    legs = _build_legs(
        _solve_for(anchor), "BTN", "AsAd", [flop_trace, anchor],
        include_preflop=False,
    )
    assert legs is not None and [leg.node_id for leg in legs] == [
        flop_trace.node_id, anchor.node_id,
    ]
    _, correct = build_options(legs[0].spot, style="gto")
    assert correct == "Always Bet"

    # MATERIAL mid leg (70/30): skipped like a forced move -- the hand
    # survives, the line is narrated, no question pauses there.
    flop_material = _flop_cbet_node({"AsAd": {"Bet 60%": 0.70, "All-in": 0.30}})
    legs = _build_legs(
        _solve_for(anchor), "BTN", "AsAd", [flop_material, anchor],
        include_preflop=False,
    )
    assert legs is not None and [leg.node_id for leg in legs] == [anchor.node_id]


# --- preflop deep packs: the same rule via pipeline.artifact_strip -----------
def test_preflop_deep_pack_allin_strip() -> None:
    """Deep packs' AllIn files are tree artifacts: the label leaves the mix
    outright (so the option builders, which read the canonical strategy's
    keys, can never ship "All-in" -- even at zero mass), trace dust is
    renormalised away, and a material jam mix marks the spot unaskable."""
    from types import SimpleNamespace

    from pipeline.preflop.grammars.types import PreflopActionType
    from pipeline.preflop.spot_sampler import PreflopSpot, strip_artifact_allins

    acts = (
        SimpleNamespace(action_type=PreflopActionType.FOLD, label="Fold"),
        SimpleNamespace(action_type=PreflopActionType.RAISE, label="Raise 100%"),
        SimpleNamespace(action_type=PreflopActionType.ALL_IN, label="AllIn"),
    )

    def _spot(freqs):
        dom = max(freqs, key=freqs.get)
        return PreflopSpot(
            node=SimpleNamespace(actions=acts), hero_hand_class="A2o",
            hero_card_combo="Ah2c", action_frequencies=freqs,
            dominant_action=dom, dominant_frequency=freqs[dom], presence=1.0,
        )

    # Zero-mass AllIn (the leaked "All-in as option 3" case): key removed.
    out = strip_artifact_allins(_spot({"Fold": 0.55, "Raise 100%": 0.45, "AllIn": 0.0}))
    assert "AllIn" not in out.action_frequencies and not out.artifact_material
    assert out.dominant_action == "Fold"
    # Trace dust: stripped + renormalised.
    out = strip_artifact_allins(_spot({"Fold": 0.55, "Raise 100%": 0.42, "AllIn": 0.03}))
    assert "AllIn" not in out.action_frequencies
    assert abs(sum(out.action_frequencies.values()) - 1.0) < 1e-9
    assert abs(out.stripped_artifact_freq - 0.03) < 1e-9
    # Material mix: never asked.
    out = strip_artifact_allins(_spot({"Fold": 0.55, "Raise 100%": 0.35, "AllIn": 0.10}))
    assert out.artifact_material


def test_hand_cannot_end_on_a_material_node() -> None:
    # Swap the anchor's sized raise for an artifact jam so its 10% mass is
    # material.
    material_anchor = replace(
        _turn_anchor_node({"AsAd": {"Call": 0.9, "All-in": 0.1}}),
        actions=(
            NodeAction(label="Fold", verb="fold", freq=0.3),
            NodeAction(label="Call", verb="call", freq=0.6, to_bb=10.0),
            NodeAction(label="All-in", verb="raise", freq=0.1, to_bb=190.0),
        ),
    )
    flop_clean = _flop_cbet_node({"AsAd": {"Bet 60%": 1.0}})
    legs = _build_legs(
        _solve_for(material_anchor), "BTN", "AsAd",
        [flop_clean, material_anchor], include_preflop=False,
    )
    assert legs is None
