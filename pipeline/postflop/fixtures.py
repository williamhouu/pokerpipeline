"""Synthetic postflop solves for tests and the end-to-end demo.

These build a :class:`~pipeline.postflop.solve.PostflopSolve` entirely in
memory, so the whole postflop pipeline runs with NO external solver file.
That matters for two reasons: this machine cannot run PioSolver (Windows
only), and the real trial solves are multi-GB and live outside the repo.

The numbers here are *plausible but not solver-exact*. The fixture's job is
to exercise the pipeline plumbing and to be a stable, deterministic input for
tests -- not to be a production equity source. When a real solve is adapted
into the IR, the pipeline code does not change; only the input does.

The flagship fixture is :func:`btn_vs_bb_srp_2cJs7s` -- BTN opens, BB calls,
flop ``2c Js 7s`` (the repo's canonical Phase-0 board). It carries four nodes
chosen to exercise every pipeline path:

    flop_ip_cbet         BB checks, BTN to act (c-bet decision; sizing options)
    flop_ip_facing_bet   BB bets, BTN to act (call/fold/raise; pot-odds path)
    flop_oop_lead        BB first to act on the flop (check/bet)
    turn_oop             BB checks, BTN bets, BB calls; BB to act on the turn
                         (multi-street: the question must show preflop + flop)
"""

from __future__ import annotations

from pipeline.cards import RANKS, SUITS
from pipeline.postflop.solve import (
    NodeAction,
    PostflopNode,
    PostflopSolve,
    PostflopStep,
    PreflopStep,
)


# --- range helpers ----------------------------------------------------------
def _expand(spec: str) -> list[str]:
    """Expand a hand spec into concrete 4-char combos.

    ``"JJ"`` -> 6 pair combos; ``"AJs"`` -> 4 suited; ``"AJo"`` -> 12 offsuit;
    a 4-char combo like ``"AhKd"`` -> itself. Suits use the canonical order in
    :data:`pipeline.cards.SUITS`.
    """
    if len(spec) == 4 and spec[1] in SUITS and spec[3] in SUITS:
        return [spec]
    if len(spec) == 2:  # pocket pair, e.g. "JJ"
        r = spec[0]
        return [f"{r}{a}{r}{b}" for i, a in enumerate(SUITS) for b in SUITS[i + 1:]]
    hi, lo, kind = spec[0], spec[1], spec[2]
    if kind == "s":
        return [f"{hi}{s}{lo}{s}" for s in SUITS]
    return [f"{hi}{a}{lo}{b}" for a in SUITS for b in SUITS if a != b]


def _range(specs: dict[str, float]) -> dict[str, float]:
    """Expand a ``{spec: weight}`` map to a ``{combo: weight}`` range."""
    out: dict[str, float] = {}
    for spec, weight in specs.items():
        for combo in _expand(spec):
            out[combo] = weight
    return out


# A plausible BB flat-call range vs a BTN open (the villain range BTN faces
# after BB checks the flop). Weights coarsely model how often each class
# continued. Not solver-exact -- a stable, sane spread for equity.
_BB_CALL_RANGE = _range({
    "JJ": 0.5, "TT": 1.0, "99": 1.0, "88": 1.0, "77": 1.0, "66": 1.0, "55": 1.0,
    "AJs": 1.0, "KJs": 1.0, "QJs": 1.0, "JTs": 1.0, "T9s": 1.0, "98s": 1.0,
    "87s": 1.0, "76s": 1.0, "65s": 1.0, "A7s": 0.6, "A5s": 0.6, "KQs": 1.0,
    "AJo": 0.7, "KJo": 0.7, "QJo": 0.7, "JTo": 0.8, "T9o": 0.5,
})

# A plausible BTN single-raised-pot range arriving on the flop (the villain
# range BB faces). Wider and more ace-heavy than the BB call range.
_BTN_SRP_RANGE = _range({
    "AA": 0.4, "KK": 0.6, "QQ": 1.0, "JJ": 1.0, "TT": 1.0, "99": 1.0, "88": 1.0,
    "77": 1.0, "66": 1.0, "55": 1.0, "AKs": 1.0, "AQs": 1.0, "AJs": 1.0,
    "ATs": 1.0, "KQs": 1.0, "KJs": 1.0, "QJs": 1.0, "JTs": 1.0, "T9s": 1.0,
    "98s": 1.0, "87s": 1.0, "76s": 1.0, "AQo": 0.8, "AJo": 0.8, "KQo": 0.8,
})


def _action(label, verb, freq, *, to_bb=None, pot_fraction=None, ev_bb=None):
    return NodeAction(
        label=label, verb=verb, freq=freq, to_bb=to_bb,
        pot_fraction=pot_fraction, ev_bb=ev_bb,
    )


def btn_vs_bb_srp_2cJs7s() -> PostflopSolve:
    """The canonical synthetic solve: BTN open, BB call, flop ``2c Js 7s``.

    100bb start, 0.5/1 blinds. Preflop: BTN opens to 2.5bb, BB calls. Pot
    entering the flop is 5.5bb (2.5 + 2.5 + 0.5 dead SB); effective stack
    97.5bb. See the module docstring for the four nodes.
    """
    flop = ("2c", "Js", "7s")
    pre = (
        PreflopStep("BTN", "open", to_bb=2.5),
        PreflopStep("BB", "call"),
    )

    # --- node 1: BB checks, BTN to act (c-bet decision) --------------------
    # Range-aggregate action EVs (vs Check baseline) give the EV-gap signal;
    # per-combo EVs (combo_evs) refine it for each worthy hero hand.
    cbet_actions = (
        _action("Check", "check", 0.45, ev_bb=2.05),
        _action("Bet 2bb", "bet", 0.40, to_bb=1.8, pot_fraction=0.33, ev_bb=2.30),
        _action("Bet 4bb", "bet", 0.15, to_bb=4.1, pot_fraction=0.75, ev_bb=2.10),
    )
    cbet = PostflopNode(
        node_id="flop_ip_cbet",
        street="flop",
        board=flop,
        actor="BTN",
        villain="BB",
        pot_bb=5.5,
        effective_stack_bb=97.5,
        to_call_bb=0.0,
        actions=cbet_actions,
        history=(PostflopStep("flop", "BB", "check"),),
        hero_range=dict(_BTN_SRP_RANGE, **{"AcJc": 1.0, "9c8c": 1.0, "QdQh": 1.0}),
        villain_range=_BB_CALL_RANGE,
        strategy={
            # Top pair top kicker -- wants to bet big; checking clearly worse.
            "AcJc": {"Bet 4bb": 0.70, "Bet 2bb": 0.20, "Check": 0.10},
            # Air with a backdoor -- give up; betting clearly worse.
            "9c8c": {"Check": 0.85, "Bet 2bb": 0.15},
            # Overpair to the board -- bet for value; size mix but bet-dominant.
            "QdQh": {"Bet 2bb": 0.78, "Bet 4bb": 0.12, "Check": 0.10},
        },
        combo_evs={
            "AcJc": {"Bet 4bb": 3.2, "Bet 2bb": 2.6, "Check": 2.0},
            "9c8c": {"Check": 0.9, "Bet 2bb": 0.3, "Bet 4bb": -0.2},
            "QdQh": {"Bet 2bb": 3.6, "Bet 4bb": 3.0, "Check": 2.7},
        },
    )

    # --- node 2: BB bets 33%, BTN to act (facing a bet; pot-odds path) -----
    facing_actions = (
        _action("Fold", "fold", 0.30, ev_bb=0.0),
        _action("Call", "call", 0.55, to_bb=1.8, ev_bb=0.95),
        _action("Raise", "raise", 0.15, to_bb=6.0, pot_fraction=0.82, ev_bb=0.70),
    )
    facing = PostflopNode(
        node_id="flop_ip_facing_bet",
        street="flop",
        board=flop,
        actor="BTN",
        villain="BB",
        pot_bb=7.3,  # 5.5 + BB's 1.8 bet
        effective_stack_bb=97.5,
        to_call_bb=1.8,
        actions=facing_actions,
        history=(PostflopStep("flop", "BB", "bet", to_bb=1.8),),
        hero_range=dict(_BTN_SRP_RANGE, **{"KsJd": 1.0, "Ah5h": 1.0}),
        villain_range=_BB_CALL_RANGE,
        strategy={
            # Top pair good kicker -- call down; raising/folding both worse.
            "KsJd": {"Call": 0.80, "Raise": 0.15, "Fold": 0.05},
            # Nut-flush draw -- mostly call, sometimes raise.
            "Ah5h": {"Call": 0.62, "Raise": 0.30, "Fold": 0.08},
        },
        combo_evs={
            "KsJd": {"Call": 1.2, "Raise": 0.6, "Fold": 0.0},
            "Ah5h": {"Call": 1.4, "Raise": 1.1, "Fold": 0.0},
        },
    )

    # --- node 3: BB first to act on the flop (lead/check decision) ---------
    lead_actions = (
        _action("Check", "check", 0.80, ev_bb=2.4),
        _action("Bet 2bb", "bet", 0.20, to_bb=1.8, pot_fraction=0.33, ev_bb=2.0),
    )
    lead = PostflopNode(
        node_id="flop_oop_lead",
        street="flop",
        board=flop,
        actor="BB",
        villain="BTN",
        pot_bb=5.5,
        effective_stack_bb=97.5,
        to_call_bb=0.0,
        actions=lead_actions,
        history=(),
        hero_range=dict(_BB_CALL_RANGE, **{"7h6h": 1.0}),
        villain_range=_BTN_SRP_RANGE,
        strategy={
            # Bottom pair + backdoors -- check, betting clearly worse.
            "7h6h": {"Check": 0.82, "Bet 2bb": 0.18},
        },
        combo_evs={
            "7h6h": {"Check": 1.6, "Bet 2bb": 1.0},
        },
    )

    # --- node 4: turn, BB to act after check/bet/call on the flop ----------
    # MULTI-STREET: the question here must render preflop (BTN open, BB call)
    # AND the flop action (BB check, BTN bet 1.8, BB call) before the turn.
    turn_board = ("2c", "Js", "7s", "2h")
    turn_actions = (
        _action("Check", "check", 0.70, ev_bb=4.3),
        _action("Bet 4.5bb", "bet", 0.30, to_bb=4.55, pot_fraction=0.50, ev_bb=3.7),
    )
    turn = PostflopNode(
        node_id="turn_oop",
        street="turn",
        board=turn_board,
        actor="BB",
        villain="BTN",
        pot_bb=9.1,  # 5.5 + 1.8 + 1.8
        effective_stack_bb=95.7,  # 97.5 - 1.8
        to_call_bb=0.0,
        actions=turn_actions,
        history=(
            PostflopStep("flop", "BB", "check"),
            PostflopStep("flop", "BTN", "bet", to_bb=1.8),
            PostflopStep("flop", "BB", "call", to_bb=1.8),
        ),
        hero_range=dict(_BB_CALL_RANGE, **{"Jh9c": 1.0}),
        villain_range=_BTN_SRP_RANGE,
        strategy={
            # Top pair on a paired turn -- check (pot control), betting worse.
            "Jh9c": {"Check": 0.72, "Bet 4.5bb": 0.28},
        },
        combo_evs={
            "Jh9c": {"Check": 4.6, "Bet 4.5bb": 4.0},
        },
    )

    return PostflopSolve(
        solve_id="btn_vs_bb_srp_2cJs7s",
        positions=("BB", "BTN"),
        effective_stack_bb=100.0,
        starting_pot_bb=5.5,
        flop=flop,
        preflop_summary=pre,
        nodes={
            n.node_id: n for n in (cbet, facing, lead, turn)
        },
        game_format="cash",
        bb_in_dollars=1.0,
        stakes="$0.50/$1",
        live_or_online="Online",
        table_size=6,
        source_reference="synthetic/btn_vs_bb_srp/2cJs7s",
        # Raw preflop-entry weights per seat (open / call-vs-open frequencies).
        # The BB call range carries genuine mixes (JJ 0.5, A7s 0.6, AJo 0.7),
        # so preflop-entry questions have worthy (mixed-frequency) hands.
        preflop_entry_ranges={"BB": dict(_BB_CALL_RANGE), "BTN": dict(_BTN_SRP_RANGE)},
    )


def btn_vs_bb_full_hand_2cJs7s() -> PostflopSolve:
    """A CONNECTED single-hand line for the play-through assembler.

    Unlike :func:`btn_vs_bb_srp_2cJs7s` (four illustrative, disconnected nodes),
    this is one coherent runout where BB (the hero) check-calls down: flop
    ``2c Js 7s``, turn ``2h``, river ``Kd``. Node ids use the ``.db`` grammar
    with its CUMULATIVE ``b`` tokens (each ``b<chips>`` is the actor's total
    postflop chips committed for the WHOLE line, Pio semantics -- so the 4.55bb
    turn barrel after a 1.8bb flop bet is ``b635`` = 180 + 455, and the 9bb
    river bet is ``b1535`` = 635 + 900) and every node carries the full action
    ``history``, so
    the assembler's connectivity test -- a node is an ancestor of another when
    its board AND history are prefixes -- is exercised end to end. BB holds
    ``Jh9c`` (a one-pair bluff-catcher) throughout, making five hero decisions
    (flop lead, flop facing the c-bet, turn lead, river lead, river facing the
    bet) plus the preflop entry. Two BTN nodes are included so hero-only
    filtering (skip the villain's decisions) is testable.
    """
    flop = ("2c", "Js", "7s")
    turn_board = ("2c", "Js", "7s", "2h")
    river_board = ("2c", "Js", "7s", "2h", "Kd")
    pre = (PreflopStep("BTN", "open", to_bb=2.5), PreflopStep("BB", "call"))
    hero = "Jh9c"

    # The flop action sequence shared by every later node (BB check, BTN bet, BB
    # call), then the turn sequence, then the river sequence.
    f_check = PostflopStep("flop", "BB", "check")
    f_bet = PostflopStep("flop", "BTN", "bet", to_bb=1.8)
    f_call = PostflopStep("flop", "BB", "call", to_bb=1.8)
    t_check = PostflopStep("turn", "BB", "check")
    t_bet = PostflopStep("turn", "BTN", "bet", to_bb=4.55)
    t_call = PostflopStep("turn", "BB", "call", to_bb=4.55)
    r_check = PostflopStep("river", "BB", "check")
    r_bet = PostflopStep("river", "BTN", "bet", to_bb=9.0)

    def _node(node_id, street, board, actor, villain, pot, eff, to_call,
              actions, history, strat):
        vill_range = _BTN_SRP_RANGE if actor == "BB" else dict(_BB_CALL_RANGE, **{hero: 1.0})
        hero_range = dict(_BB_CALL_RANGE, **{hero: 1.0}) if actor == "BB" else _BTN_SRP_RANGE
        return PostflopNode(
            node_id=node_id, street=street, board=board, actor=actor, villain=villain,
            pot_bb=pot, effective_stack_bb=eff, to_call_bb=to_call, actions=actions,
            history=history, hero_range=hero_range, villain_range=vill_range,
            strategy=strat,
        )

    lead_a = (_action("Check", "check", 0.80, ev_bb=2.4),
              _action("Bet 2bb", "bet", 0.20, to_bb=1.8, pot_fraction=0.33, ev_bb=2.0))
    face_a = (_action("Fold", "fold", 0.20, ev_bb=0.0),
              _action("Call", "call", 0.70, to_bb=1.8, ev_bb=0.9),
              _action("Raise to 6bb", "raise", 0.10, to_bb=6.0, pot_fraction=0.82, ev_bb=0.6))
    cbet_a = (_action("Check", "check", 0.40, ev_bb=2.0),
              _action("Bet 2bb", "bet", 0.60, to_bb=1.8, pot_fraction=0.33, ev_bb=2.3))

    n1 = _node("r:0", "flop", flop, "BB", "BTN", 5.5, 97.5, 0.0, lead_a, (),
               {hero: {"Check": 0.80, "Bet 2bb": 0.20}})
    n2 = _node("r:0:c", "flop", flop, "BTN", "BB", 5.5, 97.5, 0.0, cbet_a, (f_check,),
               {"AcJc": {"Bet 2bb": 0.7, "Check": 0.3}})
    n3 = _node("r:0:c:b180", "flop", flop, "BB", "BTN", 7.3, 97.5, 1.8, face_a,
               (f_check, f_bet), {hero: {"Call": 0.70, "Fold": 0.20, "Raise to 6bb": 0.10}})
    t_lead = (_action("Check", "check", 0.75, ev_bb=4.3),
              _action("Bet 4.5bb", "bet", 0.25, to_bb=4.55, pot_fraction=0.50, ev_bb=3.7))
    n4 = _node("r:0:c:b180:c:2h", "turn", turn_board, "BB", "BTN", 9.1, 95.7, 0.0,
               t_lead, (f_check, f_bet, f_call), {hero: {"Check": 0.75, "Bet 4.5bb": 0.25}})
    t_face = (_action("Fold", "fold", 0.40, ev_bb=0.0),
              _action("Call", "call", 0.60, to_bb=4.55, ev_bb=1.2))
    n5 = _node("r:0:c:b180:c:2h:c:b635", "turn", turn_board, "BB", "BTN", 18.2, 91.2,
               4.55, t_face, (f_check, f_bet, f_call, t_check, t_bet),
               {hero: {"Call": 0.60, "Fold": 0.40}})
    r_lead = (_action("Check", "check", 0.60, ev_bb=5.0),
              _action("Bet 9bb", "bet", 0.40, to_bb=9.0, pot_fraction=0.50, ev_bb=4.6))
    n6 = _node("r:0:c:b180:c:2h:c:b635:c:Kd", "river", river_board, "BB", "BTN", 18.2,
               91.2, 0.0, r_lead, (f_check, f_bet, f_call, t_check, t_bet, t_call),
               {hero: {"Check": 0.60, "Bet 9bb": 0.40}})
    # 70/30 (not the old 55/45): the river facing-bet node must be WORTHY
    # (65-99 window) so the fixture's canonical hand is seeded HERE and ends
    # on the river -- the July 22 2026 no-mid-hand-endings rule outlaws
    # non-fold flop/turn enders, so a fixture whose deepest worthy seed was
    # the turn check-lead could no longer assemble a single legal hand.
    r_face = (_action("Fold", "fold", 0.30, ev_bb=0.0),
              _action("Call", "call", 0.70, to_bb=9.0, ev_bb=0.8))
    n7 = _node("r:0:c:b180:c:2h:c:b635:c:Kd:c:b1535", "river", river_board, "BB", "BTN",
               36.2, 82.2, 9.0, r_face,
               (f_check, f_bet, f_call, t_check, t_bet, t_call, r_check, r_bet),
               {hero: {"Call": 0.70, "Fold": 0.30}})

    # BTN's river barrel decision after BB checks (July 22 2026): BTN's line
    # needs a RIVER decision node, or every BTN/villain-frame hand ends on
    # the flop check-back -- a non-fold early ending the no-mid-hand-endings
    # rule rightly drops, which would leave the fixture with a single
    # assemblable hand (and no villain-frame hand at all).
    n8 = _node("r:0:c:b180:c:2h:c:b635:c:Kd:c", "river", river_board, "BTN", "BB",
               18.2, 91.2, 0.0,
               (_action("Check", "check", 0.30, ev_bb=4.4),
                _action("Bet 9bb", "bet", 0.70, to_bb=9.0, pot_fraction=0.50,
                        ev_bb=5.1)),
               (f_check, f_bet, f_call, t_check, t_bet, t_call, r_check),
               {"AcJc": {"Bet 9bb": 0.70, "Check": 0.30}})

    return PostflopSolve(
        solve_id="btn_vs_bb_full_hand_2cJs7s",
        positions=("BB", "BTN"),
        effective_stack_bb=100.0,
        starting_pot_bb=5.5,
        flop=flop,
        preflop_summary=pre,
        nodes={n.node_id: n for n in (n1, n2, n3, n4, n5, n6, n7, n8)},
        game_format="cash",
        bb_in_dollars=1.0,
        stakes="$0.50/$1",
        live_or_online="Online",
        table_size=6,
        source_reference="synthetic/btn_vs_bb_full_hand/2cJs7s",
        preflop_entry_ranges={
            "BB": dict(_BB_CALL_RANGE, **{hero: 0.65}),
            "BTN": dict(_BTN_SRP_RANGE),
        },
    )


# Sanity: the canonical flop is drawn from the standard deck (guards typos).
assert all(c[0] in RANKS and c[1] in SUITS for c in btn_vs_bb_srp_2cJs7s().flop)


__all__ = ["btn_vs_bb_full_hand_2cJs7s", "btn_vs_bb_srp_2cJs7s"]
