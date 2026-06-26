"""Adapter: a PioSolver Edge ``.cfr`` solve (driven over UPI) -> the ``PostflopSolve`` IR.

This is the second postflop adapter (after :mod:`pipeline.postflop.adapters.
sqlite_db`). Where the ``.db`` adapter reads a static SQLite export, this one
drives a live PioSolver process over its **UPI protocol** (the same
``show_node`` / ``show_children`` / ``show_strategy`` / ``show_range`` /
``calc_ev`` / ``show_hand_order`` commands the Phase-0 client uses) and maps the
responses into the solver-agnostic IR. Everything above the adapter is
unchanged -- it runs on :mod:`pipeline.postflop.solve`, never a vendor format.

What UPI gives us, and how it maps to the IR
--------------------------------------------
* **Node string grammar is Pio's own** -- ``r:0`` (the flop OOP decision),
  ``:c`` (check / call), ``:b<chips>`` (bet/raise TO ``<chips>`` this street),
  ``:f`` (fold), ``:<card>`` (a chance card -> next street). Identical to the
  grammar :mod:`sqlite_db` parses (that ``.db`` is a Pio-family export), so the
  betting-state walk here mirrors that adapter's. Bet/raise vs check/call is
  derived from the betting STATE the node string implies, never from a label.
* **Ranges come straight from the solver.** ``show_range OOP|IP <node>`` returns
  the reach-weighted range at the node directly, so -- unlike the ``.db``
  adapter -- we do NOT reconstruct reach by multiplying action probabilities
  down the tree. ``show_hand_order`` gives the 1326-combo index map.
* **Strategy** is ``show_strategy <node>``: one 1326-float row per child, in
  ``show_children`` order, where ``row[a][i]`` is ``P(action a | combo i)``.
* **Per-action EVs** are ``calc_ev <actor> <child>`` -- the actor's EV (in chips)
  of taking the action that leads to ``child``. Best-effort: a child whose EV
  can't be read just drops that action's EV (the IR tolerates partial EVs).

Scenario geometry (chips per bb, the flop-entry pot, the effective stack, the
preflop line) is supplied by the caller, exactly as the ``.db`` adapter reads it
from its metadata table -- a raw ``.cfr`` is a postflop solve whose preflop
context lives in the scenario spec it was built from, not in the file.

v1 builds **flop** decision nodes (matching the repo's flop test solve and the
``.db`` adapter's default). Turn/river would descend through the chance-card
nodes; the betting walk already handles chance tokens, so that is a BFS
extension, not an IR change.

.. important::
   **This Mac cannot run PioSolver (Windows only), so this adapter is NOT
   verified end-to-end here.** It is written against a small UPI ``Protocol``
   (:class:`UpiClient`) and unit-tested with a mocked client. Full integration
   needs a Windows host with PioSolver Edge and a ``.cfr`` (e.g. the repo's
   ``test_solves/btn_vs_bb_srp_2cJs7s.cfr``); drive it via
   :func:`load_postflop_cfr`, which wires the real :class:`pipeline.piosolver.
   PioSolverClient` into the adapter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

from pipeline.bb_display import round_to_half_bb
from pipeline.postflop.solve import (
    STREETS,
    NodeAction,
    PostflopNode,
    PostflopSolve,
    PostflopStep,
    PreflopStep,
)

logger = logging.getLogger(__name__)

# A reach weight at/below this is treated as "this combo never arrives" (matches
# the .db adapter's _REACH_EPS).
_REACH_EPS = 1e-4
# Per-combo conditional frequencies below this are dropped from the stored mix.
_FREQ_EPS = 1e-4
HAND_COUNT = 1326


class UpiClient(Protocol):
    """The slice of the UPI client this adapter drives.

    :class:`pipeline.piosolver.PioSolverClient` satisfies this structurally, and
    a test fake implements the same methods -- so the adapter never imports a
    real solver and is fully unit-testable.
    """

    def load_tree(self, cfr_path: str, mode: str = ..., timeout: float = ...) -> None: ...
    def show_node(self, node: str) -> dict: ...
    def show_children(self, node: str) -> list[str]: ...
    def show_strategy(self, node: str) -> list[list[float]]: ...
    def show_range(self, player: str, node: str) -> list[float]: ...
    def calc_ev(self, player: str, node: str) -> dict: ...
    def show_hand_order(self) -> list[str]: ...
    def show_effective_stack(self) -> int | None: ...


def _node_tokens(node_id: str) -> list[str]:
    """Action tokens after the ``r:0`` root: ``r:0:c:b22`` -> ``["c", "b22"]``."""
    return node_id.split(":")[2:]


def _board_cards(board_line: str) -> tuple[str, ...]:
    """Parse a UPI board line (``"2c Js 7s"``) into card tokens."""
    return tuple(board_line.split())


def _actor_side(node_type: str) -> str | None:
    """``"OOP"`` / ``"IP"`` from a UPI node type, or ``None`` if not a decision.

    ``"OOP"`` is checked first since it does not contain ``"IP"`` as a substring,
    but the explicit order documents the intent.
    """
    t = node_type.upper()
    if "DEC" not in t:
        return None
    if "OOP" in t:
        return "OOP"
    if "IP" in t:
        return "IP"
    return None


@dataclass
class _BetState:
    """Running betting state as the node string is walked (chips)."""

    pot: float
    invested: dict[str, float]  # this-street chips, resets each street
    behind: dict[str, float]  # chips remaining behind, carries across streets
    to_act: str
    other: str
    street_idx: int
    board: list[str]

    def to_call(self) -> float:
        return max(self.invested.values()) - self.invested[self.to_act]


class CfrPioAdapter:
    """Build a :class:`PostflopSolve` from a loaded ``.cfr`` via a UPI client.

    The caller supplies the scenario geometry (chips/bb, the flop-entry pot, the
    effective stack, the preflop line); the adapter queries the solver for the
    tree, ranges, strategies, and EVs.
    """

    def __init__(
        self,
        client: UpiClient,
        *,
        cfr_path: str = "",
        oop_position: str = "BB",
        ip_position: str = "BTN",
        bb_chips: float = 100.0,
        starting_pot_chips: float | None = None,
        effective_stack_chips: float | None = None,
        preflop_summary: tuple[PreflopStep, ...] | None = None,
        table_size: int = 6,
        game_format: str = "cash",
        live_or_online: str = "Online",
        stakes: str = "",
        bb_in_dollars: float = 1.0,
        source_reference: str = "",
        load_mode: str = "auto",
        include_evs: bool = True,
    ):
        self.client = client
        self.cfr_path = cfr_path
        self.oop = oop_position
        self.ip = ip_position
        self.bb_chips = float(bb_chips)
        self.starting_pot_chips = starting_pot_chips
        self.effective_stack_chips = effective_stack_chips
        self.preflop_summary = preflop_summary
        self.table_size = table_size
        self.game_format = game_format
        self.live_or_online = live_or_online
        self.stakes = stakes
        self.bb_in_dollars = bb_in_dollars
        self.source_reference = source_reference
        self.load_mode = load_mode
        self.include_evs = include_evs

    # -- public entry ----------------------------------------------------
    def build(self, *, streets: tuple[str, ...] = ("flop",), max_nodes: int = 400) -> PostflopSolve:
        """Load the tree (if a ``cfr_path`` was given) and build the IR.

        ``streets`` selects which streets to materialise (v1: ``("flop",)``);
        ``max_nodes`` caps the BFS so a pathological tree can't run away.
        """
        bad = [s for s in streets if s not in STREETS]
        if bad:
            raise ValueError(f"unknown street(s) {bad}; expected a subset of {STREETS}")
        if streets != ("flop",):
            raise NotImplementedError(
                "the .cfr adapter builds flop nodes for v1; turn/river is a "
                "BFS-through-chance-nodes extension (the betting walk already "
                "handles chance tokens)."
            )
        if self.cfr_path:
            self.client.load_tree(self.cfr_path, self.load_mode)

        hand_order = self.client.show_hand_order()
        if len(hand_order) != HAND_COUNT:
            raise ValueError(
                f"show_hand_order returned {len(hand_order)} hands, expected {HAND_COUNT}"
            )

        decision_root = self._first_decision_node("r")
        flop = _board_cards(self.client.show_node(decision_root)["board"])
        if len(flop) != 3:
            raise ValueError(f"expected a 3-card flop at {decision_root}, got {flop!r}")
        eff_chips = self._effective_stack_chips()
        start_pot = self._starting_pot_chips(decision_root)

        node_ids = self._collect_flop_decision_nodes(decision_root, max_nodes=max_nodes)
        nodes: dict[str, PostflopNode] = {}
        for node_id in node_ids:
            node = self._build_node(
                node_id, hand_order=hand_order, flop=flop,
                start_pot=start_pot, eff_chips=eff_chips,
            )
            if node is not None:
                nodes[node.node_id] = node

        return PostflopSolve(
            solve_id=f"cfr_{''.join(flop)}",
            positions=(self.oop, self.ip),
            effective_stack_bb=round(eff_chips / self.bb_chips, 2),
            starting_pot_bb=round(start_pot / self.bb_chips, 2),
            flop=flop,  # type: ignore[arg-type]
            preflop_summary=self.preflop_summary or (
                PreflopStep(self.ip, "open", to_bb=2.5),
                PreflopStep(self.oop, "call"),
            ),
            nodes=nodes,
            game_format=self.game_format,
            bb_in_dollars=self.bb_in_dollars,
            stakes=self.stakes,
            live_or_online=self.live_or_online,
            table_size=self.table_size,
            source_reference=self.source_reference or f"cfr/{''.join(flop)}",
        )

    # -- tree navigation -------------------------------------------------
    def _first_decision_node(self, start: str) -> str:
        """Descend wrapper (ROOT/chance) nodes to the first decision node."""
        node = start
        for _ in range(8):  # guard a pathological tree
            info = self.client.show_node(node)
            if _actor_side(info.get("node_type", "")) is not None:
                return node
            children = self.client.show_children(node)
            if not children:
                raise ValueError(f"no decision node descending from {start!r}")
            node = children[0]
        raise ValueError(f"no decision node within 8 levels of {start!r}")

    def _collect_flop_decision_nodes(self, root: str, *, max_nodes: int) -> list[str]:
        """BFS from ``root`` collecting flop (3-card-board) decision nodes.

        Descends only through flop nodes; a chance (turn) node ends a branch for
        the v1 flop-only build. Deterministic (children are in solver order)."""
        out: list[str] = []
        todo = [root]
        seen: set[str] = set()
        while todo and len(seen) < max_nodes:
            node_id = todo.pop(0)
            if node_id in seen:
                continue
            seen.add(node_id)
            info = self.client.show_node(node_id)
            board = _board_cards(info.get("board", ""))
            if len(board) != 3:
                continue  # off the flop (a turn/river node) -> don't collect/descend
            if _actor_side(info.get("node_type", "")) is not None:
                out.append(node_id)
            todo.extend(self.client.show_children(node_id))
        return out

    # -- one node --------------------------------------------------------
    def _build_node(
        self, node_id: str, *, hand_order: list[str], flop: tuple[str, ...],
        start_pot: float, eff_chips: float,
    ) -> PostflopNode | None:
        info = self.client.show_node(node_id)
        side = _actor_side(info.get("node_type", ""))
        if side is None:
            return None
        actor = self.oop if side == "OOP" else self.ip
        villain = self.ip if side == "OOP" else self.oop

        state, history = self._walk(node_id, flop=flop, start_pot=start_pot, eff_chips=eff_chips)
        # The walk alternates strictly OOP->IP within a street; cross-check it
        # agrees with the solver's own node type (a guard against a malformed line).
        if state.to_act != actor:
            logger.warning(
                "cfr adapter: %s node-type says %s acts but the walk reached %s; "
                "trusting node type", node_id, actor, state.to_act,
            )
        to_call = max(state.invested.values()) - state.invested[actor]
        pot_before = state.pot - to_call
        eff_remaining = min(state.behind.values())

        children = self.client.show_children(node_id)
        strategy_rows = self.client.show_strategy(node_id)
        if len(strategy_rows) != len(children):
            logger.warning(
                "cfr adapter: %s has %d children but %d strategy rows; skipping",
                node_id, len(children), len(strategy_rows),
            )
            return None

        labels: list[tuple[str, str, float | None, float | None]] = [
            self._derive(_node_tokens(child)[-1], facing=to_call > 0,
                         pot_before=pot_before, to_call=to_call, eff_remaining=eff_remaining)
            for child in children
        ]

        actor_range = self.client.show_range(side, node_id)
        villain_range = self.client.show_range("IP" if side == "OOP" else "OOP", node_id)
        if len(actor_range) != HAND_COUNT or len(villain_range) != HAND_COUNT:
            return None

        # Per-action EVs (chips) for the ACTOR: calc_ev(actor, child) is the
        # actor's EV of the action leading to child. Best-effort per child.
        child_evs: list[list[float] | None] = []
        for child in children:
            if not self.include_evs:
                child_evs.append(None)
                continue
            try:
                child_evs.append(self.client.calc_ev(side, child)["ev"])
            except Exception as exc:  # noqa: BLE001 -- a missing EV just drops that action's
                logger.warning("cfr adapter: calc_ev %s %s failed: %s", side, child, exc)
                child_evs.append(None)

        strategy: dict[str, dict[str, float]] = {}
        combo_evs: dict[str, dict[str, float]] = {}
        denom = 0.0
        agg_freq = [0.0] * len(children)
        agg_ev_num = [0.0] * len(children)
        agg_ev_den = [0.0] * len(children)
        for i in range(HAND_COUNT):
            w = actor_range[i]
            if w <= _REACH_EPS:
                continue
            combo = hand_order[i]
            denom += w
            mix: dict[str, float] = {}
            evs: dict[str, float] = {}
            for a, (label, _verb, _to_bb, _pf) in enumerate(labels):
                p = strategy_rows[a][i]
                if p > _FREQ_EPS:
                    mix[label] = p
                agg_freq[a] += w * p
                ev_row = child_evs[a]
                if ev_row is not None and math.isfinite(ev_row[i]):
                    evs[label] = ev_row[i] / self.bb_chips
                    agg_ev_num[a] += w * ev_row[i]
                    agg_ev_den[a] += w
            if mix:
                strategy[combo] = mix
            if evs:
                combo_evs[combo] = evs

        if not strategy or denom <= 0:
            return None
        hero_range = {hand_order[i]: actor_range[i] for i in range(HAND_COUNT)
                      if actor_range[i] > _REACH_EPS}
        villain_rng = {hand_order[i]: villain_range[i] for i in range(HAND_COUNT)
                       if villain_range[i] > _REACH_EPS}
        if not villain_rng:
            return None  # no opponent reaches here -> no equity to measure

        final_actions = tuple(
            NodeAction(
                label=label, verb=verb, freq=agg_freq[a] / denom,
                to_bb=to_bb, pot_fraction=pf,
                ev_bb=(agg_ev_num[a] / agg_ev_den[a] / self.bb_chips
                       if agg_ev_den[a] > 0 else None),
            )
            for a, (label, verb, to_bb, pf) in enumerate(labels)
        )
        return PostflopNode(
            node_id=node_id,
            street="flop",
            board=tuple(state.board),
            actor=actor,
            villain=villain,
            pot_bb=round(state.pot / self.bb_chips, 4),
            effective_stack_bb=round(eff_remaining / self.bb_chips, 4),
            actions=final_actions,
            strategy=strategy,
            hero_range=hero_range,
            villain_range=villain_rng,
            history=history,
            to_call_bb=round(to_call / self.bb_chips, 4),
            combo_evs=combo_evs,
        )

    # -- betting-state walk (chips) --------------------------------------
    def _walk(
        self, node_id: str, *, flop: tuple[str, ...], start_pot: float, eff_chips: float,
    ) -> tuple[_BetState, tuple[PostflopStep, ...]]:
        """Walk the node string to the decision, tracking pot / invested /
        behind / board / history. Mirrors the ``.db`` adapter's walk minus the
        reach products (Pio gives ranges directly)."""
        state = _BetState(
            pot=start_pot,
            invested={self.oop: 0.0, self.ip: 0.0},
            behind={self.oop: eff_chips, self.ip: eff_chips},
            to_act=self.oop, other=self.ip, street_idx=0, board=list(flop),
        )
        history: list[PostflopStep] = []
        for token in _node_tokens(node_id):
            if _is_card(token):  # chance card -> next street (turn/river path)
                state.board.append(token)
                state.invested = {self.oop: 0.0, self.ip: 0.0}
                state.to_act, state.other = self.oop, self.ip
                state.street_idx += 1
                continue
            street = STREETS[min(state.street_idx, 2)]
            acting = state.to_act
            to_call_now = state.to_call()
            if token.startswith("b"):
                size = float(token[1:])
                verb = "raise" if to_call_now > 0 else "bet"
                history.append(PostflopStep(street, acting, verb, to_bb=round(size / self.bb_chips, 2)))
                added = size - state.invested[acting]
                state.pot += added
                state.invested[acting] = size
                state.behind[acting] -= added
            elif token == "c":
                if to_call_now > 0:
                    history.append(PostflopStep(street, acting, "call"))
                    state.pot += to_call_now
                    state.invested[acting] += to_call_now
                    state.behind[acting] -= to_call_now
                else:
                    history.append(PostflopStep(street, acting, "check"))
            elif token == "f":
                history.append(PostflopStep(street, acting, "fold"))
            else:
                raise ValueError(f"{node_id}: unrecognised node token {token!r}")
            state.to_act, state.other = state.other, state.to_act
        return state, tuple(history)

    def _derive(
        self, token: str, *, facing: bool, pot_before: float, to_call: float, eff_remaining: float,
    ) -> tuple[str, str, float | None, float | None]:
        """``(label, verb, to_bb, pot_fraction)`` for one child action, by context.

        Mirrors the ``.db`` adapter's label derivation so both sources produce
        identical option labels for the same betting state."""
        if token == "f":
            return "Fold", "fold", None, None
        if token == "c":
            if facing:
                return "Call", "call", round(to_call / self.bb_chips, 2), None
            return "Check", "check", None, None
        if token.startswith("b"):
            size = float(token[1:])
            to_bb = round(size / self.bb_chips, 2)
            disp = round_to_half_bb(to_bb)  # 0.5bb display grid (label only)
            if size >= eff_remaining - 1:
                return "All-in", ("raise" if facing else "bet"), to_bb, None
            if facing:
                return f"Raise to {disp:g}bb", "raise", to_bb, None
            pf = size / pot_before if pot_before > 0 else None
            label = f"Bet {round(pf * 100)}%" if pf is not None else f"Bet {disp:g}bb"
            return label, "bet", to_bb, pf
        raise ValueError(f"unrecognised child token {token!r}")

    # -- geometry helpers ------------------------------------------------
    def _effective_stack_chips(self) -> float:
        if self.effective_stack_chips is not None:
            return float(self.effective_stack_chips)
        eff = self.client.show_effective_stack()
        if eff is None or eff <= 0:
            raise ValueError(
                "could not read the effective stack; pass effective_stack_chips"
            )
        return float(eff)

    def _starting_pot_chips(self, decision_root: str) -> float:
        if self.starting_pot_chips is not None:
            return float(self.starting_pot_chips)
        # The flop-entry pot is the settled pot at the OOP root (no bets yet).
        # UPI's pot line is "<oop_pending> <ip_pending> <settled_pot>"; at the
        # root the pendings are 0, so the settled pot is the last value. We
        # require the caller to pass it explicitly for any non-root inference,
        # since the pot-line semantics aren't worth guessing for pricing.
        pot_line = str(self.client.show_node(decision_root).get("pot", "")).split()
        nums = [float(x) for x in pot_line if _is_float(x)]
        if not nums:
            raise ValueError(
                "could not read the flop-entry pot; pass starting_pot_chips"
            )
        return nums[-1]


def _is_card(token: str) -> bool:
    return len(token) == 2 and token[0] in "23456789TJQKA" and token[1] in "cdhs"


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_postflop_cfr(
    cfr_path: str,
    *,
    pio_exe: str | None = None,
    streets: tuple[str, ...] = ("flop",),
    max_nodes: int = 400,
    **scenario: object,
) -> PostflopSolve:
    """Load a ``.cfr`` via a real PioSolver process and build the IR.

    Locates PioSolver (``pio_exe`` / ``$PIOSOLVER_EXE`` / known install dirs),
    launches it, loads ``cfr_path``, and runs :class:`CfrPioAdapter`. ``scenario``
    is forwarded to the adapter (``oop_position`` / ``ip_position`` / ``bb_chips``
    / ``starting_pot_chips`` / ``effective_stack_chips`` / ``preflop_summary`` /
    display fields).

    .. note::
       Requires a Windows host with PioSolver Edge -- it CANNOT run on this Mac.
       The adapter itself is unit-tested with a mocked UPI client; this function
       is the thin real-solver wiring and is intentionally import-light so the
       package loads without a solver present.
    """
    from pipeline.piosolver import PioSolverClient, find_piosolver  # noqa: PLC0415

    exe = find_piosolver(pio_exe)
    if exe is None:
        raise RuntimeError(
            "PioSolver executable not found (pass pio_exe or set $PIOSOLVER_EXE). "
            "The .cfr adapter needs a Windows host with PioSolver Edge."
        )
    with PioSolverClient(exe) as client:
        adapter = CfrPioAdapter(client, cfr_path=cfr_path, **scenario)  # type: ignore[arg-type]
        return adapter.build(streets=streets, max_nodes=max_nodes)


__all__ = ["CfrPioAdapter", "UpiClient", "load_postflop_cfr"]
