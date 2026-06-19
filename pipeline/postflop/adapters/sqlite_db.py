"""Adapter: the third-party SQLite ``.db`` postflop solve -> the ``PostflopSolve`` IR.

This is the ONLY module that knows the vendor's ``.db`` layout. Everything above
runs on the solver-agnostic IR (:mod:`pipeline.postflop.solve`); when a new solve
arrives, only this adapter sees the vendor format.

The format (verified against ``BTN_vs_BB_SRP_100bb_QsJd9s_v8.db``; see the audit
notes in memory / ``docs``):

* tables: ``gto_postflop(node, action, freq_blob, ev_blob_oop, ev_blob_ip)``,
  ``hand_index(idx, hand)``, ``metadata(key, value)``,
  ``preflop_ranges(player, hand, weight)``.
* ``freq_blob``: 1326 bytes, one per ``hand_index`` combo, **0-255 CONDITIONAL**
  strategy (a combo's bytes across the node's actions sum to ~255). Divide by 255.
* ``ev_blob_oop`` / ``ev_blob_ip``: 1326 float32 LE, per-combo EV in **chips**,
  for the OOP / IP player. Use the **acting** player's blob at each node.
* node strings: ``"r:0"`` is the root (OOP acts first), then ``":c"`` (the passive
  action -- check OR call), ``":b<chips>"`` (bet/raise **to** ``<chips>`` total this
  street), ``":f"`` (fold), ``":<card>"`` (a chance card -> next street).

**Action labels are unreliable** (the vendor stores a check-back as ``CALL`` and a
call as ``CHECK``); we derive check-vs-call and bet-vs-raise from the betting
**state** implied by the node string, never from the ``action`` column. Bet SIZES
(the ``b<chips>`` token) are reliable.

v1 builds **flop** decision nodes only (node strings with no chance-card token).
Turn/river is a clean extension: the same walk generalises once a chance token
resets the street's invested amounts. Geometry, ranges, the 0-255 scale, and the
board mask were verified on this exact solve before wiring it in.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass

from pipeline.postflop.solve import (
    NodeAction,
    PostflopNode,
    PostflopSolve,
    PostflopStep,
    PreflopStep,
)

# A combo byte at/below this (out of 255) is treated as "not played".
_FREQ_EPS_BYTES = 0
# A reach probability at/below this is treated as "this combo never arrives".
_REACH_EPS = 1e-4
_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")


@dataclass(frozen=True)
class _DBAction:
    """One raw action row at a node: the vendor token + its decoded blobs."""

    db_name: str  # "BET_216" | "CHECK" | "CALL" | "FOLD"
    freq: list[int]  # 1326 bytes, 0-255
    ev_oop: tuple[float, ...]  # 1326 float32, chips
    ev_ip: tuple[float, ...]


def _split_cards(s: str) -> list[str]:
    return [s[i : i + 2] for i in range(0, len(s), 2)]


def _node_tokens(node_id: str) -> list[str]:
    """Action tokens after the ``r:0`` root, e.g. ``r:0:c:b216`` -> ``["c","b216"]``."""
    parts = node_id.split(":")
    return parts[2:]


def _is_flop_node(node_id: str) -> bool:
    """True when no chance-card token appears (i.e. still on the flop street)."""
    return not any(_CARD_RE.match(t) for t in _node_tokens(node_id))


def _decode_freq(blob: bytes) -> list[int]:
    return list(blob)


def _decode_ev(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 4}f", blob)


class _Solve:
    """Lazy reader over one ``.db`` (queries + small caches)."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.idx_to_hand: dict[int, str] = {}
        self.hand_to_idx: dict[str, int] = {}
        for idx, hand in con.execute("SELECT idx, hand FROM hand_index"):
            self.idx_to_hand[idx] = hand
            self.hand_to_idx[hand] = idx
        self.n = len(self.idx_to_hand)
        self.meta = dict(con.execute("SELECT key, value FROM metadata"))
        self._actions_cache: dict[str, list[_DBAction]] = {}

    def preflop_weights(self, player: str) -> dict[str, float]:
        rows = self.con.execute(
            "SELECT hand, weight FROM preflop_ranges WHERE player=?", (player,)
        )
        return {hand: float(w) for hand, w in rows}

    def actions(self, node_id: str) -> list[_DBAction]:
        cached = self._actions_cache.get(node_id)
        if cached is not None:
            return cached
        rows = self.con.execute(
            "SELECT action, freq_blob, ev_blob_oop, ev_blob_ip "
            "FROM gto_postflop WHERE node=?",
            (node_id,),
        ).fetchall()
        out = [
            _DBAction(a, _decode_freq(fb), _decode_ev(eo), _decode_ev(ei))
            for a, fb, eo, ei in rows
        ]
        self._actions_cache[node_id] = out
        return out


# --- the betting walk -------------------------------------------------------
@dataclass
class _BettingState:
    """The running game state as we walk a node string (flop-only for v1)."""

    pot_chips: float
    invested: dict[str, float]  # this-street chips committed, by side
    to_act: str  # side to act NEXT
    other: str

    def to_call(self) -> float:
        return max(self.invested.values()) - self.invested[self.to_act]


def _token_to_db_name(token: str, node_actions: list[_DBAction]) -> str:
    """Map a node-string token to the vendor ``action`` name at the PARENT node.

    ``b<X>`` -> ``BET_<X>``; ``f`` -> ``FOLD``; ``c`` -> the single passive action
    present (the vendor labels it ``CHECK`` or ``CALL`` interchangeably).
    """
    if token.startswith("b"):
        return f"BET_{token[1:]}"
    if token == "f":
        return "FOLD"
    if token == "c":
        passive = [a.db_name for a in node_actions if a.db_name in ("CHECK", "CALL")]
        if len(passive) != 1:
            raise ValueError(
                f"expected exactly one passive action, got {passive} at a parent node"
            )
        return passive[0]
    raise ValueError(f"unrecognised node token {token!r}")


class SqliteDbAdapter:
    """Builds a :class:`PostflopSolve` from one vendor ``.db`` file."""

    def __init__(
        self,
        db_path: str,
        *,
        oop_position: str = "BB",
        ip_position: str = "BTN",
        table_size: int = 9,
        game_format: str = "cash",
        live_or_online: str = "Live",
        stakes: str = "$1/$2",
        bb_in_dollars: float = 2.0,
    ):
        self.db_path = db_path
        self.oop = oop_position
        self.ip = ip_position
        self.table_size = table_size
        self.game_format = game_format
        self.live_or_online = live_or_online
        self.stakes = stakes
        self.bb_in_dollars = bb_in_dollars

    # -- geometry from metadata ------------------------------------------
    def _bb_chips(self, s: _Solve) -> float:
        """Chips per big blind. Derived from the verified pot identity, with a
        metadata cross-check (open size + eff-stack must agree on 100 chips/bb)."""
        eff = float(s.meta["eff_stack"])
        stack_bb = float(s.meta["stack_bb"])
        m = re.search(r"([\d.]+)\s*bb", s.meta.get("btn_open", ""))
        open_bb = float(m.group(1)) if m else 0.0
        if open_bb and stack_bb > open_bb:
            bb = eff / (stack_bb - open_bb)  # eff_stack = (stack - open) bb
        else:
            bb = 100.0
        return round(bb)

    def build(self, *, streets: tuple[str, ...] = ("flop",)) -> PostflopSolve:
        con = sqlite3.connect(self.db_path)
        try:
            s = _Solve(con)
            return self._build(s, streets=streets)
        finally:
            con.close()

    def _build(self, s: _Solve, *, streets: tuple[str, ...]) -> PostflopSolve:
        if streets != ("flop",):
            raise NotImplementedError(
                "v1 of the .db adapter builds flop nodes only; turn/river is the "
                "documented next step (chance-token street reset)."
            )
        bb = self._bb_chips(s)
        flop = tuple(_split_cards(s.meta["flop"]))
        board_set = set(flop)
        start_pot = float(s.meta["pot"])  # chips
        eff_flop = float(s.meta["eff_stack"])  # chips behind at the flop start
        # Open size for the preflop narrative.
        m = re.search(r"([\d.]+)\s*bb", s.meta.get("btn_open", ""))
        open_bb = float(m.group(1)) if m else 3.0

        # Reach bases: each side's preflop range, board-masked, idx-aligned.
        pre = {self.oop: s.preflop_weights("BB"), self.ip: s.preflop_weights("BTN")}
        base_reach = {
            side: [
                (
                    pre[side].get(s.idx_to_hand[i], 0.0)
                    if not (set(_split_cards(s.idx_to_hand[i])) & board_set)
                    else 0.0
                )
                for i in range(s.n)
            ]
            for side in (self.oop, self.ip)
        }

        # All flop decision nodes (drop terminal/fold-leaf strings: a node we
        # build must have >=1 action row).
        all_nodes = [r[0] for r in con_distinct_nodes(s.con)]
        flop_nodes = sorted(n for n in all_nodes if _is_flop_node(n))

        nodes: dict[str, PostflopNode] = {}
        for node_id in flop_nodes:
            node = self._build_node(
                s, node_id, bb=bb, start_pot=start_pot, eff_flop=eff_flop,
                base_reach=base_reach, flop=flop,
            )
            if node is not None:
                nodes[node_id] = node

        spot = s.meta.get("spot", "")
        return PostflopSolve(
            solve_id=f"{spot}_{''.join(flop)}",
            positions=(self.oop, self.ip),
            effective_stack_bb=float(s.meta.get("stack_bb", 100)),
            starting_pot_bb=start_pot / bb,
            flop=flop,  # type: ignore[arg-type]
            preflop_summary=(
                PreflopStep(self.ip, "open", to_bb=round(open_bb, 2)),
                PreflopStep(self.oop, "call"),
            ),
            nodes=nodes,
            game_format=self.game_format,
            bb_in_dollars=self.bb_in_dollars,
            stakes=self.stakes,
            live_or_online=self.live_or_online,
            table_size=self.table_size,
            source_reference=f"db/{spot}/{''.join(flop)}",
        )

    # -- one node --------------------------------------------------------
    def _build_node(
        self, s, node_id, *, bb, start_pot, eff_flop, base_reach, flop,
    ) -> PostflopNode | None:
        actions = s.actions(node_id)
        if not actions:
            return None

        # 1. Walk the node string: betting state + reach products to this node.
        state, reach, history = self._walk(
            s, node_id, bb=bb, start_pot=start_pot, base_reach=base_reach
        )
        actor, villain = state.to_act, state.other
        to_call_chips = state.to_call()
        pot_chips = state.pot_chips
        pot_before = pot_chips - to_call_chips  # pot before the bet hero faces
        eff_remaining = eff_flop - max(state.invested.values())

        # 2. Build the display actions (labels/verbs derived from betting state).
        node_actions, token_map = self._node_actions(
            actions, to_call_chips=to_call_chips, pot_before=pot_before,
            bb=bb, eff_remaining=eff_remaining,
        )

        actor_reach = reach[actor]
        denom = sum(actor_reach)
        if denom <= 0:
            return None  # actor's range never arrives here

        # 3. Per-combo strategy, combo EVs, and reach-weighted aggregates.
        ev_attr = "ev_oop" if actor == self.oop else "ev_ip"
        strategy: dict[str, dict[str, float]] = {}
        combo_evs: dict[str, dict[str, float]] = {}
        agg_freq = {na.label: 0.0 for na in node_actions}
        agg_ev = {na.label: 0.0 for na in node_actions}

        for i in range(s.n):
            r = actor_reach[i]
            if r <= _REACH_EPS:
                continue
            combo = s.idx_to_hand[i]
            total = sum(a.freq[i] for a in actions)
            if total <= 0:
                continue  # not in the actor's range at this node
            mix: dict[str, float] = {}
            evs: dict[str, float] = {}
            for a in actions:
                label = self._label_for(a, token_map)
                p = a.freq[i] / total
                if a.freq[i] > _FREQ_EPS_BYTES:
                    mix[label] = p
                evs[label] = getattr(a, ev_attr)[i] / bb
                agg_freq[label] += r * p
                agg_ev[label] += r * (getattr(a, ev_attr)[i] / bb)
            strategy[combo] = mix
            combo_evs[combo] = evs

        if not strategy:
            return None

        final_actions = tuple(
            NodeAction(
                label=na.label, verb=na.verb, freq=agg_freq[na.label] / denom,
                to_bb=na.to_bb, pot_fraction=na.pot_fraction,
                ev_bb=agg_ev[na.label] / denom,
            )
            for na in node_actions
        )

        hero_range = {
            s.idx_to_hand[i]: reach[actor][i]
            for i in range(s.n) if reach[actor][i] > _REACH_EPS
        }
        villain_range = {
            s.idx_to_hand[i]: reach[villain][i]
            for i in range(s.n) if reach[villain][i] > _REACH_EPS
        }

        # decision street's board (flop-only for v1).
        return PostflopNode(
            node_id=node_id,
            street="flop",
            board=flop,
            actor=actor,
            villain=villain,
            pot_bb=pot_chips / bb,
            effective_stack_bb=eff_remaining / bb,
            actions=final_actions,
            strategy=strategy,
            hero_range=hero_range,
            villain_range=villain_range,
            history=history,
            to_call_bb=to_call_chips / bb,
            combo_evs=combo_evs,
        )

    # -- the walk: betting state + reach + history -----------------------
    def _walk(self, s, node_id, *, bb, start_pot, base_reach):
        state = _BettingState(
            pot_chips=start_pot,
            invested={self.oop: 0.0, self.ip: 0.0},
            to_act=self.oop,
            other=self.ip,
        )
        reach = {self.oop: list(base_reach[self.oop]), self.ip: list(base_reach[self.ip])}
        history: list[PostflopStep] = []

        cur = "r:0"
        for token in _node_tokens(node_id):
            parent_actions = s.actions(cur)
            acting = state.to_act
            db_name = _token_to_db_name(token, parent_actions)
            pa = next((a for a in parent_actions if a.db_name == db_name), None)
            if pa is None:
                raise ValueError(f"{cur}: no action {db_name!r} for token {token!r}")

            # reach update: multiply the acting side's combos by this action's prob.
            for i in range(s.n):
                total = sum(a.freq[i] for a in parent_actions)
                p = (pa.freq[i] / total) if total > 0 else 0.0
                reach[acting][i] *= p

            # history step + betting-state update.
            to_call_now = state.to_call()
            if token.startswith("b"):
                size = float(token[1:])
                if to_call_now > 0:  # raise TO size
                    history.append(
                        PostflopStep("flop", acting, "raise", to_bb=round(size / bb, 2))
                    )
                else:  # bet OF size
                    history.append(
                        PostflopStep("flop", acting, "bet", to_bb=round(size / bb, 2))
                    )
                added = size - state.invested[acting]
                state.pot_chips += added
                state.invested[acting] = size
            elif token == "c":
                if to_call_now > 0:  # call
                    history.append(PostflopStep("flop", acting, "call"))
                    state.pot_chips += to_call_now
                    state.invested[acting] += to_call_now
                else:  # check
                    history.append(PostflopStep("flop", acting, "check"))
            elif token == "f":
                history.append(PostflopStep("flop", acting, "fold"))

            state.to_act, state.other = state.other, state.to_act
            cur = cur + ":" + token
        return state, reach, tuple(history)

    # -- action label/verb derivation ------------------------------------
    def _node_actions(self, actions, *, to_call_chips, pot_before, bb, eff_remaining):
        facing = to_call_chips > 0
        out = []
        token_map: dict[str, str] = {}  # db_name -> label
        for a in actions:
            label, verb, to_bb, pf = self._derive(
                a.db_name, facing=facing, pot_before=pot_before, bb=bb,
                to_call_chips=to_call_chips, eff_remaining=eff_remaining,
            )
            token_map[a.db_name] = label
            out.append(
                NodeAction(label=label, verb=verb, freq=0.0, to_bb=to_bb, pot_fraction=pf)
            )
        return out, token_map

    def _label_for(self, a: _DBAction, token_map: dict[str, str]) -> str:
        return token_map[a.db_name]

    def _derive(self, db_name, *, facing, pot_before, bb, to_call_chips, eff_remaining):
        """(label, verb, to_bb, pot_fraction) for one vendor action, by context."""
        if db_name == "FOLD":
            return "Fold", "fold", None, None
        if db_name in ("CHECK", "CALL"):
            if facing:
                return "Call", "call", round(to_call_chips / bb, 2), None
            return "Check", "check", None, None
        if db_name.startswith("BET_"):
            size = float(db_name[4:])
            to_bb = round(size / bb, 2)
            # all-in: the bet commits (about) the whole remaining stack.
            if size >= eff_remaining - 1:
                return "All-in", ("raise" if facing else "bet"), to_bb, None
            if facing:  # raise TO size
                return f"Raise to {to_bb:g}bb", "raise", to_bb, None
            pf = size / pot_before if pot_before > 0 else None
            label = f"Bet {round(pf * 100)}%" if pf is not None else f"Bet {to_bb:g}bb"
            return label, "bet", to_bb, pf
        raise ValueError(f"unrecognised vendor action {db_name!r}")


def con_distinct_nodes(con: sqlite3.Connection):
    return con.execute("SELECT DISTINCT node FROM gto_postflop")


def load_postflop_db(db_path: str, **kwargs) -> PostflopSolve:
    """Convenience: build a flop-only :class:`PostflopSolve` from a vendor ``.db``.

    Keyword args override display/position defaults (see :class:`SqliteDbAdapter`).
    """
    return SqliteDbAdapter(db_path, **kwargs).build()


__all__ = ["SqliteDbAdapter", "load_postflop_db"]
