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

Builds **flop, turn, and river** decision nodes. A chance-card token (``:2c``)
ends a street: the walk deals the card onto the board, resets the this-street
invested chips (the pot and the chips-behind carry over), and hands the action
to OOP (who acts first on every postflop street). Hole-card combos that contain
a dealt board card are removed from each side's reach. The ``streets`` arg to
:meth:`SqliteDbAdapter.build` selects which streets to materialise, and
``max_nodes_per_street`` deterministically down-samples the (very large) turn/
river node sets to keep build time + memory bounded for a batch.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass

from pipeline.bb_display import round_to_half_bb
from pipeline.postflop.solve import (
    STREETS,
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

# ``preflop_line`` metadata verb -> the IR's PreflopStep verb.
_PREFLOP_LINE_VERBS = {
    "open": "open", "raise": "raise", "call": "call", "fold": "fold",
    "check": "check", "limp": "call",
    "3bet": "3-bet", "3-bet": "3-bet",
    "4bet": "4-bet", "4-bet": "4-bet",
    "5bet": "5-bet", "5-bet": "5-bet",
}
# The IR verbs that put in a raise (mirrors facts._RAISE_VERBS, which counts
# them on a built solve; here we count on the parsed steps pre-build).
_RAISE_STEP_VERBS = ("open", "raise", "3-bet", "4-bet", "5-bet")
_SIZE_BB_RE = re.compile(r"([\d.]+)\s*bb", re.I)


def _parse_preflop_line(line: str) -> tuple[PreflopStep, ...] | None:
    """The ``preflop_line`` metadata as :class:`PreflopStep`\\ s, or None.

    The July-2026 exports state the exact preflop line, e.g.
    ``"BTN open 3bb, BB 3bet 17bb, BTN call"`` -> open / 3-bet / call steps.
    This is what makes a 3-BET POT solve render correctly (prose, pot-type
    column, aggressor, seat-token stack walk); the legacy files without the
    key are all SRP and fall back to the hardcoded open+call summary.
    Returns None when the line is absent or any step is unparseable (never
    guess a preflop line).
    """
    steps: list[PreflopStep] = []
    for part in (p.strip() for p in line.split(",") if p.strip()):
        tokens = part.split()
        if len(tokens) < 2:
            return None
        position, verb_token = tokens[0], tokens[1].lower()
        verb = _PREFLOP_LINE_VERBS.get(verb_token)
        if verb is None:
            return None
        m = _SIZE_BB_RE.search(part[len(position):])
        to_bb = float(m.group(1)) if m and verb in ("open", "raise", "3-bet", "4-bet", "5-bet") else None
        steps.append(PreflopStep(position, verb, to_bb=to_bb))
    return tuple(steps) if steps else None


@dataclass(frozen=True)
class _DBAction:
    """One raw action row at a node: the vendor token + its decoded blobs."""

    db_name: str  # "BET_216" | "CHECK" | "CALL" | "FOLD"
    freq: list[int]  # 1326 bytes, 0-255
    ev_oop: tuple[float, ...]  # 1326 float32, chips
    ev_ip: tuple[float, ...]


def _resolve_preflop_summary(
    meta: dict, *, ip: str, oop: str, open_bb: float, source: str = "",
) -> tuple[PreflopStep, ...]:
    """The solve's preflop summary from its metadata.

    The exact ``preflop_line`` when the file states it (3-bet pots NEED
    this: prose, the pot-type column, the aggressor, and the seat-token
    stack walk all derive from the summary). INVARIANT: a PRESENT but
    unparseable ``preflop_line`` must refuse loudly, never fall back -- the
    fallback says "open + call", and every SRP-only gate downstream trusts
    the summary, so silently guessing on a 3-bet-pot file would ship a
    wrong preflop story on every question. Only files WITHOUT the key (the
    legacy exports, all SRP) use the hardcoded open+call summary.
    """
    line = str(meta.get("preflop_line", "") or "")
    if line:
        parsed = _parse_preflop_line(line)
        if parsed is None:
            raise ValueError(
                f"unparseable preflop_line metadata {line!r} in {source or 'solve'};"
                " refusing to guess the preflop line"
            )
        return parsed
    return (
        PreflopStep(ip, "open", to_bb=round(open_bb, 2)),
        PreflopStep(oop, "call"),
    )


def _split_cards(s: str) -> list[str]:
    return [s[i : i + 2] for i in range(0, len(s), 2)]


def _node_tokens(node_id: str) -> list[str]:
    """Action tokens after the ``r:0`` root, e.g. ``r:0:c:b216`` -> ``["c","b216"]``."""
    parts = node_id.split(":")
    return parts[2:]


def _is_flop_node(node_id: str) -> bool:
    """True when no chance-card token appears (i.e. still on the flop street)."""
    return not any(_CARD_RE.match(t) for t in _node_tokens(node_id))


def _node_street(node_id: str) -> str:
    """The street a node lives on, from its count of chance-card tokens
    (0 -> flop, 1 -> turn, 2+ -> river)."""
    n_cards = sum(1 for t in _node_tokens(node_id) if _CARD_RE.match(t))
    return STREETS[min(n_cards, 2)]


def _stride_sample(items: list[str], k: int) -> list[str]:
    """``k`` evenly-spaced items from a sorted list (deterministic, order-preserving).

    Used to cap the turn/river node sets (thousands -> tens of thousands) to a
    representative, byte-stable subset for a batch. Returns all items when
    ``len(items) <= k``."""
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


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
    """The running game state as we walk a node string across streets."""

    pot_chips: float
    invested: dict[str, float]  # THIS-street chips committed, resets each street
    behind: dict[str, float]  # chips remaining behind, carries across streets
    to_act: str  # side to act NEXT
    other: str
    street_idx: int  # 0=flop, 1=turn, 2=river
    board: list[str]  # community cards revealed so far (grows on chance tokens)

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
        """Chips per big blind.

        Preferred: the exact chips/bb ratio the file itself states (the
        July-2026 exports carry ``pot``/``pot_bb`` and ``eff_stack``/
        ``eff_stack_bb`` pairs). Legacy fallback: the SRP open-size identity
        ``eff_stack = (stack - open) bb`` from ``btn_open`` (v8) -- SRP-ONLY,
        a 3-bet pot's eff_stack subtracts the 3-bet, not the open, so any
        multi-raise file MUST ship the bb-pair keys -- else 100."""
        for chips_key, bb_key in (("pot", "pot_bb"), ("eff_stack", "eff_stack_bb")):
            chips = _safe_float(s.meta.get(chips_key))
            in_bb = _safe_float(s.meta.get(bb_key))
            if chips and in_bb:
                return round(chips / in_bb)
        eff = float(s.meta["eff_stack"])
        stack_bb = float(s.meta["stack_bb"])
        m = re.search(r"([\d.]+)\s*bb", s.meta.get("btn_open", ""))
        open_bb = float(m.group(1)) if m else 0.0
        if open_bb and stack_bb > open_bb:
            bb = eff / (stack_bb - open_bb)  # eff_stack = (stack - open) bb
        else:
            bb = 100.0
        return round(bb)

    def build(
        self,
        *,
        streets: tuple[str, ...] = ("flop",),
        max_nodes_per_street: int | None = None,
        include_ancestors: bool = False,
    ) -> PostflopSolve:
        con = sqlite3.connect(self.db_path)
        try:
            s = _Solve(con)
            return self._build(
                s, streets=streets, max_nodes_per_street=max_nodes_per_street,
                include_ancestors=include_ancestors,
            )
        finally:
            con.close()

    def _build(
        self,
        s: _Solve,
        *,
        streets: tuple[str, ...],
        max_nodes_per_street: int | None,
        include_ancestors: bool = False,
    ) -> PostflopSolve:
        requested = tuple(streets)
        bad = [st for st in requested if st not in STREETS]
        if bad:
            raise ValueError(f"unknown street(s) {bad}; expected a subset of {STREETS}")
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

        # Decision nodes for the requested streets. The turn/river node sets are
        # huge (thousands -> tens of thousands), so each street is deterministically
        # down-sampled to ``max_nodes_per_street`` -- enough variety for a batch
        # without building every node. (A node we build must have >=1 action row.)
        all_nodes = [r[0] for r in con_distinct_nodes(s.con)]
        by_street: dict[str, list[str]] = {st: [] for st in STREETS}
        for n in all_nodes:
            st = _node_street(n)
            if st in requested:
                by_street[st].append(n)
        selected: list[str] = []
        for st in STREETS:  # deterministic street order
            ranked = sorted(by_street[st])
            if max_nodes_per_street is not None:
                ranked = _stride_sample(ranked, max_nodes_per_street)
            selected.extend(ranked)

        # Line-closure for play-through assembly: a down-sampled deep node's
        # decision-node ANCESTORS may not be in `selected`, which breaks
        # reconstructing the connected hand that leads to it. When requested, add
        # every decision-node prefix of every selected node (cheap: ~6-10 extra
        # per deep node, all sharing cached action reads). Restricted to the
        # requested streets so it never pulls in a street the caller excluded.
        if include_ancestors:
            extra: set[str] = set()
            for nid in selected:
                parts = nid.split(":")
                for i in range(1, len(parts)):  # 'r:0' .. parent (exclude self)
                    pref = ":".join(parts[:i + 1])
                    if pref == nid or pref in extra:
                        continue
                    if _node_street(pref) not in requested:
                        continue
                    if s.actions(pref):  # a real decision node (has action rows)
                        extra.add(pref)
            present = set(selected)
            selected.extend(sorted(p for p in extra if p not in present))

        nodes: dict[str, PostflopNode] = {}
        for node_id in selected:
            node = self._build_node(
                s, node_id, bb=bb, start_pot=start_pot, eff_flop=eff_flop,
                base_reach=base_reach, flop=flop,
            )
            if node is not None:
                nodes[node_id] = node

        preflop_summary = _resolve_preflop_summary(
            s.meta, ip=self.ip, oop=self.oop, open_bb=open_bb,
            source=self.db_path,
        )

        spot = s.meta.get("spot", "")
        return PostflopSolve(
            solve_id=f"{spot}_{''.join(flop)}",
            positions=(self.oop, self.ip),
            effective_stack_bb=float(s.meta.get("stack_bb", 100)),
            starting_pot_bb=start_pot / bb,
            flop=flop,  # type: ignore[arg-type]
            preflop_summary=preflop_summary,
            nodes=nodes,
            game_format=self.game_format,
            bb_in_dollars=self.bb_in_dollars,
            stakes=self.stakes,
            live_or_online=self.live_or_online,
            table_size=self.table_size,
            rake=str(s.meta.get("rake", "") or ""),
            source_reference=f"db/{spot}/{''.join(flop)}",
            # RAW (un-board-masked) preflop-entry weights per seat -- the source
            # for preflop-entry questions / the preflop leg of a play-through.
            # The opener's weights are ~1.0 (opens always); the caller's are a
            # genuine call-vs-fold mix (e.g. 22 at 0.83 = "calls ~83% vs the open").
            preflop_entry_ranges={
                self.oop: dict(pre[self.oop]),
                self.ip: dict(pre[self.ip]),
            },
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
            s, node_id, bb=bb, start_pot=start_pot, eff_flop=eff_flop,
            base_reach=base_reach, flop=flop,
        )
        actor, villain = state.to_act, state.other
        to_call_chips = state.to_call()
        pot_chips = state.pot_chips
        pot_before = pot_chips - to_call_chips  # pot before the bet hero faces
        eff_remaining = min(state.behind.values())  # shorter stack, across streets

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
        # A node where the villain's range never arrives (deep, ~never-played
        # reraise lines whose reach products underflow) has no opponent to
        # measure equity against -- skip it rather than emit a 0%-equity spot.
        if not villain_range:
            return None

        # decision street + the board revealed at it (3/4/5 cards).
        return PostflopNode(
            node_id=node_id,
            street=STREETS[state.street_idx],
            board=tuple(state.board),
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
    def _walk(self, s, node_id, *, bb, start_pot, eff_flop, base_reach, flop):
        state = _BettingState(
            pot_chips=start_pot,
            invested={self.oop: 0.0, self.ip: 0.0},
            behind={self.oop: eff_flop, self.ip: eff_flop},
            to_act=self.oop,
            other=self.ip,
            street_idx=0,
            board=list(flop),
        )
        reach = {self.oop: list(base_reach[self.oop]), self.ip: list(base_reach[self.ip])}
        history: list[PostflopStep] = []

        cur = "r:0"
        for token in _node_tokens(node_id):
            # A chance-card token ends the street: deal the card, reset the
            # this-street betting, hand action to OOP, and drop combos that
            # contain the dealt card. The pot and chips-behind carry over.
            if _CARD_RE.match(token):
                state.board.append(token)
                state.invested = {self.oop: 0.0, self.ip: 0.0}
                state.to_act, state.other = self.oop, self.ip
                state.street_idx += 1
                for side in (self.oop, self.ip):
                    side_reach = reach[side]
                    for i in range(s.n):
                        if side_reach[i] and token in _split_cards(s.idx_to_hand[i]):
                            side_reach[i] = 0.0
                cur = cur + ":" + token
                continue

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

            # history step + betting-state update (street-labelled).
            street_name = STREETS[state.street_idx]
            to_call_now = state.to_call()
            if token.startswith("b"):
                size = float(token[1:])
                verb = "raise" if to_call_now > 0 else "bet"  # raise TO / bet OF size
                added_full = size - state.invested[acting]
                # The vendor encodes an all-in as "bet your FLOP-START stack"
                # (the eff_stack token), which OVERSTATES the wager by whatever
                # the actor already put in on earlier streets (e.g. a turn call).
                # Cap the wager at what the actor actually has behind, so the
                # all-in size, the opponent's to-call, the pot, and the pot-odds
                # are all exact (a deep river jam was reading ~2bb too big and the
                # to-call exceeded the caller's stack). All-in == the token wants
                # at least the whole remaining stack.
                all_in = added_full >= state.behind[acting] - 1.0
                added = state.behind[acting] if all_in else added_full
                actual_to = state.invested[acting] + added  # true total this street
                history.append(
                    PostflopStep(
                        street_name, acting, verb, to_bb=round(actual_to / bb, 2),
                        all_in=all_in,
                    )
                )
                state.pot_chips += added
                state.invested[acting] = actual_to
                state.behind[acting] -= added
            elif token == "c":
                if to_call_now > 0:  # call
                    history.append(PostflopStep(street_name, acting, "call"))
                    state.pot_chips += to_call_now
                    state.invested[acting] += to_call_now
                    state.behind[acting] -= to_call_now
                else:  # check
                    history.append(PostflopStep(street_name, acting, "check"))
            elif token == "f":
                history.append(PostflopStep(street_name, acting, "fold"))

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
            # Label bb amount snapped to the 0.5bb display grid (display-only;
            # to_bb field + pot_fraction stay exact for the math).
            disp = round_to_half_bb(to_bb)
            if facing:  # raise TO size
                return f"Raise to {disp:g}bb", "raise", to_bb, None
            pf = size / pot_before if pot_before > 0 else None
            label = f"Bet {round(pf * 100)}%" if pf is not None else f"Bet {disp:g}bb"
            return label, "bet", to_bb, pf
        raise ValueError(f"unrecognised vendor action {db_name!r}")


def con_distinct_nodes(con: sqlite3.Connection):
    return con.execute("SELECT DISTINCT node FROM gto_postflop")


# --- solve discovery + self-describing metadata -----------------------------
# Each ``.db`` carries a ``metadata`` table that fully describes its scenario
# (table size, positions, stack, flop, rake, format). The library below reads
# it so a solve describes ITSELF -- the admin picker lists solves by their own
# metadata, and ``load_postflop_db`` derives the scenario from the file instead
# of relying on caller-passed defaults (which silently mislabel, e.g. a 6-max
# solve as the hardcoded 9-max). Explicit kwargs still override.

# Longest-first so "UTG+2" matches before "UTG".
_POSITIONS = ("UTG+2", "UTG+1", "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB")


def read_db_metadata(db_path: str) -> dict[str, str]:
    """Read just the ``metadata`` table (fast: no node walk). Keys -> values."""
    con = sqlite3.connect(db_path)
    try:
        return {str(k): str(v) for k, v in con.execute("SELECT key, value FROM metadata")}
    finally:
        con.close()


def _safe_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pot_type_label(meta: dict[str, str]) -> str:
    """The pot-type display label ("SRP" / "3-bet pot" / "4-bet pot"), from the
    ``preflop_line`` metadata, falling back to the spot name. "" when unknown.

    Shown in the picker label so two solves of the SAME flop at different pot
    types (e.g. the Kd7s3s SRP and 3-bet pot) stay distinguishable."""
    line = meta.get("preflop_line", "")
    if line:
        parsed = _parse_preflop_line(line)
        if parsed:
            raises = sum(
                1 for st in parsed
                if st.verb in ("open", "raise", "3-bet", "4-bet", "5-bet")
            )
            return {0: "limped", 1: "SRP", 2: "3-bet pot", 3: "4-bet pot"}.get(
                raises, f"{raises}-raise pot"
            )
    spot = meta.get("spot", "")
    for token, label in (("4BP", "4-bet pot"), ("3BP", "3-bet pot"), ("SRP", "SRP")):
        if token in spot:
            return label
    return ""


def _first_position(text: str) -> str | None:
    """The leading position token of ``text`` (e.g. a range filename / spot).

    Takes the first ``_``- or space-delimited token -- ``"BTN_open_81pot.txt"``
    -> ``"BTN"``, ``"BB_call_vs_BTN..."`` -> ``"BB"`` -- and returns it only if
    it's a known position. (A plain ``\\b`` fails here: ``_`` is a word char, so
    ``BTN\\b`` doesn't match ``BTN_open``.)"""
    token = text.strip().split("_", 1)[0].split()[0] if text.strip() else ""
    return token if token in _POSITIONS else None


def derive_scenario(meta: dict[str, str]) -> dict[str, object]:
    """Adapter kwargs implied by the ``.db`` metadata (table size, positions,
    format). Best-effort: only keys it can confidently parse are returned, so a
    caller's explicit overrides win and an unparseable field falls back to the
    adapter default."""
    out: dict[str, object] = {}
    blob = f"{meta.get('game_format', '')} {meta.get('spot', '')}"
    m = re.search(r"(\d+)\s*max", blob, re.I)
    if m:
        out["table_size"] = int(m.group(1))
    # Positions: the range filenames are the most reliable source
    # ("BTN_open_..." / "BB_call_vs_BTN_..."); fall back to the spot string.
    ip = _first_position(meta.get("ip_range", "")) or _first_position(meta.get("spot", ""))
    oop = _first_position(meta.get("oop_range", ""))
    if ip:
        out["ip_position"] = ip
    if oop:
        out["oop_position"] = oop
    ante = (meta.get("ante") or "").strip().lower()
    fmt = meta.get("game_format", "").lower()
    is_tourney = ante not in ("", "none", "0", "0%", "-") or any(
        k in fmt for k in ("mtt", "icm", "tournament", "tourney")
    )
    out["game_format"] = "tournament" if is_tourney else "cash"
    return out


@dataclass(frozen=True)
class DbSolveSummary:
    """A one-glance description of a ``.db`` solve, read from its metadata.

    Drives the admin solve picker (each solve self-describes); ``ok=False`` with
    an ``error`` when the file isn't a readable solve (so the picker can show it
    greyed rather than crash the scan)."""

    path: str
    ok: bool
    spot: str = ""
    flop: str = ""  # raw, e.g. "QsJd9s"
    flop_cards: tuple[str, ...] = ()
    table_size: int | None = None
    stack_bb: float | None = None
    oop_position: str = "BB"
    ip_position: str = "BTN"
    game_format: str = "cash"
    rake: str = ""
    ante: str = ""  # raw metadata value, e.g. "0" / "12.5%" / "1bb"
    # Pot-type label ("SRP" / "3-bet pot" / ...), "" when the file doesn't say.
    pot_type: str = ""
    customer: str = ""
    solve_date: str = ""
    label: str = ""  # compact one-liner for the picker
    error: str = ""

    @property
    def flop_pretty(self) -> str:
        return " ".join(self.flop_cards) if self.flop_cards else self.flop

    @property
    def rake_pretty(self) -> str:
        """The rake structure for display, e.g. "8% cap 2bb" or "none"."""
        r = (self.rake or "").strip()
        return r if r and r.lower() != "none" else "none"

    @property
    def ante_pretty(self) -> str:
        """The ante for display: "none" for 0/absent, else the raw value."""
        a = (self.ante or "").strip()
        return "none" if a.lower() in ("", "0", "0%", "none", "-") else a


def summarize_db(db_path: str) -> DbSolveSummary:
    """Read one ``.db``'s metadata into a :class:`DbSolveSummary` (no node walk)."""
    path = str(db_path)
    try:
        meta = read_db_metadata(path)
    except Exception as exc:  # noqa: BLE001 -- any unreadable file -> greyed entry
        return DbSolveSummary(path=path, ok=False, error=str(exc))
    if "flop" not in meta or "spot" not in meta:
        return DbSolveSummary(
            path=path, ok=False, error="missing 'flop'/'spot' metadata (not a postflop solve?)"
        )
    sc = derive_scenario(meta)
    flop_raw = meta.get("flop", "")
    flop_cards = tuple(_split_cards(flop_raw))
    ts = sc.get("table_size")
    stack = _safe_float(meta.get("stack_bb"))
    ip = str(sc.get("ip_position", "BTN"))
    oop = str(sc.get("oop_position", "BB"))
    fmt = str(sc.get("game_format", "cash"))
    rake = meta.get("rake", "") or ""
    rake_short = rake.split(" cap")[0].strip() if rake and rake.lower() != "none" else ""
    ante = (meta.get("ante", "") or "").strip()
    has_ante = ante.lower() not in ("", "0", "0%", "none", "-")
    pot_type = _pot_type_label(meta)
    bits = [f"{ip} vs {oop}"]
    if pot_type:
        bits.append(pot_type)
    if ts:
        bits.append(f"{ts}-max")
    if stack:
        bits.append(f"{stack:g}bb")
    bits.append(" ".join(flop_cards) if flop_cards else flop_raw)
    if rake_short:
        bits.append(f"{rake_short} rake")
    # Only surface the ante in the one-liner when there IS one (tournaments) --
    # cash solves stay clean. The full structure always shows in the picker details.
    if has_ante:
        bits.append(f"{ante} ante")
    return DbSolveSummary(
        path=path,
        ok=True,
        spot=meta.get("spot", ""),
        flop=flop_raw,
        flop_cards=flop_cards,
        table_size=ts if ts is None else int(ts),
        stack_bb=stack,
        oop_position=oop,
        ip_position=ip,
        game_format=fmt,
        rake=rake,
        ante=ante,
        pot_type=pot_type,
        customer=meta.get("customer_id", ""),
        solve_date=meta.get("solve_date", ""),
        label=" · ".join(bits),
    )


def discover_db_solves(directory: str) -> list[DbSolveSummary]:
    """Every ``.db`` under ``directory`` (recursive), summarised, sorted by path.

    Returns ``[]`` when the directory is absent. Unreadable files come back as
    ``ok=False`` summaries rather than being dropped, so the picker can explain
    why a file isn't usable."""
    from pathlib import Path  # noqa: PLC0415 -- local: module stays import-light

    d = Path(directory).expanduser()
    if not d.is_dir():
        return []
    return [summarize_db(str(p)) for p in sorted(d.rglob("*.db"))]


# Default per-street node cap for a batch load. Flop is tiny (~25 nodes, so the
# cap is a no-op there); turn (~2k) and river (~130k) are down-sampled to this
# many representative nodes. Plenty of variety for a batch; keeps build bounded.
DEFAULT_MAX_NODES_PER_STREET = 600


def load_postflop_db(
    db_path: str,
    *,
    streets: tuple[str, ...] = ("flop",),
    max_nodes_per_street: int | None = DEFAULT_MAX_NODES_PER_STREET,
    include_ancestors: bool = False,
    **overrides: object,
) -> PostflopSolve:
    """Build a :class:`PostflopSolve` from a vendor ``.db``.

    The scenario (table size, positions, cash/tournament) is derived from the
    file's own metadata via :func:`derive_scenario`; ``**overrides`` win over
    the derived values (used for display-only fields the metadata doesn't carry,
    e.g. ``stakes`` / ``live_or_online`` / ``bb_in_dollars``).

    ``streets`` selects which streets to materialise (default flop-only, for
    backward compatibility). ``max_nodes_per_street`` caps each street's node
    count -- the turn/river sets are huge, so they're deterministically
    down-sampled; pass ``None`` to build every node (use only for flop/small
    solves, or you may exhaust memory on a river-heavy solve).
    ``include_ancestors`` additionally materialises every selected node's
    decision-node ancestors, so a down-sampled deep node's connected line is
    fully present (needed by the play-through assembler).
    """
    derived = derive_scenario(read_db_metadata(db_path))
    return SqliteDbAdapter(db_path, **{**derived, **overrides}).build(  # type: ignore[arg-type]
        streets=streets, max_nodes_per_street=max_nodes_per_street,
        include_ancestors=include_ancestors,
    )


__all__ = [
    "DbSolveSummary",
    "SqliteDbAdapter",
    "derive_scenario",
    "discover_db_solves",
    "load_postflop_db",
    "read_db_metadata",
    "summarize_db",
]
