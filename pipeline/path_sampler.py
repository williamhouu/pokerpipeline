"""Layer 3: Path Sampler.

Walks a solved PioSolver tree and enumerates the individual decision spots
inside it -- the input Layer 4 (Question Extractor) filters and Layer 5 (Fact
Extractor) turns into data blocks.

Per docs/engineering_brief.docx, "Layer 3: Path Sampler", each decision spot
carries: the action sequence leading to it, the street and board, the pot and
effective stack (computed from the action history), the players' ranges, and
the available actions with their solver frequencies / EVs. Every spot is also
tagged with parent_node_id + action_to_reach so full-hand replay can be added
later without a schema change.

Usage:

    from pipeline.piosolver import PioSolverClient, find_piosolver
    with PioSolverClient(find_piosolver()) as client:
        client.load_tree("solve.cfr")
        sampler = PathSampler(client, oop_position="BB", ip_position="BTN")
        for node in sampler.enumerate_decision_nodes():
            ctx = sampler.build_spot_context(node)   # heavy: ranges, strategy, EVs

`enumerate_decision_nodes` is cheap (tree metadata only); `build_spot_context`
fetches the per-spot solver data. Splitting them keeps enumeration fast so
Layer 4 can filter before the expensive range/EV reads happen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pipeline.piosolver import HAND_COUNT, PioSolverClient

_RANKS = "AKQJT98765432"
_SUITS = "shdc"
_IN_RANGE_EPS = 1e-4               # range weight at or below this == not in range


# --- pure helpers (no solver needed) -----------------------------------------
def _is_card(token: str) -> bool:
    """Whether a node-id segment is a board card (e.g. 'Kh') rather than an action."""
    return len(token) == 2 and token[0] in _RANKS and token[1] in _SUITS


def total_pot(pot_field: str) -> float:
    """Total pot from PioSolver's 3-token pot field ('36 0 55' -> 91).

    The tokens are (OOP invested, IP invested, pot carried from prior streets);
    their sum is the chips in play at the node regardless of whether a bet is
    still uncalled or already collected.
    """
    return float(sum(float(t) for t in pot_field.split()))


def effective_stack(pot_field: str, starting_pot: float,
                    starting_stack: float) -> float:
    """Effective stack remaining at a node, from its pot field.

    Each chip in the carry beyond `starting_pot` was matched, so it splits
    evenly; the player who is in for more (max of the two uncalled amounts) has
    that much less behind. Clamped at 0 (an all-in node).
    """
    oop, ip, carry = (float(t) for t in pot_field.split())
    called = carry - starting_pot
    remaining = starting_stack - called / 2.0 - max(oop, ip)
    return max(0.0, remaining)


def amount_to_call(pot_field: str, hero_is_oop: bool) -> float:
    """Chips hero must put in to call -- the unmatched portion of the pot field."""
    oop, ip, _ = (float(t) for t in pot_field.split())
    return max(0.0, (ip - oop) if hero_is_oop else (oop - ip))


def street_of(board: list[str]) -> str:
    """The street implied by the board: 3 cards = flop, 4 = turn, 5 = river."""
    return {3: "flop", 4: "turn", 5: "river"}.get(len(board), "preflop")


def label_action(token: str, bet_pending: bool) -> tuple[str, bool]:
    """Semantic label for an action token, plus the new bet-pending state.

    `bet_pending` is whether the actor faces an unanswered bet, which decides
    check-vs-call and bet-vs-raise. PioSolver writes every bet or raise as a
    `b<amount>` token; the amount is the cumulative street wager.
    """
    if token == "f":
        return "fold", bet_pending
    if token == "c":
        return ("call" if bet_pending else "check"), False
    if token[:1] == "b" and token[1:].replace(".", "", 1).isdigit():
        amount = token[1:]
        return (f"raise {amount}" if bet_pending else f"bet {amount}"), True
    return token, bet_pending          # unrecognised -- pass through


def _sample_evenly(items: list, limit: int) -> list:
    """Up to `limit` items, evenly spaced across the list (representative sample)."""
    if limit <= 0 or len(items) <= limit:
        return list(items)
    return [items[(i * len(items)) // limit] for i in range(limit)]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Range-weighted mean over in-range, finite values (Pio uses nan elsewhere)."""
    num = den = 0.0
    for value, weight in zip(values, weights):
        if weight > _IN_RANGE_EPS and math.isfinite(value):
            num += value * weight
            den += weight
    return num / den if den else 0.0


# --- the data the sampler produces -------------------------------------------
@dataclass
class DecisionNode:
    """A decision spot located by the tree walk -- metadata only, no ranges."""

    node_id: str                                   # PioSolver node id, e.g. "r:0:b36"
    node_type: str                                 # "OOP_DEC" or "IP_DEC"
    street: str                                    # flop / turn / river
    board: list[str]
    pot: float
    effective_stack: float
    amount_to_call: float                          # chips hero must call (0 if no bet faced)
    hero_position: str                             # whoever acts at this node
    villain_position: str
    hero_is_oop: bool
    parent_node_id: str                            # immediate tree parent
    action_to_reach: str                           # action on the edge from the parent
    action_sequence: list[tuple[str, str]]         # (actor, label); actor OOP/IP/deal
    available_actions: list[str]                   # labels of the offered actions


@dataclass
class ActionOption:
    """One available action at a spot, with its solver frequency and EV."""

    label: str
    node_id: str                                   # child node the action leads to
    frequency: float                               # range-weighted aggregate strategy
    ev: float                                      # range-weighted aggregate EV


@dataclass
class SpotContext:
    """A decision spot with its full solver context -- the Layer 3 output."""

    node: DecisionNode
    hero_range: dict[str, float]                   # combo -> weight, in-range only
    villain_range: dict[str, float]                # villain's continuing range
    actions: list[ActionOption] = field(default_factory=list)


# --- the sampler -------------------------------------------------------------
class PathSampler:
    """Enumerates decision spots in a loaded PioSolver tree."""

    def __init__(self, client: PioSolverClient,
                 oop_position: str = "OOP", ip_position: str = "IP",
                 starting_pot: float | None = None,
                 starting_stack: float | None = None):
        self.client = client
        self.oop_position = oop_position
        self.ip_position = ip_position
        root = client.show_node("r")
        self._starting_pot = (total_pot(root["pot"]) if starting_pot is None
                              else starting_pot)
        self._starting_stack = (float(client.show_effective_stack() or 0.0)
                                if starting_stack is None else starting_stack)
        self._hand_order: list[str] | None = None     # cached on first heavy read

    # -- enumeration (cheap: tree metadata only) ------------------------------
    def enumerate_decision_nodes(self, max_chance_children: int = 6,
                                 max_nodes: int = 20000):
        """Yield every decision spot reachable from the root.

        Decision nodes (OOP_DEC / IP_DEC) are yielded; ROOT and chance
        (SPLIT_NODE) nodes are walked through; END_NODE terminals stop a line.

        Chance nodes deal 48-49 cards each; `max_chance_children` samples an
        even spread of those runouts (the brief's "representative board set")
        so a flop solve enumerates in the hundreds rather than the millions.
        `max_nodes` is a hard safety cap on the walk.
        """
        stack: list[tuple[str, list, bool]] = [("r", [], False)]
        visited = 0
        while stack and visited < max_nodes:
            node_id, sequence, bet_pending = stack.pop()
            visited += 1
            info = self.client.show_node(node_id)
            node_type = info.get("node_type", "")
            children = self.client.show_children(node_id)

            if "DEC" in node_type:
                yield self._decision_node(node_id, node_type, info,
                                          sequence, bet_pending, children)
                side = "OOP" if node_type.startswith("OOP") else "IP"
                for child in children:
                    label, pend = label_action(_segment(child), bet_pending)
                    stack.append((child, sequence + [(side, label)], pend))
            elif "SPLIT" in node_type:
                for child in _sample_evenly(children, max_chance_children):
                    stack.append((child, sequence + [("deal", _segment(child))], False))
            elif "ROOT" in node_type:
                for child in children:            # structural edge, not an action
                    stack.append((child, sequence, bet_pending))
            # END_NODE: terminal, nothing to do

    def _decision_node(self, node_id, node_type, info, sequence,
                       bet_pending, children) -> DecisionNode:
        board = info.get("board", "").split()
        is_oop = node_type.startswith("OOP")
        return DecisionNode(
            node_id=node_id,
            node_type=node_type,
            street=street_of(board),
            board=board,
            pot=total_pot(info["pot"]),
            effective_stack=effective_stack(info["pot"], self._starting_pot,
                                            self._starting_stack),
            amount_to_call=amount_to_call(info["pot"], is_oop),
            hero_position=self.oop_position if is_oop else self.ip_position,
            villain_position=self.ip_position if is_oop else self.oop_position,
            hero_is_oop=is_oop,
            parent_node_id=node_id.rsplit(":", 1)[0] if ":" in node_id else "",
            action_to_reach=sequence[-1][1] if sequence else "",
            action_sequence=list(sequence),
            available_actions=[label_action(_segment(c), bet_pending)[0]
                               for c in children],
        )

    # -- spot context (heavy: ranges, strategy, EVs) --------------------------
    def build_spot_context(self, node: DecisionNode) -> SpotContext:
        """Fetch the full solver context for one decision spot.

        Reads both players' ranges at the node, the strategy, and the EV of
        each action -- so this is far heavier than enumeration. Call it only on
        the spots that survive Layer 4's filters.
        """
        if self._hand_order is None:
            self._hand_order = self.client.show_hand_order()
        hero_side = "OOP" if node.hero_is_oop else "IP"
        villain_side = "IP" if node.hero_is_oop else "OOP"

        hero_weights = self.client.show_range(hero_side, node.node_id)
        villain_weights = self.client.show_range(villain_side, node.node_id)
        strategy = self.client.show_strategy(node.node_id)
        children = self.client.show_children(node.node_id)
        hero_total = sum(hero_weights)

        actions: list[ActionOption] = []
        for index, child in enumerate(children):
            label = (node.available_actions[index]
                     if index < len(node.available_actions)
                     else label_action(_segment(child), False)[0])
            if index < len(strategy) and hero_total > 0:
                frequency = sum(strategy[index][i] * hero_weights[i]
                                for i in range(HAND_COUNT)) / hero_total
            else:
                frequency = 0.0
            ev_row = self.client.calc_ev(hero_side, child)["ev"]
            actions.append(ActionOption(label=label, node_id=child,
                                        frequency=frequency,
                                        ev=_weighted_mean(ev_row, hero_weights)))

        return SpotContext(
            node=node,
            hero_range=self._range_dict(hero_weights),
            villain_range=self._range_dict(villain_weights),
            actions=actions,
        )

    def _range_dict(self, weights: list[float]) -> dict[str, float]:
        assert self._hand_order is not None
        return {self._hand_order[i]: weights[i]
                for i in range(HAND_COUNT) if weights[i] > _IN_RANGE_EPS}


def _segment(node_id: str) -> str:
    """The last node-id segment -- the action token or card of the edge into it."""
    return node_id.rsplit(":", 1)[-1]
