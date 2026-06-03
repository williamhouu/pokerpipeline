"""PLO preflop spot sampling.

A :class:`PloSpot` pairs a :class:`~pipeline.plo.node_enumerator.PloDecisionNode`
with a specific hero hand (one of the 16,432 suit-isomorphic PLO starting
hands, identified by its `.rng` index) and carries that hand's strategy at the
node -- everything Layer 6 needs to write an explanation and everything the
worthiness filter needs to grade the spot.

This mirrors :mod:`pipeline.preflop.spot_sampler`. Two PLO-specific points:

1. **Normalisation is the same as NLHE.** Monker's `.rng` weights are the
   JOINT probability (hand reaches this node AND takes the action); a hand
   that opens 0.5% of the time reaches a facing-3-bet node with presence
   0.005. So :func:`sample_plo_spot` sums the sibling weights to get the
   presence, then divides to get the CONDITIONAL strategy at the decision --
   exactly as the NLHE sampler does for Pio weights.

2. **EV comes for free.** Monker stores per-action EV for *every* hand at
   every node, including the actions it never takes (the counterfactual
   value). So a spot carries a real ``ev_by_action`` and a real
   ``ev_gap_sb`` -- the EV cost of the best alternative -- for *every* spot,
   including raise spots (which the NLHE EV engine cannot score without a
   postflop solve).

Typical use::

    from pipeline.plo.pack import discover_plo_pack
    from pipeline.plo.node_enumerator import enumerate_plo_nodes
    from pipeline.plo.spot_sampler import sample_plo_spot

    pack = discover_plo_pack(Path("plo_ranges"))
    node = enumerate_plo_nodes(pack)[0]
    spot = sample_plo_spot(node, hero_index=0)  # AAAA
    print(spot.action_frequencies, spot.dominant_action, spot.ev_gap_sb)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pipeline.plo.hand_order import HAND_COUNT, cards_at, monker_label
from pipeline.plo.node_enumerator import PloDecisionNode
from pipeline.plo.pack import read_rng_values

_MIN_ACTIONS_FOR_GAP = 2


@dataclass(frozen=True)
class PloSpot:
    """One PLO preflop decision spot, ready for fact extraction.

    Fields:
        node: The decision node this spot is sampled from.
        hero_index: The hero hand's `.rng` index (0..16431).
        hero_label: The Monker hand string, e.g. ``'(AK)(AK)'``.
        hero_cards: A representative concrete 4-card hand, e.g.
            ``('Ac', 'Kc', 'Ad', 'Kd')`` -- the visible cards for the
            ``User Cards`` column and the equity engine.
        action_frequencies: CONDITIONAL strategy at this decision -- action
            ``label`` -> hero's frequency given the hand reaches the node.
            Sums to 1.0 (modulo rounding) when ``presence > 0``; all zeros
            when ``presence == 0`` (the hand never reaches here).
        ev_by_action: action ``label`` -> the hero hand's EV (in small
            blinds) for taking that action, straight from the solver. Defined
            for *every* action, including ones the hero never takes.
        presence: How often the hero hand reaches this node, in [0, 1] -- the
            sum of the raw (joint) weights before normalisation. The signal
            the worthiness filter uses to drop hands that were almost always
            folded earlier in the tree.
    """

    node: PloDecisionNode
    hero_index: int
    hero_label: str
    hero_cards: tuple[str, ...]
    action_frequencies: dict[str, float] = field(default_factory=dict)
    ev_by_action: dict[str, float] = field(default_factory=dict)
    presence: float = 0.0

    @property
    def dominant_action(self) -> str:
        """The action label with the highest conditional frequency."""
        if not self.action_frequencies:
            return ""
        return max(self.action_frequencies.items(), key=lambda kv: kv[1])[0]

    @property
    def dominant_frequency(self) -> float:
        """The highest conditional frequency, in [0, 1]."""
        if not self.action_frequencies:
            return 0.0
        return max(self.action_frequencies.values())

    @property
    def ev_gap_sb(self) -> float | None:
        """EV of the best action minus the second-best, in small blinds.

        Measures how costly the top mistake is -- the worthiness/difficulty
        signal. ``None`` when the node has fewer than two actions. Because
        the dominated actions (e.g. a clearly-bad fold) sort below the top
        two, this naturally ignores them: for a value hand choosing between
        4-bet (22.9) and call (20.7) the gap is 2.2, not the distance to a
        -7 fold.
        """
        if len(self.ev_by_action) < _MIN_ACTIONS_FOR_GAP:
            return None
        evs = sorted(self.ev_by_action.values(), reverse=True)
        return evs[0] - evs[1]


# --- range-file caching -----------------------------------------------------
# A `.rng` is 16,432 entries; enumerating all hands at a node reads each of
# the node's (<= 4) sibling files once and looks each hand up. The cache keeps
# a node's files hot across its hands, and a working set of recent nodes hot
# across a batch. Bounded small because each entry is ~1 MB.
@lru_cache(maxsize=64)
def _cached_node_values(path_str: str) -> tuple[tuple[float, float], ...]:
    return tuple(read_rng_values(Path(path_str)))


def sample_plo_spot(node: PloDecisionNode, hero_index: int) -> PloSpot:
    """Build a :class:`PloSpot` for one ``(node, hero hand)`` pair.

    Reads the hero hand's ``(p, ev)`` from each of the node's sibling action
    files, normalises the weights into a conditional strategy, and keeps the
    raw per-action EVs.

    Raises ``ValueError`` if ``hero_index`` is outside ``0..16431``.
    """
    if not 0 <= hero_index < HAND_COUNT:
        msg = f"hero_index {hero_index} out of range 0..{HAND_COUNT - 1}"
        raise ValueError(msg)

    raw: dict[str, tuple[float, float]] = {}
    for opt in node.actions:
        p, ev = _cached_node_values(str(opt.path))[hero_index]
        raw[opt.label] = (p, ev)

    presence = sum(p for p, _ in raw.values())
    if presence > 0:
        action_freqs = {label: p / presence for label, (p, _) in raw.items()}
    else:
        # Hand never reaches this node. Keep zeros so the dict structurally
        # matches the action set; the worthiness filter drops it on presence.
        action_freqs = {label: 0.0 for label in raw}
    ev_by_action = {label: ev for label, (_, ev) in raw.items()}

    return PloSpot(
        node=node,
        hero_index=hero_index,
        hero_label=monker_label(hero_index),
        hero_cards=tuple(cards_at(hero_index)),
        action_frequencies=action_freqs,
        ev_by_action=ev_by_action,
        presence=presence,
    )


def enumerate_plo_spots_for_node(
    node: PloDecisionNode,
    *,
    min_presence: float = 0.0,
) -> Iterator[PloSpot]:
    """Yield a :class:`PloSpot` for every hand at this node.

    Args:
        node: The node to sample.
        min_presence: Skip hands whose ``presence`` (how often the hand
            reaches this node) is below this threshold. Default 0 = include
            every hand; 0.01 is a sensible floor for "this hand actually
            shows up here".

    Yields one spot per non-filtered hand (up to 16,432).
    """
    for hero_index in range(HAND_COUNT):
        spot = sample_plo_spot(node, hero_index)
        if spot.presence < min_presence:
            continue
        yield spot


def sample_plo_spots_across_nodes(
    nodes: Iterable[PloDecisionNode],
    *,
    min_presence: float = 0.01,
) -> Iterator[PloSpot]:
    """Yield spots across many nodes, dropping hands with no real presence."""
    for node in nodes:
        yield from enumerate_plo_spots_for_node(node, min_presence=min_presence)
