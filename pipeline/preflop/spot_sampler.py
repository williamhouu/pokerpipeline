"""Preflop spot sampling.

A ``PreflopSpot`` pairs a ``PreflopDecisionNode`` (a decision point in
the tree) with a specific hero hand class (e.g. ``'AKo'``), and carries
the per-action frequencies for that hand class -- everything Layer 6
needs to write an explanation, plus everything the question extractor
needs to filter by difficulty.

Typical use::

    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import discover_packs
    from pipeline.preflop.spot_sampler import sample_spot

    packs = discover_packs("ranges")
    nodes = enumerate_nodes(packs)
    spot = sample_spot(nodes[0], hero_hand_class="AKo")
    print(spot.action_frequencies)
    print(spot.dominant_action, spot.dominant_frequency)

Spots are produced lazily -- ``sample_spot`` only reads the range files
for the one node passed in, not the whole pack. The
``enumerate_spots_for_node`` helper iterates over all 169 hand classes
in one node, and ``sample_spots_across_nodes`` iterates across many
nodes; both lazy generators so callers can short-circuit.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pipeline.preflop.node_enumerator import (
    PreflopActionOption,
    PreflopDecisionNode,
)
from pipeline.preflop_ranges import canonical_169_hand_classes, parse_range_file


@dataclass(frozen=True)
class PreflopSpot:
    """One preflop decision spot ready for question generation.

    Fields:
        node: The decision node this spot is sampled from.
        hero_hand_class: One of the 169 preflop class labels
            (``'AA'``, ``'A2s'``, ``'AKo'``, etc.).
        hero_card_combo: A specific 4-char combo (``'AhKc'``) chosen as
            the visible representative for the hand class. Used by Layer
            8's ``User Cards`` column.
        action_frequencies: CONDITIONAL strategy at this decision -- the
            mapping from action ``label`` to hero's frequency given the
            hand reaches the node. e.g. ``{"Fold": 0.0, "Call": 0.18,
            "Raise 60%": 0.82}``. The values sum to 1.0 (modulo rounding)
            when ``presence > 0``, or all zeros when ``presence == 0``.
        dominant_action: The action label with the highest conditional
            frequency.
        dominant_frequency: That highest conditional frequency, in
            [0.0, 1.0]. Used by the worthiness filter and difficulty
            rating -- both want "given the hand is here, how dominant
            is the best action".
        presence: How often this hand reaches the node, in [0.0, 1.0].
            Equal to the sum of the RAW (presence-weighted) Pio weights
            before normalisation. Used by the worthiness filter's
            "does the hand actually reach this decision" check (a hand
            with presence < 0.01 was almost always folded earlier in
            the tree and isn't a real spot).

    Normalisation note: Pio's range files give us raw weights that are
    the JOINT probability (hand reaches node AND takes action). For
    question writing we want the CONDITIONAL strategy at this decision,
    so :func:`sample_spot` divides each raw weight by their sum (the
    presence) to get the normalised dict stored here. Without that
    step, a hand that reaches the node only 62% of the time but folds
    100% of the time it does would show as ``Fold: 62%`` -- looking
    like a mixed strategy when it's actually pure-fold.

    The default ``presence = -1.0`` is a sentinel for "auto-compute from
    action_frequencies on init", so test fixtures that construct
    ``PreflopSpot`` directly with already-normalised data don't need
    to track this field explicitly. :func:`sample_spot` always passes
    an explicit value.
    """

    node: PreflopDecisionNode
    hero_hand_class: str
    hero_card_combo: str
    action_frequencies: dict[str, float] = field(default_factory=dict)
    dominant_action: str = ""
    dominant_frequency: float = 0.0
    presence: float = -1.0

    def __post_init__(self) -> None:
        if self.presence < 0:
            # Auto-compute from action_frequencies. For real spots
            # (built by sample_spot) presence is always passed explicitly;
            # this branch is just for test-fixture convenience.
            object.__setattr__(
                self, "presence", float(sum(self.action_frequencies.values()))
            )


# --- range-file caching -----------------------------------------------------
# Range files are small (~1-2 KB each) but a generation run reads them many
# times across many spots. lru_cache keeps the parsed-dict form hot.
@lru_cache(maxsize=2048)
def _cached_parse_range_file(path_str: str) -> dict[str, float]:
    return parse_range_file(Path(path_str))


def _frequencies_for_hand_class(
    actions: tuple[PreflopActionOption, ...],
    hand_class: str,
) -> dict[str, float]:
    """Load the range file for each action; look up the hand class's weight."""
    return {
        opt.label: _cached_parse_range_file(str(opt.range_file.path)).get(
            hand_class, 0.0
        )
        for opt in actions
    }


# --- hand-class to representative combo ------------------------------------
# Pick a deterministic representative card combo for each hand class so the
# rendered question always shows real cards. For pairs (e.g. "AA"), uses
# clubs+diamonds. For suited (e.g. "AKs"), uses spades. For offsuit (e.g.
# "AKo"), uses hearts + clubs.
_RANK_TO_CHAR = {
    "A": "A",
    "K": "K",
    "Q": "Q",
    "J": "J",
    "T": "T",
    "9": "9",
    "8": "8",
    "7": "7",
    "6": "6",
    "5": "5",
    "4": "4",
    "3": "3",
    "2": "2",
}


def representative_combo_for_class(hand_class: str) -> str:
    """Return a fixed 4-char combo representing the hand class.

    Examples::

        representative_combo_for_class("AA")   -> "AcAd"
        representative_combo_for_class("AKs")  -> "AsKs"
        representative_combo_for_class("AKo")  -> "AhKc"
        representative_combo_for_class("22")   -> "2c2d"

    Deterministic. Callers wanting random combos within a class can call
    :func:`random_combo_for_class` instead.
    """
    if len(hand_class) == 2:  # pair: "AA", "KK"...
        rank = _RANK_TO_CHAR[hand_class[0]]
        return f"{rank}c{rank}d"
    if len(hand_class) != 3:
        raise ValueError(f"invalid hand_class: {hand_class!r}")
    high, low, kind = hand_class[0], hand_class[1], hand_class[2]
    if kind == "s":
        return f"{high}s{low}s"
    if kind == "o":
        return f"{high}h{low}c"
    raise ValueError(f"hand_class must end in 's' or 'o', got {hand_class!r}")


def random_combo_for_class(hand_class: str, *, rng: random.Random | None = None) -> str:
    """Random representative combo for the hand class.

    Pairs have 6 combos, suited have 4, offsuit have 12. Useful when
    generating many questions from the same hand class -- avoids visual
    repetition.
    """
    rng = rng or random.Random()
    suits = "cdhs"
    if len(hand_class) == 2:  # pair
        rank = hand_class[0]
        pair = rng.sample(suits, 2)
        return f"{rank}{pair[0]}{rank}{pair[1]}"
    if len(hand_class) != 3:
        raise ValueError(f"invalid hand_class: {hand_class!r}")
    high, low, kind = hand_class[0], hand_class[1], hand_class[2]
    if kind == "s":
        s = rng.choice(suits)
        return f"{high}{s}{low}{s}"
    if kind == "o":
        suit_pairs = [(a, b) for a in suits for b in suits if a != b]
        a, b = rng.choice(suit_pairs)
        return f"{high}{a}{low}{b}"
    raise ValueError(f"hand_class must end in 's' or 'o', got {hand_class!r}")


# --- the main API -----------------------------------------------------------
def sample_spot(
    node: PreflopDecisionNode,
    hero_hand_class: str,
    *,
    combo: str | None = None,
) -> PreflopSpot:
    """Build a PreflopSpot for one (node, hand-class) pair.

    Args:
        node: The decision node.
        hero_hand_class: One of the 169 hand class labels (``'AKo'`` etc).
        combo: Optional explicit card combo. If None, uses the
            deterministic representative combo for the class.

    Returns:
        A fully populated PreflopSpot.

    Raises:
        ValueError: if ``hero_hand_class`` isn't among the 169 canonical
            classes.
    """
    if hero_hand_class not in canonical_169_hand_classes():
        raise ValueError(
            f"unknown hand_class {hero_hand_class!r} "
            f"(expected one of the 169 canonical classes)"
        )
    chosen_combo = combo or representative_combo_for_class(hero_hand_class)

    # Pio's range files give us PRESENCE-WEIGHTED weights (joint
    # probability of reaching this node AND taking the action). Sum
    # them to get presence (how often the hand reaches here), then
    # divide each by presence to get the CONDITIONAL strategy at this
    # decision -- which is what the worthiness filter, difficulty
    # rating, options selection, and the CSV display all want.
    raw_freqs = _frequencies_for_hand_class(node.actions, hero_hand_class)
    presence = sum(raw_freqs.values())
    if presence > 0:
        action_freqs = {label: weight / presence for label, weight in raw_freqs.items()}
    else:
        # Hand never reaches this node. Keep zeros so action_frequencies
        # structurally matches the action set; downstream worthiness
        # checks will drop the spot via the presence filter.
        action_freqs = dict(raw_freqs)

    dominant_label, dominant_freq = (
        max(action_freqs.items(), key=lambda kv: kv[1]) if action_freqs else ("", 0.0)
    )
    return PreflopSpot(
        node=node,
        hero_hand_class=hero_hand_class,
        hero_card_combo=chosen_combo,
        action_frequencies=action_freqs,
        dominant_action=dominant_label,
        dominant_frequency=dominant_freq,
        presence=presence,
    )


def enumerate_spots_for_node(
    node: PreflopDecisionNode,
    *,
    min_total_weight: float = 0.0,
) -> Iterator[PreflopSpot]:
    """Yield a PreflopSpot for every hand class at this node.

    Args:
        node: The node to sample.
        min_total_weight: Skip hand classes whose ``presence`` (how often
            the hand reaches this node) is below this threshold.
            Default 0 = include every class (even hands the actor never
            has). 0.01 is a sensible minimum for "this hand actually
            shows up here".

    Yields:
        One PreflopSpot per non-filtered hand class.
    """
    for hand_class in canonical_169_hand_classes():
        spot = sample_spot(node, hand_class)
        # presence is the un-normalised sum -- the right signal for
        # "does this hand reach the node at all". Summing
        # action_frequencies (now conditional) would always give ~1.0
        # for present hands and 0 for absent ones, losing resolution.
        if spot.presence < min_total_weight:
            continue
        yield spot


def sample_spots_across_nodes(
    nodes: Iterable[PreflopDecisionNode],
    *,
    min_total_weight: float = 0.01,
) -> Iterator[PreflopSpot]:
    """Yield spots across many nodes, filtering hands with no presence.

    Useful for generating a large batch without manually iterating.
    """
    for node in nodes:
        yield from enumerate_spots_for_node(node, min_total_weight=min_total_weight)
