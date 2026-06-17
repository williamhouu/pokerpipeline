"""Layer 3 (postflop): turn a decision node + a hero combo into a spot.

A :class:`PostflopSpot` pairs a :class:`~pipeline.postflop.solve.PostflopNode`
with one specific hero hand (a 4-char combo like ``"AcJc"``) and carries that
hand's normalised action mix at the node -- everything the worthiness gate,
the fact extractor, and the option builder need.

Mirrors :mod:`pipeline.preflop.spot_sampler` (one node + one hand -> one
spot). Postflop the "hand" is a concrete combo on a board, not a 169 class,
because the board makes two combos of the same preflop class play
differently (``AhKh`` with a flush draw vs ``AhKd`` without).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from pipeline.postflop.solve import PostflopNode


@dataclass(frozen=True)
class PostflopSpot:
    """One postflop decision spot ready for question generation.

    ``action_frequencies`` maps action *label* -> hero's frequency for THIS
    combo (normalised to sum ~1). ``dominant_action`` / ``dominant_frequency``
    are the modal action and its frequency -- the worthiness gate and
    difficulty read these. ``node`` carries the board, pot, ranges, and the
    range-aggregate action data.
    """

    node: PostflopNode
    hero_combo: str
    action_frequencies: dict[str, float]
    dominant_action: str
    dominant_frequency: float

    @property
    def hero_cards(self) -> list[str]:
        """Hero's two hole cards, e.g. ``["Ac", "Jc"]``."""
        return [self.hero_combo[:2], self.hero_combo[2:]]

    @property
    def dominant_verb(self) -> str:
        """The canonical verb of the dominant action (e.g. "bet" for "Bet 33%")."""
        action = self.node.action_by_label(self.dominant_action)
        return action.verb if action else ""


def sample_spot(node: PostflopNode, hero_combo: str) -> PostflopSpot:
    """Build a :class:`PostflopSpot` for one (node, combo) pair.

    Reads the per-combo strategy at the node and normalises it (solver mixes
    sum to ~1 already, but we divide by the total defensively so a slightly
    off mix still yields clean conditional frequencies). Actions the combo
    never takes are omitted from the strategy and treated as 0.

    Raises:
        KeyError: if ``hero_combo`` has no strategy at ``node``.
    """
    raw = node.strategy[hero_combo]
    total = sum(raw.values())
    if total > 0:
        freqs = {label: weight / total for label, weight in raw.items()}
    else:
        freqs = dict(raw)
    dominant = max(freqs, key=lambda label: freqs[label]) if freqs else ""
    return PostflopSpot(
        node=node,
        hero_combo=hero_combo,
        action_frequencies=freqs,
        dominant_action=dominant,
        dominant_frequency=freqs.get(dominant, 0.0),
    )


def enumerate_spots(node: PostflopNode) -> Iterator[PostflopSpot]:
    """Yield a spot for every hero combo with a defined strategy at ``node``."""
    for hero_combo in node.strategy:
        yield sample_spot(node, hero_combo)


def spot_ev_gap_bb(spot: PostflopSpot) -> float | None:
    """The EV gap (bb) between the best and second-best action for THIS hand.

    The cost of taking the best *wrong* answer -- the signal the worthiness
    gate and difficulty use. Prefers the node's per-combo EVs
    (``combo_evs[hero_combo]``, the faithful hand-specific gap); falls back to
    the range-mean action EVs (``actions[*].ev_bb``); returns ``None`` when
    fewer than two actions carry an EV (the gap is undefined).

    Returned as an absolute, non-negative gap.
    """
    node = spot.node
    combo_evs = node.combo_evs.get(spot.hero_combo)
    if combo_evs:
        evs = sorted(combo_evs.values(), reverse=True)
    else:
        evs = sorted(
            (a.ev_bb for a in node.actions if a.ev_bb is not None),
            reverse=True,
        )
    if len(evs) < 2:
        return None
    return float(evs[0] - evs[1])


__all__ = ["PostflopSpot", "enumerate_spots", "sample_spot", "spot_ev_gap_bb"]
