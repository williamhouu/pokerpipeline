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
from dataclasses import dataclass, field

from pipeline.artifact_strip import strip_artifact_mass
from pipeline.postflop.premise import artifact_allin_action_labels
from pipeline.postflop.solve import NodeAction, PostflopNode


@dataclass(frozen=True)
class PostflopSpot:
    """One postflop decision spot ready for question generation.

    ``action_frequencies`` maps action *label* -> hero's frequency for THIS
    combo (normalised to sum ~1). ``dominant_action`` / ``dominant_frequency``
    are the modal action and its frequency -- the worthiness gate and
    difficulty read these. ``node`` carries the board, pot, ranges, and the
    range-aggregate action data.

    ARTIFACT-STRIP (July 2026, team standing rule): ``artifact_labels`` are
    the node's artifact-jam actions (:func:`pipeline.postflop.premise.
    artifact_allin_action_labels`). When this combo's frequency mass on them
    is a convergence-sliver trace (< ``ARTIFACT_MATERIALITY``), the mass is
    STRIPPED from ``action_frequencies`` and the rest renormalised -- the
    stripped strategy then drives every question surface (worthiness,
    options and their Always/Mostly qualifiers, difficulty, the SOLVER DATA
    block, the action_frequencies column, neutral_credit);
    ``stripped_artifact_freq`` records the removed mass for reviewer
    transparency. At/above materiality the solver genuinely wants the jam
    sometimes (mixing is EV-parity), so ``artifact_material`` is set and the
    spot is NEVER asked anywhere (narrated only). Realistic short-stack jams
    are not artifacts and pass through untouched.
    """

    node: PostflopNode
    hero_combo: str
    action_frequencies: dict[str, float]
    dominant_action: str
    dominant_frequency: float
    artifact_labels: frozenset[str] = field(default_factory=frozenset)
    stripped_artifact_freq: float = 0.0
    artifact_material: bool = False

    @property
    def hero_cards(self) -> list[str]:
        """Hero's two hole cards, e.g. ``["Ac", "Jc"]``."""
        return [self.hero_combo[:2], self.hero_combo[2:]]

    @property
    def dominant_verb(self) -> str:
        """The canonical verb of the dominant action (e.g. "bet" for "Bet 33%")."""
        action = self.node.action_by_label(self.dominant_action)
        return action.verb if action else ""

    @property
    def live_actions(self) -> tuple[NodeAction, ...]:
        """The node's actions minus artifact jams -- what the option builder
        and every other question surface may show. INVARIANT: no artifact
        all-in label may ever appear in a shipped option; anything that
        builds an option menu must read this, not ``node.actions``."""
        if not self.artifact_labels:
            return self.node.actions
        return tuple(
            a for a in self.node.actions if a.label not in self.artifact_labels
        )


def sample_spot(node: PostflopNode, hero_combo: str) -> PostflopSpot:
    """Build a :class:`PostflopSpot` for one (node, combo) pair.

    Reads the per-combo strategy at the node and normalises it (solver mixes
    sum to ~1 already, but we divide by the total defensively so a slightly
    off mix still yields clean conditional frequencies). Actions the combo
    never takes are omitted from the strategy and treated as 0.

    Applies the ARTIFACT-STRIP (see :class:`PostflopSpot`): trace-frequency
    artifact-jam mass is removed and the mix renormalised; material mass
    flags the spot unaskable instead (its frequencies stay honest/unstripped).
    Always on -- generation and the audit re-verifiers share this seam, so
    rebuilt rows stay byte-identical.

    Raises:
        KeyError: if ``hero_combo`` has no strategy at ``node``.
    """
    raw = node.strategy[hero_combo]
    total = sum(raw.values())
    if total > 0:
        freqs = {label: weight / total for label, weight in raw.items()}
    else:
        freqs = dict(raw)

    artifact = artifact_allin_action_labels(node)
    freqs, artifact_freq, material = strip_artifact_mass(freqs, artifact)

    dominant = max(freqs, key=lambda label: freqs[label]) if freqs else ""
    return PostflopSpot(
        node=node,
        hero_combo=hero_combo,
        action_frequencies=freqs,
        dominant_action=dominant,
        dominant_frequency=freqs.get(dominant, 0.0),
        artifact_labels=artifact,
        stripped_artifact_freq=0.0 if material else artifact_freq,
        artifact_material=material,
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

    Returned as an absolute, non-negative gap. Artifact-jam actions are
    excluded (artifact-strip: the jam no longer exists to be the tempting
    second-best, so the gap -- and difficulty's ``easy_ev`` -- measures the
    real alternatives).
    """
    node = spot.node
    combo_evs = node.combo_evs.get(spot.hero_combo)
    if combo_evs:
        evs = sorted(
            (ev for label, ev in combo_evs.items() if label not in spot.artifact_labels),
            reverse=True,
        )
    else:
        evs = sorted(
            (a.ev_bb for a in spot.live_actions if a.ev_bb is not None),
            reverse=True,
        )
    if len(evs) < 2:
        return None
    return float(evs[0] - evs[1])


def spot_action_evs_bb(spot: PostflopSpot) -> dict[str, float] | None:
    """Per-action EV (bb) for THIS hand, keyed by action label.

    The faithful hand-specific values (``combo_evs[hero_combo]``) when the
    solve exposes per-combo EVs; otherwise the range-mean action EVs
    (``actions[*].ev_bb``). ``None`` when no action carries an EV (the source
    solve doesn't expose EVs) -- the ``action_ev_bb`` column is then blank,
    mirroring the preflop EV-less-pack case. Same source-preference order as
    :func:`spot_ev_gap_bb`, so the per-action column and the (internal) gap
    agree on which EVs they read. Artifact-jam actions are excluded (the
    stripped action must not resurface via the EV column or chat context).
    """
    node = spot.node
    combo_evs = node.combo_evs.get(spot.hero_combo)
    if combo_evs:
        filtered = {
            label: ev for label, ev in combo_evs.items()
            if label not in spot.artifact_labels
        }
        return filtered or None
    range_mean = {a.label: a.ev_bb for a in spot.live_actions if a.ev_bb is not None}
    return range_mean or None


__all__ = [
    "PostflopSpot",
    "enumerate_spots",
    "sample_spot",
    "spot_action_evs_bb",
    "spot_ev_gap_bb",
]
