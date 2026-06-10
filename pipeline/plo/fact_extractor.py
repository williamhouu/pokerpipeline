"""PLO preflop fact extraction (Layer 5).

Given a :class:`~pipeline.plo.spot_sampler.PloSpot`, computes the per-spot facts
Layer 6 needs to write a grounded explanation. The architectural rule holds: the
LLM never invents a strategic claim -- every claim in a generated explanation
must trace back to a field this module computes.

This chunk (the equity-free "structural" facts) mirrors
:mod:`pipeline.preflop.fact_extractor`:

  * hero hand classification (:func:`pipeline.plo.hand_model.classify_plo_hand`)
  * villain identification (the most recent raiser/jammer)
  * the villain's range at the node where they acted -- combo count, combo-
    weighted "% of hands", and top hands
  * a strategic archetype (open / 3-bet / squeeze / call-jam / fold ...)

Two PLO-specific choices vs the NLHE extractor:

1. **The EV gap comes for free.** Monker stores per-action EV, so the spot
   already carries ``ev_gap_sb``; :attr:`PloFacts.ev_gap_bb` just exposes it in
   big blinds (the pack is 100bb, SB = 1 chip, so 1 bb = 2 sb). No EV engine.
2. **value-vs-bluff keys off hand STRUCTURE, not equity.** PLO preflop equities
   run close (a weak value signal), and PLO players read value off the hand's
   shape -- premium pairs / big double-suited = value, speculative shapes =
   light. So the archetype's value/bluff split uses
   :attr:`~pipeline.plo.hand_model.PloHandClass.strength`. The threshold (a
   ``premium``/``strong`` hand is "value") is a tunable starting value, to be
   refined against graded output the same way the NLHE thresholds are.

Equity-vs-range, range-vs-range equity, and blockers are added in the next
chunk; those fields are present here but ``None`` / empty.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pipeline.plo.equity import (
    preflop_equity_vs_range,
    preflop_range_vs_range_equity,
)
from pipeline.plo.hand_model import PloHandClass, classify_plo_hand
from pipeline.plo.hand_order import cards_at, combo_multiplicity
from pipeline.plo.node_enumerator import PloDecisionNode, action_label
from pipeline.plo.pack import PloAction, PloActionType, PloPack, read_rng
from pipeline.plo.spot_sampler import PloSpot

logger = logging.getLogger(__name__)

_TOTAL_COMBOS = 270_725  # C(52, 4): every concrete PLO starting hand
_SB_PER_BB = 2  # this pack is 100bb with SB = 1 chip, BB = 2
_MIN_PRESENCE = 0.01  # below this a hand never really reaches the node
DEFAULT_TOP_HAND_COUNT = 5

# Equity is Monte Carlo and the PLO evaluator is ~60x heavier than NLHE's, so
# we keep the work bounded: a range is approximated by a capped, frequency-
# weighted sample of representative combos, and boards-per-combo is modest.
# These are estimates (PloFacts carries the runout count so Layer 6 avoids
# false precision); bump them if a quality review shows the noise mattering.
DEFAULT_EQUITY_RUNOUTS = 60  # boards per villain combo (hero vs range)
DEFAULT_RANGE_MATCHUPS = 150  # combo pairs sampled (range vs range)
DEFAULT_RANGE_RUNOUTS = 30  # boards per matchup (range vs range)
_COMBO_CAP = 60  # representative combos kept per sampled range
# Fixed seed for the (hero-independent) range->combo sampling so the bounded
# dicts are reproducible and cacheable; board runouts use the caller's rng.
_COMBO_SAMPLE_SEED = 1_000_003

# Actions that put chips in aggressively -- a hero facing one of these has a
# "villain"; a prior one of these is a raise level.
_AGGRESSIVE = {
    PloActionType.RAISE,
    PloActionType.MIN_RAISE,
    PloActionType.ALL_IN,
}
# Hand-strength buckets that read as "value" for the value/bluff split (tunable).
_VALUE_STRENGTHS = {"premium", "strong"}


@dataclass(frozen=True)
class PloVillainStats:
    """The villain's range at the node where they took their last raise/jam.

    Fields:
        seat: e.g. ``'BB'`` -- whose range this describes.
        action_label: the action that produced it, e.g. ``'Raise 100%'``.
        weighted_combo_count: sum of ``p * multiplicity`` over the villain's
            present hands -- the real concrete-combo count of the range.
        pct_of_dealt_hands: ``weighted_combo_count / 270725 * 100`` -- the
            "X% of hands" figure a coach quotes.
        top_hands: up to :data:`DEFAULT_TOP_HAND_COUNT` ``(monker_label,
            weight)`` pairs, the highest-weight hands in the range.
    """

    seat: str
    action_label: str
    weighted_combo_count: float
    pct_of_dealt_hands: float
    top_hands: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class PloFacts:
    """Pre-computed facts for one PLO preflop spot.

    Layer 6 reads this when writing the explanation -- every strategic claim in
    the prose must trace back to a field here.
    """

    spot: PloSpot
    hand_class: PloHandClass  # hero's hand structure (shape, strength, ...)
    archetype: str  # strategic frame (see classify_plo_archetype)

    # The primary villain's range (most recent raiser). None when hero is first
    # to act, or when the range file couldn't be loaded.
    villain_stats: PloVillainStats | None = None

    # --- equity (filled by the next chunk; None/empty here) ---
    hero_equity_vs_villain: float | None = None
    hero_equity_runouts_used: int = 0
    hero_range_equity_vs_villain: float | None = None
    blockers: dict[str, int] = field(default_factory=dict)

    @property
    def ev_gap_bb(self) -> float | None:
        """The spot's EV gap (best minus second-best action) in big blinds.

        Straight from the solver via the spot's small-blind ``ev_gap_sb``.
        Available for every spot, raise spots included.
        """
        gap = self.spot.ev_gap_sb
        return None if gap is None else gap / _SB_PER_BB


# --- villain identification -------------------------------------------------
def identify_villain(node: PloDecisionNode) -> tuple[int, PloAction] | None:
    """The most recent aggressive action (raise / jam) before hero, with its
    index in ``node.history_before``.

    Returns ``None`` for a first-in spot (hero opening, or only folds before
    them) -- there is no specific villain yet.
    """
    for i in range(len(node.history_before) - 1, -1, -1):
        action = node.history_before[i]
        if action.action in _AGGRESSIVE:
            return i, action
    return None


def villain_range_stem(node: PloDecisionNode, villain_index: int) -> str:
    """The `.rng` stem of the node where the villain took their action.

    The villain's range file is the prefix of this node's path ending at the
    villain's own token: ``history_stem`` tokens ``[: villain_index + 1]``.
    (Each ``history_before[i]`` is token ``history_stem.split('.')[i]``.)
    """
    tokens = node.history_stem.split(".")
    return ".".join(tokens[: villain_index + 1])


# --- villain range stats ----------------------------------------------------
@lru_cache(maxsize=256)
def _villain_range_summary(
    path_str: str,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Combo-weighted size + top hands of a range file. Cached: many hero hands
    share one villain node, and the file is 16,432 lines."""
    entries = read_rng(Path(path_str))
    weighted = sum(e.p * combo_multiplicity(e.index) for e in entries)
    top = tuple(
        (e.label, e.p)
        for e in sorted(entries, key=lambda e: (-e.p, e.index))[
            :DEFAULT_TOP_HAND_COUNT
        ]
    )
    return weighted, top


def compute_villain_stats(
    seat: str, action_label_str: str, villain_path: Path
) -> PloVillainStats:
    """Build :class:`PloVillainStats` from a villain range file."""
    weighted, top = _villain_range_summary(str(villain_path))
    return PloVillainStats(
        seat=seat,
        action_label=action_label_str,
        weighted_combo_count=weighted,
        pct_of_dealt_hands=weighted / _TOTAL_COMBOS * 100.0,
        top_hands=top,
    )


# --- range -> bounded concrete-combo dict (for equity) ----------------------
def _combo_str(index: int) -> str:
    """Representative 8-char combo for a hand index, e.g. ``'AcAdAhAs'``."""
    return "".join(cards_at(index))


def _sample_combo_dict(
    weighted_indices: list[tuple[int, float]], cap: int
) -> dict[str, float]:
    """Frequency-weighted sample of ``cap`` representative combos.

    ``weighted_indices`` is ``[(hand_index, p * multiplicity), ...]``. Sampling
    with replacement by weight gives a ``{combo: count}`` dict whose counts
    track the true combo distribution while staying bounded for Monte Carlo.
    Uses each hand's representative suits (a small, documented blocker bias vs
    realizing hero-avoiding suits -- fine for an estimate, refine if needed).
    Deterministic via a fixed seed so the dict is reproducible and cacheable.
    """
    positive = [(i, w) for i, w in weighted_indices if w > 0]
    if not positive:
        return {}
    rng = random.Random(_COMBO_SAMPLE_SEED)
    indices = [i for i, _ in positive]
    weights = [w for _, w in positive]
    out: dict[str, float] = {}
    for index in rng.choices(indices, weights=weights, k=cap):
        combo = _combo_str(index)
        out[combo] = out.get(combo, 0.0) + 1.0
    return out


@lru_cache(maxsize=256)
def _villain_combo_dict(path_str: str, cap: int) -> dict[str, float]:
    """A bounded ``{combo: weight}`` sample of one range file (hero-independent,
    so cached across all the hero hands that face this villain node)."""
    entries = read_rng(Path(path_str))
    return _sample_combo_dict(
        [(e.index, e.p * combo_multiplicity(e.index)) for e in entries], cap
    )


@lru_cache(maxsize=128)
def _hero_range_combo_dict(action_paths: tuple[str, ...], cap: int) -> dict[str, float]:
    """A bounded combo dict for hero's whole reaching range at a node.

    Hero's range = the union of every action they take here; summing ``p`` over
    the node's action files gives each hand's reach probability. Cached per node.
    """
    totals: dict[int, float] = {}
    for path_str in action_paths:
        for entry in read_rng(Path(path_str)):
            totals[entry.index] = totals.get(entry.index, 0.0) + entry.p
    return _sample_combo_dict(
        [(i, p * combo_multiplicity(i)) for i, p in totals.items()], cap
    )


@lru_cache(maxsize=128)
def _node_range_equity(
    hero_action_paths: tuple[str, ...], villain_path_str: str
) -> float | None:
    """Hero's RANGE equity vs the villain's range at a node.

    A node-level fact (independent of which hero hand we're on), so it's cached
    per node and computed with a fixed seed -- one range-vs-range Monte Carlo
    serves every hero hand sampled from the node, and the result is stable
    rather than re-noised per hand. ``None`` if either range is empty.
    """
    hero = _hero_range_combo_dict(hero_action_paths, _COMBO_CAP)
    villain = _villain_combo_dict(villain_path_str, _COMBO_CAP)
    if not hero or not villain:
        return None
    return preflop_range_vs_range_equity(
        hero,
        villain,
        max_matchups=DEFAULT_RANGE_MATCHUPS,
        n_samples_per_matchup=DEFAULT_RANGE_RUNOUTS,
        rng=random.Random(_COMBO_SAMPLE_SEED),
    )


# --- archetype classifier ---------------------------------------------------
def _is_value(strength: str) -> bool:
    """Whether a hand-strength bucket reads as value (vs bluff/speculative)."""
    return strength in _VALUE_STRENGTHS


def _dominant_action_type(spot: PloSpot) -> PloActionType | None:
    """The :class:`PloActionType` of the spot's dominant (highest-freq) action."""
    label = spot.dominant_action
    for opt in spot.node.actions:
        if opt.label == label:
            return opt.action.action
    return None


def classify_plo_archetype(
    spot: PloSpot,
    hand_class: PloHandClass,
    *,
    villain_action: PloAction | None,
) -> str:
    """Pick a strategic archetype label for the spot.

    Keys off hero's dominant action, the raise level / squeeze context, and the
    hand's structural strength (value vs light). Returns a snake_case label or
    ``"unclassified"``. Hands that never really reach the node (presence < 1%)
    are ``"unclassified"`` -- the worthiness filter drops them anyway, but
    ad-hoc callers might not.
    """
    if spot.presence < _MIN_PRESENCE:
        return "unclassified"

    dominant = _dominant_action_type(spot)
    value = _is_value(hand_class.strength)

    n_prior_raises = sum(
        1 for a in spot.node.history_before if a.action in _AGGRESSIVE
    )
    # Squeeze = a prior raise AND at least one caller between it and hero.
    n_callers_after_raise = 0
    seen_raise = False
    for a in spot.node.history_before:
        if a.action in _AGGRESSIVE:
            seen_raise = True
            n_callers_after_raise = 0  # reset on each new raise
        elif seen_raise and a.action is PloActionType.CALL:
            n_callers_after_raise += 1

    # No raise faced. Only the BB can reach here without a bet to call -- they
    # posted the blind, so a limped/checked-around pot leaves them a CHECK, not
    # an open-fold. The SB first-in can CALL too: completing the half bet (a
    # limp), which is neither an open nor a fold -- labelling it "open_fold"
    # handed the LLM a fold frame for a Call answer. Everyone else here is
    # genuinely first-in (open / open-fold).
    if villain_action is None:
        if spot.node.actor == "BB" and dominant is PloActionType.CALL:
            return "bb_check"
        if spot.node.actor == "SB" and dominant is PloActionType.CALL:
            return "sb_complete"
        if dominant in (PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN):
            return "open_for_value"
        return "open_fold"

    if dominant is PloActionType.FOLD:
        return "fold_dominated" if not value else "fold_pot_odds"

    if dominant is PloActionType.CALL:
        # Calling a jam caps the betting -- no future streets, so it's a pure
        # pot-odds call, not implied odds (no draws to chase, no stacks to win).
        if any(a.action is PloActionType.ALL_IN for a in spot.node.history_before):
            return "call_allin"
        return "call_for_value" if value else "call_for_implied_odds"

    if dominant is PloActionType.ALL_IN:
        return "all_in_for_value" if value else "all_in_as_bluff"

    if dominant in (PloActionType.RAISE, PloActionType.MIN_RAISE):
        if n_callers_after_raise > 0 and n_prior_raises == 1:
            return "squeeze_for_value" if value else "squeeze_as_bluff"
        if n_prior_raises == 1:
            return "3bet_for_value" if value else "3bet_as_bluff"
        if n_prior_raises == 2:  # noqa: PLR2004
            return "4bet_for_value" if value else "4bet_as_bluff"
        if n_prior_raises >= 3:  # noqa: PLR2004
            return "5bet_for_value" if value else "5bet_as_bluff"

    return "unclassified"


# --- main entry point -------------------------------------------------------
def extract_plo_facts(
    spot: PloSpot,
    pack: PloPack,
    *,
    compute_equity: bool = True,
    equity_runouts: int = DEFAULT_EQUITY_RUNOUTS,
    rng: random.Random | None = None,
) -> PloFacts:
    """Compute the per-spot facts Layer 6 needs.

    Args:
        spot: the sampled spot.
        pack: the source pack (for villain-range-file lookup).
        compute_equity: when ``False``, skip the (expensive) Monte Carlo equity
            and leave the equity fields ``None`` -- e.g. for a cheap worthiness
            pre-filter before paying for equity on the survivors.
        equity_runouts: boards per villain combo for the hand-vs-range equity.
        rng: optional ``random.Random`` for reproducible equity runouts.

    Always returns a valid :class:`PloFacts` -- if villain range loading fails,
    the villain/equity fields stay ``None`` rather than raising, so one
    malformed spot can't bring down a batch.
    """
    hand_class = classify_plo_hand(spot.hero_cards)
    villain = identify_villain(spot.node)

    if villain is None:
        archetype = classify_plo_archetype(spot, hand_class, villain_action=None)
        return PloFacts(spot=spot, hand_class=hand_class, archetype=archetype)

    villain_index, villain_action = villain
    archetype = classify_plo_archetype(
        spot, hand_class, villain_action=villain_action
    )

    try:
        stem = villain_range_stem(spot.node, villain_index)
        villain_path = pack.root / f"{stem}.rng"
        if not villain_path.is_file():
            logger.warning(
                "extract_plo_facts: villain range missing for %s: %s.rng",
                spot.node.node_id,
                stem,
            )
            return PloFacts(spot=spot, hand_class=hand_class, archetype=archetype)

        villain_stats = compute_villain_stats(
            villain_action.seat, action_label(villain_action), villain_path
        )

        hero_eq: float | None = None
        hero_range_eq: float | None = None
        runouts_used = 0
        if compute_equity:
            rng = rng or random.Random()
            villain_combos = _villain_combo_dict(villain_path.as_posix(), _COMBO_CAP)
            if villain_combos:
                hero_eq = preflop_equity_vs_range(
                    spot.hero_cards, villain_combos, n_samples=equity_runouts, rng=rng
                )
                runouts_used = equity_runouts
                # Range-vs-range is a node-level fact -> cached, hero-independent.
                hero_range_eq = _node_range_equity(
                    tuple(opt.path.as_posix() for opt in spot.node.actions),
                    villain_path.as_posix(),
                )

        return PloFacts(
            spot=spot,
            hand_class=hand_class,
            archetype=archetype,
            villain_stats=villain_stats,
            hero_equity_vs_villain=hero_eq,
            hero_equity_runouts_used=runouts_used,
            hero_range_equity_vs_villain=hero_range_eq,
        )
    except (ValueError, OSError) as exc:
        logger.warning(
            "extract_plo_facts: failed for %s: %s", spot.node.node_id, exc
        )
        return PloFacts(spot=spot, hand_class=hand_class, archetype=archetype)
