"""Preflop fact extraction.

Layer 5 (preflop edition). Given a PreflopSpot, computes the per-spot
facts Layer 6 needs to write a grounded explanation -- hero equity vs
villain's range, villain range stats (combo count, top hands), etc.

The architectural rule that drives the entire pipeline applies here too:
the LLM must never invent strategic claims. Every claim in a generated
explanation must be backed by a field this module computes.

For Phase A this module computes:

  * villain identification (the most recent raiser/aggressor)
  * villain range loading + characterization (combo count, % of hands)
  * hero equity vs villain's range
  * top villain combos (the most-weighted hands in their range)

Phase B will add: hero RANGE vs villain RANGE equity, blocker counts,
strategic archetype classification, pot odds, SPR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline.preflop.equity import preflop_equity_vs_range
from pipeline.preflop.grammars.types import (
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import PreflopDecisionNode
from pipeline.preflop.pack import PreflopPack
from pipeline.preflop.spot_sampler import PreflopSpot
from pipeline.preflop_ranges import (
    HAND_COUNT,
    canonical_169_hand_classes,
    combo_label,
    parse_range_file,
)

logger = logging.getLogger(__name__)

# How many 5-card boards to sample per villain combo when computing
# hero's equity. 200 gives ~1-2% noise on the per-spot equity number;
# adequate for filter thresholds and prose, not for a published equity
# figure. Increase if quality review shows the noise mattering;
# decrease if batch speed matters more than the second decimal.
DEFAULT_EQUITY_RUNOUTS = 200

# How many top combos to surface in VillainRangeStats.top_combos. Layer 6
# uses these for "villain's range has hands like AKs, AKo, KQs, ..."
# style citations. 5 is enough for prose without overwhelming the
# context window.
DEFAULT_TOP_COMBO_COUNT = 5


@dataclass(frozen=True)
class VillainRangeStats:
    """Stats describing one villain's preflop range at the time of their
    most recent action.

    Fields:
        position: e.g. ``'BTN'`` -- whose range this describes.
        action_label: human-readable label of the action that produced
            this range (e.g. ``'Raise 60%'`` for an open, ``'Raise 182%'``
            for a 3-bet).
        weighted_combo_count: sum of weights over all 169 hand classes
            after expansion to 1326 combos. Equals raw combo count if all
            weights are 1.0; less otherwise.
        pct_of_dealt_hands: ``weighted_combo_count / 1326`` as a
            percentage in [0, 100]. The "X% of hands" number coaches
            quote.
        top_combos: tuple of ``(hand_class, weight)`` pairs, descending
            by weight, length up to ``DEFAULT_TOP_COMBO_COUNT``.
    """

    position: str
    action_label: str
    weighted_combo_count: float
    pct_of_dealt_hands: float
    top_combos: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class PreflopFacts:
    """Pre-computed facts for one preflop spot.

    Layer 6 reads this dataclass when writing the explanation -- every
    strategic claim in the prose must trace back to a field here.
    """

    spot: PreflopSpot

    # The "primary villain" -- the player whose range matters most for
    # hero's current decision (usually the most recent raiser). None if
    # there is no clear villain (e.g. hero is first to act preflop).
    villain_stats: VillainRangeStats | None = None

    # Hero hand's equity vs villain's range, in [0.0, 1.0]. None if no
    # villain or equity couldn't be computed.
    hero_equity_vs_villain: float | None = None

    # Sampling-noise estimate on hero_equity_vs_villain (rough ~1-2% at
    # DEFAULT_EQUITY_RUNOUTS). Carried so Layer 6 can avoid claiming
    # false precision in prose.
    hero_equity_runouts_used: int = 0


# --- villain identification -------------------------------------------------
def identify_villain(
    node: PreflopDecisionNode,
) -> ParsedAction | None:
    """Return the most-recent action that involved putting chips in (raise
    or all-in) before hero's decision, or None if there's no such action.

    For a node where hero is first to act (history_before is empty or all
    folds), returns None -- the question is "what should you open?" and
    there is no specific villain yet.
    """
    for action in reversed(node.history_before):
        if action.action_type in (
            PreflopActionType.RAISE,
            PreflopActionType.ALL_IN,
        ):
            return action
    return None


# --- villain range file path -----------------------------------------------
def construct_villain_range_path(
    node: PreflopDecisionNode,
    villain: ParsedAction,
    pack: PreflopPack,
) -> Path:
    """Build the absolute path to the villain's range file at the moment
    they took their last raise/all-in action.

    The grammar is: take the action history up to and including the
    villain's last raise, format as ``<Pos>_<Action>_..._<VillainPos>_<VillainAction>``,
    and put it in the villain-position folder under the pack root.
    """
    # Find the index of the villain's last raise/all-in action in history.
    villain_index = None
    for i in range(len(node.history_before) - 1, -1, -1):
        a = node.history_before[i]
        if (
            a.position == villain.position
            and a.action_type is villain.action_type
            and a.raise_size_pct == villain.raise_size_pct
        ):
            villain_index = i
            break
    if villain_index is None:
        raise ValueError(
            f"villain {villain.position} not found in node history: {node.node_id}"
        )

    chain = node.history_before[: villain_index + 1]
    tokens = []
    for a in chain:
        if a.action_type is PreflopActionType.RAISE:
            verb = f"{a.raise_size_pct:g}%"
        else:
            verb = (
                a.action_type.value
                if a.action_type is not PreflopActionType.ALL_IN
                else "AI"
            )
        tokens.append(f"{a.position}_{verb}")
    filename = "_".join(tokens) + ".txt"
    return pack.root_path / villain.position / filename


# --- range loading & expansion --------------------------------------------
@lru_cache(maxsize=2048)
def _cached_load_combo_weights(path_str: str) -> dict[str, float]:
    """Load a range file and return a dict of {combo_label: weight} (1326
    entries, with zeros for combos hero never has). Cached so repeated
    villain-range queries inside one batch are fast.
    """
    hand_class_weights = parse_range_file(Path(path_str))
    return _expand_to_combo_dict(hand_class_weights)


def _expand_to_combo_dict(
    hand_class_weights: dict[str, float],
) -> dict[str, float]:
    """Expand a 169-class range to a 1326-combo dict.

    Equity helpers in pipeline.fact_extractor.equity expect a dict keyed
    by 4-char combo strings (``'AhKh'``), not 169-class strings. Each of
    the 1326 combos inherits the weight of its hand class.
    """
    # combo_to_hand_class is the official one-way mapping; we walk all
    # combo positions and look up each combo's class.
    from pipeline.preflop_ranges import combo_cards, combo_to_hand_class

    out: dict[str, float] = {}
    for pos in range(HAND_COUNT):
        a, b = combo_cards(pos)
        hand_class = combo_to_hand_class(a, b)
        out[combo_label(pos)] = hand_class_weights.get(hand_class, 0.0)
    return out


# --- range stats ----------------------------------------------------------
def compute_villain_range_stats(
    villain: ParsedAction,
    range_file_path: Path,
    *,
    top_n: int = DEFAULT_TOP_COMBO_COUNT,
) -> VillainRangeStats:
    """Load a villain's range file; compute combo count, % of dealt hands,
    top combos."""
    hand_class_weights = parse_range_file(range_file_path)

    # Weighted combo count: each hand class's weight times its combo count
    # (pairs=6, suited=4, offsuit=12). Easier to compute by expanding to
    # 1326 combos and summing.
    combo_weights = _expand_to_combo_dict(hand_class_weights)
    weighted_combos = sum(combo_weights.values())
    pct = (weighted_combos / HAND_COUNT) * 100.0

    # Top hand classes by weight (not combos -- "AA 100%" reads cleaner
    # than "AhAd 100%, AhAc 100%, ..."). Skip 0-weight entries. Ties
    # broken by canonical Ryan-pack order so premium hands (AA, AKs,
    # AKo, AQs, ...) surface first rather than 22 / 32s (alphabetical).
    canonical_index = {c: i for i, c in enumerate(canonical_169_hand_classes())}
    nonzero = [(c, w) for c, w in hand_class_weights.items() if w > 0]
    nonzero.sort(key=lambda x: (-x[1], canonical_index.get(x[0], 999)))
    top = tuple(nonzero[:top_n])

    if villain.action_type is PreflopActionType.RAISE:
        action_label = f"Raise {villain.raise_size_pct:g}%"
    elif villain.action_type is PreflopActionType.ALL_IN:
        action_label = "AllIn"
    else:
        action_label = villain.action_type.value

    return VillainRangeStats(
        position=villain.position,
        action_label=action_label,
        weighted_combo_count=weighted_combos,
        pct_of_dealt_hands=pct,
        top_combos=top,
    )


# --- hero equity vs villain range -----------------------------------------
def compute_hero_equity_vs_range(
    hero_combo: str,
    villain_combo_weights: dict[str, float],
    *,
    max_runouts: int = DEFAULT_EQUITY_RUNOUTS,
) -> float:
    """Hero hand vs villain's full range, no board cards.

    Returns a single float in [0.0, 1.0]. Uses
    ``pipeline.preflop.equity.preflop_equity_vs_range``, which Monte
    Carlos 5-card boards per villain combo. Carries 1-2pp noise at
    default settings.
    """
    hero_cards = [hero_combo[:2], hero_combo[2:]]
    return preflop_equity_vs_range(
        hero=hero_cards,
        villain_range=villain_combo_weights,
        n_samples=max_runouts,
    )


# --- main entry point -----------------------------------------------------
def extract_facts(
    spot: PreflopSpot,
    pack: PreflopPack,
    *,
    equity_runouts: int = DEFAULT_EQUITY_RUNOUTS,
    top_combo_count: int = DEFAULT_TOP_COMBO_COUNT,
) -> PreflopFacts:
    """Compute the per-spot facts Layer 6 needs.

    Args:
        spot: The PreflopSpot produced by spot_sampler.
        pack: The source PreflopPack (for villain-range-file path lookup).
        equity_runouts: Per-combo runout sample count for the equity
            calculation. Higher = slower + more accurate.
        top_combo_count: How many of villain's most-weighted hand classes
            to surface for Layer 6 citations.

    Returns:
        PreflopFacts. Even if villain identification or range loading
        fails, returns a valid PreflopFacts with None fields rather than
        raising -- a malformed villain spot doesn't bring down a batch.
    """
    villain = identify_villain(spot.node)
    if villain is None:
        # First-to-act spot: hero is opening or facing only folds. No
        # specific villain range to characterize. Still a valid spot.
        return PreflopFacts(spot=spot)

    try:
        villain_path = construct_villain_range_path(spot.node, villain, pack)
        if not villain_path.is_file():
            logger.warning(
                "extract_facts: villain range file missing for %s: %s",
                spot.node.node_id,
                villain_path.name,
            )
            return PreflopFacts(spot=spot)

        villain_stats = compute_villain_range_stats(
            villain,
            villain_path,
            top_n=top_combo_count,
        )
        combo_weights = _cached_load_combo_weights(str(villain_path))
        hero_eq = compute_hero_equity_vs_range(
            spot.hero_card_combo,
            combo_weights,
            max_runouts=equity_runouts,
        )
        return PreflopFacts(
            spot=spot,
            villain_stats=villain_stats,
            hero_equity_vs_villain=hero_eq,
            hero_equity_runouts_used=equity_runouts,
        )
    except (ValueError, OSError) as exc:
        logger.warning(
            "extract_facts: failed for %s: %s",
            spot.node.node_id,
            exc,
        )
        return PreflopFacts(spot=spot)
