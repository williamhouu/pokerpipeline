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
  * hero RANGE equity vs villain RANGE (range-vs-range)
  * blockers (count of villain combos hero blocks, grouped by class)
  * strategic archetype classifier (3bet_for_value / squeeze_as_bluff / ...)

Phase B will add: pot odds, SPR, multiway-villain handling, additional
concept tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pipeline.preflop.equity import (
    preflop_equity_vs_range,
    preflop_range_vs_range_equity,
)
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
    combo_str_to_hand_class,
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

    # Hero's RANGE equity vs villain's range, in [0.0, 1.0]. Distinct
    # from hero_equity_vs_villain (which is *this hand* vs villain's
    # range). Layer 6 uses this for "your range has 53% equity here"
    # framing. None if not computable.
    hero_range_equity_vs_villain: float | None = None

    # How many villain combos hero's specific cards block, grouped by
    # hand class. e.g. ``{"AA": 2, "AKs": 3, "AKo": 4}`` means hero's
    # cards remove 2 AA combos, 3 AKs combos, 4 AKo combos from villain's
    # range. Empty dict if no villain or no blockers computed.
    blockers: dict[str, int] = field(default_factory=dict)

    # Strategic archetype label (e.g. "3bet_for_value", "squeeze_as_bluff",
    # "open_for_value", "fold_dominated"). Empty string if not classified.
    # See classify_archetype() in this module for the full list.
    archetype: str = ""

    # Break-even equity for calling, in [0.0, 1.0] -- the pot-odds threshold
    # hero needs to call (call_cost / (pot + call_cost)). Layer 6 cites this
    # instead of computing pot odds itself ("you need about 41% to call").
    # Populated by the batch for call/fold spots (needs the pack's chip
    # geometry); None when hero faces no bet to call or it isn't computable.
    break_even_equity: float | None = None


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
        # First-to-act spot: hero is opening or facing only folds. There's
        # no specific villain range to characterize, but we STILL classify
        # the archetype (open_for_value / fold_outranged) -- otherwise opens
        # get a blank archetype, which starves both the LLM's strategic
        # frame and the "Preflop Hand Selection" skill. classify_archetype
        # handles villain=None (its open/fold branch). (June 2026 fix: this
        # branch used to return early without an archetype.)
        return PreflopFacts(
            spot=spot, archetype=classify_archetype(spot, None, None)
        )

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
        villain_combos = _cached_load_combo_weights(str(villain_path))
        hero_eq = compute_hero_equity_vs_range(
            spot.hero_card_combo,
            villain_combos,
            max_runouts=equity_runouts,
        )
        # Chunk 2 facts: range-vs-range equity, blockers, archetype.
        hero_combos = _hero_range_at_node_cached(spot.node)
        hero_range_eq = preflop_range_vs_range_equity(
            hero_range=hero_combos,
            villain_range=villain_combos,
            max_matchups=200,
            n_samples_per_matchup=50,
        )
        blockers = compute_blockers(spot.hero_card_combo, villain_combos)
        archetype = classify_archetype(spot, villain, hero_eq)
        return PreflopFacts(
            spot=spot,
            villain_stats=villain_stats,
            hero_equity_vs_villain=hero_eq,
            hero_equity_runouts_used=equity_runouts,
            hero_range_equity_vs_villain=hero_range_eq,
            blockers=blockers,
            archetype=archetype,
        )
    except (ValueError, OSError) as exc:
        logger.warning(
            "extract_facts: failed for %s: %s",
            spot.node.node_id,
            exc,
        )
        return PreflopFacts(spot=spot)


# --- chunk 2: hero's full range at this node -------------------------------
def _hero_range_at_node_cached(
    node: PreflopDecisionNode,
) -> dict[str, float]:
    """Sum all of hero's action ranges into one "reaching range" combo dict.

    At any decision node, hero's "current range" = union of every range
    they take at the node (each action's range file represents one part).
    Adding them up gives the combo distribution over hands that *reached*
    the decision point. Used for range-vs-range equity and other
    range-level facts.

    Cached at the node level since many spots from the same node need
    the same answer.
    """
    return _hero_range_combos_uncached(
        tuple(str(opt.range_file.path) for opt in node.actions)
    )


@lru_cache(maxsize=2048)
def _hero_range_combos_uncached(
    range_file_paths: tuple[str, ...],
) -> dict[str, float]:
    """Inner cached function -- tuple of path-strings is hashable."""
    combined: dict[str, float] = {}
    for path_str in range_file_paths:
        combos = _cached_load_combo_weights(path_str)
        for combo, weight in combos.items():
            if weight <= 0:
                continue
            combined[combo] = combined.get(combo, 0.0) + weight
    return combined


# --- chunk 2: blockers -----------------------------------------------------
def compute_blockers(
    hero_combo: str,
    villain_combos: dict[str, float],
) -> dict[str, int]:
    """Count villain combos hero's specific cards block, grouped by hand class.

    A combo is "blocked" if hero's hand shares at least one card with it.
    For citation prose ("your A♠ blocks 2 AA combos"), the per-class
    count is what reads naturally.

    Args:
        hero_combo: 4-char combo like ``'AsKh'``.
        villain_combos: ``{combo: weight}`` (1326 entries; only the
            positive-weight ones are inspected).

    Returns:
        ``{hand_class: count}`` for every class hero blocks at least one
        combo of. Empty dict if hero blocks nothing in the range.
    """
    hero_cards = {hero_combo[:2], hero_combo[2:]}
    counts: dict[str, int] = {}
    for combo, weight in villain_combos.items():
        if weight <= 0:
            continue
        cards = {combo[:2], combo[2:]}
        if not (cards & hero_cards):
            continue
        hc = combo_str_to_hand_class(combo)
        counts[hc] = counts.get(hc, 0) + 1
    return counts


# --- chunk 2: archetype classifier -----------------------------------------
def classify_archetype(
    spot: PreflopSpot,
    villain: ParsedAction | None,
    hero_equity_vs_villain: float | None,
) -> str:
    """Pick a strategic archetype label for the spot.

    Looks at hero's dominant action + the action context (open, facing
    raise, 3-bet, 4-bet, squeeze, all-in) + hero's equity vs villain to
    classify as value vs bluff vs fold reason.

    The 14 archetypes returned are listed in the module docstring.
    Returns a snake_case string. Defaults to ``"unclassified"`` if the
    pattern doesn't fit or hero's hand has near-zero presence at the node
    (i.e. hero would have folded earlier in the tree and never reached
    this decision).
    """
    # Defensive: hands that never actually reach the node (total weight
    # ~0 across all actions) shouldn't get a meaningful archetype --
    # they're filtered out by question_extractor anyway, but downstream
    # callers (smoke tests, ad-hoc tooling) might bypass that filter.
    total_presence = sum(spot.action_frequencies.values())
    if total_presence < 0.01:
        return "unclassified"

    dominant = spot.dominant_action

    # Count prior raises in history (excludes hero's own pending action).
    n_prior_raises = sum(
        1
        for a in spot.node.history_before
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )
    # Squeeze = there's a prior raise AND at least one caller between
    # raiser and hero.
    n_callers_after_raise = 0
    seen_raise = False
    for a in spot.node.history_before:
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN):
            seen_raise = True
            n_callers_after_raise = 0  # reset on each new raise
        elif seen_raise and a.action_type is PreflopActionType.CALL:
            n_callers_after_raise += 1

    # No villain faced. Only the BB can reach here without a bet to call -- they
    # posted the blind, so a limped/checked-around pot leaves them a CHECK (the
    # pack still files it under "Call"), not an open-fold. Everyone else here is
    # genuinely first-in with a raising option.
    if villain is None:
        if spot.node.actor == "BB" and dominant == "Call":
            return "bb_check"
        if "Raise" in dominant:
            return "open_for_value"
        if "AllIn" in dominant:
            return "open_for_value"  # rare preflop but treat as value
        if "Fold" in dominant:
            return "fold_outranged"
        return "unclassified"

    # Hero is facing a villain action. Classify by hero's dominant action.
    def value_or_bluff(suffix_v: str, suffix_b: str) -> str:
        if hero_equity_vs_villain is None:
            return suffix_v  # default to value when equity unknown
        return suffix_v if hero_equity_vs_villain >= 0.50 else suffix_b

    if dominant == "Fold":
        if hero_equity_vs_villain is not None and hero_equity_vs_villain < 0.40:
            return "fold_dominated"
        return "fold_pot_odds"  # folding despite some equity = wrong-priced

    if dominant == "Call":
        # Calling an all-in means no future streets -- it's a pure pot-odds /
        # raw-equity decision, NOT implied odds or postflop play. At 100bb an
        # all-in caps the betting, so once one exists in the history any call
        # here is an all-in call. (June 2026: previously these got
        # call_for_implied_odds, whose frame told the LLM to talk about
        # chasing draws / stacking villains in spots where everyone is already
        # all-in.)
        if any(
            a.action_type is PreflopActionType.ALL_IN
            for a in spot.node.history_before
        ):
            return "call_allin"
        return value_or_bluff("call_for_value", "call_for_implied_odds")

    if dominant == "AllIn":
        return value_or_bluff("all_in_for_value", "all_in_as_bluff")

    if "Raise" in dominant:
        if n_callers_after_raise > 0 and n_prior_raises == 1:
            # Open + at least one caller, then hero raises = squeeze.
            return value_or_bluff("squeeze_for_value", "squeeze_as_bluff")
        if n_prior_raises == 1:
            return value_or_bluff("3bet_for_value", "3bet_as_bluff")
        if n_prior_raises == 2:
            return value_or_bluff("4bet_for_value", "4bet_as_bluff")
        if n_prior_raises >= 3:
            return value_or_bluff("5bet_for_value", "5bet_as_bluff")

    return "unclassified"
