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
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pipeline.preflop.equity import (
    preflop_equity_vs_field,
    preflop_equity_vs_range,
    preflop_range_vs_range_equity,
)
from pipeline.preflop.grammars.types import (
    ParsedAction,
    PreflopActionType,
    render_raise_size_token,
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
# hero's equity. 400 (raised from 200, June 2026) roughly halves the
# per-spot noise; together with the per-spot seeded RNG (_spot_rng) the
# estimate is also REPRODUCIBLE, so threshold-fed values (archetype
# frames, dominated/coinflip tags, ev_gap, difficulty) can't flip
# between runs of the same spot.
DEFAULT_EQUITY_RUNOUTS = 400

# How many top combos to surface in VillainRangeStats.top_combos. Layer 6
# uses these for "villain's range has hands like AKs, AKo, KQs, ..."
# style citations. 5 is enough for prose without overwhelming the
# context window.
DEFAULT_TOP_COMBO_COUNT = 5

# Coverage-based selection for the user-facing "what you're up against"
# digest (VillainRangeStats.top_combos_covering): take the heaviest classes
# by combo share until ~two-thirds of the range is covered, but never fewer
# than COVERAGE_FLOOR nor more than COVERAGE_CAP. A fixed top-N misleads --
# it's ~all of a 5-hand 4-bet range but a quarter of a 30-hand flat range.
COVERAGE_TARGET = 0.67
COVERAGE_FLOOR = 3
COVERAGE_CAP = 6


def _combos_in_class(hand_class: str) -> int:
    """Combos in a 169-grid class: pairs 6, suited 4, offsuit 12."""
    if len(hand_class) == 2:  # a pair, e.g. "AA"
        return 6
    return 4 if hand_class.endswith("s") else 12


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
            by WEIGHT (ties broken premium-first), length up to
            ``DEFAULT_TOP_COMBO_COUNT``. Internal use only -- the input to
            the domination map (where a full-weight AA/KK must surface even
            though it's only 6 combos) and the exploit/capped-range checks.
            NOT what Layer 6 names as "villain's likely hands" -- use
            ``most_common_combos`` for that.
        most_common_combos: tuple of ``(hand_class, weight)`` pairs ranked
            by COMBO SHARE (weight x combo count), so the hands villain is
            most LIKELY to actually hold lead -- AKo's 12 combos outrank a
            pair's 6. Length up to ``DEFAULT_TOP_COMBO_COUNT``. This is the
            LLM-facing "most common combos" the SOLVER DATA block reads;
            distinct from ``top_combos`` (weight-sorted) which would mislead
            ("AA, A5s, A8s, ..." on a wide range -- all full weight, premium-
            and alpha-first, but not the bulk of the range).
        top_combos_covering: the most-weighted hand classes (by COMBO
            share, so AKo's 12 combos outweigh AKs's 4) needed to cover
            ~two-thirds of the range, capped at 6 / floored at 3. The
            user-facing "what you're up against" digest -- adaptive to
            range width where a fixed top-N misleads (a 30-hand flat-call
            range vs a 5-hand 4-bet range).
        top_combos_coverage_pct: what % of the range ``top_combos_covering``
            actually covers, in [0, 100] -- shown so the digest reads as a
            summary, not the whole range.
    """

    position: str
    action_label: str
    weighted_combo_count: float
    pct_of_dealt_hands: float
    top_combos: tuple[tuple[str, float], ...] = ()
    most_common_combos: tuple[tuple[str, float], ...] = ()
    top_combos_covering: tuple[str, ...] = ()
    top_combos_coverage_pct: float = 0.0


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

    # Hero hand's equity vs the PRIMARY villain's range, in [0.0, 1.0]. None
    # if no villain or equity couldn't be computed. NOTE: this is a HEADS-UP
    # number -- hero vs the single most-recent raiser. On a MULTI-WAY all-in
    # (two or more players already all-in / called the jam) it OVERSTATES
    # hero's real chance, because hero must beat everyone, not just the
    # shover -- use hero_equity_vs_field for the true equity-to-call there.
    hero_equity_vs_villain: float | None = None

    # Hero hand's equity (pot share) vs the FULL field of all-in opponents,
    # in [0.0, 1.0] -- the CORRECT equity to call when 2+ players are all-in
    # at showdown (hero must out-rank ALL of them). None on heads-up spots
    # (where hero_equity_vs_villain already is the right number) and when it
    # couldn't be computed. See showdown_opponents for who's in the field.
    hero_equity_vs_field: float | None = None

    # Positions of every opponent still in the pot with hero (the raiser plus
    # everyone who called / shoved and hasn't folded), e.g. ``("BTN", "HJ")``.
    # Length >= 2 means a MULTI-WAY pot (all-in or not) -- hero_equity_vs_field
    # is the equity vs everyone. Empty on heads-up / first-in spots.
    showdown_opponents: tuple[str, ...] = ()

    # Per-opponent equity breakdown: ``{position: hero's heads-up equity vs
    # that opponent's range}``, e.g. ``{"BTN": 0.42, "HJ": 0.55}``. Measured
    # on the same boards as hero_equity_vs_field (so they're consistent) and
    # populated alongside it on multi-way pots. Each value is >= the field
    # number (beating one opponent is easier than beating them all). Empty on
    # heads-up / first-in spots. Layer 6 uses it to say "X% vs BTN, Y% vs HJ,
    # Z% against both" instead of pretending the pot is heads-up.
    per_opponent_equity: dict[str, float] = field(default_factory=dict)

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

    # EV gap (in bb) between the dominant action and the second-best action --
    # the cost of taking the tempting wrong answer. Populated by the batch
    # (compute_ev_gap_bb, which needs the pack). None for raise-involved spots
    # (we don't model raise EVs in v1) or when equity isn't computable, so
    # Layer 6 must treat it as "cite only if present". See ev_engine.py.
    ev_gap_bb: float | None = None

    # The set of 169 hand-class labels the PRIMARY villain actually plays at
    # the node where they last acted -- every class with weight > 0 in their
    # range, e.g. ``frozenset({"AA", "AKs", "AKo", ...})``. This is the FULL
    # range (not the top subset in ``villain_stats.most_common_combos``), so a
    # soft validator can check a "villain has X" prose claim against the real
    # range and flag a class villain never holds. Empty on open / first-in
    # spots (no villain) and when range loading failed.
    villain_played_classes: frozenset[str] = field(default_factory=frozenset)


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


def identify_showdown_opponents(
    node: PreflopDecisionNode,
) -> list[ParsedAction]:
    """Every live opponent's last COMMITTING action -- the field hero faces
    at showdown if they call.

    An opponent is "live" when their MOST RECENT action in the history is a
    commitment (raise / call / all-in), not a fold. Keying on the last
    action per position is what makes this correct: a player who raised then
    folded to a re-jam (the opener folding to a squeeze-jam) drops out, while
    a player who called the jam stays in. Hero is excluded.

    Each returned ParsedAction is that opponent's last commit, so the caller
    can resolve the range file at the node where THEY acted (the shover's
    jamming range, a caller's call-the-jam range, ...). On a multi-way all-in
    this has length >= 2; heads-up it has length <= 1. This is the multi-way
    analogue of :func:`identify_villain` (which returns only the single most
    recent raiser).
    """
    hero = node.actor
    last_by_pos: dict[str, ParsedAction] = {}
    for action in node.history_before:
        last_by_pos[action.position] = action  # later action overwrites earlier
    committed = (
        PreflopActionType.RAISE,
        PreflopActionType.CALL,
        PreflopActionType.ALL_IN,
    )
    return [
        a
        for pos, a in last_by_pos.items()
        if pos != hero and a.action_type in committed
    ]


# --- villain range file path -----------------------------------------------
def construct_villain_range_path(
    node: PreflopDecisionNode,
    villain: ParsedAction,
    pack: PreflopPack,
) -> Path:
    """Build the absolute path to the villain's range file at the moment
    they took their last raise/all-in action.

    Ryan-pack grammar: take the action history up to and including the
    villain's last raise, format as
    ``<Pos>_<Action>_..._<VillainPos>_<VillainAction>``, and put it in the
    villain-position folder under the pack root. Monker packs are flat and
    name files by token chain alone, so the villain's file is simply the
    node's own stem truncated after the villain's action token -- sliced
    from a real option filename rather than re-encoded, so the pack's own
    token spelling (zero-padded raise sizes etc.) is preserved exactly.
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

    if pack.grammar_name == "monker_nlhe":
        # Any option file's stem = the history tokens + the option's own
        # token; the villain's range file is the prefix ending at their
        # action.
        option_path = node.actions[0].range_file.path
        stem_tokens = option_path.stem.split(".")
        villain_stem = ".".join(stem_tokens[: villain_index + 1])
        return pack.root_path / f"{villain_stem}{option_path.suffix}"

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

    # "Most common combos": the hands villain is most LIKELY to be holding,
    # ranked by COMBO SHARE (weight x combo count) so a 12-combo offsuit hand
    # outranks a single 6-combo pair at the same weight. This is what coaches
    # mean by "what they show up with" -- distinct from `top` above, which
    # sorts by weight alone and so degenerates to premium-/alpha-first when
    # most of the range is full weight (e.g. "AA, A5s, A8s, A9s, ATs" on a
    # wide open). Premium-first tie-break only matters between equal combo
    # shares. Layer 6's data block reads this; `top` stays internal.
    most_common = tuple(
        sorted(
            nonzero,
            key=lambda x: (
                -(x[1] * _combos_in_class(x[0])),
                canonical_index.get(x[0], 999),
            ),
        )[:top_n]
    )

    # Coverage-based digest: rank classes by COMBO share (weight x combos)
    # and take the heaviest until ~COVERAGE_TARGET of the range is covered,
    # bounded by [COVERAGE_FLOOR, COVERAGE_CAP]. Premium-first ties (same
    # canonical order as `top`). weighted_combos is the same combo total, so
    # the running sum / weighted_combos is the true fraction of the range.
    by_combo_share = sorted(
        ((c, w * _combos_in_class(c)) for c, w in nonzero),
        key=lambda x: (-x[1], canonical_index.get(x[0], 999)),
    )
    covering: list[str] = []
    acc = 0.0
    for hand_class, class_combos in by_combo_share:
        covering.append(hand_class)
        acc += class_combos
        if len(covering) >= COVERAGE_CAP:
            break
        if len(covering) >= COVERAGE_FLOOR and weighted_combos > 0 and (
            acc / weighted_combos >= COVERAGE_TARGET
        ):
            break
    coverage_pct = (acc / weighted_combos * 100.0) if weighted_combos > 0 else 0.0

    if villain.action_type is PreflopActionType.RAISE:
        action_label = f"Raise {render_raise_size_token(villain.raise_size_pct)}"
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
        most_common_combos=most_common,
        top_combos_covering=tuple(covering),
        top_combos_coverage_pct=coverage_pct,
    )


# --- hero equity vs villain range -----------------------------------------
def compute_hero_equity_vs_range(
    hero_combo: str,
    villain_combo_weights: dict[str, float],
    *,
    max_runouts: int = DEFAULT_EQUITY_RUNOUTS,
    rng: random.Random | None = None,
) -> float:
    """Hero hand vs villain's full range, no board cards.

    Returns a single float in [0.0, 1.0]. Uses
    ``pipeline.preflop.equity.preflop_equity_vs_range``, which Monte
    Carlos 5-card boards per villain combo. Pass a seeded ``rng`` for
    reproducible results (see :func:`_spot_rng`).
    """
    hero_cards = [hero_combo[:2], hero_combo[2:]]
    return preflop_equity_vs_range(
        hero=hero_cards,
        villain_range=villain_combo_weights,
        n_samples=max_runouts,
        rng=rng,
    )


def _spot_rng(spot: PreflopSpot) -> random.Random:
    """A deterministic RNG derived from the spot's identity.

    June 2026 audit finding: unseeded Monte-Carlo equity fed hard
    thresholds (the 0.40 fold_dominated/fold_pot_odds frame split, the
    dominated/coinflip tags, ev_gap_bb, and through it the difficulty
    score), so regenerating the SAME spot could flip its strategic frame
    and move its difficulty hundreds of points. Seeding by
    (node_id, hand class, combo) makes every recomputation of a spot
    byte-identical -- across batches, audits, and Compare runs -- while
    different spots still get independent samples.
    """
    return random.Random(
        f"{spot.node.node_id}|{spot.hero_hand_class}|{spot.hero_card_combo}"
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
        # Every hand class villain actually plays here (weight > 0). Used by
        # the soft "named combo in range" validator to catch prose that says
        # villain holds a hand class that's at 0% in their range.
        villain_played_classes = frozenset(
            combo_str_to_hand_class(combo)
            for combo, weight in villain_combos.items()
            if weight > 0
        )
        rng = _spot_rng(spot)
        hero_eq = compute_hero_equity_vs_range(
            spot.hero_card_combo,
            villain_combos,
            max_runouts=equity_runouts,
            rng=rng,
        )
        # Chunk 2 facts: range-vs-range equity, blockers, archetype.
        hero_combos = _hero_range_at_node_cached(spot.node)
        hero_range_eq = preflop_range_vs_range_equity(
            hero_range=hero_combos,
            villain_range=villain_combos,
            max_matchups=200,
            n_samples_per_matchup=50,
            rng=rng,
        )
        # Restrict to villain's TOP VALUE (~top 35% by strength) so the
        # blockers fact -- and the blocks_villain_top_value tag + the prose
        # citing it -- only count blocking hands that actually matter, not the
        # bottom of the range (22 clipping A2s).
        blockers = compute_blockers(
            spot.hero_card_combo, top_value_combos(villain_combos)
        )
        # Multi-way field equity. When 2+ opponents are still in the pot (all-in
        # OR not), hero's heads-up-vs-the-raiser number (hero_eq) overstates the
        # truth -- hero must beat the WHOLE field. Compute equity vs every
        # opponent's range, plus the per-opponent breakdown. Done AFTER the
        # heads-up calc (reusing the same seeded rng) so every existing field
        # stays byte-identical; only the new fields are added.
        hero_field_eq, per_opp_eq, showdown_positions = (
            _compute_multiway_field_equity(spot, pack, rng, equity_runouts)
        )
        # Archetype classified AFTER field equity so the value/bluff + fold
        # splits can use the FIELD number on multi-way must-beat-everyone spots
        # (H2). classify_archetype consumes NO rng, so moving it here leaves
        # every equity value byte-identical -- only the label changes.
        archetype = classify_archetype(
            spot,
            villain,
            hero_eq,
            hero_equity_vs_field=hero_field_eq,
            is_multiway=len(showdown_positions) >= 2,
        )
        return PreflopFacts(
            spot=spot,
            villain_stats=villain_stats,
            hero_equity_vs_villain=hero_eq,
            hero_equity_vs_field=hero_field_eq,
            showdown_opponents=showdown_positions,
            per_opponent_equity=per_opp_eq,
            hero_equity_runouts_used=equity_runouts,
            hero_range_equity_vs_villain=hero_range_eq,
            blockers=blockers,
            archetype=archetype,
            villain_played_classes=villain_played_classes,
        )
    except (ValueError, OSError) as exc:
        logger.warning(
            "extract_facts: failed for %s: %s",
            spot.node.node_id,
            exc,
        )
        return PreflopFacts(spot=spot)


def _compute_multiway_field_equity(
    spot: PreflopSpot,
    pack: PreflopPack,
    rng: random.Random,
    equity_runouts: int,
) -> tuple[float | None, dict[str, float], tuple[str, ...]]:
    """Hero's equity vs the FULL field, the per-opponent breakdown, and the
    field's positions -- on ANY multi-way pot (all-in or not).

    Returns ``(None, {}, ())`` unless 2+ opponents are still in the pot with
    hero (the raiser plus everyone who called / shoved and hasn't folded). On
    a heads-up pot (one opponent) ``hero_equity_vs_villain`` is already the
    right number, so we skip. Returns ``(None, {}, ())`` too if any opponent's
    range file can't be resolved -- the caller keeps the heads-up number.

    Fires for multi-way all-ins AND non-all-in multi-way pots (3-bet-plus-a-
    caller, squeezes, ...). On all-in pots the field number IS the decision;
    on non-all-in pots it is hero's RAW showdown equity (accurate, but
    postflop play still matters, so the archetype frame -- not this number
    alone -- drives a non-all-in decision).

    Each opponent's range is the action-conditional range at the node where
    they last committed (the same source the heads-up path uses for the
    primary villain). ``n_samples`` scales with ``equity_runouts`` so fast-test
    mode stays fast, floored so the estimate stays stable.
    """
    opponents = identify_showdown_opponents(spot.node)
    if len(opponents) < 2:
        return None, {}, ()  # heads-up -> hero_equity_vs_villain is correct
    opp_ranges: list[dict[str, float]] = []
    for opp in opponents:
        try:
            opp_path = construct_villain_range_path(spot.node, opp, pack)
        except ValueError:
            return None, {}, ()
        if not opp_path.is_file():
            return None, {}, ()
        opp_ranges.append(_cached_load_combo_weights(str(opp_path)))
    field_eq, per_opp = preflop_equity_vs_field(
        [spot.hero_card_combo[:2], spot.hero_card_combo[2:]],
        opp_ranges,
        n_samples=max(1500, equity_runouts * 5),
        rng=rng,
    )
    positions = tuple(opp.position for opp in opponents)
    per_opp_dict = {pos: per_opp[i] for i, pos in enumerate(positions)}
    return field_eq, per_opp_dict, positions


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
# Fraction of villain's range (by combo weight, strongest first) that counts
# as "top value" for the blocks_villain_top_value tag. Blocking a combo in this
# top tier is meaningful; blocking the bottom of the range (e.g. 22 clipping 2
# combos of A2s) is not. Strength is the Chen formula -- a standard,
# deterministic preflop hand-strength score, so no precomputed equity table is
# needed. 0.35 sits in the requested 30-40% band.
_TOP_VALUE_RANGE_FRACTION = 0.35
_CHEN_RANK_VALUE = {
    "A": 14, "K": 13, "Q": 12, "J": 11, "T": 10,
    "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2,
}
_CHEN_HIGH_POINTS = {"A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0}
_CHEN_GAP_PENALTY = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}


def _chen_score(hand_class: str) -> float:
    """Chen-formula preflop strength for a 169 hand-class label (e.g. 'AKs',
    '22', 'T7o'). The classic deterministic hand-ranking heuristic: high-card
    points, pair doubling, suited bonus, gap penalty, straight bonus. Used only
    to RANK villain's range, so the raw (un-rounded) score is fine.
    """
    def _pts(rank: str) -> float:
        return _CHEN_HIGH_POINTS.get(rank, _CHEN_RANK_VALUE[rank] / 2.0)

    if len(hand_class) == 2:  # pocket pair
        return max(_pts(hand_class[0]) * 2.0, 5.0)
    hi, lo, suit = hand_class[0], hand_class[1], hand_class[2]
    score = _pts(hi)
    if suit == "s":
        score += 2.0
    gap = _CHEN_RANK_VALUE[hi] - _CHEN_RANK_VALUE[lo] - 1
    score -= _CHEN_GAP_PENALTY.get(gap, 5.0)
    if gap <= 1 and _CHEN_RANK_VALUE[hi] <= 11:  # connected and below a queen
        score += 1.0
    return score


def top_value_combos(villain_combos: dict[str, float]) -> dict[str, float]:
    """The strongest ~35% of villain's range (by combo weight), as a
    ``{combo: weight}`` dict.

    Ranks villain's hand classes by Chen strength, walks from the top
    accumulating combo weight until it reaches
    :data:`_TOP_VALUE_RANGE_FRACTION` of the range, and returns every combo
    whose class is in that top tier. This is what "top value" means for
    :func:`compute_blockers` / the ``blocks_villain_top_value`` tag: blocking
    the bottom of the range (22 clipping A2s) no longer counts as blocking
    value, but blocking AA/AK (e.g. holding an ace) still does.
    """
    by_class: dict[str, float] = {}
    for combo, weight in villain_combos.items():
        if weight > 0:
            hc = combo_str_to_hand_class(combo)
            by_class[hc] = by_class.get(hc, 0.0) + weight
    total = sum(by_class.values())
    if total <= 0:
        return {}
    target = total * _TOP_VALUE_RANGE_FRACTION
    top_classes: set[str] = set()
    acc = 0.0
    for hc, weight in sorted(by_class.items(), key=lambda kv: -_chen_score(kv[0])):
        top_classes.add(hc)
        acc += weight
        if acc >= target:
            break
    return {
        combo: weight
        for combo, weight in villain_combos.items()
        if weight > 0 and combo_str_to_hand_class(combo) in top_classes
    }


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
    *,
    hero_equity_vs_field: float | None = None,
    is_multiway: bool = False,
) -> str:
    """Pick a strategic archetype label for the spot.

    Looks at hero's dominant action + the action context (open, facing
    raise, 3-bet, 4-bet, squeeze, all-in) + hero's equity to classify as
    value vs bluff vs fold reason.

    Which equity drives the value/bluff + dominated splits depends on the
    spot (June 2026, H2):

      * On a MULTI-WAY pot a fold / call / all-in is a "beat the whole field"
        decision, so the split uses ``hero_equity_vs_field`` (hero must beat
        EVERYONE). The heads-up-vs-one-villain number overstates equity
        multi-way and would mislabel near-threshold spots (e.g. a 52%-vs-the-
        raiser call that is 41% vs the field is an implied-odds call, not a
        value call).
      * A RAISE (3-bet / 4-bet / squeeze) isolates -- hero is usually heads-up
        vs the original raiser when called -- so it keeps
        ``hero_equity_vs_villain`` (voice rule 18 frames the prose the same
        way). Heads-up spots use it too.

    The full label catalog (with the frame each demands) is
    ``pipeline.preflop.explanation_generator.PREFLOP_ARCHETYPE_GUIDANCE``.
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

    # Which equity drives the value/bluff + dominated splits. On a multi-way
    # pot, fold/call/all-in are "beat the whole field" decisions -> use the
    # field number (hero must beat everyone). A raise isolates -> keep the
    # heads-up-vs-raiser number. Heads-up spots also keep it. (H2.)
    beats_field_decision = (
        is_multiway
        and hero_equity_vs_field is not None
        and dominant in ("Fold", "Call", "AllIn")
    )
    eq_for_split = (
        hero_equity_vs_field if beats_field_decision else hero_equity_vs_villain
    )

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
    # pack still files it under "Call"), not an open-fold. The SB first-in can
    # CALL too: completing the half bet (a limp), which is neither an open nor
    # a fold -- the Monker 9-max pack offers it (the Pio 6-max pack didn't).
    # Everyone else here is genuinely first-in with a raising option.
    if villain is None:
        if spot.node.actor == "BB" and dominant == "Call":
            return "bb_check"
        if spot.node.actor == "SB" and dominant == "Call":
            return "sb_complete"
        if "Raise" in dominant:
            return "open_for_value"
        if "AllIn" in dominant:
            return "open_for_value"  # rare preflop but treat as value
        if "Fold" in dominant:
            return "fold_outranged"
        return "unclassified"

    # Hero is facing a villain action. Classify by hero's dominant action,
    # using the equity that matches the decision (eq_for_split: field on
    # multi-way fold/call/all-in, heads-up otherwise -- see the docstring).
    def value_or_bluff(suffix_v: str, suffix_b: str) -> str:
        if eq_for_split is None:
            return suffix_v  # default to value when equity unknown
        return suffix_v if eq_for_split >= 0.50 else suffix_b

    if dominant == "Fold":
        # No CALL action offered at this node (e.g. SB facing a 3-bet with
        # only fold / 4-bet) means there is no price to weigh -- the choice is
        # raise-or-fold. Framing such a fold around pot odds ("your equity
        # sits under the price you'd need to call") is category-wrong AND was
        # observed reversing the equity-vs-price relationship (June 2026 A5o
        # SB-vs-3bet misfire). Route these to a domination/realization frame
        # that never mentions a calling price. Detected by the node's action
        # set, not hero's frequency, so a hand that folds 100% at a node that
        # DOES offer a call (a normal BB-vs-open fold) still gets the
        # pot-odds frame.
        if "Call" not in spot.action_frequencies:
            return "fold_no_continue"
        if eq_for_split is not None and eq_for_split < 0.40:
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
