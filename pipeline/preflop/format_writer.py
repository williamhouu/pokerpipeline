"""Layer 8 (preflop edition): Format Writer for preflop spots.

The preflop sibling of ``pipeline.format_writer``. Same role: produce a CSV
row from a populated ``PreflopFacts`` + a ``GeneratedExplanation`` + the
source ``PreflopPack``. The schema is ``PREFLOP_CSV_COLUMNS`` -- the shared
``CSV_COLUMNS`` (defined in the postflop module) minus the columns the NLHE
preflop path dropped in June 2026 (the stat_notes-duplicate decision-math
columns, ev_gap_bb, and the easy_* difficulty diagnostics -- see
``_PREFLOP_DROPPED_COLUMNS``). The PLO writer keeps the full shared schema.

What differs from postflop:

  * **Hand Stage** is always ``"Preflop"`` (no street to encode).
  * **Cards on Table** is empty -- no board has come yet.
  * **Question** is built by a new deterministic preflop renderer
    (:func:`format_preflop_question`) that produces a coaching-voice
    narrative from the action history -- it never calls the LLM. Mirrors
    ``pipeline.action_history.format_action_history`` for postflop but
    only renders the preflop slice.
  * **board_texture** is empty (no board).
  * **hand_class** carries the 169-class label (``"AKo"``, ``"JTs"``)
    rather than the postflop made-hand + draw breakdown.
  * **concept_tags** is empty for now; the preflop concept tagger is
    deferred to Phase B (see ``pipeline.preflop.question_extractor``
    module docstring for the rationale).
  * **ev_gap_bb** is empty; preflop range files don't carry per-action
    EVs, so the gap can't be computed without an equity-driven EV
    engine. Also deferred to Phase B.
  * **solver_reference** is a pack-relative descriptive path
    (``ranges/<pack>/<actor>/<history>``).
  * **ip_range** / **oop_range** are empty for now; computing the 169-
    class snapshots at a preflop node requires summing action ranges
    (hero) and parsing villain's range file. Worth doing once the first
    review batch confirms the rest of the schema is right; deferred to
    a follow-up turn so step 8 stays focused on the Hand Stage / Cards
    on Table / narrative differences.

Stake context is configurable via the ``stakes_bb_dollars`` kwarg
(defaults to $0.50 = Tier 1 default). SB is derived from
``pack.sb_to_bb_ratio`` (0.5 for cash, but tournaments use other ratios).

Usage::

    from pipeline.preflop.format_writer import build_preflop_row, write_preflop_csv
    row = build_preflop_row(facts, explanation, pack=pack,
                            difficulty_score=1500, number=1)
    write_preflop_csv("out.csv",
                      [(facts1, exp1, 1500), (facts2, exp2, 1700)],
                      pack=pack)

Uses the standard library ``csv`` module; no pandas dependency.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pipeline.chat_context import StrategyEntry, build_chat_context
from pipeline.explanation_generator import GeneratedExplanation
from pipeline.format_writer import CSV_COLUMNS
from pipeline.neutral_credit import format_neutral_credit, neutral_credit_options
from pipeline.provenance import build_notes
from pipeline.preflop.action_history import (
    ResolvedPreflopState,
    raise_to_bb,
    resolve_preflop_history,
)
from pipeline.preflop.app_table_format import build_app_table_columns
from pipeline.preflop.concept_tags import (
    _non_fold_actor_count,
    compute_concept_tags,
)
from pipeline.preflop.difficulty import DifficultyResult
from pipeline.preflop.domination import dominating_map
from pipeline.preflop.exploit import exploit_adjustments, exploit_notes_to_json
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType
from pipeline.preflop.node_enumerator import (
    PreflopActionOption,
    PreflopDecisionNode,
)
from pipeline.preflop.options import (
    canonicalize_action_label,
    canonicalize_strategy,
    is_check_spot,
)
from pipeline.preflop.pack import PreflopPack
from pipeline.preflop.position import hero_relative_position
from pipeline.preflop.stat_notes import (
    build_stat_notes,
    stat_notes_to_json,
)
from pipeline.preflop_ranges import (
    canonical_169_hand_classes,
    parse_monker_rng_file,
    parse_range_file,
)

# --- the NLHE preflop CSV schema (June 2026 trim) ----------------------------
# Columns the NLHE preflop writer drops from the SHARED schema
# (``pipeline.format_writer.CSV_COLUMNS``). The PLO writer keeps the full
# shared schema, so it imports CSV_COLUMNS directly and is unaffected -- this
# trim is NLHE-preflop-only (the team's call, June 2026). Reasons:
#   * pot_odds / hero_equity / blocker_combos / top_villain_combos -- flat
#     duplicates of values already inside the ``stat_notes`` JSON (the app
#     reads stat_notes for the "Show the math" panel).
#   * range_equity -- a QA-only number no longer surfaced in the panel.
#   * ev_gap_bb -- the single-number gap; the per-action ``action_ev_bb``
#     column (which the Review page charts) supersedes it. ev_gap_bb is still
#     computed INTERNALLY for difficulty + the worthiness gate; only the CSV
#     column is dropped.
#   * easy_freq / easy_ev / easy_concept / easy_hand -- difficulty-algorithm
#     diagnostics; the ``Difficulty Rating`` itself stays.
_PREFLOP_DROPPED_COLUMNS = frozenset({
    "pot_odds", "hero_equity", "range_equity", "blocker_combos",
    "top_villain_combos", "ev_gap_bb",
    "easy_freq", "easy_ev", "easy_concept", "easy_hand",
})
# The preflop schema: the shared column order, minus the dropped set. Derived
# from CSV_COLUMNS so any future shared-schema column is inherited unless it's
# explicitly dropped here.
PREFLOP_CSV_COLUMNS: list[str] = [
    c for c in CSV_COLUMNS if c not in _PREFLOP_DROPPED_COLUMNS
]

# --- position-prose vocabulary -----------------------------------------------
# Mirrors ``pipeline.action_history``'s mappings but redefined locally so the
# preflop module isn't reaching into the postflop file's private API. Identical
# vocabulary on purpose -- the two layers should render position prose the
# same way; if the postflop file's wording changes, mirror the update here.
_HERO_PHRASE: dict[str, str] = {
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "UTG+2": "UTG+2",
    "LJ": "in the Lojack",
    "HJ": "in the Hijack",
    "CO": "in the Cutoff",
    "BTN": "on the Button",
    "SB": "in the Small Blind",
    "BB": "in the Big Blind",
}
_VILLAIN_REF: dict[str, str] = {
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "UTG+2": "UTG+2",
    "LJ": "The Lojack",
    "HJ": "The Hijack",
    "CO": "The Cutoff",
    "BTN": "The Button",
    "SB": "The Small Blind",
    "BB": "The Big Blind",
}

# Preflop raise levels: how many raises happened before this action.
# 1 = open, 2 = 3-bet, 3 = 4-bet, 4 = 5-bet, anything beyond falls back
# to "raise" (very rare; defensive fallback only).
_RAISE_LEVEL_VERB: dict[int, str] = {
    1: "opens",
    2: "3-bets",
    3: "4-bets",
    4: "5-bets",
}

# Preflop pot type by raise count. Mirrors ``_PREFLOP_POT_TYPE`` in the
# postflop format_writer so the same column reads identically across both
# paths (a 3-bet pot is a 3-bet pot regardless of whether the question is
# preflop or postflop).
_PREFLOP_POT_TYPE: dict[int, str] = {
    0: "Limped pot",
    1: "Single raise pot",
    2: "Three bet pot",
    3: "Four bet pot",
}

# Game-format prose form for the Cash/Tourney column. Matches the postflop
# format_writer's casing convention ("Cash" not "cash").
_GAME_FORMAT_PROSE: dict[str, str] = {
    "cash": "Cash",
    "tournament": "Tournament",
}


# --- small formatting helpers ------------------------------------------------
def _dollars(amount: float) -> str:
    """Cash amount as a string. Integer dollars render '$50'; cents '$1.25'.

    Mirrors ``pipeline.format_writer._dollars`` so the two layers stay in
    sync. Ryan-feedback Fix 1 (May 2026): drop trailing '.00' on whole-
    dollar amounts.
    """
    if isinstance(amount, float) and not amount.is_integer():
        return f"${amount:,.2f}"
    return f"${int(amount):,}"


def _stack_depth_bucket(effective_stack_bb: float) -> str:
    """The prose Stack Depth bucket from the effective stack (in BB).

    v1 thresholds match ``pipeline.format_writer._stack_depth_bucket``:
    < 40bb short, 40-150bb standard, > 150bb deep.
    """
    if effective_stack_bb < 40:
        return "Short stack"
    if effective_stack_bb <= 150:
        return "Standard Stack"
    return "Deep stack"


def _format_action_frequencies(strategy: dict[str, float]) -> str:
    """The action_frequencies CSV column value.

    Renders the strategy as a comma-separated list of
    ``<label>: <integer>%`` entries, ordered by descending frequency.

    Integer percentages are rounded using the **largest-remainder
    method** (Hare-Niemeyer) so they sum to exactly 100. Naive
    per-entry rounding produces visible totals like
    ``"Fold: 94%, 4-bet: 5%"`` (sums to 99) or
    ``"Call: 60%, Fold: 25%, 4-bet: 16%"`` (sums to 101). The
    largest-remainder method floors each entry then distributes the
    deficit by handing +1 to the entries with the largest fractional
    parts -- mathematically fair (no single column absorbs all the
    rounding error) and the standard fix for this class of bug.

    Actions the solver essentially never takes (< 0.5%) are dropped
    before rounding (team, July 2026) -- "All-in: 0%" rows are display
    noise; the postflop column applies the same trim. Empty strategy ->
    empty string. All-zero strategy (hand doesn't reach the node) ->
    empty string too, since 0% × N doesn't sum to 100 and showing
    "Fold: 0%, Call: 0%" is misleading.
    """
    if not strategy:
        return ""
    total = sum(strategy.values())
    if total <= 0:
        # All-zero strategy. Nothing to render -- the column is
        # best-effort and a "0%" row carries no information.
        return ""
    # Drop never-taken actions (< 0.5%) before rounding -- "All-in: 0%"
    # rows are noise (July 2026; matches the postflop column's trim). The
    # remaining entries are renormalised by the rounding to sum to 100.
    strategy = {lbl: f for lbl, f in strategy.items() if f >= 0.005}  # noqa: PLR2004
    if not strategy:
        return ""
    by_freq_desc = sorted(strategy.items(), key=lambda kv: -kv[1])
    # Largest-remainder rounding.
    raw = [(label, freq * 100.0) for label, freq in by_freq_desc]
    floors = [(label, int(value), value - int(value)) for label, value in raw]
    deficit = 100 - sum(floor for _, floor, _ in floors)
    if deficit != 0:
        # Top `deficit` entries by remainder get +1. (For negative deficit
        # -- impossible with non-negative inputs that sum to <=1 -- the
        # bottom |deficit| entries would get -1; not exercised.)
        ranked_by_remainder = sorted(enumerate(floors), key=lambda kv: -kv[1][2])[
            : max(deficit, 0)
        ]
        bumps = {idx for idx, _ in ranked_by_remainder}
    else:
        bumps = set()
    parts = [
        f"{label}: {floor + (1 if i in bumps else 0)}%"
        for i, (label, floor, _) in enumerate(floors)
    ]
    return ", ".join(parts)


def _hero_raise_level_for_node(facts: PreflopFacts) -> int:
    """How many raises hero's prospective raise would be the (1+N)-th of.

    Mirrors ``pipeline.preflop.options._hero_raise_level`` (kept local to
    avoid importing a private helper) -- counts raises + all-ins in the
    history so raise labels canonicalise to the right verb level.
    """
    return 1 + sum(
        1
        for a in facts.spot.node.history_before
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )


def action_evs_bb(facts: PreflopFacts, pack: PreflopPack) -> dict[str, float] | None:
    """Per-action solver EV (bb) for the hero hand, keyed by canonical label.

    Public so the batch's EV-coherence guard
    (:func:`pipeline.preflop.batch.spot_mix_incoherent`) can reuse the exact
    same per-action EV read that fills the ``action_ev_bb`` column.

    Reads each action's range file at hero's node and pulls the hero
    hand's solver EV, converting from the pack's raw unit to bb via
    ``pack.ev_units_per_bb``. Keyed by the same canonical labels as
    ``canonicalize_strategy`` so it lines up with ``action_frequencies``.

    Returns ``None`` when the pack carries no per-action EVs
    (``ev_units_per_bb`` is ``None`` -- e.g. the PioViewer Ryan pack stores
    weights only), so the ``action_ev_bb`` column is left blank. A file
    that can't be read (a ``.txt`` pack slipping through, or a gap) is
    skipped defensively rather than crashing the row.
    """
    units = pack.ev_units_per_bb
    if units is None:
        return None
    raise_level = _hero_raise_level_for_node(facts)
    check_spot = is_check_spot(facts)
    hero = facts.spot.hero_hand_class
    evs: dict[str, float] = {}
    for option in facts.spot.node.actions:
        try:
            data = parse_monker_rng_file(option.range_file.path)
        except (OSError, ValueError):
            continue
        entry = data.get(hero)
        if entry is None:
            continue
        label = canonicalize_action_label(
            option.label, raise_level=raise_level, check_spot=check_spot
        )
        # Preflop nodes carry one raise size, so canonical labels are 1:1
        # with options; if two ever collided, keep the first (deterministic).
        evs.setdefault(label, entry[1] / units)
    return evs or None


# A deep-stacked spot (e.g. 200bb facing an open or a 3-bet) carries an all-in
# option the solver never takes. Its EV is a huge negative number that is pure
# noise in the "EV of each action" panel AND squashes every other bar on the
# chart's shared scale. We drop the All-in EV row when BOTH hold: the solver
# effectively never jams here (freq ~0) AND the jam is clearly dominated (well
# below the best action). Both guards together mean we never hide an all-in that
# is a live, mixed, or near-even option -- at short stacks a jam IS a real line
# and carries real frequency, so it stays. Tune against reviewer feedback.
_ALLIN_DISPLAY_MAX_FREQ = 0.02          # jams below this are "never played"
_ALLIN_DISPLAY_MIN_DOMINATION_BB = 1.0  # ...and this far below the best action


def _drop_unreasonable_all_in(
    evs: dict[str, float], strategy: dict[str, float]
) -> dict[str, float]:
    """Filter an unreasonable all-in out of the per-action EV map (display only).

    The all-in is canonically labelled ``"All-in"`` (see
    :func:`pipeline.preflop.options.canonicalize_action_label`). Keeps it when
    the solver plays it with any real frequency, or when it isn't clearly
    dominated -- so a live short-stack jam is always shown. No-op when there's
    no all-in at the node.
    """
    all_in_ev = evs.get("All-in")
    if all_in_ev is None:
        return evs
    if strategy.get("All-in", 0.0) >= _ALLIN_DISPLAY_MAX_FREQ:
        return evs  # the solver actually jams here -- keep it
    if max(evs.values()) - all_in_ev < _ALLIN_DISPLAY_MIN_DOMINATION_BB:
        return evs  # near-even jam (short stacks) -- keep it
    return {label: ev for label, ev in evs.items() if label != "All-in"}


def _format_action_evs(
    evs: dict[str, float] | None,
    strategy: dict[str, float],
) -> str:
    """The ``action_ev_bb`` CSV value, e.g. ``"Call: +0.15, Fold: -1.00"``.

    Ordered by descending action frequency so it aligns column-for-column
    with ``action_frequencies``. Each EV is in bb to two decimals with an
    explicit sign. ``None`` (pack has no EVs) -> empty string. Shown for every
    spot, including near-pure ones (the real per-action EVs are the truth) --
    except an unreasonable deep-stack all-in, which is dropped as noise (see
    :func:`_drop_unreasonable_all_in`).
    """
    if not evs:
        return ""
    evs = _drop_unreasonable_all_in(evs, strategy)
    # EV DISPLAY NORMALIZATION (team, July 2026): different solve sources
    # measure EV from different zero points (monker packs from the hand
    # START, so a BB fold shows -1.00; Ryan-style packs from the decision,
    # so Fold shows 0.00; vendor .dbs from prior investment). Normalize the
    # DISPLAYED numbers so Fold always reads 0: every EV becomes "vs
    # folding", one convention everywhere. The differences between actions
    # (the decision signal) are unchanged by a constant shift; internal
    # math (difficulty, worthiness) never reads this column.
    fold_ev = evs.get("Fold")
    if fold_ev is not None:
        evs = {label: ev - fold_ev for label, ev in evs.items()}
    ordered = sorted(evs.keys(), key=lambda label: -strategy.get(label, 0.0))
    return ", ".join(f"{label}: {evs[label]:+.2f}" for label in ordered)


# --- pot reconstruction ------------------------------------------------------
def _compute_pot_bb(
    facts: PreflopFacts,
    pack: PreflopPack,
) -> float:
    """Reconstruct the pot in bb at hero's decision point.

    Thin wrapper over the shared pot accounting in
    :func:`pipeline.preflop.action_history.resolve_preflop_history` (the
    same walk the question-narrative renderer and the EV engine use, so
    the prose ``"opens to $1.25"`` and the POT column stay in sync).
    All-ins are approximated as the effective stack depth -- precise
    side-pot math would need stack tracking per position, deferred.

    Returns the pot in big blinds. Callers convert to dollars (for
    cash) or render directly as bb (for tournaments).
    """
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        resolve_preflop_history,
    )

    return resolve_preflop_history(facts.spot.node.history_before, pack).pot_bb


def _compute_preflop_skills(
    facts: PreflopFacts,
    *,
    game_format: str = "cash",
    stack_depth_bb: int = 100,
) -> str:
    """Comma-separated user-facing skills from pipeline.skill_tagger.

    Defensive: if the tagger import or computation fails, returns ''
    rather than crashing the batch. The skill column is metadata for
    the app -- a tagging failure shouldn't drop the whole question.
    """
    try:
        from pipeline.skill_tagger import (  # noqa: PLC0415
            compute_skills,
            from_preflop_facts,
        )

        ctx = from_preflop_facts(
            facts,
            game_format=game_format,
            stack_depth_bb=stack_depth_bb,
        )
        return ", ".join(compute_skills(ctx))
    except Exception:  # noqa: BLE001 - tagging never blocks a batch
        return ""


# --- raise-count math --------------------------------------------------------
def _count_prior_raises(history: tuple[ParsedAction, ...]) -> int:
    """How many raise/all-in actions appear in this prior-action history.

    Limps and folds don't count; only actions that put fresh money in at
    a new level (raise or all-in). Used to derive Preflop Pot Type and
    the raise verb in the narrative renderer.
    """
    return sum(
        1
        for a in history
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )


def _pot_type_for_facts(facts: PreflopFacts) -> str:
    """The Preflop Pot Type column value for these facts.

    Counts raises across the WHOLE node history -- both prior raises in
    ``history_before`` AND the raise hero is about to make (if dominant
    action is a raise). The CSV value labels the spot's pot type from
    the post-decision perspective, matching the postflop convention.
    """
    raises = _count_prior_raises(facts.spot.node.history_before)
    dominant = facts.spot.dominant_action
    if dominant.startswith("Raise") or dominant.startswith("AllIn"):
        raises += 1
    return _PREFLOP_POT_TYPE.get(raises, "Multi-raised pot")


def _pot_participant_for_facts(facts: PreflopFacts) -> str:
    """The Pot Participant column value (Heads-Up vs Multi-Way).

    Counts UNIQUE non-fold positions still in the pot (incl. hero) via
    the same set-dedup helper the ``multiway_pot`` concept tag uses, so
    the two never disagree. Counting raw actions instead double-counts
    hero's own earlier open when hero later faces a 3-bet (e.g. "HJ
    opens, SB 3-bets, HJ decides" is heads-up, not multi-way).
    """
    return "Heads-Up" if _non_fold_actor_count(facts) <= 2 else "Multi-Way"


def _position_matchup(facts: PreflopFacts) -> str:
    """The Position Matchup column value -- hero-vs-villain seats.

    ``"BB_vs_BTN"`` style (mirrors the postflop ``position_dynamic``),
    with a fallback to just hero's seat when there's no specific villain
    (open spots).
    """
    hero = facts.spot.node.actor
    if facts.villain_stats is None:
        return hero
    return f"{hero}_vs_{facts.villain_stats.position}"


def _relative_position(facts: PreflopFacts) -> str:
    """The Relative Position column value -- hero's IP/OOP standing.

    Thin wrapper over :func:`pipeline.preflop.position.hero_relative_position`
    (the shared single source so the CSV and the Layer 6 prompt agree).
    """
    return hero_relative_position(facts)


# A single position's preflop chart: the full action mix per hand at the
# node where that position acts. hand_class -> action_label -> metrics,
# where metrics is {"freq": <0-1>} plus {"to_bb": <size>} for raise/all-in.
# Pure-fold hands are omitted (a renderer treats a missing hand as 100%
# fold), which keeps the cell small. This replaced the old "presence"
# snapshot, which summed every action's weight and so collapsed to a
# useless all-1s mask for the hero (it carried no strategy). (June 2026.)
_PositionChart = dict[str, dict[str, dict[str, float]]]

# {(pack_id, root): {(actor, history_before): node}} so a villain's OWN
# decision node resolves in O(1) without re-walking the ~9k-node tree per
# row. The tree is deterministic per pack, so memoising for the process
# lifetime is safe.
_NODE_INDEX_CACHE: dict[
    tuple[str, str],
    dict[tuple[str, tuple[ParsedAction, ...]], PreflopDecisionNode],
] = {}


def _node_index(
    pack: PreflopPack,
) -> dict[tuple[str, tuple[ParsedAction, ...]], PreflopDecisionNode]:
    """All of the pack's decision nodes, keyed by ``(actor, history_before)``.

    Lets us recover any player's OWN decision node (where they faced the
    spot) from a later node's history -- needed to show each villain's full
    action mix at the moment it was on them, not just the action they took.
    """
    cache_key = (pack.pack_id, str(pack.root_path))
    cached = _NODE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from pipeline.preflop.node_enumerator import (  # noqa: PLC0415
        enumerate_nodes_by_actor,
    )

    index: dict[tuple[str, tuple[ParsedAction, ...]], PreflopDecisionNode] = {}
    for nodes in enumerate_nodes_by_actor([pack]).values():
        for n in nodes:
            index[(n.actor, n.history_before)] = n
    _NODE_INDEX_CACHE[cache_key] = index
    return index


def _villain_decision_node(
    hero_node: PreflopDecisionNode,
    villain_pos: str,
    pack: PreflopPack,
) -> PreflopDecisionNode | None:
    """The node where ``villain_pos`` made their last non-fold decision.

    That node carries the villain's FULL action menu (every option's range
    file), so we render their complete mix -- the chart for that seat at the
    moment they acted -- not just the single action they took. Returns
    ``None`` if it can't be resolved (defensive; the caller skips it).
    """
    last_index: int | None = None
    for i, a in enumerate(hero_node.history_before):
        if a.position == villain_pos and a.action_type is not PreflopActionType.FOLD:
            last_index = i
    if last_index is None:
        return None
    prefix = hero_node.history_before[:last_index]
    return _node_index(pack).get((villain_pos, prefix))


def _normalize_action(
    option: PreflopActionOption,
    actor: str,
    state: ResolvedPreflopState,
    pack: PreflopPack,
) -> tuple[str, float | None]:
    """Map a node action option to a ``(label, to_bb)`` pair.

    Labels are normalised verbs the app renders directly
    (fold/call/raise/allin); raises and jams carry their size in big blinds
    (matching the Question prose), folds and calls carry no size. ``state``
    is the resolved pot accounting at the node, which sizes the raise
    options (Monker packs derive the raise-to from the running pot).
    """
    action_type = option.action_type
    if action_type is PreflopActionType.FOLD:
        return "fold", None
    if action_type is PreflopActionType.CALL:
        return "call", None
    if action_type is PreflopActionType.ALL_IN:
        # A preflop jam commits the effective stack.
        return "allin", float(pack.stack_depth_bb)
    if action_type is PreflopActionType.RAISE:
        size = raise_to_bb(
            state,
            position=actor,
            raise_size_pct=option.raise_size_pct,
            pack=pack,
        )
        return "raise", round(size, 2)
    return action_type.value.lower(), None


def _action_mix_for_node(
    node: PreflopDecisionNode,
    pack: PreflopPack,
) -> _PositionChart:
    """The full per-hand action mix at ``node`` -- this position's preflop chart.

    For each hand class, the frequency of every action the solver offers
    here (fold / call / raise / all-in), with raise/jam sizes in bb. Hands
    that pure-fold are omitted (the renderer treats a missing hand as 100%
    fold), so a typical chart is ~30-60 hands rather than all 169.
    """
    state = resolve_preflop_history(node.history_before, pack)
    # Read each action's range file once, paired with its normalised label.
    loaded: list[tuple[str, float | None, dict[str, float]]] = [
        (
            *_normalize_action(option, node.actor, state, pack),
            parse_range_file(option.range_file.path),
        )
        for option in node.actions
    ]

    chart: _PositionChart = {}
    for hand in canonical_169_hand_classes():
        entry: dict[str, dict[str, float]] = {}
        for label, to_bb, weights in loaded:
            freq = weights.get(hand, 0.0)
            if freq <= 0.0:
                continue
            metrics = {"freq": round(freq, 4)}
            if to_bb is not None:
                metrics["to_bb"] = to_bb
            entry[label] = metrics
        # Omit hands that only ever fold -- the renderer defaults them to fold.
        if any(label != "fold" for label in entry):
            chart[hand] = entry
    return chart


# --- multiway range column (position-labeled, active players only) -----------
# Preflop seat order, for deterministic JSON key ordering. Positions not
# listed sort last (defensive; shouldn't happen for the Ryan pack).
_PREFLOP_SEAT_ORDER = ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB")


def _active_villain_actions(
    node: PreflopDecisionNode,
) -> dict[str, ParsedAction]:
    """Map each STILL-ACTIVE villain position to its last non-fold action.

    Active = put chips in (raise / call / all-in) and did NOT subsequently
    fold. Excludes hero, folded players, and seats that never acted (an
    action hasn't reached them yet). The value is that position's most
    recent non-fold action, which names their range file.
    """
    last_action: dict[str, ParsedAction] = {}
    for a in node.history_before:
        if a.position == node.actor:
            continue  # hero -- handled separately
        if a.action_type is PreflopActionType.FOLD:
            last_action.pop(a.position, None)  # folded -> drop
            continue
        last_action[a.position] = a
    return last_action


def _compute_active_ranges(
    facts: PreflopFacts,
    pack: PreflopPack,
) -> dict[str, _PositionChart]:
    """Full preflop chart for every still-active player, keyed by position.

    Hero's chart comes from their current decision node; each active
    villain's comes from the node where THEY last acted -- so it's the range
    + mix they had when it was on them, not just the one action they took. A
    villain whose node or range files can't be resolved is skipped
    defensively so a pack gap can't crash the row.
    """
    node = facts.spot.node
    ranges: dict[str, _PositionChart] = {
        node.actor: _action_mix_for_node(node, pack),
    }
    for pos in _active_villain_actions(node):
        villain_node = _villain_decision_node(node, pos, pack)
        if villain_node is None:
            continue
        try:
            ranges[pos] = _action_mix_for_node(villain_node, pack)
        except (OSError, ValueError):
            continue
    return ranges


def _render_active_ranges(facts: PreflopFacts, pack: PreflopPack) -> str:
    """The ``ranges`` CSV column: a JSON object mapping each active player's
    position to its full preflop chart, ordered by seat. Compact JSON (no
    spaces); empty string if nothing resolved.

    Shape: ``{"<POS>":{"<hand>":{"<action>":{"freq":f,"to_bb":b}}}}`` --
    action is fold/call/raise/allin; raise/allin carry a bb size; pure-fold
    hands are omitted. The app renders each hand cell as that action mix.
    """
    ranges = _compute_active_ranges(facts, pack)
    if not ranges:
        return ""
    ordered = sorted(
        ranges.items(),
        key=lambda kv: (
            _PREFLOP_SEAT_ORDER.index(kv[0])
            if kv[0] in _PREFLOP_SEAT_ORDER
            else len(_PREFLOP_SEAT_ORDER)
        ),
    )
    return json.dumps(dict(ordered), separators=(",", ":"))


# --- question-narrative renderer ---------------------------------------------
# Delegates to pipeline.preflop.action_history, which builds the brief-spec
# `hand` dict and calls pipeline.action_history.format_action_history. This
# fixes two failure modes in the previous local renderer:
#   1. Preflop folds were listed verbatim ("UTG folds. The Hijack folds. ...")
#      instead of being dropped per the brief's Fold Rule.
#   2. Raises had no dollar amounts ("The Button opens" instead of "The Button
#      opens to $1.25"), making it hard to size the action from the prose.
def format_preflop_question(
    facts: PreflopFacts,
    *,
    pack: PreflopPack,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
) -> str:
    """Deterministic action-history narrative for the Question column.

    Returns the brief-spec ``You're [POS] with [CARDS].\\n[action sequence]``
    block. Preflop folds are dropped (implied by absence, per the
    brief's Fold Rule); raises are rendered as ``"<actor> opens to
    $X"`` / ``"<actor> 3-bets to $Y"`` etc., with hero using the base
    verb (``"you open to $X"``) and villains using third-person.

    The Question deliberately does NOT include:

      * Stakes / table size / stack depth -- those live in the Context
        column. Duplicating reads as repetitive in the UI.
      * A trailing "What's your play?" prompt -- the UI implies the
        question by rendering an answer-options widget below the prose.

    Args:
        facts: The PreflopFacts.
        pack: Source PreflopPack (for open-size + table-size + sb ratio).
        stakes_bb_dollars: BB size in dollars. Used for raise-amount
            dollar conversion in the action history.
        live_or_online: Cosmetic; cash-only.
        game_format: "cash" or "tournament".

    Returns:
        Multi-line action-history string, ready for the Question column.
    """
    from pipeline.preflop.action_history import (
        format_preflop_action_history,
    )

    return format_preflop_action_history(
        facts,
        pack=pack,
        stakes_bb_dollars=stakes_bb_dollars,
        live_or_online=live_or_online,
        game_format=game_format,
        display_in_bb=display_in_bb,
    )


# Per-pack rake descriptor surfaced in the Context column so a learner (and
# reviewer) sees the rake that shaped these ranges -- rake materially changes
# preflop strategy (it's why cold-call ranges run tight). Keyed by pack_id; a
# pack not listed gets no rake suffix (e.g. tournaments). Add new packs here.
_PACK_RAKE_NOTES: dict[str, str] = {
    "ryan_preflop_tree_6max_100bb": "4% / 0.3bb cap",
    "monker_nlhe_9max_100bb": "10% / 3bb cap",
}


def _context_column(
    pack: PreflopPack,
    stakes_bb_dollars: float,
    game_format: str,
    live_or_online: str = "Online",
    display_in_bb: bool = False,
) -> str:
    """The Context column value -- short prose mirroring the postflop format.

    Shape: ``"<Online|Live> · <stakes> · Rake <r>"``. The venue leads so a
    beginner immediately sees the format; the rake (from
    :data:`_PACK_RAKE_NOTES`, keyed by pack_id) trails with a ``Rake`` label
    so it's unambiguous. Venue is omitted when not Online/Live (e.g. "Not
    specified"); rake is omitted for packs without a listed note.

    The table size is NOT shown (the dedicated Table Size column carries it).
    The **stack depth IS shown whenever it isn't the assumed 100bb** (June 2026):
    readers default to 100bb, so a 20bb / 30bb pack must say so even though the
    Default Stack column also carries it -- the Context is where the eye lands.
    100bb packs stay clean (no stack shown).

    Stakes appear only on cash + dollar display: ``"<stakes>"``. When the
    question renders in big blinds (``display_in_bb``) the stakes are dropped
    too (every amount the player sees is in bb, so "$1/$2" is noise), and a
    tournament has no stakes -- so the Context can be as short as
    ``"Live · 20bb · Rake 5% / 0.5bb cap"``.
    """
    parts: list[str] = []
    if live_or_online in ("Online", "Live"):
        parts.append(live_or_online)
    if game_format == "cash" and not display_in_bb:
        sb_dollars = round(stakes_bb_dollars * pack.sb_to_bb_ratio, 2)
        parts.append(f"{_dollars(sb_dollars)}/{_dollars(stakes_bb_dollars)}")
    # Stack depth when it isn't the assumed 100bb (in bb, stake-independent).
    if round(pack.stack_depth_bb) != 100:  # noqa: PLR2004
        parts.append(f"{pack.stack_depth_bb:g}bb")
    rake = _PACK_RAKE_NOTES.get(pack.pack_id, "")
    if rake:
        parts.append(f"Rake {rake}")
    return " · ".join(parts)


# --- the main row builder ----------------------------------------------------
def _is_pocket_pair_class(hand_class: str) -> bool:
    """True when hero's hand is a pocket pair (the only hand that flops a set).
    Gates set-mining wording in the exploit engine off suited connectors /
    suited aces, which are speculative but never flop sets. Uses the same
    playability source as the archetype's speculative-hand gate."""
    from pipeline.preflop.playability import hand_playability  # noqa: PLC0415

    play = hand_playability(hand_class)
    return bool(play.get("recognized") and play.get("pocket_pair"))


def _exploit_notes_for_facts(facts: PreflopFacts) -> str:
    """Deterministic exploitative adjustments for the CSV ``exploit_notes``
    column, computed from the archetype + this hand's facts (equity, domination
    vs villain's heaviest in-range classes, blockers). "" when the archetype
    has no exploit role. Same computation the admin exploit panel shows."""
    if facts.villain_stats is not None:
        # in_range_classes (the FULL weight-gated range), never a top-N
        # digest: any cap silently drops real dominators, so the exploit
        # clauses named A3s-A6s while AKo was missing (QC 2026-07-03) and
        # A8o vs a BTN open showed no dominators at all (QC 2026-07-04;
        # same fix as the data block's domination_vs_villain_range).
        dom = dominating_map(
            facts.spot.hero_hand_class,
            [c for c, _ in facts.villain_stats.in_range_classes],
        )
    else:
        dom = {"dominated_by": [], "you_dominate": [], "coinflips": []}
    notes = exploit_adjustments(
        facts.archetype,
        hero_equity=facts.hero_equity_vs_villain,
        dominated_by=dom["dominated_by"] or None,
        you_dominate=dom["you_dominate"] or None,
        # Most-blocked classes FIRST (same order as the blocker_combos cell):
        # the refinement clause names the first few, and naming 1-combo junk
        # ("J9s, JTs") instead of the 3-combo premiums (QQ, AQo) undercuts it.
        blockers=sorted(facts.blockers, key=lambda c: (-facts.blockers[c], c)) or None,
        # Raises hero faces (1 = a single open, 2 = a 3-bet, 3+ = a 4-bet+): names
        # hero's isolation re-raise correctly (a 4-bet vs a 3-bet) and flips the
        # fold_or_raise nit advice.
        facing_raise_level=_count_prior_raises(facts.spot.node.history_before),
        # Gates set-mining wording to actual pocket pairs (a suited connector /
        # suited ace on a speculative fold does not flop a set).
        is_pocket_pair=_is_pocket_pair_class(facts.spot.hero_hand_class),
    )
    return exploit_notes_to_json(notes)


# Notes provenance string for every preflop row.
_PREFLOP_NOTES = "Auto-generated by poker-pipeline (preflop path)."


def _node_reference(facts: PreflopFacts, pack: PreflopPack) -> str:
    """The machine node reference (the old ``solver_reference`` value).

    Now carried in the ``Notes`` column's ``Node:`` field; byte-identical to
    what solver_reference held so the Compare/Review parsers are unaffected.
    """
    return f"{pack.pack_id}/{facts.spot.node.actor}/{facts.spot.node.node_id}"


def _preflop_notes(facts: PreflopFacts, pack: PreflopPack) -> str:
    """The enriched ``Notes`` value: provenance + chart + situation + node."""
    situation = (
        f"{_position_matchup(facts).replace('_', ' ')}, "
        f"{_pot_type_for_facts(facts).lower()}, {facts.archetype}"
    )
    return build_notes(
        _PREFLOP_NOTES,
        chart=pack.pack_id,
        situation=situation,
        node_ref=_node_reference(facts, pack),
    )


def _preflop_animation_script(
    facts: PreflopFacts, pack: PreflopPack, *,
    stakes_bb_dollars: float, game_format: str,
) -> str:
    """The app's animation timeline for a standalone preflop question.

    The pack node's ``history_before`` already lists every prior action
    INCLUDING the folds (unlike the postflop solves' line-only summaries),
    so no fold synthesis is needed: map each parsed action onto the shared
    :mod:`pipeline.animation` walk with its resolved bb size and stop at
    hero's decision. Built from the same resolved sizes as the Question
    prose, so the two can never disagree."""
    from pipeline.animation import AnimationTable, walk_explicit_preflop  # noqa: PLC0415

    node = facts.spot.node
    resolved = resolve_preflop_history(node.history_before, pack)
    actions: list[tuple[str, str, float | None, bool]] = []
    for parsed, size in zip(node.history_before, resolved.sizes_bb):
        t = parsed.action_type
        if t is PreflopActionType.FOLD:
            actions.append((parsed.position, "fold", None, False))
        elif t is PreflopActionType.CALL:
            actions.append((parsed.position, "call", None, False))
        elif t is PreflopActionType.RAISE:
            actions.append((parsed.position, "raise", size, False))
        elif t is PreflopActionType.ALL_IN:
            # An all-in commits the effective stack (side-pot math deferred,
            # matching resolve_preflop_history's documented convention).
            actions.append(
                (parsed.position, "raise", size or pack.stack_depth_bb, True)
            )
        else:  # a BB check option etc.
            actions.append((parsed.position, "check", None, False))
    table = AnimationTable(
        table_size=pack.table_size,
        starting_stack_bb=float(pack.stack_depth_bb),
        bb_in_dollars=(
            float(stakes_bb_dollars) if game_format != "tournament" else None
        ),
    )
    walk_explicit_preflop(table, actions, decision_seat=node.actor)
    return table.render(node.actor)


def _preflop_chat_context(
    facts: PreflopFacts,
    explanation: GeneratedExplanation,
    pack: PreflopPack,
    difficulty: DifficultyResult,
    *,
    game_format: str,
    question_text: str,
    user_cards: str,
    neutral_list: list[str],
) -> str:
    """The per-question chatbot JSON blob (all deterministic facts). See
    :mod:`pipeline.chat_context`."""
    from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
        PREFLOP_ARCHETYPE_GUIDANCE,
    )

    strat = canonicalize_strategy(facts)
    evs = action_evs_bb(facts, pack) or {}
    strategy = [
        StrategyEntry(action=lbl, frequency_pct=fr * 100.0, ev_bb=evs.get(lbl))
        for lbl, fr in strat.items()
    ]
    key_facts: dict[str, object] = {}
    if facts.hero_equity_vs_villain is not None:
        key_facts["your_hand_equity_vs_villain_range_pct"] = round(
            facts.hero_equity_vs_villain * 100
        )
    if facts.hero_equity_vs_field is not None:
        key_facts["your_equity_vs_all_in_field_pct"] = round(
            facts.hero_equity_vs_field * 100
        )
    if facts.hero_range_equity_vs_villain is not None:
        key_facts["your_range_equity_vs_villain_pct"] = round(
            facts.hero_range_equity_vs_villain * 100
        )
    if facts.break_even_equity is not None:
        key_facts["break_even_equity_to_continue_pct"] = round(
            facts.break_even_equity * 100
        )
    if facts.blockers:
        key_facts["combos_you_block_of_villains_value"] = dict(facts.blockers)
    villain = None
    vs = facts.villain_stats
    if vs is not None:
        villain = {
            "seat": vs.position,
            "action": vs.action_label,
            "range_pct_of_all_hands": round(vs.pct_of_dealt_hands),
            "likely_hands": [hc for hc, _w in vs.most_common_combos],
        }
    skills_str = _compute_preflop_skills(
        facts, game_format=game_format, stack_depth_bb=pack.stack_depth_bb
    )
    return build_chat_context(
        pipeline="preflop",
        situation=question_text,
        hero_hand=user_cards,
        hand_summary=facts.spot.hero_hand_class,
        recommended_action=explanation.correct_answer,
        also_acceptable=neutral_list,
        full_strategy=strategy,
        key_facts=key_facts,
        villain=villain,
        strategic_frame=f"{facts.archetype}: "
        f"{PREFLOP_ARCHETYPE_GUIDANCE.get(facts.archetype, '')}",
        concept_tags=compute_concept_tags(facts),
        skills_tested=[s for s in skills_str.split(", ") if s],
        difficulty=difficulty.score,
        coaching_answer=explanation.answer_explanation,
    )


def build_preflop_row(
    facts: PreflopFacts,
    explanation: GeneratedExplanation,
    *,
    pack: PreflopPack,
    difficulty: DifficultyResult,
    number: int,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
    validation_status: str = "auto_approved",
    claim_check: str = "",
) -> dict[str, str]:
    """Turn one populated PreflopFacts + its LLM-written explanation into a CSV row.

    Args:
        facts: The Layer 5 data block for the spot.
        explanation: The six LLM-written CSV columns from Layer 6.
        pack: The source preflop pack (table size, stack depth, sb ratio).
        difficulty: The :class:`DifficultyResult` from
            :func:`pipeline.preflop.difficulty.compute_difficulty`. Its
            ``score`` populates the Difficulty Rating column; the
            per-axis breakdown populates the diagnostic columns
            (easy_freq, easy_ev, easy_concept, easy_hand, plus the
            bumps_applied audit column).
        number: The ``No`` column value -- per-batch auto-increment.
        stakes_bb_dollars: BB size in dollars. Default 0.50 = Tier 1.
        live_or_online: "Online" or "Live". Cosmetic.
        game_format: "cash" or "tournament".

    Returns:
        A dict keyed by ``PREFLOP_CSV_COLUMNS`` (the shared schema minus the
        columns the NLHE preflop path dropped in June 2026); every column is
        present.

    Raises:
        KeyError: never -- every PREFLOP_CSV_COLUMNS entry is filled (some with
            empty strings where the preflop path doesn't yet compute the
            field).
    """
    # The 7 "table-state" columns (User Seat / User Cards / Cards on Table
    # / Table Size / Default Stack / Seats / POT) in the Runout app's exact
    # poker-table format -- seat-stack-amount-action tokens the app renders
    # as chips. Built natively from structured facts (see
    # pipeline.preflop.app_table_format), reusing the same resolved dollar
    # amounts the Question prose uses so the two never disagree.
    table_cols = build_app_table_columns(
        facts,
        pack,
        stakes_bb_dollars=stakes_bb_dollars,
        live_or_online=live_or_online,
        game_format=game_format,
        display_in_bb=display_in_bb,
    )
    # Computed once, reused by their CSV column AND the chatbot context blob.
    question_text = format_preflop_question(
        facts,
        pack=pack,
        stakes_bb_dollars=stakes_bb_dollars,
        live_or_online=live_or_online,
        game_format=game_format,
        display_in_bb=display_in_bb,
    )
    neutral_list = neutral_credit_options(
        explanation.options(),
        explanation.correct_answer,
        canonicalize_strategy(facts),
    )

    return {
        "No": str(number),
        # UI rendering -- the app's table-state format.
        "User Seat": table_cols["user_seat"],
        "User Cards": table_cols["user_cards"],
        "Cards on Table": table_cols["cards_on_table"],
        "Table Size": table_cols["table_size"],
        "Default Stack": table_cols["default_stack"],
        "Seats": table_cols["seats"],
        "POT": table_cols["pot"],
        # Question content. Context + Question are deterministic; the
        # LLM (Layer 6) writes the four options + correct answer +
        # explanation.
        "Context": _context_column(
            pack, stakes_bb_dollars, game_format,
            live_or_online=live_or_online,
            display_in_bb=display_in_bb,
        ),
        "Question": question_text,
        "Question Type": "Hand scenario question",  # sentence case, no period (July 2026)
        "Hand Stage": "Preflop",
        "option 1": explanation.option_1,
        "option 2": explanation.option_2,
        "option 3": explanation.option_3,
        "option 4": explanation.option_4,
        "Correct Answer": explanation.correct_answer,
        # Deterministic neutral-credit options (the 20-point rule). Computed
        # from the SAME canonical strategy that built the options.
        "neutral_credit": format_neutral_credit(neutral_list),
        "Answer Explanation": explanation.answer_explanation,
        # Classification.
        "Cash/Tourney": _GAME_FORMAT_PROSE.get(game_format, game_format.capitalize()),
        "Live or Online": live_or_online,
        # Hero's IP/OOP standing; the seat matchup ("BB_vs_BTN") moves to
        # the "Position Matchup" column.
        "Relative Position": _relative_position(facts),
        "Preflop Pot Type": _pot_type_for_facts(facts),
        "Pot Participant": _pot_participant_for_facts(facts),
        "Stack Depth": _stack_depth_bucket(pack.stack_depth_bb),
        "Difficulty Rating": str(difficulty.score),
        "Position Matchup": _position_matchup(facts),
        # Notes carries provenance + chart + situation + node reference
        # (the old solver_reference value, now the Node: field). July 2026.
        "Notes": _preflop_notes(facts, pack),
        # Pipeline columns.
        # concept_tags: comma-separated list of firing preflop tags from
        # pipeline.preflop.concept_tags. Each tag is a boolean Python
        # function -- the LLM uses these in the SOLVER DATA block to
        # frame the explanation; Layer 7 validators check the prose
        # mentions tags it should.
        "concept_tags": ", ".join(compute_concept_tags(facts)),
        # board_texture: empty (no board preflop).
        "board_texture": "",
        # (ev_gap_bb column dropped June 2026 -- superseded by the per-action
        # action_ev_bb column. The gap is still computed internally for
        # difficulty + the worthiness gate; it's just no longer a CSV column.)
        # "auto_approved" unless the caller overrides -- the batch driver
        # passes "flagged" when a soft validator warned on this row (the
        # warnings themselves live in the meta sidecar's question record).
        "validation_status": validation_status,
        # Use canonical labels here too so the QA column reads as
        # "Raise: 70%, Fold: 30%" not "Raise 60%: 70%, Fold: 30%"
        # (the % token from Pio's internal labels would be misread as
        # an additional frequency).
        "action_frequencies": _format_action_frequencies(
            canonicalize_strategy(facts),
        ),
        # Range column: JSON {position: 169-class range} for every
        # still-active player (hero + non-folded actors). The app's range
        # UI reads this. (Replaced the heads-up-only ip_range/oop_range
        # pair, dropped May 2026.)
        "ranges": _render_active_ranges(facts, pack),
        # Phase 3: user-facing skill labels for the app's "study X"
        # features. Distinct from concept_tags (which carries the
        # computational atoms) -- skills is the user-readable mapping
        # over the 42-skill catalog defined in pipeline.skill_tagger.
        # Strict tagging: typically 2-4 skills per preflop spot.
        "skills": _compute_preflop_skills(
            facts, game_format=game_format, stack_depth_bb=pack.stack_depth_bb
        ),
        # Strategic archetype label (one of 16) from
        # pipeline.preflop.fact_extractor.classify_archetype. The LLM
        # gets this in its SOLVER DATA block as the strategic frame;
        # surfacing it in the CSV makes "show me all 3bet_as_bluff
        # spots" and similar analytics trivial.
        "archetype": facts.archetype,
        # (easy_freq / easy_ev / easy_concept / easy_hand difficulty-diagnostic
        # columns dropped June 2026 -- the Difficulty Rating itself stays; the
        # per-axis breakdown is computed internally but no longer exported.)
        # Decision math: the deterministic "Show the math" panel rows, as a
        # compact JSON the app renders (pipeline.preflop.stat_notes). The old
        # flat duplicate columns (pot_odds / hero_equity / range_equity /
        # blocker_combos / top_villain_combos) were dropped June 2026 -- their
        # values already live inside this JSON. Blank on opens (no villain).
        "stat_notes": stat_notes_to_json(build_stat_notes(
            facts,
            display_in_bb=display_in_bb,
            bb_in_dollars=stakes_bb_dollars,
        )),
        # Layer-7 claim-checker output (opt-in); "" when the checker didn't run.
        "claim_check": claim_check,
        # Deterministic exploitative adjustments (vs nit/station/maniac).
        "exploit_notes": _exploit_notes_for_facts(facts),
        # Per-action solver EV (bb) for the hero hand, ordered to match
        # action_frequencies. Populated only for packs that ship per-action
        # EVs (the Monker packs); blank for the EV-less Ryan pack.
        "action_ev_bb": _format_action_evs(
            action_evs_bb(facts, pack), canonicalize_strategy(facts)
        ),
        # Per-question chatbot context (all deterministic facts as one JSON blob).
        "animation_script": _preflop_animation_script(
            facts, pack, stakes_bb_dollars=stakes_bb_dollars,
            game_format=game_format,
        ),
        "chat_context": _preflop_chat_context(
            facts, explanation, pack, difficulty,
            game_format=game_format,
            question_text=question_text,
            user_cards=table_cols["user_cards"],
            neutral_list=neutral_list,
        ),
    }


# --- CSV writer --------------------------------------------------------------
def write_preflop_csv(
    path: Path | str,
    rows: Iterable[tuple[PreflopFacts, GeneratedExplanation, DifficultyResult]],
    *,
    pack: PreflopPack,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
    row_statuses: Iterable[str | None] | None = None,
    claim_checks: Iterable[str] | None = None,
) -> int:
    """Write preflop question rows to a CSV at ``path``; return rows written.

    ``rows`` is an iterable of ``(facts, explanation, difficulty)``
    triples where ``difficulty`` is the
    :class:`pipeline.preflop.difficulty.DifficultyResult` from the
    4-axis algorithm. The ``No`` column auto-increments from 1
    across the output. Parent directories are created if needed;
    the full CSV header is always written.

    Args:
        path: Output CSV path (Path or string). Parent created if missing.
        rows: Iterable of triples to write.
        pack: Source preflop pack -- forwarded to every row.
        stakes_bb_dollars: BB size in dollars. Default 0.50 = Tier 1.
        live_or_online: "Online" or "Live". Cosmetic.
        game_format: "cash" or "tournament".
        row_statuses: Optional per-row ``validation_status`` overrides,
            positionally aligned with ``rows``. ``None`` entries (and a
            ``None`` argument) keep the default ``auto_approved``. The
            batch driver passes ``"flagged"`` for rows a soft validator
            warned on.

    Returns:
        Number of rows written (excludes the header).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_list = list(row_statuses) if row_statuses is not None else []
    claim_list = list(claim_checks) if claim_checks is not None else []
    written = 0
    # utf-8-sig writes a BOM so Excel on Windows auto-detects UTF-8 instead
    # of falling back to cp1252 and mojibake-ing the suit emoji bytes.
    # Matches pipeline.format_writer.write_csv's encoding choice.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREFLOP_CSV_COLUMNS)
        writer.writeheader()
        for number, (facts, explanation, difficulty) in enumerate(rows, start=1):
            status = (
                status_list[number - 1]
                if number - 1 < len(status_list) and status_list[number - 1]
                else "auto_approved"
            )
            row: dict[str, Any] = build_preflop_row(
                facts,
                explanation,
                pack=pack,
                difficulty=difficulty,
                number=number,
                stakes_bb_dollars=stakes_bb_dollars,
                live_or_online=live_or_online,
                game_format=game_format,
                display_in_bb=display_in_bb,
                validation_status=status,
                claim_check=(
                    claim_list[number - 1]
                    if number - 1 < len(claim_list) else ""
                ),
            )
            writer.writerow(row)
            written += 1
    return written


__all__ = [
    "PREFLOP_CSV_COLUMNS",
    "action_evs_bb",
    "build_preflop_row",
    "format_preflop_question",
    "write_preflop_csv",
]
