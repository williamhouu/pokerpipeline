"""Layer 8 (preflop edition): Format Writer for preflop spots.

The preflop sibling of ``pipeline.format_writer``. Same role: produce a CSV
row in the 44-column team template (``CSV_COLUMNS`` is the canonical schema
defined in the postflop module and reused verbatim here) from a populated
``PreflopFacts`` + a ``GeneratedExplanation`` + the source ``PreflopPack``.

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

from pipeline.explanation_generator import GeneratedExplanation
from pipeline.format_writer import CSV_COLUMNS
from pipeline.preflop.app_table_format import build_app_table_columns
from pipeline.preflop.concept_tags import (
    _non_fold_actor_count,
    compute_concept_tags,
)
from pipeline.preflop.difficulty import DifficultyResult
from pipeline.preflop.fact_extractor import (
    PreflopFacts,
    construct_villain_range_path,
)
from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType
from pipeline.preflop.node_enumerator import PreflopDecisionNode
from pipeline.preflop.options import canonicalize_strategy
from pipeline.preflop.pack import PreflopPack
from pipeline.preflop.position import hero_relative_position
from pipeline.preflop_ranges import (
    canonical_169_hand_classes,
    format_hand_class_range,
    parse_range_file,
)

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

    Empty strategy -> empty string. All-zero strategy (hand doesn't
    reach the node) -> empty string too, since 0% × N doesn't sum to
    100 and showing "Fold: 0%, Call: 0%" is misleading.
    """
    if not strategy:
        return ""
    total = sum(strategy.values())
    if total <= 0:
        # All-zero strategy. Nothing to render -- the column is
        # best-effort and a "0%" row carries no information.
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


# --- pot reconstruction ------------------------------------------------------
def _compute_pot_bb(
    facts: PreflopFacts,
    pack: PreflopPack,
) -> float:
    """Reconstruct the pot in bb at hero's decision point.

    Walks ``facts.spot.node.history_before`` maintaining each position's
    committed chips. The pot at any point is the sum of all committed
    amounts. Starts with SB and BB posting their blinds.

    For raise actions, the raise size in bb comes from
    :func:`pipeline.preflop.action_history._raise_size_bb` (same lookup
    table the question-narrative renderer uses, so the prose
    ``"opens to $1.25"`` and the POT column stay in sync). For all-ins
    we approximate the shove as the effective stack depth -- precise
    side-pot math would need stack tracking per position, deferred.

    Returns the pot in big blinds. Callers convert to dollars (for
    cash) or render directly as bb (for tournaments).
    """
    from pipeline.preflop.action_history import _raise_size_bb  # noqa: PLC0415

    committed: dict[str, float] = {"SB": pack.sb_to_bb_ratio, "BB": 1.0}
    current_max_bet = 1.0  # everyone has to match the BB to enter
    raise_level = 0
    for parsed in facts.spot.node.history_before:
        pos = parsed.position
        if parsed.action_type is PreflopActionType.FOLD:
            # Fold doesn't add chips. Any chips already committed (SB / BB
            # blinds, or anyone who raised then we later see them fold,
            # though that shouldn't appear in history_before) stay in.
            continue
        if parsed.action_type is PreflopActionType.CALL:
            # Caller matches the current bet level.
            committed[pos] = current_max_bet
            continue
        if parsed.action_type is PreflopActionType.RAISE:
            raise_level += 1
            bb_size = _raise_size_bb(parsed, raise_level, pack)
            committed[pos] = bb_size
            current_max_bet = bb_size
            continue
        if parsed.action_type is PreflopActionType.ALL_IN:
            raise_level += 1
            # Approximation: shove to effective stack. Precise side-pot
            # math would need per-position stack tracking.
            committed[pos] = float(pack.stack_depth_bb)
            current_max_bet = float(pack.stack_depth_bb)
            continue
    return sum(committed.values())


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


def _render_ev_gap(facts: PreflopFacts, pack: PreflopPack) -> str:
    """The ev_gap_bb column value. Two decimals when computed; empty
    string when the v1 EV engine can't compute reliably (raise actions
    in the top-2, no equity data, etc -- see ev_engine module docstring)."""
    from pipeline.preflop.ev_engine import compute_ev_gap_bb  # noqa: PLC0415

    gap = compute_ev_gap_bb(facts, pack)
    if gap is None:
        return ""
    return f"{gap:.2f}"


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


def _solver_reference(facts: PreflopFacts, pack: PreflopPack) -> str:
    """A descriptive pack-relative path identifying the preflop spot.

    Example: ``ryan_preflop_tree_6max_100bb/BB/UTG_Fold_HJ_Fold_CO_Fold_
    BTN_60%_SB_Fold_BB_Decision`` -- the actor folder + the action chain
    that led to the decision. Mirrors how the range files themselves are
    laid out on disk so a QA reviewer can find the source range
    directly. Pure prose; the path doesn't have to exist on disk.
    """
    return f"{pack.pack_id}/{facts.spot.node.actor}/{facts.spot.node.node_id}"


def _compute_hero_range_snapshot(
    node: PreflopDecisionNode,
) -> dict[str, float]:
    """Hero's 169-class range at the decision node.

    Sums the per-hand-class weights across all of hero's action range
    files. Each cell is the hand class's PRESENCE at this node (how
    often it reaches here from the parent decisions) -- which is what
    a range-grid UI shows. Range cells in [0, 1]; cells with zero
    weight (hand class never reaches the node) stay at 0.

    Returns a complete 169-entry dict so the renderer downstream can
    rely on canonical-order iteration.
    """
    out: dict[str, float] = {cls: 0.0 for cls in canonical_169_hand_classes()}
    for opt in node.actions:
        weights = parse_range_file(opt.range_file.path)
        for cls in canonical_169_hand_classes():
            out[cls] += weights.get(cls, 0.0)
    return out


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
) -> dict[str, dict[str, float]]:
    """169-class range for every still-active player, keyed by position.

    Always includes hero (their range entering this decision). Each active
    villain's range is read from the file naming their last non-fold action;
    a missing file is skipped defensively so a pack gap can't crash the row.
    """
    node = facts.spot.node
    ranges: dict[str, dict[str, float]] = {
        node.actor: _compute_hero_range_snapshot(node),
    }
    for pos, action in _active_villain_actions(node).items():
        try:
            path = construct_villain_range_path(node, action, pack)
        except (ValueError, KeyError):
            continue
        if not path.is_file():
            continue
        ranges[pos] = parse_range_file(path)
    return ranges


def _render_active_ranges(facts: PreflopFacts, pack: PreflopPack) -> str:
    """The ``ranges`` CSV column: a JSON object mapping each active player's
    position to its 169-class range string, ordered by seat. Compact JSON
    (no spaces) so the cell stays small; empty string if nothing resolved.
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
    serialized = {pos: format_hand_class_range(r) for pos, r in ordered}
    return json.dumps(serialized, separators=(",", ":"))


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
}


def _context_column(
    pack: PreflopPack,
    stakes_bb_dollars: float,
    game_format: str,
    live_or_online: str = "Online",
) -> str:
    """The Context column value -- short prose mirroring the postflop format.

    Shape: ``"<Online|Live> · <core> · Rake <r>"``. The venue leads so a
    beginner immediately sees the format; the rake (from
    :data:`_PACK_RAKE_NOTES`, keyed by pack_id) trails with a ``Rake`` label
    so it's unambiguous. Venue is omitted when not Online/Live (e.g. "Not
    specified"); rake is omitted for packs without a listed note.

    Core (cash): ``"<n>-Handed, <stakes>"``; tournament: ``"<n>-Handed"``.
    The stack size is intentionally NOT shown here -- the dedicated
    ``Default Stack`` column already carries it, so repeating it in the
    Context was redundant (dropped June 2026 per Zach's feedback). This
    also makes the Context independent of the ``display_in_bb`` toggle.
    """
    if game_format == "cash":
        sb_dollars = round(stakes_bb_dollars * pack.sb_to_bb_ratio, 2)
        stakes_str = f"{_dollars(sb_dollars)}/{_dollars(stakes_bb_dollars)}"
        core = f"{pack.table_size}-Handed, {stakes_str}"
    else:
        core = f"{pack.table_size}-Handed"

    parts: list[str] = []
    if live_or_online in ("Online", "Live"):
        parts.append(live_or_online)
    parts.append(core)
    rake = _PACK_RAKE_NOTES.get(pack.pack_id, "")
    if rake:
        parts.append(f"Rake {rake}")
    return " · ".join(parts)


# --- the main row builder ----------------------------------------------------
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
        A dict keyed by ``CSV_COLUMNS``; every column is present.

    Raises:
        KeyError: never -- every CSV_COLUMNS entry is filled (some with
            empty strings where the preflop path doesn't yet compute the
            field).
    """
    spot = facts.spot

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
        ),
        "Question": format_preflop_question(
            facts,
            pack=pack,
            stakes_bb_dollars=stakes_bb_dollars,
            live_or_online=live_or_online,
            game_format=game_format,
            display_in_bb=display_in_bb,
        ),
        "Question Type": "Hand Scenario Question.",
        "Hand Stage": "Preflop",
        "option 1": explanation.option_1,
        "option 2": explanation.option_2,
        "option 3": explanation.option_3,
        "option 4": explanation.option_4,
        "Correct Answer": explanation.correct_answer,
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
        "Notes": "Auto-generated by poker-pipeline (preflop path).",
        # Pipeline columns.
        # concept_tags: comma-separated list of firing preflop tags from
        # pipeline.preflop.concept_tags. Each tag is a boolean Python
        # function -- the LLM uses these in the SOLVER DATA block to
        # frame the explanation; Layer 7 validators check the prose
        # mentions tags it should.
        "concept_tags": ", ".join(compute_concept_tags(facts)),
        # hand_class: the 169-class label (e.g. "AKo"), not the postflop
        # made-hand + draw breakdown.
        "hand_class": spot.hero_hand_class,
        # board_texture: empty (no board preflop).
        "board_texture": "",
        "solver_reference": _solver_reference(facts, pack),
        # ev_gap_bb: gap between dominant and 2nd-most-frequent action,
        # in bb. Computed by pipeline.preflop.ev_engine for call/fold
        # spots; empty for spots where the top-2 actions involve a
        # raise (the v1 engine doesn't model raise EVs -- see the
        # engine module docstring for the scope rationale).
        "ev_gap_bb": _render_ev_gap(facts, pack),
        "validation_status": "auto_approved",
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
        # Diagnostic columns: per-axis breakdown of the difficulty
        # score (May 2026). Each axis is in [0, 1] where 1 = "easy on
        # this dimension". The Difficulty Rating column is computed
        # from a weighted sum of these four. Surfacing them lets the
        # reviewer see WHY a spot got a particular rating without
        # re-running the algorithm. ``easy_ev`` is empty when the EV
        # engine couldn't score the spot (raise-involved spots in v1
        # -- see pipeline.preflop.ev_engine).
        "easy_freq": f"{difficulty.easy_freq:.3f}",
        "easy_ev": (
            f"{difficulty.easy_ev:.3f}" if difficulty.ev_available else ""
        ),
        "easy_concept": f"{difficulty.easy_concept:.3f}",
        "easy_hand": f"{difficulty.easy_hand:.3f}",
        # Names of any BUMP_RULES that fired for this spot. Currently
        # the table is empty so this is always ''; populated as bumps
        # are added in pipeline.preflop.difficulty.BUMP_RULES.
        "difficulty_bumps": ", ".join(difficulty.bumps_applied),
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

    Returns:
        Number of rows written (excludes the header).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # utf-8-sig writes a BOM so Excel on Windows auto-detects UTF-8 instead
    # of falling back to cp1252 and mojibake-ing the suit emoji bytes.
    # Matches pipeline.format_writer.write_csv's encoding choice.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for number, (facts, explanation, difficulty) in enumerate(rows, start=1):
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
            )
            writer.writerow(row)
            written += 1
    return written


__all__ = [
    "build_preflop_row",
    "format_preflop_question",
    "write_preflop_csv",
]
