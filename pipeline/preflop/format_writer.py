"""Layer 8 (preflop edition): Format Writer for preflop spots.

The preflop sibling of ``pipeline.format_writer``. Same role: produce a CSV
row in the 38-column team template (``CSV_COLUMNS`` is the canonical schema
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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pipeline.explanation_generator import GeneratedExplanation
from pipeline.format_writer import CSV_COLUMNS
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.grammars.types import ParsedAction, PreflopActionType
from pipeline.preflop.pack import PreflopPack

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

# Card-suit to emoji. Mirrors action_history._SUIT_EMOJI. ``❤️`` uses U+2764
# (heavy heart) + U+FE0F variation selector; the other three use the standard
# suit codepoints + U+FE0F. Matches the team's voice in the Question column.
_SUIT_EMOJI: dict[str, str] = {
    "s": "♠️",
    "h": "❤️",
    "d": "♦️",
    "c": "♣️",
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


def _bb(value: float) -> str:
    """A big-blind count formatted as e.g. '100bb'."""
    return f"{round(value)}bb"


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

    Renders the dominant ``{action_label: freq}`` strategy as a comma-
    separated list of ``<label>: <integer>%`` entries, ordered by
    descending frequency. Mirrors the postflop helper.

    Empty strategy -> empty string (the column is best-effort).
    """
    if not strategy:
        return ""
    by_freq_desc = sorted(strategy.items(), key=lambda kv: -kv[1])
    parts = [f"{label}: {round(100 * freq)}%" for label, freq in by_freq_desc]
    return ", ".join(parts)


def _format_card(card: str) -> str:
    """One card as rank + suit emoji ('Th' -> 'T<heart>').

    Mirrors ``pipeline.action_history.format_card``. Lowercases the suit
    letter for emoji lookup so 'TH' / 'Th' / 'th' all work.
    """
    if len(card) != 2:
        raise ValueError(f"card must be 2 chars (rank+suit), got {card!r}")
    rank, suit = card[0], card[1].lower()
    if suit not in _SUIT_EMOJI:
        raise ValueError(f"unknown suit {suit!r} in {card!r}")
    return rank + _SUIT_EMOJI[suit]


def _split_combo(combo: str) -> tuple[str, str]:
    """Split a 4-char combo string ('AhKc') into its two 2-char cards."""
    if len(combo) != 4:
        raise ValueError(f"combo must be 4 chars, got {combo!r}")
    return combo[:2], combo[2:]


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

    Heuristic: count non-fold actors in ``history_before`` plus hero.
    If <=2 active actors (one villain + hero), it's heads-up. Otherwise
    multi-way. This is approximate -- seats still to act may also enter
    -- but matches what the postflop column does (which is also a
    snapshot at the decision point, not the final hand state).
    """
    active = sum(
        1
        for a in facts.spot.node.history_before
        if a.action_type is not PreflopActionType.FOLD
    )
    # +1 for hero (about to act).
    return "Heads-Up" if active + 1 <= 2 else "Multi-Way"


def _relative_position(facts: PreflopFacts) -> str:
    """The Relative Position column value.

    Postflop uses ``"BB_vs_BTN"`` (hero vs villain). Mirror that for
    preflop, with a fallback to just hero's seat when there's no
    specific villain (open spots).
    """
    hero = facts.spot.node.actor
    if facts.villain_stats is None:
        return hero
    return f"{hero}_vs_{facts.villain_stats.position}"


def _solver_reference(facts: PreflopFacts, pack: PreflopPack) -> str:
    """A descriptive pack-relative path identifying the preflop spot.

    Example: ``ryan_preflop_tree_6max_100bb/BB/UTG_Fold_HJ_Fold_CO_Fold_
    BTN_60%_SB_Fold_BB_Decision`` -- the actor folder + the action chain
    that led to the decision. Mirrors how the range files themselves are
    laid out on disk so a QA reviewer can find the source range
    directly. Pure prose; the path doesn't have to exist on disk.
    """
    return f"{pack.pack_id}/{facts.spot.node.actor}/{facts.spot.node.node_id}"


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
    )


def _context_column(
    pack: PreflopPack,
    stakes_bb_dollars: float,
    game_format: str,
) -> str:
    """The Context column value -- short prose mirroring the postflop format.

    Format: ``"<table_size>-Handed, <stakes>, Stacks <stack>"`` for cash;
    ``"<table_size>-Handed, <stack>bb stacks"`` for tournament. Matches the
    postflop ``ScenarioConfig.context`` shape so the column reads
    identically across both paths.
    """
    if game_format == "cash":
        sb_dollars = round(stakes_bb_dollars * pack.sb_to_bb_ratio, 2)
        stack_dollars = round(pack.stack_depth_bb * stakes_bb_dollars, 2)
        stakes_str = f"{_dollars(sb_dollars)}/{_dollars(stakes_bb_dollars)}"
        return (
            f"{pack.table_size}-Handed, {stakes_str}, Stacks {_dollars(stack_dollars)}"
        )
    return f"{pack.table_size}-Handed, {pack.stack_depth_bb}bb stacks"


# --- the main row builder ----------------------------------------------------
def build_preflop_row(
    facts: PreflopFacts,
    explanation: GeneratedExplanation,
    *,
    pack: PreflopPack,
    difficulty_score: int,
    number: int,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
) -> dict[str, str]:
    """Turn one populated PreflopFacts + its LLM-written explanation into a CSV row.

    Args:
        facts: The Layer 5 data block for the spot.
        explanation: The six LLM-written CSV columns from Layer 6.
        pack: The source preflop pack (table size, stack depth, sb ratio).
        difficulty_score: Layer 4's difficulty rating (500-3000).
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
    node = spot.node

    card_a, card_b = _split_combo(spot.hero_card_combo)
    user_cards = _format_card(card_a) + " " + _format_card(card_b)

    # Stack in dollars/bb derived from pack + stakes config.
    if game_format == "cash":
        default_stack = _dollars(pack.stack_depth_bb * stakes_bb_dollars)
    else:
        default_stack = _bb(pack.stack_depth_bb)

    return {
        "No": str(number),
        # UI rendering. Cards on Table is empty for preflop (no board).
        "User Seat": node.actor,
        "User Cards": user_cards,
        "Cards on Table": "",
        "Table Size": str(pack.table_size),
        "Default Stack": default_stack,
        # Seats / POT: the postflop renderer carries a richer
        # per-villain stack/commitment view. Preflop pot tracking from
        # %-of-pot raise tokens needs the pack's sizing conventions
        # decoded (see docs/ryan_range_pack_index.md). Deferred to a
        # follow-up so step 8 stays focused.
        "Seats": (facts.villain_stats.position if facts.villain_stats else ""),
        "POT": "",  # TODO(phase-b): reconstruct pot from raise %-tokens.
        # Question content. Context + Question are deterministic; the
        # LLM (Layer 6) writes the four options + correct answer +
        # explanation.
        "Context": _context_column(pack, stakes_bb_dollars, game_format),
        "Question": format_preflop_question(
            facts,
            pack=pack,
            stakes_bb_dollars=stakes_bb_dollars,
            live_or_online=live_or_online,
            game_format=game_format,
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
        "Relative Position": _relative_position(facts),
        "Preflop Pot Type": _pot_type_for_facts(facts),
        "Pot Participant": _pot_participant_for_facts(facts),
        "Stack Depth": _stack_depth_bucket(pack.stack_depth_bb),
        "Difficulty Rating": str(difficulty_score),
        # Skill tags -- Phase 3 taxonomy, not yet mapped (same as postflop).
        "tag_1": "",
        "tag_2": "",
        "tag_3": "",
        "Notes": "Auto-generated by poker-pipeline (preflop path).",
        # Pipeline columns.
        # concept_tags: empty for preflop -- the preflop concept tagger is
        # deferred to Phase B (see pipeline.preflop.question_extractor's
        # module docstring for why).
        "concept_tags": "",
        # hand_class: the 169-class label (e.g. "AKo"), not the postflop
        # made-hand + draw breakdown.
        "hand_class": spot.hero_hand_class,
        # board_texture: empty (no board preflop).
        "board_texture": "",
        "solver_reference": _solver_reference(facts, pack),
        # ev_gap_bb: empty -- preflop range files don't carry per-action
        # EVs. Phase B will add an equity-driven EV engine.
        "ev_gap_bb": "",
        "validation_status": "auto_approved",
        "action_frequencies": _format_action_frequencies(spot.action_frequencies),
        # ip_range / oop_range: 169-class snapshots at the preflop node.
        # Computing them requires summing hero's per-action ranges and
        # parsing villain's range file into a 169-class dict; the
        # building blocks exist in pipeline.preflop.fact_extractor but
        # the snapshotting helper isn't wired yet. Deferred to a
        # follow-up turn; empty for now so the CSV stays valid.
        "ip_range": "",
        "oop_range": "",
    }


# --- CSV writer --------------------------------------------------------------
def write_preflop_csv(
    path: Path | str,
    rows: Iterable[tuple[PreflopFacts, GeneratedExplanation, int]],
    *,
    pack: PreflopPack,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
) -> int:
    """Write preflop question rows to a CSV at ``path``; return rows written.

    ``rows`` is an iterable of ``(facts, explanation, difficulty_score)``
    triples. The ``No`` column auto-increments from 1 across the output.
    Parent directories are created if needed; the 38-column header is
    always written.

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
                difficulty_score=difficulty,
                number=number,
                stakes_bb_dollars=stakes_bb_dollars,
                live_or_online=live_or_online,
                game_format=game_format,
            )
            writer.writerow(row)
            written += 1
    return written


__all__ = [
    "build_preflop_row",
    "format_preflop_question",
    "write_preflop_csv",
]
