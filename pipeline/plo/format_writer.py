"""Layer 8 for PLO: assemble one CSV row (and write the CSV).

build_plo_row turns a populated PloFacts -- plus the deterministic options +
(eventually) the LLM explanation + the difficulty result -- into a dict keyed by
:data:`PLO_CSV_COLUMNS`. Ports :mod:`pipeline.preflop.format_writer`, with two
differences:

* **No ``ranges`` column, plus a ``hand_shape`` column.** PLO range *display*
  was dropped; ``PLO_CSV_COLUMNS`` is the shared schema minus ``ranges`` plus a
  PLO-only ``hand_shape`` descriptor right after ``archetype`` (40 columns).
* **Decoupled from Layer 6.** The four options + correct answer are passed in
  (computed by :mod:`pipeline.plo.options`); the explanation is a plain string
  defaulting to ``""`` -- so a full CSV row is producible today, before the LLM
  prose layer exists. When Layer 6 lands the batch just passes its text.

Every other column is computed from the facts the analytical layers already
produce (difficulty, concept tags, skills, archetype, ev_gap_bb), and the
table-state + Question columns reuse the resolved pot-limit amounts so the chip
tokens, the prose, and the pot never disagree.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pipeline.chat_context import StrategyEntry, build_chat_context
from pipeline.format_writer import CSV_COLUMNS
from pipeline.neutral_credit import format_neutral_credit, neutral_credit_options
from pipeline.plo.action_history import (
    display_seat,
    format_plo_action_history,
    format_plo_context,
)
from pipeline.plo.app_table_format import build_plo_app_table_columns
from pipeline.plo.difficulty import PloDifficultyResult
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.node_enumerator import plo_active_player_count
from pipeline.plo.options import canonicalize_strategy
from pipeline.plo.pack import PloActionType
from pipeline.plo.position import hero_relative_position
from pipeline.plo.skill_tagger import compute_plo_skills
from pipeline.plo.spot_tags import compute_plo_concept_tags


# The PLO schema = the shared template minus the dropped range-display column,
# plus a PLO-only `hand_shape` column (the descriptor: "double-suited rundown"
# etc.) inserted right after `archetype`. PLO suits/shape are load-bearing, so
# the shape is worth surfacing for QA/analytics (NLHE has no such descriptor).
def _plo_columns() -> tuple[str, ...]:
    cols = [c for c in CSV_COLUMNS if c != "ranges"]
    cols.insert(cols.index("archetype") + 1, "hand_shape")
    return tuple(cols)


PLO_CSV_COLUMNS: tuple[str, ...] = _plo_columns()

_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
_PREFLOP_POT_TYPE: dict[int, str] = {
    0: "Limped pot",
    1: "Single raise pot",
    2: "Three bet pot",
    3: "Four bet pot",
}
_GAME_FORMAT_PROSE: dict[str, str] = {"cash": "Cash", "tournament": "Tournament"}
_PERCENT = 100
_HEADS_UP_MAX = 2
_MAX_OPTIONS = 4


def _format_action_frequencies(strategy: dict[str, float]) -> str:
    """``{label: freq}`` -> ``"Call: 60%, Raise: 30%, Fold: 10%"``.

    Integer percents via the largest-remainder method so they sum to exactly
    100 (naive rounding shows totals like 99 or 101). Empty / all-zero -> "".
    """
    if not strategy or sum(strategy.values()) <= 0:
        return ""
    by_freq = sorted(strategy.items(), key=lambda kv: -kv[1])
    floors = [(label, int(v * _PERCENT), (v * _PERCENT) % 1) for label, v in by_freq]
    deficit = _PERCENT - sum(floor for _, floor, _ in floors)
    bumps = {
        i
        for i, _ in sorted(enumerate(floors), key=lambda kv: -kv[1][2])[: max(deficit, 0)]
    }
    return ", ".join(
        f"{label}: {floor + (1 if i in bumps else 0)}%"
        for i, (label, floor, _) in enumerate(floors)
    )


def _active_count(facts: PloFacts) -> int:
    """Players still in (last-action non-folders + hero), for Pot Participant."""
    return plo_active_player_count(facts.spot.node)


def _pot_type(facts: PloFacts) -> str:
    raises = sum(1 for a in facts.spot.node.history_before if a.action in _AGGRESSIVE)
    if facts.spot.dominant_action.startswith(("Raise", "Min-raise", "All-in")):
        raises += 1
    return _PREFLOP_POT_TYPE.get(raises, "Multi-raised pot")


def _position_matchup(facts: PloFacts) -> str:
    # App-facing seat codes (UTG/BTN), matching the NLHE matchup strings.
    hero = display_seat(facts.spot.node.actor)
    if facts.villain_stats is None:
        return hero
    return f"{hero}_vs_{display_seat(facts.villain_stats.seat)}"


def _solver_reference(facts: PloFacts, pack_label: str) -> str:
    return f"{pack_label}/{facts.spot.node.actor}/{facts.spot.node.node_id}"


def _plo_chat_context(
    facts: PloFacts,
    *,
    options: list[str],
    correct_answer: str,
    explanation: str,
    difficulty: PloDifficultyResult,
    question_text: str,
    user_cards: str,
    neutral_list: list[str],
) -> str:
    """The per-question chatbot JSON blob (all deterministic facts). Reuses PLO's
    rich generation block (:func:`build_solver_data`) for the PLO-specific facts
    (flush potential, nut ranking, card redundancy). See :mod:`pipeline.chat_context`."""
    from pipeline.plo.explanation_generator import build_solver_data  # noqa: PLC0415

    sd = build_solver_data(facts, options, correct_answer, include_skills=False)
    # Keys mapped to dedicated top-level fields -- everything else in the rich
    # generation block becomes key_facts (flush potential, redundancy, ev note,
    # equity, leaning examples, ...).
    _top = {
        "situation", "your_hand", "your_hand_shape", "your_position", "options",
        "correct_action", "action_strategy", "strategic_frame", "concept_tags",
        "villain",
    }
    key_facts = {k: v for k, v in sd.items() if k not in _top}
    strategy = [
        StrategyEntry(action=lbl, frequency_pct=fr * 100.0)
        for lbl, fr in canonicalize_strategy(facts).items()
    ]
    return build_chat_context(
        pipeline="plo",
        situation=question_text,
        hero_hand=user_cards,
        hand_summary=sd.get("your_hand_shape", facts.hand_class.descriptor),
        recommended_action=correct_answer,
        also_acceptable=neutral_list,
        full_strategy=strategy,
        key_facts=key_facts,
        villain=sd.get("villain"),
        strategic_frame=sd.get("strategic_frame", facts.archetype),
        concept_tags=compute_plo_concept_tags(facts),
        skills_tested=compute_plo_skills(facts),
        difficulty=difficulty.score,
        coaching_answer=explanation,
    )


def build_plo_row(
    facts: PloFacts,
    *,
    difficulty: PloDifficultyResult,
    options: list[str],
    correct_answer: str,
    explanation: str = "",
    number: int,
    pack_label: str = "plo_6max_100bb",
    stakes_bb_dollars: float = 1.0,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
    stack_bb: float = 100.0,
    validation_status: str = "draft",
) -> dict[str, str]:
    """Build one PLO CSV row (a dict keyed by :data:`PLO_CSV_COLUMNS`).

    ``options`` / ``correct_answer`` come from :func:`pipeline.plo.options.
    build_options`; ``explanation`` is the Layer 6 text (``""`` until it exists).
    Every column is filled.
    """
    table = build_plo_app_table_columns(
        facts,
        stakes_bb_dollars=stakes_bb_dollars,
        game_format=game_format,
        display_in_bb=display_in_bb,
        stack_bb=stack_bb,
    )
    opts = [*options[:_MAX_OPTIONS], *([""] * (_MAX_OPTIONS - len(options)))]
    ev_gap = facts.ev_gap_bb
    # Computed once, reused by their CSV column AND the chatbot context blob.
    question_text = format_plo_action_history(
        facts,
        stakes_bb_dollars=stakes_bb_dollars,
        game_format=game_format,
        display_in_bb=display_in_bb,
        stack_bb=stack_bb,
    )
    neutral_list = neutral_credit_options(
        options, correct_answer, canonicalize_strategy(facts)
    )

    row = {
        "No": str(number),
        "User Seat": table["user_seat"],
        "User Cards": table["user_cards"],
        "Cards on Table": table["cards_on_table"],
        "Table Size": table["table_size"],
        "Default Stack": table["default_stack"],
        "Seats": table["seats"],
        "POT": table["pot"],
        "Context": format_plo_context(
            stakes_bb_dollars=stakes_bb_dollars,
            stack_bb=stack_bb,
            game_format=game_format,
            display_in_bb=display_in_bb,
            live_or_online=live_or_online,
        ),
        "Question": question_text,
        "Question Type": "Hand scenario question",  # sentence case, no period (July 2026)
        "Hand Stage": "Preflop",
        "option 1": opts[0],
        "option 2": opts[1],
        "option 3": opts[2],
        "option 4": opts[3],
        "Correct Answer": correct_answer,
        # Deterministic neutral-credit options (the 20-point rule), computed
        # from the same canonical strategy that built the options.
        "neutral_credit": format_neutral_credit(neutral_list),
        "Answer Explanation": explanation,
        "Cash/Tourney": _GAME_FORMAT_PROSE.get(game_format, game_format.capitalize()),
        "Live or Online": live_or_online,
        "Relative Position": hero_relative_position(facts),
        "Preflop Pot Type": _pot_type(facts),
        "Pot Participant": "Heads-Up" if _active_count(facts) <= _HEADS_UP_MAX else "Multi-Way",
        "Stack Depth": "Standard Stack",
        "Difficulty Rating": str(difficulty.score),
        "skills": ", ".join(compute_plo_skills(facts)),
        "action_frequencies": _format_action_frequencies(canonicalize_strategy(facts)),
        "ev_gap_bb": f"{ev_gap:.2f}" if ev_gap is not None else "",
        "concept_tags": ", ".join(compute_plo_concept_tags(facts)),
        "Notes": "Auto-generated by poker-pipeline (PLO preflop path).",
        "Position Matchup": _position_matchup(facts),
        "archetype": facts.archetype,
        "hand_shape": facts.hand_class.descriptor,
        "board_texture": "",
        "solver_reference": _solver_reference(facts, pack_label),
        "validation_status": validation_status,
        "easy_freq": f"{difficulty.easy_freq:.3f}",
        "easy_ev": f"{difficulty.easy_ev:.3f}" if difficulty.ev_available else "",
        "easy_concept": f"{difficulty.easy_concept:.3f}",
        "easy_hand": f"{difficulty.easy_hand:.3f}",
        # Decision-math columns (NLHE-only for now): PLO's equity / blockers
        # are 4-card and don't map onto the NLHE stat_notes helpers, so these
        # are blank on PLO rows. The "Show the math" panel hides itself when
        # stat_notes is empty. Present here purely to satisfy the shared
        # schema (PLO_CSV_COLUMNS derives from CSV_COLUMNS).
        "pot_odds": "",
        "hero_equity": "",
        "range_equity": "",
        "blocker_combos": "",
        "top_villain_combos": "",
        "stat_notes": "",
        "claim_check": "",  # NLHE claim checker not wired for PLO yet
        "exploit_notes": "",  # NLHE exploit tagger not wired for PLO yet
        "action_ev_bb": "",  # per-action EV column not wired for PLO yet
        # Per-question chatbot context (all deterministic facts as one JSON blob),
        # reusing PLO's rich generation block for the PLO-specific facts.
        "chat_context": _plo_chat_context(
            facts,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            difficulty=difficulty,
            question_text=question_text,
            user_cards=table["user_cards"],
            neutral_list=neutral_list,
        ),
    }
    # Defensive: guarantee exact schema coverage (every column, no extras).
    missing = set(PLO_CSV_COLUMNS) - set(row)
    if missing:
        msg = f"build_plo_row missing columns: {sorted(missing)}"
        raise AssertionError(msg)
    return {col: row[col] for col in PLO_CSV_COLUMNS}


def write_plo_csv(rows: list[dict[str, str]], path: Path | str) -> int:
    """Write rows to ``path`` with the PLO header. Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLO_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


__all__ = ["PLO_CSV_COLUMNS", "build_plo_row", "write_plo_csv"]
