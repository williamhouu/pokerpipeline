"""Layer 8 (postflop): write the question rows to CSV.

Builds a row dict per question in a column layout aligned with the team's
output format (the table-state columns the app renders, the question/options/
explanation, the classification block, and the postflop diagnostics), then
writes the CSV. Deterministic; no LLM.

The table-state columns (User Seat / User Cards / Cards on Table / Table Size /
Default Stack / Seats / POT) are the Runout app's exact chip/seat/board render
tokens, built by :mod:`pipeline.postflop.app_table_format` from the same
bb-denominated amounts the question prose uses. They render in big blinds
(``display_in_bb=True``) or in dollars (``solve.bb_in_dollars``).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from pipeline.chat_context import StrategyEntry, build_chat_context
from pipeline.explanation_generator import GeneratedExplanation
from pipeline.format_writer import CSV_COLUMNS
from pipeline.neutral_credit import format_neutral_credit, neutral_credit_options
from pipeline.postflop.action_history import build_context_line, format_question
from pipeline.provenance import build_notes
from pipeline.postflop.animation_script import build_postflop_animation_script
from pipeline.postflop.app_table_format import build_postflop_app_table_columns
from pipeline.postflop.difficulty import PostflopDifficulty
from pipeline.postflop.exploit import (
    exploit_adjustments_for_facts,
    exploit_notes_to_json,
)
from pipeline.postflop.stat_notes import build_stat_notes, stat_notes_to_json
from pipeline.postflop.facts import PostflopFacts
from pipeline.postflop.options import frequencies_for_options
from pipeline.postflop.range_export import build_active_ranges_json
from pipeline.postflop.skills import compute_postflop_skills
from pipeline.postflop.solve import PostflopSolve
from pipeline.postflop.spot_sampler import spot_action_evs_bb

# The postflop CSV schema, UNIFIED with the preflop schema (June 2026). The
# FIRST columns are byte-identical -- same names, same order -- to
# ``PREFLOP_CSV_COLUMNS``, so the app reads ONE column layout for both paths;
# postflop then appends its extra columns at the END (it carries more
# decision-math + the play-through tags than the preflop CSV keeps). A parity
# test (tests/test_postflop_pipeline.py) asserts the shared prefix stays aligned.
#
# The shared prefix = the shared ``CSV_COLUMNS`` minus exactly the columns the
# preflop writer drops. That drop set is MIRRORED here (not imported) so the
# postflop package does not depend on the preflop writer -- it stays
# self-contained; the parity test guards against the two drifting apart.
_SHARED_PREFIX_DROP: frozenset[str] = frozenset({
    "pot_odds", "hero_equity", "range_equity", "blocker_combos",
    "top_villain_combos", "ev_gap_bb",
    "easy_freq", "easy_ev", "easy_concept", "easy_hand",
})
_SHARED_PREFIX: tuple[str, ...] = tuple(
    c for c in CSV_COLUMNS if c not in _SHARED_PREFIX_DROP
)
# Postflop-only extras, appended AFTER the shared prefix:
#   * play-through tags  -- preflop has no linked preflop->river sequences. The
#     app GROUPS BY hand_id + ORDERS BY sequence_index; reading by header name,
#     so their position does not matter (moved from the front, June 2026).
#   * decision-math + difficulty diagnostics the preflop CSV drops but postflop
#     keeps (pot_odds / hero_equity / range_equity / easy_*), plus ``spr``
#     (postflop-only, not in the shared schema at all).
_POSTFLOP_EXTRA_COLUMNS: tuple[str, ...] = (
    "hand_id", "sequence_index", "sequence_total",
    "pot_odds", "hero_equity", "range_equity", "spr",
    "easy_freq", "easy_ev", "easy_concept", "easy_hand",
    # Full-hand play-throughs only (July 2026): the whole hand's difficulty
    # = the MAX over its legs' Difficulty Ratings (the hand demands what its
    # hardest decision demands; connective easy legs don't dilute it). Same
    # value on every leg of a hand so the app can select play-throughs for a
    # user's level with one filter. Blank on standalone rows.
    "hand_difficulty",
)
# The SUPERSET of every column a postflop row dict carries. Row builders
# (build_postflop_row, the preflop-entry + pack-leg adapters) always emit this
# full key set; each writer then trims via its ``columns`` list. CONTRACT
# (same as the PLO writer): keep building superset rows -- the batch layer,
# meta records, and tests read the dropped values even though the CSVs no
# longer carry them.
POSTFLOP_ROW_COLUMNS: tuple[str, ...] = _SHARED_PREFIX + _POSTFLOP_EXTRA_COLUMNS

# STANDALONE batches (July 2026, team request -- the last writer to get the
# declutter the NLHE preflop CSV got in June 2026 and the PLO + full-hand
# CSVs got in July): drop the trailing diagnostics (flat duplicates of
# numbers already inside the kept ``stat_notes`` JSON, which the admin math
# panel falls back to, or difficulty diagnostics) AND the play-through tag
# columns, which are always blank on standalone rows (hand_id /
# sequence_index / sequence_total / hand_difficulty are full-hand concepts).
# The 📏 bet-sizing trainer writes through this schema too.
_STANDALONE_DROPPED_COLUMNS: frozenset[str] = frozenset({
    "pot_odds", "hero_equity", "range_equity", "spr",
    "easy_freq", "easy_ev", "easy_concept", "easy_hand",
    "hand_id", "sequence_index", "sequence_total", "hand_difficulty",
})
POSTFLOP_CSV_COLUMNS: tuple[str, ...] = tuple(
    c for c in POSTFLOP_ROW_COLUMNS if c not in _STANDALONE_DROPPED_COLUMNS
)

# FULL-HAND (play-through) batches trim the trailing diagnostic columns
# (July 2026, team request): the dropped values are flat duplicates of
# numbers already inside the kept ``stat_notes`` JSON (which the admin math
# panel falls back to) or difficulty diagnostics. ``sequence_total`` is also
# dropped (team, July 2026): the app groups on ``hand_id`` and orders by
# ``sequence_index``, and a leg count is just the group size. ``hand_id`` /
# ``sequence_index`` and ``hand_difficulty`` (the app's hand-level selector)
# are KEPT -- the play-through columns are the one place the full-hand schema
# is WIDER than the standalone one.
_FULL_HAND_DROPPED_COLUMNS: frozenset[str] = frozenset({
    "pot_odds", "hero_equity", "range_equity", "spr",
    "easy_freq", "easy_ev", "easy_concept", "easy_hand",
    "sequence_total",
})
FULL_HAND_CSV_COLUMNS: tuple[str, ...] = tuple(
    c for c in POSTFLOP_ROW_COLUMNS if c not in _FULL_HAND_DROPPED_COLUMNS
)

_NOTES = "Auto-generated by poker-pipeline (postflop path)."

# Preflop pot type by raise count -- prose form, matching the preflop/shared
# writers so a "Three bet pot" reads identically whether the question is preflop
# or postflop. (1 raise = a single raised pot, the SRP baseline.)
_PREFLOP_POT_TYPE: dict[int, str] = {
    0: "Limped pot", 1: "Single raise pot",
    2: "Three bet pot", 3: "Four bet pot",
}


def _stack_depth_bucket(effective_stack_bb: float) -> str:
    """Prose Stack Depth bucket -- thresholds match the preflop/shared writers
    (< 40bb short, 40-150bb standard, > 150bb deep)."""
    if effective_stack_bb < 40:  # noqa: PLR2004
        return "Short stack"
    if effective_stack_bb <= 150:  # noqa: PLR2004
        return "Standard Stack"
    return "Deep stack"


def _emoji_cards(cards: Iterable[str]) -> str:
    from pipeline.action_history import format_card  # local: pure leaf

    return " ".join(format_card(c) for c in cards)


def _format_frequencies(action_frequencies: dict[str, float]) -> str:
    """'Bet 75%: 70%, Bet 33%: 20%, Check: 10%' (descending), skipping ~0."""
    items = sorted(action_frequencies.items(), key=lambda kv: -kv[1])
    return ", ".join(
        f"{label}: {freq * 100:.0f}%" for label, freq in items if freq >= 0.005
    )


# A deep-stacked node (e.g. the 200bb tree's direct flop all-in) carries a jam
# the solver never takes. Its EV is a big negative number that is pure noise in
# the "EV of each action" panel AND squashes every other bar on the chart's
# shared scale (team, July 2026: "makes it obvious nobody is going all-in 200bb
# deep"). Drop the All-in EV row only when BOTH hold: the solver effectively
# never jams (freq ~0) AND the jam is clearly dominated (well below the best
# action) -- so a live, mixed, or near-even jam (short stacks, river shoves) is
# always shown. MIRRORS pipeline.preflop.format_writer._drop_unreasonable_all_in
# (not imported: the postflop package stays self-contained).
_ALLIN_DISPLAY_MAX_FREQ = 0.02          # jams below this are "never played"
_ALLIN_DISPLAY_MIN_DOMINATION_BB = 1.0  # ...and this far below the best action


def _drop_unreasonable_all_in(
    evs: dict[str, float], action_frequencies: dict[str, float]
) -> dict[str, float]:
    """Filter an unreasonable all-in out of the per-action EV map (display
    only -- the solve, options, worthiness, and chat_context keep the full
    truth). No-op when there's no ``"All-in"`` action at the node."""
    all_in_ev = evs.get("All-in")
    if all_in_ev is None:
        return evs
    if action_frequencies.get("All-in", 0.0) >= _ALLIN_DISPLAY_MAX_FREQ:
        return evs  # the solver actually jams here -- keep it
    if max(evs.values()) - all_in_ev < _ALLIN_DISPLAY_MIN_DOMINATION_BB:
        return evs  # near-even jam -- keep it
    return {label: ev for label, ev in evs.items() if label != "All-in"}


def _format_action_evs(
    evs: dict[str, float] | None, action_frequencies: dict[str, float]
) -> str:
    """The ``action_ev_bb`` CSV value, e.g. ``"Check: +4.10, Bet 75%: +4.05"``.

    Ordered by descending action frequency so it lines up column-for-column
    with ``action_frequencies``; each EV is in bb to two decimals with an
    explicit sign. ``None`` (the solve exposes no per-action EVs) -> "".
    An unreasonable deep-stack all-in is dropped as display noise (see
    :func:`_drop_unreasonable_all_in`).
    """
    if not evs:
        return ""
    evs = _drop_unreasonable_all_in(evs, action_frequencies)
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
    ordered = sorted(evs, key=lambda label: -action_frequencies.get(label, 0.0))
    return ", ".join(f"{label}: {evs[label]:+.2f}" for label in ordered)


def _postflop_chat_context(
    facts: PostflopFacts,
    explanation: GeneratedExplanation,
    solve: PostflopSolve,
    difficulty: PostflopDifficulty,
    *,
    display_in_bb: bool,
    neutral: list[str],
) -> str:
    """The per-question chatbot JSON blob (all deterministic facts). See
    :mod:`pipeline.chat_context`."""
    from pipeline.postflop.explanation_generator import (  # noqa: PLC0415
        POSTFLOP_ARCHETYPE_GUIDANCE,
    )
    from pipeline.postflop.skills import compute_postflop_skills  # noqa: PLC0415

    evs = spot_action_evs_bb(facts.spot) or {}
    strategy = [
        StrategyEntry(action=label, frequency_pct=freq * 100.0, ev_bb=evs.get(label))
        for label, freq in facts.spot.action_frequencies.items()
    ]
    key_facts: dict[str, object] = {
        "hero_equity_vs_villain_pct": round(facts.hero_equity_vs_villain * 100),
        # Showdown-equity composition: what share of villain's range hero beats
        # RIGHT NOW (so the chatbot can explain "ahead of air" vs "drawing").
        "currently_ahead_of_villain_range_pct": round(facts.currently_ahead_pct * 100),
        "your_range_equity_pct": round(facts.hero_range_equity * 100),
        "range_advantage": facts.range_advantage,
        "nut_advantage": facts.nut_advantage,
        "spr": round(facts.spr, 1),
        "pot_bb": round(facts.pot_bb, 1),
        "board_texture": facts.board_texture.get("composite", ""),
    }
    if facts.break_even_equity is not None:
        key_facts["break_even_equity_pct"] = round(facts.break_even_equity * 100)
        key_facts["to_call_bb"] = round(facts.to_call_bb, 1)
    if facts.blocker_effect != "neutral":
        key_facts["blockers"] = (
            f"you remove ~{facts.blocked_value_pct * 100:.0f}% of villain's value "
            f"and ~{facts.blocked_bluff_pct * 100:.0f}% of their bluffs "
            f"(mainly block their {facts.blocker_effect})"
        )
    villain = {
        "seat": facts.villain_position,
        "strong_hands_pct": round(facts.villain_nut_share * 100),
        "has_range_edge": facts.range_advantage == "villain",
    }
    return build_chat_context(
        pipeline="postflop",
        situation=format_question(facts.spot, solve, display_in_bb=display_in_bb),
        hero_hand=_emoji_cards(facts.spot.hero_cards),
        hand_summary=f"{facts.made_hand.replace('_', ' ')} ({facts.strength_bucket})"
        + (f"; draws: {', '.join(facts.draws)}" if facts.has_strong_draw else ""),
        recommended_action=explanation.correct_answer,
        also_acceptable=neutral,
        full_strategy=strategy,
        key_facts=key_facts,
        villain=villain,
        strategic_frame=f"{facts.archetype}: "
        f"{POSTFLOP_ARCHETYPE_GUIDANCE.get(facts.archetype, '')}",
        concept_tags=facts.concept_tags,
        skills_tested=compute_postflop_skills(facts, game_format=solve.game_format),
        difficulty=difficulty.score,
        coaching_answer=explanation.answer_explanation,
    )


def build_postflop_row(
    facts: PostflopFacts,
    explanation: GeneratedExplanation,
    solve: PostflopSolve,
    difficulty: PostflopDifficulty,
    number: int,
    *,
    validation_status: str = "draft",
    display_in_bb: bool = True,
    hand_id: str = "",
    sequence_index: int | str = "",
    sequence_total: int | str = "",
) -> dict[str, str]:
    """Build one CSV row dict from all the layer outputs.

    Amounts in the Context / Question / POT / Default Stack render in big blinds
    (``display_in_bb=True``) or in dollars (``solve.bb_in_dollars``). The
    ``action_ev_bb`` / SPR / equity columns are unit-fixed and unaffected.

    ``hand_id`` / ``sequence_index`` / ``sequence_total`` tag this row as one
    leg of a play-through hand (Option B). They stay blank for a standalone
    question, so an ordinary batch is byte-identical aside from the new (empty)
    columns; the full-hand assembler fills them in.
    """
    node = facts.spot.node
    # The app's poker-table render tokens (seats + chips + board), so the CSV
    # feeds the chip/seat/board renderer directly. Built from the same resolved
    # amounts as the Question prose, so the two never disagree.
    table = build_postflop_app_table_columns(facts, solve, display_in_bb=display_in_bb)
    opts = (explanation.options() + ["", "", "", ""])[:4]
    pot_odds = (
        f"{facts.break_even_equity * 100:.0f}%"
        if facts.break_even_equity is not None
        else ""
    )
    # Neutral-credit options from the hand's own action mix (the 20-point rule);
    # computed once, reused by the neutral_credit column + the chatbot context.
    neutral_list = neutral_credit_options(
        explanation.options(),
        explanation.correct_answer,
        frequencies_for_options(facts.spot.action_frequencies, explanation.options()),
    )
    row = {
        "No": str(number),
        "hand_id": hand_id,
        "sequence_index": str(sequence_index),
        "sequence_total": str(sequence_total),
        # Stamped by the full-hand driver (max of the hand's legs); blank on
        # standalone rows and until the driver assigns it.
        "hand_difficulty": "",
        "Hand Stage": facts.street.capitalize(),
        "Context": build_context_line(solve, display_in_bb=display_in_bb),
        # App table-state tokens (User Seat / User Cards / Cards on Table /
        # Table Size / Default Stack / Seats / POT) -- see app_table_format.py.
        "User Seat": table["user_seat"],
        "User Cards": table["user_cards"],
        "Cards on Table": table["cards_on_table"],
        "Table Size": table["table_size"],
        "Default Stack": table["default_stack"],
        "Seats": table["seats"],
        "POT": table["pot"],
        "Question": format_question(facts.spot, solve, display_in_bb=display_in_bb),
        # Same fixed label as the preflop/PLO writers, no trailing period
        # (June 2026 -- was "Postflop Decision"; unified per team feedback).
        "Question Type": "Hand scenario question",  # sentence case, no period (July 2026)
        "option 1": opts[0],
        "option 2": opts[1],
        "option 3": opts[2],
        "option 4": opts[3],
        "Correct Answer": explanation.correct_answer,
        # frequencies_for_options sums collapsed verb families (a "Bet" option =
        # "Bet 33%" + "Bet 50%" + ...) so neutral credit matches the options even
        # when gto collapsed a multi-size spot to Check-vs-Bet.
        "neutral_credit": format_neutral_credit(neutral_list),
        "Answer Explanation": explanation.answer_explanation,
        "Cash/Tourney": "Tournament" if solve.game_format == "tournament" else "Cash",
        "Live or Online": solve.live_or_online,
        "Relative Position": "In Position" if facts.hero_in_position else "Out of Position",
        # Shared-schema classification columns, now populated for postflop too
        # (June 2026 unification) so the first 41 columns match the preflop CSV.
        "Preflop Pot Type": _PREFLOP_POT_TYPE.get(
            facts.preflop_raise_count, "Multi-raised pot"
        ),
        "Pot Participant": "Heads-Up" if facts.n_players <= 2 else "Multi-Way",  # noqa: PLR2004
        "Stack Depth": _stack_depth_bucket(solve.effective_stack_bb),
        "Position Matchup": f"{facts.hero_position}_vs_{facts.villain_position}",
        "Difficulty Rating": str(difficulty.score),
        "action_frequencies": _format_frequencies(facts.spot.action_frequencies),
        "action_ev_bb": _format_action_evs(
            spot_action_evs_bb(facts.spot), facts.spot.action_frequencies
        ),
        "hero_equity": f"{facts.hero_equity_vs_villain * 100:.0f}%",
        "range_equity": f"{facts.hero_range_equity * 100:.0f}%",
        "pot_odds": pot_odds,
        "spr": f"{facts.spr:.1f}",
        "stat_notes": stat_notes_to_json(build_stat_notes(
            facts,
            display_in_bb=display_in_bb,
            bb_in_dollars=solve.bb_in_dollars,
        )),
        # Deterministic exploitative adjustments (nit / station / maniac), keyed
        # on the postflop archetype + refined by this hand's equity / showdown
        # standing / draw / blockers. Same JSON shape as the preflop column, so
        # the admin exploit panel renders both. See pipeline.postflop.exploit.
        "exploit_notes": exploit_notes_to_json(exploit_adjustments_for_facts(facts)),
        "concept_tags": ", ".join(facts.concept_tags),
        "skills": ", ".join(
            compute_postflop_skills(facts, game_format=solve.game_format)
        ),
        "archetype": facts.archetype,
        "board_texture": facts.board_texture.get("composite", ""),
        # Both active players' current-street ranges (169-class JSON) for the
        # app's range UI -- accurate, from the node's reach-weighted ranges.
        "ranges": build_active_ranges_json(node, solve),
        "easy_freq": f"{difficulty.easy_freq:.2f}",
        "easy_ev": f"{difficulty.easy_ev:.2f}" if difficulty.easy_ev is not None else "",
        "easy_concept": f"{difficulty.easy_concept:.2f}",
        "easy_hand": f"{difficulty.easy_hand:.2f}",
        # Notes carries provenance + chart (the solve) + situation + node
        # reference (the old solver_reference value, now the Node: field, still
        # ".../<node_id>/<hero_combo>" so the postflop parsers are unaffected).
        # July 2026.
        "Notes": build_notes(
            _NOTES,
            chart=solve.source_reference,
            situation=(
                f"{facts.hero_position} vs {facts.villain_position}, "
                f"{_PREFLOP_POT_TYPE.get(facts.preflop_raise_count, 'multi-raised pot').lower()}, "
                f"{facts.archetype}, {facts.board_texture.get('composite', '')}"
            ),
            node_ref=f"{solve.source_reference}/{node.node_id}/{facts.spot.hero_combo}",
        ),
        "validation_status": validation_status,
        "chat_context": _postflop_chat_context(
            facts, explanation, solve, difficulty,
            display_in_bb=display_in_bb, neutral=neutral_list,
        ),
        # The app's animation timeline; see the schema note above.
        "animation_script": build_postflop_animation_script(node, solve),
        # "" = the Layer-7 claim checker did not run for this row. The batch
        # overwrites this with the checker's JSON when the opt-in pass runs.
        "claim_check": "",
    }
    return row


def write_postflop_csv(
    output_path: Path | str,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] = POSTFLOP_CSV_COLUMNS,
) -> int:
    """Write row dicts to ``output_path`` (UTF-8 BOM for Excel). Returns count.

    ``columns`` selects the emitted schema: the default full
    ``POSTFLOP_CSV_COLUMNS`` for standalone batches, or the trimmed
    ``FULL_HAND_CSV_COLUMNS`` for full-hand play-through batches. Row dicts
    always carry the full key set; the writer filters.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return len(rows)


__all__ = [
    "FULL_HAND_CSV_COLUMNS",
    "POSTFLOP_CSV_COLUMNS",
    "POSTFLOP_ROW_COLUMNS",
    "build_postflop_row",
    "write_postflop_csv",
]
