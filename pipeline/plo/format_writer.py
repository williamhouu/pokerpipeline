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
import json
from pathlib import Path

from pipeline.chat_context import StrategyEntry, build_chat_context
from pipeline.format_writer import CSV_COLUMNS
from pipeline.neutral_credit import format_neutral_credit, neutral_credit_options
from pipeline.provenance import build_notes
from pipeline.plo.action_history import (
    call_price,
    display_seat,
    format_plo_action_history,
    format_plo_context,
    price_is_live,
    resolve_pot_limit,
)
from pipeline.plo.app_table_format import build_plo_app_table_columns
from pipeline.plo.difficulty import PloDifficultyResult
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.hand_model import (
    dead_weight_stat_entries,
    flush_ceiling_stat_entries,
)
from pipeline.plo.node_enumerator import plo_active_player_count
from pipeline.plo.options import (
    _hero_raise_level,
    canonicalize_action_label,
    canonicalize_strategy,
    is_check_spot,
)
from pipeline.plo.pack import PloActionType, PloPack
from pipeline.plo.position import hero_relative_position
from pipeline.plo.range_breakdown import build_range_breakdown
from pipeline.plo.skill_tagger import compute_plo_skills
from pipeline.plo.spot_tags import compute_plo_concept_tags


# Columns the PLO CSV drops from the shared template (July 16 2026, team
# ask -- the same declutter the NLHE preflop writer did in June). The
# decision-math values still live INSIDE the kept `stat_notes` JSON (the
# admin equity bar + Show-the-math panel read the columns first and fall
# back to stat_notes), and the per-question lifecycle status moved to the
# meta sidecar record (`validation_status` there), which is what the Review
# flags render from anyway. blocker_combos / top_villain_combos were always
# blank on PLO rows (2-card-class helpers).
_PLO_DROPPED_COLUMNS: tuple[str, ...] = (
    "validation_status",
    "easy_freq",
    "easy_ev",
    "easy_concept",
    "easy_hand",
    "pot_odds",
    "hero_equity",
    "range_equity",
    "blocker_combos",
    "top_villain_combos",
)


# The PLO schema = the shared template minus the dropped range-display column
# and the July-16 declutter list above, plus a PLO-only `hand_shape` column
# (the descriptor: "double-suited rundown" etc.) inserted right after
# `archetype`. PLO suits/shape are load-bearing, so the shape is worth
# surfacing for QA/analytics (NLHE has no such descriptor).
def _plo_columns() -> tuple[str, ...]:
    cols = [
        c for c in CSV_COLUMNS
        if c != "ranges" and c not in _PLO_DROPPED_COLUMNS
    ]
    cols.insert(cols.index("archetype") + 1, "hand_shape")
    # The hand-type range breakdown (hero + every still-active opponent, by
    # shape) sits right after animation_script -- the team's requested slot
    # for this "GTO Wizard Categories" equivalent (July 2026). PLO-only:
    # the pair x suit hand-type axes are a PLO construct (NLHE has the grid).
    cols.insert(cols.index("animation_script") + 1, "range_breakdown")
    return tuple(cols)


PLO_CSV_COLUMNS: tuple[str, ...] = _plo_columns()

_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
_SB_PER_BB = 2.0  # pack EVs are in small blinds; the CSV speaks big blinds
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
    ts = facts.spot.node.table_size
    hero = display_seat(facts.spot.node.actor, table_size=ts)
    if facts.villain_stats is None:
        return hero
    return f"{hero}_vs_{display_seat(facts.villain_stats.seat, table_size=ts)}"


def _node_reference(facts: PloFacts, pack_label: str) -> str:
    """The machine node reference (the old ``solver_reference`` value).

    Now carried in the ``Notes`` column's ``Node:`` field instead of its own
    column; the string is byte-identical to what solver_reference held, so the
    Compare/Review consumers that parse it are unaffected.
    """
    return f"{pack_label}/{facts.spot.node.actor}/{facts.spot.node.node_id}"


def _plo_notes(facts: PloFacts, pack_label: str) -> str:
    """The enriched ``Notes`` value: provenance + chart + situation + node."""
    situation = (
        f"{_position_matchup(facts).replace('_', ' ')}, "
        f"{_pot_type(facts).lower()}, {facts.archetype}"
    )
    return build_notes(
        "Auto-generated by poker-pipeline (PLO preflop path).",
        chart=pack_label,
        situation=situation,
        node_ref=_node_reference(facts, pack_label),
    )


def _plo_animation_script(
    facts: PloFacts, *, stack_bb: float, stakes_bb_dollars: float,
    game_format: str,
) -> str:
    """The app's animation timeline for a PLO preflop question.

    The node's ``history_before`` lists every prior action including folds;
    sizes come from the same pot-limit resolution the Question prose uses
    (:func:`resolve_pot_limit`), so the two can never disagree. Seats are
    mapped to the player-facing names (LJ -> UTG, BU -> BTN) to match the
    Seats column."""
    from pipeline.animation import AnimationTable, walk_explicit_preflop  # noqa: PLC0415

    node = facts.spot.node
    ts = node.table_size
    resolved, _pot = resolve_pot_limit(node.history_before, stack_bb=stack_bb)
    actions: list[tuple[str, str, float | None, bool]] = []
    for ra in resolved:
        seat = display_seat(ra.seat, table_size=ts)
        if ra.verb == "fold":
            actions.append((seat, "fold", None, False))
        elif ra.verb == "call":
            actions.append((seat, "call", None, False))
        elif ra.verb == "all-in":
            actions.append((seat, "raise", ra.to_bb, True))
        else:  # open / 3-bet / 4-bet / 5-bet / raise, with a raise-to size
            actions.append((seat, "raise", ra.to_bb, False))
    table = AnimationTable(
        table_size=ts,
        starting_stack_bb=float(stack_bb),
        bb_in_dollars=(
            float(stakes_bb_dollars) if game_format != "tournament" else None
        ),
    )
    walk_explicit_preflop(
        table, actions, decision_seat=display_seat(node.actor, table_size=ts),
    )
    return table.render(display_seat(node.actor, table_size=ts))


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
    pack: PloPack | None = None,
) -> dict[str, str]:
    """Build one PLO CSV row (a dict keyed by :data:`PLO_CSV_COLUMNS`).

    ``options`` / ``correct_answer`` come from :func:`pipeline.plo.options.
    build_options`; ``explanation`` is the Layer 6 text (``""`` until it exists).
    Every column is filled.

    ``pack`` (the discovered :class:`PloPack`) is needed only to resolve the
    still-active opponents' ranges for the ``range_breakdown`` column; when it
    is omitted (some unit tests) that column ships empty and nothing else
    changes.
    """
    table = build_plo_app_table_columns(
        facts,
        stakes_bb_dollars=stakes_bb_dollars,
        game_format=game_format,
        display_in_bb=display_in_bb,
        stack_bb=stack_bb,
        table_size=facts.spot.node.table_size,
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

    # Decision math (July 2026 -- PLO "shows the math" like NLHE): the raw
    # pot-odds price from the pot-limit walk, the sampled equities when the
    # batch computed them, and the written-out arithmetic for the admin
    # Show-the-math panel (same {key,label,value,note} JSON entry shape as
    # pipeline.preflop.stat_notes, so the generic panel/equity-bar readers
    # work unchanged).
    pot_now, to_call = call_price(
        facts.spot.node.history_before, facts.spot.node.actor, stack_bb=stack_bb
    )
    # Same gate as the SOLVER DATA price block (price_is_live): an open
    # decision has no call on the menu, so no pot-odds cells there.
    break_even = (
        to_call / (pot_now + to_call)
        if to_call > 0
        and price_is_live(facts.spot.node.history_before, facts.spot.node.actor)
        else None
    )
    hero_eq = facts.hero_equity_vs_villain
    range_eq = facts.hero_range_equity_vs_villain
    strategy = canonicalize_strategy(facts)
    # Per-action EVs keyed by the CANONICAL labels the rest of the row speaks
    # ("3-bet", not the raw "Raise 100%"). Duplicate raw labels collapsing
    # into one canonical label (two raise sizes) merge frequency-weighted;
    # zero-frequency raw labels keep their counterfactual EV via the epsilon.
    _raise_level = _hero_raise_level(facts)
    _check_spot = is_check_spot(facts)
    _canon_ev: dict[str, tuple[float, float]] = {}
    for raw_label, ev_sb in facts.spot.ev_by_action.items():
        canon = canonicalize_action_label(
            raw_label, raise_level=_raise_level, check_spot=_check_spot
        )
        weight = max(facts.spot.action_frequencies.get(raw_label, 0.0), 1e-9)
        acc_ev, acc_w = _canon_ev.get(canon, (0.0, 0.0))
        _canon_ev[canon] = (acc_ev + ev_sb * weight, acc_w + weight)
    action_ev_cell = ", ".join(
        f"{label}: {_canon_ev[label][0] / _canon_ev[label][1] / _SB_PER_BB:.2f}"
        for label in strategy
        if label in _canon_ev
    )
    stat_entries: list[dict[str, str]] = []
    if break_even is not None:
        stat_entries.append({
            "key": "pot_odds",
            "label": "Pot odds",
            # Bare percentage (team, July 2026): the panel renders
            # "Pot odds · 41%", not "need 41%" -- whether THIS hand should
            # call at that price is the explanation's job (implied odds can
            # justify a sub-threshold call), matching the preflop/postflop
            # pot-odds rows.
            "value": f"{break_even * 100:.0f}%",
            "note": (
                f"Facing {to_call:.1f}bb into the {pot_now:.1f}bb pot: "
                f"break-even equity = {to_call:.1f} / ({pot_now:.1f} + "
                f"{to_call:.1f}) = {break_even * 100:.0f}%."
            ),
        })
    if hero_eq is not None and facts.villain_stats is not None:
        stat_entries.append({
            "key": "hero_equity",
            "label": "Your equity",
            "value": f"{hero_eq * 100:.0f}%",
            "note": (
                f"Your exact four cards vs the "
                f"{display_seat(facts.villain_stats.seat, table_size=facts.spot.node.table_size)}"
                f" range ({facts.villain_stats.pct_of_dealt_hands:.1f}% of all "
                f"hands), sampled over {facts.hero_equity_runouts_used} runouts."
            ),
        })
    if range_eq is not None:
        stat_entries.append({
            "key": "range_equity",
            "label": "Your range's equity",
            "value": f"{range_eq * 100:.0f}%",
            "note": "Your ENTIRE range at this node vs the same villain range.",
        })
    # Flush ceiling + dead weight (July 16 2026, panel ideas 1+2): the
    # 4-card structural facts, one row per suit hero can flush in (or the
    # "no flush possible" row) plus at most one redundant-card row. Pure
    # functions of hero's cards; the notes reuse the exact describe_*
    # sentences the SOLVER DATA block ships, so panel and prose can never
    # disagree. Flush ceiling always emits >= 1 entry, so PLO stat_notes
    # is now never empty.
    stat_entries.extend(flush_ceiling_stat_entries(facts.spot.hero_cards))
    stat_entries.extend(dead_weight_stat_entries(facts.spot.hero_cards))

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
            table_size=facts.spot.node.table_size,
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
        # Notes carries provenance + chart + situation + the node reference
        # (the old solver_reference value, now in the Node: field). July 2026.
        "Notes": _plo_notes(facts, pack_label),
        "Position Matchup": _position_matchup(facts),
        "archetype": facts.archetype,
        "hand_shape": facts.hand_class.descriptor,
        "board_texture": "",
        "validation_status": validation_status,
        "easy_freq": f"{difficulty.easy_freq:.3f}",
        "easy_ev": f"{difficulty.easy_ev:.3f}" if difficulty.ev_available else "",
        "easy_concept": f"{difficulty.easy_concept:.3f}",
        "easy_hand": f"{difficulty.easy_hand:.3f}",
        # Decision-math columns (July 2026: PLO now shows the math). Blank
        # cells remain blank when the input doesn't exist (no bet faced -> no
        # pot_odds; equity-off batch -> no equities). blocker_combos /
        # top_villain_combos / exploit_notes stay blank by design -- those
        # NLHE helpers are 2-card-class-specific.
        "pot_odds": f"{break_even * 100:.1f}%" if break_even is not None else "",
        "hero_equity": f"{hero_eq * 100:.1f}%" if hero_eq is not None else "",
        "range_equity": f"{range_eq * 100:.1f}%" if range_eq is not None else "",
        "blocker_combos": "",
        "top_villain_combos": "",
        "stat_notes": (
            json.dumps(stat_entries, separators=(",", ":")) if stat_entries else ""
        ),
        "claim_check": "",  # filled by the batch when Layer-7 runs
        "exploit_notes": "",  # NLHE exploit tagger not wired for PLO (2-card)
        # Per-action solver EVs in bb (drives the Review EV panel, like NLHE).
        "action_ev_bb": action_ev_cell,
        # Per-question chatbot context (all deterministic facts as one JSON blob),
        # reusing PLO's rich generation block for the PLO-specific facts.
        "animation_script": _plo_animation_script(
            facts, stack_bb=stack_bb, stakes_bb_dollars=stakes_bb_dollars,
            game_format=game_format,
        ),
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
        # Hand-type range breakdown (hero + every still-active opponent, by
        # shape, with hero's action split) + deterministic copy. One JSON
        # cell right after animation_script. Needs the pack to resolve the
        # opponents' ranges; empty when the pack isn't supplied.
        "range_breakdown": (
            json.dumps(build_range_breakdown(facts, pack), separators=(",", ":"))
            if pack is not None
            else ""
        ),
    }
    # Defensive: guarantee full schema coverage. The dict deliberately keeps
    # MORE keys than the schema -- the July-16 dropped columns (validation
    # status, equity/pot-odds flats, difficulty diagnostics) still feed the
    # batch layer, the meta sidecar, and tests; write_plo_csv trims to
    # PLO_CSV_COLUMNS at write time (extrasaction="ignore").
    missing = set(PLO_CSV_COLUMNS) - set(row)
    if missing:
        msg = f"build_plo_row missing columns: {sorted(missing)}"
        raise AssertionError(msg)
    return row


def write_plo_csv(rows: list[dict[str, str]], path: Path | str) -> int:
    """Write rows to ``path`` with the PLO header. Returns the row count.

    ``extrasaction="ignore"``: ``build_plo_row`` deliberately returns MORE
    keys than the CSV schema (the July-16 dropped columns -- validation
    status, equity/pot-odds flats, difficulty diagnostics -- stay in the
    dict for the batch layer, the meta sidecar, and tests); only the
    ``PLO_CSV_COLUMNS`` subset reaches the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(PLO_CSV_COLUMNS), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


__all__ = ["PLO_CSV_COLUMNS", "build_plo_row", "write_plo_csv"]
