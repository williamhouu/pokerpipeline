"""Layer 8: Format Writer.

Writes pipeline questions out as CSV rows in the team's question-template
column format. The authoritative layout is docs/output_format_examples.xlsx
(Sheet 1, "Example formatting") -- 35 baseline columns: a leading `No`, the
28 template columns, the 5 pipeline columns (concept_tags, hand_class,
board_texture, solver_reference, ev_gap_bb), and `validation_status`.

A 36th column `action_frequencies` was added per Ryan-feedback Fix 3 (Apr
2026): a comma-separated `<verb>: <integer>%` summary of the solver's
range-aggregate strategy at the decision point, so the QA reviewer can see
the mix at a glance without opening the .cfr.

Columns the pipeline cannot fill yet carry an explicit placeholder rather than
being left blank:
  * `[TBD by Layer 6]` -- the question stem, the four options, and the answer
    explanation, all written by the (not-yet-built) LLM Explanation Generator;
  * `[TBD]` -- the cols-A-H UI-rendering fields, whose formatting algorithm the
    brief defers.

TODO(Layer 6): `Correct Answer` must equal one of the four `option N` strings
*verbatim*. It currently holds the solver's correct-action verb as a stand-in;
once Layer 6 generates the option strings, set Correct Answer to the matching
option's exact text and enforce that equality (the CSV consumer pairs them by
exact string match).

Uses the standard library `csv` module -- no pandas dependency.
"""
from __future__ import annotations

import csv
from pathlib import Path

from pipeline.action_history import format_action_history
from pipeline.fact_extractor.spot_data import SpotData
from pipeline.scenario_config import ScenarioConfig, spot_to_hand

_LLM_TBD = "[TBD by Layer 6]"     # LLM-generated content
_TBD = "[TBD]"                    # deferred cols-A-H UI formatting

# The 35 question-template columns, in order (docs/output_format_examples.xlsx).
CSV_COLUMNS = [
    "No",
    # UI rendering.
    "User Seat", "User Cards", "Cards on Table", "Table Size",
    "Default Stack", "Seats", "POT",
    # Question content.
    "Context", "Question", "Question Type", "Hand Stage",
    "option 1", "option 2", "option 3", "option 4",
    "Correct Answer", "Answer Explanation",
    # Classification.
    "Cash/Tourney", "Live or Online", "Relative Position", "Preflop Pot Type",
    "Pot Participant", "Stack Depth", "Difficulty Rating",
    # Skill tags + misc.
    "tag_1", "tag_2", "tag_3", "Notes",
    # New pipeline columns.
    "concept_tags", "hand_class", "board_texture", "solver_reference",
    "ev_gap_bb", "validation_status",
    # Ryan-feedback Fix 3 (Apr 2026): "fold: 10%, call: 60%, raise: 20%, all-in: 10%".
    "action_frequencies",
]

# Preflop pot type -- prose form for the CSV (matches the sample's wording).
_PREFLOP_POT_TYPE = {0: "Limped pot", 1: "Single raise pot",
                     2: "Three bet pot", 3: "Four bet pot"}
# Preflop pot type -- slug form for the solver_reference path.
_POT_TYPE_SLUG = {0: "limped_pot", 1: "single_raised_pot",
                  2: "3bet_pot", 3: "4bet_pot"}
# Reconcile the board_texture module's rank vocabulary with the sample's.
_RANK_ALIASES = {"broadway_heavy": "broadway"}


def _bb(value: float) -> str:
    """A stack/blind count formatted as e.g. '98bb'."""
    return f"{round(value)}bb"


def _stack_depth_bucket(effective_stack_bb: float) -> str:
    """The prose Stack Depth bucket the sample uses, from the effective stack.

    v1 thresholds: < 40bb short, 40-150bb standard, > 150bb deep.
    """
    if effective_stack_bb < 40:
        return "Short stack"
    if effective_stack_bb <= 150:
        return "Standard Stack"
    return "Deep stack"


def _board_texture_string(board_texture) -> str:
    """The 3-axis board_texture string: '{suit}_{connectedness}_{rank}'.

    Example: 'monotone_connected_broadway'. The single composite descriptor
    (e.g. 'very_wet') is deliberately not emitted -- the sample uses the axes.
    """
    if board_texture is None:
        return ""
    rank = _RANK_ALIASES.get(board_texture.rank_distribution,
                             board_texture.rank_distribution)
    return f"{board_texture.suit_distribution}_{board_texture.connectedness}_{rank}"


def _solver_reference(meta) -> str:
    """A descriptive cache-style path identifying the solve and the spot.

    Built from the scenario metadata on the SpotData, e.g.
    'PioSolver_Cash_100bb/BB_vs_BTN/single_raised_pot/flop_2cJs7s/AhKh'.

    TODO(Layer 1): fully-formed paths -- the solver build, the table size, and
    the exact preflop action line (BTN_open/BB_call/...) -- need the scenario
    registry to carry richer identifiers than a postflop solve provides.
    """
    pot_type = _POT_TYPE_SLUG.get(meta.preflop_raise_count, "multi_raised_pot")
    board = "".join(meta.board)
    street_segment = f"{meta.street}_{board}" if board else meta.street
    segments = [
        f"PioSolver_{meta.game_format.capitalize()}_{round(meta.stack_depth_bb)}bb",
        meta.position_dynamic or "unknown",
        pot_type,
        street_segment,
        "".join(meta.hero_cards),
    ]
    return "/".join(segment for segment in segments if segment)


def _dollars(amount: float) -> str:
    """Cash amount as a string. Integer dollars render '$50'; cents '$1.25'."""
    if isinstance(amount, float) and not amount.is_integer():
        return f"${amount:,.2f}"
    return f"${int(amount):,}"


def _format_action_frequencies(strategy: dict[str, float]) -> str:
    """The Fix-3 action_frequencies column value.

    Renders `decision.range_aggregate_strategy` (a `{verb: freq}` map)
    as a comma-separated list of `<verb>: <integer>%` entries, ordered by
    descending frequency. Example: `{"call": 0.66, "fold": 0.34}` ->
    `"call: 66%, fold: 34%"`.

    Empty strategy -> empty string (the column is best-effort: spots
    without a populated strategy still produce a valid CSV row).
    """
    if not strategy:
        return ""
    by_freq_desc = sorted(strategy.items(), key=lambda kv: -kv[1])
    parts = [f"{verb}: {round(100 * freq)}%" for verb, freq in by_freq_desc]
    return ", ".join(parts)


def _villain_seat(meta, scenario: ScenarioConfig) -> str:
    """Simplified Seats column -- the villain's seat and remaining stack.

    The .xlsx sample uses a richer 'SB-$29.65-$11.60-bet' format that encodes
    each villain's last action; replicating that needs per-villain commitment
    tracking. For Tier 1 heads-up SRP the simple `<position>-$<stack>` form
    is enough to populate the column from real data; the richer form is left
    for a later pass.
    """
    villain_remaining = meta.effective_stack_bb * scenario.dollars_per_bb
    return f"{meta.villain_position}-{_dollars(round(villain_remaining, 2))}"


def build_row(spot_data: SpotData, difficulty_score: int, number: int, *,
              scenario: ScenarioConfig | None = None) -> dict[str, str]:
    """Turn one populated SpotData plus its difficulty into a CSV row dict.

    `number` is the row's `No` value -- a per-output auto-increment. Every
    column in CSV_COLUMNS is present; LLM and deferred-UI columns carry an
    explicit placeholder.

    `scenario` carries the per-solve facts the postflop `.cfr` file doesn't
    know -- table size, blinds, online/live, the preflop action line. When
    passed (the normal pipeline path via `pipeline.scenario_config.get_scenario`)
    the Context, Question, Table Size, Default Stack, Live or Online, Seats,
    and POT columns are filled from real data; when omitted (legacy tests
    that haven't been updated) those columns stay as `[TBD]` so the row
    structure is unchanged.
    """
    meta = spot_data.spot_metadata
    decision = spot_data.decision_data

    if scenario is not None:
        context = scenario.context
        hand = spot_to_hand(spot_data, scenario)
        question = format_action_history(hand)
        table_size = str(scenario.table_size)
        default_stack = _dollars(scenario.default_stack_dollars)
        live_or_online = scenario.live_or_online
        seats = _villain_seat(meta, scenario)
        pot = _dollars(round(meta.pot_bb * scenario.dollars_per_bb, 2))
    else:
        context = _LLM_TBD
        question = _LLM_TBD
        table_size = _TBD
        default_stack = _bb(meta.effective_stack_bb)
        live_or_online = _TBD
        seats = _TBD
        pot = _TBD

    return {
        "No": str(number),
        # UI rendering -- now filled from the scenario when one is provided.
        "User Seat": meta.hero_position or _TBD,
        "User Cards": " ".join(meta.hero_cards) or _TBD,
        "Cards on Table": " ".join(meta.board) or _TBD,
        "Table Size": table_size,
        "Default Stack": default_stack,
        "Seats": seats,
        "POT": pot,
        # Question content -- Context and Question are deterministic
        # (action_history.format_action_history); the LLM writes options +
        # correct answer + explanation (Layer 6).
        "Context": context,
        "Question": question,
        "Question Type": "Multiple Choice",
        "Hand Stage": meta.street.capitalize(),
        "option 1": _LLM_TBD,
        "option 2": _LLM_TBD,
        "option 3": _LLM_TBD,
        "option 4": _LLM_TBD,
        "Correct Answer": decision.correct_action or _TBD,   # see module TODO
        "Answer Explanation": _LLM_TBD,
        # Classification -- filled from the data block.
        "Cash/Tourney": meta.game_format.capitalize(),
        "Live or Online": live_or_online,
        "Relative Position": meta.position_dynamic or _TBD,
        "Preflop Pot Type": _PREFLOP_POT_TYPE.get(meta.preflop_raise_count,
                                                  "Multi-raised pot"),
        "Pot Participant": ("Heads-Up" if meta.active_players_on_flop <= 2
                            else "Multi-Way"),
        "Stack Depth": _stack_depth_bucket(meta.effective_stack_bb),
        "Difficulty Rating": str(difficulty_score),
        # Skill tags -- the Phase 3 skill taxonomy, not yet mapped.
        "tag_1": "", "tag_2": "", "tag_3": "",
        "Notes": "Auto-generated by poker-pipeline.",
        # The 5 new pipeline columns.
        "concept_tags": ", ".join(spot_data.concept_tags),
        "hand_class": spot_data.hand_class.label if spot_data.hand_class else "",
        "board_texture": _board_texture_string(spot_data.board_texture),
        "solver_reference": _solver_reference(meta),
        "ev_gap_bb": f"{decision.ev_gap_bb:.2f}",
        "validation_status": "auto_approved",      # pipeline-generated default
        # Ryan-feedback Fix 3: surface the action mix at the decision node
        # so the QA reviewer can read it without opening the .cfr.
        "action_frequencies": _format_action_frequencies(
            decision.range_aggregate_strategy),
    }


def write_csv(path, rows, *, scenario: ScenarioConfig | None = None) -> int:
    """Write question rows to a CSV at `path`; return the number of rows written.

    `rows` is an iterable of (SpotData, difficulty_score) pairs. The `No` column
    auto-increments from 1 across the output. The parent directory is created
    if needed; the header is always written. `scenario` is forwarded to every
    `build_row` call so Context / Question / etc. fill from real data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # utf-8-sig writes a BOM so Excel on Windows auto-detects UTF-8 instead
    # of falling back to cp1252 and mojibake-ing the suit emoji bytes
    # (♥ ♦ ♣ ♠ + U+FE0F variation selector) as Latin-1.
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for number, (spot_data, difficulty) in enumerate(rows, start=1):
            writer.writerow(build_row(spot_data, difficulty, number,
                                      scenario=scenario))
            written += 1
    return written
