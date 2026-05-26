"""Tests for pipeline.preflop.format_writer (Layer 8, preflop).

Sibling of ``tests/test_format_writer.py``. Uses synthetic PreflopFacts +
GeneratedExplanation fixtures -- no PioSolver and no Anthropic API
needed. Asserts the 38-column structure, the preflop-specific column
values (Hand Stage / Cards on Table / Question), and that the
write-csv round-trip preserves the schema.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import GeneratedExplanation  # noqa: E402
from pipeline.format_writer import CSV_COLUMNS  # noqa: E402
from pipeline.preflop.fact_extractor import (  # noqa: E402
    PreflopFacts,
    VillainRangeStats,
)
from pipeline.preflop.format_writer import (  # noqa: E402
    _pot_participant_for_facts,
    _pot_type_for_facts,
    _relative_position,
    build_preflop_row,
    format_preflop_question,
    write_preflop_csv,
)
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.pack import PreflopPack  # noqa: E402
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


# --- fixtures ---------------------------------------------------------------
def _pack() -> PreflopPack:
    """A Ryan-pack-shaped fixture. root_path can be any path -- the row
    builder never touches the filesystem."""
    return PreflopPack(
        pack_id="ryan_preflop_tree_6max_100bb",
        root_path=Path("/tmp/test_pack"),
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
        sb_to_bb_ratio=0.5,
        description="test pack",
    )


def _node(
    actor: str = "BTN",
    history: tuple[ParsedAction, ...] = (),
) -> PreflopDecisionNode:
    """Minimal node fixture with empty `actions` -- the row builder only
    reads `actor` and `history_before`."""
    return PreflopDecisionNode(
        pack_id="ryan_preflop_tree_6max_100bb",
        actor=actor,
        history_before=history,
        actions=(),
    )


def _open_facts() -> PreflopFacts:
    """UTG opens AKo at full frequency. No prior history -- first-to-act."""
    spot = PreflopSpot(
        node=_node(actor="UTG", history=()),
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies={"Fold": 0.0, "Raise 60%": 1.0},
        dominant_action="Raise 60%",
        dominant_frequency=1.0,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=None,
        archetype="open_for_value",
    )


def _facing_open_facts() -> PreflopFacts:
    """SB facing BTN open with AQs -- 60% call, 40% 3-bet."""
    spot = PreflopSpot(
        node=_node(
            actor="SB",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.FOLD),
                ParsedAction("CO", PreflopActionType.FOLD),
                ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
            ),
        ),
        hero_hand_class="AQs",
        hero_card_combo="AsQs",
        action_frequencies={
            "Fold": 0.0,
            "Call": 0.60,
            "Raise 308%": 0.40,
        },
        dominant_action="Call",
        dominant_frequency=0.60,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="BTN",
            action_label="Raise 60%",
            weighted_combo_count=618.0,
            pct_of_dealt_hands=46.6,
            top_combos=(("AA", 1.0), ("KK", 1.0), ("AKs", 1.0)),
        ),
        hero_equity_vs_villain=0.474,
        archetype="3bet_as_bluff",
    )


def _facing_3bet_facts() -> PreflopFacts:
    """BTN facing BB 3-bet after BTN opened -- 4-bet spot."""
    spot = PreflopSpot(
        node=_node(
            actor="BTN",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.FOLD),
                ParsedAction("CO", PreflopActionType.FOLD),
                ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
                ParsedAction("SB", PreflopActionType.FOLD),
                ParsedAction("BB", PreflopActionType.RAISE, 182.0),
            ),
        ),
        hero_hand_class="JJ",
        hero_card_combo="JcJd",
        action_frequencies={
            "Fold": 0.15,
            "Call": 0.70,
            "Raise 50%": 0.15,
        },
        dominant_action="Call",
        dominant_frequency=0.70,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="BB",
            action_label="Raise 182%",
            weighted_combo_count=80.0,
            pct_of_dealt_hands=6.0,
            top_combos=(("AA", 1.0), ("KK", 1.0), ("QQ", 0.5)),
        ),
        hero_equity_vs_villain=0.51,
        archetype="4bet_for_value",
    )


def _squeeze_facts() -> PreflopFacts:
    """3-way preflop: HJ opens, CO calls, hero=BTN faces squeeze decision."""
    spot = PreflopSpot(
        node=_node(
            actor="BTN",
            history=(
                ParsedAction("UTG", PreflopActionType.FOLD),
                ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
                ParsedAction("CO", PreflopActionType.CALL),
            ),
        ),
        hero_hand_class="AKs",
        hero_card_combo="AsKs",
        action_frequencies={
            "Fold": 0.0,
            "Call": 0.30,
            "Raise 85%": 0.70,
        },
        dominant_action="Raise 85%",
        dominant_frequency=0.70,
    )
    return PreflopFacts(
        spot=spot,
        villain_stats=VillainRangeStats(
            position="HJ",
            action_label="Raise 60%",
            top_combos=(("AA", 1.0),),
            weighted_combo_count=200.0,
            pct_of_dealt_hands=15.0,
        ),
        hero_equity_vs_villain=0.55,
        archetype="squeeze_for_value",
    )


def _explanation() -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1="Fold",
        option_2="Call",
        option_3="Raise 308%",
        option_4="",
        correct_answer="Call",
        answer_explanation="The best play is to call.",
    )


# --- 38-column structure ----------------------------------------------------
def test_thirty_eight_column_structure() -> None:
    """Every preflop row covers all 38 CSV_COLUMNS."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    assert set(row.keys()) == set(CSV_COLUMNS)
    assert len(row) == 38


def test_no_column_auto_increments() -> None:
    row1 = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=7,
    )
    assert row1["No"] == "7"


# --- preflop-specific column values -----------------------------------------
def test_hand_stage_is_preflop() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    assert row["Hand Stage"] == "Preflop"


def test_cards_on_table_is_empty_preflop() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    assert row["Cards on Table"] == ""


def test_user_cards_split_into_space_separated() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    # combo "AsQs" renders with suit emojis, space-separated.
    assert row["User Cards"] == "A♠️ Q♠️"


def test_hand_class_is_169_label() -> None:
    """Preflop hand_class is the 169-class label (AKo), not a postflop
    made-hand breakdown."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    assert row["hand_class"] == "AQs"


def test_board_texture_is_empty_preflop() -> None:
    row = build_preflop_row(
        _open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    assert row["board_texture"] == ""


def test_ev_gap_and_concept_tags_and_range_columns_empty_for_phase_a() -> None:
    """ev_gap_bb, concept_tags, ip_range, oop_range are deferred to Phase B
    (per the module docstring) and must render as empty strings rather than
    placeholders in step 8 v1."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    assert row["ev_gap_bb"] == ""
    assert row["concept_tags"] == ""
    assert row["ip_range"] == ""
    assert row["oop_range"] == ""


def test_solver_reference_is_pack_relative_path() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    # pack_id / actor / node_id
    assert row["solver_reference"].startswith("ryan_preflop_tree_6max_100bb/SB/")
    # Node id encodes the action history of the node.
    assert "BTN_60%" in row["solver_reference"]


def test_llm_columns_pass_through_from_explanation() -> None:
    explanation = GeneratedExplanation(
        option_1="Fold",
        option_2="Call",
        option_3="Raise 308%",
        option_4="",
        correct_answer="Call",
        answer_explanation="The best play is to call.",
    )
    row = build_preflop_row(
        _facing_open_facts(),
        explanation,
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    assert row["option 1"] == "Fold"
    assert row["option 2"] == "Call"
    assert row["option 3"] == "Raise 308%"
    assert row["option 4"] == ""
    assert row["Correct Answer"] == "Call"
    assert row["Answer Explanation"] == "The best play is to call."


def test_difficulty_rating_passes_through() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=2300,
        number=1,
    )
    assert row["Difficulty Rating"] == "2300"


def test_action_frequencies_descending_percentages() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    # Call 60%, Raise 40%, Fold 0% (dropped from the column? actually
    # postflop helper renders all entries. Let's verify: postflop helper
    # renders descending including zeros).
    assert row["action_frequencies"].startswith("Call: 60%")
    assert "Raise 308%: 40%" in row["action_frequencies"]


# --- pot type, participant, relative position --------------------------------
def test_pot_type_for_open_is_single_raise() -> None:
    """Hero is opening (dominant=Raise, history empty) -> Single raise pot."""
    assert _pot_type_for_facts(_open_facts()) == "Single raise pot"


def test_pot_type_for_facing_open_call_is_single_raise() -> None:
    """Hero facing one prior raise, dominant action is Call -> 1 raise total
    after the decision -> Single raise pot."""
    assert _pot_type_for_facts(_facing_open_facts()) == "Single raise pot"


def test_pot_type_for_facing_3bet_with_call_is_3bet_pot() -> None:
    """Hero facing 2 prior raises, dominant action is Call -> 2 raises total
    -> Three bet pot."""
    assert _pot_type_for_facts(_facing_3bet_facts()) == "Three bet pot"


def test_pot_participant_heads_up_for_facing_open() -> None:
    """SB faces BTN open: only one non-fold actor + hero -> heads-up."""
    assert _pot_participant_for_facts(_facing_open_facts()) == "Heads-Up"


def test_pot_participant_multi_way_for_squeeze() -> None:
    """HJ opens, CO calls, hero=BTN -> 2 non-folds + hero = 3 players."""
    assert _pot_participant_for_facts(_squeeze_facts()) == "Multi-Way"


def test_relative_position_with_villain() -> None:
    """Relative Position = hero_vs_villain when there's a villain."""
    assert _relative_position(_facing_open_facts()) == "SB_vs_BTN"


def test_relative_position_open_is_just_hero() -> None:
    """No villain (open spot) -> Relative Position = just hero's seat."""
    assert _relative_position(_open_facts()) == "UTG"


# --- question narrative (deterministic, no LLM) -----------------------------
def test_question_for_open_no_prior_history() -> None:
    """UTG opens, no prior actions -> just the hero line, no action history.
    Cards are concatenated without a space per the brief ('two cards with no
    space between them'). The Question does NOT carry context (table size /
    stakes / stack) -- that lives in the Context column. It does NOT end
    with 'What's your play?' -- the UI implies the question."""
    q = format_preflop_question(_open_facts(), pack=_pack())
    assert "You're UTG with A❤️K♣️" in q
    # Context info must NOT appear in the Question column.
    assert "6-handed" not in q
    assert "cash game" not in q
    assert "$0.25/$0.50" not in q
    assert "effective stacks" not in q
    # UI implies the question; no trailing prompt.
    assert "What's your play?" not in q


def test_question_for_facing_open_renders_villain_history() -> None:
    """Per the brief's Fold Rule: preflop folds are DROPPED from the action
    history (implied by absence). Only non-fold actions appear; raises
    render with dollar amounts."""
    q = format_preflop_question(_facing_open_facts(), pack=_pack())
    assert "You're in the Small Blind with A♠️Q♠️" in q
    # Per the brief: preflop folds are implied by absence, NOT listed.
    assert "UTG folds" not in q
    assert "The Hijack folds" not in q
    assert "The Cutoff folds" not in q
    # The non-fold action does appear, with its dollar amount.
    assert "The Button opens to $1.25" in q


def test_question_for_facing_3bet_uses_3bet_verb() -> None:
    """A second raise (after an open) renders as '3-bets'. Hero (BTN in
    this fixture) opened, so their own action renders with the base verb
    ('you open') -- only villains get third-person."""
    q = format_preflop_question(_facing_3bet_facts(), pack=_pack())
    # Hero opened (BTN is hero in this fixture). Hero verb is "open" not
    # "opens" (the brief: hero uses base verb, villain uses third-person).
    assert "You open to $1.25" in q
    # BB 3-bets (with size from the Ryan-pack lookup: 182% -> 12bb -> $6).
    assert "The Big Blind 3-bets to $6" in q


def test_question_renderer_scales_amounts_with_stakes() -> None:
    """Raise amounts in the Question scale with stakes_bb_dollars: at $0.25/
    $0.50 the open is "$1.25"; at $1/$2 the open is "$5"."""
    cash_low = format_preflop_question(
        _facing_open_facts(), pack=_pack(), stakes_bb_dollars=0.50
    )
    cash_high = format_preflop_question(
        _facing_open_facts(), pack=_pack(), stakes_bb_dollars=2.0
    )
    assert "$1.25" in cash_low
    assert "$5" in cash_high


def test_question_renderer_tournament_uses_bb() -> None:
    """Tournament-format hands render amounts in bb, not dollars."""
    tourney = format_preflop_question(
        _facing_open_facts(), pack=_pack(), game_format="tournament"
    )
    assert "2.5bb" in tourney  # 2.5x open in bb
    assert "$" not in tourney


def test_question_renderer_omits_context_info() -> None:
    """Stakes / table size / stack depth never leak into the Question
    column -- those go in the Context column."""
    q = format_preflop_question(_facing_open_facts(), pack=_pack())
    # No "X-handed", no "cash game" phrase, no "effective stacks" phrase.
    assert "6-handed" not in q
    assert "cash game" not in q
    assert "effective stacks" not in q


# --- defaults + kwargs ------------------------------------------------------
def test_defaults_render_cash_at_default_stakes() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
    )
    # $0.25/$0.50 cash, 100bb -> $50 default stack.
    assert row["Cash/Tourney"] == "Cash"
    assert row["Live or Online"] == "Online"
    assert row["Default Stack"] == "$50"
    assert "$0.25/$0.50" in row["Context"]
    assert "Stacks $50" in row["Context"]


def test_stakes_kwarg_scales_dollar_columns() -> None:
    """A higher BB rate scales Default Stack + Context proportionally."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
        stakes_bb_dollars=2.0,
    )
    assert row["Default Stack"] == "$200"
    assert "$1/$2" in row["Context"]


def test_tournament_game_format_uses_bb_stack() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
        game_format="tournament",
    )
    assert row["Cash/Tourney"] == "Tournament"
    assert row["Default Stack"] == "100bb"


def test_live_or_online_kwarg() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty_score=1500,
        number=1,
        live_or_online="Live",
    )
    assert row["Live or Online"] == "Live"


# --- write_preflop_csv round-trip -------------------------------------------
def test_write_preflop_csv_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "preflop.csv"
        count = write_preflop_csv(
            out,
            [
                (_facing_open_facts(), _explanation(), 1500),
                (_facing_3bet_facts(), _explanation(), 2200),
            ],
            pack=_pack(),
        )
        assert count == 2

        with open(out, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            data_rows = list(reader)
        assert header == CSV_COLUMNS
        assert len(data_rows) == 2

        no_index = CSV_COLUMNS.index("No")
        assert [r[no_index] for r in data_rows] == ["1", "2"]

        stage_index = CSV_COLUMNS.index("Hand Stage")
        assert all(r[stage_index] == "Preflop" for r in data_rows)

        board_index = CSV_COLUMNS.index("Cards on Table")
        assert all(r[board_index] == "" for r in data_rows)


def test_write_preflop_csv_creates_parent_dirs() -> None:
    """Parent directory is auto-created -- matches the postflop writer."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "a" / "b" / "c" / "preflop.csv"
        count = write_preflop_csv(
            out,
            [(_open_facts(), _explanation(), 800)],
            pack=_pack(),
        )
        assert count == 1
        assert out.is_file()


# --- bad-input guards -------------------------------------------------------
def test_invalid_combo_length_raises() -> None:
    """A malformed hero_card_combo (not 4 chars) triggers ValueError --
    defensive: shouldn't normally happen, but catches data corruption."""
    spot = PreflopSpot(
        node=_node(actor="BTN", history=()),
        hero_hand_class="AKo",
        hero_card_combo="AhKcQd",  # 6 chars -- invalid
        action_frequencies={"Fold": 0.0, "Raise 60%": 1.0},
        dominant_action="Raise 60%",
        dominant_frequency=1.0,
    )
    facts = PreflopFacts(spot=spot, archetype="open_for_value")
    with pytest.raises(ValueError, match="combo must be 4 chars"):
        build_preflop_row(
            facts,
            _explanation(),
            pack=_pack(),
            difficulty_score=1500,
            number=1,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
