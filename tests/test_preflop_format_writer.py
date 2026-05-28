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
from pipeline.preflop.difficulty import DifficultyResult  # noqa: E402
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


def _difficulty(score: int = 1500) -> DifficultyResult:
    """Test fixture: a DifficultyResult with the given score and
    plausible per-axis values. The axes are wired up so reviewers
    inspecting test output won't be confused by zeros everywhere."""
    return DifficultyResult(
        score=score,
        easy_freq=0.5,
        easy_ev=0.5,
        easy_concept=0.5,
        easy_hand=0.5,
        easy_blend=0.5,
        bumps_applied=(),
        ev_available=True,
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


# --- 45-column structure (post-difficulty-diagnostics columns) ------------
def test_forty_five_column_structure() -> None:
    """Every preflop row covers all 45 CSV_COLUMNS.

    Column-count history:
      * 38: baseline schema (~Apr 2026)
      * 39: + skills (Phase 3 user-facing skill labels)
      * 40: + archetype (preflop strategic frame for QA)
      * 45: + easy_freq / easy_ev / easy_concept / easy_hand /
             difficulty_bumps (May 2026 difficulty algorithm redesign;
             per-axis breakdown of the rating for reviewer QA).
    """
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
    )
    assert set(row.keys()) == set(CSV_COLUMNS)
    assert len(row) == 45
    # Preflop rows ALWAYS populate archetype (it's a preflop-only
    # classifier). One of 16 labels or "unclassified".
    assert row["archetype"] != ""
    # Difficulty diagnostic columns are populated (the test fixture
    # uses easy_*=0.5 across the board).
    assert row["easy_freq"] == "0.500"
    assert row["easy_concept"] == "0.500"


def test_no_column_auto_increments() -> None:
    row1 = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=7,
    )
    assert row1["No"] == "7"


# --- preflop-specific column values -----------------------------------------
def test_hand_stage_is_preflop() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
    )
    assert row["Hand Stage"] == "Preflop"


def test_cards_on_table_is_empty_preflop() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
    )
    assert row["Cards on Table"] == ""


def test_user_cards_split_into_space_separated() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
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
        difficulty=_difficulty(1500),
        number=1,
    )
    assert row["hand_class"] == "AQs"


def test_board_texture_is_empty_preflop() -> None:
    row = build_preflop_row(
        _open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
    )
    assert row["board_texture"] == ""


def test_ip_oop_positions_blind_vs_button() -> None:
    """Non-BvB: the non-blind position is IP, BB is OOP."""
    from pipeline.preflop.format_writer import _ip_oop_positions

    assert _ip_oop_positions("BB", "BTN") == ("BTN", "BB")
    assert _ip_oop_positions("BTN", "BB") == ("BTN", "BB")
    assert _ip_oop_positions("HJ", "BB") == ("HJ", "BB")
    assert _ip_oop_positions("BB", "UTG") == ("UTG", "BB")


def test_ip_oop_positions_bvb_heads_up() -> None:
    """BvB heads-up: SB is the dealer and acts LAST postflop -> SB is IP,
    BB is OOP. This is the standard HU exception to the postflop-order
    rule."""
    from pipeline.preflop.format_writer import _ip_oop_positions

    assert _ip_oop_positions("SB", "BB") == ("SB", "BB")
    assert _ip_oop_positions("BB", "SB") == ("SB", "BB")


def test_ip_oop_positions_two_non_blind_seats() -> None:
    """Two non-blind seats: later in postflop order is IP."""
    from pipeline.preflop.format_writer import _ip_oop_positions

    # BTN later than CO, CO later than HJ, etc.
    assert _ip_oop_positions("BTN", "CO") == ("BTN", "CO")
    assert _ip_oop_positions("CO", "UTG") == ("CO", "UTG")


def test_ip_oop_positions_sb_vs_non_blind() -> None:
    """SB is the very first to act postflop (in non-HU), so SB is OOP
    against any non-blind."""
    from pipeline.preflop.format_writer import _ip_oop_positions

    assert _ip_oop_positions("BTN", "SB") == ("BTN", "SB")
    assert _ip_oop_positions("SB", "BTN") == ("BTN", "SB")


def test_pot_open_spot_is_just_blinds() -> None:
    """First-to-act spot (no history) -> pot is SB + BB = $0.75 at $0.50/bb."""
    from pipeline.preflop.format_writer import _compute_pot_bb

    facts = _open_facts()
    # No raises in history -> pot = sb + bb = 0.5 + 1.0 = 1.5bb.
    assert _compute_pot_bb(facts, _pack()) == pytest.approx(1.5)


def test_pot_after_open_includes_open_size() -> None:
    """After BTN opens 2.5bb, the pot has BTN's 2.5 + SB 0.5 + BB 1.0 = 4bb."""
    from pipeline.preflop.format_writer import _compute_pot_bb

    facts = _facing_open_facts()  # SB facing BTN open
    # SB hasn't acted yet but has the SB in; BB has the BB in; BTN
    # opened to 2.5bb. Hero's choice hasn't entered the pot yet.
    # Pot = 0.5 (SB) + 1.0 (BB) + 2.5 (BTN open) = 4.0bb.
    assert _compute_pot_bb(facts, _pack()) == pytest.approx(4.0)


def test_pot_after_3bet_includes_open_plus_3bet() -> None:
    """Open + 3-bet pot. BTN opens 2.5, BB 3-bets to 12 (Ryan-pack 182%
    token in the lookup), hero (BTN) deciding. Pot = 12 + 2.5 (BTN's
    open commit, NOT matched) + 0.5 SB. Wait actually BB's 3-bet
    REPLACES the BTN's 2.5? No -- BTN's 2.5 stays in but doesn't
    increase. Total = BB's 12 + BTN's 2.5 + SB's 0.5 = 15."""
    from pipeline.preflop.format_writer import _compute_pot_bb

    facts = _facing_3bet_facts()
    # _facing_3bet_facts has actor=BTN with history:
    #   UTG fold, HJ fold, CO fold, BTN raise 60%, SB fold, BB raise 182%
    # Open size (60% -> 2.5bb), 3-bet size (182% -> 12bb).
    # Committed: SB=0.5, BB=12.0, BTN=2.5. Pot = 15.0bb.
    assert _compute_pot_bb(facts, _pack()) == pytest.approx(15.0)


def test_pot_column_renders_dollars_for_cash() -> None:
    """POT column shows cash dollars when game_format=cash."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
        stakes_bb_dollars=0.50,
    )
    # 4.0bb * $0.50 = $2.00.
    assert row["POT"] == "$2"


def test_pot_column_scales_with_stakes() -> None:
    """At $1/$2, the same 4bb pot renders as $8."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
        stakes_bb_dollars=2.0,
    )
    # 4.0bb * $2 = $8.
    assert row["POT"] == "$8"


def test_pot_column_renders_bb_for_tournament() -> None:
    """Tournament mode renders pot in bb."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
        game_format="tournament",
    )
    assert row["POT"] == "4bb"


def test_range_snapshots_empty_for_open_spot() -> None:
    """No villain (open spot) -> both ip_range and oop_range empty.
    There's no IP/OOP frame without a 2-player anchor."""
    row = build_preflop_row(
        _open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
    )
    assert row["ip_range"] == ""
    assert row["oop_range"] == ""


def test_phase_b_columns_populated_when_data_available() -> None:
    """All Phase B columns now populate from their respective engines.
    Empty values only when the upstream data is missing (e.g. no villain
    -> ev_gap_bb can't compute; synthetic pack without villain range
    file -> ip_range/oop_range empty)."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
    )
    # concept_tags from Phase B step 1 -- populated when there are
    # firing tags (always at least position).
    assert row["concept_tags"]
    tag_list = [t.strip() for t in row["concept_tags"].split(",")]
    assert "small_blind" in tag_list
    assert "facing_single_raise" in tag_list
    assert "ace_blocker" in tag_list
    # ip_range / oop_range from Phase B step 2. The synthetic fixture
    # has villain_stats but the synthetic pack lacks the villain's
    # range file on disk, so the snapshot helper returns empty --
    # defensive fallback. The real-pack smoke test covers the populated
    # path.
    # ev_gap_bb from Phase B step 4. The fixture is BB facing BTN open
    # with action_frequencies {Call: 0.60, Raise 308%: 0.40, Fold: 0.0}.
    # Top-2 canonical actions are Call and 3-bet -- the engine returns
    # None because raises in the top-2 aren't computable in v1 EV. So
    # ev_gap_bb is empty for this specific fixture.
    assert row["ev_gap_bb"] == ""


def test_solver_reference_is_pack_relative_path() -> None:
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
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
        difficulty=_difficulty(1500),
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
        difficulty=_difficulty(2300),
        number=1,
    )
    assert row["Difficulty Rating"] == "2300"


def _sum_pcts(action_frequencies_str: str) -> int:
    """Pull the integer percentages out of an action_frequencies cell and
    return their sum. Helper for the rounding tests."""
    if not action_frequencies_str:
        return 0
    total = 0
    for chunk in action_frequencies_str.split(","):
        if ":" in chunk:
            _, pct = chunk.split(":")
            total += int(pct.strip().rstrip("%"))
    return total


def test_action_frequencies_sums_to_100_with_naive_rounding_case() -> None:
    """Cases that would round to 99% or 101% under naive per-entry
    rounding now sum to exactly 100% via the largest-remainder method.
    """
    from pipeline.preflop.format_writer import _format_action_frequencies

    # Case 1: would round to 99% naively (94 + 5 = 99, plus 0s).
    out1 = _format_action_frequencies({"Fold": 0.941, "4-bet": 0.054, "Call": 0.005})
    assert _sum_pcts(out1) == 100

    # Case 2: would round to 101% naively (60 + 25 + 16 = 101, plus 0).
    out2 = _format_action_frequencies({"Call": 0.604, "Fold": 0.247, "4-bet": 0.149})
    assert _sum_pcts(out2) == 100

    # Case 3: every freq has a .5 fractional remainder -- worst case for
    # naive rounding.
    out3 = _format_action_frequencies({"Call": 0.335, "Fold": 0.335, "Raise": 0.330})
    assert _sum_pcts(out3) == 100


def test_action_frequencies_order_preserved_after_rounding() -> None:
    """The rounding fix must not change the descending-frequency order."""
    from pipeline.preflop.format_writer import _format_action_frequencies

    out = _format_action_frequencies({"Fold": 0.604, "Call": 0.247, "Raise": 0.149})
    # The first label is the dominant action; the order matches descending
    # frequency.
    labels = [chunk.split(":")[0].strip() for chunk in out.split(",")]
    assert labels == ["Fold", "Call", "Raise"]


def test_action_frequencies_pure_strategy_sums_to_100() -> None:
    """A pure strategy ({Call: 1.0}) renders as "Call: 100%" and sums to 100."""
    from pipeline.preflop.format_writer import _format_action_frequencies

    out = _format_action_frequencies({"Call": 1.0, "Fold": 0.0, "Raise": 0.0})
    assert _sum_pcts(out) == 100
    assert out.startswith("Call: 100%")


def test_action_frequencies_empty_strategy_empty_string() -> None:
    """Empty strategy dict -> empty string column."""
    from pipeline.preflop.format_writer import _format_action_frequencies

    assert _format_action_frequencies({}) == ""


def test_action_frequencies_all_zero_strategy_empty_string() -> None:
    """All-zero strategy (hand never reaches node) -> empty string. A
    '0%' row carries no information and showing 'Fold: 0%, Call: 0%' as
    the action_frequencies cell would be confusing."""
    from pipeline.preflop.format_writer import _format_action_frequencies

    assert _format_action_frequencies({"Fold": 0.0, "Call": 0.0}) == ""


def test_action_frequencies_descending_percentages() -> None:
    """action_frequencies uses canonical labels ('3-bet', not 'Raise 308%')
    so the column doesn't read like 'Raise 308%: 40%' which players /
    reviewers parse as two different percent values."""
    row = build_preflop_row(
        _facing_open_facts(),
        _explanation(),
        pack=_pack(),
        difficulty=_difficulty(1500),
        number=1,
    )
    # Fixture is SB facing a BTN open -- hero's raise is the 2nd of the
    # hand, so 'Raise 308%' canonicalises to '3-bet'.
    assert row["action_frequencies"].startswith("Call: 60%")
    assert "3-bet: 40%" in row["action_frequencies"]
    # Raw Pio token must NOT leak into the column.
    assert "Raise 308%" not in row["action_frequencies"]


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
        difficulty=_difficulty(1500),
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
        difficulty=_difficulty(1500),
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
        difficulty=_difficulty(1500),
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
        difficulty=_difficulty(1500),
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
                (_facing_open_facts(), _explanation(), _difficulty(1500)),
                (_facing_3bet_facts(), _explanation(), _difficulty(2200)),
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
            [(_open_facts(), _explanation(), _difficulty(800))],
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
            difficulty=_difficulty(1500),
            number=1,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
