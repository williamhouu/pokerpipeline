"""Tests for the July-2026 PLO audit-wave fixes: show-the-math columns,
multiway-awareness facts, the position-fact skill rules, and the limp
terminology validator (all found by the first deep 9-max batch audit)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import GeneratedExplanation  # noqa: E402
from pipeline.plo.action_history import call_price, resolve_pot_limit  # noqa: E402
from pipeline.plo.explanation_generator import build_solver_data  # noqa: E402
from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import (  # noqa: E402
    PloDecisionNode,
    plo_pending_after_hero,
)
from pipeline.plo.pack import (  # noqa: E402
    SEATS_9MAX,
    PloAction,
    PloActionType,
    parse_node_path,
)
from pipeline.plo.skill_tagger import compute_plo_skills  # noqa: E402
from pipeline.plo.validators import validate_terminology  # noqa: E402

R, C, F = PloActionType.RAISE, PloActionType.CALL, PloActionType.FOLD
CARDS = ("As", "Ks", "Ah", "Kh")


def _facts(
    history: tuple[PloAction, ...],
    actor: str,
    *,
    villain_seat: str | None = None,
    table_size: int = 9,
    freqs: dict[str, float] | None = None,
    hero_eq: float | None = None,
) -> PloFacts:
    from pipeline.plo.spot_sampler import PloSpot

    node = PloDecisionNode(
        actor=actor, history_before=history, actions=(), history_stem="",
        table_size=table_size,
    )
    spot = PloSpot(
        node=node, hero_index=0, hero_label="x", hero_cards=CARDS,
        action_frequencies=freqs or {"Call": 0.7, "Fold": 0.3},
        ev_by_action={"Call": 2.0, "Fold": -1.0},
        presence=1.0,
    )
    villain = (
        PloVillainStats(
            seat=villain_seat, action_label="Raise 100%",
            weighted_combo_count=1.0, pct_of_dealt_hands=4.0,
        )
        if villain_seat else None
    )
    return PloFacts(
        spot=spot, hand_class=classify_plo_hand(CARDS),
        archetype="call_for_value", villain_stats=villain,
        hero_equity_vs_villain=hero_eq,
    )


# --- call_price ---------------------------------------------------------------
def test_call_price_simple_open_defend():
    # UTG opens pot (3.5bb), folds to the BB: pot 5bb, BB owes 2.5bb.
    history = parse_node_path("2." + ".".join(["0"] * 7), seats=SEATS_9MAX)
    pot, to_call = call_price(history, "BB")
    assert pot == 5.0
    assert to_call == 2.5


def test_call_price_pot_matches_resolve_pot_limit():
    # The two walks share the arithmetic -- any drift is a bug.
    stem = "2.1.0.0.1.0.0.2.1.1"  # open, calls, SB 3-bet pot, calls around
    history = parse_node_path(stem, seats=SEATS_9MAX)
    _actions, pot_a = resolve_pot_limit(history)
    pot_b, _to_call = call_price(history, history[-1].seat)
    assert abs(pot_a - pot_b) < 1e-9


# --- pending-after-hero ---------------------------------------------------------
def test_pending_empty_when_hero_closes_the_action():
    # UTG opens, UTG+2 calls, BTN 3-bets, SB calls, everyone else folds ->
    # back on UTG+2 (UTG folded): nobody pending, a call closes the action.
    stem = "2.0.1.0.0.0.2.1.0.0"
    history = parse_node_path(stem, seats=SEATS_9MAX)
    node = PloDecisionNode(
        actor="UTG+2", history_before=history, actions=(), history_stem="",
        table_size=9,
    )
    still, closes = plo_pending_after_hero(node)
    assert still == ()
    assert closes is True


def test_pending_lists_seats_that_have_not_faced_the_raise():
    # UTG+2 opens, HJ + BTN call, BB 3-bets, UTG+2 folds -> HJ decides with
    # BTN still to act on the 3-bet behind them.
    history = (
        PloAction("UTG", F), PloAction("UTG+1", F),
        PloAction("UTG+2", R, 100), PloAction("LJ", F), PloAction("HJ", C),
        PloAction("CO", F), PloAction("BTN", C), PloAction("SB", F),
        PloAction("BB", R, 100), PloAction("UTG+2", F),
    )
    node = PloDecisionNode(
        actor="HJ", history_before=history, actions=(), history_stem="",
        table_size=9,
    )
    still, closes = plo_pending_after_hero(node)
    assert still == ("BTN",)
    assert closes is False


# --- SOLVER DATA additions --------------------------------------------------------
def test_solver_data_has_multiway_awareness_and_price():
    # BB defending vs a UTG open: no one behind, price = 2.5 into 5.
    history = parse_node_path("2." + ".".join(["0"] * 7), seats=SEATS_9MAX)
    facts = _facts(history, "BB", villain_seat="UTG", hero_eq=0.55)
    data = build_solver_data(facts, ["Fold", "Call"], "Call")
    assert data["still_to_act_behind_you"] == "nobody"
    assert data["your_call_or_fold_closes_the_action"] is True
    assert data["players_still_in_the_hand"] == 2
    assert data["price"]["pot_bb"] == 5.0
    assert data["price"]["to_call_bb"] == 2.5
    assert data["price"]["break_even_equity_pct"] == 33
    assert "ABOVE" in data["price"]["your_equity_vs_break_even"]


def test_solver_data_price_absent_when_not_facing_a_bet():
    facts = _facts((), "UTG", freqs={"Raise 100%": 0.7, "Fold": 0.3})
    data = build_solver_data(facts, ["Fold", "Raise"], "Raise")
    assert "price" not in data


# --- position-fact skill rules (the #6 mis-tag from the live audit) ---------------
def test_co_caller_facing_btn_3bet_is_out_of_position_play():
    # CO called an open, BTN 3-bet: hero is OOP vs the 3-bettor. The old
    # "late seat = IP" heuristic tagged this In Position Play.
    history = (
        PloAction("UTG+1", R, 100), PloAction("CO", C),
        PloAction("BTN", R, 100),
    )
    skills = compute_plo_skills(
        _facts(history, "CO", villain_seat="BTN", freqs={"Call": 1.0})
    )
    assert "Out of Position Play" in skills
    assert "In Position Play" not in skills


def test_btn_facing_sb_squeeze_is_in_position_play():
    history = (
        PloAction("UTG", R, 100), PloAction("BTN", C),
        PloAction("SB", R, 100),
    )
    skills = compute_plo_skills(
        _facts(history, "BTN", villain_seat="SB", freqs={"Call": 1.0})
    )
    assert "In Position Play" in skills
    assert "Out of Position Play" not in skills


# --- terminology validator ------------------------------------------------------
def _gen(prose: str) -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1="Fold", option_2="Call", option_3="", option_4="",
        correct_answer="Call", answer_explanation=prose,
    )


def test_limp_language_rejected_when_every_call_followed_a_raise():
    history = (PloAction("UTG", R, 100), PloAction("UTG+2", C),
               PloAction("LJ", R, 100))
    facts = _facts(history, "UTG+2", villain_seat="LJ", freqs={"Call": 1.0})
    result = validate_terminology(
        _gen("After limping in you now face a squeeze, so call."), facts
    )
    assert not result.is_valid
    assert "limp" in result.error_message


def test_limp_language_allowed_in_an_unraised_pot():
    # SB completing first-in: 'limp' is the correct word there.
    history = tuple(PloAction(s, F) for s in SEATS_9MAX[:7])  # folds to SB
    facts = _facts(history, "SB", freqs={"Call": 0.6, "Fold": 0.4})
    result = validate_terminology(
        _gen("Completing here is a profitable limp with this shape."), facts
    )
    assert result.is_valid


def test_no_limp_word_passes():
    history = (PloAction("UTG", R, 100),)
    facts = _facts(history, "BB", villain_seat="UTG", freqs={"Call": 1.0})
    assert validate_terminology(_gen("Calling is fine at this price."), facts).is_valid


# --- diversify (balanced action mix) -------------------------------------------
def _write_rng(path: Path, p: float) -> None:
    from pipeline.plo.hand_order import HAND_COUNT

    out = []
    for _ in range(HAND_COUNT):
        out.append("x")
        out.append(f"{p};0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_diversify_interleaves_action_contexts(tmp_path):
    """With one worthy OPEN node and many worthy facing-raise nodes, a
    diversified 2-question batch must include the open; the raw draw is free
    to (and with this seed does) take two facing-raise spots."""
    from pipeline.plo.batch import generate_plo_batch
    from pipeline.plo.pack import discover_plo_pack

    root = tmp_path / "ranges" / "Omaha" / "9-way" / "100bb"
    # Worthy UTG open decision (65/35 mix).
    _write_rng(root / "0.rng", 0.35)
    _write_rng(root / "2.rng", 0.65)
    # Several worthy facing-the-open nodes (UTG+1, and after a fold UTG+2).
    for prefix in ("2", "2.0"):
        _write_rng(root / f"{prefix}.0.rng", 0.3)
        _write_rng(root / f"{prefix}.1.rng", 0.7)
        _write_rng(root / f"{prefix}.2.rng", 0.0)
    pack = discover_plo_pack(tmp_path)

    def contexts(diversify: bool) -> list[str]:
        out = tmp_path / f"b_{diversify}.csv"
        generate_plo_batch(
            pack, output_path=out, total_questions=2, seed=3,
            compute_equity=False, diversify=diversify,
        )
        import json as _json

        meta = _json.loads(out.with_suffix(".meta.json").read_text())
        from pipeline.plo.node_enumerator import (
            enumerate_plo_nodes,
            plo_node_action_context,
        )

        nodes = {n.node_id: n for n in enumerate_plo_nodes(pack)}
        return [
            plo_node_action_context(nodes[q["node_id"]])
            for q in meta["questions"]
        ]

    assert "Opening" in contexts(True)  # the round-robin reaches the open
    meta = json.loads((tmp_path / "b_True.meta.json").read_text())
    assert meta["run_settings"]["diversify"] is True


# --- writer math columns -----------------------------------------------------------
def test_row_shows_the_math():
    from pipeline.plo.difficulty import compute_plo_difficulty
    from pipeline.plo.format_writer import build_plo_row

    history = parse_node_path("2." + ".".join(["0"] * 7), seats=SEATS_9MAX)
    facts = _facts(history, "BB", villain_seat="UTG", hero_eq=0.55)
    row = build_plo_row(
        facts, difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call"], correct_answer="Call", number=1,
    )
    assert row["pot_odds"] == "33.3%"
    assert row["hero_equity"] == "55.0%"
    assert row["action_ev_bb"] == "Call: 1.00, Fold: -0.50"  # sb -> bb
    notes = json.loads(row["stat_notes"])
    keys = {n["key"] for n in notes}
    assert {"pot_odds", "hero_equity"} <= keys
    pot_note = next(n for n in notes if n["key"] == "pot_odds")
    # Grid amounts print in the same clean form as the Question prose
    # ("5", never "5.0" -- July 23 2026).
    assert "2.5 / (5 + 2.5)" in pot_note["note"]
    assert "Facing 2.5bb into the 5bb pot" in pot_note["note"]
    # The Show-the-math value is the BARE percentage, not "need 33%" (team,
    # July 2026) -- whether to call at that price is the explanation's job.
    assert pot_note["value"] == "33%"
    assert "need" not in pot_note["value"]


def test_pot_odds_note_quotes_the_displayed_sizes_not_the_exact_walk():
    """USER RULE (July 23 2026): the pot-odds math must quote exactly the
    numbers the player sees in the Question. MTT pot-relative raise tokens
    resolve to off-grid sizes (a 72%-pot BB raise over a limp = 3.52bb, a
    92%-pot 3-bet over it = 11.38bb) which DISPLAY on the 0.5bb grid as
    3.5bb / 11.5bb -- so the note must say "Facing 8bb into the 16.5bb pot"
    (the subtraction the player does: 11.5 - 3.5), never the exact
    "7.9bb into the 16.4bb pot" that shipped and read as an error."""
    import json

    from pipeline.plo.action_history import call_price, display_call_price
    from pipeline.plo.difficulty import compute_plo_difficulty
    from pipeline.plo.format_writer import build_plo_row
    from pipeline.plo.pack import PloAction, PloActionType

    # The shipped P308 line: LJ/HJ fold, CO limps, BU/SB fold, BB raise72,
    # CO raise92, back on the BB (25bb MTT, 1bb ante).
    history = (
        PloAction("LJ", PloActionType.FOLD),
        PloAction("HJ", PloActionType.FOLD),
        PloAction("CO", PloActionType.CALL),
        PloAction("BU", PloActionType.FOLD),
        PloAction("SB", PloActionType.FOLD),
        PloAction("BB", PloActionType.RAISE, 72),
        PloAction("CO", PloActionType.RAISE, 92),
    )
    # The exact walk stays exact (strategic gates keep using it)...
    pot_exact, call_exact = call_price(
        history, "BB", stack_bb=25.0, ante_bb=1.0
    )
    assert round(pot_exact, 2) == 16.4
    assert round(call_exact, 2) == 7.86
    # ...while the displayed price matches the Question's rounded sizes.
    disp_pot, disp_call, break_even = display_call_price(
        history, "BB", stack_bb=25.0, ante_bb=1.0
    )
    assert (disp_pot, disp_call) == (16.5, 8.0)
    assert round(break_even * 100) == 33

    facts = _facts(history, "BB", villain_seat="CO", table_size=6, hero_eq=0.55)
    row = build_plo_row(
        facts, difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call", "All-in"], correct_answer="Call", number=1,
        stack_bb=25.0, ante_bb=1.0, game_format="tournament",
    )
    pot_note = next(
        n for n in json.loads(row["stat_notes"]) if n["key"] == "pot_odds"
    )
    assert pot_note["note"] == (
        "Facing 8bb into the 16.5bb pot: "
        "break-even equity = 8 / (16.5 + 8) = 33%."
    )
    assert pot_note["value"] == "33%"


def test_range_width_renders_right_after_pot_odds(monkeypatch):
    # User feedback: "what percent of hands are played" as a Show-the-math row,
    # right below Pot odds. The share comes from a pure frequency read; here we
    # stub it so the test pins the WIRING (position + wording), not the pack.
    from pipeline.plo import format_writer
    from pipeline.plo.difficulty import compute_plo_difficulty
    from pipeline.plo.format_writer import build_plo_row

    monkeypatch.setattr(
        format_writer, "hero_range_action_shares",
        lambda facts: ({"Call": 0.13, "Fold": 0.87}, 100.0),
    )
    history = parse_node_path("2." + ".".join(["0"] * 7), seats=SEATS_9MAX)
    facts = _facts(history, "BB", villain_seat="UTG",
                   freqs={"Call": 0.7, "Fold": 0.3}, hero_eq=0.55)
    row = build_plo_row(
        facts, difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call"], correct_answer="Call", number=1,
    )
    notes = json.loads(row["stat_notes"])
    keys = [n["key"] for n in notes]
    assert "pot_odds" in keys and "range_width" in keys
    assert keys.index("range_width") == keys.index("pot_odds") + 1  # right after
    rw = next(n for n in notes if n["key"] == "range_width")
    assert rw["value"] == "13%"
    # Facing a raise -> denominator is the reaching range, not all hands.
    assert "the hands you reach this spot with" in rw["note"]
    assert "13% call here" in rw["note"]


def test_range_width_open_says_all_starting_hands(monkeypatch):
    # A first-in (all-fold history) spot: the denominator is all starting hands
    # (the classic RFI width the feedback describes), and with no call on the
    # menu there is no Pot odds row, so Range width leads.
    from pipeline.plo import format_writer
    from pipeline.plo.difficulty import compute_plo_difficulty
    from pipeline.plo.format_writer import build_plo_row

    monkeypatch.setattr(
        format_writer, "hero_range_action_shares",
        lambda facts: ({"Raise": 0.18, "Fold": 0.82}, 200.0),
    )
    history = parse_node_path(".".join(["0"] * 3), seats=SEATS_9MAX)  # folds to LJ
    facts = _facts(history, "LJ", freqs={"Raise": 0.9, "Fold": 0.1})
    row = build_plo_row(
        facts, difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Raise"], correct_answer="Raise", number=1,
    )
    notes = json.loads(row["stat_notes"])
    keys = [n["key"] for n in notes]
    assert "pot_odds" not in keys  # no call to make -> no price row
    rw = next(n for n in notes if n["key"] == "range_width")
    assert rw["value"] == "18%"
    assert "all starting hands" in rw["note"]


def test_row_action_evs_use_canonical_labels():
    """The EV cell must speak the same labels as the rest of the row: the raw
    'Raise 100%' EV shows up as the canonical '3-bet' when facing one raise
    (the first live batch dropped every raise EV over this mismatch)."""
    from pipeline.plo.difficulty import compute_plo_difficulty
    from pipeline.plo.format_writer import build_plo_row
    from pipeline.plo.spot_sampler import PloSpot

    history = (PloAction("UTG", R, 100),)
    node = PloDecisionNode(
        actor="BB", history_before=history, actions=(), history_stem="",
        table_size=9,
    )
    spot = PloSpot(
        node=node, hero_index=0, hero_label="x", hero_cards=CARDS,
        action_frequencies={"Raise 100%": 0.8, "Fold": 0.1, "Call": 0.1},
        ev_by_action={"Raise 100%": 4.0, "Fold": -2.0, "Call": 1.0},
        presence=1.0,
    )
    facts = PloFacts(
        spot=spot, hand_class=classify_plo_hand(CARDS),
        archetype="3bet_for_value",
        villain_stats=PloVillainStats(
            seat="UTG", action_label="Raise 100%",
            weighted_combo_count=1.0, pct_of_dealt_hands=10.0,
        ),
    )
    row = build_plo_row(
        facts, difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Call", "3-bet"], correct_answer="3-bet", number=1,
    )
    assert "3-bet: 2.00" in row["action_ev_bb"]  # 4.0 sb -> 2.0 bb
    assert "Fold: -1.00" in row["action_ev_bb"]


def test_open_spot_has_no_pot_odds_cells():
    """A non-SB open decision has no call on the menu -- pot_odds and the
    stat_notes pot-odds entry must be blank (the SOLVER DATA price block
    already suppresses it; the shared price_is_live rule keeps them agreeing)."""
    from pipeline.plo.difficulty import compute_plo_difficulty
    from pipeline.plo.format_writer import build_plo_row

    facts = _facts((), "UTG", freqs={"Raise 100%": 0.7, "Fold": 0.3})
    row = build_plo_row(
        facts, difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Raise"], correct_answer="Raise", number=1,
    )
    assert row["pot_odds"] == ""
    assert "pot_odds" not in row["stat_notes"]


# --- flush ceiling + dead weight panel rows (July 16 2026, ideas 1+2) --------
# INVARIANTS pinned here: (a) flush_ceiling always emits >= 1 entry (the
# no-flush row when no two cards share a suit); (b) dead_weight emits at most
# ONE entry (rank vs suit redundancy are mutually exclusive by construction)
# and NOTHING for plain pairs; (c) every claim is a ceiling claim about
# hero's own hand, sourced from the same flush_suits/describe_* vocabulary
# the SOLVER DATA block uses.
from pipeline.plo.hand_model import (  # noqa: E402
    dead_weight_stat_entries,
    describe_card_redundancy,
    describe_suit_redundancy,
    flush_ceiling_stat_entries,
)


def _by_key(entries):
    return {e["key"]: e for e in entries}


def test_flush_ceiling_rainbow_emits_the_no_flush_row():
    # July 2026 user wording: value is the terse "NA"; the note explains in
    # plain English (rainbow hand + the two-cards-one-suit rule).
    entries = flush_ceiling_stat_entries(("As", "Kd", "Qh", "Jc"))
    assert len(entries) == 1
    assert entries[0]["value"] == "NA"
    assert "rainbow hand" in entries[0]["note"]
    assert "can never make one" in entries[0]["note"]


def test_flush_ceiling_double_suited_orders_high_suit_first():
    # The real plo9_factor_list_check #4 hand: nut diamonds + weak clubs.
    entries = flush_ceiling_stat_entries(("9c", "Jc", "8d", "Ad"))
    assert len(entries) == 2
    assert entries[0]["key"] == "flush_ceiling_diamonds"
    assert entries[0]["value"] == "nut flush possible"
    assert "A♦ 8♦" in entries[0]["label"]
    assert entries[1]["key"] == "flush_ceiling_clubs"
    assert entries[1]["value"] == "J-high flush at best (weak)"
    assert "J♣ 9♣" in entries[1]["label"]


def test_flush_ceiling_ladder_labels_match_flush_suits_vocabulary():
    king = flush_ceiling_stat_entries(("Kc", "Kd", "4c", "4d"))
    assert all(e["value"] == "second-nut flush at best" for e in king)
    queen = flush_ceiling_stat_entries(("5c", "Qd", "5h", "Qh"))
    assert queen[0]["value"] == "third-nut flush at best"


def test_flush_ceiling_three_suited_shows_only_the_two_highest():
    # The real plo9_audit_batch2 #7 hand: the 3rd diamond belongs to the
    # dead-weight row, never the ceiling label.
    entries = flush_ceiling_stat_entries(("3c", "Td", "Jd", "Qd"))
    assert len(entries) == 1
    assert "Q♦ J♦" in entries[0]["label"]
    assert "T♦" not in entries[0]["label"]


def test_flush_ceiling_never_claims_unbeatable():
    # Ceiling claims only: a straight flush outranks even the nut flush, so
    # unbeatability wording must never appear.
    for hand in [("9c", "Jc", "8d", "Ad"), ("As", "Ks", "Qh", "Jc")]:
        for e in flush_ceiling_stat_entries(hand):
            assert "beat" not in e["note"].lower()
            assert "unbeatable" not in e["note"].lower()


def test_dead_weight_trips_counts_one_redundant_card():
    entries = dead_weight_stat_entries(("Qs", "Qd", "Qh", "7c"))
    assert len(entries) == 1
    assert entries[0]["label"] == "Dead weight: three queens"
    assert entries[0]["value"] == "one card is redundant"
    # The note is the exact SOLVER DATA sentence, sentence-cased.
    src = describe_card_redundancy(("Qs", "Qd", "Qh", "7c"))
    assert entries[0]["note"] == src[:1].upper() + src[1:]


def test_dead_weight_trips_with_suited_fourth_names_no_specific_card():
    # QsQdQh7h: the Q♥ carries flush value its siblings don't, so the row
    # must count the redundancy without striking a named card.
    entries = dead_weight_stat_entries(("Qs", "Qd", "Qh", "7h"))
    assert len(entries) == 1
    for glyph in ("♠", "♦", "♥", "♣"):
        assert glyph not in entries[0]["label"] + entries[0]["value"]
    # And the hearts flush row still fires alongside it.
    flush = _by_key(flush_ceiling_stat_entries(("Qs", "Qd", "Qh", "7h")))
    assert "flush_ceiling_hearts" in flush


def test_dead_weight_quads_and_monotone():
    quads = dead_weight_stat_entries(("Qs", "Qd", "Qh", "Qc"))
    assert quads[0]["value"] == "two cards are redundant, no set possible"
    mono = dead_weight_stat_entries(("3d", "Td", "Jd", "Qd"))
    assert mono[0]["label"] == "Dead weight: four diamonds"
    assert mono[0]["value"] == "two cards add no flush value"
    src = describe_suit_redundancy(("3d", "Td", "Jd", "Qd"))
    assert mono[0]["note"] == src[:1].upper() + src[1:]


def test_dead_weight_never_fires_on_plain_pairs():
    for hand in [
        ("Kc", "Kd", "4c", "4d"),  # two pair
        ("5c", "Qd", "5h", "Qh"),  # double-paired double-suited
        ("As", "Ks", "Qh", "Jc"),  # unpaired
    ]:
        assert dead_weight_stat_entries(hand) == []


def test_dead_weight_emits_at_most_one_row():
    # Rank vs suit redundancy are mutually exclusive by construction (trips
    # occupy three different suits, quads four); pin it across shapes.
    for hand in [
        ("Qs", "Qd", "Qh", "7h"),
        ("3c", "Td", "Jd", "Qd"),
        ("3d", "Td", "Jd", "Qd"),
        ("Qs", "Qd", "Qh", "Qc"),
        ("7d", "Qd", "Ad", "7c"),  # suit redundancy + a pair
    ]:
        assert len(dead_weight_stat_entries(hand)) <= 1


def test_row_stat_notes_carry_flush_ceiling_and_dead_weight():
    """build_plo_row integration: the new panel rows ship in stat_notes on
    every row (flush ceiling always; dead weight when the hand has it)."""
    from pipeline.plo.difficulty import compute_plo_difficulty
    from pipeline.plo.format_writer import build_plo_row

    facts = _facts((), "UTG", freqs={"Raise 100%": 0.7, "Fold": 0.3})
    row = build_plo_row(
        facts, difficulty=compute_plo_difficulty(facts),
        options=["Fold", "Raise"], correct_answer="Raise", number=1,
    )
    notes = json.loads(row["stat_notes"])
    keys = [n["key"] for n in notes]
    # CARDS = As Ks Ah Kh: double-suited, nut ceiling in both suits.
    assert keys.count("flush_ceiling_spades") == 1
    assert keys.count("flush_ceiling_hearts") == 1
    assert "dead_weight" not in keys
    by_key = {n["key"]: n for n in notes}
    assert by_key["flush_ceiling_spades"]["value"] == "nut flush possible"
