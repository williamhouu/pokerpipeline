"""Tests for the Layer 5 fact extractor -- equity, range, blockers, orchestration.

Run directly (`python tests/test_fact_extractor.py`) or under pytest. Every test
is pure: SpotContexts are built synthetically, so no PioSolver is needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor import extract_facts                     # noqa: E402
from pipeline.fact_extractor.equity import (                          # noqa: E402
    equity_vs_range, hand_equity, rank_hand,
)
from pipeline.fact_extractor.equity_range_blockers import (           # noqa: E402
    _board_top_combos, _range_weight_in, compute_equity_data,
    compute_range_data,
)
from pipeline.path_sampler import ActionOption, DecisionNode, SpotContext  # noqa: E402


def _context(board, hero_range, villain_range, *, pot=200.0, to_call=0.0,
             hero_is_oop=True):
    """A synthetic SpotContext (no solver) for testing the fact extractor."""
    street = {3: "flop", 4: "turn", 5: "river"}[len(board)]
    node = DecisionNode(
        node_id="r:0", node_type="OOP_DEC" if hero_is_oop else "IP_DEC",
        street=street, board=board, pot=pot, effective_stack=900.0,
        amount_to_call=to_call,
        hero_position="BB" if hero_is_oop else "BTN",
        villain_position="BTN" if hero_is_oop else "BB",
        hero_is_oop=hero_is_oop, parent_node_id="r", action_to_reach="",
        action_sequence=[("OOP", "check"), ("IP", "bet 50")],
        available_actions=["fold", "call"])
    actions = [ActionOption("fold", "r:0:f", 0.3, 0.0),
               ActionOption("call", "r:0:c", 0.7, 12.0)]
    return SpotContext(node=node, hero_range=hero_range,
                       villain_range=villain_range, actions=actions)


# --- hand evaluator ----------------------------------------------------------
def test_rank_hand_category_ordering():
    straight_flush = rank_hand(["As", "Ks", "Qs", "Js", "Ts"])
    quads = rank_hand(["Ac", "Ah", "Ad", "As", "Kh"])
    flush = rank_hand(["Ah", "Jh", "9h", "6h", "3h"])
    straight = rank_hand(["Ah", "Kd", "Qc", "Js", "Th"])
    assert straight_flush > quads > flush > straight
    # The category element settles every cross-category comparison.
    assert straight_flush[0] == 8 and quads[0] == 7 and flush[0] == 5


def test_rank_hand_tiebreakers_and_wheel():
    # Pair rank outranks kickers: KK beats QQ even with QQ holding A-K kickers.
    assert (rank_hand(["Kh", "Kd", "9c", "5s", "2h"])
            > rank_hand(["Qh", "Qd", "Ac", "Ks", "2h"]))
    # The wheel is a 5-high straight.
    assert rank_hand(["Ah", "2d", "3c", "4s", "5h"]) == (4, 5)
    # A 7-card hand picks its best five.
    assert rank_hand(["As", "Ks", "Qs", "Js", "Ts", "2c", "3d"]) == (8, 14)


# --- equity ------------------------------------------------------------------
def test_hand_equity_exact_river():
    river = ["As", "Kd", "9h", "4c", "2s"]
    # Trip aces vs a pair -> hero always wins (no cards left to come).
    assert hand_equity(["Ah", "Ac"], ["Qh", "Qc"], river) == 1.0
    # Both players play the board's straight -> a tie is 0.5 equity.
    straight_board = ["5h", "6s", "7d", "8c", "9h"]
    assert hand_equity(["2c", "3c"], ["2d", "3d"], straight_board) == 0.5


def test_hand_equity_flop_dominant():
    # An overpair crushes air on a dry flop.
    equity = hand_equity(["Ah", "Ad"], ["7c", "2c"], ["Ks", "8h", "3d"])
    assert equity > 0.80


def test_equity_vs_range():
    river = ["As", "Kd", "9h", "4c", "2s"]
    villain = {"QhQc": 1.0, "JsTs": 1.0, "8h7h": 1.0}
    hero_equity, per_combo = equity_vs_range(["Ah", "Ac"], villain, river)
    assert hero_equity == 1.0                       # trip aces beats them all
    assert all(eq == 0.0 for eq in per_combo.values())


# --- equity_data / range_data ------------------------------------------------
def test_compute_equity_data():
    ctx = _context(["As", "Kd", "9h", "4c", "2s"], {"AhAc": 1.0},
                   {"QhQc": 1.0}, pot=200.0, to_call=50.0)
    hero_equity, _ = equity_vs_range(["Ah", "Ac"], ctx.villain_range,
                                     ctx.node.board)
    equity_data = compute_equity_data(ctx, "AhAc", hero_equity)
    assert equity_data.hero_raw_equity_vs_continuing == 1.0
    assert equity_data.pot_odds_required == 0.25     # 50 to call into a 200 pot
    assert equity_data.mdf == 0.75


def test_compute_range_data():
    board = ["As", "Kd", "9h", "4c", "2s"]
    hero_range = {"AhAc": 1.0}
    villain_range = {"QhQc": 1.0, "JsTs": 1.0, "8h7h": 1.0}
    ctx = _context(board, hero_range, villain_range)
    _, per_combo = equity_vs_range(["Ah", "Ac"], villain_range, board)
    range_data = compute_range_data(ctx, "AhAc", per_combo)
    assert len(range_data.villain_range) == 3
    assert len(range_data.hero_range) == 1
    # Hero holds trip aces; villain's range has no value (premium/strong) hands.
    assert range_data.hero_strong_hand_count > 0
    assert range_data.villain_strong_hand_count == 0
    assert 0.0 <= range_data.hero_total_equity <= 1.0


# --- top-5% combos -----------------------------------------------------------
def test_board_top_combos_contains_made_nuts():
    # Dry As-Kd-9h-4c-2s river: nut combos are sets / two pair on aces+kings.
    board = ["As", "Kd", "9h", "4c", "2s"]
    top = _board_top_combos(board, pct=0.05)
    # AA combos (set of aces) sit at the top.
    assert frozenset(("Ah", "Ac")) in top
    assert frozenset(("Ah", "Ad")) in top
    # 7-deuce offsuit (air) does not.
    assert frozenset(("7h", "2c")) not in top
    # Size: 5% of C(47, 2) = 0.05 * 1081 ~= 54 combos.
    assert 40 <= len(top) <= 70, len(top)


def test_range_weight_in_top_combos():
    board = ["As", "Kd", "9h", "4c", "2s"]
    top = _board_top_combos(board, pct=0.05)
    # A hero range of mostly nut combos -- weight in top should be the full sum.
    hero_range = {"AhAc": 1.0, "AhAd": 0.5, "7h2c": 1.0}
    in_top = _range_weight_in(hero_range, top)
    # Aces are in top, 7-2 is not; weight = 1.0 + 0.5 = 1.5.
    assert in_top == 1.5


def test_compute_range_data_uses_universal_top_combos():
    """nut_advantage now reads the board's universal top-5% pool, not the
    premium-bucket count. On a dry A-K-9-4-2 board hero (with AA only) holds
    more universal top-5% weight than villain (with QQ/JT/87)."""
    board = ["As", "Kd", "9h", "4c", "2s"]
    hero_range = {"AhAc": 1.0}                          # set of aces -> in top 5%
    villain_range = {"QhQc": 1.0, "JsTs": 1.0, "8h7h": 1.0}   # none nutted
    ctx = _context(board, hero_range, villain_range)
    _, per_combo = equity_vs_range(["Ah", "Ac"], villain_range, board)
    rd = compute_range_data(ctx, "AhAc", per_combo)
    assert rd.hero_top_5pct_combos > 0
    assert rd.villain_top_5pct_combos == 0


# --- Ryan-feedback Fix 4: villain_top_value_combos --------------------------
def test_villain_top_value_combos_ranks_premium_first():
    """Premium-bucket classes outrank lower buckets when weight is similar.
    On 9-7-2: QQ (overpair = strong) vs JT (no-pair air); QQ should win.
    """
    board = ["9h", "7c", "2d"]
    hero_range = {"AhAc": 1.0}
    villain_range = {"QhQc": 1.0, "JsTs": 1.0, "8h6h": 1.0}
    ctx = _context(board, hero_range, villain_range)
    _, per_combo = equity_vs_range(["Ah", "Ac"], villain_range, board)
    rd = compute_range_data(ctx, "AhAc", per_combo)
    assert rd.villain_top_value_combos, "field should be populated"
    # Entries are sorted by total_weight * bucket_score desc.
    # QhQc is the only "overpair" combo -> strong-bucket entry should rank
    # ahead of the air/marginal-bucket no_pair entries.
    top_entry = rd.villain_top_value_combos[0]
    assert top_entry["bucket"] in ("premium", "strong"), \
        f"expected premium/strong on top, got {top_entry}"


def test_villain_top_value_combos_carries_emoji_examples():
    """example_combos are formatted in suit-emoji notation, matching the
    Question column convention (rank + ♠️ / ❤️ / ♦️ / ♣️)."""
    board = ["9h", "7c", "2d"]
    hero_range = {"AhAc": 1.0}
    villain_range = {"QhQc": 1.0, "QsQd": 0.5}
    ctx = _context(board, hero_range, villain_range)
    _, per_combo = equity_vs_range(["Ah", "Ac"], villain_range, board)
    rd = compute_range_data(ctx, "AhAc", per_combo)
    overpair_entries = [e for e in rd.villain_top_value_combos
                        if "overpair" in e["hand_class_label"]]
    assert overpair_entries, (
        f"expected at least one overpair entry, got "
        f"{rd.villain_top_value_combos}")
    examples = overpair_entries[0]["example_combos"]
    # Every example contains a suit emoji.
    assert all("♠" in c or "❤" in c or "♦" in c or "♣" in c for c in examples)
    # The dominant combo (QhQc has full weight 1.0) is in the example set.
    assert any("Q❤" in c and "Q♣" in c for c in examples)


def test_villain_top_value_combos_empty_when_no_range():
    """Empty villain range -> empty field (no crash, just no entries)."""
    board = ["As", "Kd", "9h"]
    hero_range = {"AhAc": 1.0}
    ctx = _context(board, hero_range, {})
    _, per_combo = equity_vs_range(["Ah", "Ac"], {}, board)
    rd = compute_range_data(ctx, "AhAc", per_combo)
    assert rd.villain_top_value_combos == []


# --- Ryan ask (May 2026): ip_range_snapshot / oop_range_snapshot ------------
def test_extract_facts_populates_range_snapshots():
    """extract_facts maps hero/villain to ip/oop based on hero_in_position
    and populates the 169-class snapshots from each player's combo range."""
    board = ["As", "Kd", "9h"]
    # hero is OOP (hero_is_oop=True), so hero -> oop_range_snapshot.
    hero_range = {"AhAc": 1.0, "QcQd": 0.5}
    villain_range = {"KhKs": 1.0, "JhJs": 1.0}
    ctx = _context(board, hero_range, villain_range)
    spot = extract_facts(ctx, hero_hand="AhAc")
    # 169-entry snapshots on both sides.
    assert len(spot.ip_range_snapshot) == 169
    assert len(spot.oop_range_snapshot) == 169
    # hero (OOP) had AA at full weight; AcAd is the only unblocked AA combo
    # (AhAc shares Ah... wait actually only AhAc is in the range). Mean of
    # the one entry = 1.0.
    assert spot.oop_range_snapshot["AA"] == 1.0
    # Hero's QQ class: QcQd at 0.5 weight, one combo present -> mean 0.5.
    assert spot.oop_range_snapshot["QQ"] == 0.5
    # IP (villain) has KK and JJ at full weight (KhKs and JhJs each).
    assert spot.ip_range_snapshot["KK"] == 1.0
    assert spot.ip_range_snapshot["JJ"] == 1.0
    # Classes absent from villain's range remain 0.
    assert spot.ip_range_snapshot["AA"] == 0.0


def test_extract_facts_snapshots_swap_when_hero_is_ip():
    """Mirror test: when hero is IP, hero_range -> ip_range_snapshot."""
    board = ["As", "Kd", "9h"]
    hero_range = {"AhAc": 1.0}
    villain_range = {"QhQd": 1.0}
    ctx = _context(board, hero_range, villain_range, hero_is_oop=False)
    spot = extract_facts(ctx, hero_hand="AhAc")
    assert spot.spot_metadata.hero_in_position is True
    # hero (IP) has AA -> shows up in ip_range_snapshot, NOT oop.
    assert spot.ip_range_snapshot["AA"] == 1.0
    assert spot.oop_range_snapshot["AA"] == 0.0
    # villain (OOP) has QQ -> shows up in oop_range_snapshot.
    assert spot.oop_range_snapshot["QQ"] == 1.0
    assert spot.ip_range_snapshot["QQ"] == 0.0


# --- extract_facts orchestration ---------------------------------------------
def test_extract_facts_populates_spotdata():
    board = ["As", "Kd", "9h", "4c", "2s"]
    hero_range = {"AhAc": 1.0, "AhKh": 1.0, "KhKs": 0.8, "9s9c": 0.6}
    villain_range = {"QhQc": 1.0, "JsTs": 1.0, "8h7h": 1.0, "6c5c": 1.0}
    spot = extract_facts(_context(board, hero_range, villain_range,
                                  pot=200.0, to_call=50.0))

    # equity_data and range_data are no longer empty.
    assert spot.equity_data.hero_raw_equity_vs_continuing == 1.0
    assert spot.equity_data.pot_odds_required == 0.25
    assert spot.range_data.villain_range and spot.range_data.hero_range
    assert spot.range_data.hero_total_equity > 0.55

    # hand class and board texture are filled.
    assert spot.hand_class is not None and spot.board_texture is not None

    # The tagger now fires on real fact data -- hero crushes villain here.
    assert spot.concept_tags, "no concept tags fired"
    assert "range_advantage_hero" in spot.concept_tags


def test_ev_gap_bb_convention():
    """Pins the ev_gap_bb formula: action EVs are CHIPS at each child node;
    `ev_gap_bb = (best - second) / big_blind`, where `big_blind = effective_stack
    / 100`. A silent regression here (forgetting the divide, dividing twice,
    using per-combo spreads instead of action means) would make every Layer 4
    decision wrong, so we check the math directly against a synthetic spot.
    """
    board = ["As", "Kd", "9h", "4c", "2s"]
    # Synthetic node: effective_stack=900 chips, so a 100bb solve has big_blind=9.
    # Action child-mean EVs: fold=0 chips, call=+12 chips (ours, via _context).
    ctx = _context(board, {"AhAc": 1.0}, {"QhQc": 1.0})
    spot = extract_facts(ctx, big_blind=9.0)
    # (12 - 0) / 9 == 1.333... bb. Use approx because we are dividing floats.
    assert abs(spot.decision_data.ev_gap_bb - (12.0 / 9.0)) < 1e-9
    # And per-action EVs in the data block are in bb, not chips.
    assert abs(spot.decision_data.range_mean_evs_per_action["call"] - (12.0 / 9.0)) < 1e-9
    assert spot.decision_data.range_mean_evs_per_action["fold"] == 0.0
    # Stack depth derives from the same big_blind: 900 / 9 == 100bb.
    assert abs(spot.spot_metadata.effective_stack_bb - 100.0) < 1e-9


def test_extract_facts_picks_most_likely_hero_hand():
    board = ["As", "Kd", "9h", "4c", "2s"]
    spot = extract_facts(_context(board, {"AhAc": 0.2, "7h7d": 0.9},
                                  {"QhQc": 1.0}))
    # 7h7d carries the most weight, so it is the hand the spot is built around.
    assert spot.hand_class.made_hand in ("two_pair_mid", "second_pair",
                                         "pocket_pair_below_overcards")


if __name__ == "__main__":
    suite = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in suite:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
