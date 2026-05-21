"""Tests for pipeline.fact_extractor.board_texture.

Run directly (`python tests/test_board_texture.py`) or under pytest. Cases are
drawn from the worked examples in docs/engineering_brief.docx plus the Phase 0
test-solve board (2c Js 7s).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cards import parse_board                      # noqa: E402
from pipeline.fact_extractor.board_texture import classify_board  # noqa: E402


def test_composite_matches_brief_examples():
    # Every labelled composite example in the brief.
    cases = {
        "As 7h 2d": "dry",        # rainbow, disconnected, draw-light
        "Kc 8c 3d": "semi_wet",   # unconnected two-tone
        "Jh Th 9c": "wet",        # connected two-tone, broadway
        "Jh Th 9h": "very_wet",   # monotone
        "8c 7c 6d": "dynamic",    # connected two-tone, low
        "As Kh 2d": "static",     # high rainbow, no draws
        "Ks Kc 4d": "static",     # paired rainbow
    }
    for board, expected in cases.items():
        got = classify_board(board)["composite"]
        assert got == expected, f"{board}: expected {expected}, got {got}"


def test_suit_distribution():
    # Flop: 'monotone' is only used for the all-3-same-suit case.
    assert classify_board("Jh Th 9h")["suit_distribution"] == "monotone"
    assert classify_board("Jh Th 9c")["suit_distribution"] == "two_tone"
    assert classify_board("As Kh 2d")["suit_distribution"] == "rainbow"
    # Turn / river: 4+ of one suit is a board flush; exactly 3 is a flush
    # threat (three_of_suit) -- NOT 'monotone'; 2 is two_tone; 1 is rainbow.
    assert classify_board("Jh Th 9h 2h")["suit_distribution"] == "flush_on_board"
    assert classify_board("Jh Th 9h 2c")["suit_distribution"] == "three_of_suit"
    # The runout from the test solve: 2c Js 7s 8h As (3 spades, 2 off suit)
    # is three_of_suit on the river, not monotone.
    assert classify_board("2c Js 7s 8h As")["suit_distribution"] == "three_of_suit"
    assert classify_board("Jh Th 9c 8d")["suit_distribution"] == "two_tone"
    assert classify_board("Jh Tc 9d 8s")["suit_distribution"] == "rainbow"
    # 5-card boards.
    assert classify_board("Jh Th 9h 2h 5h")["suit_distribution"] == "flush_on_board"
    assert classify_board("Jh Th 9h 2c 5d")["suit_distribution"] == "three_of_suit"


def test_pair_status():
    assert classify_board("2c Js 7s")["pair_status"] == "unpaired"
    assert classify_board("Ks Kc 4d")["pair_status"] == "paired"
    assert classify_board("7c 7d 7h")["pair_status"] == "trips_on_board"


def test_connectedness():
    assert classify_board("8c 7c 6d")["connectedness"] == "connected"
    assert classify_board("Ac 2d 3h")["connectedness"] == "connected"   # wheel
    assert classify_board("7c 9d Tc")["connectedness"] == "one_gap"
    assert classify_board("7c Td Jc")["connectedness"] == "two_gap"
    assert classify_board("As 7h 2d")["connectedness"] == "disconnected"


def test_rank_distribution():
    assert classify_board("As 7h 2d")["rank_distribution"] == "ace_high"
    assert classify_board("Kh Qd Jc")["rank_distribution"] == "high"
    assert classify_board("Jh Th 9c")["rank_distribution"] == "broadway_heavy"
    assert classify_board("Ks 8d 3c")["rank_distribution"] == "king_high"
    assert classify_board("8c 7c 6d")["rank_distribution"] == "low"
    assert classify_board("2c Js 7s")["rank_distribution"] == "middling"


def test_phase0_test_solve_board():
    # 2c Js 7s -- the BTN-vs-BB SRP flop used for the Phase 0 verification.
    texture = classify_board("2c Js 7s")
    assert texture == {
        "suit_distribution": "two_tone",
        "pair_status": "unpaired",
        "connectedness": "disconnected",
        "rank_distribution": "middling",
        "composite": "semi_wet",
    }


def test_accepts_list_and_string_input():
    assert classify_board(["2c", "Js", "7s"]) == classify_board("2c Js 7s")


def test_rejects_invalid_boards():
    for bad in ("Ah Kd", "Ah Kd Qc Js Ts 9h", "Ah Ah Kd"):
        try:
            classify_board(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for board: {bad!r}")
    # parse_board is the gate for the duplicate / count checks above
    assert len(parse_board("2c Js 7s")) == 3


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
