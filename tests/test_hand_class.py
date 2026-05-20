"""Tests for pipeline.fact_extractor.hand_class.

Run directly (`python tests/test_hand_class.py`) or under pytest. Cases cover
every made-hand category, every draw type, the strength buckets, the composite
label, and the Phase 0 test-solve board (2c Js 7s).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fact_extractor.hand_class import (        # noqa: E402
    DRAW_TYPES, MADE_HAND_CATEGORIES, STRENGTH_BUCKETS, classify_hand,
)


def _made(hole, board):
    return classify_hand(hole, board)["made_hand"]


def test_made_hand_categories():
    cases = {
        ("9h Th", "Jh Qh Kh"): "straight_flush",
        ("7c 7d", "7h 7s Kc"): "quads",
        ("7c 7d", "7h Ks Kc"): "full_house_set_plus_board",
        ("9c 9d", "7h 7s 7c"): "full_house_trips_plus_pocket",
        ("Ah 2h", "Jh Th 9h"): "flush_nut",
        ("Kh 2h", "Jh Th 9h"): "flush_second_nut",
        ("Qh 2h", "Jh Th 9h"): "flush_weak",
        ("9c Tc", "8h 7s 6d"): "straight_nut",
        ("5c 4d", "8h 7s 6d"): "straight_weak",
        ("7c 7d", "7h Ks 2c"): "set",
        ("7c 3d", "7h 7s Kc"): "trips",
        ("Ac Kd", "As Kh 5c"): "two_pair_top",
        ("Ac 5d", "As Kh 5c"): "two_pair_top_and_bottom",
        ("Kc 5d", "As Kh 5c"): "two_pair_mid",
        ("Ah Ad", "8c 5h 2d"): "overpair",
        ("Kc Ad", "Ks 8h 3c"): "top_pair_top_kicker",
        ("Kc Qd", "Ks 8h 3c"): "top_pair_good_kicker",
        ("Kc 7d", "Ks 8h 3c"): "top_pair_weak_kicker",
        ("7c 7d", "Kc Qh 2s"): "pocket_pair_below_overcards",
        ("8c 9d", "Ks 8h 3c"): "second_pair",
        ("8c 2h", "Ks Qd 8h 3c"): "third_pair",
        ("3c 9d", "Ks 8h 3d"): "bottom_pair",
        ("Ac 9d", "Ks 8h 3c"): "ace_high",
        ("Jc 9d", "Ks 8h 3c"): "no_pair_air",
    }
    for (hole, board), expected in cases.items():
        got = _made(hole, board)
        assert got == expected, f"{hole} on {board}: expected {expected}, got {got}"
    # Every one of the brief's 24 categories is exercised above.
    assert set(cases.values()) == set(MADE_HAND_CATEGORIES)


def test_draws():
    def draws(hole, board):
        return classify_hand(hole, board)["draws"]

    assert "flush_draw_nut" in draws("Ah 5h", "Jh Th 2c")
    assert "flush_draw_weak" in draws("Qh 5h", "Jh Th 2c")
    assert "straight_draw_open_ended" in draws("9c Tc", "8h 7s 2d")
    assert "gutshot" in draws("8d 7c", "Jh Th 2c")
    assert "combo_draw" in draws("9h Th", "8h 7h 2c")
    assert "backdoor_flush_draw" in draws("Ah 9c", "Kh 5h 2s")
    assert "backdoor_straight_draw" in draws("9d 8c", "Kh 7s 2c")
    # No draws on a complete (river) hand.
    assert draws("Ah Ad", "Kc Qd 7s 4h 2c") == []
    # combo_draw co-exists with its component draws for filtering.
    combo = draws("9h Th", "8h 7h 2c")
    assert "flush_draw_weak" in combo and "straight_draw_open_ended" in combo


def test_strength_buckets():
    def bucket(hole, board):
        return classify_hand(hole, board)["strength_bucket"]

    assert bucket("7c 7d", "7h Ks 2c") == "premium"      # set
    assert bucket("Ah Ad", "8c 5h 2d") == "strong"       # overpair, dry board
    assert bucket("Kc Qd", "Ks 8h 3c") == "medium"       # top pair good kicker
    assert bucket("8c 9d", "Ks 8h 3c") == "vulnerable"   # second pair
    assert bucket("3c 9d", "Ks 8h 3d") == "marginal"     # bottom pair
    assert bucket("Jc 9d", "Ks 8h 3c") == "air"          # no pair


def test_weak_overpair_on_wet_board_is_vulnerable():
    # v1 refinement: a weak overpair (JJ or lower) on a wet board drops to
    # vulnerable; a high overpair, or any overpair on a dry board, does not.
    assert classify_hand("9c 9d", "8c 7c 6d")["strength_bucket"] == "vulnerable"
    assert classify_hand("Ah Ad", "8c 7c 6d")["strength_bucket"] == "strong"
    assert classify_hand("9c 9d", "8h 5s 2d")["strength_bucket"] == "strong"


def test_label_format():
    assert classify_hand("Ah Ad", "8c 5h 2d")["label"] == "overpair_no_draws"
    assert classify_hand("Kc Ad", "Ks 8h 3c")["label"] == "top_pair_top_kicker_no_draws"
    assert classify_hand("9h Th", "8h 7h 2c")["label"] == "no_pair_air_with_combo_draw"
    assert classify_hand("8d 7c", "Jh Th 2c")["label"] == "no_pair_air_with_gutshot"


def test_phase0_test_solve_board():
    # 2c Js 7s -- the BTN-vs-BB SRP flop used for the Phase 0 verification.
    result = classify_hand("Jh Td", "2c Js 7s")
    assert result["made_hand"] == "top_pair_good_kicker"
    assert result["strength_bucket"] == "medium"


def test_output_structure():
    # Shape and value domains hold across a spread of hands.
    boards = ["2c Js 7s", "Jh Th 9h", "8c 7c 6d", "As Kd Qc 7h 2s"]
    holes = ["Ah Ad", "Jh Td", "9c 8c", "7s 2h", "Ah Kh"]
    checked = 0
    for board in boards:
        for hole in holes:
            try:
                result = classify_hand(hole, board)
            except ValueError:
                continue                       # hole card clashes with this board
            checked += 1
            assert set(result) == {"made_hand", "draws", "strength_bucket", "label"}
            assert result["made_hand"] in MADE_HAND_CATEGORIES
            assert result["strength_bucket"] in STRENGTH_BUCKETS
            assert all(d in DRAW_TYPES for d in result["draws"])
            assert result["label"].startswith(result["made_hand"])
    assert checked >= 12


def test_accepts_list_and_string_input():
    assert classify_hand(["Ah", "Ad"], ["8c", "5h", "2d"]) == \
        classify_hand("Ah Ad", "8c 5h 2d") == classify_hand("AhAd", "8c 5h 2d")


def test_rejects_invalid_input():
    bad = [
        ("Ah Ah", "8c 5h 2d"),    # duplicated hole card
        ("Ah Kd Qc", "8c 5h 2d"), # three hole cards
        ("Ah Kd", "8c 5h"),       # two-card board
        ("8c 5h", "8c 5h 2d"),    # hole card also on the board
    ]
    for hole, board in bad:
        try:
            classify_hand(hole, board)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for hole={hole!r} board={board!r}")


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
