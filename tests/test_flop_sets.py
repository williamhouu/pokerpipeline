"""Tests for pipeline.flop_sets.

Pure unit tests -- no PioSolver. Locks the catalog shape so a future edit
that breaks 25-flop coverage or the MINIMAL_DEBUG reproduce-test target
fails loudly.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.flop_sets import (                                          # noqa: E402
    FLOP_SETS, MINIMAL_DEBUG, STANDARD_25_FLOPS, flop_board_string,
    flop_filename_stem, normalize_flop, select_flops,
)


# --- normalisation + format helpers -----------------------------------------
def test_normalize_flop_accepts_multiple_input_forms():
    expected = ("2c", "Js", "7s")
    assert normalize_flop(("2c", "Js", "7s")) == expected
    assert normalize_flop(["2c", "Js", "7s"]) == expected
    assert normalize_flop("2c Js 7s") == expected
    assert normalize_flop("2cJs7s") == expected
    # Lowercase rank normalised by parse_card.
    assert normalize_flop("2C jS 7s")[0] == "2c"


def test_normalize_flop_rejects_bad_input():
    for bad in (("2c", "Js"), ("2c", "Js", "7s", "8h"), ("2c", "Js", "2c")):
        try:
            normalize_flop(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_flop_board_string_uses_space_form():
    """Pio's set_board takes space-separated cards."""
    assert flop_board_string(("2c", "Js", "7s")) == "2c Js 7s"


def test_flop_filename_stem_is_filesystem_safe():
    """Cache paths can't have spaces; the stem is concatenated cards."""
    assert flop_filename_stem(("2c", "Js", "7s")) == "2cJs7s"
    assert flop_filename_stem(("As", "Kd", "9h")) == "AsKd9h"


# --- MINIMAL_DEBUG ----------------------------------------------------------
def test_minimal_debug_reproduces_existing_test_solve_board():
    """MINIMAL_DEBUG's single flop matches the existing hand-solved file
    test_solves/btn_vs_bb_srp_2cJs7s.cfr so we can compare structurally."""
    assert len(MINIMAL_DEBUG) == 1
    assert MINIMAL_DEBUG[0] == ("2c", "Js", "7s")
    assert flop_filename_stem(MINIMAL_DEBUG[0]) == "2cJs7s"


# --- STANDARD_25_FLOPS ------------------------------------------------------
def test_standard_25_has_25_distinct_flops():
    assert len(STANDARD_25_FLOPS) == 25
    # All distinct after normalisation.
    stems = {flop_filename_stem(f) for f in STANDARD_25_FLOPS}
    assert len(stems) == 25, f"duplicates in STANDARD_25: {len(stems)} unique stems"


def test_standard_25_no_card_repeats_within_a_flop():
    """Each flop has three different cards (no duplicates)."""
    for flop in STANDARD_25_FLOPS:
        cards = list(flop)
        assert len(set(cards)) == 3, f"duplicate card in {flop!r}"


def test_standard_25_covers_all_suit_distributions():
    """Mix of monotone, two-tone, and rainbow boards."""
    suit_dist: Counter = Counter()
    for flop in STANDARD_25_FLOPS:
        suits = Counter(c[1] for c in flop)
        most_common_count = max(suits.values())
        if most_common_count == 3:
            suit_dist["monotone"] += 1
        elif most_common_count == 2:
            suit_dist["two_tone"] += 1
        else:
            suit_dist["rainbow"] += 1
    # The sample should not skew degenerately toward one suit shape.
    assert suit_dist["monotone"] >= 2, suit_dist
    assert suit_dist["two_tone"] >= 8, suit_dist
    assert suit_dist["rainbow"] >= 6, suit_dist


def test_standard_25_covers_paired_and_unpaired():
    """Some paired boards, mostly unpaired."""
    paired = 0
    for flop in STANDARD_25_FLOPS:
        ranks = Counter(c[0] for c in flop)
        if max(ranks.values()) >= 2:
            paired += 1
    assert 2 <= paired <= 6, f"paired board count out of range: {paired}"


def test_standard_25_covers_rank_distribution():
    """High, middling, and low boards all represented."""
    rank_value = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                  "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    high = mid = low = 0
    for flop in STANDARD_25_FLOPS:
        top_rank = max(rank_value[c[0]] for c in flop)
        if top_rank >= 13:                       # K-high or higher
            high += 1
        elif top_rank >= 9:                      # 9-high to Q-high
            mid += 1
        else:                                    # 8-high or lower
            low += 1
    assert high >= 5
    assert mid >= 5
    assert low >= 3


# --- registry ---------------------------------------------------------------
def test_select_flops_known_set():
    assert select_flops("MINIMAL_DEBUG") == MINIMAL_DEBUG
    assert select_flops("STANDARD_25_FLOPS") == STANDARD_25_FLOPS


def test_select_flops_unknown_set_raises():
    try:
        select_flops("not_a_set")
    except KeyError as exc:
        assert "not_a_set" in str(exc)
        assert "MINIMAL_DEBUG" in str(exc)
        assert "STANDARD_25_FLOPS" in str(exc)
        return
    raise AssertionError("expected KeyError")


def test_flop_sets_registry_keys_match_module_constants():
    """Sanity: FLOP_SETS dict and the top-level constants agree."""
    assert FLOP_SETS["MINIMAL_DEBUG"] is MINIMAL_DEBUG
    assert FLOP_SETS["STANDARD_25_FLOPS"] is STANDARD_25_FLOPS


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
