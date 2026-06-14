"""Deterministic preflop domination classifier + map."""
from pipeline.preflop.domination import (
    AHEAD,
    BEHIND,
    DOMINATES_YOU,
    FLIP,
    YOU_DOMINATE,
    classify_matchup,
    dominating_map,
)


def test_weak_ace_is_dominated_by_better_aces_and_set():
    # A9s: the textbook fold reason -- crushed by better aces and AA.
    for v in ("AKo", "AKs", "AQs", "AJo", "ATs"):
        assert classify_matchup("A9s", v) == DOMINATES_YOU, v
    assert classify_matchup("A9s", "AA") == DOMINATES_YOU      # set/overpair on the ace
    assert classify_matchup("A9s", "99") == DOMINATES_YOU      # villain pairs your nine


def test_weak_ace_is_behind_a_bigger_pair_but_not_dominated():
    # The lesson the classifier must preserve: vs QQ/KK the ace is LIVE, so
    # this is "behind", not structural domination.
    assert classify_matchup("A9s", "QQ") == BEHIND
    assert classify_matchup("A9s", "KK") == BEHIND
    assert classify_matchup("A9s", "TT") == BEHIND


def test_you_dominate_weaker_versions():
    assert classify_matchup("AKo", "AQs") == YOU_DOMINATE      # share A, K>Q
    assert classify_matchup("A9s", "A5s") == YOU_DOMINATE      # share A, 9>5
    assert classify_matchup("A9s", "T9s") == YOU_DOMINATE      # share 9, A>T
    assert classify_matchup("99", "A9o") == YOU_DOMINATE       # your set vs his pair-of-9s


def test_flips_and_live_matchups():
    assert classify_matchup("99", "AKo") == FLIP               # pair vs two overcards
    assert classify_matchup("A9s", "KQo") == AHEAD             # ace high, no share
    assert classify_matchup("KQo", "AKo") == DOMINATES_YOU     # share K, A>Q


def test_dominating_map_buckets_a9s_vs_a_tight_3bet_range():
    villain = ["AA", "KK", "AKs", "AKo", "AQs", "QQ", "A5s"]
    m = dominating_map("A9s", villain)
    assert "AA" in m["dominated_by"] and "AKs" in m["dominated_by"]
    assert "AQs" in m["dominated_by"] and "AKo" in m["dominated_by"]
    assert "A5s" in m["you_dominate"]
    assert "QQ" not in m["dominated_by"]          # behind, not dominated -> not listed here
    assert "KK" not in m["dominated_by"]


def test_dominating_map_dedupes_and_caps_order_preserved():
    villain = ["AKo", "AKo", "AQs"]
    m = dominating_map("A9s", villain)
    assert m["dominated_by"] == ["AKo", "AQs"]    # deduped, order preserved


def test_malformed_labels_are_ignored():
    assert classify_matchup("A9s", "ZZ") is None
    m = dominating_map("A9s", ["AKo", "garbage", "A5s"])
    assert m["dominated_by"] == ["AKo"]
    assert m["you_dominate"] == ["A5s"]
