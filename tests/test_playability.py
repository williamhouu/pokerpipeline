"""Deterministic hand-playability profile (the 'wheelish straights' fix)."""
from pipeline.preflop.playability import hand_playability


def test_a9s_is_nut_flush_disconnected_weak_kicker():
    # The bug: the LLM called A9s "wheelish". A9s is disconnected (the ace and
    # the nine can never share a straight), its only value is the nut flush,
    # and its top pair (aces) is dominated by a weak kicker.
    p = hand_playability("A9s")
    assert p["flush_potential"] == "nut_flush"
    assert p["straight_potential"] == "disconnected"
    assert p["makes_wheel_straight"] is False
    assert p["makes_broadway_straight"] is False
    assert p["domination_risk"] == "ace_weak_kicker"
    assert p["primary_value"] == ["the nut flush"]


def test_wheel_aces_make_the_wheel():
    for hc in ("A2s", "A3s", "A4s", "A5s"):
        p = hand_playability(hc)
        assert p["makes_wheel_straight"] is True, hc
        assert p["makes_broadway_straight"] is False, hc
        assert p["flush_potential"] == "nut_flush", hc
        assert "the wheel" in p["primary_value"], hc
    # A2s is connected to the wheel (A-2 adjacent); A5s is a three-gapper to it.
    assert hand_playability("A2s")["straight_potential"] == "open_ender"
    assert hand_playability("A5s")["straight_potential"] == "three_gapper"


def test_broadway_hands_make_the_nut_straight():
    for hc in ("AKs", "AKo", "KQs", "ATs", "JTs", "QTo"):
        p = hand_playability(hc)
        assert p["makes_broadway_straight"] is True, hc
        assert p["makes_wheel_straight"] is False, hc
    aks = hand_playability("AKs")
    assert aks["flush_potential"] == "nut_flush"
    assert "the nut straight (Broadway)" in aks["primary_value"]
    assert aks["top_pair_kicker"] == "top"


def test_pairs_make_sets_not_straights():
    for hc in ("AA", "77", "22"):
        p = hand_playability(hc)
        assert p["makes_sets"] is True, hc
        assert p["straight_potential"] == "none", hc
        assert p["flush_potential"] == "none", hc
        assert p["primary_value"] == ["sets"], hc


def test_suited_connector_makes_flush_and_straights():
    p = hand_playability("76s")
    assert p["flush_potential"] == "non_nut_flush"
    assert p["straight_potential"] == "open_ender"
    assert p["makes_broadway_straight"] is False
    assert "a flush (not the nuts)" in p["primary_value"]
    assert "straights" in p["primary_value"]


def test_gappers_and_king_domination():
    assert hand_playability("T8s")["straight_potential"] == "one_gapper"
    assert hand_playability("J9s")["straight_potential"] == "one_gapper"
    k9 = hand_playability("K9o")          # K-9: span 4 -> three-gapper (makes 9TJQK)
    assert k9["straight_potential"] == "three_gapper"
    assert k9["domination_risk"] == "king_weak_kicker"
    assert k9["flush_potential"] == "none"


def test_offsuit_junk_has_no_flush():
    p = hand_playability("72o")
    assert p["flush_potential"] == "none"
    assert p["suited"] is False
    # 7-2 is disconnected (span 5)
    assert p["straight_potential"] == "disconnected"


def test_unrecognized_label_is_flagged_not_crashed():
    assert hand_playability("XYZ") == {"recognized": False}
    assert hand_playability("")["recognized"] is False
