"""Tests for pipeline.plo.position (IP/OOP standing)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.position import hero_relative_position, ip_oop_positions  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

CARDS = ("As", "Ks", "Ah", "Kh")


def _facts(actor: str, *, villain_seat: str | None) -> PloFacts:
    node = PloDecisionNode(actor=actor, history_before=(), actions=(), history_stem="")
    spot = PloSpot(node=node, hero_index=0, hero_label="x", hero_cards=CARDS, presence=1.0)
    vstats = (
        PloVillainStats(seat=villain_seat, action_label="Raise 100%", weighted_combo_count=1.0, pct_of_dealt_hands=1.0)
        if villain_seat
        else None
    )
    return PloFacts(spot=spot, hand_class=classify_plo_hand(CARDS), archetype="", villain_stats=vstats)


def test_ip_oop_button_acts_last():
    assert ip_oop_positions("BU", "LJ") == ("BU", "LJ")
    assert ip_oop_positions("LJ", "BU") == ("BU", "LJ")


def test_ip_oop_blind_vs_blind_sb_is_ip():
    assert ip_oop_positions("SB", "BB") == ("SB", "BB")
    assert ip_oop_positions("BB", "SB") == ("SB", "BB")


def test_relative_position_with_villain():
    assert hero_relative_position(_facts("BU", villain_seat="LJ")) == "In Position"
    assert hero_relative_position(_facts("BB", villain_seat="LJ")) == "Out of Position"
    # BvB: SB is in position even though it acts first preflop.
    assert hero_relative_position(_facts("SB", villain_seat="BB")) == "In Position"
    assert hero_relative_position(_facts("BB", villain_seat="SB")) == "Out of Position"


def test_relative_position_on_open():
    assert hero_relative_position(_facts("BU", villain_seat=None)) == "In Position"
    assert hero_relative_position(_facts("SB", villain_seat=None)) == "In Position"
    assert hero_relative_position(_facts("LJ", villain_seat=None)) == "Out of Position"
    assert hero_relative_position(_facts("CO", villain_seat=None)) == "Out of Position"
