"""Tests for pipeline.plo.app_table_format (the app's table-state columns)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.app_table_format import build_plo_app_table_columns  # noqa: E402
from pipeline.plo.fact_extractor import PloFacts  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402


def R(seat: str) -> PloAction:
    return PloAction(seat, PloActionType.RAISE, 100)


def F(seat: str) -> PloAction:
    return PloAction(seat, PloActionType.FOLD)


def _cols(actor: str, history: tuple[PloAction, ...], cards: tuple[str, str, str, str], **kw):
    node = PloDecisionNode(actor=actor, history_before=history, actions=(), history_stem="")
    spot = PloSpot(node=node, hero_index=0, hero_label="x", hero_cards=cards, presence=1.0)
    facts = PloFacts(spot=spot, hand_class=classify_plo_hand(cards), archetype="")
    return build_plo_app_table_columns(facts, **kw)


AAKK = ("Ac", "Ad", "Kc", "Kd")


def test_open_node_full_stacks_and_blind_pot():
    cols = _cols("LJ", (), AAKK)
    assert cols["user_seat"] == "UTG-$100"  # display code for the pack's LJ seat
    assert cols["pot"] == "$1.5"
    assert cols["default_stack"] == "$100"
    assert cols["table_size"] == "6"
    assert "SB-$100-$0.5" in cols["seats"]
    assert "BB-$99-$1" in cols["seats"]


def test_facing_open_shows_opener_token():
    cols = _cols("HJ", (R("LJ"),), AAKK)
    assert cols["user_seat"] == "HJ-$100"  # hero hasn't acted
    assert "UTG-$97-$3.5-raise" in cols["seats"]
    assert cols["pot"] == "$5"


def test_facing_3bet_hero_open_token_and_fold_rules():
    # LJ opens, HJ/CO/BU/SB fold, BB 3-bets, LJ faces it.
    cols = _cols(
        "LJ",
        (R("LJ"), F("HJ"), F("CO"), F("BU"), F("SB"), R("BB")),
        AAKK,
    )
    assert cols["user_seat"] == "UTG-$97-$3.5-raise"  # hero's own open
    assert "BB-$89-$11-3-bet" in cols["seats"]
    assert "SB-$100-$0.5-FOLD" in cols["seats"]  # blind fold shown
    assert "HJ-" not in cols["seats"]  # silent non-blind fold omitted
    assert "CO-" not in cols["seats"]
    assert cols["pot"] == "$15"


def test_user_cards_renders_four_cards():
    cols = _cols("LJ", (), ("Ac", "Ad", "4h", "8h"))
    assert cols["user_cards"] == "A-clubs, A-diamonds, 4-hearts, 8-hearts"
    assert cols["cards_on_table"] == ""  # preflop


def test_bb_display_uses_bb_not_dollars():
    cols = _cols("HJ", (R("LJ"),), AAKK, display_in_bb=True)
    assert "$" not in cols["seats"]
    # bb display keeps fractional remaining (no whole-dollar rounding).
    assert "UTG-96.5BB-3.5BB-raise" in cols["seats"]
    assert cols["pot"] == "5BB"
    assert cols["default_stack"] == "100BB"
