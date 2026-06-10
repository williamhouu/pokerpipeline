"""Tests for pipeline.plo.range_examples (leaning-hand contrast examples)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo import range_examples  # noqa: E402
from pipeline.plo.fact_extractor import PloFacts  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.range_examples import (  # noqa: E402
    format_examples,
    leaning_examples_for_spot,
    render_hand_class,
)
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

AKQJ_DS = ("Ah", "Kh", "Qd", "Jd")
KK44_DS = ("Kc", "Kd", "4c", "4d")
QQTK_SS = ("Qc", "Qd", "Th", "Kh")
JT98_R = ("Jc", "Td", "9h", "8s")


# --- rendering --------------------------------------------------------------
def test_render_hand_class_ranks_desc_plus_suit_word():
    assert render_hand_class(AKQJ_DS) == "AKQJ double-suited"
    assert render_hand_class(KK44_DS) == "KK44 double-suited"
    assert render_hand_class(QQTK_SS) == "KQQT single-suited"
    assert render_hand_class(JT98_R) == "JT98 rainbow"


def test_render_never_names_specific_suits():
    # The card-fabrication audit rejects rank+suit mentions outside hero's
    # hand -- rendered examples must never contain one.
    for cards in (AKQJ_DS, KK44_DS, QQTK_SS):
        name = render_hand_class(cards)
        assert not any(s in name for s in ("♠", "♥", "❤", "♦", "♣"))
        assert classify_plo_hand(cards)  # sanity: the input itself is valid


# --- band selection ---------------------------------------------------------
def test_format_examples_band_dedupe_cap_and_wording():
    rows = [
        (0.9, 0.99, AKQJ_DS),  # too pure -> "super obvious", excluded
        (0.9, 0.70, KK44_DS),  # in band, high presence -> kept, "mostly"
        (0.02, 0.70, QQTK_SS),  # presence floor -> excluded
        (0.5, 0.50, JT98_R),  # in band -> kept, "often"
        (0.4, 0.66, ("Kh", "Ks", "4h", "4s")),  # same class as KK44 -> deduped
        (0.3, 0.30, ("Ac", "2d", "7h", "9s")),  # leans the wrong way -> out
    ]
    out = format_examples(rows, "calls")
    assert out == ["KK44 double-suited (mostly calls)", "JT98 rainbow (often calls)"]


def test_format_examples_caps_at_three():
    rows = [
        (0.9, 0.8, AKQJ_DS),
        (0.8, 0.8, KK44_DS),
        (0.7, 0.8, QQTK_SS),
        (0.6, 0.8, JT98_R),
    ]
    assert len(format_examples(rows, "calls")) == 3


# --- spot-level wiring ------------------------------------------------------
def _facts(freqs: dict[str, float]) -> PloFacts:
    node = PloDecisionNode(actor="SB", history_before=(), actions=(), history_stem="")
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=QQTK_SS,
        action_frequencies=freqs,
        presence=1.0,
    )
    return PloFacts(
        spot=spot, hand_class=classify_plo_hand(QQTK_SS), archetype="fold_pot_odds"
    )


def test_leaning_examples_picks_runner_up_action(monkeypatch):
    seen: dict[str, str] = {}

    def _fake(node, raw_label, verb):
        seen["raw"] = raw_label
        seen["verb"] = verb
        return ("KK44 double-suited (mostly calls)",)

    monkeypatch.setattr(range_examples, "hands_leaning_to_option", _fake)
    out = leaning_examples_for_spot(
        _facts({"Fold": 0.95, "Call": 0.05, "Raise 100%": 0.0})
    )
    assert seen["raw"] == "Call"  # runner-up by hero's frequencies
    assert out == {
        "action": "Call",
        "hands": ["KK44 double-suited (mostly calls)"],
    }


def test_leaning_examples_none_when_no_options_or_no_hands(monkeypatch):
    assert leaning_examples_for_spot(_facts({"Fold": 1.0})) is None
    monkeypatch.setattr(
        range_examples, "hands_leaning_to_option", lambda *a: ()
    )
    assert leaning_examples_for_spot(_facts({"Fold": 0.9, "Call": 0.1})) is None
