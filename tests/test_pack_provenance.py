"""Browserless tests for the Review pack-provenance captions (July 2026).

The captions themselves are tiny st.caption calls; ALL the logic lives in
two pure functions in admin_panel.review (per the fix-durability rule:
logic OUT of the Streamlit seam, tested without a browser):

* ``preflop_leg_provenance`` -- which source built a full-hand PREFLOP leg
  (a matched range pack vs the solve's entry ranges), joined on
  ``(hand_id, street == "preflop")``.
* ``batch_pack_id`` -- the range pack a standalone preflop batch came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel.review import (  # noqa: E402
    batch_pack_id,
    preflop_leg_provenance,
)


def _meta(questions: list[dict]) -> dict:
    return {"mode": "full_hand", "questions": questions}


# --- preflop_leg_provenance --------------------------------------------------
def test_pack_leg_caption_names_the_pack() -> None:
    meta = _meta([
        {"hand_id": "h1", "street": "preflop",
         "preflop_leg_source": "pack", "pack_id": "preflop_8max_200bb_IMPROVED"},
        {"hand_id": "h1", "street": "flop", "node_id": "r:0:c"},
    ])
    caption = preflop_leg_provenance(meta, hand_id="h1")
    assert caption == "Preflop leg from range pack: preflop_8max_200bb_IMPROVED"


def test_entry_fallback_leg_caption() -> None:
    meta = _meta([
        {"hand_id": "h2", "street": "preflop", "hero_position": "BB"},
    ])
    caption = preflop_leg_provenance(meta, hand_id="h2")
    assert caption == (
        "Preflop leg from the solve's entry ranges (no matching range pack)"
    )


def test_provenance_joins_on_hand_id() -> None:
    meta = _meta([
        {"hand_id": "h1", "street": "preflop",
         "preflop_leg_source": "pack", "pack_id": "pack_a"},
        {"hand_id": "h2", "street": "preflop", "hero_position": "BB"},
    ])
    assert "pack_a" in (preflop_leg_provenance(meta, hand_id="h1") or "")
    assert "entry ranges" in (preflop_leg_provenance(meta, hand_id="h2") or "")


def test_provenance_none_when_undeterminable() -> None:
    # No meta at all (older batch / missing sidecar).
    assert preflop_leg_provenance(None, hand_id="h1") is None
    # Blank hand_id (a standalone row) never gets a leg caption.
    assert preflop_leg_provenance(_meta([]), hand_id="") is None
    assert preflop_leg_provenance(_meta([]), hand_id="   ") is None
    # No preflop record for that hand.
    meta = _meta([{"hand_id": "h1", "street": "flop"}])
    assert preflop_leg_provenance(meta, hand_id="h1") is None
    # Malformed questions payloads are graceful, never raising.
    assert preflop_leg_provenance({"questions": "nope"}, hand_id="h1") is None
    assert preflop_leg_provenance({"questions": [None, 3]}, hand_id="h1") is None


def test_pack_leg_without_pack_id_still_says_pack() -> None:
    meta = _meta([
        {"hand_id": "h1", "street": "preflop", "preflop_leg_source": "pack"},
    ])
    assert preflop_leg_provenance(meta, hand_id="h1") == (
        "Preflop leg from a range pack"
    )


# --- empty_batch_diagnosis ----------------------------------------------------
def test_empty_diagnosis_names_the_band_and_observed_max() -> None:
    from admin_panel.review import empty_batch_diagnosis

    meta = {
        "mode": "full_hand",
        "run_settings": {"min_hand_difficulty": 2100, "max_hand_difficulty": 3200},
        "counters": {
            "questions_written": 0, "hands_assembled": 14,
            "hands_difficulty_filtered": 14,
            "hand_difficulty_observed_max": 1750,
        },
    }
    diag = empty_batch_diagnosis(meta)
    assert diag is not None
    assert "2100-3200" in diag and "14 scanned" in diag and "1750" in diag


def test_empty_diagnosis_no_hands_assembled() -> None:
    from admin_panel.review import empty_batch_diagnosis

    meta = {
        "mode": "full_hand",
        "run_settings": {},
        "counters": {"questions_written": 0, "hands_assembled": 0,
                     "hands_difficulty_filtered": 0},
    }
    diag = empty_batch_diagnosis(meta)
    assert diag is not None and "No hands could be assembled" in diag


def test_empty_diagnosis_none_when_not_applicable() -> None:
    from admin_panel.review import empty_batch_diagnosis

    assert empty_batch_diagnosis(None) is None
    assert empty_batch_diagnosis({}) is None
    # Questions were written -> the batch isn't empty; no diagnosis.
    assert empty_batch_diagnosis({
        "mode": "full_hand", "run_settings": {},
        "counters": {"questions_written": 9, "hands_assembled": 3},
    }) is None
    # Not a full-hand batch -> out of scope (for now).
    assert empty_batch_diagnosis({
        "mode": "spots", "run_settings": {},
        "counters": {"questions_written": 0},
    }) is None


# --- batch_pack_id -----------------------------------------------------------
def test_batch_pack_id_reads_top_level_pack() -> None:
    assert batch_pack_id({"pack_id": "ryan_preflop_tree_6max_100bb"}) == (
        "ryan_preflop_tree_6max_100bb"
    )


def test_batch_pack_id_none_when_blank_or_missing() -> None:
    assert batch_pack_id(None) is None
    assert batch_pack_id({}) is None
    assert batch_pack_id({"pack_id": ""}) is None
    assert batch_pack_id({"pack_id": "   "}) is None
    assert batch_pack_id("not a dict") is None
