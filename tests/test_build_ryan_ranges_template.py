"""Tests for `scripts.build_ryan_ranges_template`.

Pure unit tests -- do not require PioSolver install or read Ryan's range
pack from disk. Coverage:

  * Pot-type auto-inference for srp / 3bp / 4bp keys, including the
    4bet-takes-precedence-over-3bet edge case.
  * The scenario registry contains the expected per-pot-type entries.
  * Each registered scenario's range paths land in the right
    position-folder (BB/, BTN/, etc.) and have a `.txt` extension.
  * Each registered scenario's auto-inferred pot type matches the
    pot type expected for its key (SRP/3BP/4BP).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_ryan_ranges_template as builder                                # noqa: E402


# --- pot-type auto-inference ------------------------------------------------
def test_srp_key_infers_srp():
    assert builder.infer_pot_type("BTN_open_BB_call") == "srp"
    assert builder.infer_pot_type("CO_open_BB_call") == "srp"
    assert builder.infer_pot_type("SB_open_BB_call") == "srp"
    assert builder.infer_pot_type("HJ_open_BB_call") == "srp"
    # Thin SRP scenario (#3).
    assert builder.infer_pot_type("BTN_open_SB_call") == "srp"


def test_3bet_key_infers_3bp():
    assert builder.infer_pot_type("BTN_open_BB_3bet_BTN_call") == "3bp"
    assert builder.infer_pot_type("CO_open_BTN_3bet_CO_call") == "3bp"
    assert builder.infer_pot_type("HJ_open_BB_3bet_HJ_call") == "3bp"
    assert builder.infer_pot_type("BTN_open_SB_3bet_BTN_call") == "3bp"
    assert builder.infer_pot_type("UTG_open_BB_3bet_UTG_call") == "3bp"


def test_4bet_key_infers_4bp_even_when_3bet_also_present():
    # 4bp keys ALSO contain "3bet" because the 4-bet sequence is
    # BTN open + BB 3-bet + BTN 4-bet + BB call. The infer function must
    # check 4bet first so the key resolves to 4bp, not 3bp.
    assert builder.infer_pot_type(
        "BTN_open_BB_3bet_BTN_4bet_BB_call") == "4bp"
    assert builder.infer_pot_type(
        "CO_open_BTN_3bet_CO_4bet_BTN_call") == "4bp"
    assert builder.infer_pot_type(
        "HJ_open_BB_3bet_HJ_4bet_BB_call") == "4bp"
    assert builder.infer_pot_type(
        "UTG_open_BB_3bet_UTG_4bet_BB_call") == "4bp"


# --- source template registry -----------------------------------------------
def test_source_templates_has_all_three_pot_types():
    assert set(builder.SOURCE_TEMPLATES) == {"srp", "3bp", "4bp"}


def test_pot_types_constant_matches_source_templates_keys():
    assert set(builder.POT_TYPES) == set(builder.SOURCE_TEMPLATES)


def test_3bp_and_4bp_source_templates_exist_in_repo():
    # The SRP source template is Pio's shipped 2bpot-full.txt and lives
    # under C:\PioSOLVER\TreeBuilding\ (only present on a Pio-installed
    # box). The 3bp/4bp templates are committed to the repo so the build
    # is hermetic for the new pot types.
    assert builder.SOURCE_TEMPLATES["3bp"].is_file()
    assert builder.SOURCE_TEMPLATES["4bp"].is_file()


# --- scenario registry ------------------------------------------------------
def test_registry_contains_all_14_scenarios():
    expected = {
        # SRP
        "BTN_open_BB_call", "CO_open_BB_call", "SB_open_BB_call",
        "HJ_open_BB_call", "BTN_open_SB_call",
        # 3bp
        "BTN_open_BB_3bet_BTN_call", "CO_open_BTN_3bet_CO_call",
        "HJ_open_BB_3bet_HJ_call", "BTN_open_SB_3bet_BTN_call",
        "UTG_open_BB_3bet_UTG_call",
        # 4bp
        "BTN_open_BB_3bet_BTN_4bet_BB_call",
        "CO_open_BTN_3bet_CO_4bet_BTN_call",
        "HJ_open_BB_3bet_HJ_4bet_BB_call",
        "UTG_open_BB_3bet_UTG_4bet_BB_call",
    }
    assert set(builder.SCENARIO_REGISTRY) == expected


def test_each_scenario_has_well_formed_range_paths():
    valid_folders = {"BB", "BTN", "CO", "HJ", "SB", "UTG"}
    for key, scenario in builder.SCENARIO_REGISTRY.items():
        for label, path in (("OOP", scenario.oop_range_path),
                            ("IP", scenario.ip_range_path)):
            folder = path.split("/", 1)[0]
            assert folder in valid_folders, (
                f"{key}: {label} path {path!r} starts with unknown folder "
                f"{folder!r}")
            assert path.endswith(".txt"), (
                f"{key}: {label} path {path!r} must end in .txt")


def test_each_scenario_oop_ip_positions_differ():
    for key, scenario in builder.SCENARIO_REGISTRY.items():
        assert scenario.oop_position != scenario.ip_position, (
            f"{key}: OOP/IP positions both {scenario.oop_position!r}")


def test_each_scenario_pot_type_matches_inference():
    # For each registered scenario, the auto-inferred pot type from the
    # key must match the pot type expected for its action shape.
    expected_pot_types = {
        # SRP
        "BTN_open_BB_call": "srp",
        "CO_open_BB_call": "srp",
        "SB_open_BB_call": "srp",
        "HJ_open_BB_call": "srp",
        "BTN_open_SB_call": "srp",
        # 3bp
        "BTN_open_BB_3bet_BTN_call": "3bp",
        "CO_open_BTN_3bet_CO_call": "3bp",
        "HJ_open_BB_3bet_HJ_call": "3bp",
        "BTN_open_SB_3bet_BTN_call": "3bp",
        "UTG_open_BB_3bet_UTG_call": "3bp",
        # 4bp
        "BTN_open_BB_3bet_BTN_4bet_BB_call": "4bp",
        "CO_open_BTN_3bet_CO_4bet_BTN_call": "4bp",
        "HJ_open_BB_3bet_HJ_4bet_BB_call": "4bp",
        "UTG_open_BB_3bet_UTG_4bet_BB_call": "4bp",
    }
    for key in builder.SCENARIO_REGISTRY:
        assert builder.infer_pot_type(key) == expected_pot_types[key], (
            f"{key}: inferred {builder.infer_pot_type(key)!r}, "
            f"expected {expected_pot_types[key]!r}")
