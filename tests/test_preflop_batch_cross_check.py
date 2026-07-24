"""Tests for the deterministic post-batch cross-checks (July 2026).

Each check is exercised with a synthetic BROKEN row (must fire) and the
batch-integration test proves a clean dry-run batch runs the pass
automatically and reports zero problems.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

import pytest  # noqa: E402

from pipeline.preflop.batch import generate_preflop_batch  # noqa: E402
from pipeline.preflop.batch_cross_check import (  # noqa: E402
    cross_check_batch,
    cross_check_row,
    expected_relative_position,
)
from pipeline.preflop.pack import (  # noqa: E402
    PreflopPack,
    clear_registry,
    register_pack,
)
from pipeline.preflop_ranges import canonical_169_hand_classes  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _row(**overrides) -> dict:
    """A fully-consistent baseline row; each test breaks ONE thing."""
    base = {
        "No": "1",
        "Position Matchup": "SB_vs_BB",
        "Relative Position": "Out of Position",
        "skills": "Facing a 3-Bet, Blind vs. Blind Play, Out of Position Play",
        "Difficulty Rating": "1800",
        "action_frequencies": "Fold: 100%, Call: 0%, 4-bet: 0%",
        "Question": "You're in the Small Blind with K❤️T♣️.\n"
                    "You open to 3bb. The Big Blind 3-bets to 9bb.",
        "Answer Explanation": "The best play is to fold.",
    }
    base.update(overrides)
    return base


def _record(**overrides) -> dict:
    base = {
        "hand_class": "KTo",
        "solver_data": {
            "domination_vs_villain_range": {
                "dominated_by": ["AKo", "KQo", "AA"],
                "you_dominate": ["K9o", "QTo"],
            },
            "villain_stats": {
                "most_common_combos": [
                    {"hand_class": "AKo", "weight": 1.0},
                    {"hand_class": "K9o", "weight": 1.0},
                ],
            },
        },
    }
    base.update(overrides)
    return base


# --- the seat rule itself -----------------------------------------------------
def test_expected_relative_position_ring_rules() -> None:
    assert expected_relative_position("SB", "BB") == "Out of Position"
    assert expected_relative_position("BB", "SB") == "In Position"
    assert expected_relative_position("BTN", "CO") == "In Position"
    assert expected_relative_position("BB", "BTN") == "Out of Position"
    # Open spots: only the BTN is guaranteed position.
    assert expected_relative_position("BTN", None) == "In Position"
    assert expected_relative_position("SB", None) == "Out of Position"
    assert expected_relative_position("??", "BB") is None


# --- clean baseline -------------------------------------------------------------
def test_consistent_row_is_clean() -> None:
    assert cross_check_row(_row(), _record()) == []


# --- each check fires on a broken row --------------------------------------------
def test_catches_wrong_relative_position() -> None:
    issues = cross_check_row(_row(**{"Relative Position": "In Position"}),
                             _record())
    assert any("Relative Position" in i for i in issues)


def test_catches_wrong_position_skill() -> None:
    issues = cross_check_row(
        _row(skills="Blind vs. Blind Play, In Position Play"), _record()
    )
    assert any("'In Position Play'" in i for i in issues)


def test_catches_bvb_skill_hygiene() -> None:
    issues = cross_check_row(
        _row(skills="Blind Defense, Out of Position Play"), _record()
    )
    assert any("missing the 'Blind vs. Blind Play'" in i for i in issues)
    assert any("wrongly tagged 'Blind Defense'" in i for i in issues)


def test_catches_swapped_domination_lists() -> None:
    rec = _record()
    dom = rec["solver_data"]["domination_vs_villain_range"]
    dom["dominated_by"], dom["you_dominate"] = (
        dom["you_dominate"], dom["dominated_by"],
    )
    issues = cross_check_row(_row(), rec)
    assert any("classifies as you_dominate" in i for i in issues)
    assert any("classifies as dominates_you" in i for i in issues)


def test_catches_empty_dominated_by_with_visible_dominators() -> None:
    rec = _record()
    rec["solver_data"]["domination_vs_villain_range"]["dominated_by"] = []
    issues = cross_check_row(_row(), rec)
    assert any("dominated_by is EMPTY" in i and "AKo" in i for i in issues)


def test_catches_out_of_band_difficulty() -> None:
    issues = cross_check_row(
        _row(**{"Difficulty Rating": "900"}), _record(),
        min_difficulty=1500, max_difficulty=2750,
    )
    assert any("outside the batch's requested band" in i for i in issues)


def test_catches_frequency_sum_drift() -> None:
    issues = cross_check_row(
        _row(action_frequencies="Fold: 60%, Call: 30%"), _record()
    )
    assert any("sum to 90%" in i for i in issues)


def test_catches_rio_on_all_in() -> None:
    issues = cross_check_row(
        _row(
            skills="Blind vs. Blind Play, Reverse Implied Odds",
            Question="You're in the Small Blind.\nThe Big Blind moves all-in.",
        ),
        _record(),
    )
    assert any("Reverse Implied Odds" in i for i in issues)


def test_catches_prose_position_contradiction() -> None:
    issues = cross_check_row(
        _row(**{"Answer Explanation": "Fold, even though you're in position "
                                      "against this 3-bet."}),
        _record(),
    )
    assert any("prose says" in i for i in issues)


def test_batch_helper_maps_by_row_index() -> None:
    rows = [_row(), _row(**{"Relative Position": "In Position", "No": "2"})]
    recs = [_record(), _record()]
    findings = cross_check_batch(rows, recs)
    assert list(findings.keys()) == [1]


# --- integration: runs automatically at batch time --------------------------------
def test_batch_runs_cross_checks_automatically(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    utg = pack_root / "UTG"
    classes = canonical_169_hand_classes()
    raise_weights = {c: 0.0 for c in classes}
    raise_weights.update({"AA": 1.0, "A5s": 0.6, "77": 0.7})
    fold_weights = {c: 1.0 - raise_weights[c] for c in classes}
    utg.mkdir(parents=True)
    (utg / "UTG_60%.txt").write_text(
        ",".join(f"{c}:{raise_weights[c]}" for c in classes)
    )
    (utg / "UTG_Fold.txt").write_text(
        ",".join(f"{c}:{fold_weights[c]}" for c in classes)
    )
    pack = PreflopPack(
        pack_id="xcheck_pack", root_path=pack_root, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
        description="cross-check fixture",
    )
    register_pack(pack)
    out = tmp_path / "out.csv"
    result = generate_preflop_batch(
        pack=pack, output_path=out, total_questions=10,
        dry_run=True, random_seed=3,
    )
    assert result.questions_written > 0
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    # The pass ran (counter present) and a healthy pipeline is clean.
    assert meta["counters"]["cross_check_problems"] == 0
    assert all("cross_check_issues" not in q for q in meta["questions"])


# --- check 9: GTO secondary must be EV-ranked (the standing rule) --------------
def _gto_row(**overrides) -> dict:
    base = _row(
        **{
            "option 1": "Always Fold",
            "option 2": "Mostly Fold",
            "option 3": "Mostly Call",
            "option 4": "Always Call",
            "Correct Answer": "Always Call",
            "action_frequencies": "Fold: 0%, Call: 100%, 4-bet: 0%",
            "action_ev_bb": "Call: +2.87, Fold: +0.00, 4-bet: +1.95",
        }
    )
    base.update(overrides)
    return base


def test_catches_gto_secondary_not_ev_ranked() -> None:
    """The AQs Review catch: pure Call with 4-bet at +1.95bb must pair with
    4-bet, not Fold. A batch shipping the Fold pairing gets flagged."""
    issues = cross_check_row(_gto_row(), _record())
    assert any("second-best action by EV" in i for i in issues)


def test_gto_secondary_correctly_ev_ranked_is_clean() -> None:
    row = _gto_row(
        **{
            "option 1": "Always Call",
            "option 2": "Mostly Call",
            "option 3": "Mostly 4-bet",
            "option 4": "Always 4-bet",
        }
    )
    assert cross_check_row(row, _record()) == []


def test_gto_secondary_fold_ok_when_fold_is_second_best_ev() -> None:
    """Fold competes inside the EV ranking: when raising is -EV, the Fold
    pairing is correct and must not be flagged."""
    row = _gto_row(
        **{"action_ev_bb": "Call: +1.10, Fold: +0.00, 4-bet: -0.55"}
    )
    assert cross_check_row(row, _record()) == []


def test_gto_secondary_check_skips_mixed_spots() -> None:
    """Non-pure rows pick the secondary by FREQUENCY (a real mix), so the
    EV rule doesn't apply."""
    row = _gto_row(
        **{"action_frequencies": "Fold: 15%, Call: 85%, 4-bet: 0%"}
    )
    assert cross_check_row(row, _record()) == []


def test_gto_secondary_check_skips_ev_less_rows() -> None:
    row = _gto_row(**{"action_ev_bb": ""})
    assert cross_check_row(row, _record()) == []


def test_gto_secondary_ignores_artifact_allin_at_deep_stacks() -> None:
    """July 23 2026 (200bb pack-leg false flags): the artifact rule strips
    All-in from every option surface at stacks > 40bb, so the EV-secondary
    audit must not demand it as the wrong answer even when its pack EV beats
    the shipped secondary's. At realistic short stacks All-in competes."""
    deep = _gto_row(
        **{
            "Default Stack": "200BB",
            "option 1": "Always Fold",
            "option 2": "Mostly Fold",
            "option 3": "Mostly 3-bet",
            "option 4": "Always 3-bet",
            "Correct Answer": "Always 3-bet",
            "action_frequencies": "Fold: 0%, Call: 0%, 3-bet: 100%",
            "action_ev_bb": "3-bet: +3.72, All-in: +2.76, Fold: +0.00",
        }
    )
    assert cross_check_row(deep, _record()) == []
    short = dict(deep, **{"Default Stack": "20BB"})
    issues = cross_check_row(short, _record())
    assert any("All-in is the better alternative" in i for i in issues)
