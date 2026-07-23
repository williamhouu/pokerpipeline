"""Tests for the 🎛️ fully-balanced FULL-HAND mode (July 2026).

Pins the Hold'em adaptation of the PLO balanced-batch rules:
* the final-answer verb axis reads the DOMINANT ACTION (style-independent,
  the user's rule), with preflop fold-/raise-enders mapping to fold/raise;
* ending street stays quota-owned by the length profile (not an axis);
* the meta carries run_settings.fully_balanced + a balance_report whose
  difficulty axis reports only the SCORED pool (honesty rule);
* determinism: same solve + seed -> byte-identical CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.postflop.balanced_hands import (  # noqa: E402
    hand_balance_attrs,
    hand_difficulty_band,
    order_pool_for_balance,
)
from pipeline.postflop.fixtures import btn_vs_bb_full_hand_2cJs7s  # noqa: E402
from pipeline.postflop.full_hand_batch import generate_full_hand_batch  # noqa: E402
from pipeline.postflop.play_through import assemble_hands  # noqa: E402
from pipeline.postflop.batch import _collect_worthy  # noqa: E402


def _pool(solve, seed=3):
    worthy, _lq, _ps, _am = _collect_worthy(
        solve, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None,
    )
    return assemble_hands(
        solve, seeds=worthy, heroes=(), max_hands=None,
        include_preflop=True, include_villain=False, variety_seed=seed,
    )


def test_hand_attrs_verb_reads_dominant_action_and_terminal_enders():
    solve = btn_vs_bb_full_hand_2cJs7s()
    hands = _pool(solve)
    assert hands
    for hand in hands:
        attrs = hand_balance_attrs(hand, solve)
        deepest = hand.legs[-1]
        if getattr(deepest, "spot", None) is not None:
            dominant = deepest.spot.dominant_action.lower()
            if dominant.startswith("fold"):
                assert attrs["answer_verb"] == "fold"
            elif dominant.startswith(("check", "call")):
                assert attrs["answer_verb"] == "call/check"
            else:
                assert attrs["answer_verb"] == "raise"
        elif getattr(deepest, "terminal_raise", False):
            assert attrs["answer_verb"] == "raise"
        elif getattr(deepest, "terminal_fold", False):
            assert attrs["answer_verb"] == "fold"
        assert attrs["hero"] in solve.positions


def test_order_pool_is_deterministic_and_a_permutation():
    solve = btn_vs_bb_full_hand_2cJs7s()
    hands = _pool(solve)
    ordered = order_pool_for_balance(hands, solve)
    assert len(ordered) == len(hands)
    assert {id(h) for h in ordered} == {id(h) for h in hands}
    assert [h.hand_id for h in ordered] == [
        h.hand_id for h in order_pool_for_balance(hands, solve)
    ]


def test_hand_difficulty_band_edges_match_admin_presets():
    assert hand_difficulty_band(1299) == "Easy"
    assert hand_difficulty_band(1300) == "Medium"
    assert hand_difficulty_band(2100) == "Hard"


def test_fully_balanced_batch_writes_report_and_is_deterministic(tmp_path):
    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "fh.csv"
    res = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=4, dry_run=True,
        fully_balanced=True, variety_seed=3,
    )
    meta = json.loads(Path(res.meta_path).read_text())
    assert meta["run_settings"]["fully_balanced"] is True
    report = meta["balance_report"]
    assert report["selected"] == meta["counters"]["hands_written"]
    axis_keys = [a["axis"] for a in report["axes"]]
    # Difficulty leads; ending street is deliberately NOT an axis (the
    # length profile owns it via quotas).
    assert axis_keys[0] == "difficulty_band"
    assert "answer_verb" in axis_keys and "hero" in axis_keys
    assert "ending_street" not in axis_keys
    for ax in report["axes"]:
        assert (
            sum(v["achieved"] for v in ax["values"]) == report["selected"]
        )
    assert "balance_swaps_made" in meta["counters"]
    # Byte-determinism: same solve + same seed -> identical CSV.
    first = out.read_bytes()
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=4, dry_run=True,
        fully_balanced=True, variety_seed=3,
    )
    assert out.read_bytes() == first


def test_fully_balanced_off_leaves_meta_without_report(tmp_path):
    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "fh_plain.csv"
    res = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        variety_seed=3,
    )
    meta = json.loads(Path(res.meta_path).read_text())
    assert meta["run_settings"]["fully_balanced"] is False
    assert "balance_report" not in meta


# --- 🧼 strict-clean hands (July 22 2026, phase 2 of removing manual review) --


def test_leg_is_fully_clean_mirrors_review_rules():
    from pipeline.postflop.full_hand_batch import leg_is_fully_clean

    # Auto-fix lifecycle: clean and fixed+re-passed are clean; everything
    # else is not.
    assert leg_is_fully_clean({}, {"revise": {"status": "clean"}})
    assert leg_is_fully_clean(
        {}, {"revise": {"status": "fixed", "final_audit_issues": []}}
    )
    assert not leg_is_fully_clean(
        {}, {"revise": {"status": "fixed", "final_audit_issues": ["x"]}}
    )
    assert not leg_is_fully_clean({}, {"revise": {"status": "discarded"}})
    # Flag-only mode: the claim_check cell decides.
    assert leg_is_fully_clean({"claim_check": "[]"}, {})
    assert not leg_is_fully_clean({"claim_check": '[{"claim": "x"}]'}, {})
    # Soft flags and cross-check findings always disqualify.
    assert not leg_is_fully_clean({}, {"validator_warnings": ["w"]})
    assert not leg_is_fully_clean({}, {"cross_check_issues": ["w"]})
    # A leg that CANNOT be audited (entry fallback, blank cell) passes.
    assert leg_is_fully_clean({"claim_check": ""}, {})


def test_strict_clean_rebuilds_then_ships_flagged_when_reserve_is_dry(
    tmp_path, monkeypatch
):
    """Control-flow pin: a hand with an unclean leg is rebuilt exactly once;
    still unclean, it looks for a replacement -- and on this tiny fixture
    (empty reserve) it SHIPS FLAGGED rather than silently shrinking the
    batch. The counters narrate every step."""
    import json

    import pipeline.postflop.full_hand_batch as fhb

    solve = btn_vs_bb_full_hand_2cJs7s()
    # Patch the cleanliness rule: every leg of the FIRST hand_id seen is
    # forever unclean; everything else clean.
    seen: dict[str, int] = {}

    def _fake_clean(row, record):
        hand_id = str((row or {}).get("hand_id", ""))
        if not seen or hand_id == next(iter(seen)):
            seen[hand_id] = seen.get(hand_id, 0) + 1
            return False
        return True

    monkeypatch.setattr(fhb, "leg_is_fully_clean", _fake_clean)
    out = tmp_path / "strict.csv"
    res = fhb.generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        balanced_lengths=True, variety_seed=3, strict_clean_hands=True,
    )
    meta = json.loads(Path(res.meta_path).read_text())
    ctr = meta["counters"]
    assert meta["run_settings"]["strict_clean_hands"] is True
    assert ctr["hands_regenerated_for_flags"] == 1  # one rebuild attempt
    # This fixture has no reserve to pull from, so the still-flagged hand
    # ships (flags visible) instead of vanishing -- the batch stays full.
    assert ctr["hands_dropped_still_flagged"] == 0
    assert ctr["hands_shipped_flagged_budget"] == 1
    assert ctr["hands_written"] == 2  # noqa: PLR2004


def test_strict_clean_is_a_noop_on_clean_dry_runs(tmp_path):
    """With nothing flagged (dry runs carry no audit), strict mode changes
    NOTHING: byte-identical output to the same batch without it."""
    solve = btn_vs_bb_full_hand_2cJs7s()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    generate_full_hand_batch(
        solve=solve, output_path=a, total_hands=2, dry_run=True,
        variety_seed=3,
    )
    generate_full_hand_batch(
        solve=solve, output_path=b, total_hands=2, dry_run=True,
        variety_seed=3, strict_clean_hands=True,
    )
    assert a.read_bytes() == b.read_bytes()


def test_strict_clean_circuit_breaker_ships_flagged_after_budget(tmp_path, monkeypatch):
    """The July 22 live lesson: on a flag-prone solve the rebuild/replace
    loop must NOT churn forever. With every leg permanently 'unclean', the
    budgets (total_hands rebuilds, 3x total_hands total churn) spend out
    and the batch SHIPS the remaining hands flagged -- it always ends, and
    the counter says how many shipped that way."""
    import json

    import pipeline.postflop.full_hand_batch as fhb

    solve = btn_vs_bb_full_hand_2cJs7s()
    monkeypatch.setattr(fhb, "leg_is_fully_clean", lambda r, rec: False)
    out = tmp_path / "budget.csv"
    res = fhb.generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        balanced_lengths=True, variety_seed=3, strict_clean_hands=True,
    )
    meta = json.loads(Path(res.meta_path).read_text())
    ctr = meta["counters"]
    # The batch finished and shipped hands despite universal flags.
    assert ctr["hands_written"] >= 1
    assert ctr["hands_shipped_flagged_budget"] >= 1
    # Churn stayed within the budgets.
    assert ctr["hands_regenerated_for_flags"] <= 2
    assert (
        ctr["hands_regenerated_for_flags"] + ctr["hands_dropped_still_flagged"]
        <= 3 * 2
    )


# --- incremental commit + graceful stop (July 22 2026, from the cancelled run)


def test_incremental_commit_final_output_matches_old_single_write(tmp_path):
    """The per-hand flush must not change the FINAL bytes: complete=True on
    the last flush, and the CSV/meta equal a from-scratch rebuild."""
    import json

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "inc.csv"
    res = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        variety_seed=3,
    )
    meta = json.loads(Path(res.meta_path).read_text())
    assert meta["complete"] is True
    assert meta["counters"]["stopped_early"] is False
    first = out.read_bytes()
    generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        variety_seed=3,
    )
    assert out.read_bytes() == first


def test_graceful_stop_keeps_committed_hands(tmp_path):
    """stop_check firing after the first hand: the batch ends early, ships
    every committed hand as a COMPLETE batch, and says why it's short."""
    import json

    solve = btn_vs_bb_full_hand_2cJs7s()
    out = tmp_path / "stopped.csv"
    calls = {"n": 0}

    def _stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # allow the first hand, stop before the second

    res = generate_full_hand_batch(
        solve=solve, output_path=out, total_hands=2, dry_run=True,
        variety_seed=3, stop_check=_stop,
    )
    meta = json.loads(Path(res.meta_path).read_text())
    assert meta["complete"] is True  # a graceful stop is a CLEAN finish
    assert meta["counters"]["stopped_early"] is True
    assert 1 <= meta["counters"]["hands_written"] < 2  # noqa: PLR2004
    assert out.exists() and out.read_bytes()  # the committed hand is on disk
