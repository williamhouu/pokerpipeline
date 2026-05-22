"""Tests for pipeline.scenario_spec (Layer 2 solver specs).

Pure unit tests -- no PioSolver, no Anthropic. Covers:

  * the Cash6max_100bb_BTN_open_BB_call registration matches the geometry of
    the existing hand-solved test .cfr (so a re-solve via Layer 2 produces a
    structurally equivalent file);
  * registry lookup + clear error on miss;
  * the SolverSpec invariants (positive stack/pot, OOP != IP, etc.);
  * file-vs-string detection on the range field.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.scenario_spec import (                                       # noqa: E402
    BB_CALL_VS_BTN_OPEN_PLACEHOLDER, BTN_OPEN_100BB_PLACEHOLDER,
    SOLVER_SPECS, SolverSpec, get_solver_spec,
)


# --- the registered spec matches the existing hand-solved geometry ----------
def test_cash6max_btn_vs_bb_registered():
    """The Cash 6-max 100bb BTN-vs-BB SRP spec is registered with the same
    pot/stack geometry as the existing hand-solved test_solves/btn_vs_bb_srp_2cJs7s.cfr.

    Hand-solved geometry (from the existing path_sampler reading of the .cfr):
      starting_stack ~= 8775 chips, big_blind ~= 87.75 chips, pot ~= 495 chips.
    Note Pio reports the POSTFLOP effective stack via show_effective_stack().
    """
    spec = SOLVER_SPECS["Cash6max_100bb_BTN_open_BB_call"]
    assert spec.format == "cash"
    assert spec.stack_bb == 100
    assert spec.oop_position == "BB" and spec.ip_position == "BTN"
    # Stack & pot match the existing .cfr geometry (give or take Pio's iso
    # adjustments). bb_in_chips=90 gives a clean integer pot of 495 (= 5.5bb).
    assert spec.bb_in_chips == 90
    assert spec.pot_after_preflop_chips == 495
    assert spec.starting_postflop_stack_chips == 8775
    # Brief: at least two bet sizes per actor + at least one raise size.
    assert len(spec.bet_sizes_oop_pct) >= 2
    assert len(spec.bet_sizes_ip_pct) >= 2
    assert len(spec.raise_sizes_pct) >= 1
    # Accuracy target: 2.5 chips ~= 0.5% of pot, the brief's quality bar.
    assert spec.accuracy_target_chips == 2.5
    # Ranges are inline Pio strings, not file paths.
    assert spec.range_is_file("OOP") is False
    assert spec.range_is_file("IP") is False
    assert "22+" in spec.ip_range            # BTN open includes all pairs
    assert "65s" in spec.oop_range           # BB call has suited connectors


def test_get_solver_spec_known_and_unknown():
    spec = get_solver_spec("Cash6max_100bb_BTN_open_BB_call")
    assert spec.name == "Cash6max_100bb_BTN_open_BB_call"
    try:
        get_solver_spec("does_not_exist")
    except KeyError as exc:
        msg = str(exc)
        assert "does_not_exist" in msg
        assert "Cash6max_100bb_BTN_open_BB_call" in msg
        assert "pipeline/scenario_spec.py" in msg
        return
    raise AssertionError("expected KeyError")


def test_cache_dir_name_is_the_scenario_name():
    spec = SOLVER_SPECS["Cash6max_100bb_BTN_open_BB_call"]
    assert spec.cache_dir_name == "Cash6max_100bb_BTN_open_BB_call"


# --- validation -------------------------------------------------------------
def _valid_kwargs():
    return dict(
        name="Test", format="cash", stack_bb=100,
        oop_position="BB", ip_position="BTN",
        oop_range="QQ+", ip_range="22+",
        pot_after_preflop_chips=495,
        starting_postflop_stack_chips=8775,
        bb_in_chips=90,
        bet_sizes_oop_pct=(33, 75), bet_sizes_ip_pct=(33, 75),
        raise_sizes_pct=(50,),
        accuracy_target_chips=2.5,
    )


def test_invalid_format_rejected():
    try:
        SolverSpec(**{**_valid_kwargs(), "format": "zynga"})
    except ValueError as exc:
        assert "format" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_oop_must_differ_from_ip():
    try:
        SolverSpec(**{**_valid_kwargs(), "ip_position": "BB"})
    except ValueError as exc:
        assert "differ" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_pot_must_be_positive():
    try:
        SolverSpec(**{**_valid_kwargs(), "pot_after_preflop_chips": 0})
    except ValueError as exc:
        assert "pot" in str(exc).lower()
        return
    raise AssertionError("expected ValueError")


def test_bet_sizes_must_be_present():
    try:
        SolverSpec(**{**_valid_kwargs(), "bet_sizes_oop_pct": ()})
    except ValueError as exc:
        assert "bet size" in str(exc).lower()
        return
    raise AssertionError("expected ValueError")


def test_bet_size_pct_in_sane_range():
    try:
        SolverSpec(**{**_valid_kwargs(), "bet_sizes_oop_pct": (-50, 75)})
    except ValueError as exc:
        assert "1-1000" in str(exc) or "% of pot" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_accuracy_must_be_positive():
    try:
        SolverSpec(**{**_valid_kwargs(), "accuracy_target_chips": 0})
    except ValueError as exc:
        assert "accuracy" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_geometry_consistency_check():
    """If pot + 2*stack is smaller than stack_bb*bb_in_chips, the spec is
    inconsistent (some chips would be unaccounted for)."""
    try:
        # 100bb at 90 chips/bb = 9000 chips total in the hand. If pot+2*stack
        # comes to only 4000, something's wrong.
        SolverSpec(**{**_valid_kwargs(),
                      "starting_postflop_stack_chips": 1000,
                      "pot_after_preflop_chips": 2000})
    except ValueError as exc:
        assert "geometry" in str(exc).lower()
        return
    raise AssertionError("expected ValueError")


# --- range file detection ---------------------------------------------------
def test_range_is_file_detection():
    """A .txt extension AND existing file -> True; inline string -> False."""
    # Inline string (the registered default).
    spec = SOLVER_SPECS["Cash6max_100bb_BTN_open_BB_call"]
    assert spec.range_is_file("OOP") is False
    assert spec.range_is_file("IP") is False
    # A real .txt path that exists.
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"22+,A2s+\n")
        tmp_path = tmp.name
    try:
        file_spec = SolverSpec(**{**_valid_kwargs(),
                                  "oop_range": tmp_path,
                                  "ip_range": "22+"})
        assert file_spec.range_is_file("OOP") is True
        assert file_spec.range_is_file("IP") is False        # inline string
    finally:
        Path(tmp_path).unlink()
    # A .txt path that doesn't exist -> False (defensive: we won't error
    # construct-time on a missing file; the batch solver surfaces it later
    # with context).
    bad = SolverSpec(**{**_valid_kwargs(),
                        "oop_range": "/path/that/does/not/exist.txt"})
    assert bad.range_is_file("OOP") is False


def test_placeholder_constants_exported():
    """The two placeholder range strings are public so future commits can
    swap them out and the test that locks the geometry doesn't have to
    chase string-by-string."""
    assert "22+" in BTN_OPEN_100BB_PLACEHOLDER
    assert "22-TT" in BB_CALL_VS_BTN_OPEN_PLACEHOLDER


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
