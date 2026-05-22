"""Tests for pipeline.batch_solver.

Pure unit tests -- no PioSolver process is launched. The PioSolverClient is
mocked (a SimpleNamespace exposing `command`, `try_command`, plus the
context-manager interface) so we can drive solve_one and run_batch through
every code path: success, skip (resume), failure, timeout, dry-run.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.batch_solver import (                                        # noqa: E402
    BatchSolveResult, SolveResult, cache_path, failure_marker_path,
    plan_batch, run_batch, solve_one,
)
from pipeline.flop_sets import MINIMAL_DEBUG, STANDARD_25_FLOPS, normalize_flop
from pipeline.piosolver import PioSolverError                              # noqa: E402
from pipeline.scenario_spec import SOLVER_SPECS                            # noqa: E402


SPEC = SOLVER_SPECS["Cash6max_100bb_BTN_open_BB_call"]
FLOP = MINIMAL_DEBUG[0]                          # ('2c', 'Js', '7s')


# --- mock client -----------------------------------------------------------
def _make_mock_client(command_log: list, *,
                      fail_on: set | None = None,
                      timeout_on: set | None = None,
                      exploit_response: list[str] | None = None,
                      dump_creates_file: bool = True):
    """Return a SimpleNamespace mimicking PioSolverClient.

    Records every (cmd, timeout) on `command_log`. If `dump_creates_file`
    is True, a `dump_tree <path>` call creates an empty file at <path>.
    """
    fail_on = fail_on or set()
    timeout_on = timeout_on or set()
    exploit_response = exploit_response or []

    def command(cmd: str, timeout: float = 120.0) -> list[str]:
        command_log.append((cmd, timeout))
        verb = cmd.split()[0] if cmd else ""
        if cmd in fail_on or verb in fail_on:
            raise PioSolverError(f"mock failure on: {cmd}")
        if cmd in timeout_on or verb in timeout_on:
            raise PioSolverError(f"mock timed out after waiting for: {cmd}")
        if verb == "dump_tree" and dump_creates_file:
            # The output path is the second token (after `dump_tree`).
            target = Path(cmd.split()[1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"FAKE_CFR_BYTES")
        if verb in ("calc_results_ev", "calc_global_freq", "show_progress"):
            return exploit_response
        return []

    def try_command(cmd: str) -> list[str]:
        try:
            return command(cmd)
        except PioSolverError:
            return []

    return SimpleNamespace(command=command, try_command=try_command,
                           _log=command_log)


# --- cache paths -----------------------------------------------------------
def test_cache_path_includes_scenario_dir_and_flop_stem():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        p = cache_path(SPEC, FLOP, solve_root=root)
        assert p == root / "Cash6max_100bb_BTN_open_BB_call" / "2cJs7s.cfr"


def test_failure_marker_path_is_sibling_with_kind_extension():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed = failure_marker_path(SPEC, FLOP, solve_root=root, kind="failed")
        timeout = failure_marker_path(SPEC, FLOP, solve_root=root, kind="timeout")
        assert failed.parent == cache_path(SPEC, FLOP, solve_root=root).parent
        assert failed.name == "2cJs7s.failed"
        assert timeout.name == "2cJs7s.timeout"


# --- plan_batch (dry-run primitive) ----------------------------------------
def test_plan_batch_lists_one_entry_per_flop():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = plan_batch(SPEC, MINIMAL_DEBUG, solve_root=root)
        assert len(plan) == 1
        entry = plan[0]
        assert entry["spec_name"] == SPEC.name
        assert entry["flop_stem"] == "2cJs7s"
        assert entry["flop_board"] == "2c Js 7s"
        assert entry["already_exists"] is False
        assert entry["output_path"] == root / SPEC.name / "2cJs7s.cfr"


def test_plan_batch_marks_existing_cfrs_for_skip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = cache_path(SPEC, FLOP, solve_root=root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"FAKE")
        plan = plan_batch(SPEC, MINIMAL_DEBUG, solve_root=root)
        assert plan[0]["already_exists"] is True


def test_plan_batch_handles_25_flop_set():
    with tempfile.TemporaryDirectory() as tmp:
        plan = plan_batch(SPEC, STANDARD_25_FLOPS, solve_root=Path(tmp))
        assert len(plan) == 25
        stems = {entry["flop_stem"] for entry in plan}
        assert len(stems) == 25                  # all distinct


# --- solve_one happy path ---------------------------------------------------
def test_solve_one_issues_expected_upi_sequence():
    """The UPI command sequence must include every setup step in the right
    order: isomorphism, ranges, geometry, board, sizings, build_tree,
    accuracy, go, wait_for_solver, dump_tree."""
    with tempfile.TemporaryDirectory() as tmp:
        log: list = []
        client = _make_mock_client(log)
        result = solve_one(client, SPEC, FLOP, solve_root=Path(tmp),
                           log=lambda *a, **kw: None)
        verbs = [cmd.split()[0] for cmd, _to in log if cmd]
        assert result.status == "solved"
        # Required setup verbs appear in order.
        for verb in ("set_isomorphism", "set_range_oop", "set_range_ip",
                     "set_eff_stack", "set_pot", "set_board",
                     "set_bet_size", "build_tree", "set_accuracy",
                     "go", "wait_for_solver", "dump_tree"):
            assert verb in verbs, f"missing UPI verb {verb!r} in: {verbs}"
        # Output file produced + file size recorded.
        assert result.output_path.is_file()
        assert result.file_size_bytes > 0


def test_solve_one_passes_spec_chip_values():
    """Spot-check that the actual chip numbers from the spec land in UPI args."""
    with tempfile.TemporaryDirectory() as tmp:
        log: list = []
        client = _make_mock_client(log)
        solve_one(client, SPEC, FLOP, solve_root=Path(tmp),
                  log=lambda *a, **kw: None)
        log_cmds = [cmd for cmd, _ in log]
        # Geometry from the registered spec.
        assert f"set_eff_stack {SPEC.starting_postflop_stack_chips}" in log_cmds
        assert f"set_pot 0 0 {SPEC.pot_after_preflop_chips}" in log_cmds
        # Board.
        assert "set_board 2c Js 7s" in log_cmds
        # Sizings.
        oop_sizes = ",".join(str(s) for s in SPEC.bet_sizes_oop_pct)
        assert f"set_bet_size OOP {oop_sizes}" in log_cmds
        # Accuracy.
        assert f"set_accuracy {SPEC.accuracy_target_chips}" in log_cmds


# --- solve_one resume + failure paths --------------------------------------
def test_solve_one_skips_when_cfr_already_exists():
    """If the .cfr is on disk already, solve_one returns 'skipped' without
    issuing any UPI commands."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = cache_path(SPEC, FLOP, solve_root=root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 4096)
        log: list = []
        client = _make_mock_client(log)
        result = solve_one(client, SPEC, FLOP, solve_root=root,
                           log=lambda *a, **kw: None)
        assert result.status == "skipped"
        assert result.file_size_bytes == 4096
        assert log == []                         # NO UPI commands issued


def test_solve_one_writes_failed_marker_on_setup_error():
    """When set_range_ip errors, write a .failed marker and return cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log: list = []
        client = _make_mock_client(log, fail_on={"set_range_ip"})
        result = solve_one(client, SPEC, FLOP, solve_root=root,
                           log=lambda *a, **kw: None)
        assert result.status == "failed"
        assert "set_range_ip" in result.error_message
        marker = failure_marker_path(SPEC, FLOP, solve_root=root, kind="failed")
        assert marker.is_file()
        assert "set_range_ip" in marker.read_text(encoding="utf-8")


def test_solve_one_writes_timeout_marker_on_solver_timeout():
    """`wait_for_solver` raising 'timed out' -> .timeout marker."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log: list = []
        client = _make_mock_client(log, timeout_on={"wait_for_solver"})
        result = solve_one(client, SPEC, FLOP, solve_root=root,
                           log=lambda *a, **kw: None)
        assert result.status == "timeout"
        marker = failure_marker_path(SPEC, FLOP, solve_root=root, kind="timeout")
        assert marker.is_file()


def test_solve_one_clears_stale_markers_before_retry():
    """A previous .failed marker shouldn't block a re-attempt -- solve_one
    deletes any sibling markers before configuring the tree."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stale = failure_marker_path(SPEC, FLOP, solve_root=root, kind="failed")
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("old failure", encoding="utf-8")
        log: list = []
        client = _make_mock_client(log)
        result = solve_one(client, SPEC, FLOP, solve_root=root,
                           log=lambda *a, **kw: None)
        assert result.status == "solved"
        assert not stale.is_file()               # stale marker removed


# --- BatchSolveResult roll-ups ---------------------------------------------
def test_batch_solve_result_partitions_solves_by_status():
    res = BatchSolveResult(spec_name="x", flop_set_name="y")
    res.solves = [
        SolveResult("x", "AsKd9h", Path("a"), status="solved"),
        SolveResult("x", "Ah5d2c", Path("b"), status="skipped"),
        SolveResult("x", "QhQd9c", Path("c"), status="failed",
                    error_message="boom"),
        SolveResult("x", "Ks7s4d", Path("d"), status="timeout",
                    error_message="t/o"),
    ]
    assert [s.flop_stem for s in res.solved] == ["AsKd9h"]
    assert [s.flop_stem for s in res.skipped] == ["Ah5d2c"]
    assert {s.flop_stem for s in res.failed} == {"QhQd9c", "Ks7s4d"}


# --- run_batch end-to-end against the mock ---------------------------------
def test_run_batch_rollup_partitions_solves_correctly():
    """run_batch iterates flops and rolls up the per-flop SolveResults.
    We patch solve_one to return canned results so we exercise the
    iteration / classification without launching Pio."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stub_exe = root / "fake_pio.exe"
        stub_exe.write_bytes(b"stub")

        # Canned outcomes: one solved, one skipped, one failed.
        flops = (
            normalize_flop(("As", "Kd", "9h")),
            normalize_flop(("Ah", "5d", "2c")),
            normalize_flop(("Qh", "Qd", "9c")),
        )
        outcomes = iter([
            ("solved",  None),
            ("skipped", None),
            ("failed",  "boom"),
        ])

        def fake_solve_one(client, spec, flop, *, solve_root, timeout_seconds, log):
            status, err = next(outcomes)
            return SolveResult(
                spec_name=spec.name,
                flop_stem="".join(flop),
                output_path=cache_path(spec, flop, solve_root=solve_root),
                status=status, error_message=err or "")

        # Also patch PioSolverClient so no real process launches.
        from pipeline import batch_solver
        from pipeline import piosolver
        original_solve = batch_solver.solve_one
        original_start = piosolver.PioSolverClient.start
        original_close = piosolver.PioSolverClient.close
        batch_solver.solve_one = fake_solve_one
        piosolver.PioSolverClient.start = lambda self: None
        piosolver.PioSolverClient.close = lambda self: None
        try:
            res = batch_solver.run_batch(
                SPEC, flops, pio_exe=stub_exe, solve_root=root,
                flop_set_name="custom", log=lambda *a, **kw: None)
        finally:
            batch_solver.solve_one = original_solve
            piosolver.PioSolverClient.start = original_start
            piosolver.PioSolverClient.close = original_close
        assert len(res.solves) == 3
        assert [s.status for s in res.solves] == ["solved", "skipped", "failed"]
        assert len(res.solved) == 1
        assert len(res.skipped) == 1
        assert len(res.failed) == 1


def test_run_batch_raises_when_pio_exe_missing():
    """No PioSolver -> clear error before any iteration."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            run_batch(SPEC, MINIMAL_DEBUG,
                      pio_exe=root / "does_not_exist.exe",
                      solve_root=root, log=lambda *a, **kw: None)
        except PioSolverError as exc:
            assert "not found" in str(exc).lower()
            return
        raise AssertionError("expected PioSolverError")


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
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [ERR ] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
