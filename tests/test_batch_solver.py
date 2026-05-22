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


# --- solve_one happy path (template-driven) ---------------------------------
def _stub_template(path: Path, board_token: str = "AsKdQh") -> None:
    """Write a minimal Pio-shaped template the batch_solver can replay."""
    path.write_text(
        "#Type#NoLimit\n"
        "#Board#As Kd Qh\n"
        "#Pot#55\n"
        "set_isomorphism 1 0\n"
        "set_eff_stack 975\n"
        "set_pot 0 0 55\n"
        "set_range OOP " + " ".join(["1.0"] * 1326) + "\n"
        "set_range IP " + " ".join(["1.0"] * 1326) + "\n"
        f"set_board {board_token}\n"
        "add_line 0 0 0 0 0 41 112 257 553 975\n"
        "build_tree\n",
        encoding="utf-8")


def _spec_with_template(template_path: Path):
    """Return a SOLVER_SPECS entry with pio_template_path pointing at our stub."""
    from dataclasses import replace
    return replace(SPEC, pio_template_path=str(template_path))


def test_solve_one_issues_template_lines_and_solve_commands():
    """Template-driven: every non-comment template line is issued via UPI,
    then set_accuracy / go / wait_for_solver / dump_tree."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template = tmp_path / "stub_template.txt"
        _stub_template(template)
        spec = _spec_with_template(template)
        log: list = []
        client = _make_mock_client(log)
        result = solve_one(client, spec, FLOP, solve_root=tmp_path,
                           log=lambda *a, **kw: None)
        assert result.status == "solved"
        verbs = [cmd.split()[0] for cmd, _ in log]
        # Every UPI verb that appeared in the stub template must be issued.
        for verb in ("set_isomorphism", "set_eff_stack", "set_pot",
                     "set_range", "set_board", "add_line", "build_tree",
                     "set_accuracy", "go", "wait_for_solver", "dump_tree"):
            assert verb in verbs, f"missing UPI verb {verb!r} in: {verbs}"
        # Metadata header lines (starting with '#') are NOT issued.
        assert not any(cmd.startswith("#") for cmd, _ in log)


def test_solve_one_substitutes_set_board_for_target_flop():
    """The template's `set_board AsKdQh` line must be rewritten to the
    target flop's concatenated stem when MINIMAL_DEBUG (2cJs7s) is solved."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template = tmp_path / "stub_template.txt"
        _stub_template(template, board_token="AsKdQh")     # template's default
        spec = _spec_with_template(template)
        log: list = []
        client = _make_mock_client(log)
        solve_one(client, spec, FLOP, solve_root=tmp_path,
                  log=lambda *a, **kw: None)
        boards_issued = [cmd for cmd, _ in log if cmd.startswith("set_board")]
        # Only the swapped board appears; the template's stub board doesn't.
        assert boards_issued == ["set_board 2cJs7s"], boards_issued


def test_solve_one_passes_accuracy_from_spec():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template = tmp_path / "stub_template.txt"
        _stub_template(template)
        spec = _spec_with_template(template)
        log: list = []
        client = _make_mock_client(log)
        solve_one(client, spec, FLOP, solve_root=tmp_path,
                  log=lambda *a, **kw: None)
        log_cmds = [cmd for cmd, _ in log]
        assert f"set_accuracy {spec.accuracy_target_chips}" in log_cmds


def test_solve_one_errors_clearly_when_template_missing():
    """A missing template path produces an actionable error, not a cryptic
    UPI failure halfway through the run."""
    from dataclasses import replace
    with tempfile.TemporaryDirectory() as tmp:
        spec = replace(SPEC, pio_template_path=str(Path(tmp) / "nope.txt"))
        log: list = []
        client = _make_mock_client(log)
        result = solve_one(client, spec, FLOP, solve_root=Path(tmp),
                           log=lambda *a, **kw: None)
        assert result.status == "failed"
        assert "template not found" in result.error_message.lower()
        assert log == []                       # no UPI commands issued


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
    """When a template UPI command errors, write a .failed marker and
    return cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        template = root / "stub_template.txt"
        _stub_template(template)
        spec = _spec_with_template(template)
        log: list = []
        # set_range errors -- simulates a corrupt template / Pio rejecting
        # the range argument format.
        client = _make_mock_client(log, fail_on={"set_range"})
        result = solve_one(client, spec, FLOP, solve_root=root,
                           log=lambda *a, **kw: None)
        assert result.status == "failed"
        assert "set_range" in result.error_message
        marker = failure_marker_path(spec, FLOP, solve_root=root, kind="failed")
        assert marker.is_file()
        assert "set_range" in marker.read_text(encoding="utf-8")


def test_solve_one_writes_timeout_marker_on_solver_timeout():
    """`wait_for_solver` raising 'timed out' -> .timeout marker."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        template = root / "stub_template.txt"
        _stub_template(template)
        spec = _spec_with_template(template)
        log: list = []
        client = _make_mock_client(log, timeout_on={"wait_for_solver"})
        result = solve_one(client, spec, FLOP, solve_root=root,
                           log=lambda *a, **kw: None)
        assert result.status == "timeout"
        marker = failure_marker_path(spec, FLOP, solve_root=root, kind="timeout")
        assert marker.is_file()


def test_solve_one_clears_stale_markers_before_retry():
    """A previous .failed marker shouldn't block a re-attempt -- solve_one
    deletes any sibling markers before configuring the tree."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        template = root / "stub_template.txt"
        _stub_template(template)
        spec = _spec_with_template(template)
        stale = failure_marker_path(spec, FLOP, solve_root=root, kind="failed")
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("old failure", encoding="utf-8")
        log: list = []
        client = _make_mock_client(log)
        result = solve_one(client, spec, FLOP, solve_root=root,
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
