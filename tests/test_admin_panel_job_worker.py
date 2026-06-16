"""Tests for the out-of-process job runner.

Two layers:
  * ``run_spec`` unit tests -- exercise the worker's spec->call->envelope
    logic in-process (fast, deterministic, no subprocess).
  * ``start_subprocess_job`` integration tests -- actually spawn a child
    interpreter and verify progress mirroring, result capture, failure
    capture, and cancellation through the public jobs API.

Both use :func:`admin_panel.job_worker._echo_job` as the target: a tiny
top-level function the child can import by (module, qualname), so the
tests don't depend on a test module being importable from a fresh
interpreter.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import pytest

from admin_panel import job_worker, jobs


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Clean the single-slot registry + subprocess state between tests."""
    jobs._CURRENT_JOB = None  # noqa: SLF001
    jobs._CURRENT_PROC = None  # noqa: SLF001
    jobs._CANCEL_REQUESTED = False  # noqa: SLF001
    yield
    # Make sure a stray child from a failed test never leaks.
    jobs.request_cancel_current_job()
    jobs._CURRENT_JOB = None  # noqa: SLF001
    jobs._CURRENT_PROC = None  # noqa: SLF001
    jobs._CANCEL_REQUESTED = False  # noqa: SLF001


def _wait_for_done(job: jobs.Job[object], timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.is_done:
            return
        time.sleep(0.02)
    raise AssertionError(f"job did not finish within {timeout}s")


# --- run_spec unit tests (in-process, no subprocess) ------------------------
def _write_spec(tmp_path: Path, **kwargs: object) -> tuple[Path, Path, Path]:
    spec_path = tmp_path / "spec.pkl"
    progress_path = tmp_path / "progress.json"
    result_path = tmp_path / "result.pkl"
    spec_path.write_bytes(
        pickle.dumps(
            {
                "fn_module": "admin_panel.job_worker",
                "fn_qualname": "_echo_job",
                "kwargs": kwargs,
                "progress_callback_kwarg": "progress_callback",
                "progress_path": str(progress_path),
                "result_path": str(result_path),
            }
        )
    )
    return spec_path, progress_path, result_path


def test_run_spec_success_writes_result_and_progress(tmp_path: Path) -> None:
    spec_path, progress_path, result_path = _write_spec(
        tmp_path, value="hello", steps=3
    )
    code = job_worker.run_spec(str(spec_path))
    assert code == 0
    envelope = pickle.loads(result_path.read_bytes())
    assert envelope == {"ok": True, "result": "hello"}
    # Last progress tick persisted.
    import json

    progress = json.loads(progress_path.read_text())
    assert progress == {"message": "step 3", "current": 3, "total": 3}


def test_run_spec_failure_writes_traceback_envelope(tmp_path: Path) -> None:
    spec_path, _progress_path, result_path = _write_spec(
        tmp_path, value="x", steps=1, fail=True
    )
    code = job_worker.run_spec(str(spec_path))
    assert code == 1
    envelope = pickle.loads(result_path.read_bytes())
    assert envelope["ok"] is False
    assert "echo job asked to fail" in envelope["error"]
    assert "RuntimeError" in envelope["error"]


# --- start_subprocess_job integration tests (real child process) ------------
def test_subprocess_job_runs_to_completion() -> None:
    job = jobs.start_subprocess_job(
        job_worker._echo_job, label="echo", value="done!", steps=3
    )
    assert job.cancellable is True
    _wait_for_done(job)

    assert job.status is jobs.JobStatus.COMPLETED
    assert job.result == "done!"
    assert job.error is None
    assert job.finished_at is not None
    # Progress was mirrored from the child's file into Job.progress.
    assert job.progress.total == 3
    assert job.progress.current == 3


def test_subprocess_job_failure_captures_traceback() -> None:
    job = jobs.start_subprocess_job(
        job_worker._echo_job, label="echo-fail", steps=1, fail=True
    )
    _wait_for_done(job)
    assert job.status is jobs.JobStatus.FAILED
    assert job.result is None
    assert job.error is not None
    assert "echo job asked to fail" in job.error


def test_subprocess_job_can_be_cancelled() -> None:
    # A long-running job so we have time to cancel it mid-flight.
    job = jobs.start_subprocess_job(
        job_worker._echo_job,
        label="echo-long",
        value="never",
        steps=10_000,
        sleep_s=0.05,
    )
    # Wait until it's actually running (child spawned + first tick).
    deadline = time.time() + 10.0
    while time.time() < deadline and job.status is not jobs.JobStatus.RUNNING:
        time.sleep(0.02)
    assert job.status is jobs.JobStatus.RUNNING

    assert jobs.request_cancel_current_job() is True
    _wait_for_done(job)
    assert job.status is jobs.JobStatus.CANCELLED
    assert job.result is None


def test_second_subprocess_job_while_active_raises() -> None:
    job = jobs.start_subprocess_job(
        job_worker._echo_job, label="first", steps=10_000, sleep_s=0.05
    )
    try:
        with pytest.raises(RuntimeError, match="Another job"):
            jobs.start_subprocess_job(
                job_worker._echo_job, label="second", steps=1
            )
    finally:
        jobs.request_cancel_current_job()
        _wait_for_done(job)


def test_request_cancel_no_job_returns_false() -> None:
    assert jobs.request_cancel_current_job() is False
