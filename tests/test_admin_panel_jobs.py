"""Unit tests for admin_panel.jobs.

The background-job module is small but easy to get wrong (threading +
shared state), so the tests cover:
 * happy path -- a job finishes, result + progress are captured
 * progress callback flows through to Job.progress
 * a thrown exception transitions the job to FAILED with a traceback
 * a second start_job while one is active raises RuntimeError
 * clear_current_job refuses while active, succeeds when done
"""

from __future__ import annotations

import threading
import time

import pytest

from admin_panel import jobs


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Force a clean slot between tests.

    The registry is a module-level global, so a test that leaves a job
    behind would poison its neighbors. We blow it away here regardless
    of state -- the tests own the slot.
    """
    jobs._CURRENT_JOB = None  # noqa: SLF001
    yield
    jobs._CURRENT_JOB = None  # noqa: SLF001


def _wait_for_done(job: jobs.Job[object], timeout: float = 2.0) -> None:
    """Spin until the job finishes or the timeout trips."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.is_done:
            return
        time.sleep(0.01)
    raise AssertionError(f"job did not finish within {timeout}s")


def test_happy_path_runs_to_completion() -> None:
    def fn(*, progress_callback: jobs.Callable[[str, int, int], None]) -> str:
        progress_callback("starting", 0, 3)
        progress_callback("midway", 1, 3)
        progress_callback("almost", 2, 3)
        return "done!"

    job = jobs.start_job(fn, label="test happy")
    _wait_for_done(job)

    assert job.status is jobs.JobStatus.COMPLETED
    assert job.result == "done!"
    assert job.error is None
    assert job.finished_at is not None
    # Last callback wins.
    assert job.progress.message == "almost"
    assert job.progress.current == 2
    assert job.progress.total == 3


def test_progress_is_visible_mid_run() -> None:
    """The script thread can read progress while the worker is alive."""
    saw_progress = threading.Event()
    can_finish = threading.Event()

    def fn(*, progress_callback: jobs.Callable[[str, int, int], None]) -> int:
        progress_callback("step 1", 1, 5)
        saw_progress.set()
        # Block until the test thread has verified mid-run state.
        assert can_finish.wait(timeout=2.0)
        return 42

    job = jobs.start_job(fn, label="test mid-run progress")

    assert saw_progress.wait(timeout=2.0)
    # Mid-run: status is RUNNING and progress reflects step 1.
    assert job.status is jobs.JobStatus.RUNNING
    assert job.progress.message == "step 1"
    assert job.progress.current == 1
    assert job.progress.total == 5
    assert jobs.has_active_job()

    can_finish.set()
    _wait_for_done(job)
    assert job.result == 42


def test_exception_marks_failed_with_traceback() -> None:
    def fn(*, progress_callback: jobs.Callable[[str, int, int], None]) -> None:
        raise ValueError("kaboom")

    job = jobs.start_job(fn, label="test failure")
    _wait_for_done(job)

    assert job.status is jobs.JobStatus.FAILED
    assert job.result is None
    assert job.error is not None
    # Traceback text should mention the exception type and message so
    # users can copy-paste it back to us.
    assert "ValueError" in job.error
    assert "kaboom" in job.error


def test_second_start_while_active_raises() -> None:
    block = threading.Event()

    def fn(*, progress_callback: jobs.Callable[[str, int, int], None]) -> None:
        # Hold the slot until the test releases us.
        assert block.wait(timeout=2.0)

    job = jobs.start_job(fn, label="first")
    # Race-safe: even before the worker thread sets RUNNING, the slot
    # is held (start_job assigns _CURRENT_JOB under the lock first).
    with pytest.raises(RuntimeError, match="Another job"):
        jobs.start_job(fn, label="second")

    block.set()
    _wait_for_done(job)

    # Once done, the slot accepts a new job.
    def fn2(*, progress_callback: jobs.Callable[[str, int, int], None]) -> str:
        return "ok"

    job2 = jobs.start_job(fn2, label="after first")
    _wait_for_done(job2)
    assert job2.result == "ok"


def test_clear_current_job_refuses_while_active() -> None:
    block = threading.Event()

    def fn(*, progress_callback: jobs.Callable[[str, int, int], None]) -> None:
        assert block.wait(timeout=2.0)

    jobs.start_job(fn, label="blocking")
    with pytest.raises(RuntimeError, match="active job"):
        jobs.clear_current_job()

    block.set()
    job = jobs.get_current_job()
    assert job is not None
    _wait_for_done(job)

    # Now clearing is allowed.
    jobs.clear_current_job()
    assert jobs.get_current_job() is None


def test_custom_progress_callback_kwarg() -> None:
    """Callers can pass a different kwarg name than 'progress_callback'."""

    def fn(*, on_progress: jobs.Callable[[str, int, int], None]) -> str:
        on_progress("tick", 1, 1)
        return "ok"

    job = jobs.start_job(
        fn, label="custom kwarg", progress_callback_kwarg="on_progress"
    )
    _wait_for_done(job)
    assert job.status is jobs.JobStatus.COMPLETED
    assert job.progress.message == "tick"


def test_no_job_state() -> None:
    """Sanity: a fresh registry reports no job, no activity."""
    assert jobs.get_current_job() is None
    assert not jobs.has_active_job()
    # clear is a no-op when empty.
    jobs.clear_current_job()
    assert jobs.get_current_job() is None
