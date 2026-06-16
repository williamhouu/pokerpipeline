"""Background job registry for the admin panel.

Streamlit re-executes the whole script on every interaction -- sidebar
clicks, button presses, tab switches. Any Python work running in the
script thread when a rerun happens is abandoned. This module runs long
jobs (today: preflop batch generation) in the background so they survive
across reruns. The UI polls a small thread-safe state object each rerun
and renders progress + result.

Scope: single concurrent job per admin-panel process. The admin panel
is single-user, the batch is the only long-running operation, and
queuing isn't needed yet. If multiple parallel jobs become real later,
swap the single slot for a dict keyed by id.

Two runners share one registry slot:

* :func:`start_job` runs any callable on a background **thread** -- for
  light work where staying in-process is fine.
* :func:`start_subprocess_job` runs an importable function in a separate
  **process** -- used for batch generation, which is heavy enough that an
  in-process thread starves the Streamlit UI of the GIL (a running batch
  froze the panel). The child has its own interpreter + GIL, so the UI
  stays responsive; progress comes back via a polled JSON file and the
  return value via a pickle file. Because it crosses a process boundary,
  the target must be a top-level importable function and its kwargs +
  return value must be picklable. A subprocess also makes real
  cancellation possible (see :func:`request_cancel_current_job`).
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobStatus(StrEnum):
    """Lifecycle of a job. Strings so logs/UI can print them as-is."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Latest progress snapshot. Mutated under the registry lock."""

    message: str = ""
    current: int = 0
    total: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass
class Job[T]:
    """One background job -- id, status, progress, result, error."""

    id: str
    label: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    result: T | None = None
    # Full traceback text on failure. Surfaced verbatim in the UI so the
    # user can copy-paste it back to us without digging through logs.
    error: str | None = None
    # True for subprocess jobs, which can be terminated. Threaded jobs
    # have no kill path, so the UI hides the Cancel button for them.
    cancellable: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def is_active(self) -> bool:
        return self.status in (JobStatus.PENDING, JobStatus.RUNNING)

    @property
    def is_done(self) -> bool:
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)


# Module-level single-slot registry. Streamlit reruns don't blow this
# away because the Python interpreter stays alive across reruns -- only
# the script's locals are re-executed. The lock guards mutations from
# the background thread vs. reads from the script thread.
_LOCK = threading.Lock()
_CURRENT_JOB: Job[Any] | None = None

# Subprocess-job state (single slot, like _CURRENT_JOB): the running child
# process and a cancel flag, both guarded by _LOCK. Only one job runs at a
# time, so a single proc handle + flag suffice.
_CURRENT_PROC: subprocess.Popen[bytes] | None = None
_CANCEL_REQUESTED: bool = False

# How often the supervisor thread mirrors the child's progress file into
# Job.progress. Sub-second so the 1s UI fragment always has fresh data;
# the supervisor sleeps between polls, holding no GIL while it waits.
_POLL_SECONDS = 0.4

# Repo root: admin_panel/jobs.py -> admin_panel -> <root>. Used as the
# child's cwd so `python -m admin_panel.job_worker` resolves its imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_current_job() -> Job[Any] | None:
    """Return the latest job (active OR finished), or None if never started."""
    with _LOCK:
        return _CURRENT_JOB


def has_active_job() -> bool:
    """True iff a job is pending or running."""
    job = get_current_job()
    return job is not None and job.is_active


def clear_current_job() -> None:
    """Drop the slot. Only allowed when the current job is done (or absent).

    Raises:
        RuntimeError: if a job is still active. The caller should refuse
            with a friendly message rather than racing the worker.
    """
    global _CURRENT_JOB
    with _LOCK:
        if _CURRENT_JOB is not None and _CURRENT_JOB.is_active:
            raise RuntimeError(
                f"Cannot clear an active job (id={_CURRENT_JOB.id}, "
                f"status={_CURRENT_JOB.status.value})."
            )
        _CURRENT_JOB = None


def start_job[T](
    fn: Callable[..., T],
    *,
    label: str,
    progress_callback_kwarg: str = "progress_callback",
    **kwargs: Any,
) -> Job[T]:
    """Start ``fn(**kwargs)`` on a background thread, return the Job handle.

    The function is invoked with ``**kwargs`` plus an extra kwarg
    (default name ``progress_callback``) that takes ``(message, current,
    total)`` and writes into ``Job.progress``.

    Use this for LIGHT work. For heavy work (batch generation) use
    :func:`start_subprocess_job` so the job doesn't contend with the UI
    for the GIL.

    Args:
        fn: The callable to run in the background.
        label: Short human-readable description for UI.
        progress_callback_kwarg: Name of the kwarg ``fn`` accepts for
            its progress reporter. Default matches the pipeline batch.
        **kwargs: Forwarded to ``fn``.

    Returns:
        The newly-created Job. Reads of ``.progress`` / ``.status`` /
        ``.result`` after this point are thread-safe.

    Raises:
        RuntimeError: if another job is already active.
    """
    global _CURRENT_JOB
    with _LOCK:
        if _CURRENT_JOB is not None and _CURRENT_JOB.is_active:
            raise RuntimeError(
                f"Another job is already running (id={_CURRENT_JOB.id}, "
                f"status={_CURRENT_JOB.status.value}). "
                "Wait for it to finish (or clear it) before starting a new one."
            )
        job: Job[T] = Job(id=uuid.uuid4().hex[:12], label=label)
        _CURRENT_JOB = job

    def _progress(msg: str, current: int, total: int) -> None:
        # The pipeline batch already calls back once per spot. Updating
        # under the lock keeps the dataclass swap atomic from the reader
        # side -- a read either sees the old snapshot or the new one.
        with _LOCK:
            job.progress = JobProgress(
                message=msg, current=current, total=total
            )

    def _runner() -> None:
        try:
            with _LOCK:
                job.status = JobStatus.RUNNING
            result = fn(**{progress_callback_kwarg: _progress, **kwargs})
            with _LOCK:
                job.result = result
                job.status = JobStatus.COMPLETED
                job.finished_at = time.time()
        except Exception:  # noqa: BLE001
            with _LOCK:
                job.error = traceback.format_exc()
                job.status = JobStatus.FAILED
                job.finished_at = time.time()

    # daemon=True so Ctrl+C on the streamlit server actually exits. If
    # the user closes the browser tab, the job keeps running -- the
    # whole point of this module.
    thread = threading.Thread(target=_runner, daemon=True, name=f"job-{job.id}")
    thread.start()
    return job


def start_subprocess_job[T](
    fn: Callable[..., T],
    *,
    label: str,
    progress_callback_kwarg: str = "progress_callback",
    **kwargs: Any,
) -> Job[T]:
    """Run ``fn(**kwargs)`` in a SEPARATE PROCESS; return the Job handle.

    Use this (not :func:`start_job`) for heavy work like batch generation:
    the child runs in its own interpreter, so it never contends with the
    Streamlit UI for the GIL. A lightweight supervisor thread in this
    process watches the child, mirrors its progress file into
    ``Job.progress``, and captures its pickled return value (or
    traceback) when it exits -- so the Job/JobProgress/JobStatus contract
    the UI reads is identical to the threaded runner.

    Constraints from the process boundary:
      * ``fn`` must be a TOP-LEVEL importable function (resolved in the
        child by ``fn.__module__`` + ``fn.__qualname__``). Closures and
        lambdas can't cross the boundary -- use :func:`start_job` for those.
      * ``**kwargs`` must be picklable, and ``fn``'s return value must be
        picklable (it comes back via a pickle file).
      * ``fn`` should persist its real outputs itself (the batch already
        writes the CSV + meta.json); the return value is for the UI's
        summary, not the payload.

    Raises:
        RuntimeError: if another job is already active.
    """
    global _CURRENT_JOB, _CURRENT_PROC, _CANCEL_REQUESTED
    with _LOCK:
        if _CURRENT_JOB is not None and _CURRENT_JOB.is_active:
            raise RuntimeError(
                f"Another job is already running (id={_CURRENT_JOB.id}, "
                f"status={_CURRENT_JOB.status.value}). "
                "Wait for it to finish (or clear it) before starting a new one."
            )
        job: Job[T] = Job(
            id=uuid.uuid4().hex[:12], label=label, cancellable=True
        )
        _CURRENT_JOB = job
        _CURRENT_PROC = None
        _CANCEL_REQUESTED = False

    work_dir = Path(tempfile.mkdtemp(prefix=f"pp_job_{job.id}_"))
    spec_path = work_dir / "spec.pkl"
    progress_path = work_dir / "progress.json"
    result_path = work_dir / "result.pkl"
    spec_path.write_bytes(
        pickle.dumps(
            {
                "fn_module": fn.__module__,
                "fn_qualname": fn.__qualname__,
                "kwargs": kwargs,
                "progress_callback_kwarg": progress_callback_kwarg,
                "progress_path": str(progress_path),
                "result_path": str(result_path),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )

    def _supervise() -> None:
        global _CURRENT_PROC
        try:
            proc: subprocess.Popen[bytes] = subprocess.Popen(
                [sys.executable, "-m", "admin_panel.job_worker", str(spec_path)],
                cwd=str(_PROJECT_ROOT),
                env=os.environ.copy(),
            )
            with _LOCK:
                _CURRENT_PROC = proc
                job.status = JobStatus.RUNNING
                cancel_now = _CANCEL_REQUESTED
            if cancel_now:
                # Cancel arrived before the process existed -- honor it now.
                proc.terminate()
            while proc.poll() is None:
                _mirror_progress(job, progress_path)
                time.sleep(_POLL_SECONDS)
            _mirror_progress(job, progress_path)  # final flush
            _finish_subprocess_job(job, proc.returncode, result_path)
        except Exception:  # noqa: BLE001
            with _LOCK:
                job.error = traceback.format_exc()
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
        finally:
            with _LOCK:
                _CURRENT_PROC = None
            shutil.rmtree(work_dir, ignore_errors=True)

    thread = threading.Thread(
        target=_supervise, daemon=True, name=f"job-{job.id}"
    )
    thread.start()
    return job


def _mirror_progress(job: Job[Any], progress_path: Path) -> None:
    """Copy the child's latest progress file into Job.progress.

    Best-effort: a missing file (no progress yet) or a torn read just
    leaves the previous snapshot in place until the next poll.
    """
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return
    with _LOCK:
        job.progress = JobProgress(
            message=str(data.get("message", "")),
            current=int(data.get("current", 0)),
            total=int(data.get("total", 0)),
        )


def _finish_subprocess_job(
    job: Job[Any], returncode: int | None, result_path: Path
) -> None:
    """Set the job's terminal state from the child's result file / exit code."""
    with _LOCK:
        cancelled = _CANCEL_REQUESTED
    if cancelled:
        with _LOCK:
            job.status = JobStatus.CANCELLED
            job.error = "Cancelled by user."
            job.finished_at = time.time()
        return
    try:
        envelope = pickle.loads(result_path.read_bytes())
    except (FileNotFoundError, OSError, pickle.UnpicklingError, EOFError) as exc:
        with _LOCK:
            job.status = JobStatus.FAILED
            job.error = (
                f"The generation process exited (code {returncode}) without "
                f"a readable result: {exc}. It may have been killed (out of "
                "memory, or terminated externally)."
            )
            job.finished_at = time.time()
        return
    with _LOCK:
        if envelope.get("ok"):
            job.result = envelope.get("result")
            job.status = JobStatus.COMPLETED
        else:
            job.status = JobStatus.FAILED
            job.error = envelope.get("error", "Unknown worker error.")
        job.finished_at = time.time()


def request_cancel_current_job() -> bool:
    """Ask the active subprocess job to stop. Returns True iff a cancel was
    accepted.

    Terminates the child process; the supervisor thread observes the exit
    and marks the job CANCELLED. No-op (returns False) when there is no
    active job, or the active job is a threaded :func:`start_job` (which
    has no kill path).
    """
    global _CANCEL_REQUESTED
    with _LOCK:
        job = _CURRENT_JOB
        if job is None or not job.is_active or not job.cancellable:
            return False
        _CANCEL_REQUESTED = True
        proc = _CURRENT_PROC
    if proc is not None:
        proc.terminate()  # outside the lock; supervisor sees poll() != None
    return True


__all__ = [
    "Job",
    "JobProgress",
    "JobStatus",
    "clear_current_job",
    "get_current_job",
    "has_active_job",
    "request_cancel_current_job",
    "start_job",
    "start_subprocess_job",
]
