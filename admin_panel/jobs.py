"""Background job registry for the admin panel.

Streamlit re-executes the whole script on every interaction -- sidebar
clicks, button presses, tab switches. Any Python work running in the
script thread when a rerun happens is abandoned. This module runs long
jobs (today: preflop batch generation) on a background thread so they
survive across reruns. The UI polls a small thread-safe state object
each rerun and renders progress + result.

Scope: single concurrent job per admin-panel process. The admin panel
is single-user, the batch is the only long-running operation, and
queuing/cancellation aren't needed yet. If multiple parallel jobs
become real later, swap the single slot for a dict keyed by id.

Threads, not subprocesses: keeps everything in one Python interpreter
(the result is a dataclass; no IPC). Trade-off: a streamlit server
restart kills the job. Today's pain (tab switch kills work) is fully
fixed by threading. If durability across restarts becomes a real ask,
we promote to subprocess + on-disk progress.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    """Lifecycle of a job. Strings so logs/UI can print them as-is."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def is_active(self) -> bool:
        return self.status in (JobStatus.PENDING, JobStatus.RUNNING)

    @property
    def is_done(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)

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
    total)`` and writes into ``Job.progress``. Pipeline batch entry
    points already accept ``progress_callback`` -- no wrapper needed.

    Args:
        fn: The callable to run in the background.
        label: Short human-readable description for UI. e.g.
            "Generating 30 preflop questions (Sonnet 4.6)".
        progress_callback_kwarg: Name of the kwarg ``fn`` accepts for
            its progress reporter. Default matches the pipeline batch.
        **kwargs: Forwarded to ``fn``.

    Returns:
        The newly-created Job. Reads of ``.progress`` / ``.status`` /
        ``.result`` after this point are thread-safe.

    Raises:
        RuntimeError: if another job is already active. Check
            :func:`has_active_job` first if you want a friendlier
            "job already running" UI rather than an exception.
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


__all__ = [
    "Job",
    "JobProgress",
    "JobStatus",
    "clear_current_job",
    "get_current_job",
    "has_active_job",
    "start_job",
]
