"""Background job registry for the admin panel.

Streamlit re-executes the whole script on every interaction -- sidebar
clicks, button presses, tab switches. Any Python work running in the
script thread when a rerun happens is abandoned. This module runs long
jobs (today: preflop batch generation) in the background so they survive
across reruns. The UI polls a small thread-safe state object each rerun
and renders progress + result.

Scope: ONE job runs at a time per admin-panel process (the admin panel is
single-user and batches contend for the same API budget), plus a FIFO
queue of pending jobs (July 2026): :func:`enqueue_subprocess_job` starts
the job immediately when the slot is idle, otherwise appends it to the
queue; when the running job reaches a terminal state the next queued job
starts automatically. Finished jobs are appended to a bounded history
(:func:`job_history`) so the UI can show every batch's outcome even after
the slot has moved on to the next one.

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
    # True when the job's target accepts a graceful-stop check (July 2026):
    # the UI can offer "finish the current unit of work, keep what's done"
    # in addition to the hard Cancel.
    graceful_stoppable: bool = False
    # Set by request_graceful_stop_current_job so the UI can show
    # "stopping..." while the job wraps up its current unit of work.
    stop_requested: bool = False
    # Caller-supplied breadcrumbs (e.g. {"kind": "plo_generate",
    # "output_name": ...}). Lets each admin page recognise ITS jobs in the
    # shared slot/queue/history and carry click-time context (layer-7 mode,
    # requested count) to the completion renderer. Never read by this module.
    meta: dict[str, Any] = field(default_factory=dict)
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


@dataclass
class QueuedRequest:
    """One not-yet-started job waiting in the FIFO queue."""

    id: str
    label: str
    fn: Callable[..., Any]
    kwargs: dict[str, Any]
    progress_callback_kwarg: str = "progress_callback"
    meta: dict[str, Any] = field(default_factory=dict)
    stop_check_kwarg: str | None = None
    queued_at: float = field(default_factory=time.time)


# FIFO of pending jobs + bounded history of finished ones (July 2026).
# Both live at module level for the same reason as _CURRENT_JOB, guarded
# by the same lock. History exists because the queue auto-advances: a
# finished job is evicted from the slot the moment the next one starts,
# so the UI needs somewhere durable (per-process) to read its outcome.
_PENDING: list[QueuedRequest] = []
_HISTORY: list[Job[Any]] = []
_HISTORY_MAX = 30

# Subprocess-job state (single slot, like _CURRENT_JOB): the running child
# process and a cancel flag, both guarded by _LOCK. Only one job runs at a
# time, so a single proc handle + flag suffice.
_CURRENT_PROC: subprocess.Popen[bytes] | None = None
_CANCEL_REQUESTED: bool = False
# Sentinel file the CURRENT job's child polls for a graceful stop (set only
# for jobs started with stop_check_kwarg). Touching it asks the child to
# finish its current unit of work and return normally.
_CURRENT_STOP_PATH: Path | None = None

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


def pending_jobs() -> list[QueuedRequest]:
    """Snapshot of the FIFO queue (next-to-run first)."""
    with _LOCK:
        return list(_PENDING)


def job_history() -> list[Job[Any]]:
    """Snapshot of finished jobs, oldest first (bounded at _HISTORY_MAX)."""
    with _LOCK:
        return list(_HISTORY)


def remove_queued(request_id: str) -> bool:
    """Drop one not-yet-started request from the queue. True iff removed."""
    with _LOCK:
        for i, req in enumerate(_PENDING):
            if req.id == request_id:
                del _PENDING[i]
                return True
    return False


def _on_job_finished(job: Job[Any]) -> None:
    """Record a terminal job in history, then start the next queued job.

    Called by BOTH runners exactly once, after the job's terminal state is
    set and (for subprocess jobs) after the supervisor has released its
    process handle -- starting the successor earlier would let the old
    supervisor's cleanup clobber the new job's _CURRENT_PROC.
    """
    nxt: QueuedRequest | None = None
    with _LOCK:
        _HISTORY.append(job)
        del _HISTORY[:-_HISTORY_MAX]
        if _PENDING:
            nxt = _PENDING.pop(0)
    if nxt is None:
        return
    try:
        start_subprocess_job(
            nxt.fn,
            label=nxt.label,
            progress_callback_kwarg=nxt.progress_callback_kwarg,
            meta=nxt.meta,
            stop_check_kwarg=nxt.stop_check_kwarg,
            **nxt.kwargs,
        )
    except RuntimeError:
        # Slot got taken in the gap (e.g. the user started a job manually).
        # Put the request back at the front; the taker's completion will
        # advance the queue again.
        with _LOCK:
            _PENDING.insert(0, nxt)


def enqueue_subprocess_job[T](
    fn: Callable[..., T],
    *,
    label: str,
    progress_callback_kwarg: str = "progress_callback",
    meta: dict[str, Any] | None = None,
    stop_check_kwarg: str | None = None,
    **kwargs: Any,
) -> tuple[Job[T] | None, QueuedRequest | None, int]:
    """Start ``fn`` as a subprocess job now, or queue it behind the active one.

    Returns ``(job, None, 0)`` when the job started immediately, or
    ``(None, request, position)`` when it was queued (``position`` is
    1-based: 1 = next to run). Same picklability constraints as
    :func:`start_subprocess_job`.
    """
    try:
        job = start_subprocess_job(
            fn,
            label=label,
            progress_callback_kwarg=progress_callback_kwarg,
            meta=meta,
            stop_check_kwarg=stop_check_kwarg,
            **kwargs,
        )
    except RuntimeError:
        req = QueuedRequest(
            id=uuid.uuid4().hex[:12],
            label=label,
            fn=fn,
            kwargs=kwargs,
            progress_callback_kwarg=progress_callback_kwarg,
            meta=dict(meta or {}),
            stop_check_kwarg=stop_check_kwarg,
        )
        with _LOCK:
            _PENDING.append(req)
            position = len(_PENDING)
        return None, req, position
    return job, None, 0


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
    meta: dict[str, Any] | None = None,
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
        job: Job[T] = Job(
            id=uuid.uuid4().hex[:12], label=label, meta=dict(meta or {})
        )
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
        _on_job_finished(job)

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
    meta: dict[str, Any] | None = None,
    stop_check_kwarg: str | None = None,
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
            id=uuid.uuid4().hex[:12],
            label=label,
            cancellable=True,
            graceful_stoppable=stop_check_kwarg is not None,
            meta=dict(meta or {}),
        )
        _CURRENT_JOB = job
        _CURRENT_PROC = None
        _CANCEL_REQUESTED = False

    work_dir = Path(tempfile.mkdtemp(prefix=f"pp_job_{job.id}_"))
    spec_path = work_dir / "spec.pkl"
    progress_path = work_dir / "progress.json"
    result_path = work_dir / "result.pkl"
    stop_path = work_dir / "stop.requested"
    # Durable descriptor (July 2026): lets a RESTARTED panel process
    # rediscover this job from disk (see adopt_disk_jobs). Written before
    # the child spawns; the supervisor fills in the pid afterwards.
    _write_descriptor(
        work_dir,
        {
            "id": job.id,
            "label": label,
            "meta": job.meta,
            "started_at": job.started_at,
            "stop_check_kwarg": stop_check_kwarg,
            "pid": None,
        },
    )
    spec_path.write_bytes(
        pickle.dumps(
            {
                "fn_module": fn.__module__,
                "fn_qualname": fn.__qualname__,
                "kwargs": kwargs,
                "progress_callback_kwarg": progress_callback_kwarg,
                "progress_path": str(progress_path),
                "result_path": str(result_path),
                # Graceful stop (July 2026): when set, the worker injects a
                # callable under this kwarg that returns True once the
                # sentinel file exists (touched by request_graceful_stop).
                "stop_check_kwarg": stop_check_kwarg,
                "stop_path": str(stop_path),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    if stop_check_kwarg is not None:
        global _CURRENT_STOP_PATH
        with _LOCK:
            _CURRENT_STOP_PATH = stop_path

    def _supervise() -> None:
        global _CURRENT_PROC
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "admin_panel.job_worker", str(spec_path)],
                cwd=str(_PROJECT_ROOT),
                env=os.environ.copy(),
            )
            _update_descriptor(work_dir, pid=proc.pid)
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
            global _CURRENT_STOP_PATH
            with _LOCK:
                # Only clear OUR handles: _on_job_finished may auto-start the
                # next queued job, whose supervisor sets a NEW _CURRENT_PROC /
                # stop path; an unconditional None here would break its
                # cancel + graceful-stop paths.
                if _CURRENT_PROC is proc:
                    _CURRENT_PROC = None
                if _CURRENT_STOP_PATH == stop_path:
                    _CURRENT_STOP_PATH = None
            shutil.rmtree(work_dir, ignore_errors=True)
        _on_job_finished(job)

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


def request_graceful_stop_current_job() -> bool:
    """Ask the active job to finish its current unit of work, then stop.

    Touches the job's stop-sentinel file; the child's injected stop_check
    sees it between units (e.g. between questions) and returns normally
    with everything committed so far -- the job completes (not cancels).
    Returns True iff a stop was accepted. No-op (False) when there is no
    active job or the job wasn't started with ``stop_check_kwarg``.
    """
    with _LOCK:
        job = _CURRENT_JOB
        stop_path = _CURRENT_STOP_PATH
        if (
            job is None
            or not job.is_active
            or not job.graceful_stoppable
            or stop_path is None
        ):
            return False
        job.stop_requested = True
    try:
        stop_path.touch()
    except OSError:
        return False
    return True


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


# --- disk re-attach (July 2026) ----------------------------------------------
# INVARIANT: a panel-process restart must NEVER hide a running or finished
# batch. Job history/queue live in process memory, but the subprocess child
# survives a restart -- before this section, a restarted panel simply forgot
# its jobs ("started but never finished"), their results went unread and
# their token spend never reached the usage ledger. Every subprocess job now
# leaves a durable ``job.json`` descriptor in its ``pp_job_*`` work dir, and
# :func:`adopt_disk_jobs` rediscovers those dirs:
#
#   * still running  -> ADOPTED into a side registry (never the single
#     _CURRENT_JOB slot: two orphans can be running at once, and the slot
#     must stay free for new work), with a watcher thread that mirrors
#     progress and harvests the result exactly like a supervisor.
#   * finished       -> the result envelope is harvested straight into
#     _HISTORY (a work dir still on disk == its parent died before the
#     harvest, so this is never a double-read; the normal path removes the
#     dir the moment the supervisor finishes) and the dir is cleaned up.
#     History is what the ledger sweeps read, so recovered spend gets logged.
#   * dead (no process, no result) -> a FAILED history entry that says so.
#
# Dirs made by the pre-descriptor code (no job.json) are handled by
# reconstructing label/meta from spec.pkl; their pid is recovered via pgrep.

_ADOPTED: dict[str, tuple[Job[Any], "DiskJobInfo"]] = {}
# Serializes whole adoption passes: two Streamlit sessions rendering at once
# must not both adopt the same dir (double watcher, double history entry).
_ADOPT_LOCK = threading.Lock()


@dataclass
class DiskJobInfo:
    """What we know about one ``pp_job_*`` work dir found on disk."""

    work_dir: Path
    job_id: str
    label: str
    meta: dict[str, Any]
    started_at: float
    pid: int | None
    stop_check_kwarg: str | None
    legacy: bool  # True when reconstructed from spec.pkl (no job.json)

    @property
    def spec_path(self) -> Path:
        return self.work_dir / "spec.pkl"

    @property
    def progress_path(self) -> Path:
        return self.work_dir / "progress.json"

    @property
    def result_path(self) -> Path:
        return self.work_dir / "result.pkl"

    @property
    def stop_path(self) -> Path:
        return self.work_dir / "stop.requested"


def _write_descriptor(work_dir: Path, data: dict[str, Any]) -> None:
    """Write ``job.json`` (best-effort: never let bookkeeping kill a job)."""
    try:
        tmp = work_dir / "job.json.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, work_dir / "job.json")
    except OSError:
        pass


def _update_descriptor(work_dir: Path, **updates: Any) -> None:
    try:
        path = work_dir / "job.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(updates)
        _write_descriptor(work_dir, data)
    except (OSError, ValueError):
        pass


def _legacy_label_meta(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Reconstruct a display label + meta for a pre-descriptor work dir.

    Mirrors the labels/meta the Generate pages attach at enqueue time, so an
    adopted legacy job renders in the same panels a remembered one would.
    """
    qualname = str(spec.get("fn_qualname", ""))
    module = str(spec.get("fn_module", ""))
    kwargs = spec.get("kwargs", {}) or {}
    out_name = Path(str(kwargs.get("output_path", ""))).name
    db_name = Path(str(kwargs.get("db_path", ""))).name
    if "full_hand" in qualname:
        n = kwargs.get("total_hands")
        label = f"Postflop full hands: {db_name}" + (f" ({n} hands)" if n else "")
        return label, {}
    if module.startswith("pipeline.plo"):
        label = out_name or qualname
        return label, {"kind": "plo_generate", "output_name": out_name}
    if "preflop_entry" in qualname:
        return f"Preflop entry: {db_name}", {}
    if "postflop" in module:
        return f"Postflop: {db_name or out_name or qualname}", {}
    return out_name or qualname or "recovered job", {}


def _read_disk_job(work_dir: Path) -> DiskJobInfo | None:
    """Parse one ``pp_job_*`` dir into a :class:`DiskJobInfo`, or None."""
    name = work_dir.name  # pp_job_<id>_<random>
    parts = name.split("_")
    job_id = parts[2] if len(parts) >= 4 else name
    desc_path = work_dir / "job.json"
    if desc_path.is_file():
        try:
            d = json.loads(desc_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return DiskJobInfo(
            work_dir=work_dir,
            job_id=str(d.get("id", job_id)),
            label=str(d.get("label", "recovered job")),
            meta=dict(d.get("meta") or {}),
            started_at=float(d.get("started_at") or work_dir.stat().st_mtime),
            pid=int(d["pid"]) if d.get("pid") else None,
            stop_check_kwarg=d.get("stop_check_kwarg"),
            legacy=False,
        )
    spec_path = work_dir / "spec.pkl"
    if not spec_path.is_file():
        return None
    try:
        spec = pickle.loads(spec_path.read_bytes())
        label, meta = _legacy_label_meta(spec)
    except Exception:  # noqa: BLE001 -- unreadable spec: not adoptable
        return None
    try:
        started_at = spec_path.stat().st_mtime
    except OSError:
        started_at = time.time()
    return DiskJobInfo(
        work_dir=work_dir,
        job_id=job_id,
        label=label,
        meta=meta,
        started_at=started_at,
        pid=None,
        stop_check_kwarg=spec.get("stop_check_kwarg"),
        legacy=True,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _find_worker_pid(info: DiskJobInfo) -> int | None:
    """The child's pid: from the descriptor, else pgrep on the spec path
    (legacy dirs -- the spec path is unique per job, so a match is exact)."""
    if info.pid is not None:
        return info.pid
    if shutil.which("pgrep") is None:
        return None
    try:
        out = subprocess.run(
            ["pgrep", "-f", str(info.spec_path)],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return pids[0] if pids else None


def _probe_disk_job(info: DiskJobInfo) -> str:
    """``"finished"`` (result file present) / ``"running"`` / ``"dead"``."""
    if info.result_path.is_file():
        return "finished"
    pid = _find_worker_pid(info)
    if pid is not None and _pid_alive(pid):
        info.pid = pid  # remember for cancel support
        return "running"
    return "dead"


def _known_job_ids() -> set[str]:
    with _LOCK:
        ids = {j.id for j in _HISTORY}
        ids.update(_ADOPTED.keys())
        if _CURRENT_JOB is not None:
            ids.add(_CURRENT_JOB.id)
    return ids


def _harvest_result(job: Job[Any], info: DiskJobInfo) -> None:
    """Terminal state from the result envelope (adopted-job analogue of
    :func:`_finish_subprocess_job`; no cancel flag -- nobody asked)."""
    try:
        envelope = pickle.loads(info.result_path.read_bytes())
        finished_at = info.result_path.stat().st_mtime
    except Exception as exc:  # noqa: BLE001 -- incl. unpickle of stale classes
        with _LOCK:
            job.status = JobStatus.FAILED
            job.error = (
                f"Recovered batch finished but its result could not be read: {exc}. "
                "The output CSV (if any) is still on disk."
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
        job.finished_at = finished_at


def _retire_adopted(job: Job[Any], info: DiskJobInfo) -> None:
    """Move a terminal adopted job into history + clean its work dir."""
    with _LOCK:
        _ADOPTED.pop(job.id, None)
        _HISTORY.append(job)
        del _HISTORY[:-_HISTORY_MAX]
    shutil.rmtree(info.work_dir, ignore_errors=True)


def _watch_adopted(job: Job[Any], info: DiskJobInfo) -> None:
    """Supervisor for an adopted RUNNING job: mirror progress, then harvest.

    Harvests as soon as the result file appears (the child writes it
    atomically right before exiting); a vanished process with no result
    file means the child died -> FAILED with an honest message.
    """
    while True:
        _mirror_progress(job, info.progress_path)
        if info.result_path.is_file():
            _harvest_result(job, info)
            break
        pid = info.pid
        if pid is not None and not _pid_alive(pid):
            # Grace re-check: the child may have written the result between
            # our file check and the pid check.
            time.sleep(_POLL_SECONDS)
            if info.result_path.is_file():
                _harvest_result(job, info)
            else:
                with _LOCK:
                    job.status = JobStatus.FAILED
                    job.error = (
                        "The recovered batch's worker process exited without "
                        "writing a result (killed or crashed). Its output CSV "
                        "may be partial -- check the batch folder."
                    )
                    job.finished_at = time.time()
            break
        time.sleep(_POLL_SECONDS)
    _mirror_progress(job, info.progress_path)
    _retire_adopted(job, info)


def adopt_disk_jobs(tmp_root: str | None = None) -> list[Job[Any]]:
    """Rediscover subprocess jobs from ``pp_job_*`` dirs on disk.

    Idempotent and cheap (skips every already-known job id), so callers can
    run it on every panel render. Returns the jobs adopted THIS call:
    running ones (now watched, visible via :func:`adopted_jobs`) and
    finished/dead ones (already moved into :func:`job_history`).
    """
    root = Path(tmp_root or tempfile.gettempdir())
    try:
        candidates = sorted(
            p for p in root.iterdir() if p.is_dir() and p.name.startswith("pp_job_")
        )
    except OSError:
        return []
    if not candidates:
        return []
    with _ADOPT_LOCK:
        return _adopt_candidates(candidates)


def _adopt_candidates(candidates: list[Path]) -> list[Job[Any]]:
    known = _known_job_ids()
    adopted: list[Job[Any]] = []
    for work_dir in candidates:
        info = _read_disk_job(work_dir)
        if info is None or info.job_id in known:
            continue
        state = _probe_disk_job(info)
        job: Job[Any] = Job(
            id=info.job_id,
            label=info.label,
            meta=dict(info.meta),
            cancellable=False,
            graceful_stoppable=False,
            started_at=info.started_at,
        )
        if state == "running":
            job.status = JobStatus.RUNNING
            job.cancellable = info.pid is not None
            job.graceful_stoppable = bool(info.stop_check_kwarg)
            _mirror_progress(job, info.progress_path)
            with _LOCK:
                _ADOPTED[job.id] = (job, info)
            threading.Thread(
                target=_watch_adopted, args=(job, info), daemon=True,
                name=f"adopted-{job.id}",
            ).start()
        elif state == "finished":
            _harvest_result(job, info)
            _mirror_progress(job, info.progress_path)
            with _LOCK:
                _HISTORY.append(job)
                del _HISTORY[:-_HISTORY_MAX]
            shutil.rmtree(work_dir, ignore_errors=True)
        else:  # dead
            job.status = JobStatus.FAILED
            job.error = (
                "This batch was recovered from a previous panel session: its "
                "worker process is gone and no result was written. Its output "
                "CSV may be partial -- check the batch folder."
            )
            job.finished_at = time.time()
            _mirror_progress(job, info.progress_path)
            with _LOCK:
                _HISTORY.append(job)
                del _HISTORY[:-_HISTORY_MAX]
            shutil.rmtree(work_dir, ignore_errors=True)
        known.add(info.job_id)
        adopted.append(job)
    return adopted


def adopted_jobs() -> list[Job[Any]]:
    """Snapshot of currently-adopted (recovered, still running) jobs."""
    with _LOCK:
        return [job for job, _info in _ADOPTED.values()]


@dataclass
class JobBoard:
    """One consistent snapshot of EVERYTHING in flight, for the always-on
    sidebar board (July 2026, user ask: a batch must never 'stop showing
    up' while it is still generating).

    ``active`` is every job currently running or pending, whatever started
    it: the registry slot's job AND every adopted (recovered) one --
    ordered oldest-first so the board reads top-down in start order.
    ``queued`` is the FIFO of not-yet-started requests. ``last_done`` is
    the most recently finished job (slot or history), for the idle state.
    """

    active: list[Job[Any]] = field(default_factory=list)
    queued: list[QueuedRequest] = field(default_factory=list)
    last_done: Job[Any] | None = None


def job_board() -> JobBoard:
    """Snapshot the whole job landscape under one lock acquisition."""
    with _LOCK:
        active: list[Job[Any]] = []
        if _CURRENT_JOB is not None and _CURRENT_JOB.is_active:
            active.append(_CURRENT_JOB)
        active.extend(job for job, _info in _ADOPTED.values() if job.is_active)
        active.sort(key=lambda j: j.started_at)
        queued = list(_PENDING)
        done_candidates = [j for j in _HISTORY if j.is_done]
        if _CURRENT_JOB is not None and _CURRENT_JOB.is_done:
            done_candidates.append(_CURRENT_JOB)
        last_done = max(
            done_candidates,
            key=lambda j: j.finished_at or 0.0,
            default=None,
        )
    return JobBoard(active=active, queued=queued, last_done=last_done)


def request_adopted_stop(job_id: str) -> bool:
    """Graceful-stop an adopted job (touch its stop sentinel). True iff sent."""
    with _LOCK:
        entry = _ADOPTED.get(job_id)
    if entry is None:
        return False
    job, info = entry
    if not job.graceful_stoppable:
        return False
    try:
        info.stop_path.touch()
    except OSError:
        return False
    with _LOCK:
        job.stop_requested = True
    return True


def request_adopted_cancel(job_id: str) -> bool:
    """Terminate an adopted job's worker process. True iff a signal was sent."""
    with _LOCK:
        entry = _ADOPTED.get(job_id)
    if entry is None:
        return False
    job, info = entry
    if info.pid is None:
        return False
    try:
        os.kill(info.pid, 15)  # SIGTERM; the watcher observes the exit
    except OSError:
        return False
    return True


__all__ = [
    "DiskJobInfo",
    "Job",
    "JobProgress",
    "JobStatus",
    "QueuedRequest",
    "JobBoard",
    "adopt_disk_jobs",
    "adopted_jobs",
    "clear_current_job",
    "job_board",
    "enqueue_subprocess_job",
    "get_current_job",
    "has_active_job",
    "job_history",
    "pending_jobs",
    "remove_queued",
    "request_adopted_cancel",
    "request_adopted_stop",
    "request_cancel_current_job",
    "request_graceful_stop_current_job",
    "start_job",
    "start_subprocess_job",
]
