"""Out-of-process entry point for a background job.

Run as a fresh interpreter:

    python -m admin_panel.job_worker <spec_path>

``<spec_path>`` is a pickle of a spec dict (written by
:func:`admin_panel.jobs.start_subprocess_job`):

    {
      "fn_module":  "pipeline.preflop.batch",   # importable module
      "fn_qualname": "generate_preflop_batch",   # attribute on it
      "kwargs": { ... },                          # picklable kwargs for fn
      "progress_callback_kwarg": "progress_callback",
      "progress_path": "/tmp/.../progress.json",  # this process writes it
      "result_path":   "/tmp/.../result.pkl",     # this process writes it
    }

Why a separate process? Generation is API-latency-bound but still does
real Python work between calls (equity sims, the SDK's response parsing).
Run inside the Streamlit process -- as the old in-process thread did -- it
contends with the UI for the single Python GIL, so a running batch freezes
the panel. A child process has its own interpreter and its own GIL, so the
UI stays responsive while the batch runs here.

This module is deliberately **Streamlit-free** and imports nothing heavy at
module load: the target module is imported lazily by name, so the child
starts fast and never re-imports the admin app.

Contract with the parent:
  * progress -- the injected callback writes ``{message, current, total}``
    to ``progress_path`` with an atomic replace; the parent polls it.
  * result   -- on exit this process writes a pickle envelope to
    ``result_path``: ``{"ok": True, "result": <return value>}`` on success
    or ``{"ok": False, "error": "<traceback>"}`` on failure. The parent
    treats a missing/garbled file (e.g. the process was killed) as a
    failure/cancellation.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _atomic_write(path: str, write: Callable[[str], None]) -> None:
    """Write via a temp file + ``os.replace`` so a reader never sees a
    half-written file (replace is atomic on POSIX)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    write(tmp)
    os.replace(tmp, path)


def _write_progress(path: str, message: str, current: int, total: int) -> None:
    def _w(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {"message": message, "current": current, "total": total}, fh
            )

    _atomic_write(path, _w)


def _write_result(path: str, envelope: dict[str, Any]) -> None:
    def _w(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            pickle.dump(envelope, fh, protocol=pickle.HIGHEST_PROTOCOL)

    _atomic_write(path, _w)


def _resolve_fn(module: str, qualname: str) -> Any:
    """Import ``module`` and walk ``qualname`` (dotted) to the callable.

    Top-level functions (the only thing the parent sends today) resolve in
    one ``getattr``; the dotted walk is just defensive."""
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def run_spec(spec_path: str) -> int:
    """Execute the job described by the pickled spec at ``spec_path``.

    Returns a process exit code (0 on success, 1 on failure). The actual
    return value / traceback is communicated via the result file, not the
    exit code -- the code is just a coarse signal for the parent."""
    spec = pickle.loads(Path(spec_path).read_bytes())
    progress_path: str = spec["progress_path"]
    result_path: str = spec["result_path"]
    progress_kwarg: str = spec.get("progress_callback_kwarg", "progress_callback")

    def _progress(message: str, current: int, total: int) -> None:
        # Best-effort: a progress-write failure must never crash the job.
        try:
            _write_progress(progress_path, message, int(current), int(total))
        except Exception:  # noqa: BLE001
            pass

    # Graceful stop (July 2026): when the spec names a stop kwarg, inject a
    # callable that reports True once the parent touches the sentinel file.
    # The target checks it between units of work (e.g. between questions)
    # and returns normally with everything committed so far.
    injected: dict[str, Any] = {progress_kwarg: _progress}
    stop_kwarg = spec.get("stop_check_kwarg")
    stop_path = spec.get("stop_path")
    if stop_kwarg and stop_path:
        injected[stop_kwarg] = lambda: os.path.exists(stop_path)

    try:
        fn = _resolve_fn(spec["fn_module"], spec["fn_qualname"])
        result = fn(**{**injected, **spec["kwargs"]})
        _write_result(result_path, {"ok": True, "result": result})
        return 0
    except BaseException:  # noqa: BLE001 -- capture EVERYTHING for the parent
        # KeyboardInterrupt/SystemExit included: if we got far enough to be
        # here (not killed by a signal), record why so the parent can show it.
        try:
            _write_result(
                result_path, {"ok": False, "error": traceback.format_exc()}
            )
        except Exception:  # noqa: BLE001
            pass
        return 1


# --- self-test target -------------------------------------------------------
# A tiny importable job used by the jobs integration test (and as a smoke
# target). It lives here so the test has a top-level, picklable-by-reference
# function the child can import by (module, qualname) without depending on
# the test module being importable from a fresh interpreter.
def _echo_job(
    *,
    progress_callback: Callable[[str, int, int], None],
    value: str = "ok",
    steps: int = 2,
    fail: bool = False,
    sleep_s: float = 0.0,
) -> str:
    import time as _time

    for i in range(steps):
        progress_callback(f"step {i + 1}", i + 1, steps)
        if sleep_s:
            _time.sleep(sleep_s)
    if fail:
        raise RuntimeError("echo job asked to fail")
    return value


def _stoppable_job(
    *,
    progress_callback: Callable[[str, int, int], None],
    stop_check: Callable[[], bool] | None = None,
    steps: int = 100,
    sleep_s: float = 0.1,
) -> int:
    """Self-test target for the graceful-stop path: loops ``steps`` times,
    breaking early the moment ``stop_check`` reports True. Returns the
    number of completed steps."""
    import time as _time

    done = 0
    for i in range(steps):
        if stop_check is not None and stop_check():
            break
        _time.sleep(sleep_s)
        done = i + 1
        progress_callback(f"step {done}", done, steps)
    return done


if __name__ == "__main__":
    sys.exit(run_spec(sys.argv[1]))
