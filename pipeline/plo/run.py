"""Picklable job entry point for PLO batch generation (July 2026).

The admin panel runs PLO generation as a background SUBPROCESS job
(``admin_panel.jobs``) so a Streamlit rerun -- any click, any page switch --
can never kill a batch mid-flight (the old inline call died at its next
``st.progress`` update and lost the whole batch while the API spend had
already happened). The job worker resolves its target by module + qualname,
so the target must be a top-level importable function with picklable kwargs:
this module is that seam, the PLO analogue of :mod:`pipeline.postflop.run`.

The only logic here is protocol adaptation: the job worker injects a
3-arg progress callback ``(message, current, total)`` while
:func:`pipeline.plo.batch.generate_plo_batch` takes a 2-arg
``(done, total)`` one. Everything else passes straight through, and the
returned :class:`~pipeline.plo.batch.PloBatchResult` (which carries the
batch's token totals) pickles back to the parent for the spend logger.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pipeline.plo.batch import PloBatchResult, generate_plo_batch


def run_plo_generate_job(
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
    **kwargs: Any,
) -> PloBatchResult:
    """Run one PLO batch; adapt the job progress protocol; return the result.

    ``kwargs`` are exactly :func:`generate_plo_batch`'s (all picklable --
    ``pack`` is a frozen dataclass of paths/strings). The child process pays
    one cold pack enumeration (~seconds); acceptable next to a multi-minute
    LLM batch. ``stop_check`` (injected by the job worker when the parent
    supports graceful stop) passes straight through: the batch checks it
    between questions and ships everything committed so far.
    """
    total = int(kwargs.get("total_questions", 0) or 0)
    if progress_callback is not None:
        # The batch only reports after its first committed question; the pack
        # walk + spot sampling before that can take a while, so say so.
        progress_callback("Loading pack + sampling spots…", 0, total)

    def _adapt(done: int, batch_total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                f"Generated {done} / {batch_total} questions…",
                done,
                batch_total,
            )

    return generate_plo_batch(
        progress_callback=_adapt, stop_check=stop_check, **kwargs
    )


__all__ = ["run_plo_generate_job"]
