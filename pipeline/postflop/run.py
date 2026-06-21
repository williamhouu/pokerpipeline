"""One-call postflop generation from a ``.db`` path -- the orchestration entry.

Ties the vendor adapter, the spot selector, and the batch driver together
behind a single top-level function so BOTH the CLI and the admin panel's
subprocess job runner share one path. ``generate_postflop_batch_from_db`` is a
plain module-level function with picklable arguments, which is exactly what
:func:`admin_panel.jobs.start_subprocess_job` needs (it ships the ``.db`` PATH
to the child and loads the solve there, rather than pickling a multi-hundred-MB
in-memory solve across the process boundary).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pipeline.postflop.adapters.sqlite_db import load_postflop_db
from pipeline.postflop.batch import PostflopBatchResult, generate_postflop_batch
from pipeline.postflop.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    load_postflop_system_prompt,
)
from pipeline.postflop.question_extractor import MAX_FREQUENCY, MIN_FREQUENCY
from pipeline.postflop.spot_selection import make_spot_selector

# Where the admin writes postflop batches (sibling of the preflop output dir).
POSTFLOP_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "test_output" / "postflop_batches"
)


def generate_postflop_batch_from_db(
    *,
    db_path: str,
    output_path: str | Path,
    total_questions: int,
    heroes: tuple[str, ...] = (),
    diversify: bool = False,
    stakes: str = "$1/$2",
    live_or_online: str = "Live",
    bb_in_dollars: float = 2.0,
    answer_style: str = "auto",
    display_in_bb: bool = True,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    min_ev_gap_bb: float | None = None,
    equity_runouts: int | None = None,
    system_prompt: str | None = None,
    progress_callback: Any = None,
) -> PostflopBatchResult:
    """Load the ``.db`` solve, curate spots, and generate a batch.

    The solve's strategic scenario (table size, positions, cash/tournament) comes
    from the file's own metadata; only the display framing (``stakes`` /
    ``live_or_online`` / ``bb_in_dollars``) is passed in here, since the solve
    doesn't carry stakes. ``heroes`` keeps only those acting positions (empty =
    both); ``diversify`` round-robins the decision types. A real run needs
    ``ANTHROPIC_API_KEY``; without it (or with ``dry_run``) the deterministic
    placeholder prose is used.
    """
    solve = load_postflop_db(
        db_path,
        stakes=stakes,
        live_or_online=live_or_online,
        bb_in_dollars=bb_in_dollars,
    )
    selector = make_spot_selector(heroes=tuple(heroes) or None, diversify=diversify)

    client = None
    if not dry_run and os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic  # noqa: PLC0415 -- only for a real run

        client = Anthropic()

    kwargs: dict[str, Any] = {}
    if equity_runouts is not None:
        kwargs["equity_runouts"] = equity_runouts

    # Resolve the system prompt in the child (the admin override takes effect on
    # the next run without restarting). An explicit system_prompt still wins.
    prompt = system_prompt if system_prompt is not None else load_postflop_system_prompt()

    return generate_postflop_batch(
        solve=solve,
        output_path=output_path,
        total_questions=total_questions,
        client=client,
        model=model,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        dry_run=dry_run or client is None,
        answer_style=answer_style,
        display_in_bb=display_in_bb,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        min_ev_gap_bb=min_ev_gap_bb,
        system_prompt=prompt,
        progress_callback=progress_callback,
        spot_selector=selector,
        **kwargs,
    )


__all__ = ["POSTFLOP_OUTPUT_DIR", "generate_postflop_batch_from_db"]
