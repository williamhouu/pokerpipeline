"""Postflop batch driver -- the end-to-end orchestrator.

Takes a :class:`~pipeline.postflop.solve.PostflopSolve` (from the synthetic
fixture or a real-solve adapter) and produces a CSV of questions plus a
``.meta.json`` provenance sidecar, exactly like the preflop batch driver but
for postflop. The per-question loop is:

    enumerate nodes -> sample spots -> worthiness gate -> extract facts ->
    build options + difficulty -> write explanation (LLM, or a deterministic
    placeholder on a dry run) -> soft validators -> CSV row + meta record

It runs with NO API key when ``dry_run=True`` (or ``client=None``): every
strategic fact is deterministic and the explanation is the placeholder. A
real run passes an Anthropic ``client`` and gets LLM prose with one corrective
retry; a spot whose explanation can't pass the hard validators is recorded as
a failure (with its attempted text) and the batch continues -- one bad spot
never aborts the run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.explanation_generator import ExplanationValidationError
from pipeline.postflop.difficulty import compute_difficulty
from pipeline.postflop.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    build_solver_data_block,
    generate_postflop_explanation,
    placeholder_explanation,
)
from pipeline.postflop.facts import DEFAULT_EQUITY_RUNOUTS, extract_facts
from pipeline.postflop.format_writer import build_postflop_row, write_postflop_csv
from pipeline.postflop.options import build_options
from pipeline.postflop.question_extractor import (
    MAX_FREQUENCY,
    MIN_FREQUENCY,
    evaluate_spot,
)
from pipeline.postflop.solve import PostflopSolve, validate_solve
from pipeline.postflop.spot_sampler import enumerate_spots

ProgressCallback = Any  # callable(message: str, done: int, total: int) | None


@dataclass
class PostflopBatchResult:
    """Summary of one postflop batch run."""

    output_path: Path
    questions_written: int
    questions_attempted: int
    worthy_spots_available: int
    requested_questions: int
    dry_run: bool
    model_used: str
    failures: list[dict[str, Any]] = field(default_factory=list)
    soft_flagged_rows: int = 0
    meta_path: Path | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def _collect_worthy(
    solve: PostflopSolve,
    *,
    min_frequency: float,
    max_frequency: float,
    min_ev_gap_bb: float | None,
) -> list[Any]:
    """Every worthy spot in the solve, in a deterministic (node, combo) order."""
    worthy = []
    for node_id in sorted(solve.nodes):
        node = solve.nodes[node_id]
        for spot in enumerate_spots(node):
            ev = evaluate_spot(
                spot,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                min_ev_gap_bb=min_ev_gap_bb,
            )
            if ev.is_worthy:
                worthy.append(spot)
    return worthy


def generate_postflop_batch(
    *,
    solve: PostflopSolve,
    output_path: Path | str,
    total_questions: int,
    client: object | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    dry_run: bool = False,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    min_ev_gap_bb: float | None = None,
    equity_runouts: int = DEFAULT_EQUITY_RUNOUTS,
    system_prompt: str | None = None,
    progress_callback: ProgressCallback = None,
    write_meta: bool = True,
    spot_selector: Callable[[list[Any]], list[Any]] | None = None,
) -> PostflopBatchResult:
    """Generate up to ``total_questions`` postflop questions from ``solve``.

    When ``dry_run`` is True (or ``client`` is None) the explanation is the
    deterministic placeholder and no API key is needed. Returns a
    :class:`PostflopBatchResult`; writes the CSV and (unless ``write_meta`` is
    False) a ``<stem>.meta.json`` sidecar.

    ``spot_selector`` (optional) receives the full worthy-spot list and returns
    the curated/ordered subset to actually generate -- used to diversify a
    fill-to-N batch across node types so it is not dominated by one archetype.
    It must be deterministic to preserve byte-identical output. Default: the
    worthy list is used as-is (sorted by node, then combo).

    Raises:
        ValueError: if ``solve`` fails the structural check in
            :func:`pipeline.postflop.solve.validate_solve`.
    """
    problems = validate_solve(solve)
    if problems:
        raise ValueError(f"solve {solve.solve_id} is malformed: {problems}")

    use_placeholder = dry_run or client is None
    worthy = _collect_worthy(
        solve,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        min_ev_gap_bb=min_ev_gap_bb,
    )
    worthy_total = len(worthy)
    # Optional caller-supplied curation/ordering (e.g. diversify across node
    # types so a fill-to-N batch is not dominated by one archetype). Must be
    # deterministic to preserve byte-identical output. Default: take as-is.
    if spot_selector is not None:
        worthy = spot_selector(worthy)

    rows: list[dict[str, str]] = []
    question_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempted = 0
    soft_flagged = 0
    in_tokens = out_tokens = 0

    def _record_usage(usage: object) -> None:
        nonlocal in_tokens, out_tokens
        in_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    for spot in worthy:
        if len(rows) >= total_questions:
            break
        attempted += 1
        if progress_callback is not None:
            progress_callback(
                f"Generating {len(rows) + 1}/{total_questions} "
                f"({spot.node.node_id} / {spot.hero_combo})",
                len(rows),
                total_questions,
            )

        facts = extract_facts(spot, solve, equity_runouts=equity_runouts)
        options, correct = build_options(spot)
        difficulty = compute_difficulty(facts)

        try:
            if use_placeholder:
                explanation = placeholder_explanation(facts, options, correct)
            else:
                explanation = generate_postflop_explanation(
                    facts, options, correct, solve,
                    client=client, model=model, temperature=temperature,
                    max_tokens=max_tokens, system_prompt=system_prompt,
                    usage_callback=_record_usage,
                )
        except ExplanationValidationError as exc:
            failures.append({
                "node_id": spot.node.node_id,
                "hero_combo": spot.hero_combo,
                "error_message": str(exc),
                "attempt_text": exc.last_attempt_text,
            })
            continue

        from pipeline.postflop.validators import run_postflop_soft_validators

        soft_warnings = (
            [] if use_placeholder
            else run_postflop_soft_validators(explanation, facts)
        )
        status = "flagged" if soft_warnings else "draft"
        if soft_warnings:
            soft_flagged += 1

        row = build_postflop_row(
            facts, explanation, solve, difficulty, len(rows) + 1,
            validation_status=status,
        )
        rows.append(row)

        record: dict[str, Any] = {
            "node_id": spot.node.node_id,
            "hero_combo": spot.hero_combo,
            "street": facts.street,
            "correct_answer": correct,
            "options": options,
            "archetype": facts.archetype,
            "difficulty": difficulty.score,
            "hero_equity": round(facts.hero_equity_vs_villain, 4),
            "ev_gap_bb": facts.ev_gap_bb,
            "concept_tags": facts.concept_tags,
            "solver_data": build_solver_data_block(facts),
        }
        if soft_warnings:
            record["validator_warnings"] = soft_warnings
        question_records.append(record)

    output_path = Path(output_path)
    write_postflop_csv(output_path, rows)

    meta_path: Path | None = None
    if write_meta:
        meta_path = output_path.with_suffix(".meta.json")
        meta = {
            "solve_id": solve.solve_id,
            "source_reference": solve.source_reference,
            "model": model if not use_placeholder else "(dry-run placeholder)",
            "dry_run": use_placeholder,
            "run_settings": {
                "total_questions": total_questions,
                "min_frequency": min_frequency,
                "max_frequency": max_frequency,
                "min_ev_gap_bb": min_ev_gap_bb,
                "equity_runouts": equity_runouts,
                "temperature": temperature,
            },
            "counters": {
                "worthy_spots_available": worthy_total,
                "worthy_spots_selected": len(worthy),
                "questions_attempted": attempted,
                "questions_written": len(rows),
                "soft_flagged_rows": soft_flagged,
            },
            "questions": question_records,
            "failures": failures,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return PostflopBatchResult(
        output_path=output_path,
        questions_written=len(rows),
        questions_attempted=attempted,
        worthy_spots_available=worthy_total,
        requested_questions=total_questions,
        dry_run=use_placeholder,
        model_used=model if not use_placeholder else "(dry-run placeholder)",
        failures=failures,
        soft_flagged_rows=soft_flagged,
        meta_path=meta_path,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
    )


__all__ = ["PostflopBatchResult", "generate_postflop_batch"]
