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
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.explanation_generator import ExplanationValidationError
from pipeline.postflop.action_history import format_question
from pipeline.postflop.claim_checker import POSTFLOP_CHECKER_SYSTEM_PROMPT
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
from pipeline.postflop.layer7 import run_layer7_audit
from pipeline.postflop.options import build_options
from pipeline.postflop.premise import DEFAULT_MIN_PREMISE_FREQ, line_premise_min_freq
from pipeline.postflop.quality import node_quality_issue
from pipeline.postflop.question_extractor import (
    MAX_FREQUENCY,
    MIN_FREQUENCY,
    evaluate_spot,
)
from pipeline.postflop.solve import PostflopSolve, validate_solve
from pipeline.postflop.spot_sampler import enumerate_spots

ProgressCallback = Any  # callable(message: str, done: int, total: int) | None

logger = logging.getLogger(__name__)


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
    quality_gate: bool = True,
    min_premise_freq: float | None = None,
) -> tuple[list[Any], int, int]:
    """Every worthy spot in the solve, in a deterministic (node, combo) order.

    Returns ``(worthy_spots, low_quality_nodes_skipped, premise_filtered_nodes)``.
    When ``quality_gate`` is on (default), nodes that fail the convergence / reach
    guard (:func:`pipeline.postflop.quality.node_quality_issue`) are skipped
    whole. When ``min_premise_freq`` is set, a node whose line includes a PRIOR
    action taken below that frequency
    (:func:`pipeline.postflop.premise.line_premise_min_freq`) is also skipped --
    a question built on an action someone almost never takes. Both are per-NODE
    gates (the verdict is the same for every hero hand at the node), applied
    BEFORE any equity sim.
    """
    worthy: list[Any] = []
    low_quality = 0
    premise_skipped = 0
    for node_id in sorted(solve.nodes):
        node = solve.nodes[node_id]
        if quality_gate and node_quality_issue(node) is not None:
            low_quality += 1
            continue
        if min_premise_freq is not None:
            pmf = line_premise_min_freq(node, solve)
            if pmf is not None and pmf < min_premise_freq:
                premise_skipped += 1
                continue
        for spot in enumerate_spots(node):
            ev = evaluate_spot(
                spot,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                min_ev_gap_bb=min_ev_gap_bb,
            )
            if ev.is_worthy:
                worthy.append(spot)
    return worthy, low_quality, premise_skipped


def _node_range_snapshots(node: Any) -> dict[str, dict[str, float]]:
    """169-class range snapshot for BOTH players at ``node``, keyed by position.

    Reuses the pure preflop-pack aggregator (combo-range + board -> mean weight
    per UNBLOCKED class) so the Review page's postflop range grids read exactly
    like the preflop range charts. Returns ``{position: {hand_class: weight}}``
    with weights in [0, 1]. Deterministic from solver output (goes in meta.json,
    never the CSV, so byte-identical-CSV determinism is unaffected).
    """
    # Pure leaf util (combo -> 169-class), reused like cards / fact_extractor;
    # lazy-imported to keep the postflop core decoupled at module load.
    from pipeline.preflop_ranges import (  # noqa: PLC0415
        aggregate_combo_range_to_classes,
    )

    board = list(node.board)
    return {
        node.actor: aggregate_combo_range_to_classes(dict(node.hero_range), board),
        node.villain: aggregate_combo_range_to_classes(dict(node.villain_range), board),
    }


def _node_actor_strategy(node: Any) -> dict[str, dict[str, float]]:
    """The ACTOR's per-action 169-class strategy at ``node`` -- the data for an
    ACTION-COLOURED grid (check / bet / call / raise per hand), like the preflop
    range charts, instead of flat presence.

    Returns ``{action_label: {hand_class: weight}}`` where each weight is the
    reach-weighted ``P(action)`` for that class, so a class's per-action
    segments sum to its presence (a hand that is 50% of the range and bets it
    all shows one half-height bet band). Reuses the board-aware aggregator.
    """
    from pipeline.preflop_ranges import (  # noqa: PLC0415
        aggregate_combo_range_to_classes,
    )

    board = list(node.board)
    out: dict[str, dict[str, float]] = {}
    for action in {a.label for a in node.actions}:
        # weight of (combo plays action) = reach(combo) * P(action | combo).
        per_combo = {
            c: node.hero_range.get(c, 0.0) * node.strategy.get(c, {}).get(action, 0.0)
            for c in node.hero_range
        }
        out[action] = aggregate_combo_range_to_classes(per_combo, board)
    return out


def _villain_decision_node(node: Any, solve: Any) -> Any:
    """The most recent node where ``node.villain`` acted -- so we can show the
    VILLAIN's own strategy (e.g. BB's check/donk) rather than flat presence.

    The villain's decision is the longest node-id PREFIX of the current node
    whose actor is the villain. Returns None when the villain has not acted on
    this line yet (they are first to act on the current street), or when that
    node was down-sampled out of the solve.
    """
    villain = node.villain
    nid = node.node_id
    best = None
    for cid, cand in solve.nodes.items():
        if cand.actor == villain and nid.startswith(cid + ":"):
            if best is None or len(cid) > len(best.node_id):
                best = cand
    return best


# The street immediately before each decision street (flop's prior is preflop).
_PRIOR_STREET = {"turn": "flop", "river": "turn"}


def _prior_street_node(node: Any, solve: Any) -> Any:
    """The decision node on the street BEFORE this one whose ranges to show.

    For the range visuals: a turn question should show the FLOP range (not the
    preflop / flop-entry range), a river question the TURN range -- "the street
    before". Returns the deepest node-id PREFIX of the current node that is a
    real decision node on the prior street (so its ``hero_range`` /
    ``villain_range`` are the ranges as they were on that street). ``None`` for a
    flop question (its prior is preflop -> the caller uses the shared flop-entry
    ranges) or if no prior-street ancestor survived down-sampling (turn nodes are
    sampled; flop nodes are always kept, so a turn question always resolves).
    """
    prior = _PRIOR_STREET.get(node.street)
    if prior is None:
        return None
    parts = node.node_id.split(":")
    for i in range(len(parts) - 1, 1, -1):  # longest prefix first, excluding self
        cand = solve.nodes.get(":".join(parts[:i]))
        if cand is not None and cand.street == prior:
            return cand
    return None


def _street_strategies(node: Any, solve: Any) -> dict[str, dict]:
    """Both players' action strategies for the current street, keyed by
    position: the ACTOR at this node, and the VILLAIN at the node where THEY
    last acted (so BB's flop check/donk shows on BB's grid even when BTN is the
    hero). Villain omitted when they have not acted on this street yet.
    """
    out: dict[str, dict] = {node.actor: _node_actor_strategy(node)}
    vnode = _villain_decision_node(node, solve)
    if vnode is not None:  # vnode.actor == node.villain
        out[vnode.actor] = _node_actor_strategy(vnode)
    return out


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
    answer_style: str = "auto",
    display_in_bb: bool = True,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    min_ev_gap_bb: float | None = None,
    quality_gate: bool = True,
    min_premise_freq: float | None = DEFAULT_MIN_PREMISE_FREQ,
    equity_runouts: int = DEFAULT_EQUITY_RUNOUTS,
    system_prompt: str | None = None,
    trap_difficulty: bool = False,
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    revise_pass: bool = False,
    final_audit: bool = False,
    progress_callback: ProgressCallback = None,
    write_meta: bool = True,
    spot_selector: Callable[[list[Any]], list[Any]] | None = None,
    provenance: dict[str, Any] | None = None,
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
    worthy, low_quality_filtered_out, premise_filtered_out = _collect_worthy(
        solve,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        min_ev_gap_bb=min_ev_gap_bb,
        quality_gate=quality_gate,
        min_premise_freq=min_premise_freq,
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
    # Layer-7 LLM audit tallies (move only when the opt-in passes run):
    #   claim_flagged -- shipped rows the claim checker flagged (flag-only path);
    #   revise_* -- the auto-fix lifecycle (flagged = fixed + discarded + unchanged).
    claim_flagged = 0
    revise_flagged = 0
    revise_fixed = 0
    revise_discarded = 0
    revise_unchanged = 0
    # Counterintuitive "trap" spots floored to Hard by the opt-in trap_difficulty
    # mode (0 when off) -- lets the user see how many spots it re-rated.
    trap_floored = 0
    checker_prompt = claim_checker_prompt or POSTFLOP_CHECKER_SYSTEM_PROMPT

    # Visual range data for the Review page (preflop-parity). The FLOP ROOT --
    # OOP to act, no postflop action yet -- holds each player's flop-ENTRY
    # ("preflop") range; it's the same for the whole batch. Each question then
    # carries its own CURRENT-street snapshot. Both are 169-class, keyed by
    # position, so the Review page renders standard range grids per player.
    _flop_root = next(
        (n for n in solve.nodes.values() if n.street == "flop" and not n.history),
        None,
    )
    preflop_ranges = (
        _node_range_snapshots(_flop_root) if _flop_root is not None else {}
    )
    # How each player REACHED this flop, so the Review page can colour the
    # preflop grid by entry action (makes "BB shows no raises" self-evident: it
    # CALLED). Single-raised pot => the IP player opened (raise), OOP called.
    # (A future 3-bet-pot solve would set these from its own line.)
    preflop_entry_actions = {
        solve.ip_position: "raise",
        solve.oop_position: "call",
    }

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
        options, correct = build_options(spot, style=answer_style)
        difficulty = compute_difficulty(facts, apply_trap_bump=trap_difficulty)
        if difficulty.trap_bump_applied:
            trap_floored += 1

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

        solver_data_block = build_solver_data_block(facts)
        # The action-history narrative (the SAME string the reader + the writer
        # see). The data block never restates the line, so the Layer-7 checker /
        # reviser need this to verify line references (a donk-lead, a check-raise)
        # instead of false-flagging them as invented action history.
        question_text = format_question(facts.spot, solve, display_in_bb=display_in_bb)

        # --- Layer-7 LLM audit / revise passes (opt-in extra LLM calls) ------
        # Shared with the full-hand driver (pipeline.postflop.layer7) so a leg
        # gets the same QA as a standalone spot. Real runs only (a placeholder is
        # never checked/revised).
        claim_check_json = ""
        claim_issues: list[str] = []
        revise_record: dict[str, Any] | None = None
        remaining_issues: list[str] = []
        if not use_placeholder and (run_claim_checker or revise_pass):
            l7 = run_layer7_audit(
                explanation, facts,
                solver_data_block=solver_data_block, question_text=question_text,
                node_id=spot.node.node_id, client=client, model=model,
                temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, checker_prompt=checker_prompt,
                run_claim_checker=run_claim_checker, revise_pass=revise_pass,
                final_audit=final_audit, usage_callback=_record_usage,
            )
            explanation = l7.explanation  # possibly the rewrite
            claim_check_json = l7.claim_check_json
            claim_issues = l7.claim_issues
            revise_record = l7.revise_record
            remaining_issues = l7.remaining_issues
            claim_flagged += l7.claim_flagged
            revise_flagged += l7.revise_flagged
            revise_fixed += l7.revise_fixed
            revise_discarded += l7.revise_discarded
            revise_unchanged += l7.revise_unchanged

        from pipeline.postflop.validators import run_postflop_soft_validators

        # Soft validators run on the FINAL (possibly revised) explanation.
        soft_warnings = (
            [] if use_placeholder
            else run_postflop_soft_validators(explanation, facts)
        )
        status = "flagged" if (soft_warnings or remaining_issues) else "draft"
        if soft_warnings:
            soft_flagged += 1

        row = build_postflop_row(
            facts, explanation, solve, difficulty, len(rows) + 1,
            validation_status=status, display_in_bb=display_in_bb,
        )
        row["claim_check"] = claim_check_json
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
            # Node-level range-vs-range stats (the "who is ahead here" facts).
            "range_equity": round(facts.hero_range_equity, 4),
            "range_advantage": facts.range_advantage,
            "hero_nut_share": round(facts.hero_nut_share, 4),
            "villain_nut_share": round(facts.villain_nut_share, 4),
            "nut_advantage": facts.nut_advantage,
            # Blocker value/bluff decomposition.
            "blocked_value_pct": round(facts.blocked_value_pct, 4),
            "blocked_bluff_pct": round(facts.blocked_bluff_pct, 4),
            "blocker_effect": facts.blocker_effect,
            "ev_gap_bb": facts.ev_gap_bb,
            "concept_tags": facts.concept_tags,
            "solver_data": solver_data_block,
            # Each player's range at THIS (current-street) node, 169-class,
            # keyed by position -- the RIGHT range grid on the Review page.
            "street_ranges": _node_range_snapshots(spot.node),
            # BOTH players' per-action strategy for the current street (the actor
            # at this node + the villain at the node where THEY acted), keyed by
            # position -- the action-coloured grids. + who the hero/actor is.
            "street_strategy": _street_strategies(spot.node, solve),
            "street_actor": spot.node.actor,
        }
        # The LEFT range grid = "the street before" (June 2026): a turn question
        # shows the FLOP range, a river question the TURN range. Flop questions
        # leave this empty -> the Review page falls back to the shared flop-entry
        # (preflop) ranges. So the two grids are always (prior street, current).
        _prior_node = _prior_street_node(spot.node, solve)
        if _prior_node is not None:
            record["prior_street_ranges"] = _node_range_snapshots(_prior_node)
            record["prior_street_label"] = _prior_node.street
        if soft_warnings:
            record["validator_warnings"] = soft_warnings
        if claim_issues:
            record["claim_check_issues"] = claim_issues
        if revise_record is not None:
            record["revise"] = revise_record
        question_records.append(record)

    output_path = Path(output_path)
    write_postflop_csv(output_path, rows)

    meta_path: Path | None = None
    if write_meta:
        meta_path = output_path.with_suffix(".meta.json")
        meta = {
            "solve_id": solve.solve_id,
            "source_reference": solve.source_reference,
            # Flop-entry ("preflop") range per player (169-class), shared by
            # every question -- the Review page pairs it with each question's
            # street_ranges to show preflop + current-street grids per player.
            "preflop_ranges": preflop_ranges,
            "preflop_entry_actions": preflop_entry_actions,
            "model": model if not use_placeholder else "(dry-run placeholder)",
            "dry_run": use_placeholder,
            # How to reload the exact source solve for the batch re-verifier
            # (db_path / streets / max_nodes_per_street). Empty for fixture runs.
            "provenance": provenance or {},
            "run_settings": {
                "total_questions": total_questions,
                "answer_style": answer_style,
                "display_in_bb": display_in_bb,
                "min_frequency": min_frequency,
                "max_frequency": max_frequency,
                "min_ev_gap_bb": min_ev_gap_bb,
                "quality_gate": quality_gate,
                "min_premise_freq": min_premise_freq,
                "equity_runouts": equity_runouts,
                "temperature": temperature,
                "trap_difficulty": trap_difficulty,
                "run_claim_checker": run_claim_checker,
                "revise_pass": revise_pass,
                "final_audit": final_audit and revise_pass,
            },
            "counters": {
                "worthy_spots_available": worthy_total,
                "worthy_spots_selected": len(worthy),
                "low_quality_nodes_skipped": low_quality_filtered_out,
                "premise_filtered_nodes": premise_filtered_out,
                "questions_attempted": attempted,
                "questions_written": len(rows),
                "soft_flagged_rows": soft_flagged,
                "trap_floored": trap_floored,
                # Layer-7 audit tallies (0 unless the opt-in passes ran).
                "claim_flagged_rows": claim_flagged,
                "revise_flagged": revise_flagged,
                "revise_fixed": revise_fixed,
                "revise_discarded": revise_discarded,
                "revise_unchanged": revise_unchanged,
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
