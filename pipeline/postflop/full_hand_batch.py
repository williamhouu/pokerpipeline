"""Batch drivers for play-through (full-hand) and standalone-preflop questions.

Kept separate from :mod:`pipeline.postflop.batch` (the standalone per-spot
driver) so the tested standalone path is untouched. Both drivers here reuse the
same leaf layers -- the fact extractor, option builder, difficulty, the Layer-6
explanation generator, and the CSV row builder -- and the preflop-entry module,
adding only the play-through assembly + the Option-B sequence tags.

* :func:`generate_full_hand_batch` -- assemble up to ``total_hands`` connected
  hands and emit one CSV row per leg (preflop entry, then each street's hero
  decision), all sharing a ``hand_id`` with an ascending ``sequence_index``.
* :func:`generate_preflop_entry_batch` -- standalone preflop-entry questions
  from the solve's flop-entry ranges (no play-through linkage; ``hand_id`` blank).

Both run with NO API key when ``dry_run=True`` (or ``client=None``): every
strategic fact is deterministic and the explanation is the placeholder.
Deterministic ordering + a content-hash ``hand_id`` keep the CSV byte-identical.

The opt-in Layer-7 audit/revise runs on the full-hand POSTFLOP legs via the
shared :func:`pipeline.postflop.layer7.run_layer7_audit` (so a leg gets the same
QA as a standalone spot); the preflop-entry leg is skipped (the checker prompt is
postflop-specific). Re-verify a finished batch with
``scripts/audit_full_hand_batch.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline.explanation_generator import ExplanationValidationError
from pipeline.postflop.action_history import format_question
from pipeline.postflop.batch import (
    PostflopBatchResult,
    _collect_worthy,
    _node_range_snapshots,
    _prior_street_node,
    _street_strategies,
)
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
from pipeline.postflop.play_through import assemble_hands
from pipeline.postflop.preflop_leg_pack import (
    _build_pack_facts,
    _facts_difficulty,
    build_pack_preflop_leg_row,
    find_pack_leg_source,
)
from pipeline.postflop.preflop_entry import (
    build_preflop_entry_options,
    build_preflop_entry_row,
    enumerate_preflop_entry_facts,
    generate_preflop_entry_explanation,
    placeholder_preflop_entry_explanation,
    preflop_entry_is_worthy,
    standalone_entry_is_reliable,
)
from pipeline.postflop.premise import DEFAULT_MIN_PREMISE_FREQ
from pipeline.postflop.question_extractor import MAX_FREQUENCY, MIN_FREQUENCY
from pipeline.postflop.solve import PostflopSolve, validate_solve
from pipeline.postflop.validators import run_postflop_soft_validators

logger = logging.getLogger(__name__)

ProgressCallback = Any


# Counter keys a leg reports back so the batch can aggregate them.
_LEG_COUNTER_KEYS = (
    "soft_flagged", "claim_flagged", "revise_flagged", "revise_fixed",
    "revise_discarded", "revise_unchanged",
    "preflop_leg_pack_used", "preflop_leg_entry_fallback",
)


def _zero_counters() -> dict[str, int]:
    return dict.fromkeys(_LEG_COUNTER_KEYS, 0)


# --- per-leg row builders ---------------------------------------------------
def _postflop_leg_row(
    spot, solve, *, number, hand_id, sequence_index, sequence_total,
    use_placeholder, client, model, temperature, max_tokens, answer_style,
    display_in_bb, equity_runouts, trap_difficulty, system_prompt, usage_cb,
    run_claim_checker=False, claim_checker_prompt=None, revise_pass=False,
    final_audit=False, facts=None,
) -> tuple[dict[str, str] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, int]]:
    """Build one postflop-leg row. Returns (row, meta_record, failure, counters).

    Runs the SAME Layer-6 generation + (opt-in) Layer-7 audit/revise + soft
    validators as a standalone spot, via the shared
    :func:`pipeline.postflop.layer7.run_layer7_audit`, so a play-through leg gets
    identical QA. ``counters`` is the per-leg delta the batch aggregates."""
    counters = _zero_counters()
    if facts is None:  # the pre-pass may have already paid the equity sim
        facts = extract_facts(spot, solve, equity_runouts=equity_runouts)
    options, correct = build_options(spot, style=answer_style)
    difficulty = compute_difficulty(facts, apply_trap_bump=trap_difficulty)
    try:
        if use_placeholder:
            explanation = placeholder_explanation(facts, options, correct)
        else:
            explanation = generate_postflop_explanation(
                facts, options, correct, solve, client=client, model=model,
                temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, usage_callback=usage_cb,
            )
    except ExplanationValidationError as exc:
        return None, None, {
            "node_id": spot.node.node_id, "hero_combo": spot.hero_combo,
            "hand_id": hand_id, "error_message": str(exc),
            "attempt_text": exc.last_attempt_text,
        }, counters

    solver_data_block = build_solver_data_block(facts)
    question_text = format_question(facts.spot, solve, display_in_bb=display_in_bb)

    # --- Layer-7 (opt-in; real runs only) -- identical to the standalone path ---
    claim_check_json = ""
    revise_record: dict[str, Any] | None = None
    remaining_issues: list[str] = []
    claim_issues: list[str] = []
    if not use_placeholder and (run_claim_checker or revise_pass):
        l7 = run_layer7_audit(
            explanation, facts,
            solver_data_block=solver_data_block, question_text=question_text,
            node_id=spot.node.node_id, client=client, model=model,
            temperature=temperature, max_tokens=max_tokens,
            system_prompt=system_prompt,
            checker_prompt=claim_checker_prompt or POSTFLOP_CHECKER_SYSTEM_PROMPT,
            run_claim_checker=run_claim_checker, revise_pass=revise_pass,
            final_audit=final_audit, usage_callback=usage_cb,
        )
        explanation = l7.explanation
        claim_check_json = l7.claim_check_json
        claim_issues = l7.claim_issues
        revise_record = l7.revise_record
        remaining_issues = l7.remaining_issues
        counters["claim_flagged"] += l7.claim_flagged
        counters["revise_flagged"] += l7.revise_flagged
        counters["revise_fixed"] += l7.revise_fixed
        counters["revise_discarded"] += l7.revise_discarded
        counters["revise_unchanged"] += l7.revise_unchanged

    soft_warnings = (
        [] if use_placeholder else run_postflop_soft_validators(explanation, facts)
    )
    if soft_warnings:
        counters["soft_flagged"] += 1
    status = "flagged" if (soft_warnings or remaining_issues) else "draft"

    row = build_postflop_row(
        facts, explanation, solve, difficulty, number,
        validation_status=status, display_in_bb=display_in_bb,
        hand_id=hand_id, sequence_index=sequence_index, sequence_total=sequence_total,
    )
    row["claim_check"] = claim_check_json
    record = {
        "node_id": spot.node.node_id, "hero_combo": spot.hero_combo,
        "street": facts.street, "hand_id": hand_id,
        "sequence_index": sequence_index, "sequence_total": sequence_total,
        "correct_answer": correct, "options": options,
        "archetype": facts.archetype, "difficulty": difficulty.score,
        "solver_data": solver_data_block,
        # Range-visual metadata, identical to generate_postflop_batch so the
        # grouped (play-through) Review shows the SAME panel as a standalone
        # batch: the current-street ranges (right grid), both players'
        # per-action strategy (action-coloured), and the street-before ranges
        # (left grid). Without these the panel degraded to a misleading green
        # "holdings" grid + a "Preflop -- n/a" left grid.
        "street_ranges": _node_range_snapshots(spot.node),
        "street_strategy": _street_strategies(spot.node, solve),
        "street_actor": spot.node.actor,
    }
    _prior_node = _prior_street_node(spot.node, solve)
    if _prior_node is not None:
        record["prior_street_ranges"] = _node_range_snapshots(_prior_node)
        record["prior_street_label"] = _prior_node.street
    if revise_record is not None:
        record["revise"] = revise_record
    if claim_issues:
        record["claim_check_issues"] = claim_issues
    if soft_warnings:
        record["validator_warnings"] = soft_warnings
    return row, record, None, counters


def _preflop_leg_row_entry(
    entry_facts, *, number, hand_id, sequence_index, sequence_total,
    use_placeholder, client, model, temperature, max_tokens, answer_style,
    display_in_bb, preflop_system_prompt, usage_cb, as_played=False,
) -> tuple[dict[str, str] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, int]]:
    """Build one preflop-entry-leg row. Returns (row, meta_record, failure, counters).

    ``as_played`` marks a play-through leg (the entry action is what this hand
    did to reach the flop, so it is the correct answer, not Fold). The
    preflop-entry leg does NOT run the postflop Layer-7 audit (the checker prompt
    is postflop-specific), so its counters are always zero."""
    options, correct = build_preflop_entry_options(
        entry_facts, style=answer_style, display_in_bb=display_in_bb,
        as_played=as_played,
    )
    try:
        if use_placeholder:
            explanation = placeholder_preflop_entry_explanation(
                entry_facts, options, correct, display_in_bb=display_in_bb,
                as_played=as_played,
            )
        else:
            explanation = generate_preflop_entry_explanation(
                entry_facts, options, correct, client=client, model=model,
                temperature=temperature, max_tokens=max_tokens,
                system_prompt=preflop_system_prompt, display_in_bb=display_in_bb,
                usage_callback=usage_cb, as_played=as_played,
            )
    except ExplanationValidationError as exc:
        return None, None, {
            "node_id": "", "hero_combo": entry_facts.hero_combo,
            "hand_id": hand_id, "error_message": str(exc),
            "attempt_text": exc.last_attempt_text,
        }, _zero_counters()
    row = build_preflop_entry_row(
        entry_facts, explanation, number, validation_status="draft",
        display_in_bb=display_in_bb, hand_id=hand_id,
        sequence_index=sequence_index, sequence_total=sequence_total,
        as_played=as_played,
    )
    record = {
        "node_id": "", "hero_combo": entry_facts.hero_combo, "street": "preflop",
        # hero_position + as_played let the re-verifier rebuild this leg exactly.
        "hero_position": entry_facts.hero_position, "as_played": as_played,
        "hand_id": hand_id, "sequence_index": sequence_index,
        "sequence_total": sequence_total, "correct_answer": correct,
        "options": options, "archetype": entry_facts.archetype,
        "difficulty": entry_facts.difficulty,
    }
    return row, record, None, _zero_counters()


def _write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(meta, indent=2, default=str))


# --- full-hand (play-through) batch -----------------------------------------
def generate_full_hand_batch(
    *,
    solve: PostflopSolve,
    output_path: Path | str,
    total_hands: int,
    client: object | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    dry_run: bool = False,
    answer_style: str = "auto",
    display_in_bb: bool = True,
    heroes: tuple[str, ...] = (),
    include_villain: bool = False,
    include_preflop: bool = True,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    min_ev_gap_bb: float | None = None,
    quality_gate: bool = True,
    min_premise_freq: float | None = DEFAULT_MIN_PREMISE_FREQ,
    equity_runouts: int = DEFAULT_EQUITY_RUNOUTS,
    system_prompt: str | None = None,
    preflop_system_prompt: str | None = None,
    trap_difficulty: bool = False,
    razor_difficulty: bool = False,
    preflop_leg_pack_root: Path | str | None = None,
    min_hand_difficulty: int | None = None,
    max_hand_difficulty: int | None = None,
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    revise_pass: bool = False,
    final_audit: bool = False,
    progress_callback: ProgressCallback = None,
    write_meta: bool = True,
    provenance: dict[str, Any] | None = None,
) -> PostflopBatchResult:
    """Generate up to ``total_hands`` play-through hands (each = several linked
    questions sharing a ``hand_id``).

    Worthy spots seed the hands (so each has an interesting decision); each hand
    is the hero's full decision line plus the preflop entry. ``include_villain``
    additionally emits the villain's line on the same runout as its own hand.
    The opt-in Layer-7 audit (``run_claim_checker`` / ``revise_pass`` /
    ``final_audit``) runs on the POSTFLOP legs via the shared
    :func:`pipeline.postflop.layer7.run_layer7_audit` (the preflop-entry leg is
    skipped -- the checker prompt is postflop-specific). Returns a
    :class:`PostflopBatchResult` whose ``questions_written`` is the total ROW
    count across all hands.
    """
    problems = validate_solve(solve)
    if problems:
        raise ValueError(f"solve {solve.solve_id} is malformed: {problems}")

    use_placeholder = dry_run or client is None
    worthy, low_quality, premise_skipped = _collect_worthy(
        solve, min_frequency=min_frequency, max_frequency=max_frequency,
        min_ev_gap_bb=min_ev_gap_bb, quality_gate=quality_gate,
        min_premise_freq=min_premise_freq,
    )
    hands = assemble_hands(
        solve, seeds=worthy, heroes=tuple(heroes) or (), max_hands=total_hands,
        include_preflop=include_preflop, include_villain=include_villain,
    )

    # --- pack-backed preflop legs (July 2026) -----------------------------
    # When a preflop range pack provably matches this solve's preflop line
    # (same table size / stack / open size), the preflop leg is built with
    # the FULL preflop pipeline (EVs, ranges, stat_notes, 4-axis difficulty)
    # instead of the entry-derived approximation. Per-hand coherence gate
    # inside the builder; entry-derived leg is the fallback.
    pack_source = None
    if include_preflop and preflop_leg_pack_root is not None:
        pack_source = find_pack_leg_source(solve, preflop_leg_pack_root)

    # --- hand-difficulty pre-pass (July 2026) ------------------------------
    # hand_difficulty = MAX over the legs' Difficulty Ratings: the hand
    # demands what its hardest decision demands (a 2400 river bluff-catch
    # behind three easy calls is a hard HAND; a mean would wash it out to
    # "easy"). Computed BEFORE any LLM call so the optional band filter
    # costs no tokens; the facts cache hands each postflop leg's equity sim
    # to the generation loop so nothing is computed twice.
    facts_cache: dict[tuple[str, str], Any] = {}
    pack_facts_cache: dict[tuple[str, str], Any] = {}
    hand_difficulties: dict[str, int] = {}
    for hand in hands:
        leg_scores: list[int] = []
        for leg in hand.legs:
            if leg.kind == "preflop_entry":
                score = None
                if pack_source is not None:
                    built = _build_pack_facts(
                        pack_source, leg.entry_facts.hero_position,
                        leg.entry_facts.hero_combo, solve,
                        equity_runouts=equity_runouts,
                    )
                    if built is not None:
                        pf, pd = _facts_difficulty(
                            built, trap_difficulty=trap_difficulty,
                            razor_difficulty=razor_difficulty,
                        )
                        pack_facts_cache[
                            (leg.entry_facts.hero_position,
                             leg.entry_facts.hero_combo)
                        ] = (pf, pd)
                        score = pd.score
                if score is None:
                    score = leg.entry_facts.difficulty
            else:
                key = (leg.spot.node.node_id, leg.spot.hero_combo)
                if key not in facts_cache:
                    facts_cache[key] = extract_facts(
                        leg.spot, solve, equity_runouts=equity_runouts,
                    )
                score = compute_difficulty(
                    facts_cache[key], apply_trap_bump=trap_difficulty,
                ).score
            leg_scores.append(score)
        hand_difficulties[hand.hand_id] = max(leg_scores) if leg_scores else 0

    hands_difficulty_filtered = 0
    if min_hand_difficulty is not None or max_hand_difficulty is not None:
        lo = min_hand_difficulty if min_hand_difficulty is not None else 0
        hi = max_hand_difficulty if max_hand_difficulty is not None else 10_000
        kept = [h for h in hands if lo <= hand_difficulties[h.hand_id] <= hi]
        hands_difficulty_filtered = len(hands) - len(kept)
        hands = kept

    in_tokens = out_tokens = 0

    def _usage(usage: object) -> None:
        nonlocal in_tokens, out_tokens
        in_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    rows: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    hand_index: list[dict[str, Any]] = []
    agg = _zero_counters()  # batch-wide Layer-7 / soft-validator tallies

    total_legs = sum(h.total for h in hands)
    done = 0
    for hand in hands:
        leg_records: list[int] = []
        for i, leg in enumerate(hand.legs, start=1):
            done += 1
            if progress_callback is not None:
                progress_callback(
                    f"Hand {hand.hand_id[:22]} leg {i}/{hand.total}", done, total_legs,
                )
            number = len(rows) + 1
            if leg.kind == "preflop_entry":
                row = record = failure = None
                counters = _zero_counters()
                prebuilt = (
                    pack_facts_cache.get(
                        (leg.entry_facts.hero_position, leg.entry_facts.hero_combo)
                    )
                    if pack_source is not None else None
                )
                if prebuilt is not None:
                    row, record, failure = build_pack_preflop_leg_row(
                        pack_source, leg.entry_facts.hero_position,
                        leg.entry_facts.hero_combo, solve,
                        number=number, hand_id=hand.hand_id,
                        sequence_index=i, sequence_total=hand.total,
                        use_placeholder=use_placeholder, client=client,
                        model=model, temperature=temperature,
                        max_tokens=max_tokens, answer_style=answer_style,
                        display_in_bb=display_in_bb,
                        equity_runouts=equity_runouts,
                        trap_difficulty=trap_difficulty,
                        razor_difficulty=razor_difficulty,
                        usage_cb=_usage, prebuilt=prebuilt,
                    )
                if row is not None or failure is not None:
                    agg["preflop_leg_pack_used"] += 1 if row is not None else 0
                else:
                    agg["preflop_leg_entry_fallback"] += 1
                    row, record, failure, counters = _preflop_leg_row_entry(
                        leg.entry_facts, number=number, hand_id=hand.hand_id,
                        sequence_index=i, sequence_total=hand.total,
                        use_placeholder=use_placeholder, client=client, model=model,
                        temperature=temperature, max_tokens=max_tokens,
                        answer_style=answer_style, display_in_bb=display_in_bb,
                        preflop_system_prompt=preflop_system_prompt, usage_cb=_usage,
                        as_played=True,  # the play-through entry = what this hand did
                    )
            else:
                row, record, failure, counters = _postflop_leg_row(
                    leg.spot, solve,
                    facts=facts_cache.get(
                        (leg.spot.node.node_id, leg.spot.hero_combo)
                    ),
                    number=number, hand_id=hand.hand_id,
                    sequence_index=i, sequence_total=hand.total,
                    use_placeholder=use_placeholder, client=client, model=model,
                    temperature=temperature, max_tokens=max_tokens,
                    answer_style=answer_style, display_in_bb=display_in_bb,
                    equity_runouts=equity_runouts, trap_difficulty=trap_difficulty,
                    system_prompt=system_prompt, usage_cb=_usage,
                    run_claim_checker=run_claim_checker,
                    claim_checker_prompt=claim_checker_prompt,
                    revise_pass=revise_pass, final_audit=final_audit,
                )
            for k in _LEG_COUNTER_KEYS:
                agg[k] += counters[k]
            if failure is not None:
                failures.append(failure)
                continue
            rows.append(row)  # type: ignore[arg-type]
            records.append(record)  # type: ignore[arg-type]
            leg_records.append(number)
        if leg_records:
            hd = hand_difficulties.get(hand.hand_id, 0)
            for n in leg_records:
                rows[n - 1]["hand_difficulty"] = str(hd)
            hand_index.append({
                "hand_id": hand.hand_id, "hero": hand.hero,
                "hero_combo": hand.hero_combo, "frame": hand.frame,
                "row_numbers": leg_records, "legs": hand.total,
                "hand_difficulty": hd,
            })

    output_path = Path(output_path)
    write_postflop_csv(output_path, rows)

    meta_path: Path | None = None
    if write_meta:
        meta_path = output_path.with_suffix(".meta.json")
        _write_meta(meta_path, {
            "solve_id": solve.solve_id,
            "source_reference": solve.source_reference,
            "mode": "full_hand",
            "model": model if not use_placeholder else "(dry-run placeholder)",
            "dry_run": use_placeholder,
            "provenance": provenance or {},
            "run_settings": {
                "total_hands": total_hands, "answer_style": answer_style,
                "display_in_bb": display_in_bb, "heroes": list(heroes),
                "include_villain": include_villain, "include_preflop": include_preflop,
                "min_frequency": min_frequency, "max_frequency": max_frequency,
                "min_ev_gap_bb": min_ev_gap_bb, "quality_gate": quality_gate,
                "min_premise_freq": min_premise_freq, "equity_runouts": equity_runouts,
                "trap_difficulty": trap_difficulty,
                "razor_difficulty": razor_difficulty,
                "preflop_leg_pack": pack_source.pack_id if pack_source else None,
                "min_hand_difficulty": min_hand_difficulty,
                "max_hand_difficulty": max_hand_difficulty,
                "run_claim_checker": run_claim_checker,
                "revise_pass": revise_pass,
                "final_audit": final_audit and revise_pass,
            },
            "counters": {
                "worthy_spots_available": len(worthy),
                "low_quality_nodes_skipped": low_quality,
                "premise_filtered_nodes": premise_skipped,
                "hands_assembled": len(hands) + hands_difficulty_filtered,
                "hands_difficulty_filtered": hands_difficulty_filtered,
                "preflop_leg_pack_used": agg["preflop_leg_pack_used"],
                "preflop_leg_entry_fallback": agg["preflop_leg_entry_fallback"],
                "hands_written": len(hand_index),
                "questions_written": len(rows),
                # Layer-7 + soft-validator tallies (postflop legs only; 0 unless
                # the opt-in passes ran). Keys match the standalone batch.
                "soft_flagged_rows": agg["soft_flagged"],
                "claim_flagged_rows": agg["claim_flagged"],
                "revise_flagged": agg["revise_flagged"],
                "revise_fixed": agg["revise_fixed"],
                "revise_discarded": agg["revise_discarded"],
                "revise_unchanged": agg["revise_unchanged"],
            },
            "hands": hand_index,
            "questions": records,
            "failures": failures,
        })

    return PostflopBatchResult(
        output_path=output_path,
        questions_written=len(rows),
        questions_attempted=total_legs,
        worthy_spots_available=len(worthy),
        requested_questions=total_hands,
        dry_run=use_placeholder,
        model_used=model if not use_placeholder else "(dry-run placeholder)",
        failures=failures,
        soft_flagged_rows=agg["soft_flagged"],
        meta_path=meta_path,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
    )


# --- standalone preflop-entry batch -----------------------------------------
def generate_preflop_entry_batch(
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
    heroes: tuple[str, ...] = (),
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    preflop_system_prompt: str | None = None,
    progress_callback: ProgressCallback = None,
    write_meta: bool = True,
    provenance: dict[str, Any] | None = None,
) -> PostflopBatchResult:
    """Generate up to ``total_questions`` STANDALONE preflop-entry questions.

    Sourced from the solve's flop-entry range frequencies (no play-through
    linkage; ``hand_id`` stays blank). Worthiness = a genuinely-mixed entry
    frequency (the same window idea as elsewhere). NOT a substitute for real
    standalone preflop questions (use the preflop range-pack pipeline) -- this
    is the postflop solve's own entry decision.
    """
    use_placeholder = dry_run or client is None
    all_facts = enumerate_preflop_entry_facts(solve, heroes=tuple(heroes) or ())
    in_window = [
        f for f in all_facts
        if preflop_entry_is_worthy(f, min_frequency=min_frequency, max_frequency=max_frequency)
    ]
    # #6B: drop premium defender hands whose call/fold framing is unreliable (they
    # 3-bet, not fold; the postflop solve has no 3-bet data to show it). Play-through
    # legs keep them via as_played; this guard is standalone-only.
    worthy = [f for f in in_window if standalone_entry_is_reliable(f)]
    premium_excluded = len(in_window) - len(worthy)

    in_tokens = out_tokens = 0

    def _usage(usage: object) -> None:
        nonlocal in_tokens, out_tokens
        in_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    rows: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for facts in worthy:
        if len(rows) >= total_questions:
            break
        number = len(rows) + 1
        if progress_callback is not None:
            progress_callback(
                f"Preflop {facts.hero_position} {facts.hand_class}", len(rows), total_questions,
            )
        row, record, failure, _counters = _preflop_leg_row_entry(
            facts, number=number, hand_id="", sequence_index="", sequence_total="",
            use_placeholder=use_placeholder, client=client, model=model,
            temperature=temperature, max_tokens=max_tokens, answer_style=answer_style,
            display_in_bb=display_in_bb, preflop_system_prompt=preflop_system_prompt,
            usage_cb=_usage,
        )
        if failure is not None:
            failures.append(failure)
            continue
        rows.append(row)  # type: ignore[arg-type]
        records.append(record)  # type: ignore[arg-type]

    output_path = Path(output_path)
    write_postflop_csv(output_path, rows)

    meta_path: Path | None = None
    if write_meta:
        meta_path = output_path.with_suffix(".meta.json")
        _write_meta(meta_path, {
            "solve_id": solve.solve_id,
            "source_reference": solve.source_reference,
            "mode": "preflop_entry",
            "model": model if not use_placeholder else "(dry-run placeholder)",
            "dry_run": use_placeholder,
            "provenance": provenance or {},
            "run_settings": {
                "total_questions": total_questions, "answer_style": answer_style,
                "display_in_bb": display_in_bb, "heroes": list(heroes),
                "min_frequency": min_frequency, "max_frequency": max_frequency,
            },
            "counters": {
                "entry_spots_available": len(all_facts),
                "worthy_entry_spots": len(worthy),
                # #6B: premium defender hands dropped (call/fold framing unreliable).
                "premium_3bet_excluded": premium_excluded,
                "questions_written": len(rows),
            },
            "questions": records,
            "failures": failures,
        })

    return PostflopBatchResult(
        output_path=output_path,
        questions_written=len(rows),
        questions_attempted=len(worthy),
        worthy_spots_available=len(worthy),
        requested_questions=total_questions,
        dry_run=use_placeholder,
        model_used=model if not use_placeholder else "(dry-run placeholder)",
        failures=failures,
        meta_path=meta_path,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
    )


__all__ = ["generate_full_hand_batch", "generate_preflop_entry_batch"]
