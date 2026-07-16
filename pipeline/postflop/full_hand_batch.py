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
shared :func:`pipeline.postflop.layer7.run_layer7_audit` AND on the PACK-built
preflop legs via the PREFLOP pipeline's checker/reviser (July 2026) -- so every
leg of a hand gets audit-depth QA. Only the SRP entry-derived fallback leg is
skipped (its binary framing carries no SOLVER DATA block for a checker to
audit). A deterministic cross-check (:mod:`pipeline.preflop.batch_cross_check`,
reached through the sanctioned pack-leg seam) re-reads every written row at the
end of the batch. Re-verify a finished batch with
``scripts/audit_full_hand_batch.py``.
"""

from __future__ import annotations

import hashlib
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
from pipeline.postflop.difficulty import (
    aggregate_hand_difficulty,
    compute_difficulty,
    easy_freq_axis,
)
from pipeline.postflop.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    build_solver_data_block,
    generate_postflop_explanation,
    placeholder_explanation,
)
from pipeline.postflop.facts import (
    DEFAULT_EQUITY_RUNOUTS,
    extract_facts,
    preflop_raise_count,
)
from pipeline.postflop.format_writer import (
    FULL_HAND_CSV_COLUMNS,
    build_postflop_row,
    write_postflop_csv,
)
from pipeline.postflop.layer7 import run_layer7_audit
from pipeline.postflop.options import build_options
from pipeline.postflop.play_through import (
    LENGTH_PROFILES,
    assemble_hands,
    balanced_hand_mix,
    balanced_length_mix,
)
from pipeline.postflop.play_through import _solve_tag as _pt_solve_tag
from pipeline.postflop.preflop_leg_pack import (
    _build_pack_facts,
    _facts_difficulty,
    build_pack_preflop_leg_row,
    find_pack_leg_source,
    run_full_hand_cross_check,
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
from pipeline.postflop.showdown import attach_showdown_resolution
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
    # Multi-raise (3-bet pot) preflop legs that could not be built: no
    # matching pack, or the pack's strategy for the hand contradicts the
    # as-played line. The leg is DROPPED (never faked from entry weights);
    # the hand's postflop legs still narrate the full preflop line.
    "preflop_line_legs_dropped",
)


def _require_single_raised_pot(solve: PostflopSolve, mode_label: str) -> None:
    """SRP-ONLY GATE for STANDALONE preflop-entry mode (July 2026).

    The entry framing is a continue-or-fold binary read off the solve's
    entry weights. In a 3-bet pot the weights are 3-bet / call-vs-3bet
    frequencies whose complement is a call+fold MIX, so no honest binary
    question exists -- fail fast with a plain message. (Full-hand mode
    supports 3-bet pots via pack-backed line legs, which ask each real
    decision from a matching preflop range pack; standalone preflop
    questions belong to the preflop range-pack pipeline.) The admin UI
    mirrors this gate upfront; THIS check is the source of truth.
    """
    n_raises = preflop_raise_count(solve)
    if n_raises != 1:
        line = ", ".join(
            f"{st.position} {st.verb}" + (f" {st.to_bb:g}bb" if st.to_bb else "")
            for st in solve.preflop_summary
        )
        raise ValueError(
            f"{mode_label} supports single-raised-pot solves only. This "
            f"solve's preflop line is '{line}' ({n_raises} raises). Use "
            "full-hand mode (pack-backed preflop legs) or standalone "
            "postflop questions for it instead."
        )


def _zero_counters() -> dict[str, int]:
    return dict.fromkeys(_LEG_COUNTER_KEYS, 0)


def _pack_leg_counter_delta(record: dict[str, Any]) -> dict[str, int]:
    # The pack-leg builder reports its Layer-7 lifecycle inside the meta
    # record (mirroring the preflop batch); this maps it onto the postflop
    # batch counter keys so claim_flagged_rows / revise_* count every leg.
    d = _zero_counters()
    if record.get("validator_warnings"):
        d["soft_flagged"] = 1
    if record.get("claim_check_issues"):
        d["claim_flagged"] = 1
    status = (record.get("revise") or {}).get("status")
    if status in ("fixed", "discarded", "unchanged"):
        d["revise_flagged"] = 1
        d[f"revise_{status}"] = 1
    return d


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
    # Artifact-strip transparency (July 2026), same as the standalone batch.
    if spot.stripped_artifact_freq > 0:
        record["artifact_stripped"] = {
            "labels": sorted(spot.artifact_labels),
            "freq": round(spot.stripped_artifact_freq, 4),
        }
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


# Band-scan ordering kicks in only when the requested band floor is above the
# scale's hard floor (i.e. the user actually wants harder-than-baseline hands).
_HARD_FLOOR_FOR_SORT = 500


def _min_postflop_freq_ease(hand) -> tuple[float, str]:
    """Sort key for the band scan: the hand's LOWEST postflop-leg frequency
    ease (0 = coin-flip = the only legs that can rate very hard; 1 = pure).
    Hands with no postflop leg sort last; hand_id breaks ties for determinism.
    """
    eases = [
        easy_freq_axis(leg.spot.dominant_frequency)
        for leg in hand.legs
        if leg.kind == "postflop" and leg.spot is not None
    ]
    return (min(eases) if eases else 1.0, hand.hand_id)


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
    preflop_pack_system_prompt: str | None = None,
    trap_difficulty: bool = False,
    razor_difficulty: bool = False,
    preflop_leg_pack_root: Path | str | None = None,
    min_hand_difficulty: int | None = None,
    max_hand_difficulty: int | None = None,
    variety_seed: int | None = None,
    diversify_hands: bool = False,
    balanced_lengths: bool = False,
    length_profile: str = "river_heavy",
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    revise_pass: bool = False,
    final_audit: bool = False,
    progress_callback: ProgressCallback = None,
    write_meta: bool = True,
    provenance: dict[str, Any] | None = None,
    prompt_names: dict[str, str] | None = None,
) -> PostflopBatchResult:
    """Generate up to ``total_hands`` play-through hands (each = several linked
    questions sharing a ``hand_id``).

    Three prompts write the legs' explanations: ``system_prompt`` (postflop
    legs), ``preflop_pack_system_prompt`` (the pack-backed preflop leg,
    written by the PREFLOP pipeline's generator; None = its active prompt),
    and ``preflop_system_prompt`` (the entry-derived preflop fallback leg).
    ``prompt_names`` is an optional display-name map (e.g. ``{"postflop":
    ..., "preflop_pack": ...}``) recorded verbatim in run_settings so a
    batch always says WHICH named prompts wrote it.

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
    worthy, low_quality, premise_skipped, artifact_material_out = _collect_worthy(
        solve, min_frequency=min_frequency, max_frequency=max_frequency,
        min_ev_gap_bb=min_ev_gap_bb, quality_gate=quality_gate,
        min_premise_freq=min_premise_freq,
    )
    # BAND-SCAN INVARIANT (July 2026): ``total_hands`` caps the KEPT hands,
    # never the scanned ones. The old flow assembled exactly total_hands and
    # THEN band-filtered, so "1 Hard hand" assembled one arbitrary hand and
    # usually deleted it -- a silent empty batch. With a band set, assemble a
    # larger candidate pool and scan until enough in-band hands are found.
    band_set = min_hand_difficulty is not None or max_hand_difficulty is not None
    if balanced_lengths and not band_set:
        # Balanced lengths assembles the WHOLE pool: fold-truncated flop
        # enders come from combos that rarely reach deep streets, so they
        # live at the very END of the deepest-first seed order -- any cap
        # starves that bucket. Assembly is cheap (pure range math, no
        # equity sims); facts are only computed for the quota-mixed keep.
        scan_cap = None
    elif band_set or balanced_lengths:
        scan_cap = max(60, total_hands * 20)
    elif diversify_hands:
        # Balanced mix needs an oversized pool to round-robin across
        # (hero, depth, strength) buckets. Cheap: no facts are computed
        # until AFTER the mix trims back to total_hands.
        # 20x (like the band scan) on purpose: the deepest-first ordering
        # front-loads one actor's longest lines, so a shallow pool can
        # miss the other seat's hands entirely.
        scan_cap = max(60, total_hands * 20)
    else:
        scan_cap = total_hands
    hands = assemble_hands(
        solve, seeds=worthy, heroes=tuple(heroes) or (), max_hands=scan_cap,
        include_preflop=include_preflop, include_villain=include_villain,
        variety_seed=variety_seed,
    )

    # --- pack-backed preflop legs (July 2026) -----------------------------
    # When a preflop range pack provably matches this solve's preflop line
    # (same table size / stack / open size), the preflop leg is built with
    # the FULL preflop pipeline (EVs, ranges, stat_notes, 4-axis difficulty)
    # instead of the entry-derived approximation. Per-hand coherence gate
    # inside the builder; entry-derived leg is the fallback. (Resolved
    # BEFORE the length mix: preflop-ENDING hands are built from the pack.)
    pack_source = None
    if include_preflop and preflop_leg_pack_root is not None:
        pack_source = find_pack_leg_source(solve, preflop_leg_pack_root)

    # --- preflop-ENDING hands (balanced lengths, July 2026) ---------------
    # A quarter of a balanced batch ends preflop: hero's correct action at
    # their deepest preflop decision is FOLD (worthy window), the hand is a
    # single-leg (SRP) or open-then-fold story, and -- when folding to a
    # raise -- the raiser reveals a clearly-stronger hand. Built from the
    # matched pack only (entry weights cannot express a fold decision); with
    # no pack the quota back-fills from the other streets and the counter
    # says so.
    preflop_ender_hands: list[Any] = []
    if balanced_lengths and pack_source is not None and include_preflop:
        import zlib as _zlib  # noqa: PLC0415

        from pipeline.postflop.play_through import (  # noqa: PLC0415
            HandLeg,
            PlayThroughHand,
        )
        from pipeline.postflop.preflop_leg_pack import (  # noqa: PLC0415
            combo_for_hand_class,
            fold_ender_hand_classes,
            raise_ender_hand_classes,
        )
        import random as _random  # noqa: PLC0415

        ender_quota = total_hands // 4 + 2
        seats = tuple(heroes) or tuple(solve.positions)
        for hero in seats:
            try:
                steps = pack_source.steps_for(hero)
            except (KeyError, ValueError):
                continue
            if not steps:
                continue
            step = steps[-1]  # the hero's DEEPEST preflop decision
            fold_classes = fold_ender_hand_classes(
                pack_source, hero, step_index=step.step_index,
                min_frequency=min_frequency, max_frequency=max_frequency,
            )
            raise_classes = raise_ender_hand_classes(
                pack_source, hero, step_index=step.step_index,
                min_frequency=min_frequency, max_frequency=max_frequency,
            )
            order_rng = _random.Random(
                variety_seed if variety_seed is not None else 0
            )
            order_rng.shuffle(fold_classes)
            order_rng.shuffle(raise_classes)
            # Interleave fold- and raise-enders so the preflop quarter
            # itself is unpredictable: a preflop question's answer can be
            # fold, call (the hand continues), or the re-raise (the hand
            # ends with the pot pushed to hero).
            classes: list[tuple[str, bool]] = []
            fi = ri = 0
            while len(classes) < ender_quota and (
                fi < len(fold_classes) or ri < len(raise_classes)
            ):
                if fi < len(fold_classes):
                    classes.append((fold_classes[fi], False)); fi += 1
                if len(classes) < ender_quota and ri < len(raise_classes):
                    classes.append((raise_classes[ri], True)); ri += 1
            for hand_class, is_raise_ender in classes:
                combo = combo_for_hand_class(
                    hand_class,
                    _random.Random(_zlib.crc32(
                        f"{solve.source_reference}|ender|{hero}|{hand_class}".encode()
                    )),
                )
                digest = hashlib.sha1(
                    f"{solve.source_reference}|preflop_ender|{hero}|{combo}"
                    f"|{step.step_index}".encode()
                ).hexdigest()[:8]
                preflop_ender_hands.append(PlayThroughHand(
                    hand_id=f"{_pt_solve_tag(solve)}_{combo}_pf_{digest}",
                    hero=hero, hero_combo=combo, frame="hero",
                    legs=(HandLeg(
                        kind="preflop_line", street="preflop",
                        node_id=f"preflop_step:{step.step_index}",
                        step_index=step.step_index,
                        terminal_fold=not is_raise_ender,
                        terminal_raise=is_raise_ender,
                    ),),
                ))
        hands = hands + preflop_ender_hands

    length_reserve: dict[str, list] = {}
    diversify_backfill: list = []
    if balanced_lengths and not band_set:
        # Ending-street quotas per the length profile (river-leaning is the
        # production default). The UNSELECTED pool is kept as a per-street
        # RESERVE: when a hand dies mid-generation (whole-hand atomicity),
        # a same-ending replacement takes its slot so a small batch can't
        # silently lose its river hand. With a difficulty band the band
        # scan governs instead.
        pool = hands
        hands = balanced_length_mix(
            pool, total_hands,
            weights=LENGTH_PROFILES.get(
                length_profile, LENGTH_PROFILES["equal"]
            ),
        )
        _selected = {id(h) for h in hands}
        for h in pool:
            if id(h) not in _selected:
                length_reserve.setdefault(h.ending_street, []).append(h)
    elif diversify_hands and not band_set:
        # With a difficulty band the scan-order heuristic governs instead
        # (the band itself is the selector); mixing there would fight it.
        # The unselected remainder is kept (deterministic order) so a hand
        # dropped by the preflop-coherence filter below pulls a replacement
        # instead of shrinking the batch (July 15 2026).
        _mix_pool = hands
        hands = balanced_hand_mix(_mix_pool, total_hands)
        _mix_selected = {id(h) for h in hands}
        diversify_backfill = [
            h for h in _mix_pool if id(h) not in _mix_selected
        ]

    # --- hand-difficulty pre-pass (July 2026) ------------------------------
    # hand_difficulty = MAX over the legs' Difficulty Ratings: the hand
    # demands what its hardest decision demands (a 2400 river bluff-catch
    # behind three easy calls is a hard HAND; a mean would wash it out to
    # "easy"). Computed BEFORE any LLM call so the optional band filter
    # costs no tokens; the facts cache hands each postflop leg's equity sim
    # to the generation loop so nothing is computed twice.
    facts_cache: dict[tuple[str, str], Any] = {}
    pack_facts_cache: dict[tuple[str, str], Any] = {}
    line_facts_cache: dict[tuple[int, str], Any] = {}
    hand_difficulties: dict[str, int] = {}

    # --- multi-raise preflop line legs (3-bet pots, July 2026) -------------
    # A preflop_line leg is built EXCLUSIVELY from the matched pack: the
    # solve's entry weights cannot express a raise-or-call-or-fold decision,
    # so there is no honest entry fallback. Validate every such leg UP FRONT
    # (build + cache its pack facts). NO FLOP-START HANDS (July 15 2026, the
    # user's call, reversing the July 13 narrate-through rule): a refused
    # preflop leg now drops the WHOLE hand -- a play-through must never
    # start mid-hand, and a hand whose preflop premise the chart refuses
    # (K4o open at 200bb: the pack plays Fold 64%) is built on a decision
    # we would not teach. Same rule the postflop as-played coherence gate
    # already applies to mid-hand legs. Backfill below keeps the batch full.
    preflop_line_legs_dropped = 0
    preflop_entry_legs_dropped = 0
    hands_dropped_preflop_incoherent = 0

    def _filter_preflop_legs(hand):
        """Validate the hand's preflop legs (pack coherence, the pack-first
        rule, the no-pack dominance gate). Returns the hand unchanged when
        every preflop leg is buildable, or ``None`` when ANY is refused --
        the whole hand is then dropped (no flop-start play-throughs).
        Counters via nonlocal. Used on the selected batch AND on every
        backfill/top-up replacement, so a replacement can never skip the
        gates. Hands with no preflop legs at all (the --no-preflop-leg
        mode) pass through untouched -- that mode asks for flop-start
        hands explicitly."""
        nonlocal preflop_line_legs_dropped, preflop_entry_legs_dropped
        nonlocal hands_dropped_preflop_incoherent

        if not any(
            leg.kind in ("preflop_line", "preflop_entry") for leg in hand.legs
        ):
            return hand
        for leg in hand.legs:
            if leg.kind == "preflop_entry":
                # PACK-FIRST RULE (July 2026, the user's call): when a
                # matching preflop chart exists, it is the ONLY source
                # of preflop questions -- never the entry fallback, whose
                # as-played framing can contradict the solver ('Mostly
                # Open' correct above 'Fold: 81%'). With NO pack, the
                # entry leg needs a genuinely dominant continue (>= 50%).
                if pack_source is not None:
                    key = (leg.entry_facts.hero_position,
                           leg.entry_facts.hero_combo)
                    if key not in pack_facts_cache:
                        b = _build_pack_facts(
                            pack_source, key[0], key[1], solve,
                            equity_runouts=equity_runouts,
                        )
                        pack_facts_cache[key] = (
                            None if b is None else _facts_difficulty(
                                b, trap_difficulty=trap_difficulty,
                                razor_difficulty=razor_difficulty,
                            )
                        )
                    if pack_facts_cache[key] is None:
                        preflop_entry_legs_dropped += 1
                        hands_dropped_preflop_incoherent += 1
                        return None
                elif leg.entry_facts.continue_freq < 0.5:  # noqa: PLR2004
                    preflop_entry_legs_dropped += 1
                    hands_dropped_preflop_incoherent += 1
                    return None
                continue
            if leg.kind != "preflop_line":
                continue
            built = None
            if pack_source is not None:
                key = (leg.step_index, hand.hero_combo,
                       leg.terminal_fold, leg.terminal_raise)
                if key not in line_facts_cache:
                    b = _build_pack_facts(
                        pack_source, hand.hero, hand.hero_combo, solve,
                        equity_runouts=equity_runouts,
                        step_index=leg.step_index,
                        terminal_fold=leg.terminal_fold,
                        terminal_raise=leg.terminal_raise,
                    )
                    line_facts_cache[key] = (
                        None if b is None else _facts_difficulty(
                            b, trap_difficulty=trap_difficulty,
                            razor_difficulty=razor_difficulty,
                        )
                    )
                built = line_facts_cache[key]
            if built is None:
                preflop_line_legs_dropped += 1
                hands_dropped_preflop_incoherent += 1
                return None
        return hand

    # Filter the selected batch; a dropped hand pulls a replacement so the
    # batch stays full: balanced lengths -> the same-ending reserve (then
    # any street, deepest first); diversify -> the unselected remainder in
    # its deterministic order. Plain/band paths have their own semantics
    # (the band scan filters its whole pool below; plain assembles exactly
    # total_hands, so a drop shrinks the batch -- counted).
    _kept_hands: list[Any] = []
    for _h in hands:
        _fh = _filter_preflop_legs(_h)
        if _fh is not None:
            _kept_hands.append(_fh)
            continue
        replacement = None
        if length_reserve:
            for street in (_h.ending_street, "river", "turn", "flop",
                           "preflop"):
                queue = length_reserve.get(street) or []
                while queue:
                    cand = _filter_preflop_legs(queue.pop(0))
                    if cand is not None:
                        replacement = cand
                        break
                if replacement is not None:
                    break
        elif diversify_backfill:
            while diversify_backfill:
                cand = _filter_preflop_legs(diversify_backfill.pop(0))
                if cand is not None:
                    replacement = cand
                    break
        if replacement is not None:
            _kept_hands.append(replacement)
    hands = _kept_hands

    def _score_hand(hand) -> int:
        """hand_difficulty = peak-anchored blend over the hand's legs
        (aggregate_hand_difficulty: 0.65 x hardest + 0.35 x mean of the
        rest -- see pipeline/postflop/difficulty.py). Facts cached for the
        generation loop, so nothing is computed twice."""
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
            elif leg.kind == "preflop_line":
                # Validated + cached by the up-front filter above; a leg
                # whose facts failed was already dropped from the hand.
                _pf, pd = line_facts_cache[
                    (leg.step_index, hand.hero_combo,
                     leg.terminal_fold, leg.terminal_raise)
                ]
                score = pd.score
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
        return aggregate_hand_difficulty(leg_scores)

    hands_difficulty_filtered = 0
    hands_scanned = len(hands)
    hand_difficulty_observed_max: int | None = None
    band_scan_capped = False
    if band_set:
        lo = min_hand_difficulty if min_hand_difficulty is not None else 0
        hi = max_hand_difficulty if max_hand_difficulty is not None else 10_000
        candidates = list(hands)
        if not trap_difficulty and lo > _HARD_FLOOR_FOR_SORT:
            # Scan order heuristic: a leg's difficulty is bounded by its
            # dominant frequency alone (concept/hand ease have hard floors),
            # so hands whose postflop legs sit closest to a coin-flip are the
            # only ones that can reach a high band. Score those first so the
            # scan finds in-band hands fast. ORDER only -- never excludes a
            # hand. Skipped with trap-aware on (the trap floor can lift a
            # pure leg far past its frequency ceiling).
            candidates.sort(key=_min_postflop_freq_ease)
        kept: list[Any] = []
        scanned = 0
        for hand in candidates:
            if len(kept) >= total_hands:
                break
            scanned += 1
            score = _score_hand(hand)
            hand_difficulties[hand.hand_id] = score
            hand_difficulty_observed_max = (
                score if hand_difficulty_observed_max is None
                else max(hand_difficulty_observed_max, score)
            )
            if lo <= score <= hi:
                kept.append(hand)
        hands_scanned = scanned
        hands_difficulty_filtered = scanned - len(kept)
        band_scan_capped = (
            len(kept) < total_hands and len(candidates) >= scan_cap
        )
        hands = kept
    else:
        for hand in hands:
            score = _score_hand(hand)
            hand_difficulties[hand.hand_id] = score
            hand_difficulty_observed_max = (
                score if hand_difficulty_observed_max is None
                else max(hand_difficulty_observed_max, score)
            )

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
    hands_dropped_failed_leg = 0
    hands_topped_up = 0
    showdown_resolutions = 0
    ending_counts: dict[str, int] = {}
    from collections import deque as _deque  # noqa: PLC0415

    pending = _deque(hands)
    while pending:
        hand = pending.popleft()
        # WHOLE-HAND ATOMICITY (July 2026): a play-through with a missing
        # street is a broken story (the app would show a hand starting at
        # question 2 of 4), so legs are buffered per hand and committed only
        # when EVERY leg generated. Any leg failure discards the whole
        # hand's buffered rows (the failure record is kept for the meta);
        # because commits are atomic, the shipped question numbers stay
        # contiguous with no renumbering.
        hand_rows: list[dict[str, str]] = []
        hand_recs: list[dict[str, Any]] = []
        hand_failed = False
        for i, leg in enumerate(hand.legs, start=1):
            done += 1
            if progress_callback is not None:
                progress_callback(
                    f"Hand {hand.hand_id[:22]} leg {i}/{hand.total}", done, total_legs,
                )
            number = len(rows) + len(hand_rows) + 1
            if leg.kind == "preflop_line":
                # Multi-raise line leg: always pack-built (the up-front
                # filter guarantees the prebuilt facts exist for it).
                counters = _zero_counters()
                row, record, failure = build_pack_preflop_leg_row(
                    pack_source, hand.hero, hand.hero_combo, solve,
                    number=number, hand_id=hand.hand_id,
                    sequence_index=i, sequence_total=hand.total,
                    use_placeholder=use_placeholder, client=client,
                    model=model, temperature=temperature,
                    max_tokens=max_tokens, answer_style=answer_style,
                    display_in_bb=display_in_bb,
                    equity_runouts=equity_runouts,
                    trap_difficulty=trap_difficulty,
                    razor_difficulty=razor_difficulty,
                    system_prompt=preflop_pack_system_prompt,
                    usage_cb=_usage,
                    prebuilt=line_facts_cache[
                        (leg.step_index, hand.hero_combo,
                         leg.terminal_fold, leg.terminal_raise)
                    ],
                    step_index=leg.step_index,
                    terminal_fold=leg.terminal_fold,
                    terminal_raise=leg.terminal_raise,
                    run_claim_checker=run_claim_checker,
                    revise_pass=revise_pass,
                    final_audit=final_audit and revise_pass,
                )
                if row is not None:
                    agg["preflop_leg_pack_used"] += 1
                    counters = _pack_leg_counter_delta(record)
            elif leg.kind == "preflop_entry":
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
                        system_prompt=preflop_pack_system_prompt,
                        usage_cb=_usage, prebuilt=prebuilt,
                        run_claim_checker=run_claim_checker,
                        revise_pass=revise_pass,
                        final_audit=final_audit and revise_pass,
                    )
                if row is not None or failure is not None:
                    agg["preflop_leg_pack_used"] += 1 if row is not None else 0
                    if row is not None:
                        counters = _pack_leg_counter_delta(record)
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
                hand_failed = True
                break  # whole-hand drop: see the atomicity note above
            hand_rows.append(row)  # type: ignore[arg-type]
            hand_recs.append(record)  # type: ignore[arg-type]
        if hand_failed:
            hands_dropped_failed_leg += 1
            # TOP-UP (July 2026): whole-hand atomicity just cost the batch a
            # hand -- pull a SAME-ENDING replacement from the reserve (then
            # any street, deepest first) so a 2-hand batch can't silently
            # lose its river hand. Replacements pass the same preflop-leg
            # gates and get scored like everything else.
            for street in (hand.ending_street, "river", "turn", "flop",
                           "preflop"):
                queue = length_reserve.get(street) or []
                while queue:
                    cand = _filter_preflop_legs(queue.pop(0))
                    # None = the candidate's own preflop leg was refused
                    # (July 15: whole-hand drop) -- keep pulling.
                    if cand is not None and cand.legs:
                        hand_difficulties[cand.hand_id] = _score_hand(cand)
                        pending.append(cand)
                        total_legs += cand.total
                        hands_topped_up += 1
                        break
                else:
                    continue
                break
            continue
        if hand_rows:
            # Showdown resolution (July 2026): the hand's FINAL leg gets the
            # closing sequence (hero's correct action, an invented villain
            # call/fold where needed, reveals, pot push) appended to its
            # animation timeline -- villain's revealed hand VINDICATES the
            # correct answer (see pipeline/postflop/showdown.py). Attached
            # only when the decision honestly ends the hand.
            last_leg = hand.legs[-1]
            resolution = None
            if last_leg.kind == "postflop":
                resolution = attach_showdown_resolution(
                    hand_rows[-1],
                    node=last_leg.spot.node, solve=solve,
                    hero_combo=hand.hero_combo,
                    correct_answer=hand_recs[-1].get("correct_answer", ""),
                    hand_id=hand.hand_id,
                )
            elif (
                last_leg.kind == "preflop_line"
                and (last_leg.terminal_fold or last_leg.terminal_raise)
                and pack_source is not None
            ):
                # Preflop-ending hand: the raiser (if any) reveals a
                # clearly-stronger starting hand -- the preflop analogue of
                # the postflop vindication reveal. First-in folds get none.
                from pipeline.postflop.preflop_leg_pack import (  # noqa: PLC0415
                    build_preflop_fold_resolution,
                    build_preflop_raise_resolution,
                )

                builder = (
                    build_preflop_raise_resolution
                    if last_leg.terminal_raise else build_preflop_fold_resolution
                )
                resolution = builder(
                    pack_source, hand.hero, hand.hero_combo, solve,
                    hand_id=hand.hand_id, step_index=last_leg.step_index,
                )
                if resolution is not None:
                    payload = json.loads(hand_rows[-1]["animation_script"])
                    payload["version"] = 2
                    payload["resolution"] = resolution
                    hand_rows[-1]["animation_script"] = json.dumps(
                        payload, separators=(",", ":")
                    )
            if resolution is not None:
                showdown_resolutions += 1
                hand_recs[-1]["showdown"] = {
                    "vindicates": resolution["vindicates"],
                    "villain_cards": resolution["villain_cards"],
                    "summary": resolution["summary"],
                }
            hd = hand_difficulties.get(hand.hand_id, 0)
            start = len(rows)
            for r in hand_rows:
                r["hand_difficulty"] = str(hd)
            ending_counts[hand.ending_street] = (
                ending_counts.get(hand.ending_street, 0) + 1
            )
            rows.extend(hand_rows)
            records.extend(hand_recs)
            leg_numbers = list(range(start + 1, start + len(hand_rows) + 1))
            hand_index.append({
                "hand_id": hand.hand_id, "hero": hand.hero,
                "hero_combo": hand.hero_combo, "frame": hand.frame,
                "row_numbers": leg_numbers, "legs": hand.total,
                "hand_difficulty": hd,
            })

    # Deterministic cross-check (July 2026, ported from preflop): re-read
    # every row AS WRITTEN and verify first-principles facts (positions,
    # skills, domination direction, frequency sums, difficulty bands).
    # Zero LLM, zero false positives in calibration; findings land on the
    # question records so Review shows them as machine-verified errors.
    cross_check_map = run_full_hand_cross_check(rows, records) if rows else {}
    cross_check_problems = 0
    for idx, cc_issues in cross_check_map.items():
        records[idx]["cross_check_issues"] = cc_issues
        cross_check_problems += len(cc_issues)

    output_path = Path(output_path)
    # Full-hand batches emit the TRIMMED schema (no pot_odds..easy_hand
    # diagnostics; hand_difficulty kept) -- see FULL_HAND_CSV_COLUMNS.
    write_postflop_csv(output_path, rows, columns=FULL_HAND_CSV_COLUMNS)

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
                "prompt_names": prompt_names or {},
                "min_hand_difficulty": min_hand_difficulty,
                "max_hand_difficulty": max_hand_difficulty,
                # The hand-selection shuffle seed (None = legacy fixed order).
                # Recorded so any batch is reproducible: same solve + same
                # settings + this seed = the same hands.
                "variety_seed": variety_seed,
                "diversify_hands": diversify_hands,
                "balanced_lengths": balanced_lengths,
                "length_profile": length_profile,
                "run_claim_checker": run_claim_checker,
                "revise_pass": revise_pass,
                "final_audit": final_audit and revise_pass,
            },
            "counters": {
                "worthy_spots_available": len(worthy),
                "low_quality_nodes_skipped": low_quality,
                "premise_filtered_nodes": premise_skipped,
                # Seed spots silenced by artifact-strip materiality (their
                # real strategy mixes an unrealistic jam >= 5%); mid-hand
                # material legs are additionally skipped inside _build_legs.
                "artifact_material_spots_skipped": artifact_material_out,
                "hands_assembled": hands_scanned,
                "hands_difficulty_filtered": hands_difficulty_filtered,
                # Band-scan diagnostics (July 2026): the hardest hand seen in
                # the scan, and whether the scan pool ran out before the
                # requested count was reached -- the Review page reads these
                # to explain an empty/short band-filtered batch.
                "hand_difficulty_observed_max": hand_difficulty_observed_max,
                "band_scan_capped": band_scan_capped,
                "preflop_leg_pack_used": agg["preflop_leg_pack_used"],
                "preflop_leg_entry_fallback": agg["preflop_leg_entry_fallback"],
                "preflop_line_legs_dropped": preflop_line_legs_dropped,
                "preflop_entry_legs_dropped": preflop_entry_legs_dropped,
                # July 15 2026 (user's call): a refused preflop leg drops the
                # WHOLE hand -- no play-through ever starts at the flop. This
                # counts the hands discarded for it (backfill refills the
                # batch where a reserve/remainder exists).
                "hands_dropped_preflop_incoherent": hands_dropped_preflop_incoherent,
                "hands_dropped_failed_leg": hands_dropped_failed_leg,
                "hands_topped_up": hands_topped_up,
                "showdown_resolutions": showdown_resolutions,
                "cross_check_problems": cross_check_problems,
                # Balanced-lengths accounting: the ACTUAL ending-street mix
                # of the committed hands (quotas back-fill when a bucket
                # runs short, so this is the ground truth, not the target),
                # and how many preflop-ender candidates the pack yielded.
                "hands_by_ending": ending_counts,
                "preflop_ender_candidates": len(preflop_ender_hands),
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
    _require_single_raised_pot(solve, "Preflop-entry mode")
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
