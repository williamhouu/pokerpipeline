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

from pipeline.postflop.adapters.sqlite_db import (
    DEFAULT_MAX_NODES_PER_STREET,
    load_postflop_db,
)
from pipeline.postflop.batch import PostflopBatchResult, generate_postflop_batch
from pipeline.postflop.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    load_postflop_system_prompt,
)
from pipeline.postflop.preflop_entry import load_preflop_entry_system_prompt
from pipeline.postflop.premise import DEFAULT_MIN_PREMISE_FREQ
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
    streets: tuple[str, ...] = ("flop",),
    max_nodes_per_street: int | None = DEFAULT_MAX_NODES_PER_STREET,
    diversify: bool = False,
    strength_buckets: tuple[str, ...] = (),
    decision_types: tuple[str, ...] = (),
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
    quality_gate: bool = True,
    min_premise_freq: float | None = DEFAULT_MIN_PREMISE_FREQ,
    equity_runouts: int | None = None,
    system_prompt: str | None = None,
    trap_difficulty: bool = False,
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    revise_pass: bool = False,
    final_audit: bool = False,
    progress_callback: Any = None,
) -> PostflopBatchResult:
    """Load the ``.db`` solve, curate spots, and generate a batch.

    The solve's strategic scenario (table size, positions, cash/tournament) comes
    from the file's own metadata; only the display framing (``stakes`` /
    ``live_or_online`` / ``bb_in_dollars``) is passed in here, since the solve
    doesn't carry stakes. ``heroes`` keeps only those acting positions (empty =
    both); ``streets`` selects which streets to materialise (flop / turn / river)
    and ``max_nodes_per_street`` caps each street's (large) node set; ``diversify``
    round-robins the decision types across streets. A real run needs
    ``ANTHROPIC_API_KEY``; without it (or with ``dry_run``) the deterministic
    placeholder prose is used.
    """
    solve = load_postflop_db(
        db_path,
        streets=tuple(streets) or ("flop",),
        max_nodes_per_street=max_nodes_per_street,
        stakes=stakes,
        live_or_online=live_or_online,
        bb_in_dollars=bb_in_dollars,
    )
    # Curation filters need to know the preflop aggressor + IP seat to classify
    # the decision SITUATION (c-bet vs lead vs facing a bet) without leaking the
    # answer. Both come from the solve.
    from pipeline.postflop.facts import preflop_aggressor  # noqa: PLC0415

    selector = make_spot_selector(
        heroes=tuple(heroes) or None,
        diversify=diversify,
        strength_buckets=tuple(strength_buckets) or None,
        decision_types=tuple(decision_types) or None,
        aggressor=preflop_aggressor(solve),
        ip_position=solve.ip_position,
    )

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
        quality_gate=quality_gate,
        min_premise_freq=min_premise_freq,
        system_prompt=prompt,
        trap_difficulty=trap_difficulty,
        run_claim_checker=run_claim_checker,
        claim_checker_prompt=claim_checker_prompt,
        revise_pass=revise_pass,
        final_audit=final_audit,
        progress_callback=progress_callback,
        spot_selector=selector,
        # Provenance for the batch re-verifier (scripts/audit_postflop_batch.py):
        # how to reload THIS exact solve (same streets + down-sampling cap).
        provenance={
            "db_path": str(db_path),
            "streets": list(streets) or ["flop"],
            "max_nodes_per_street": max_nodes_per_street,
            "stakes": stakes,
            "live_or_online": live_or_online,
            "bb_in_dollars": bb_in_dollars,
        },
        **kwargs,
    )


def compare_postflop_batches_from_db(
    *,
    db_path: str,
    output_path_a: str | Path,
    output_path_b: str | Path,
    total_questions: int,
    system_prompt_a: str,
    system_prompt_b: str,
    model_a: str = DEFAULT_MODEL,
    model_b: str = DEFAULT_MODEL,
    name_a: str = "Prompt A",
    name_b: str = "Prompt B",
    heroes: tuple[str, ...] = (),
    streets: tuple[str, ...] = ("flop",),
    max_nodes_per_street: int | None = DEFAULT_MAX_NODES_PER_STREET,
    diversify: bool = False,
    strength_buckets: tuple[str, ...] = (),
    decision_types: tuple[str, ...] = (),
    stakes: str = "$1/$2",
    live_or_online: str = "Live",
    bb_in_dollars: float = 2.0,
    answer_style: str = "auto",
    display_in_bb: bool = True,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    min_ev_gap_bb: float | None = None,
    quality_gate: bool = True,
    min_premise_freq: float | None = DEFAULT_MIN_PREMISE_FREQ,
    equity_runouts: int | None = None,
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    dry_run: bool = False,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Generate TWO postflop batches on the SAME spots, A vs B, for the Compare page.

    The whole point of an A/B is that the only thing that differs is the prompt
    (or the model): so the solve is loaded ONCE and the SAME deterministic spot
    selector drives both batches. Postflop spot selection is fully deterministic
    (sorted nodes + combos, deterministic diversify), so both sides see identical
    spots with no shared RNG seed needed. Returns a picklable summary dict the
    admin stashes + renders side-by-side; each side also writes its own CSV +
    meta.json (so each is reviewable as a normal batch).

    A single subprocess job runs both sides sequentially (the job runner allows
    one active job), so the heavy equity sims never contend with the Streamlit UI.
    """
    solve = load_postflop_db(
        db_path,
        streets=tuple(streets) or ("flop",),
        max_nodes_per_street=max_nodes_per_street,
        stakes=stakes,
        live_or_online=live_or_online,
        bb_in_dollars=bb_in_dollars,
    )
    from pipeline.postflop.facts import preflop_aggressor  # noqa: PLC0415

    selector = make_spot_selector(
        heroes=tuple(heroes) or None,
        diversify=diversify,
        strength_buckets=tuple(strength_buckets) or None,
        decision_types=tuple(decision_types) or None,
        aggressor=preflop_aggressor(solve),
        ip_position=solve.ip_position,
    )

    client = None
    if not dry_run and os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic  # noqa: PLC0415

        client = Anthropic()

    kwargs: dict[str, Any] = {}
    if equity_runouts is not None:
        kwargs["equity_runouts"] = equity_runouts

    provenance = {
        "db_path": str(db_path),
        "streets": list(streets) or ["flop"],
        "max_nodes_per_street": max_nodes_per_street,
        "stakes": stakes,
        "live_or_online": live_or_online,
        "bb_in_dollars": bb_in_dollars,
    }

    # One combined progress bar across both sides (A fills [0, N), B fills [N, 2N)).
    total2 = max(1, total_questions * 2)

    def _side_cb(side: str, offset: int) -> Any:
        if progress_callback is None:
            return None

        def _cb(message: str, done: int, _total: int) -> None:
            progress_callback(f"{side} — {message}", offset + done, total2)

        return _cb

    def _run(out_path: str | Path, prompt: str, model: str, side: str, offset: int):
        return generate_postflop_batch(
            solve=solve,
            output_path=out_path,
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
            quality_gate=quality_gate,
            min_premise_freq=min_premise_freq,
            system_prompt=prompt,
            run_claim_checker=run_claim_checker,
            claim_checker_prompt=claim_checker_prompt,
            progress_callback=_side_cb(side, offset),
            spot_selector=selector,
            provenance=provenance,
            **kwargs,
        )

    res_a = _run(output_path_a, system_prompt_a, model_a, name_a, 0)
    res_b = _run(output_path_b, system_prompt_b, model_b, name_b, total_questions)

    return {
        "a_csv": str(output_path_a),
        "b_csv": str(output_path_b),
        "a_name": name_a,
        "b_name": name_b,
        "a_written": res_a.questions_written,
        "b_written": res_b.questions_written,
        "a_attempted": res_a.questions_attempted,
        "b_attempted": res_b.questions_attempted,
        "a_failures": len(res_a.failures),
        "b_failures": len(res_b.failures),
        "dry_run": res_a.dry_run,
        "model_a": res_a.model_used,
        "model_b": res_b.model_used,
        "in_tokens": res_a.total_input_tokens + res_b.total_input_tokens,
        "out_tokens": res_a.total_output_tokens + res_b.total_output_tokens,
        # Per-side splits (July 2026 spend-logger fix): the two sides can run
        # DIFFERENT models, so the lifetime ledger logs one entry per side at
        # that side's price instead of a mispriced blended total.
        "a_in_tokens": res_a.total_input_tokens,
        "a_out_tokens": res_a.total_output_tokens,
        "b_in_tokens": res_b.total_input_tokens,
        "b_out_tokens": res_b.total_output_tokens,
    }


def generate_full_hand_batch_from_db(
    *,
    db_path: str,
    output_path: str | Path,
    total_hands: int,
    heroes: tuple[str, ...] = (),
    streets: tuple[str, ...] = ("flop", "turn", "river"),
    max_nodes_per_street: int | None = DEFAULT_MAX_NODES_PER_STREET,
    include_villain: bool = False,
    include_preflop: bool = True,
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
    quality_gate: bool = True,
    min_premise_freq: float | None = DEFAULT_MIN_PREMISE_FREQ,
    equity_runouts: int | None = None,
    system_prompt: str | None = None,
    preflop_system_prompt: str | None = None,
    preflop_pack_system_prompt: str | None = None,
    prompt_names: dict[str, str] | None = None,
    trap_difficulty: bool = False,
    razor_difficulty: bool = False,
    use_pack_preflop_legs: bool = True,
    min_hand_difficulty: int | None = None,
    max_hand_difficulty: int | None = None,
    variety_seed: int | None = None,
    diversify_hands: bool = False,
    balanced_lengths: bool = False,
    fully_balanced: bool = False,
    length_profile: str = "river_heavy",
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    revise_pass: bool = False,
    final_audit: bool = False,
    strict_clean_hands: bool = False,
    action_heavy: bool = True,
    llm_workers: int = 1,
    stop_check: Any = None,
    progress_callback: Any = None,
):
    """Load the ``.db`` (line-closed) and generate ``total_hands`` play-throughs.

    Each hand is several LINKED questions (preflop entry + the hero's decision on
    each street of one connected line) sharing a ``hand_id`` with an ascending
    ``sequence_index`` (Option B). ``include_villain`` also emits the villain's
    line on the same runout. The opt-in Layer-7 audit runs on the postflop legs.
    Picklable for the admin subprocess job runner.
    """
    from pipeline.postflop.full_hand_batch import generate_full_hand_batch

    solve = load_postflop_db(
        db_path,
        streets=tuple(streets) or ("flop", "turn", "river"),
        max_nodes_per_street=max_nodes_per_street,
        include_ancestors=True,  # the connected line must be fully materialised
        stakes=stakes,
        live_or_online=live_or_online,
        bb_in_dollars=bb_in_dollars,
    )

    client = None
    if not dry_run and os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic  # noqa: PLC0415

        client = Anthropic()

    kwargs: dict[str, Any] = {}
    if equity_runouts is not None:
        kwargs["equity_runouts"] = equity_runouts
    prompt = system_prompt if system_prompt is not None else load_postflop_system_prompt()
    pre_prompt = (
        preflop_system_prompt if preflop_system_prompt is not None
        else load_preflop_entry_system_prompt()
    )

    # Pack-backed preflop legs (July 2026): auto-select the closest preflop
    # range pack from the repo's ranges/ so the preflop leg carries the full
    # math (EVs, ranges, stat_notes, 4-axis difficulty). Fallback logic and
    # the coherence gate live in pipeline.postflop.preflop_leg_pack.
    ranges_root = (
        Path(__file__).resolve().parents[2] / "ranges"
        if use_pack_preflop_legs else None
    )

    return generate_full_hand_batch(
        solve=solve,
        output_path=output_path,
        total_hands=total_hands,
        client=client,
        model=model,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        dry_run=dry_run or client is None,
        answer_style=answer_style,
        display_in_bb=display_in_bb,
        heroes=tuple(heroes),
        include_villain=include_villain,
        include_preflop=include_preflop,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        min_ev_gap_bb=min_ev_gap_bb,
        quality_gate=quality_gate,
        min_premise_freq=min_premise_freq,
        system_prompt=prompt,
        preflop_system_prompt=pre_prompt,
        preflop_pack_system_prompt=preflop_pack_system_prompt,
        prompt_names=prompt_names,
        trap_difficulty=trap_difficulty,
        razor_difficulty=razor_difficulty,
        preflop_leg_pack_root=ranges_root,
        min_hand_difficulty=min_hand_difficulty,
        max_hand_difficulty=max_hand_difficulty,
        variety_seed=variety_seed,
        diversify_hands=diversify_hands,
        balanced_lengths=balanced_lengths,
        fully_balanced=fully_balanced,
        length_profile=length_profile,
        run_claim_checker=run_claim_checker,
        claim_checker_prompt=claim_checker_prompt,
        revise_pass=revise_pass,
        final_audit=final_audit,
        strict_clean_hands=strict_clean_hands,
        action_heavy=action_heavy,
        llm_workers=llm_workers,
        stop_check=stop_check,
        progress_callback=progress_callback,
        provenance={
            "db_path": str(db_path),
            "streets": list(streets) or ["flop", "turn", "river"],
            "max_nodes_per_street": max_nodes_per_street,
            "include_ancestors": True,
            "stakes": stakes,
            "live_or_online": live_or_online,
            "bb_in_dollars": bb_in_dollars,
        },
        **kwargs,
    )


def generate_preflop_entry_batch_from_db(
    *,
    db_path: str,
    output_path: str | Path,
    total_questions: int,
    heroes: tuple[str, ...] = (),
    stakes: str = "$1/$2",
    live_or_online: str = "Live",
    bb_in_dollars: float = 2.0,
    answer_style: str = "auto",
    display_in_bb: bool = True,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    preflop_system_prompt: str | None = None,
    progress_callback: Any = None,
):
    """Load the ``.db`` and generate STANDALONE preflop-entry questions.

    Sourced from the solve's flop-entry range frequencies (the entry decision
    that started the hand). Only the flop is materialised (the entry ranges come
    from the preflop-ranges table, not the node walk). Picklable for the admin.
    """
    from pipeline.postflop.full_hand_batch import generate_preflop_entry_batch

    solve = load_postflop_db(
        db_path,
        streets=("flop",),
        max_nodes_per_street=1,  # entry ranges don't need the node set
        stakes=stakes,
        live_or_online=live_or_online,
        bb_in_dollars=bb_in_dollars,
    )

    client = None
    if not dry_run and os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic  # noqa: PLC0415

        client = Anthropic()

    pre_prompt = (
        preflop_system_prompt if preflop_system_prompt is not None
        else load_preflop_entry_system_prompt()
    )
    return generate_preflop_entry_batch(
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
        heroes=tuple(heroes),
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        preflop_system_prompt=pre_prompt,
        progress_callback=progress_callback,
        provenance={
            "db_path": str(db_path),
            "stakes": stakes,
            "live_or_online": live_or_online,
            "bb_in_dollars": bb_in_dollars,
        },
    )


__all__ = [
    "POSTFLOP_OUTPUT_DIR",
    "compare_postflop_batches_from_db",
    "generate_full_hand_batch_from_db",
    "generate_postflop_batch_from_db",
    "generate_preflop_entry_batch_from_db",
]
