"""📏 Bet-sizing trainer batches (July 2026) -- MULTI-SOLVE, fully balanced.

One batch of pure bet-SIZING questions drawn across EVERY installed postflop
solve at once, balanced on seven axes. A sizing-viable spot is one where the
menu offers >= 2 open-bet sizes AND the solver's dominant action IS one of
those sized bets (worthy 65-99), so the wrong options naturally include the
other sizes plus Check -- the question is literally "which size?".

Pipeline: load each solve -> collect + curate its sizing pool (class-deduped,
street/situation round-robin) -> score every candidate's balance attrs
(exact difficulty band; see below) -> ONE global greedy balanced ordering
(:mod:`pipeline.balanced_select`, the shared leaf) -> generate per solve via
the untouched :func:`pipeline.postflop.batch.generate_postflop_batch`
(``answer_style="sizing"``, spots pre-picked through ``spot_selector``) ->
merge the sub-batches into ONE CSV + ONE meta in the global balanced order.

DIFFICULTY BANDS ARE EXACT, NOT ESTIMATES: postflop difficulty reads only the
dominant frequency, the archetype + the three concept-tag modifiers
(multiway / wet board / range disadvantage), and the hand-strength bucket --
none of which depend on the sampled hero-equity number (verified by
``test_pool_difficulty_band_is_exact``). The pool pass therefore runs
``extract_facts`` with reduced equity runouts purely for speed; the band it
computes is identical to the one generation later computes at full runouts.
(The node-level range-advantage sim is memoised in ``facts.py``, which is
what makes scoring a few hundred candidates affordable.)

BALANCE AXES (weights follow the PLO/full-hand precedent -- structure first,
texture last). "Correct size" buckets the answer's pot fraction so the batch
teaches small-vs-medium-vs-big rather than 50 small c-bets:

    flop/solve 1.0 · street 1.0 · difficulty band 0.9 · correct size 0.8
    · situation 0.7 · position 0.5 · hand strength 0.25

HONESTY RULE: shortfalls are never silently padded -- the meta
``balance_report`` records achieved-vs-target per axis, and a pool that
cannot fill 50 ships what exists.

Deterministic end-to-end (seeded per-spot sims, fixed-seed range equity,
deterministic greedy ordering + merge), so the multi-solve re-verifier path
in ``scripts/audit_postflop_batch.py`` rebuilds every row byte-exactly.
"""

from __future__ import annotations

import collections
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeline.balanced_select import (
    Axis,
    balance_report as _core_balance_report,
    balanced_order as _core_balanced_order,
)
from pipeline.postflop.adapters.sqlite_db import load_postflop_db, summarize_db
from pipeline.postflop.batch import (
    DEFAULT_MODEL,
    _collect_worthy,
    generate_postflop_batch,
)
from pipeline.postflop.difficulty import compute_difficulty
from pipeline.postflop.facts import extract_facts, preflop_aggressor
from pipeline.postflop.format_writer import write_postflop_csv
from pipeline.postflop.solve import PostflopSolve
from pipeline.postflop.spot_selection import (
    spot_decision_type,
    spot_strength_bucket,
)

# Street node caps for the multi-solve load. River mirrors the full-hand
# 2500 floor (the down-sample otherwise starves river sizing spots).
SIZING_MAX_NODES_PER_STREET: dict[str, int] = {
    "flop": 600, "turn": 600, "river": 2500,
}

# Pool-scoring equity runouts. Speed only -- the difficulty band is
# equity-independent (module docstring), and generation recomputes every
# fact at the batch's full runouts.
POOL_EQUITY_RUNOUTS = 80

# Pool sizing: score up to this many candidates per requested question
# (bounded), split across the solves.
POOL_FACTOR = 8
POOL_CAP = 400

# Difficulty band edges -- the admin Generate presets' convention
# (Easy 400-1300, Medium 1300-2100, Hard 2100-3200).
_BAND_EDGES = ((1300, "Easy"), (2100, "Medium"))

# (attr key, plain-English label, weight). Order = display order.
SIZING_BALANCE_AXES: tuple[Axis, ...] = (
    ("flop", "Flop", 1.00),
    ("street", "Street", 1.00),
    ("difficulty_band", "Difficulty", 0.90),
    ("size_family", "Correct size", 0.80),
    ("situation", "Situation", 0.70),
    ("position", "Position", 0.50),
    ("strength", "Hand strength", 0.25),
)


def difficulty_band(score: float) -> str:
    for edge, name in _BAND_EDGES:
        if score < edge:
            return name
    return "Hard"


def is_bet_size_label(label: str) -> bool:
    """An open-bet action label carrying a size ("Bet 4.5bb")."""
    return label.startswith("Bet ")


def is_sizing_spot(spot: Any) -> bool:
    """Menu offers >= 2 open-bet sizes AND the dominant action is one."""
    sizes = [a for a in spot.live_actions if is_bet_size_label(a.label)]
    return len(sizes) >= 2 and is_bet_size_label(spot.dominant_action)


def size_family(spot: Any) -> str:
    """Bucket the CORRECT size's pot fraction: small / medium / big-overbet."""
    for action in spot.live_actions:
        if action.label == spot.dominant_action:
            f = action.pot_fraction
            if f is None:
                return "unsized"
            if f < 0.45:  # noqa: PLR2004
                return "small (<45% pot)"
            if f < 0.90:  # noqa: PLR2004
                return "medium (45-90%)"
            return "big/overbet (90%+)"
    return "unsized"


def hand_class_key(combo: str) -> str:
    """'5h5d' -> '55', 'Kc7c' -> 'K7s', 'Kc7d' -> 'K7o' (suit-twin dedupe)."""
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def collect_sizing_pool(
    solve: PostflopSolve,
    *,
    min_frequency: float = 0.65,
    max_frequency: float = 0.99,
    per_solve_cap: int | None = None,
) -> list[Any]:
    """This solve's curated sizing candidates, deterministic order.

    Sizing filter + one-combo-per-(node, hand class) dedupe (suit twins of
    the same class sit adjacent in worthy order and would fill a batch with
    repeats), then a (street, situation) round-robin so a per-solve cap
    keeps every category represented rather than the first bucket only.
    """
    worthy, *_ = _collect_worthy(
        solve,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        min_ev_gap_bb=None,
    )
    aggressor = preflop_aggressor(solve)
    ip = solve.ip_position
    buckets: dict[tuple, list] = collections.defaultdict(list)
    seen_class: set[tuple] = set()
    for spot in worthy:
        if not is_sizing_spot(spot):
            continue
        class_key = (spot.node.node_id, hand_class_key(spot.hero_combo))
        if class_key in seen_class:
            continue
        seen_class.add(class_key)
        try:
            situation = spot_decision_type(spot, aggressor=aggressor, ip_position=ip)
        except Exception:  # noqa: BLE001 -- classification is best-effort
            situation = "?"
        buckets[(spot.node.street, situation)].append(spot)
    ordered: list[Any] = []
    keys = sorted(buckets)
    idx = dict.fromkeys(keys, 0)
    while any(idx[k] < len(buckets[k]) for k in keys):
        for k in keys:
            if idx[k] < len(buckets[k]):
                ordered.append(buckets[k][idx[k]])
                idx[k] += 1
    if per_solve_cap is not None:
        ordered = ordered[:per_solve_cap]
    return ordered


def solve_display_key(db_path: str) -> str:
    """Short unique-ish key for one solve: '<flop> <SRP|3BP> <stack>bb'."""
    s = summarize_db(db_path)
    spot = (s.spot or "").lower()
    pot_type = "3BP" if ("3bp" in spot or "3bet" in spot) else "SRP"
    stack = f" {int(s.stack_bb)}bb" if s.stack_bb else ""
    flop = s.flop or Path(db_path).stem
    return f"{flop} {pot_type}{stack}"


def _attrs_for(
    spot: Any,
    solve: PostflopSolve,
    solve_key: str,
    *,
    equity_runouts: int = POOL_EQUITY_RUNOUTS,
) -> dict[str, str]:
    """One candidate's value on every balance axis (exact difficulty band)."""
    facts = extract_facts(spot, solve, equity_runouts=equity_runouts)
    diff = compute_difficulty(facts)
    aggressor = preflop_aggressor(solve)
    try:
        situation = spot_decision_type(
            spot, aggressor=aggressor, ip_position=solve.ip_position
        )
    except Exception:  # noqa: BLE001
        situation = "?"
    return {
        "flop": solve_key,
        "street": spot.node.street,
        "difficulty_band": difficulty_band(diff.score),
        "size_family": size_family(spot),
        "situation": situation,
        "position": spot.node.actor,
        "strength": spot_strength_bucket(spot),
    }


@dataclass
class SizingBatchResult:
    """Summary of one multi-solve sizing batch run."""

    output_path: Path
    meta_path: Path | None
    questions_written: int
    requested_questions: int
    pool_scored: int
    per_solve_written: dict[str, int] = field(default_factory=dict)
    balance_report: dict = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = True
    model_used: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    solves_skipped: dict[str, str] = field(default_factory=dict)
    # Compat with the admin done-panel (_render_postflop_result_ui reads these
    # off every postflop result): the sizing "worthy pool" is the scored pool.
    worthy_spots_available: int = 0
    soft_flagged_rows: int = 0


def generate_sizing_batch(  # noqa: PLR0912, PLR0915 -- one linear orchestration
    db_paths: Sequence[str],
    output_path: Path | str,
    *,
    total_questions: int = 50,
    client: object | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    display_in_bb: bool = True,
    display_split: bool = False,
    min_frequency: float = 0.65,
    max_frequency: float = 0.99,
    system_prompt: str | None = None,
    prompt_name: str | None = None,
    run_claim_checker: bool = False,
    revise_pass: bool = False,
    final_audit: bool = False,
    streets: tuple[str, ...] = ("flop", "turn", "river"),
    max_nodes_per_street: Mapping[str, int] | int | None = None,
    stakes: str = "",
    live_or_online: str = "Online",
    bb_in_dollars: float = 1.0,
    progress_callback: Callable[[str, int, int], None] | None = None,
    solves_loaded: Mapping[str, PostflopSolve] | None = None,
    llm_workers: int = 1,
) -> SizingBatchResult:
    """One fully-balanced bet-sizing batch across ``db_paths`` (see module doc).

    ``solves_loaded`` (tests only) maps a display key to a pre-built in-memory
    solve, bypassing the ``.db`` loader; provenance then records no db_path
    and the batch is not re-verifiable (fixture batches never are).

    The display framing (``stakes`` / ``live_or_online`` / ``bb_in_dollars``)
    applies to every solve, exactly like the single-solve run wrapper.
    """
    output_path = Path(output_path)
    caps = max_nodes_per_street or SIZING_MAX_NODES_PER_STREET

    def _progress(msg: str, done: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(msg, done, total)

    # --- 1. load every solve (unreadable ones reported, never fatal) --------
    sources: dict[str, tuple[str, PostflopSolve]] = {}
    skipped: dict[str, str] = {}
    if solves_loaded is not None:
        sources = {key: ("", solve) for key, solve in solves_loaded.items()}
    else:
        for db_path in db_paths:
            try:
                key = solve_display_key(db_path)
                solve = load_postflop_db(
                    db_path,
                    streets=streets,
                    max_nodes_per_street=caps,
                    stakes=stakes,
                    live_or_online=live_or_online,
                    bb_in_dollars=bb_in_dollars,
                )
            except Exception as exc:  # noqa: BLE001 -- report, don't die
                skipped[str(db_path)] = f"{type(exc).__name__}: {exc}"
                continue
            # Collisions (same flop/pot-type/stack twice) disambiguate by suffix.
            base, n = key, 2
            while key in sources:
                key = f"{base} ({n})"
                n += 1
            sources[key] = (str(db_path), solve)
    if not sources:
        raise ValueError(
            "no readable solves for the sizing batch: "
            + "; ".join(f"{p}: {e}" for p, e in skipped.items())
        )

    # --- 2. per-solve sizing pools ------------------------------------------
    pool_target = min(POOL_FACTOR * total_questions, POOL_CAP)
    per_solve_cap = max(24, -(-pool_target // len(sources)))  # ceil div
    pool: list[tuple[str, Any]] = []  # (solve_key, spot)
    for key in sorted(sources):
        _, solve = sources[key]
        for spot in collect_sizing_pool(
            solve,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            per_solve_cap=per_solve_cap,
        ):
            pool.append((key, spot))

    # --- 3. score balance attrs (the only pre-pass cost) --------------------
    attrs: list[dict[str, str]] = []
    for i, (key, spot) in enumerate(pool):
        _progress(f"Scoring sizing pool {i + 1}/{len(pool)}", 0, total_questions)
        attrs.append(_attrs_for(spot, sources[key][1], key))

    # --- 4. one global balanced ordering ------------------------------------
    order = _core_balanced_order(
        attrs,
        SIZING_BALANCE_AXES,
        spread_keys=[f"{key}|{spot.node.node_id}" for key, spot in pool],
    )
    ranked = [pool[i] for i in order]  # (key, spot) in global balanced order
    rank_of = {
        (key, spot.node.node_id, spot.hero_combo): r
        for r, (key, spot) in enumerate(ranked)
    }
    attrs_of = {
        (key, spot.node.node_id, spot.hero_combo): attrs[i]
        for i, (key, spot) in enumerate(pool)
    }
    intended = ranked[:total_questions]
    # 💵/bb EVEN SPLIT (Aug 2026, user ask): alternate the display currency
    # down the GLOBAL balanced order, so the shipped batch is ~half big
    # blinds and ~half dollars (and backfill after failures keeps
    # alternating). Each question's currency is recorded in its meta record;
    # options are dollarized at build time by the currency-consistency rule.
    currency_by_rank = {
        rk: (rk % 2 == 0) if display_split else display_in_bb
        for rk in range(len(ranked))
    }
    in_bb_of = {
        (key, spot.node.node_id, spot.hero_combo): currency_by_rank[rk]
        for rk, (key, spot) in enumerate(ranked)
    }
    targets = collections.Counter(
        (key, in_bb_of[(key, spot.node.node_id, spot.hero_combo)])
        for key, spot in intended
    )

    # --- 5. generate per solve through the standard batch -------------------
    sub_rows: list[tuple[int, dict[str, str]]] = []  # (global rank, row)
    sub_questions: list[tuple[int, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    per_solve_written: dict[str, int] = {}
    counters_sum: collections.Counter = collections.Counter()
    in_tokens = out_tokens = cache_c_tokens = cache_r_tokens = 0
    model_used = ""
    sub_run_settings: dict[str, Any] = {}
    done_so_far = 0

    with tempfile.TemporaryDirectory(prefix="sizing_batch_") as tmp:
        for key, group_in_bb in sorted(targets):
            db_path, solve = sources[key]
            target = targets[(key, group_in_bb)]
            # This solve's spots in global balanced order, THIS currency
            # group only: the intended picks first, then its remainder as
            # balanced backfill after failures.
            mine = [
                spot
                for k, spot in ranked
                if k == key
                and in_bb_of[(k, spot.node.node_id, spot.hero_combo)]
                == group_in_bb
            ]
            pick_keys = [(spot.node.node_id, spot.hero_combo) for spot in mine]
            pick_index = {pk: i for i, pk in enumerate(pick_keys)}

            def _selector(
                worthy: list[Any], _idx: Mapping = pick_index
            ) -> list[Any]:
                chosen = [
                    s for s in worthy
                    if (s.node.node_id, s.hero_combo) in _idx
                ]
                chosen.sort(key=lambda s: _idx[(s.node.node_id, s.hero_combo)])
                return chosen

            offset = done_so_far

            def _sub_progress(
                msg: str, done: int, _total: int, _offset: int = offset
            ) -> None:
                _progress(msg, _offset + done, total_questions)

            sub_out = Path(tmp) / f"{len(sub_rows)}_{key}_{int(group_in_bb)}.csv"
            result = generate_postflop_batch(
                solve=solve,
                output_path=sub_out,
                total_questions=target,
                client=client,
                model=model,
                dry_run=dry_run,
                answer_style="sizing",
                display_in_bb=group_in_bb,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                system_prompt=system_prompt,
                run_claim_checker=run_claim_checker,
                revise_pass=revise_pass,
                final_audit=final_audit,
                llm_workers=llm_workers,
                spot_selector=_selector,
                progress_callback=_sub_progress,
                provenance={
                    "db_path": db_path,
                    "streets": list(streets),
                    "max_nodes_per_street": dict(caps)
                    if isinstance(caps, Mapping) else caps,
                    "stakes": stakes,
                    "live_or_online": live_or_online,
                    "bb_in_dollars": bb_in_dollars,
                    "display_in_bb": group_in_bb,
                },
            )
            per_solve_written[key] = (
                per_solve_written.get(key, 0) + result.questions_written
            )
            done_so_far += result.questions_written
            in_tokens += result.total_input_tokens
            out_tokens += result.total_output_tokens
            cache_c_tokens += result.total_cache_creation_tokens
            cache_r_tokens += result.total_cache_read_tokens
            model_used = result.model_used

            sub_meta = json.loads(
                sub_out.with_suffix(".meta.json").read_text(encoding="utf-8")
            )
            sub_run_settings = sub_meta.get("run_settings", {})
            for cname, cval in sub_meta.get("counters", {}).items():
                if isinstance(cval, (int, float)):
                    counters_sum[cname] += cval
            for f in sub_meta.get("failures", []):
                failures.append({**f, "solve_key": key})
            with sub_out.open(encoding="utf-8-sig") as fh:
                import csv as _csv  # noqa: PLC0415

                rows = list(_csv.DictReader(fh))
            for row, q in zip(rows, sub_meta.get("questions", []), strict=True):
                rk = rank_of.get((key, q["node_id"], q["hero_combo"]))
                # Every shipped spot came from the ranked pool; a miss would
                # mean the sub-batch generated something we never picked.
                assert rk is not None, (key, q["node_id"], q["hero_combo"])
                sub_rows.append((rk, row))
                sub_questions.append(
                    (rk, {**q, "solve_key": key, "display_in_bb": group_in_bb})
                )

    # --- 6. merge in the global balanced order + renumber -------------------
    sub_rows.sort(key=lambda t: t[0])
    sub_questions.sort(key=lambda t: t[0])
    merged_rows: list[dict[str, str]] = []
    merged_questions: list[dict[str, Any]] = []
    shipped_attrs: list[dict[str, str]] = []
    for i, ((rk, row), (_rk2, q)) in enumerate(
        zip(sub_rows, sub_questions, strict=True)
    ):
        row = dict(row)
        row["No"] = str(i + 1)
        merged_rows.append(row)
        merged_questions.append(q)
        shipped_attrs.append(
            attrs_of[(q["solve_key"], q["node_id"], q["hero_combo"])]
        )

    write_postflop_csv(output_path, merged_rows)
    report = _core_balance_report(shipped_attrs, attrs, SIZING_BALANCE_AXES)

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "solve_id": "sizing_multi",
        "source_reference": "; ".join(sorted(sources)),
        # No shared flop-entry grid across solves -- each question record
        # carries its own street_ranges/prior_street_ranges (Review renders
        # those; the shared grid is skipped when this is empty).
        "preflop_ranges": {},
        "preflop_entry_actions": {},
        "model": model_used or ("(dry-run placeholder)" if dry_run else model),
        "dry_run": dry_run or client is None,
        "provenance": {
            "mode": "sizing_multi",
            "solves": {
                key: {
                    "db_path": sources[key][0],
                    "streets": list(streets),
                    "max_nodes_per_street": dict(caps)
                    if isinstance(caps, Mapping) else caps,
                    "stakes": stakes,
                    "live_or_online": live_or_online,
                    "bb_in_dollars": bb_in_dollars,
                }
                for key in sorted(sources)
            },
            "solves_skipped": skipped,
        },
        "run_settings": {
            **sub_run_settings,
            "display_in_bb": display_in_bb,
            "display_split": display_split,
            "total_questions": total_questions,
            "sizing_mode": True,
            # Which explanation prompt wrote this batch (the admin picker's
            # library-entry name, or the CLI's description of the default).
            "prompt_name": prompt_name or "",
            "balance_axes": [
                {"axis": k, "label": lbl, "weight": w}
                for k, lbl, w in SIZING_BALANCE_AXES
            ],
        },
        "counters": {
            **dict(counters_sum),
            "questions_written": len(merged_rows),
            "sizing_pool_scored": len(pool),
            "solves_used": len(sources),
            "solves_skipped": len(skipped),
        },
        "balance_report": report,
        "questions": merged_questions,
        "failures": failures,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return SizingBatchResult(
        output_path=output_path,
        meta_path=meta_path,
        questions_written=len(merged_rows),
        requested_questions=total_questions,
        pool_scored=len(pool),
        per_solve_written=per_solve_written,
        balance_report=report,
        failures=failures,
        dry_run=dry_run or client is None,
        model_used=model_used,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
        total_cache_creation_tokens=cache_c_tokens,
        total_cache_read_tokens=cache_r_tokens,
        solves_skipped=skipped,
        worthy_spots_available=len(pool),
        soft_flagged_rows=int(counters_sum.get("soft_flagged_rows", 0)),
    )


__all__ = [
    "POOL_EQUITY_RUNOUTS",
    "SIZING_BALANCE_AXES",
    "SIZING_MAX_NODES_PER_STREET",
    "SizingBatchResult",
    "collect_sizing_pool",
    "difficulty_band",
    "generate_sizing_batch",
    "hand_class_key",
    "is_sizing_spot",
    "size_family",
    "solve_display_key",
]
