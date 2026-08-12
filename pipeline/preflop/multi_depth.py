"""ONE merged all-depths tournament batch (Aug 2026).

The admin panel's "🏆 All tournament depths" mode used to queue SEVEN
separate batches (one per MTT pack), each with its own CSV + meta.json +
Review entry + ledger entry. This module is the merge layer -- modeled on
the bet-sizing trainer's multi-solve pattern
(:mod:`pipeline.postflop.sizing_batch`, provenance ``sizing_multi``,
per-question ``solve_key``) but for the preflop pipeline:

* :func:`generate_all_depths_batch` runs :func:`~pipeline.preflop.batch.
  generate_preflop_batch` once per pack (ascending stack depth) into
  hidden per-depth temp files, then merges them into ONE CSV (the exact
  ``PREFLOP_CSV_COLUMNS`` schema, ``No`` renumbered 1..N) + ONE
  meta.json (``run_settings.all_depths`` / ``pack_ids`` /
  ``questions_per_depth``; counters summed with a ``per_depth``
  breakdown; every question record stamped with its ``pack_id`` +
  ``table_size`` -- the re-verifier's multi-pack join key).
* It returns the SAME :class:`~pipeline.preflop.batch.BatchResult` type
  with token totals + written counts summed, so the admin panel's done
  panel and ledger sweep work unchanged (one ledger entry per run).
* A depth that RAISES is recorded in meta ``failed_depths`` and the run
  continues; only when EVERY depth raises does the whole run raise.

Picklability: this function is a module-level top-level callable with
picklable kwargs, so ``admin_panel.jobs.start_subprocess_job`` /
``enqueue_subprocess_job`` can ship it to a child process directly.

The LLM never decides poker facts here -- this layer only orchestrates
per-depth generation and does file-level merging.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from pipeline.preflop.batch import (
    BatchResult,
    PreflopFailure,
    ProgressCallback,
    generate_preflop_batch,
)
from pipeline.preflop.format_writer import PREFLOP_CSV_COLUMNS
from pipeline.preflop.pack import PreflopPack

logger = logging.getLogger(__name__)

# Hidden working dir (next to the merged CSV) for the per-depth temp
# CSV/meta files. A SUBDIRECTORY on purpose: the Review page lists
# batches via ``glob("*.csv")`` on the output dir, and pathlib's glob
# matches dotFILES -- a temp CSV sitting in the same dir would flash up
# as a phantom batch mid-run. Removed after a successful merge.
_TMP_DIR_NAME = ".all_depths_tmp"


def _depth_label(pack: PreflopPack) -> str:
    """'20bb' -- the human depth tag used in progress + temp filenames."""
    return f"{pack.stack_depth_bb:g}bb"


def _sum_counters(per_depth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sum every NUMERIC counter across the per-depth counter dicts.

    Non-numeric values (none today) are skipped rather than guessed at.
    """
    totals: dict[str, Any] = {}
    for counters in per_depth.values():
        for key, value in counters.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0) + value
    return totals


def generate_all_depths_batch(  # noqa: PLR0912, PLR0915 -- one linear orchestration
    *,
    packs: Sequence[PreflopPack],
    questions_per_depth: int,
    output_path: Path | str,
    progress_callback: ProgressCallback | None = None,
    **batch_kwargs: Any,
) -> BatchResult:
    """Generate ONE merged batch across every pack in ``packs``.

    Args:
        packs: The packs (one per stack depth) to generate from. Run
            order is ascending ``stack_depth_bb`` regardless of input
            order; the merged CSV keeps that order.
        questions_per_depth: Target question count for EACH depth (the
            merged batch targets ``len(packs) * questions_per_depth``).
        output_path: The MERGED CSV path; its ``.meta.json`` sidecar is
            written next to it. Per-depth temp files live in a hidden
            subdirectory of its parent and are deleted after the merge.
        progress_callback: Optional ``(message, current, total)``
            reporter spanning the WHOLE run ("Depth 3/7 (20bb):
            Generating question 5/12 ...", current/total across all
            depths).
        **batch_kwargs: Every other kwarg
            :func:`~pipeline.preflop.batch.generate_preflop_batch`
            accepts (minus ``pack`` / ``total_questions`` /
            ``output_path`` / ``progress_callback``), applied
            identically to each depth.

    Returns:
        A :class:`BatchResult` for the MERGED batch: token totals and
        written/attempted counts summed across depths, ``output_path`` =
        the merged CSV (None when no depth produced rows). The admin
        ledger sweep reads exactly these fields, so one run logs one
        spend entry.

    Raises:
        RuntimeError: only when EVERY depth raised (nothing to merge).
        ValueError: on reserved kwargs / empty ``packs``.
    """
    reserved = {"pack", "total_questions"} & set(batch_kwargs)
    if reserved:
        raise ValueError(
            f"kwargs {sorted(reserved)} are owned by generate_all_depths_batch; "
            "pass packs= and questions_per_depth= instead."
        )
    ordered = sorted(packs, key=lambda p: p.stack_depth_bb)
    if not ordered:
        raise ValueError("packs must contain at least one PreflopPack")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / _TMP_DIR_NAME
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_depths = len(ordered)
    overall_total = questions_per_depth * n_depths

    results: list[tuple[PreflopPack, BatchResult]] = []
    failed_depths: list[dict[str, str]] = []

    for i, pack in enumerate(ordered):
        depth = _depth_label(pack)

        def _depth_progress(
            msg: str, current: int, total: int, _i: int = i, _depth: str = depth
        ) -> None:
            # Overall position = full quotas for finished depths + this
            # depth's own progress (a depth may write fewer than its
            # quota; the bar simply jumps at the boundary).
            del total  # the inner total is the per-depth quota
            if progress_callback is not None:
                progress_callback(
                    f"Depth {_i + 1}/{n_depths} ({_depth}): {msg}",
                    _i * questions_per_depth + current,
                    overall_total,
                )

        tmp_csv = tmp_dir / f"{out_path.stem}.{depth}.csv"
        try:
            res = generate_preflop_batch(
                pack=pack,
                output_path=tmp_csv,
                total_questions=questions_per_depth,
                progress_callback=_depth_progress,
                **batch_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 -- one depth must not kill the run
            logger.warning(
                "all-depths: depth %s (%s) failed: %s", depth, pack.pack_id, exc
            )
            failed_depths.append(
                {"pack_id": pack.pack_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        results.append((pack, res))

    if not results:
        raise RuntimeError(
            "all-depths batch: every depth failed -- "
            + "; ".join(f"{f['pack_id']}: {f['error']}" for f in failed_depths)
        )

    # --- merge ---------------------------------------------------------
    if progress_callback is not None:
        progress_callback(
            f"Merging {len(results)} depths into one batch…",
            overall_total,
            overall_total,
        )

    merged_rows: list[dict[str, str]] = []
    merged_questions: list[dict[str, Any]] = []
    merged_meta_failures: list[dict[str, Any]] = []
    per_depth_counters: dict[str, dict[str, Any]] = {}
    balance_by_depth: dict[str, Any] = {}
    base_meta: dict[str, Any] | None = None
    pack_ids_written: list[str] = []

    for pack, res in results:
        if res.output_path is None:
            # A depth with zero worthy spots writes nothing -- record an
            # honest empty breakdown so the shortfall is visible in meta.
            per_depth_counters[pack.pack_id] = {
                "questions_written": 0,
                "questions_attempted": res.questions_attempted,
                "worthy_spots_available": res.worthy_spots_available,
                "nodes_after_filter": res.nodes_after_filter,
            }
            continue
        depth_meta = json.loads(
            res.output_path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        with open(res.output_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == PREFLOP_CSV_COLUMNS, (
                f"depth {pack.pack_id} CSV schema drifted: {reader.fieldnames}"
            )
            depth_rows = list(reader)
        merged_rows.extend(depth_rows)
        pack_ids_written.append(pack.pack_id)
        if base_meta is None:
            base_meta = deepcopy(depth_meta)
        # Stamp EVERY question record with its source pack -- the
        # re-verifier's + Review inspector's multi-pack join key.
        for record in depth_meta.get("questions", []):
            record.setdefault("pack_id", pack.pack_id)
            record.setdefault("table_size", pack.table_size)
            merged_questions.append(record)
        merged_meta_failures.extend(depth_meta.get("failures", []))
        per_depth_counters[pack.pack_id] = depth_meta.get("counters", {}) or {}
        if depth_meta.get("balance_report"):
            balance_by_depth[pack.pack_id] = depth_meta["balance_report"]

    written_total = sum(res.questions_written for _p, res in results)
    final_out: Path | None = None
    meta_path: Path | None = None

    if merged_rows and base_meta is not None:
        # ONE CSV: concatenated ascending depth, ``No`` renumbered from 1.
        for number, row in enumerate(merged_rows, start=1):
            row["No"] = str(number)
        with open(out_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREFLOP_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(merged_rows)
        final_out = out_path

        # ONE meta.json: the shared prompt/model/settings snapshot from the
        # first successful depth (identical across depths by construction),
        # with the multi-pack fields layered on top.
        meta = base_meta
        # Top-level pack_id: a human summary, NOT a resolvable pack id --
        # Review's "📦 Pack:" caption renders it as-is; the re-verifier
        # resolves per-question pack_id records instead.
        meta["pack_id"] = f"{len(pack_ids_written)} tournament depths"
        run_settings = dict(meta.get("run_settings") or {})
        run_settings["all_depths"] = True
        run_settings["pack_ids"] = pack_ids_written
        run_settings["questions_per_depth"] = questions_per_depth
        meta["run_settings"] = run_settings
        meta["counters"] = {
            **_sum_counters(per_depth_counters),
            "per_depth": per_depth_counters,
        }
        # Top-level balance_report stays None (its flat shape is per-batch;
        # the per-depth reports keep full provenance under their own key).
        meta["balance_report"] = None
        if balance_by_depth:
            meta["balance_reports_by_depth"] = balance_by_depth
        if failed_depths:
            meta["failed_depths"] = failed_depths
        meta["questions"] = merged_questions
        meta["failures"] = merged_meta_failures
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        # Merge succeeded -> the per-depth temp files are redundant.
        shutil.rmtree(tmp_dir, ignore_errors=True)
    elif not merged_rows:
        # Nothing written anywhere (all depths ran but produced 0 rows):
        # no CSV, no meta -- same contract as an empty single-pack batch.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    merged_failures: list[PreflopFailure] = []
    for _pack, res in results:
        merged_failures.extend(res.failures)

    return BatchResult(
        output_path=final_out,
        questions_written=written_total,
        questions_attempted=sum(r.questions_attempted for _p, r in results),
        failures=merged_failures,
        worthy_spots_available=sum(r.worthy_spots_available for _p, r in results),
        nodes_after_filter=sum(r.nodes_after_filter for _p, r in results),
        difficulty_filtered_out=sum(
            r.difficulty_filtered_out for _p, r in results
        ),
        noise_filtered_out=sum(r.noise_filtered_out for _p, r in results),
        incoherent_mix_filtered_out=sum(
            r.incoherent_mix_filtered_out for _p, r in results
        ),
        rare_line_filtered_out=sum(r.rare_line_filtered_out for _p, r in results),
        rare_premise_filtered_out=sum(
            r.rare_premise_filtered_out for _p, r in results
        ),
        soft_flagged_rows=sum(r.soft_flagged_rows for _p, r in results),
        requested_questions=overall_total,
        # Summed token totals -> the panel's ledger sweep logs ONE entry
        # with the whole run's spend.
        total_input_tokens=sum(r.total_input_tokens for _p, r in results),
        total_output_tokens=sum(r.total_output_tokens for _p, r in results),
        total_cache_creation_tokens=sum(
            r.total_cache_creation_tokens for _p, r in results
        ),
        total_cache_read_tokens=sum(
            r.total_cache_read_tokens for _p, r in results
        ),
        model_used=next(
            (r.model_used for _p, r in results if r.model_used), ""
        ),
        prompt_name=next(
            (r.prompt_name for _p, r in results if r.prompt_name), ""
        ),
        meta_path=meta_path,
    )
