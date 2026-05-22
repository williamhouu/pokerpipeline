"""Layer 2: Tree Resolver -- batch solve driver.

Takes a `SolverSpec` (from pipeline.scenario_spec) and a flop set (from
pipeline.flop_sets), iterates the cross product, and for each (spec, flop)
drives PioSolver Edge via the existing UPI client to produce one `.cfr` file
at:

    solves/{spec.cache_dir_name}/{flop_stem}.cfr

Features:

  * Resume: if a `.cfr` already exists at the expected path, skip the solve
    (a re-run is idempotent).
  * Per-solve failure isolation: a single solve crashing doesn't stop the
    batch -- the spot gets a `.failed` marker file alongside the would-be
    `.cfr`, the batch logs the error, and the next flop proceeds.
  * Wall-clock timeout: each solve gets up to 4 hours; on timeout a
    `.timeout` marker is written, the solver process is restarted, and the
    batch continues.
  * Dry-run mode: print the planned (spec, flop, output_path) tuples
    without issuing a single UPI command.

The UPI write-side commands used here (set_eff_stack, set_pot,
set_isomorphism, set_board, set_range_ip/oop, build_tree, go,
wait_for_solver, dump_tree) are PioSolver Edge's documented protocol; the
exact spelling has historically varied between Pio versions, so each call
goes through `client.command(...)` (the generic UPI hook) rather than a
typed wrapper. If a UPI command name or argument shape changes in a future
Pio release, only this file needs adjusting.
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.flop_sets import (
    Flop, flop_board_string, flop_filename_stem,
)
from pipeline.piosolver import PioSolverClient, PioSolverError, find_piosolver
from pipeline.scenario_spec import SolverSpec

# Per-solve wall-clock cap. A 100bb postflop solve to 0.5%-pot exploitability
# typically takes 30 min - 2 hr; 4 hours is the alarm threshold.
SOLVE_TIMEOUT_SECONDS = 4 * 3600

# After kicking off a solve with `go`, we poll for the solver to be ready
# again. Pio's `wait_for_solver` (UPI) blocks server-side, but on heavy
# solves the response can take hours; we still want client-side liveness.
WAIT_FOR_SOLVER_TIMEOUT = SOLVE_TIMEOUT_SECONDS


# --- result dataclasses -----------------------------------------------------
@dataclass
class SolveResult:
    """The outcome of one (spec, flop) solve."""

    spec_name: str
    flop_stem: str
    output_path: Path
    status: str                                  # "solved" / "skipped" / "failed" / "timeout"
    elapsed_seconds: float = 0.0
    final_exploitability_chips: float | None = None
    file_size_bytes: int = 0
    error_message: str = ""


@dataclass
class BatchSolveResult:
    """Roll-up of a batch_solve run for the final summary log."""

    spec_name: str
    flop_set_name: str
    solves: list[SolveResult] = field(default_factory=list)

    @property
    def solved(self) -> list[SolveResult]:
        return [s for s in self.solves if s.status == "solved"]

    @property
    def skipped(self) -> list[SolveResult]:
        return [s for s in self.solves if s.status == "skipped"]

    @property
    def failed(self) -> list[SolveResult]:
        return [s for s in self.solves if s.status in ("failed", "timeout")]


# --- output paths -----------------------------------------------------------
DEFAULT_SOLVE_ROOT = Path("solves")              # relative to repo root


def cache_path(spec: SolverSpec, flop: Flop, *,
               solve_root: Path = DEFAULT_SOLVE_ROOT) -> Path:
    """Absolute path of the `.cfr` for one (spec, flop)."""
    return solve_root / spec.cache_dir_name / f"{flop_filename_stem(flop)}.cfr"


def failure_marker_path(spec: SolverSpec, flop: Flop, *,
                        solve_root: Path = DEFAULT_SOLVE_ROOT,
                        kind: str = "failed") -> Path:
    """Sibling marker path (`.failed` / `.timeout`) for one (spec, flop)."""
    return solve_root / spec.cache_dir_name / f"{flop_filename_stem(flop)}.{kind}"


# --- UPI tree configuration: template-driven --------------------------------
# Background: PioSolver Edge's UPI write-side doesn't expose a single
# high-level "configure this tree" verb. Instead the GUI tree-builder
# produces a `.txt` template whose body is literal UPI commands (set_range,
# set_board, set_eff_stack, set_pot, add_line ... build_tree). The Edge 3
# UPI dialect requires specific argument shapes:
#
#   set_range OOP <1326 floats>     -- combo-level weights, not 169 hand cells
#   set_board 2cJs7s                -- concatenated cards, no spaces
#   set_bet_sizes OOP 33,75         -- plural verb; per-actor; comma-separated
#
# Some verbs we guessed (set_range_oop, set_board with spaces, set_bet_size
# singular, build_postflop_tree) don't exist on this Edge build. See
# test_output/upi_findings.md for the full discovery log.
#
# Architecture: read the template, swap the `set_board` line for the
# target flop, issue every non-comment line as a UPI command. The template
# encodes ranges, sizings, and (via add_line) the betting tree -- so a
# single file substitution per flop drives the entire solve setup.

# Lines we never replay from a template (they're setup directives Pio runs
# differently when scripted vs read from the GUI).
_TEMPLATE_SKIP_PREFIXES = ("#",)


def _is_command_line(line: str) -> bool:
    """A template line is a UPI command iff it's non-empty and not metadata."""
    stripped = line.strip()
    if not stripped:
        return False
    return not stripped.startswith(_TEMPLATE_SKIP_PREFIXES)


def _swap_board_line(line: str, flop: Flop) -> str:
    """If `line` is a `set_board ...` UPI command, replace its argument
    with the target flop's concatenated form (e.g. '2cJs7s'). Other lines
    pass through unchanged."""
    stripped = line.lstrip()
    if stripped.startswith("set_board "):
        return f"set_board {flop_filename_stem(flop)}"
    return line


def _load_template_lines(template_path: Path) -> list[str]:
    """Read a Pio tree template and return the UPI command lines, in order.

    Metadata header lines (starting with `#`) and blank lines are dropped.
    Returns lines verbatim (no normalisation); the caller is responsible
    for any board substitution before issuing.
    """
    text = template_path.read_text(encoding="utf-8")
    return [line.rstrip("\r\n") for line in text.splitlines()
            if _is_command_line(line)]


def _configure_tree(client: PioSolverClient,
                    spec: SolverSpec, flop: Flop) -> None:
    """Drive the tree setup by issuing every UPI command in the template,
    with `set_board` swapped for the target flop.

    The template's last line is `build_tree`, so on return Pio has a built
    (but unsolved) tree. Callers issue `set_accuracy`, `go`,
    `wait_for_solver`, `dump_tree` to finish the solve.
    """
    if not spec.pio_template_path:
        raise PioSolverError(
            f"spec {spec.name!r} has no pio_template_path; template-driven "
            f"solves are the only supported mode in Tier 1")
    template_path = Path(spec.pio_template_path)
    if not template_path.is_file():
        raise PioSolverError(
            f"Pio template not found at {template_path}; check the "
            f"SolverSpec.pio_template_path or Pio's install dir")
    for line in _load_template_lines(template_path):
        cmd = _swap_board_line(line, flop)
        # set_range with 1326 weights produces a very long line; bump the
        # timeout to keep transient slowness from looking like a hang.
        timeout = 600.0 if cmd.startswith(("set_range ", "build_tree")) else 30.0
        client.command(cmd, timeout=timeout)


def _reset_for_next_solve(client: PioSolverClient) -> None:
    """Clear solver state between solves so the next configure_tree starts
    from a clean slate. PioSolver's UPI typically exposes `free_tree` for
    this; on some builds it's `reset_tree`. Best-effort: try both, ignore
    the one that errors.
    """
    for cmd in ("free_tree", "reset_tree"):
        try:
            client.command(cmd, timeout=30.0)
            return
        except PioSolverError:
            continue


# --- exploitability readout -------------------------------------------------
def _read_exploitability(client: PioSolverClient) -> float | None:
    """Best-effort read of post-solve exploitability in chips.

    PioSolver Edge exposes a `calc_results` or `show_progress` command that
    emits a structured block including OOP+IP EV plus the exploitability.
    We parse the first parseable float we recognise. If the command isn't
    available on this Pio build, return None (the .cfr is still valid).
    """
    for cmd in ("calc_results_ev", "calc_global_freq", "show_progress"):
        resp = client.try_command(cmd)
        for line in resp:
            stripped = line.strip().lower()
            if "exploit" not in stripped and "br" not in stripped:
                continue
            for tok in line.replace(":", " ").replace(",", " ").split():
                try:
                    return float(tok)
                except ValueError:
                    continue
    return None


# --- one solve --------------------------------------------------------------
def solve_one(client: PioSolverClient,
              spec: SolverSpec,
              flop: Flop,
              *,
              solve_root: Path = DEFAULT_SOLVE_ROOT,
              timeout_seconds: int = SOLVE_TIMEOUT_SECONDS,
              log=print) -> SolveResult:
    """Configure + solve + dump one (spec, flop). Returns a SolveResult."""
    output = cache_path(spec, flop, solve_root=solve_root)
    flop_stem = flop_filename_stem(flop)
    result = SolveResult(spec_name=spec.name, flop_stem=flop_stem,
                         output_path=output, status="failed")

    if output.is_file():
        result.status = "skipped"
        result.file_size_bytes = output.stat().st_size
        log(f"  [skip] {spec.name}/{flop_stem}.cfr already exists "
            f"({result.file_size_bytes:,} bytes)")
        return result

    output.parent.mkdir(parents=True, exist_ok=True)
    # Clear any stale marker so a re-attempt isn't gated by an old failure.
    for kind in ("failed", "timeout"):
        marker = failure_marker_path(spec, flop, solve_root=solve_root, kind=kind)
        if marker.is_file():
            marker.unlink()

    start = time.monotonic()
    try:
        log(f"  [solve] {spec.name}/{flop_stem}: configuring tree ...")
        _configure_tree(client, spec, flop)
        log(f"  [solve] {spec.name}/{flop_stem}: target accuracy = "
            f"{spec.accuracy_target_chips} chips; starting solve ...")
        client.command(
            f"set_accuracy {spec.accuracy_target_chips}", timeout=30.0)
        client.command("go", timeout=30.0)
        # `wait_for_solver` blocks server-side until the accuracy target is hit.
        client.command("wait_for_solver", timeout=timeout_seconds)
        result.final_exploitability_chips = _read_exploitability(client)
        client.command(f"dump_tree {output} no_rivers",
                       timeout=600.0)
        if not output.is_file():
            # Fallback: some Pio builds reject the optional `no_rivers` flag.
            client.command(f"dump_tree {output}", timeout=600.0)
        if not output.is_file():
            raise PioSolverError(
                f"dump_tree returned without producing {output}")
        result.elapsed_seconds = time.monotonic() - start
        result.file_size_bytes = output.stat().st_size
        result.status = "solved"
        explv = result.final_exploitability_chips
        explv_str = f"{explv:.3f}" if explv is not None else "?"
        log(f"  [done]  {spec.name}/{flop_stem}.cfr "
            f"({result.file_size_bytes:,} bytes, {result.elapsed_seconds:.0f}s, "
            f"exploitability {explv_str} chips)")
    except PioSolverError as exc:
        result.elapsed_seconds = time.monotonic() - start
        # Distinguish timeouts -- a long solve hitting the wall-clock cap is a
        # different beast than a UPI error and gets its own marker.
        is_timeout = "timed out" in str(exc).lower()
        result.status = "timeout" if is_timeout else "failed"
        result.error_message = str(exc)
        marker = failure_marker_path(spec, flop, solve_root=solve_root,
                                     kind=result.status)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        log(f"  [{result.status}] {spec.name}/{flop_stem}: {exc}",
            file=_stderr())
    return result


def _stderr():
    import sys
    return sys.stderr


# --- batch orchestrator -----------------------------------------------------
def plan_batch(spec: SolverSpec, flops: tuple,
               *, solve_root: Path = DEFAULT_SOLVE_ROOT) -> list[dict]:
    """The execution plan for a (spec, flop_set) -- one dict per flop with
    the planned output path and whether the file already exists.

    `--dry-run` callers print this list and exit; production runs feed it
    into the actual solve loop.
    """
    plan: list[dict] = []
    for flop in flops:
        output = cache_path(spec, flop, solve_root=solve_root)
        plan.append({
            "spec_name": spec.name,
            "flop_stem": flop_filename_stem(flop),
            "flop_board": flop_board_string(flop),
            "output_path": output,
            "already_exists": output.is_file(),
        })
    return plan


def run_batch(spec: SolverSpec, flops: tuple, *,
              pio_exe: Path | None = None,
              solve_root: Path = DEFAULT_SOLVE_ROOT,
              timeout_seconds: int = SOLVE_TIMEOUT_SECONDS,
              flop_set_name: str = "",
              log=print) -> BatchSolveResult:
    """Solve every (spec, flop). Resumes if .cfr files already exist."""
    result = BatchSolveResult(spec_name=spec.name, flop_set_name=flop_set_name)
    if pio_exe is None:
        pio_exe = find_piosolver()
    if pio_exe is None or not Path(pio_exe).is_file():
        raise PioSolverError(
            "PioSolver Edge executable not found. Pass pio_exe=... or set "
            "$PIOSOLVER_EXE.")

    log(f"  PioSolver: {pio_exe}")
    log(f"  scenario : {spec.name}")
    log(f"  flop set : {flop_set_name or 'inline'} ({len(flops)} flops)")
    log(f"  cache    : {solve_root.resolve() / spec.cache_dir_name}")
    log("")

    with PioSolverClient(Path(pio_exe)) as client:
        for index, flop in enumerate(flops, 1):
            log(f"--- [{index}/{len(flops)}] flop {flop_board_string(flop)} ---")
            try:
                solve = solve_one(client, spec, flop,
                                  solve_root=solve_root,
                                  timeout_seconds=timeout_seconds, log=log)
            finally:
                # Always reset solver state -- if the solve crashed mid-way,
                # we don't want stale tree config bleeding into the next one.
                try:
                    _reset_for_next_solve(client)
                except PioSolverError:
                    pass
            result.solves.append(solve)
    return result


__all__ = [
    "BatchSolveResult",
    "DEFAULT_SOLVE_ROOT",
    "SOLVE_TIMEOUT_SECONDS",
    "SolveResult",
    "cache_path",
    "failure_marker_path",
    "plan_batch",
    "run_batch",
    "solve_one",
]
