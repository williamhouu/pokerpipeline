#!/usr/bin/env python3
"""Phase 0 end-to-end test: load the BTN vs BB SRP test solve and verify the stack.

This is the Phase 0 deliverable from docs/engineering_brief.docx:

    "One end-to-end test solve: Pick BTN vs BB SRP, one flop texture. Run solve,
     save output, verify can be loaded and parsed by Python. Confirms the whole
     stack works before any pipeline code gets written."

The solve already exists (test_solves/btn_vs_bb_srp_2cJs7s.cfr -- BTN vs BB
single-raised pot, flop 2c Js 7s). This script drives PioSolver Edge over its UPI
(Universal Poker Interface) protocol -- launching it as a subprocess and talking
to it over stdin/stdout -- to:

  1. Load the .cfr save.
  2. Read tree metadata (board, effective stack, root node).
  3. Extract the canonical 1326-hand ordering.
  4. Extract OOP (BB) and IP (BTN) ranges at the root.
  5. Extract the root decision strategy and per-hand EVs.
  6. Sanity-check all of it and write a JSON dump of the solve.

It prints a PASS/FAIL report and exits non-zero if any check fails.

`PioSolverClient` is intentionally self-contained. When Layer 2 (Tree Resolver) is
built, promote it into a reusable `piosolver` package.

Usage:
    python scripts/phase0_end_to_end_test.py [--pio-exe PATH] [--cfr PATH]
                                             [--out PATH] [--load-mode MODE]

PioSolver is located from, in order: --pio-exe, the PIOSOLVER_EXE environment
variable, the DEFAULT_PIO_EXE constant below, then a search of common install dirs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFR = REPO_ROOT / "test_solves" / "btn_vs_bb_srp_2cJs7s.cfr"
DEFAULT_OUT = REPO_ROOT / "test_solves" / "phase0_dump.json"

# --- EDIT ME if the automatic search fails ------------------------------------
# Absolute path to the PioSolver Edge executable, e.g.
#   DEFAULT_PIO_EXE = r"C:\PioSOLVER\PioSOLVER3-edge.exe"
DEFAULT_PIO_EXE = ""
# ------------------------------------------------------------------------------

HAND_COUNT = 1326          # heads-up combos: C(52,2) two-card hands
_EOF = object()            # sentinel pushed onto the read queue when the solver exits
_READY_ACKS = ("is_ready ok!", "is_ready")


class PioSolverError(RuntimeError):
    """Raised when the solver errors, exits, or fails to respond in time."""


class PioSolverClient:
    """Minimal UPI client: drives a PioSolver process over stdin/stdout.

    Protocol: every command is followed by an `is_ready` probe. The solver echoes
    `is_ready ok!` once it has finished the prior command, which acts as a reliable
    end-of-response sentinel without depending on `set_end_string`.
    """

    def __init__(self, exe_path: Path):
        self.exe_path = Path(exe_path)
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_q: "queue.Queue" = queue.Queue()
        self._stderr_lines: list[str] = []

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self) -> "PioSolverClient":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def start(self) -> None:
        if not self.exe_path.is_file():
            raise PioSolverError(f"PioSolver executable not found: {self.exe_path}")
        self._proc = subprocess.Popen(
            [str(self.exe_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.exe_path.parent),  # Pio resolves resources relative to cwd
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        # Drain the startup banner: send one is_ready probe and wait for the ack.
        self._write("is_ready")
        self._read_until_ready(timeout=60.0, command="<startup handshake>")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                try:
                    self._write("exit")
                except (OSError, ValueError):
                    pass
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        finally:
            self._proc = None

    # -- low-level I/O ---------------------------------------------------------
    def _pump_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self._stdout_q.put(line.rstrip("\r\n"))
        self._stdout_q.put(_EOF)

    def _pump_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr_lines.append(line.rstrip("\r\n"))

    def _write(self, line: str) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _read_until_ready(self, timeout: float, command: str) -> list[str]:
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PioSolverError(f"timed out after {timeout:.0f}s waiting for: {command}")
            try:
                item = self._stdout_q.get(timeout=remaining)
            except queue.Empty:
                raise PioSolverError(f"timed out waiting for: {command}")
            if item is _EOF:
                err = "\n".join(self._stderr_lines[-10:])
                raise PioSolverError(
                    f"PioSolver exited while running: {command}"
                    + (f"\n--- solver stderr ---\n{err}" if err else "")
                )
            if item.strip() in _READY_ACKS:
                return lines
            lines.append(item)

    def command(self, cmd: str, timeout: float = 120.0) -> list[str]:
        """Send a UPI command and return its response lines (sentinel stripped)."""
        if self._proc is None or self._proc.poll() is not None:
            raise PioSolverError("solver process is not running")
        self._write(cmd)
        self._write("is_ready")
        return self._read_until_ready(timeout=timeout, command=cmd)

    # -- typed UPI helpers -----------------------------------------------------
    def load_tree(self, cfr_path: Path, mode: str = "auto", timeout: float = 900.0) -> None:
        resp = self.command(f"load_tree {cfr_path} {mode}", timeout=timeout)
        if not any("ok!" in line.lower() for line in resp):
            raise PioSolverError("load_tree failed:\n" + ("\n".join(resp) or "<no output>"))

    def show_node(self, node: str = "r") -> dict:
        resp = [line for line in self.command(f"show_node {node}") if line.strip()]
        if not resp or _looks_like_error(resp):
            raise PioSolverError(f"show_node {node} failed: " + " | ".join(resp))
        info: dict = {"raw": resp, "node_id": resp[0]}
        if len(resp) >= 5:
            info["node_type"] = resp[1]
            info["board"] = resp[2]
            info["pot"] = resp[3]
            try:
                info["children"] = int(resp[4].split()[0])
            except (ValueError, IndexError):
                info["children"] = None
        info["flags"] = resp[5] if len(resp) >= 6 else ""
        return info

    def show_children(self, node: str = "r") -> list[str]:
        """Return the node ids of the children of `node`, in solver order.

        `show_children` prints a `show_node`-format block per child; the child
        node ids are the only tokens shaped like `r:...` with no whitespace.
        """
        child_ids: list[str] = []
        for line in self.command(f"show_children {node}"):
            stripped = line.strip()
            if stripped.startswith("r:") and " " not in stripped:
                child_ids.append(stripped)
        return child_ids

    def first_decision_node(self, start: str = "r") -> tuple[str, dict]:
        """Descend from `start` through wrapper nodes to the first decision node.

        A flop solve's root is a `ROOT` node; the actual OOP decision sits below
        it. Follows the first child of each non-decision node until a node whose
        type contains `DEC` is reached.
        """
        node = start
        for _ in range(8):  # guard against an unexpectedly deep / cyclic tree
            info = self.show_node(node)
            if "DEC" in info.get("node_type", "").upper():
                return node, info
            children = self.show_children(node)
            if not children:
                raise PioSolverError(f"no decision node found descending from {start}")
            node = children[0]
        raise PioSolverError(f"no decision node within 8 levels of {start}")

    def show_hand_order(self) -> list[str]:
        for line in self.command("show_hand_order"):
            toks = line.split()
            if len(toks) == HAND_COUNT:
                return toks
        raise PioSolverError("show_hand_order did not return 1326 hands")

    def show_range(self, player: str, node: str = "r") -> list[float]:
        rows = _numeric_rows(self.command(f"show_range {player} {node}"))
        if not rows:
            raise PioSolverError(f"show_range {player} {node} returned no 1326-value row")
        return rows[0]

    def show_strategy(self, node: str = "r") -> list[list[float]]:
        rows = _numeric_rows(self.command(f"show_strategy {node}"))
        if not rows:
            raise PioSolverError(f"show_strategy {node} returned no 1326-value rows")
        return rows

    def calc_ev(self, player: str, node: str = "r") -> dict:
        rows = _numeric_rows(self.command(f"calc_ev {player} {node}"))
        if len(rows) < 2:
            raise PioSolverError(f"calc_ev {player} {node} expected 2 rows, got {len(rows)}")
        return {"ev": rows[0], "matchups": rows[1]}

    def show_effective_stack(self) -> Optional[int]:
        for line in self.command("show_effective_stack"):
            for tok in line.split():
                try:
                    return int(float(tok))
                except ValueError:
                    continue
        return None

    def try_command(self, cmd: str) -> list[str]:
        """Best-effort command; returns [] instead of raising (for optional info)."""
        try:
            return self.command(cmd)
        except PioSolverError:
            return []


# --- parsing helpers ----------------------------------------------------------
def _looks_like_error(lines: list[str]) -> bool:
    return any("error" in line.lower() for line in lines)


def _numeric_rows(lines: list[str], expected_len: int = HAND_COUNT) -> list[list[float]]:
    """Keep only response lines that are exactly `expected_len` parseable floats."""
    rows: list[list[float]] = []
    for line in lines:
        toks = line.split()
        if len(toks) != expected_len:
            continue
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            continue
    return rows


def _clean(values: list[float], ndigits: int) -> list[Optional[float]]:
    """Round values for the JSON dump; map non-finite (Pio's nan for out-of-range
    hands) to None so the output stays strict, parseable JSON."""
    return [round(v, ndigits) if math.isfinite(v) else None for v in values]


def _action_label(child_id: str) -> str:
    """Turn a child node id (e.g. 'r:0:b53') into a human label ('bet 53')."""
    seg = child_id.rsplit(":", 1)[-1]
    if seg in ("c", "C"):
        return "check/call"
    if seg in ("f", "F"):
        return "fold"
    if seg[:1] in ("b", "B") and seg[1:].replace(".", "").isdigit():
        return f"bet {seg[1:]}"
    if seg[:1] in ("r", "R") and seg[1:].replace(".", "").isdigit():
        return f"raise {seg[1:]}"
    return seg


def find_pio_exe(explicit: Optional[str]) -> Optional[Path]:
    """Resolve the PioSolver executable from flag / env / constant / disk search."""
    for candidate in (explicit, os.environ.get("PIOSOLVER_EXE"), DEFAULT_PIO_EXE or None):
        if candidate and Path(candidate).is_file():
            return Path(candidate)

    patterns = [
        r"C:\PioSOLVER*\*.exe",
        r"C:\PioSOLVER*\**\*.exe",
        r"C:\Program Files\PioSOLVER*\**\*.exe",
        r"C:\Program Files (x86)\PioSOLVER*\**\*.exe",
    ]
    found: set[str] = set()
    for pattern in patterns:
        found.update(glob(pattern, recursive=True))

    def score(path: str) -> int:
        name = Path(path).name.lower()
        if "pio" not in name:
            return -1
        s = 0
        if "solver" in name:
            s += 2
        if "edge" in name:
            s += 3
        if any(bad in name for bad in ("viewer", "updater", "unins")):
            s -= 100
        return s

    ranked = sorted((f for f in found if score(f) >= 0), key=score, reverse=True)
    return Path(ranked[0]) if ranked else None


# --- the test -----------------------------------------------------------------
def run_phase0_test(client: PioSolverClient, cfr_path: Path, load_mode: str,
                    out_path: Path) -> bool:
    """Run the end-to-end checks. Returns True iff every check passed."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> bool:
        checks.append((name, bool(passed), detail))
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}" + (f"  --  {detail}" if detail else ""))
        return bool(passed)

    print(f"\nLoading solve ({load_mode} mode): {cfr_path.name}")
    t0 = time.monotonic()
    client.load_tree(cfr_path, mode=load_mode)
    check("load_tree", True, f"loaded in {time.monotonic() - t0:.1f}s")

    eff_stack = client.show_effective_stack()
    root = client.show_node("r")
    print(f"\nRoot node: id={root['node_id']}  type={root.get('node_type', '?')}  "
          f"board={root.get('board', '?').strip()}  children={root.get('children')}")
    print(f"Effective stack: {eff_stack}")

    # A flop solve's root is a ROOT wrapper; descend to the real flop decision.
    dnode, dinfo = client.first_decision_node("r")
    print(f"Flop decision node: id={dnode}  type={dinfo.get('node_type', '?')}  "
          f"children={dinfo.get('children')}")
    check("located flop decision node", True,
          f"{dnode} ({dinfo.get('node_type', '?')})")

    hand_order = client.show_hand_order()
    check("show_hand_order returns 1326 hands", len(hand_order) == HAND_COUNT,
          f"{len(hand_order)} hands")

    oop_range = client.show_range("OOP", dnode)
    ip_range = client.show_range("IP", dnode)
    oop_sum, ip_sum = sum(oop_range), sum(ip_range)
    oop_in = sum(1 for w in oop_range if w > 1e-4)
    ip_in = sum(1 for w in ip_range if w > 1e-4)
    check("OOP (BB) range parsed, non-empty",
          len(oop_range) == HAND_COUNT and oop_sum > 0,
          f"{oop_in} combos in range, weight sum {oop_sum:.1f}")
    check("IP (BTN) range parsed, non-empty",
          len(ip_range) == HAND_COUNT and ip_sum > 0,
          f"{ip_in} combos in range, weight sum {ip_sum:.1f}")

    child_ids = client.show_children(dnode)
    strategy = client.show_strategy(dnode)
    shapes_ok = all(len(row) == HAND_COUNT for row in strategy)
    n_actions = len(strategy)
    check("show_strategy: one 1326-value row per action",
          shapes_ok and n_actions >= 2,
          f"{n_actions} actions x {HAND_COUNT} hands")
    if dinfo.get("children") is not None:
        check("strategy action count matches node children",
              n_actions == dinfo["children"],
              f"strategy={n_actions}, node says {dinfo['children']}")

    # Per-hand action frequencies must sum to ~1 for every in-range hand.
    bad = 0
    for i, w in enumerate(oop_range):
        if w <= 1e-4:
            continue
        total = sum(strategy[a][i] for a in range(n_actions))
        if not math.isclose(total, 1.0, abs_tol=0.02):
            bad += 1
    check("per-hand strategy frequencies sum to 1.0 (in-range hands)",
          bad == 0, f"{bad} of {oop_in} in-range combos off by >0.02")

    # Range-weighted aggregate strategy -- the interpretable sanity number.
    aggregate: list[float] = []
    if oop_sum > 0 and shapes_ok:
        aggregate = [
            sum(strategy[a][i] * oop_range[i] for i in range(HAND_COUNT)) / oop_sum
            for a in range(n_actions)
        ]
        labels = ([_action_label(cid) for cid in child_ids]
                  if len(child_ids) == n_actions
                  else [f"action {a}" for a in range(n_actions)])
        summary = " | ".join(f"{lab} {freq * 100:.1f}%"
                             for lab, freq in zip(labels, aggregate))
        print(f"\nBB flop strategy (range-weighted): {summary}")
        check("aggregate strategy sums to 100%",
              math.isclose(sum(aggregate), 1.0, abs_tol=0.01),
              f"sum={sum(aggregate) * 100:.1f}%")

    oop_ev = client.calc_ev("OOP", dnode)
    ip_ev = client.calc_ev("IP", dnode)

    def finite_in_range(ev_row: list[float], rng: list[float]) -> bool:
        return all(math.isfinite(ev_row[i]) for i in range(HAND_COUNT) if rng[i] > 1e-4)

    def weighted_mean_ev(ev_row: list[float], rng: list[float]) -> float:
        # Only in-range hands have a meaningful EV; out-of-range hands carry nan.
        num = den = 0.0
        for i in range(HAND_COUNT):
            if rng[i] > 1e-4 and math.isfinite(ev_row[i]):
                num += ev_row[i] * rng[i]
                den += rng[i]
        return num / den if den else 0.0

    oop_mean = weighted_mean_ev(oop_ev["ev"], oop_range)
    ip_mean = weighted_mean_ev(ip_ev["ev"], ip_range)
    check("calc_ev OOP: finite EV for every in-range combo",
          len(oop_ev["ev"]) == HAND_COUNT and finite_in_range(oop_ev["ev"], oop_range),
          f"range-weighted mean EV {oop_mean:.3f}")
    check("calc_ev IP: finite EV for every in-range combo",
          len(ip_ev["ev"]) == HAND_COUNT and finite_in_range(ip_ev["ev"], ip_range),
          f"range-weighted mean EV {ip_mean:.3f}")

    # JSON dump of the solve -- the Phase 0 "JSON dump of one solve" artifact.
    dump = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cfr_file": str(cfr_path),
        "load_mode": load_mode,
        "root": {k: v for k, v in root.items() if k != "raw"},
        "decision_node": {k: v for k, v in dinfo.items() if k != "raw"},
        "effective_stack": eff_stack,
        "tree_info": client.try_command("show_tree_info"),
        "hand_order": hand_order,
        "child_node_ids": child_ids,
        "action_labels": [_action_label(c) for c in child_ids],
        "oop_bb_range": _clean(oop_range, 6),
        "ip_btn_range": _clean(ip_range, 6),
        "decision_strategy": [_clean(row, 6) for row in strategy],
        "decision_strategy_aggregate": _clean(aggregate, 6),
        "oop_bb_ev": _clean(oop_ev["ev"], 4),
        "ip_btn_ev": _clean(ip_ev["ev"], 4),
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "all_passed": all(p for _, p, _ in checks),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"\nJSON dump written: {out_path}")

    passed = sum(1 for _, p, _ in checks if p)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("=" * 60)
    return dump["all_passed"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0 end-to-end test: load and parse the BTN vs BB SRP solve.",
        epilog="PioSolver path resolution order: --pio-exe, $PIOSOLVER_EXE, "
               "DEFAULT_PIO_EXE constant, then an automatic disk search.",
    )
    parser.add_argument("--pio-exe", help="path to the PioSolver Edge executable")
    parser.add_argument("--cfr", type=Path, default=DEFAULT_CFR,
                        help=f"path to the .cfr solve (default: {DEFAULT_CFR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"path for the JSON dump (default: {DEFAULT_OUT})")
    parser.add_argument("--load-mode", choices=["auto", "fast", "full"], default="auto",
                        help="PioSolver load_tree mode (default: auto)")
    args = parser.parse_args(argv)

    if not args.cfr.is_file():
        print(f"ERROR: solve file not found: {args.cfr}", file=sys.stderr)
        return 2

    pio_exe = find_pio_exe(args.pio_exe)
    if pio_exe is None:
        print("ERROR: could not locate the PioSolver executable.", file=sys.stderr)
        print("Fix one of the following, then re-run:", file=sys.stderr)
        print("  - pass --pio-exe \"C:\\path\\to\\PioSOLVER-edge.exe\"", file=sys.stderr)
        print("  - set the PIOSOLVER_EXE environment variable", file=sys.stderr)
        print("  - edit the DEFAULT_PIO_EXE constant in this script", file=sys.stderr)
        return 2

    print(f"PioSolver executable: {pio_exe}")
    try:
        with PioSolverClient(pio_exe) as client:
            ok = run_phase0_test(client, args.cfr, args.load_mode, args.out)
    except PioSolverError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
