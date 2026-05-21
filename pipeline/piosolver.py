"""Reusable PioSolver Edge client (UPI protocol).

Promoted from scripts/phase0_end_to_end_test.py, whose self-contained
`PioSolverClient` was always intended to become a shared module once a pipeline
layer needed it -- Layer 3 (Path Sampler) is that layer.

`PioSolverClient` launches PioSolver Edge as a subprocess and talks to it over
stdin/stdout. Every command is followed by an `is_ready` probe; the solver
echoes `is_ready ok!` when the prior command finished, giving a reliable
end-of-response sentinel.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from glob import glob
from pathlib import Path
from typing import Optional

HAND_COUNT = 1326                  # heads-up combos: C(52, 2) two-card hands
_EOF = object()                    # sentinel pushed when the solver process exits
_READY_ACKS = ("is_ready ok!", "is_ready")


class PioSolverError(RuntimeError):
    """Raised when the solver errors, exits, or fails to respond in time."""


class PioSolverClient:
    """Minimal UPI client driving a PioSolver process over stdin/stdout."""

    def __init__(self, exe_path: Path):
        self.exe_path = Path(exe_path)
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_q: "queue.Queue" = queue.Queue()
        self._stderr_lines: list[str] = []

    # -- lifecycle ------------------------------------------------------------
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
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(self.exe_path.parent),   # Pio resolves resources relative to cwd
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._write("is_ready")              # drain the startup banner
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

    # -- low-level I/O --------------------------------------------------------
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
                    + (f"\n--- solver stderr ---\n{err}" if err else ""))
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

    # -- typed UPI helpers ----------------------------------------------------
    def load_tree(self, cfr_path, mode: str = "auto", timeout: float = 900.0) -> None:
        resp = self.command(f"load_tree {cfr_path} {mode}", timeout=timeout)
        if not any("ok!" in line.lower() for line in resp):
            raise PioSolverError("load_tree failed:\n" + ("\n".join(resp) or "<no output>"))

    def show_node(self, node: str = "r") -> dict:
        """Node metadata: node_id, node_type, board, pot, children count, flags."""
        resp = [line for line in self.command(f"show_node {node}") if line.strip()]
        if not resp or any("error" in line.lower() for line in resp):
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
        """Child node ids of `node`, in solver order."""
        return [s for s in (line.strip() for line in self.command(f"show_children {node}"))
                if s.startswith("r:") and " " not in s]

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


# --- parsing helpers ---------------------------------------------------------
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


def find_piosolver(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the PioSolver Edge executable.

    Resolution order: explicit path, $PIOSOLVER_EXE, known install locations,
    then a top-level disk search. CustomBuilds / old-CPU variants are skipped --
    they are not the standard runnable build.
    """
    for candidate in (explicit, os.environ.get("PIOSOLVER_EXE")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    for known in (r"C:\PioSOLVER\PioSOLVER3-edge.exe",
                  r"C:\PioSOLVER\PioSOLVER2-edge.exe",
                  r"C:\Program Files\PioSOLVER\PioSOLVER3-edge.exe"):
        if Path(known).is_file():
            return Path(known)

    found: set[str] = set()
    for pattern in (r"C:\PioSOLVER*\*.exe",
                    r"C:\Program Files\PioSOLVER*\*.exe",
                    r"C:\Program Files (x86)\PioSOLVER*\*.exe"):
        found.update(glob(pattern))

    def score(path: str) -> int:
        name = Path(path).name.lower()
        if "pio" not in name or "custombuilds" in path.lower():
            return -1
        if any(bad in name for bad in ("viewer", "updater", "unins", "oldcpu", "newcpu")):
            return -1
        return (2 if "solver" in name else 0) + (3 if "edge" in name else 0)

    ranked = sorted((f for f in found if score(f) >= 0), key=score, reverse=True)
    return Path(ranked[0]) if ranked else None
