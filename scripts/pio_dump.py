"""Dump a PioSolver .cfr save to plain-text files (single-file, stdlib-only).

Made to be sent to someone who has PioSolver on Windows ("run this next to
your solver, send back the zip"). It launches the solver exe, loads the
.cfr, walks the tree, and writes every dumped node's strategy/ranges/EVs
as text -- the .rng-equivalent the Pio ecosystem doesn't ship natively.

Usage (on the machine that has PioSolver)::

    python pio_dump.py "C:\\PioSOLVER\\PioSOLVER3-pro.exe" "F:\\Sims\\Qs7h2d.cfr"

It prints the path of a .zip when finished -- send that file back.
Options: --streets flop,turn,river (default flop), --no-ev, --max-nodes N.

Self-contained ON PURPOSE: this file is e-mailed around alone, so the UPI
client below is a trimmed copy of pipeline/piosolver.py rather than an
import. `--selftest` runs the whole dump against a built-in fake solver
(`--fake-solver`) so the I/O, parsing, file writing, and zipping are
exercised without a Pio installation; the only untested surface left is
the real solver's exact response phrasing, which is why every command and
response line also lands in dump.log inside the output folder.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HAND_COUNT = 1326
_EOF = object()
_READY_ACKS = ("is_ready ok!", "is_ready")


class PioError(RuntimeError):
    pass


class PioClient:
    """Minimal UPI client (trimmed copy of pipeline/piosolver.py)."""

    def __init__(self, argv: list[str], log) -> None:
        self.argv = argv
        self.log = log
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._stderr: list[str] = []

    def __enter__(self) -> PioClient:
        cwd = Path(self.argv[0]).parent if Path(self.argv[0]).is_file() else None
        self._proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._write("is_ready")
        self._read_until_ready(60.0, "<startup>")
        return self

    def __exit__(self, *_exc) -> None:
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

    def _pump_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self._q.put(line.rstrip("\r\n"))
        self._q.put(_EOF)

    def _pump_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip("\r\n"))

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
                raise PioError(f"timed out after {timeout:.0f}s on: {command}")
            try:
                item = self._q.get(timeout=remaining)
            except queue.Empty:
                raise PioError(f"timed out waiting for: {command}") from None
            if item is _EOF:
                err = "\n".join(self._stderr[-10:])
                raise PioError(
                    f"solver exited during: {command}"
                    + (f"\n--- stderr ---\n{err}" if err else "")
                )
            if item.strip() in _READY_ACKS:
                return lines
            lines.append(item)

    def command(self, cmd: str, timeout: float = 300.0) -> list[str]:
        if self._proc is None or self._proc.poll() is not None:
            raise PioError("solver process is not running")
        self.log(f">>> {cmd}")
        self._write(cmd)
        self._write("is_ready")
        lines = self._read_until_ready(timeout, cmd)
        for line in lines[:8]:
            self.log(f"    {line[:200]}")
        if len(lines) > 8:
            self.log(f"    ... ({len(lines)} lines total)")
        return lines

    def try_command(self, cmd: str, timeout: float = 60.0) -> list[str]:
        try:
            return self.command(cmd, timeout=timeout)
        except PioError as exc:
            self.log(f"    (optional command failed: {exc})")
            return []


def _numeric_rows(lines: list[str], n: int = HAND_COUNT) -> list[list[float]]:
    rows = []
    for line in lines:
        toks = line.split()
        if len(toks) != n:
            continue
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            continue
    return rows


def _fmt_row(row: list[float]) -> str:
    return " ".join(f"{v:.6g}" for v in row)


# --- the dump ----------------------------------------------------------------
def _street_of(board_line: str) -> str:
    n = len(board_line.split())
    return {3: "flop", 4: "turn", 5: "river"}.get(n, f"cards{n}")


def dump_tree(
    pio_argv: list[str],
    cfr_path: str,
    out_dir: Path,
    *,
    streets: set[str],
    with_ev: bool,
    max_nodes: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir = out_dir / "nodes"
    nodes_dir.mkdir(exist_ok=True)
    log_path = out_dir / "dump.log"
    log_file = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        log_file.write(msg + "\n")
        log_file.flush()

    manifest = (out_dir / "manifest.tsv").open("w", encoding="utf-8")
    manifest.write("node_id\tstreet\tnode_type\tpot\tn_children\tdumped\n")

    dumped = visited = 0
    with PioClient(pio_argv, log) as pio:
        resp = pio.command(f"load_tree {cfr_path}", timeout=1800.0)
        if not any("ok" in line.lower() for line in resp):
            raise PioError("load_tree failed:\n" + ("\n".join(resp) or "<no output>"))

        meta = out_dir / "meta.txt"
        with meta.open("w", encoding="utf-8") as m:
            m.write(f"cfr_file {cfr_path}\n")
            m.write(f"dumped_at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for cmd in ("show_tree_info", "show_effective_stack"):
                for line in pio.try_command(cmd):
                    m.write(f"{cmd} {line}\n")
            order = pio.command("show_hand_order")
            for line in order:
                if len(line.split()) == HAND_COUNT:
                    m.write(f"hand_order {line}\n")
                    break
            else:
                raise PioError("show_hand_order did not return 1326 hands")

        # Breadth-first walk from the root. Streets are detected from each
        # node's board length; nodes on non-requested streets are recorded
        # in the manifest (structure) but their payloads are skipped, and
        # the walk does not descend past them.
        todo: list[str] = ["r:0"]
        seen: set[str] = set()
        while todo and visited < max_nodes:
            node_id = todo.pop(0)
            if node_id in seen:
                continue
            seen.add(node_id)
            visited += 1
            try:
                info = [ln for ln in pio.command(f"show_node {node_id}") if ln.strip()]
            except PioError as exc:
                log(f"!!! show_node {node_id} failed: {exc}")
                continue
            node_type = info[1] if len(info) > 1 else "?"
            board = info[2] if len(info) > 2 else ""
            pot = info[3] if len(info) > 3 else ""
            street = _street_of(board)
            children = [
                s for s in (ln.strip() for ln in pio.command(f"show_children {node_id}"))
                if s.startswith("r:") and " " not in s
            ]
            on_street = street in streets
            is_decision = "DEC" in node_type.upper() or node_type.upper() in (
                "IP", "OOP", "IP_DEC", "OOP_DEC",
            )
            manifest.write(
                f"{node_id}\t{street}\t{node_type}\t{pot}\t{len(children)}\t"
                f"{int(on_street and is_decision)}\n"
            )

            if on_street:
                todo.extend(children)
            if not (on_street and is_decision):
                continue

            safe = node_id.replace(":", "_")
            with (nodes_dir / f"{safe}.txt").open("w", encoding="utf-8") as f:
                f.write(f"node_id {node_id}\n")
                f.write(f"node_type {node_type}\nboard {board}\npot {pot}\n")
                f.write("children " + " ".join(children) + "\n")
                strat = _numeric_rows(pio.command(f"show_strategy {node_id}"))
                # Deliberately not strict: a row-count mismatch is recorded
                # as a warning line below instead of aborting the node.
                for child, row in zip(children, strat, strict=False):
                    f.write(f"strategy {child}\n{_fmt_row(row)}\n")
                if len(strat) != len(children):
                    f.write(f"# warning: {len(strat)} strategy rows for "
                            f"{len(children)} children\n")
                for player in ("OOP", "IP"):
                    rng = _numeric_rows(pio.command(f"show_range {player} {node_id}"))
                    if rng:
                        f.write(f"range {player}\n{_fmt_row(rng[0])}\n")
                if with_ev:
                    for player in ("OOP", "IP"):
                        rows = _numeric_rows(
                            pio.try_command(f"calc_ev {player} {node_id}", timeout=600.0)
                        )
                        if len(rows) >= 2:
                            f.write(f"ev {player}\n{_fmt_row(rows[0])}\n")
                            f.write(f"matchups {player}\n{_fmt_row(rows[1])}\n")
            dumped += 1
            if dumped % 25 == 0:
                print(f"  ... {dumped} nodes dumped ({visited} visited)")

    manifest.close()
    log_file.close()
    if visited >= max_nodes:
        print(f"  note: stopped at the --max-nodes cap ({max_nodes}).")
    print(f"  dumped {dumped} decision nodes ({visited} visited).")
    zip_base = str(out_dir)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=out_dir)
    return Path(zip_path)


# --- built-in fake solver (for --selftest) -------------------------------------
def _fake_solver() -> None:
    """Reads UPI commands on stdin, answers with canned-but-plausible data."""
    hands = []
    ranks = "23456789TJQKA"
    suits = "cdhs"
    cards = [r + s for r in ranks for s in suits]
    for i in range(52):
        for j in range(i + 1, 52):
            hands.append(cards[j] + cards[i])
    row = " ".join("0.5" for _ in range(HAND_COUNT))
    children = {"r:0": ["r:0:c", "r:0:b18"], "r:0:c": [], "r:0:b18": []}
    types = {"r:0": "OOP_DEC", "r:0:c": "IP_DEC", "r:0:b18": "IP_DEC"}
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "exit":
            return
        if cmd == "is_ready":
            print("is_ready ok!", flush=True)
            continue
        if cmd.startswith("load_tree"):
            print("load_tree ok!", flush=True)
        elif cmd == "show_hand_order":
            print(" ".join(hands), flush=True)
        elif cmd.startswith("show_node"):
            node = cmd.split()[1]
            print(node, flush=True)
            print(types.get(node, "IP_DEC"), flush=True)
            print("Qs 7h 2d", flush=True)
            print("0 0 55", flush=True)
            print(f"{len(children.get(node, []))} children", flush=True)
        elif cmd.startswith("show_children"):
            for ch in children.get(cmd.split()[1], []):
                print(ch, flush=True)
        elif cmd.startswith(("show_strategy",)):
            node = cmd.split()[1]
            for _ in children.get(node, []):
                print(row, flush=True)
        elif cmd.startswith(("show_range", "calc_ev")):
            print(row, flush=True)
            if cmd.startswith("calc_ev"):
                print(row, flush=True)
        # unknown commands fall through silently; is_ready still acks


def _selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dump_test"
        zip_path = dump_tree(
            [sys.executable, str(Path(__file__).resolve()), "--fake-solver"],
            "fake.cfr",
            out,
            streets={"flop"},
            with_ev=True,
            max_nodes=50,
        )
        node_files = sorted(p.name for p in (out / "nodes").glob("*.txt"))
        text = (out / "nodes" / "r_0.txt").read_text(encoding="utf-8")
        ok = (
            zip_path.is_file()
            and node_files == ["r_0.txt", "r_0_b18.txt", "r_0_c.txt"]
            and "strategy r:0:c" in text
            and "range OOP" in text
            and "ev IP" in text
            and (out / "meta.txt").read_text(encoding="utf-8").count("hand_order") == 1
        )
        print("SELFTEST", "PASSED" if ok else "FAILED")
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Dump a PioSolver .cfr to plain text. "
        "Example: python pio_dump.py \"C:\\PioSOLVER\\PioSOLVER3-pro.exe\" "
        "\"F:\\Sims\\Qs7h2d.cfr\""
    )
    ap.add_argument("pio_exe", nargs="?", help="path to the PioSolver .exe")
    ap.add_argument("cfr_file", nargs="?", help="path to the .cfr to dump")
    ap.add_argument("--streets", default="flop",
                    help="comma list of streets to dump fully (default: flop)")
    ap.add_argument("--no-ev", action="store_true",
                    help="skip per-hand EVs (faster)")
    ap.add_argument("--max-nodes", type=int, default=5000)
    ap.add_argument("--fake-solver", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.fake_solver:
        _fake_solver()
        return 0
    if args.selftest:
        return _selftest()
    if not args.pio_exe or not args.cfr_file:
        ap.error("need both the PioSolver exe path and the .cfr path")
    if not Path(args.pio_exe).is_file():
        ap.error(f"solver exe not found: {args.pio_exe}")
    if not Path(args.cfr_file).is_file():
        ap.error(f".cfr not found: {args.cfr_file}")

    out_dir = Path.cwd() / f"pio_dump_{Path(args.cfr_file).stem}"
    print(f"Loading {args.cfr_file} (a 1GB file can take a few minutes)...")
    zip_path = dump_tree(
        [args.pio_exe],
        args.cfr_file,
        out_dir,
        streets={s.strip() for s in args.streets.split(",") if s.strip()},
        with_ev=not args.no_ev,
        max_nodes=args.max_nodes,
    )
    print("\nDONE. Send back this file:")
    print(f"  {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
