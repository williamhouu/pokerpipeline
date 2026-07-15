"""Deterministic audit for the CEV tournament preflop pack export.

Usage:
    python scripts/audit_cev_pack.py <path-to-CEV-root>

The root is the folder whose children are the per-depth packs
(``010bb/``, ``012.5bb/``, ... ``200bb/``). Re-run this on every new
export from the vendor; the July-2026 export FAILED it (see
``docs/cev_pack_audit.md``) on two counts: every all-in action file is
absent, and 139 sized-4-bet range files are missing at 50-100bb.

FORMAT SPEC (derived + verified July 2026; all 29,060 files conformed):

* Nested folders encode the action line; each path segment is a token
  ``<SEAT>_<VERB>[_<amount>]`` with SEAT in
  UTG/UTG1/LJ/HJ/CO/BTN/SB/BB (8-max) and VERB in R/C/X/3B/4B/5B.
* R, 3B, 4B, 5B amounts are raise-TO totals in bb. C amounts are chips
  ADDED by the caller (so ``SB_C_1.5`` matches a 2bb bet from the 0.5bb
  post). X is the BB check (no amount). Blinds are 0.5/1.0.
* FOLDS ARE IMPLICIT: every seat skipped between consecutive recorded
  actors folded. A node folder contains ``<its-own-name>.txt`` -- the
  per-hand frequency (0..1] of the ACTOR taking THAT action, conditional
  on reaching the node -- plus one folder/.txt per continuation. A hand
  absent from every sibling action file folds (or, in a fold-illegal
  spot, is missing data -- see check E).
* A ``.txt`` is ``class:freq`` comma-separated over the 169 hand classes;
  a bare class means 1.0.

CHECKS:
  A. every token parses; every file parses; weights in (0,1]; no dupes.
  B. rotation legality of every recorded line (implicit-fold walk).
  C. amount arithmetic: C == current_bet - committed (exact); raises
     strictly increase the bet and never exceed the stack.
  D. every node folder carries its own range file.
  E. mass conservation at fold-ILLEGAL nodes (BB facing a completed SB
     limp can only check or raise: per-hand sums must be ~1).
  F. all-in coverage: the max wager per depth should REACH the stack at
     short depths; a pack with no wager == stack has no jam branches.
  G. reach-weighted premium implied-fold: AA/KK/QQ/AKs that genuinely
     reach a node (product of the actor's own prior action freqs >= 5%)
     must not have > 5% unaccounted mass ("fold") at nodes where folding
     them is implausible; large counts indicate dropped action branches.

Exit code 0 = all checks clean; 1 = failures (printed).
"""
from __future__ import annotations

import collections
import os
import re
import sys

TOK = re.compile(r"^(UTG1|UTG|LJ|HJ|CO|BTN|SB|BB)_(R|C|X|3B|4B|5B)(?:_([0-9.]+))?$")
SEATS = ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
RANKS = "23456789TJQKA"
VALID_CLASSES: set[str] = set()
for _i, _r1 in enumerate(RANKS):
    for _j, _r2 in enumerate(RANKS):
        if _i == _j:
            VALID_CLASSES.add(_r1 + _r2)
        elif _i > _j:
            VALID_CLASSES.add(_r1 + _r2 + "s")
            VALID_CLASSES.add(_r1 + _r2 + "o")
EPS = 1e-6
PREMIUMS = ("AA", "KK", "QQ", "AKs")
REACH_FLOOR = 0.05      # a premium must reach the node this often to count
FOLD_LEAK_TOL = 0.05    # >5% unaccounted mass flags the node


def parse_token(name: str):
    m = TOK.match(name)
    return m.groups() if m else None


def load_range(path: str, problems) -> dict[str, float]:
    d: dict[str, float] = {}
    try:
        text = open(path).read().strip()
    except OSError as e:
        problems["unreadable"].append((path, str(e)))
        return d
    if not text:
        problems["empty_file"].append(path)
        return d
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            h, _, w_s = part.partition(":")
            try:
                w = float(w_s)
            except ValueError:
                problems["bad_weight"].append((path, part))
                continue
        else:
            h, w = part, 1.0
        if h not in VALID_CLASSES:
            problems["bad_class"].append((path, h))
            continue
        if w == 0.0:
            # Explicit zero = "this hand never takes this action": harmless
            # (and informative -- the mass lives in a sibling action file).
            problems["zero_weight_info"].append((path, h))
            continue
        if not 0.0 < w <= 1.0 + EPS:
            problems["weight_range"].append((path, h, w))
            continue
        if h in d:
            problems["dup_class"].append((path, h))
            continue
        d[h] = min(w, 1.0)
    return d


class RotationWalk:
    """Replay one recorded line, checking legality + amount arithmetic."""

    def __init__(self, stack: float):
        self.stack = stack
        self.committed = dict.fromkeys(SEATS, 0.0)
        self.committed["SB"] = 0.5
        self.committed["BB"] = 1.0
        self.active = list(SEATS)
        self.cursor = 0
        self.current_bet = 1.0
        self.errors: list[str] = []

    def _advance_to(self, seat: str) -> bool:
        if seat not in self.active:
            self.errors.append(f"actor {seat} already folded")
            return False
        guard = 0
        while self.active[self.cursor % len(self.active)] != seat:
            self.active.remove(self.active[self.cursor % len(self.active)])
            if self.cursor >= len(self.active):
                self.cursor = 0
            guard += 1
            if guard > len(SEATS) * 2:
                self.errors.append(f"unreachable actor {seat}")
                return False
        return True

    def apply(self, seat: str, verb: str, amt: float | None) -> None:
        if not self._advance_to(seat):
            return
        c = self.committed[seat]
        if verb == "C":
            need = self.current_bet - c
            if amt is None:
                self.errors.append(f"{seat} C missing amount")
            elif abs(amt - need) > 0.011:
                self.errors.append(
                    f"{seat} C_{amt:g} but owes {need:g} "
                    f"(bet {self.current_bet:g}, committed {c:g})"
                )
            self.committed[seat] = self.current_bet
        elif verb == "X":
            if abs(self.current_bet - c) > EPS:
                self.errors.append(f"{seat} X while facing {self.current_bet - c:g}")
        else:  # R / 3B / 4B / 5B raise TO amt
            if amt is None:
                self.errors.append(f"{seat} {verb} missing amount")
                return
            if amt <= self.current_bet + EPS:
                self.errors.append(
                    f"{seat} {verb}_{amt:g} not above bet {self.current_bet:g}"
                )
            if amt > self.stack + EPS:
                self.errors.append(f"{seat} {verb}_{amt:g} exceeds stack {self.stack:g}")
            self.current_bet = amt
            self.committed[seat] = amt
        self.cursor = (self.active.index(seat) + 1) % len(self.active)


def audit_depth(droot: str, stack: float) -> dict:
    problems: dict[str, list] = collections.defaultdict(list)
    ranges: dict[tuple, dict[str, float]] = {}
    nodes: dict[tuple, dict] = {}

    for dirpath, dirnames, filenames in os.walk(droot):
        rel = os.path.relpath(dirpath, droot)
        toks = tuple(rel.split(os.sep)) if rel != "." else ()
        for t in toks[-1:]:
            if parse_token(t) is None:
                problems["bad_dirname"].append(dirpath)
        own = (os.path.basename(dirpath) + ".txt") if toks else None
        if toks:
            nodes[toks] = {"has_own": own in filenames}
            if own in filenames:
                ranges[toks] = load_range(os.path.join(dirpath, own), problems)
        for f in filenames:
            if not f.endswith(".txt") or f == own:
                continue
            stem = f[:-4]
            if parse_token(stem) is None:
                problems["bad_filename"].append(os.path.join(dirpath, f))
                continue
            kt = toks + (stem,)
            nodes[kt] = {"has_own": True}
            ranges[kt] = load_range(os.path.join(dirpath, f), problems)

    max_wager = 0.0
    for toks in nodes:
        walk = RotationWalk(stack)
        for t in toks:
            parsed = parse_token(t)
            if parsed is None:
                break
            seat, verb, amt_s = parsed
            amt = float(amt_s) if amt_s else None
            walk.apply(seat, verb, amt)
            if amt is not None:
                max_wager = max(max_wager, amt)
        for e in walk.errors:
            problems["walk"].append(("/".join(toks), e))

    by_decision: dict[tuple, list[tuple]] = collections.defaultdict(list)
    for toks in nodes:
        parsed = parse_token(toks[-1])
        if parsed:
            by_decision[(toks[:-1], parsed[0])].append(toks)

    # E: fold-illegal conservation -- BB facing a completed limp (no raise
    # anywhere on the line) can only check or raise.
    fold_illegal_leaks = []
    for (parent, seat), options in by_decision.items():
        if seat != "BB":
            continue
        parsed_parent = [parse_token(t) for t in parent]
        if not parent or any(p is None or p[1] not in ("C",) for p in parsed_parent):
            continue  # only pure limp lines reach BB with no bet to call
        sums: dict[str, float] = collections.defaultdict(float)
        for o in options:
            for h, w in ranges.get(o, {}).items():
                sums[h] += w
        for h in VALID_CLASSES:
            if sums.get(h, 0.0) < 1.0 - FOLD_LEAK_TOL:
                fold_illegal_leaks.append(
                    ("/".join(parent), h, round(1.0 - sums.get(h, 0.0), 3))
                )

    # G: reach-weighted premium implied fold.
    prem_flags: collections.Counter = collections.Counter()
    prem_examples: dict[str, list] = collections.defaultdict(list)
    for (parent, seat), options in by_decision.items():
        prior = [
            parent[: i + 1]
            for i, t in enumerate(parent)
            if (p := parse_token(t)) and p[0] == seat
        ]
        sums = collections.defaultdict(float)
        for o in options:
            for h, w in ranges.get(o, {}).items():
                sums[h] += w
        for h in PREMIUMS:
            reach = 1.0
            for pt in prior:
                reach *= ranges.get(pt, {}).get(h, 0.0)
            if reach < REACH_FLOOR:
                continue
            leak = 1.0 - min(sums.get(h, 0.0), 1.0)
            if leak > FOLD_LEAK_TOL:
                prem_flags[h] += 1
                if len(prem_examples[h]) < 3:
                    prem_examples[h].append(
                        ("/".join(parent) or "(root)", seat, round(leak, 3))
                    )

    missing_own = ["/".join(t) for t, n in nodes.items() if not n["has_own"]]
    oversum = []
    for (parent, seat), options in by_decision.items():
        sums = collections.defaultdict(float)
        for o in options:
            for h, w in ranges.get(o, {}).items():
                sums[h] += w
        oversum.extend(
            ("/".join(parent) or "(root)", seat, h, round(s, 4))
            for h, s in sums.items()
            if s > 1.005
        )

    return {
        "problems": problems,
        "n_files": len(ranges),
        "n_decisions": len(by_decision),
        "max_wager": max_wager,
        "has_allin": max_wager >= stack - 0.01,
        "missing_own": missing_own,
        "fold_illegal_leaks": fold_illegal_leaks,
        "prem_flags": prem_flags,
        "prem_examples": prem_examples,
        "oversum": oversum,
    }


def main(root: str) -> int:
    failures = 0
    for depth in sorted(os.listdir(root)):
        droot = os.path.join(root, depth)
        if not os.path.isdir(droot) or not depth.endswith("bb"):
            continue
        stack = float(depth.removesuffix("bb"))
        r = audit_depth(droot, stack)
        p = r["problems"]
        issues = []
        for key in (
            "unreadable", "empty_file", "bad_weight", "bad_class",
            "weight_range", "dup_class", "bad_dirname", "bad_filename", "walk",
        ):
            if p[key]:
                issues.append(f"{key}={len(p[key])} e.g. {p[key][0]}")
        if r["oversum"]:
            issues.append(f"oversum={len(r['oversum'])} e.g. {r['oversum'][0]}")
        if r["missing_own"]:
            issues.append(
                f"missing-own-range-file={len(r['missing_own'])} "
                f"e.g. {r['missing_own'][0]}"
            )
        if r["fold_illegal_leaks"]:
            issues.append(
                f"fold-illegal-mass-leaks={len(r['fold_illegal_leaks'])} "
                f"e.g. {r['fold_illegal_leaks'][0]}"
            )
        if not r["has_allin"]:
            issues.append(
                f"NO ALL-IN BRANCH (max wager {r['max_wager']:g}bb "
                f"of {stack:g}bb stack)"
            )
        if r["prem_flags"]:
            ex = r["prem_examples"].get("AA") or next(iter(r["prem_examples"].values()))
            issues.append(
                f"premium-implied-fold nodes={dict(r['prem_flags'])} e.g. {ex[0]}"
            )
        status = "CLEAN" if not issues else "FAIL"
        zero_info = len(p["zero_weight_info"])
        print(
            f"{depth:>8} [{status}] files={r['n_files']} "
            f"decisions={r['n_decisions']}"
            + (f" (explicit-zero entries: {zero_info}, informational)" if zero_info else "")
        )
        for msg in issues:
            print(f"          - {msg}")
        failures += bool(issues)
    print(f"\n{'ALL CLEAN' if failures == 0 else f'{failures} depth(s) FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
