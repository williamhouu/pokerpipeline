#!/usr/bin/env python
"""Re-runnable intake audit for the PLO 9-max Monker pack family (July 2026).

Written for the 100bb export; the 60bb sibling (Aug 2026) shares the grammar,
seats, rake and EV units, so it audits with the same checks -- pass
``--pack-dir plo9_60_ranges --stack-bb 60``.

The PLO analogue of ``scripts/audit_nlhe9_pack.py``: pulls REAL numbers from
the extracted ``.rng`` files and prints a prioritized PASS / WARN / FAIL
report over the things that decide integration go/no-go:

  [1] census + node-string grammar (every stem must decode with the 9-seat
      queue and the {0 fold, 1 call, 2 pot-raise, 3 all-in} token set)
  [2] file-format integrity (2 x 16,432 lines, parseable payloads, p in [0,1],
      per-hand action frequencies summing to ~1 across sibling files)
  [3] hand-order alignment -- THE silent-corruption risk. This export carries
      real hand labels (the 6-max pack shipped "????"), so every line can be
      checked against pipeline.plo.hand_order (label dialects differ:
      "(2A)AA" vs "AA(2A)" -- compared canonically, by rank multiset +
      suited-group partition)
  [4] EV-unit calibration (fold EVs are forced: SB fold = -1 sb = -1000
      milli-sb from hand start; a non-blind seat's fold = 0)
  [5] unambiguous-hand range sanity (AAKK double-suited never folds; rainbow
      low-pair trash never opens; UTG RFI width plausible under 9-max +
      5%/2bb rake)
  [6] EV-vs-frequency inversion scan on facing-one-open nodes -- the Monker
      QQ-fold bug class (a dominant action whose OWN counterfactual EV table
      says it sacrifices big EV)

    venv/bin/python scripts/audit_plo9_pack.py [--pack-dir plo9_ranges]

Read-only; safe to re-run any time (e.g. on a re-export).
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.hand_order import HAND_COUNT, combo_multiplicity, hand_order

# 9-max preflop acting order. Monker numbers seats by action order; blinds
# have posted and act last preflop.
SEATS9: tuple[str, ...] = (
    "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB",
)
# This export's token set (measured from the archive listing): no sized-raise
# tokens (the 6-max pack's "40100") -- pot-limit's one raise is the pot.
TOKENS = {"0": "fold", "1": "call", "2": "raise_pot", "3": "all_in"}
_CONTINUES = {"call", "raise_pot"}

_TOTAL_COMBOS = 270_725  # C(52, 4)


def _flag(ok: bool, warn: bool = False) -> str:
    return "  FAIL" if not ok and not warn else ("  WARN" if warn else "  PASS")


# --- grammar -----------------------------------------------------------------
def decode_stem(stem: str) -> list[tuple[str, str]]:
    """Decode a filename stem into ``[(seat, action), ...]`` with the 9-seat
    queue (caller/raiser rotates to the back; folder/jammer leaves)."""
    queue = list(SEATS9)
    out: list[tuple[str, str]] = []
    for token in stem.split("."):
        action = TOKENS.get(token)
        if action is None:
            raise ValueError(f"unknown token {token!r} in {stem!r}")
        if not queue:
            raise ValueError(f"action {token!r} but no seat left in {stem!r}")
        seat = queue.pop(0)
        out.append((seat, action))
        if action in _CONTINUES:
            queue.append(seat)
    return out


# --- label canonicalization ----------------------------------------------------
_GROUP_RE = re.compile(r"\(([2-9TJQKA]+)\)|([2-9TJQKA])")


def canon_label(label: str) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Canonical form of a Monker hand-class label: (sorted ranks, sorted
    suited groups). Dialect-independent: "AA(2A)" == "(2A)AA"; "2AAA" == "AAA2".
    """
    ranks: list[str] = []
    groups: list[tuple[str, ...]] = []
    for grouped, single in _GROUP_RE.findall(label):
        if grouped:
            groups.append(tuple(sorted(grouped)))
            ranks.extend(grouped)
        else:
            ranks.append(single)
    return tuple(sorted(ranks)), tuple(sorted(groups))


# --- file reading ---------------------------------------------------------------
def read_labeled(path: Path) -> tuple[list[str], list[tuple[float, float]]]:
    """(labels, [(p, ev_millisb), ...]) for one .rng file. Raises on bad shape."""
    lines = path.read_text(encoding="utf-8").split("\n")
    lines = [ln for ln in lines if ln]
    if len(lines) != 2 * HAND_COUNT:
        raise ValueError(f"{path.name}: {len(lines)} lines, expected {2 * HAND_COUNT}")
    labels, values = [], []
    for i in range(HAND_COUNT):
        labels.append(lines[2 * i])
        p_str, _, ev_str = lines[2 * i + 1].partition(";")
        values.append((float(p_str), float(ev_str) if ev_str else 0.0))
    return labels, values


def audit(pack_dir: Path, stack_bb: float = 100.0) -> int:  # noqa: C901, PLR0915 - a linear report
    fails = 0
    rng_root = None
    for f in sorted(pack_dir.rglob("*.rng")):
        rng_root = f.parent
        break
    if rng_root is None:
        print(f"no .rng files under {pack_dir}")
        return 1

    print("=" * 72)
    print(f"PLO 9-MAX PACK AUDIT ({stack_bb:g}bb): {rng_root}")
    print("=" * 72)

    stems = sorted(p.stem for p in rng_root.glob("*.rng"))
    order = hand_order()
    canon_order = [canon_label(lbl) for lbl in order]
    canon_to_idx = {c: i for i, c in enumerate(canon_order)}

    # -- [1] census + grammar ------------------------------------------------
    tokens = Counter(t for s in stems for t in s.split("."))
    decode_errors = 0
    active_ok_nodes = 0  # clean-line share: <=2 prior raises, <=3 players in
    raise_counts = Counter()
    for s in stems:
        try:
            seq = decode_stem(s)
        except ValueError:
            decode_errors += 1
            continue
        n_raises = sum(1 for _s, a in seq if a in ("raise_pot", "all_in"))
        raise_counts[min(n_raises, 4)] += 1
        folded = {seat for seat, a in seq if a == "fold"}
        jammed = {seat for seat, a in seq if a == "all_in"}
        in_hand = len(SEATS9) - len(folded) - len(jammed)
        if n_raises <= 2 and in_hand <= 3:
            active_ok_nodes += 1
    print(f"\n[1] CENSUS + GRAMMAR: {len(stems):,} node files")
    print(f"    token vocabulary = {dict(tokens)}")
    unknown = set(tokens) - set(TOKENS)
    print(f"    all tokens known ({sorted(TOKENS)}){_flag(not unknown)}: unknown={sorted(unknown)}")
    print(f"    every stem decodes with the 9-seat queue{_flag(decode_errors == 0)}: {decode_errors} errors")
    print(f"    raise-count mix: {dict(sorted(raise_counts.items()))} (4 = 4+)")
    print(f"    clean-line share (<=2 raises, <=3 players): {active_ok_nodes:,} "
          f"({active_ok_nodes / len(stems) * 100:.1f}%) -- the default generation pool")
    fails += bool(unknown) + (decode_errors > 0)

    # -- [2] format integrity (sampled) ---------------------------------------
    sample = random.Random(0).sample(stems, k=min(300, len(stems)))
    bad_shape = bad_p = 0
    for s in sample:
        try:
            _labels, values = read_labeled(rng_root / f"{s}.rng")
        except (ValueError, OSError):
            bad_shape += 1
            continue
        if any(not (-1e-9 <= p <= 1 + 1e-9) for p, _ in values):
            bad_p += 1
    print(f"\n[2] FORMAT (300 sampled files)")
    print(f"    2 x {HAND_COUNT} lines, parseable{_flag(bad_shape == 0)}: {bad_shape} bad files")
    print(f"    weights in [0, 1]{_flag(bad_p == 0)}: {bad_p} files out of range")
    fails += (bad_shape > 0) + (bad_p > 0)

    # frequency closure: across sibling action files, each hand's weights sum
    # to ~1 (every hand must do SOMETHING at a decision it reaches).
    def siblings(prefix: str) -> dict[str, list[tuple[float, float]]]:
        out = {}
        for t in TOKENS:
            f = rng_root / (f"{prefix}.{t}.rng" if prefix else f"{t}.rng")
            if f.is_file():
                out[t] = read_labeled(f)[1]
        return out

    utg1_files = siblings("2")  # UTG+1 facing the UTG open
    sums = [sum(vals[i][0] for vals in utg1_files.values()) for i in range(HAND_COUNT)]
    closure_bad = sum(1 for x in sums if abs(x - 1.0) > 0.02)
    print(f"    action-frequency closure at 'UTG+1 vs UTG open' (sum ~= 1)"
          f"{_flag(closure_bad == 0, warn=0 < closure_bad <= 20)}: {closure_bad} hands off")
    fails += closure_bad > 20

    # -- [3] hand-order alignment ----------------------------------------------
    print(f"\n[3] HAND-ORDER ALIGNMENT (labels vs pipeline.plo.hand_order)")
    misaligned = 0
    checked_files = 0
    for s in [stems[0], "2", stems[len(stems) // 2], stems[-1]]:
        try:
            labels, _values = read_labeled(rng_root / f"{s}.rng")
        except (ValueError, OSError):
            continue
        checked_files += 1
        for i, lbl in enumerate(labels):
            if canon_label(lbl) != canon_order[i]:
                misaligned += 1
    print(f"    {checked_files} files x {HAND_COUNT:,} labels canonically compared"
          f"{_flag(misaligned == 0)}: {misaligned} mismatches "
          "(0 required -- the reader keys by line index)")
    fails += misaligned > 0

    # -- [4] EV units (forced fold EVs) -----------------------------------------
    print(f"\n[4] EV UNITS (milli-sb from hand start, like the 6-max pack)")
    sb_fold = rng_root / ("0" * 15 + ".rng").replace("0" * 15, ".".join(["0"] * 8))
    sb_fold = rng_root / (".".join(["0"] * 8) + ".rng")   # 8 folds -> SB folds
    btn_fold = rng_root / (".".join(["0"] * 7) + ".rng")  # 7 folds -> BTN folds
    for name, path, want in (("SB fold = -1000", sb_fold, -1000.0),
                             ("BTN fold = 0", btn_fold, 0.0)):
        if not path.is_file():
            print(f"    {name}: file missing{_flag(False)}")
            fails += 1
            continue
        _l, values = read_labeled(path)
        evs = {ev for _p, ev in values}
        ok = evs == {want}
        print(f"    {name}{_flag(ok)}: distinct EVs = {sorted(evs)[:4]}")
        fails += not ok

    # -- [5] unambiguous-hand range sanity ---------------------------------------
    print(f"\n[5] RANGE SANITY (unambiguous hands + widths)")
    aakk = canon_to_idx[canon_label("(AK)(AK)")]   # AAKK double-suited: premium
    trash = canon_to_idx[canon_label("3322")]      # rainbow low pairs: pure trash

    utg = siblings("")
    if "2" in utg and "0" in utg:
        open_w = utg["2"]
        rfi = sum(open_w[i][0] * combo_multiplicity(i) for i in range(HAND_COUNT))
        rfi_pct = rfi / _TOTAL_COMBOS * 100
        ok_width = 3.0 <= rfi_pct <= 15.0
        print(f"    UTG RFI width = {rfi_pct:.1f}% of dealt hands"
              f"{_flag(ok_width)} (expect tight under 9-max + 5%/2bb rake)")
        prem_ok = open_w[aakk][0] > 0.95
        trash_ok = open_w[trash][0] < 0.02
        print(f"    UTG opens AAKK-ds {open_w[aakk][0] * 100:.0f}%{_flag(prem_ok)} "
              f"(want ~100), opens 3322-rainbow {open_w[trash][0] * 100:.1f}%"
              f"{_flag(trash_ok)} (want ~0)")
        fails += (not ok_width) + (not prem_ok) + (not trash_ok)
    else:
        print(f"    UTG root files missing{_flag(False)}")
        fails += 1

    bb_vs_utg = siblings("2." + ".".join(["0"] * 7))  # UTG opens, folds to BB
    if "0" in bb_vs_utg:
        bb_fold_prem = bb_vs_utg["0"][aakk][0]
        ok = bb_fold_prem < 0.02
        print(f"    BB folds AAKK-ds vs the UTG open {bb_fold_prem * 100:.1f}%"
              f"{_flag(ok)} (want ~0 -- the QQ-fold-bug tripwire)")
        fails += not ok
    else:
        print(f"    BB-vs-UTG-open files missing{_flag(False)}")
        fails += 1

    # -- [6] EV-vs-frequency inversion scan (the Monker QQ-fold bug class) -------
    # For every hand at a facing-one-open node: the dominant action's OWN
    # counterfactual EV should be near the max across offered actions. A hand
    # whose dominant choice sacrifices > 500 milli-sb (0.25bb) is an inversion;
    # a systematic cluster = the EV-collapse export artifact.
    print(f"\n[6] EV-VS-FREQUENCY INVERSIONS (facing one open; sacrifice > 0.25bb)")
    nodes = {
        "UTG+1 vs UTG open": "2",
        "BB vs UTG open": "2." + ".".join(["0"] * 7),
        "SB vs BTN open": ".".join(["0"] * 6) + ".2",
        "BTN vs CO open": ".".join(["0"] * 5) + ".2",
    }
    total_inversions = 0
    for name, prefix in nodes.items():
        sib = siblings(prefix)
        if len(sib) < 2:
            print(f"    {name}: files missing{_flag(False)}")
            fails += 1
            continue
        inversions = []
        for i in range(HAND_COUNT):
            mix = {t: sib[t][i][0] for t in sib}
            if sum(mix.values()) < 0.5:
                continue  # hand doesn't (meaningfully) reach / closure hole
            dom = max(mix, key=lambda t: mix[t])
            if mix[dom] < 0.5:
                continue
            evs = {t: sib[t][i][1] for t in sib}
            sacrifice = max(evs.values()) - evs[dom]
            if sacrifice > 500.0:
                inversions.append((order[i], dom, sacrifice, mix[dom]))
        total_inversions += len(inversions)
        worst = sorted(inversions, key=lambda x: -x[2])[:3]
        detail = "; ".join(
            f"{lbl} {TOKENS[d]}@{w * 100:.0f}% sacrifices {s / 1000:.2f}sb"
            for lbl, d, s, w in worst
        )
        ok = len(inversions) <= 20
        warn = 0 < len(inversions) <= 20
        print(f"    {name}: {len(inversions)} inversion hand-classes"
              f"{_flag(ok, warn=warn)}{('  worst: ' + detail) if worst else ''}")
        fails += not ok

    # -- [7] open size from pot math ---------------------------------------------
    # Token "2" is the pot raise. First-in, blinds only (no ante): the raise
    # is TO call(1bb) + pot-after-call(0.5 + 1 + 1) = 3.5bb -- independent of
    # stack depth. Verified two ways: this first-principles arithmetic AND the
    # shared pot-limit walk (resolve_pot_limit at this pack's stack_bb), which
    # every downstream consumer (prose, chart export) uses.
    print(f"\n[7] OPEN SIZE (token 2 = pot raise, {stack_bb:g}bb stacks)")
    from pipeline.plo.action_history import resolve_pot_limit
    from pipeline.plo.pack import SEATS_9MAX, parse_node_path

    expected_open = 1.0 + (0.5 + 1.0 + 1.0)  # call + pot after the call
    history = parse_node_path("2", seats=SEATS_9MAX)
    resolved, pot_after = resolve_pot_limit(history, stack_bb=stack_bb)
    open_to = resolved[-1].to_bb or 0.0
    ok = abs(open_to - expected_open) < 1e-9
    print(f"    UTG first-in pot raise resolves to {open_to:g}bb"
          f"{_flag(ok)} (first-principles pot math: {expected_open:g}bb; "
          f"pot after = {pot_after:g}bb)")
    fails += not ok

    print("\n" + "=" * 72)
    verdict = "CLEAN" if fails == 0 else f"{fails} HARD CHECK(S) FAILED"
    print(f"VERDICT: {verdict}"
          + ("  (see WARN lines for non-blocking items)" if fails == 0 else ""))
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack-dir", default="plo9_ranges", type=Path)
    ap.add_argument(
        "--stack-bb",
        default=100.0,
        type=float,
        help="effective stack depth of the export (60 for the 60bb sibling)",
    )
    args = ap.parse_args()
    raise SystemExit(audit(args.pack_dir, stack_bb=args.stack_bb))
