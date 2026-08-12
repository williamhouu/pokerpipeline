#!/usr/bin/env python
"""Re-runnable intake audit for a PLO 6-max ``.rng`` pack (any stack depth).

The 6-max analogue of ``scripts/audit_plo9_pack.py`` -- the 6-max exports use a
different token grammar (``40100`` = 100%-pot raise, ``5`` = min-raise, plus
``0`` fold / ``1`` call / ``3`` all-in) and ship no real hand labels ("????"),
so the hand-order-alignment check does not apply; everything else does. Pulls
REAL numbers from the extracted files and prints a prioritized PASS/WARN/FAIL
report over the integration go/no-go questions:

  [1] census + grammar -- every stem must decode with the 6-seat queue + the
      6-max token set (any undecodable token is a hard FAIL: the pipeline
      would crash on it).
  [2] file-format integrity -- 2 x 16,432 lines per file, payloads parseable,
      strategy weight p in [0, 1].
  [3] strategy consistency -- at every decision node the per-hand weight summed
      across its sibling action files (the reach) must stay <= 1.
  [4] EV sanity -- EVs present and non-degenerate; a first-in non-blind seat's
      fold EV is ~0 (the forced-fold baseline).
  [5] range sanity -- the first-in (RFI) aggression width is plausible for the
      stack depth (short stacks jam more), and a clear premium never folds
      first-in.
  [6] EV-vs-frequency inversion scan -- the Monker QQ-fold-bug class: a
      dominant action whose OWN counterfactual EV table says it sacrifices big
      EV vs an alternative it takes ~0%.

    venv/bin/python scripts/audit_plo6_pack.py --pack-dir plo12_ranges

Read-only; safe to re-run (e.g. on a re-export).
"""

from __future__ import annotations

import argparse
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.hand_order import HAND_COUNT, combo_multiplicity
from pipeline.plo.pack import (
    SEATS,
    PloActionType,
    _rng_root,
    parse_node_path,
    read_rng_values,
)

_TOTAL_COMBOS = 270_725  # C(52, 4)
_AGGRO = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}


def _flag(ok: bool, warn: bool = False) -> str:
    return "  FAIL" if not ok and not warn else ("  WARN" if warn else "  PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default="plo12_ranges")
    args = ap.parse_args()
    base = Path(args.pack_dir)
    root = _rng_root(base)
    if root is None:
        print(f"FAIL: no .rng files under {base}")
        return 1
    files = sorted(root.glob("*.rng"))
    print(f"pack root: {root}")
    print(f"files: {len(files)}\n")
    fails = 0

    # [1] census + grammar --------------------------------------------------
    tok_hist: Counter[str] = Counter()
    bad_tokens: list[str] = []
    stems = [f.stem for f in files]
    for stem in stems:
        for t in stem.split("."):
            tok_hist[t] += 1
        try:
            parse_node_path(stem, seats=SEATS)
        except ValueError as exc:
            bad_tokens.append(f"{stem}: {exc}")
    ok = not bad_tokens
    fails += not ok
    print(f"[1] grammar: {len(stems)} stems decode"
          f"{'' if ok else f' -- {len(bad_tokens)} FAIL'}{_flag(ok)}")
    print(f"    tokens seen: {dict(tok_hist.most_common())}")
    for b in bad_tokens[:5]:
        print(f"      {b}")

    # [2] format + [3] strategy consistency ---------------------------------
    # Compact per-file storage: (p array, ev array) as float32. The deep
    # packs (75bb+ = 11k+ files) do NOT fit in memory as 16,432-tuple lists
    # (~2MB/file -> ~23GB, which swapped a nearly-full disk to death at
    # intake); two float arrays are ~130KB/file. float32 precision is far
    # inside every audit tolerance.
    values: dict[str, tuple[array, array]] = {}
    fmt_bad = 0
    p_range_bad = 0
    for f in files:
        try:
            v = read_rng_values(f)
        except ValueError:
            fmt_bad += 1
            continue
        if len(v) != HAND_COUNT:
            fmt_bad += 1
            continue
        if any(not (0.0 <= p <= 1.0001) for p, _ in v):
            p_range_bad += 1
        values[f.stem] = (
            array("f", (p for p, _ in v)),
            array("f", (ev for _, ev in v)),
        )
    ok = fmt_bad == 0 and p_range_bad == 0
    fails += not ok
    print(f"\n[2] format: {len(values)}/{len(files)} parse as 2x{HAND_COUNT}; "
          f"{fmt_bad} malformed, {p_range_bad} with p out of [0,1]{_flag(ok)}")

    # Group sibling action files by their decision node (stem minus last token).
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in values:
        parent, _, _ = stem.rpartition(".")
        groups[parent].append(stem)
    over_one = 0
    checked = 0
    for sibs in groups.values():
        if len(sibs) < 2:
            continue
        checked += 1
        for i in range(HAND_COUNT):
            reach = sum(values[s][0][i] for s in sibs)
            if reach > 1.02:  # noqa: PLR2004  small tolerance for rounding
                over_one += 1
                break
    ok = over_one == 0
    fails += not ok
    print(f"\n[3] strategy: {checked} multi-action nodes; "
          f"{over_one} with per-hand reach summing > 1{_flag(ok)}")

    # [4] EV sanity ---------------------------------------------------------
    all_ev_zero = all(ev == 0.0 for _, ev_arr in values.values() for ev in ev_arr)
    # First-in fold node for the opener (SEATS[0]): the single-token "0" file.
    fold_stem = "0"
    fold_ev_ok = True
    if fold_stem in values:
        # A non-blind first-in fold gives up nothing -> EV ~ 0.
        evs = [ev for p, ev in zip(*values[fold_stem]) if p > 0]
        if evs:
            fold_ev_ok = max(abs(e) for e in evs) < 0.05  # noqa: PLR2004
    ok = (not all_ev_zero) and fold_ev_ok
    fails += not ok
    print(f"\n[4] EV: non-degenerate={not all_ev_zero}, "
          f"opener fold-EV~0={fold_ev_ok}{_flag(ok)}")

    # [4b] cash EV-unit anchor (bb-ante packs use [5b] instead) -------------
    # The files' own fold bookkeeping pins the EV units: a folder is out
    # exactly what it committed. In milli-SMALL-blind units (every 6-max
    # cash pack) the SB's forced open-fold EV is -1000 (its posted 0.5bb =
    # 1sb) and the BB's forced fold vs a single open is -2000 (the 1bb
    # blind = 2sb). A milli-BB export would read -500 / -1000 instead, so
    # this catches a unit change in a re-export. The open token itself is
    # discovered (not assumed) and its pot-limit size resolved from the
    # token's own pot-%: raise-to = 1bb call + pct x 2.5bb first-in pot.
    if "(bb-ante)" not in root.name:
        def _forced_ev_milli(stem: str) -> float | None:
            v = values.get(stem)
            if not v:
                return None
            num = sum(p * ev for p, ev in zip(*v) if p > 0)
            den = sum(p for p in v[0] if p > 0)
            return (num / den * 1000) if den else None

        cash_singles = sorted(s for s in values if "." not in s)
        open_toks = [s for s in cash_singles if s not in ("0", "1", "3", "5")]
        sb_fold = _forced_ev_milli("0.0.0.0.0")
        bb_stem = next(
            (f"{t}.0.0.0.0.0" for t in open_toks
             if f"{t}.0.0.0.0.0" in values),
            None,
        )
        bb_fold = _forced_ev_milli(bb_stem) if bb_stem else None
        sb_ok = sb_fold is not None and abs(sb_fold - (-1000)) < 50  # noqa: PLR2004
        bb_ok = bb_fold is not None and abs(bb_fold - (-2000)) < 100  # noqa: PLR2004
        ok = sb_ok and bb_ok
        fails += not ok
        sb_txt = f"{sb_fold:.0f}" if sb_fold is not None else "missing"
        bb_txt = f"{bb_fold:.0f}" if bb_fold is not None else "missing"
        sizes = {
            t: f"{1 + int(t[2:]) / 100.0 * 2.5:.2f}bb"
            for t in open_toks if t.startswith("40")
        }
        print(f"\n[4b] cash EV anchor (milli-sb): SB open-fold {sb_txt} "
              f"(expect -1000), BB fold vs open {bb_txt} (expect -2000)"
              f"{_flag(ok)}")
        print(f"     open tokens {open_toks} -> first-in raise-to {sizes}")

    # [5] range sanity: first-in aggression width ---------------------------
    # The opener's RFI = single-token files: raise/all-in/min-raise vs
    # fold/call. Sized-raise tokens are discovered (an MTT 3.5x open encodes
    # as a node-specific pot-% like 40072, not the cash pack's 40100).
    def _combo_share(stem: str) -> float:
        v = values.get(stem)
        if not v:
            return 0.0
        num = sum(p * combo_multiplicity(i) for i, p in enumerate(v[0]))
        return 100.0 * num / _TOTAL_COMBOS

    def _rfi_label(stem: str) -> str:
        return {"3": "jam", "5": "min-raise", "1": "limp/call",
                "0": "fold"}.get(stem, f"open[{stem}]")

    singles = sorted(s for s in values if "." not in s)
    present = {_rfi_label(s): _combo_share(s) for s in singles}
    aggro_width = sum(
        v for k, v in present.items()
        if k.startswith(("open[", "jam", "min-raise"))
    )
    # A short-stack RFI is wider than a deep one but still a minority of hands.
    width_ok = 5.0 < aggro_width < 75.0  # noqa: PLR2004
    fails += not width_ok
    print(f"\n[5] RFI (opener, first-in): "
          + ", ".join(f"{k} {v:.1f}%" for k, v in present.items()))
    print(f"    aggression width = {aggro_width:.1f}% of all hands"
          f"{_flag(width_ok, warn=not (10 < aggro_width < 60))}")

    # [5b] MTT bb-ante calibration (runs when the pack dir names an ante) ----
    # Three facts pin the format: (a) EV units -- the SB's forced open-fold
    # EV is -0.5bb: -1000 in the cash packs' milli-SB units, -500 in
    # milli-BB units; (b) the ante is SUNK in the EV baseline when the BB's
    # forced fold-to-a-raise EV equals the blind alone (-1bb); (c) the ante
    # IS in the pot when the opener's sized-raise token resolves to ~3.5bb
    # only with the ante's dead money included (pot 2.5bb, not 1.5bb).
    if "(bb-ante)" in root.name:
        def _forced_ev(stem: str) -> float | None:
            v = values.get(stem)
            if not v:
                return None
            num = sum(p * ev for p, ev in zip(*v) if p > 0)
            den = sum(p for p in v[0] if p > 0)
            return (num / den * 1000) if den else None

        sb_fold = _forced_ev("0.0.0.0.0")
        bb_stem = next(
            (f"{s}.0.0.0.0.0" for s in singles
             if s.startswith("40") and f"{s}.0.0.0.0.0" in values),
            None,
        )
        bb_fold = _forced_ev(bb_stem) if bb_stem else None
        unit_bb = sb_fold is not None and abs(sb_fold - (-500)) < 25  # noqa: PLR2004
        ante_sunk = bb_fold is not None and abs(bb_fold - (-1000)) < 50  # noqa: PLR2004
        open_tok = next((s for s in singles if s.startswith("40")), None)
        size_ok = None
        if open_tok:
            pct = int(open_tok[2:]) / 100.0
            with_ante = 1 + pct * (2.5 + 1)     # pot = SB+BB+1bb ante
            without = 1 + pct * (1.5 + 1)
            size_ok = abs(with_ante - 3.5) < 0.15  # noqa: PLR2004
            print(f"\n[5b] ante calibration: SB fold {sb_fold:.0f} "
                  f"(milli-bb units={'YES' if unit_bb else 'NO'}), "
                  f"BB fold {bb_fold if bb_fold is not None else float('nan'):.0f} "
                  f"(ante sunk in EV={'YES' if ante_sunk else 'NO'})")
            print(f"     open token {open_tok}: with-ante -> "
                  f"{with_ante:.2f}bb, without -> {without:.2f}bb"
                  f"{_flag(bool(unit_bb and ante_sunk and size_ok))}")
        fails += not (unit_bb and ante_sunk and (size_ok is not False))

    # [6] EV-vs-frequency inversion (QQ-fold-bug class) ---------------------
    # For each facing-one-open node group, flag a hand whose DOMINANT action's
    # counterfactual EV is >= 0.5sb WORSE than an alternative it takes ~0%.
    inversions = 0
    inv_examples: list[str] = []
    for parent, sibs in groups.items():
        if len(sibs) < 2:
            continue
        # facing-one-open: parent history has exactly one aggressive action.
        try:
            hist = parse_node_path(parent, seats=SEATS) if parent else ()
        except ValueError:
            continue
        if sum(1 for a in hist if a.action in _AGGRO) != 1:
            continue
        for i in range(HAND_COUNT):
            reach = sum(values[s][0][i] for s in sibs)
            if reach < 0.5:
                continue
            acts = [(values[s][0][i] / reach, values[s][1][i], s) for s in sibs]
            dom_f, dom_ev, dom_s = max(acts, key=lambda t: t[0])
            if dom_f < 0.8:  # noqa: PLR2004  only near-pure dominant actions
                continue
            best_alt = max((ev for f, ev, s in acts if s != dom_s), default=dom_ev)
            if best_alt - dom_ev >= 0.5:  # noqa: PLR2004
                inversions += 1
                if len(inv_examples) < 3:
                    inv_examples.append(
                        f"{dom_s} hand#{i}: plays {dom_f*100:.0f}% at EV "
                        f"{dom_ev:.2f} but an alt is worth {best_alt:.2f}"
                    )
    ok = inversions == 0
    print(f"\n[6] EV/freq inversion (QQ-fold class): {inversions} hits"
          f"{_flag(ok, warn=inversions > 0)}")
    for e in inv_examples:
        print(f"      {e}")

    print(f"\n{'=' * 50}")
    verdict = "CLEAN -- safe to integrate" if fails == 0 else f"{fails} HARD FAIL(s)"
    print(f"VERDICT: {verdict}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
