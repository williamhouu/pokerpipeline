"""Spot-check that the Ryan-ranges re-solve differs meaningfully from the
old (Pio-default-ranges) solve at the same flop.

Loads two .cfr files for the same flop (the old one from the pre-Ryan
backup, the new one from the re-solve) and compares:

  1. OOP range at the flop root (the conditional range entering the flop).
     The Ryan-pack expansion sums to ~37.3% of all hands; the Pio template
     placeholder is a different weighted distribution.
  2. IP range at the flop root.
  3. Strategy at the FIRST OOP decision node (BB's check/bet decision facing
     no action). Top 10 most-different hands printed.

The script logs differences but does not assert any specific magnitude --
strategy differences are sometimes large (hand classes that were in one
range and not the other) and sometimes small (overlapping linear regions).
The point is to make the differences visible so a human can confirm the
re-solve actually consumed the new ranges.

Run AFTER the batch_solve completes; Pio is single-instance under the Edge
license, so this conflicts with a running batch.

Usage:
    python scripts/spot_check_ryan_vs_old.py
    python scripts/spot_check_ryan_vs_old.py --flop 4d4sKh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.piosolver import PioSolverClient, find_piosolver               # noqa: E402

OLD_DIR = REPO_ROOT / "solves" / "Cash6max_100bb_BTN_open_BB_call_pre_ryan"
NEW_DIR = REPO_ROOT / "solves" / "Cash6max_100bb_BTN_open_BB_call"
DEFAULT_FLOP = "4d4sKh"                              # solve #2 in sorted order


def _weighted_sum(weights: list[float]) -> float:
    """Total weighted combos (one entry per combo, weight in [0,1])."""
    return sum(weights)


def _summarise_range(client: PioSolverClient, side: str) -> dict:
    """Capture the root-node range for one side as a small summary."""
    weights = client.show_range(side, node="r")
    total = _weighted_sum(weights)
    return {
        "side": side,
        "total_weighted_combos": total,
        "pct_of_all_hands": 100 * total / 1326,
        "n_nonzero": sum(1 for w in weights if w > 0.001),
        "n_full_weight": sum(1 for w in weights if w >= 0.999),
        "weights": weights,
    }


def _root_decision_node(client: PioSolverClient) -> str:
    """Return the node_id of the root's first OOP decision child.

    The flop root is OOP_DEC (BB-to-act). show_children at the root returns
    OOP_DEC's children (one per OOP action). The first such child IS the
    decision node we want strategy for -- actually, the root IS the OOP
    decision; we want strategy AT the root. Returns 'r'.
    """
    return "r"


def _format_diff_table(old_w: list[float], new_w: list[float],
                       label: str, head: int = 10) -> list[str]:
    """Top `head` weight-difference combos between two range vectors."""
    from pipeline.preflop_ranges import combo_label, HAND_COUNT
    diffs = [(abs(new_w[i] - old_w[i]), i, new_w[i] - old_w[i])
             for i in range(HAND_COUNT)]
    diffs.sort(reverse=True)
    lines = [f"\nTop {head} largest weight changes ({label}):",
             f"  {'combo':<8s}  {'old':>6s}  {'new':>6s}  {'delta':>7s}"]
    for _, i, delta in diffs[:head]:
        sign = "+" if delta >= 0 else "-"
        lines.append(f"  {combo_label(i):<8s}  {old_w[i]:>6.3f}  {new_w[i]:>6.3f}  "
                     f"{sign}{abs(delta):>5.3f}")
    return lines


def _compare_strategies(old_client: PioSolverClient, new_client: PioSolverClient,
                        node: str) -> list[str]:
    """Pretty-print strategy weights at a single decision node, both sides."""
    lines = [f"\nStrategy at node {node!r}:"]
    for label, client in (("OLD (Pio default)", old_client),
                          ("NEW (Ryan ranges)", new_client)):
        try:
            strategy = client.show_strategy(node=node)
        except Exception as exc:
            lines.append(f"  {label}: error reading show_strategy: {exc}")
            continue
        # show_strategy returns one row per action; row[i] = combo i's weight
        # for that action. We report the marginal action frequencies.
        if not strategy:
            lines.append(f"  {label}: empty")
            continue
        n_actions = len(strategy)
        # Action frequency = sum(action_weights) / 1326 (across all combos
        # including those weighted 0 in the range, which contribute 0).
        action_totals = [sum(row) for row in strategy]
        grand = sum(action_totals) or 1.0
        action_pcts = [100.0 * t / grand for t in action_totals]
        pcts_str = " | ".join(f"action{i}={p:.1f}%"
                              for i, p in enumerate(action_pcts))
        lines.append(f"  {label}: {n_actions} actions, mix = {pcts_str}")
    return lines


def spot_check(flop_stem: str) -> int:
    old_cfr = OLD_DIR / f"{flop_stem}.cfr"
    new_cfr = NEW_DIR / f"{flop_stem}.cfr"
    if not old_cfr.is_file():
        print(f"ERROR: old .cfr missing: {old_cfr}", file=sys.stderr)
        return 2
    if not new_cfr.is_file():
        print(f"ERROR: new .cfr missing (re-solve still running?): {new_cfr}",
              file=sys.stderr)
        return 2

    exe = find_piosolver()
    if exe is None:
        print("ERROR: PioSolver not found.", file=sys.stderr)
        return 2

    print(f"PioSolver : {exe}")
    print(f"OLD       : {old_cfr.relative_to(REPO_ROOT)}")
    print(f"NEW       : {new_cfr.relative_to(REPO_ROOT)}")
    print()

    # Two separate Pio sessions for clean state. Each loads its own .cfr,
    # we extract what we need, then close.
    print("Loading OLD ...")
    with PioSolverClient(exe) as old_client:
        old_client.load_tree(old_cfr)
        old_oop = _summarise_range(old_client, "OOP")
        old_ip = _summarise_range(old_client, "IP")
        old_strategy_lines: list[str] = []
        try:
            with PioSolverClient(exe) as new_client:
                # Wait, only one Pio instance allowed under license -- nest
                # would conflict. Sequential it is.
                pass
        except Exception:
            pass

    print("Loading NEW ...")
    with PioSolverClient(exe) as new_client:
        new_client.load_tree(new_cfr)
        new_oop = _summarise_range(new_client, "OOP")
        new_ip = _summarise_range(new_client, "IP")
        # Strategy comparison needs both sessions -- do new now while we have it.
        # We already captured old's ranges; for strategies we'll re-load old.
        new_strategy = new_client.try_command(f"show_strategy r")

    print("Loading OLD again for strategy ...")
    with PioSolverClient(exe) as old_client:
        old_client.load_tree(old_cfr)
        old_strategy = old_client.try_command(f"show_strategy r")

    print()
    print("=" * 72)
    print(f"FLOP {flop_stem} -- root-node range comparison")
    print("=" * 72)
    for label, old, new in (("OOP (BB)", old_oop, new_oop),
                            ("IP (BTN)", old_ip, new_ip)):
        print(f"  {label}:")
        print(f"    OLD: {old['total_weighted_combos']:>6.1f} combos "
              f"({old['pct_of_all_hands']:.1f}% of all hands), "
              f"{old['n_nonzero']} nonzero, {old['n_full_weight']} full")
        print(f"    NEW: {new['total_weighted_combos']:>6.1f} combos "
              f"({new['pct_of_all_hands']:.1f}% of all hands), "
              f"{new['n_nonzero']} nonzero, {new['n_full_weight']} full")
    print()

    # Diff tables.
    for label, old, new in (("OOP", old_oop, new_oop), ("IP", old_ip, new_ip)):
        for line in _format_diff_table(old["weights"], new["weights"], label):
            print(line)

    # Action mixes (root is the OOP decision node).
    def _action_pcts(strategy_lines: list[str]) -> str:
        """Parse show_strategy stdout into per-action weighted sums.

        Each emitted row has 1326 floats (one per combo) representing this
        action's frequency for that combo. The sum across the row is the
        weighted combo count taking that action; dividing by 1326 gives the
        marginal frequency.
        """
        rows: list[list[float]] = []
        for line in strategy_lines:
            toks = line.strip().split()
            if len(toks) != 1326:
                continue
            try:
                rows.append([float(t) for t in toks])
            except ValueError:
                continue
        if not rows:
            return "(no parseable strategy rows)"
        totals = [sum(r) for r in rows]
        grand = sum(totals) or 1.0
        return " | ".join(
            f"action{i}={100*t/grand:.1f}%" for i, t in enumerate(totals))

    print()
    print(f"Root-node OOP strategy (action mix, weighted):")
    print(f"  OLD: {_action_pcts(old_strategy)}")
    print(f"  NEW: {_action_pcts(new_strategy)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flop", default=DEFAULT_FLOP,
                        help=f"flop stem to compare (default {DEFAULT_FLOP})")
    args = parser.parse_args(argv)
    return spot_check(args.flop)


if __name__ == "__main__":
    sys.exit(main())
