#!/usr/bin/env python3
"""inspect_spot.py -- read one hand out of the Phase 0 solve dump, poker-style.

Given a specific hand (e.g. "AhAd" or "7s2h"), this prints, in plain language:

  * whether the hand is in BB's range at the flop decision spot, and its weight;
  * the GTO strategy for the hand (how often to check vs bet);
  * the hand's EV in chips at the spot.

It reads test_solves/phase0_dump.json directly -- the data was already extracted
from PioSolver by scripts/phase0_end_to_end_test.py, so PioSolver is not launched.

Two requested fields are flagged as not-yet-available and explained in the closing
NOTES (per-action EV split, equity vs BTN's range); see that section for why.

Usage:
    python scripts/inspect_spot.py AhAd
    python scripts/inspect_spot.py AhAd KhQh 7s2h        # several at once
    python scripts/inspect_spot.py 7s2h --dump path/to/other_dump.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUMP = REPO_ROOT / "test_solves" / "phase0_dump.json"

RANKS = "23456789TJQKA"           # low -> high
SUITS = "cdhs"
RANK_WORDS = {
    "2": "deuces", "3": "threes", "4": "fours", "5": "fives", "6": "sixes",
    "7": "sevens", "8": "eights", "9": "nines", "T": "tens", "J": "jacks",
    "Q": "queens", "K": "kings", "A": "aces",
}
IN_RANGE_EPS = 1e-4               # range weight at or below this == not in range


# --- card / hand parsing ------------------------------------------------------
def parse_card(token: str) -> str:
    """Normalise a card to '<RANK><suit>' form, e.g. 'aH' -> 'Ah'. Raises on junk."""
    token = token.strip().replace("10", "T")
    if len(token) != 2:
        raise ValueError(f"not a card: {token!r}")
    rank, suit = token[0].upper(), token[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"not a card: {token!r}")
    return rank + suit


def parse_hand(text: str) -> tuple[str, str]:
    """Parse a 4-char hand string ('AhAd') into two normalised, distinct cards."""
    raw = text.strip().replace("10", "T")
    if len(raw) != 4:
        raise ValueError(f"expected a 2-card hand like 'AhKs', got {text!r}")
    c1, c2 = parse_card(raw[:2]), parse_card(raw[2:])
    if c1 == c2:
        raise ValueError(f"{text!r} repeats the same card")
    return c1, c2


def hand_label(c1: str, c2: str) -> str:
    """A human name for the hand: 'pocket aces', 'K-Q suited', '7-2 offsuit'."""
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    if r1 == r2:
        return f"pocket {RANK_WORDS[r1]}"
    hi, lo = sorted((r1, r2), key=RANKS.index, reverse=True)
    return f"{hi}-{lo} {'suited' if s1 == s2 else 'offsuit'}"


def find_hand_index(hand_order: list[str], c1: str, c2: str) -> Optional[int]:
    """Index of this hand in the solver's 1326-hand ordering, ignoring card order."""
    wanted = {c1, c2}
    for i, token in enumerate(hand_order):
        if {token[:2], token[2:]} == wanted:
            return i
    return None


# --- formatting helpers -------------------------------------------------------
def para(text: str, indent: str = "   ") -> str:
    return textwrap.fill(text, width=66, initial_indent=indent, subsequent_indent=indent)


def action_sort_key(label: str):
    """Order actions check/fold first, then bets/raises by size."""
    group = 0 if ("check" in label or "fold" in label) else 1
    nums = [int(t) for t in label.replace("/", " ").split() if t.isdigit()]
    return (group, nums[0] if nums else 0, label)


def spot_name(cfr_file: str) -> str:
    """Turn 'btn_vs_bb_srp_2cJs7s.cfr' into 'BTN vs BB, single-raised pot'."""
    stem = Path(cfr_file).stem.lower()
    pot = ("3-bet pot" if "_3b" in stem or stem.endswith("3b")
           else "single-raised pot" if "srp" in stem else "pot")
    if "btn" in stem and "_vs_bb" in stem:
        return f"BTN vs BB, {pot}"
    return f"{Path(cfr_file).stem} ({pot})"


# --- the report ---------------------------------------------------------------
def print_header(dump: dict, dump_path: Path) -> None:
    board = dump["decision_node"]["board"].strip()
    pot = dump["decision_node"]["pot"].split()[-1]
    aggregate = dump.get("decision_strategy_aggregate") or dump.get("root_strategy_aggregate", [])
    labels = dump.get("action_labels", [])
    agg_pairs = sorted(zip(labels, aggregate), key=lambda p: action_sort_key(p[0]))
    agg_text = "   ".join(f"{lab} {freq * 100:.1f}%" for lab, freq in agg_pairs)

    print("=" * 68)
    print(f" SPOT:   {spot_name(dump['cfr_file'])}")
    print(f" BOARD:  {board}        BB acts first on the flop")
    print(f" POT:    {pot} chips        Effective stacks: {dump['effective_stack']} chips")
    print(f" SOURCE: {dump_path.name}  (solve dumped {dump['generated_at'][:10]})")
    print()
    print(f" BB's overall flop strategy here:   {agg_text}")
    print("=" * 68)


def print_hand_report(dump: dict, hand_text: str) -> None:
    hand_order = dump["hand_order"]
    oop_range = dump["oop_bb_range"]
    oop_ev = dump["oop_bb_ev"]
    strategy = dump.get("decision_strategy") or dump["root_strategy"]
    labels = dump.get("action_labels") or [f"action {i}" for i in range(len(strategy))]
    board = {parse_card(c) for c in dump["decision_node"]["board"].split()}

    c1, c2 = parse_hand(hand_text)
    idx = find_hand_index(hand_order, c1, c2)

    print()
    print("-" * 68)
    print(f" {c1} {c2}   ({hand_label(c1, c2)})")
    print("-" * 68)

    if idx is None:
        print(para(f"{c1} {c2} is not a recognised combo in the solver's hand "
                   f"list. Check the input."))
        return

    on_board = sorted(board & {c1, c2})
    weight = oop_range[idx] or 0.0
    in_range = weight > IN_RANGE_EPS

    # 1. In BB's range? -------------------------------------------------------
    if in_range:
        print(f" In BB's range here?   YES   (range weight {weight:.3f})")
        if weight >= 0.999:
            print(para("BB always carries this exact combo to this flop."))
        else:
            pct = weight * 100
            print(para(f"BB reaches this flop holding exactly {c1} {c2} about "
                       f"{pct:.0f}% of the time it is dealt -- it 3-bets or folds "
                       f"the rest of this combo pre-flop, so only a fraction of it "
                       f"shows up here."))
    else:
        print(f" In BB's range here?   NO   (range weight {weight:.3f})")
        if on_board:
            board_str = dump["decision_node"]["board"].strip()
            print(para(f"This combo cannot occur at this spot: {', '.join(on_board)} "
                       f"{'is' if len(on_board) == 1 else 'are'} on the board "
                       f"({board_str})."))
        print(para("BB never brings this hand to this flop -- it is folded (or "
                    "never continued) pre-flop, so it is absent from BB's range."))
        print()
        print(" How to play it / EV:   n/a -- hand is not in BB's range.")
        return

    # 2. GTO strategy ---------------------------------------------------------
    print()
    print(f" How to play {c1} {c2} (GTO):")
    pairs = sorted(zip(labels, strategy), key=lambda p: action_sort_key(p[0]))
    width = max(len(lab) for lab, _ in pairs)
    for label, row in pairs:
        freq = (row[idx] or 0.0) * 100
        dots = "." * (width + 3 - len(label))
        print(f"   {label} {dots}  {freq:5.1f}%")

    # 3. EV at this spot ------------------------------------------------------
    ev = oop_ev[idx]
    print()
    if ev is not None and math.isfinite(ev):
        print(f" EV of {c1} {c2} at this spot:   {ev:+.1f} chips")
        print(para("(the chip value of holding this hand here, played GTO)"))
    else:
        print(f" EV of {c1} {c2} at this spot:   n/a")

    # 4. requested-but-not-in-dump fields ------------------------------------
    print()
    print(" Equity vs BTN's range:   not in this dump   (see NOTES)")
    print(" Per-action EV split:     not in this dump   (see NOTES)")


def print_notes() -> None:
    print()
    print("=" * 68)
    print(" NOTES -- two requested fields are not in phase0_dump.json yet")
    print("-" * 68)
    print(para("Per-action EV split: the dump stores each hand's EV at the "
               "decision node (one number, the GTO value of the spot), not its "
               "EV for check vs bet separately. Per-action EVs come from running "
               "calc_ev at each child node (r:0:c and r:0:b36). Adding that to "
               "the Phase 0 extraction would make this field available.", indent=" "))
    print()
    print(para("Equity vs BTN's range: showdown equity needs a poker hand "
               "evaluator that rolls out the turn and river against BTN's range. "
               "The engineering brief scopes this as its own component (the "
               "validator's equity calculator). Better built once, properly, "
               "than approximated in an exploration script.", indent=" "))
    print("=" * 68)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect how a specific hand plays at the Phase 0 solve spot.")
    parser.add_argument("hands", nargs="+", metavar="HAND",
                        help="one or more hands, e.g. AhAd KhQh 7s2h")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP,
                        help=f"path to the solve dump (default: {DEFAULT_DUMP})")
    args = parser.parse_args(argv)

    if not args.dump.is_file():
        print(f"ERROR: dump not found: {args.dump}", file=sys.stderr)
        print("Run scripts/phase0_end_to_end_test.py first to create it.",
              file=sys.stderr)
        return 2
    dump = json.loads(args.dump.read_text(encoding="utf-8"))

    print_header(dump, args.dump)
    for hand in args.hands:
        try:
            print_hand_report(dump, hand)
        except ValueError as exc:
            print()
            print("-" * 68)
            print(f" {hand}")
            print("-" * 68)
            print(para(f"Could not read this hand: {exc}"))
    print_notes()
    return 0


if __name__ == "__main__":
    sys.exit(main())
