#!/usr/bin/env python3
"""Standalone reproduction: exported-strategy vs exported-EV consistency.

Self-contained on purpose (stdlib only, no local imports) so it can be run
against any of the SQLite postflop exports with just:

    python3 vendor_repro_ev_vs_strategy.py /path/to/solve.db
    python3 vendor_repro_ev_vs_strategy.py /path/to/solve.db --dump-node "r:0:c:c:3s:b2312:c:2h:b4968"

What it checks
--------------
On a river facing-bet node a call is a pure showdown, so the file's own EV
tables imply an exact equity for every combo of the player facing the bet:

    implied_equity = (EV_call - EV_fold + to_call) / (pot_after_bet + to_call - rake)

where ``rake`` is the rake taken from the final pot on a win (parsed from the
file's ``rake`` metadata, e.g. "10% cap 3bb"): the EV tables are net of rake
while showdown equity is rake-blind, so an uncorrected denominator reads a
phantom NEGATIVE bias that is largest in small pots (~-3 to -5 points on a
checkdown-river stab at 10% rake) and negligible in big pots (the cap).

That number must match the exact showdown equity of the same combo against
the betting range reconstructed from the file's OWN frequency tables
(preflop range x the conditional freq blobs along the line, board-masked).
If the two disagree systematically, the strategy tables and the EV tables
describe two different solutions.

It prints the per-node median gap (implied minus exact, in equity points),
the pooled median, and the share of nodes past 3 points. A converged,
consistently-exported file measures ~1-2 points of noise.

File-format assumptions (all verified against the v7/v8 exports):
  * tables: gto_postflop(node, action, freq_blob, ev_blob_oop, ev_blob_ip),
    hand_index(idx, hand), metadata(key, value), preflop_ranges(player, hand, weight)
  * freq_blob: 1326 bytes, 0-255 conditional strategy per combo
  * ev_blob_oop / ev_blob_ip: 1326 float32 LE, EV in chips for OOP / IP
  * node grammar: "r:0" root (OOP first), ":c" passive (check or call),
    ":b<chips>" bet/raise to <chips> CUMULATIVE chips committed by that player
    across the WHOLE postflop line (Pio node-string semantics, NOT a fresh
    per-street amount; e.g. a river b4968 after b2312:c is a 4968-2312 = 2656
    chip bet, and an all-in token equals eff_stack exactly), ":f" fold,
    ":<card>" chance card -> next street
  * the passive action's LABEL (CHECK vs CALL) varies between exports, so
    check-vs-call is derived from the betting state, never the label.

Nodes whose reconstructed betting range holds fewer than --min-betting-combos
combos (default 6) are skipped: EVs at barely-reached nodes are the least
trained part of a solve and score as noise, not as an export defect.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import statistics
import struct
import sys
from collections import defaultdict
from itertools import combinations

CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
RANKS = "23456789TJQKA"
REACH_EPS = 1e-4


# --- 7-card evaluator (rank tuples compare correctly between hands) ---------
def _rank5(cards):
    """Rank a 5-card hand as a comparable tuple. cards: [(rank_int, suit), ...]"""
    rs = sorted((c[0] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    counts = defaultdict(int)
    for r in rs:
        counts[r] += 1
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    shape = tuple(n for _r, n in groups)
    order = tuple(r for r, _n in groups)
    uniq = sorted(counts, reverse=True)
    straight_hi = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_hi = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:  # A2345 (wheel)
            straight_hi = 3
    if flush and straight_hi:
        return (8, straight_hi)
    if shape == (4, 1):
        return (7,) + order
    if shape == (3, 2):
        return (6,) + order
    if flush:
        return (5,) + tuple(rs)
    if straight_hi:
        return (4, straight_hi)
    if shape == (3, 1, 1):
        return (3,) + order
    if shape == (2, 2, 1):
        return (2,) + order
    if shape == (2, 1, 1, 1):
        return (1,) + order
    return (0,) + tuple(rs)


def rank7(card_strs):
    """Best 5-card rank from 7 cards given as 2-char strings like 'As'."""
    cards = [(RANKS.index(c[0]), c[1]) for c in card_strs]
    return max(_rank5(list(c)) for c in combinations(cards, 5))


# --- db reading --------------------------------------------------------------
def split_cards(s):
    return [s[i:i + 2] for i in range(0, len(s), 2)]


def node_tokens(node_id):
    return node_id.split(":")[2:]


def node_street_idx(node_id):
    return sum(1 for t in node_tokens(node_id) if CARD_RE.match(t))


class Solve:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.meta = dict(self.con.execute("SELECT key, value FROM metadata"))
        self.idx_to_hand = {}
        self.hand_to_idx = {}
        for idx, hand in self.con.execute("SELECT idx, hand FROM hand_index"):
            self.idx_to_hand[idx] = hand
            self.hand_to_idx[hand] = idx
        self.n = len(self.idx_to_hand)
        self._cache = {}

    def actions(self, node_id):
        if node_id in self._cache:
            return self._cache[node_id]
        rows = self.con.execute(
            "SELECT action, freq_blob, ev_blob_oop, ev_blob_ip "
            "FROM gto_postflop WHERE node=?", (node_id,)
        ).fetchall()
        out = [
            (a, list(fb),
             struct.unpack(f"<{len(eo) // 4}f", eo),
             struct.unpack(f"<{len(ei) // 4}f", ei))
            for a, fb, eo, ei in rows
        ]
        self._cache[node_id] = out
        return out

    def preflop_weights(self, player):
        return {
            h: float(w) for h, w in self.con.execute(
                "SELECT hand, weight FROM preflop_ranges WHERE player=?", (player,)
            )
        }

    def bb_chips(self):
        for ck, bk in (("pot", "pot_bb"), ("eff_stack", "eff_stack_bb")):
            try:
                chips, in_bb = float(self.meta[ck]), float(self.meta[bk])
                if chips and in_bb:
                    return round(chips / in_bb)
            except (KeyError, TypeError, ValueError):
                pass
        return 100.0  # every export to date uses 100 chips/bb

    def rake_chips(self, final_pot):
        """Rake taken from ``final_pot`` chips on a showdown win, per the
        file's ``rake`` metadata ("8% cap 2bb" / "10% cap 3bb (300 chips)").
        0 when absent/unparseable (the check then runs uncorrected)."""
        m = re.match(r"\s*([\d.]+)\s*%\s*cap\s*([\d.]+)\s*bb", self.meta.get("rake", ""))
        if not m:
            return 0.0
        pct, cap_bb = float(m.group(1)) / 100.0, float(m.group(2))
        return min(pct * final_pot, cap_bb * self.bb_chips())


def walk(s, node_id, oop="BB", ip="BTN"):
    """Walk a node string: betting state + both sides' reach at the node.

    Returns (pot_chips_incl_bet, to_call_chips, actor, reach_by_side, board)."""
    flop = split_cards(s.meta["flop"])
    board_set = set(flop)
    pre = {oop: s.preflop_weights("BB"), ip: s.preflop_weights("BTN")}
    reach = {
        side: [
            (pre[side].get(s.idx_to_hand[i], 0.0)
             if not (set(split_cards(s.idx_to_hand[i])) & board_set) else 0.0)
            for i in range(s.n)
        ]
        for side in (oop, ip)
    }
    pot = float(s.meta["pot"])
    eff = float(s.meta["eff_stack"])
    # committed = each side's CUMULATIVE postflop chips for the whole line (the
    # quantity the b<chips> tokens state -- it never resets at a street change).
    # Both sides' totals are equal whenever a street starts, so to_call computed
    # on cumulative totals is exact within a street.
    committed = {oop: 0.0, ip: 0.0}
    behind = {oop: eff, ip: eff}
    to_act, other = oop, ip
    board = list(flop)

    cur = "r:0"
    for token in node_tokens(node_id):
        if CARD_RE.match(token):
            board.append(token)
            to_act, other = oop, ip
            for side in (oop, ip):
                r = reach[side]
                for i in range(s.n):
                    if r[i] and token in split_cards(s.idx_to_hand[i]):
                        r[i] = 0.0
            cur += ":" + token
            continue

        parent = s.actions(cur)
        if token.startswith("b"):
            db_name = "BET_" + token[1:]
        elif token == "f":
            db_name = "FOLD"
        else:  # "c" -> the single passive action, whatever it was labelled
            passive = [a for a, *_ in parent if a in ("CHECK", "CALL")]
            if len(passive) != 1:
                raise ValueError(f"{cur}: expected one passive action, got {passive}")
            db_name = passive[0]
        pa = next((row for row in parent if row[0] == db_name), None)
        if pa is None:
            raise ValueError(f"{cur}: no action {db_name!r} for token {token!r}")

        freqs = pa[1]
        for i in range(s.n):
            total = sum(row[1][i] for row in parent)
            reach[to_act][i] *= (freqs[i] / total) if total > 0 else 0.0

        to_call_now = max(committed.values()) - committed[to_act]
        if token.startswith("b"):
            # The token is the actor's cumulative committed-for-the-line total
            # after this action; the fresh wager subtracts EVERYTHING already
            # committed, earlier streets included. (An all-in token equals
            # eff_stack exactly, so the min() clamp is a no-op on well-formed
            # files.)
            size = float(token[1:])
            added = min(size - committed[to_act], behind[to_act])
            pot += added
            committed[to_act] += added
            behind[to_act] -= added
        elif token == "c" and to_call_now > 0:
            pot += to_call_now
            committed[to_act] += to_call_now
            behind[to_act] -= to_call_now

        to_act, other = other, to_act
        cur += ":" + token

    to_call = max(committed.values()) - committed[to_act]
    return pot, to_call, to_act, reach, board


def stride_sample(items, k):
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


# Skip nodes whose reconstructed betting range holds fewer combos than this:
# EVs at barely-reached nodes are the least-trained part of a solve and score
# as noise rather than as an export defect.
MIN_BETTING_COMBOS = 6
# ... and fewer combo-EQUIVALENTS of total probability mass than this. A deep
# raise line can hold 57 distinct combos each at microscopic weight (total
# ~0.01 of one combo) -- still an effectively unreached line whose EVs are
# untrained. Measured July 2026 on the 8h6h5s 200bb file: nodes with mass
# >= 0.08 score ~0.0 pts; every over-3pt node sat at mass <= 0.04.
MIN_BETTING_MASS = 0.25


def analyse_node(s, nid, oop, ip, max_combos, dump=False, min_combos=MIN_BETTING_COMBOS,
                 stats=None):
    """Per-combo implied-vs-exact gaps at one river facing-bet node, or None."""
    pot, to_call, actor, reach, board = walk(s, nid, oop, ip)
    if to_call <= 0:
        return None
    villain_side = ip if actor == oop else oop
    acts = {row[0]: row for row in s.actions(nid)}
    passive = [acts[k] for k in ("CALL", "CHECK") if k in acts]
    if len(passive) != 1 or "FOLD" not in acts:
        return None
    call_row, fold_row = passive[0], acts["FOLD"]
    ev_slot = 2 if actor == oop else 3  # ev_blob_oop vs ev_blob_ip
    call_ev, fold_ev = call_row[ev_slot], fold_row[ev_slot]
    call_f, fold_f = call_row[1], fold_row[1]

    villain = []
    for i in range(s.n):
        w = reach[villain_side][i]
        if w > REACH_EPS:
            cards = split_cards(s.idx_to_hand[i])
            villain.append((set(cards), rank7(cards + board), w))
    vmass = sum(w for _c, _r, w in villain)
    if len(villain) < min_combos or vmass < MIN_BETTING_MASS:
        # low-reach node (too few combos OR too little mass): EVs are noise
        if stats is not None:
            stats["low_reach_skipped"] = stats.get("low_reach_skipped", 0) + 1
        if dump and villain:
            print(f"\nnode {nid}\n  skipped: {len(villain)} combos / "
                  f"{vmass:.2f} combo-equivalents of mass in the reconstructed "
                  f"betting range (min {min_combos} / {MIN_BETTING_MASS})")
        return None

    heroes = sorted(
        s.idx_to_hand[i] for i in range(s.n) if reach[actor][i] > REACH_EPS
    )
    # EVs are net of rake; showdown equity is rake-blind. Correct the
    # denominator by the rake the winner pays on the final pot, or the check
    # reads a phantom negative bias in small pots.
    rake = s.rake_chips(pot + to_call)
    rows = []
    for h in stride_sample(heroes, max_combos):
        i = s.hand_to_idx[h]
        if call_f[i] + fold_f[i] <= 0:
            continue
        cards = split_cards(h)
        hero_rank = rank7(cards + board)
        blocked = set(cards)
        win = tie = tot = 0.0
        for vcards, vrank, w in villain:
            if vcards & blocked:
                continue
            tot += w
            if hero_rank > vrank:
                win += w
            elif hero_rank == vrank:
                tie += w
        if tot <= 0:
            continue
        exact = (win + 0.5 * tie) / tot
        implied = (call_ev[i] - fold_ev[i] + to_call) / (pot + to_call - rake)
        total_f = call_f[i] + fold_f[i]
        rows.append((h, implied, exact, call_f[i] / total_f))
    if not rows:
        return None
    if dump:
        price = to_call / (pot + to_call)
        print(f"\nnode {nid}")
        print(f"  pot after bet = {pot:g} chips, to_call = {to_call:g} chips, "
              f"call price = {price * 100:.1f}%, rake = {rake:g} chips, actor = {actor}")
        print(f"  {'combo':6s} {'implied':>8s} {'exact':>8s} {'gap':>7s} {'call%':>6s}")
        for h, implied, exact, cf in sorted(rows, key=lambda r: -abs(r[1] - r[2])):
            print(f"  {h:6s} {implied * 100:7.1f}% {exact * 100:7.1f}% "
                  f"{(implied - exact) * 100:+6.1f} {cf * 100:5.0f}%")
    return [implied - exact for _h, implied, exact, _cf in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db")
    ap.add_argument("--max-nodes", type=int, default=24)
    ap.add_argument("--max-combos-per-node", type=int, default=60)
    ap.add_argument("--min-betting-combos", type=int, default=MIN_BETTING_COMBOS,
                    help="skip nodes whose reconstructed betting range has fewer "
                         "combos than this (low-reach nodes score as noise)")
    ap.add_argument("--dump-node", default=None,
                    help="print per-combo implied/exact detail for one node id")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip the (slow-ish) file SHA-256")
    args = ap.parse_args()

    if not args.no_hash:
        h = hashlib.sha256()
        with open(args.db, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        print(f"sha256  {h.hexdigest()}")
    s = Solve(args.db)
    for k in ("spot_name", "spot", "flop", "solve_date", "solve_seconds", "accuracy"):
        if k in s.meta:
            print(f"{k:14s}= {s.meta[k]}")
    oop, ip = "BB", "BTN"

    if args.dump_node:
        gaps = analyse_node(s, args.dump_node, oop, ip,
                            args.max_combos_per_node, dump=True,
                            min_combos=args.min_betting_combos)
        if gaps is None:
            print("  (not a scoreable river facing-bet node)")
        else:
            print(f"  node median gap = {statistics.median(gaps) * 100:+.1f} points"
                  f" over {len(gaps)} combos")
        return 0

    fold_nodes = {r[0] for r in s.con.execute(
        "SELECT DISTINCT node FROM gto_postflop WHERE action='FOLD'")}
    candidates = sorted(n for n in fold_nodes if node_street_idx(n) >= 2)
    if not candidates:
        print("no river facing-bet nodes in this file")
        return 0

    node_medians = []
    pooled = []
    stats: dict = {}
    for nid in stride_sample(candidates, args.max_nodes):
        gaps = analyse_node(s, nid, oop, ip, args.max_combos_per_node,
                            min_combos=args.min_betting_combos, stats=stats)
        if gaps:
            node_medians.append((nid, statistics.median(gaps), len(gaps)))
            pooled.extend(gaps)
    if not pooled:
        print("no scoreable samples")
        return 0

    med = statistics.median(pooled)
    bad = sum(1 for _n, m, _k in node_medians if abs(m) > 0.03)
    print(f"\nsampled {len(node_medians)} river facing-bet nodes, "
          f"{len(pooled)} (node, combo) points"
          + (f"  [{stats['low_reach_skipped']} low-reach node(s) skipped]"
             if stats.get("low_reach_skipped") else ""))
    print(f"pooled median (implied minus exact equity) = {med * 100:+.1f} points"
          f"   [a consistent export measures ~1-2 points]")
    print(f"nodes with |median gap| > 3 points          = {bad}/{len(node_medians)}")
    print("\nworst nodes (rerun with --dump-node <id> for per-combo detail):")
    for nid, m, k in sorted(node_medians, key=lambda t: -abs(t[1]))[:8]:
        print(f"  {m * 100:+6.1f} points  ({k:3d} combos)  {nid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
