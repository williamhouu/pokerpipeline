#!/usr/bin/env python
"""Re-runnable audit for a third-party SQLite ``.db`` postflop solve.

Pulls the REAL numbers from the file (never the metadata labels) and prints a
prioritized PASS / WARN / FAIL report covering the things that killed prior
vendor attempts: format integrity, range correctness (the dealbreaker), whether
the SOLVE actually used the shipped range, board masking, geometry, coverage,
and a strategy sanity pass (c-bet + OOP-lead frequencies).

    venv/bin/python scripts/audit_postflop_db.py /path/to/solve.db

Designed for the BTN-vs-BB SRP family (OOP=BB first to act, IP=BTN). See
``docs`` / memory for the format spec. This is the postflop analogue of
``scripts/audit_nlhe9_pack.py``.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict


def _connect(path: str):
    import sqlite3

    return sqlite3.connect(path)


def _classify(hand: str) -> str:
    ranks = "23456789TJQKA"
    r1, s1, r2, s2 = hand[0], hand[1], hand[2], hand[3]
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if ranks.index(r1) > ranks.index(r2) else (r2, r1)
    return hi + lo + ("s" if s1 == s2 else "o")


def _flag(ok: bool, warn: bool = False) -> str:
    return "  FAIL" if not ok and not warn else ("  WARN" if warn else "  PASS")


def audit(path: str) -> int:  # noqa: C901 - a linear report
    con = _connect(path)
    cur = con.cursor()
    fails = 0

    print("=" * 72)
    print(f"POSTFLOP .db AUDIT: {path}")
    print("=" * 72)

    # -- metadata -------------------------------------------------------
    meta = dict(cur.execute("SELECT key, value FROM metadata"))
    flop = meta.get("flop", "")
    board = {flop[i : i + 2] for i in range(0, len(flop), 2)}
    print("\n[1] METADATA / GEOMETRY")
    for k in ("spot", "flop", "pot", "eff_stack", "stack_bb", "ante", "rake",
              "game_format", "btn_open", "oop_range", "ip_range",
              "bb_total_defense", "accuracy"):
        if k in meta:
            print(f"    {k:18s}= {meta[k]}")

    # -- hand index -----------------------------------------------------
    idx2hand = {i: h for i, h in cur.execute("SELECT idx, hand FROM hand_index")}
    print(f"\n[2] HAND INDEX{_flag(len(idx2hand) == 1326)}: {len(idx2hand)} combos (expect 1326)")
    fails += len(idx2hand) != 1326

    # -- format: blob scale + EV presence + board mask at r:0 -----------
    rows = cur.execute(
        "SELECT action, freq_blob, ev_blob_oop, ev_blob_ip FROM gto_postflop WHERE node='r:0'"
    ).fetchall()
    persum = defaultdict(int)
    maxbyte = 0
    ev_len_ok = True
    for _a, fb, eo, ei in rows:
        f = list(fb)
        maxbyte = max(maxbyte, max(f))
        for i, v in enumerate(f):
            persum[i] += v
        ev_len_ok = ev_len_ok and len(eo) == 4 * len(idx2hand) and len(ei) == 4 * len(idx2hand)
    inrange = [i for i, s in persum.items() if s > 0]
    sums = [persum[i] for i in inrange]
    scale_ok = maxbyte == 255 and (max(sums) <= 256)
    print("\n[3] FORMAT INTEGRITY")
    print(f"    freq blob 0-255 conditional{_flag(scale_ok)}: maxbyte={maxbyte}, per-combo action-sum<= {max(sums)}")
    print(f"    EV blobs present (1326 float32 each){_flag(ev_len_ok)}")
    bad_mask = [idx2hand[i] for i in inrange if (idx2hand[i][:2] in board or idx2hand[i][2:] in board)]
    print(f"    board mask clean{_flag(not bad_mask)}: {len(bad_mask)} board-card combos with nonzero freq (expect 0)")
    fails += (not scale_ok) + (not ev_len_ok) + bool(bad_mask)

    # -- ranges ---------------------------------------------------------
    def range_summary(player: str):
        rows = cur.execute(
            "SELECT hand, weight FROM preflop_ranges WHERE player=?", (player,)
        ).fetchall()
        wt = defaultdict(float)
        n = defaultdict(int)
        for hand, w in rows:
            c = _classify(hand)
            wt[c] += float(w)
            n[c] += 1
        total = sum(wt.values())
        return wt, n, total

    print("\n[4] RANGES (pulled from preflop_ranges, by 169-class)")
    btn_wt, btn_n, btn_tot = range_summary("BTN")
    bb_wt, bb_n, bb_tot = range_summary("BB")
    print(f"    BTN open width = {btn_tot / 13.26:.1f}%   BB call width = {bb_tot / 13.26:.1f}%")

    def avg(wt, n, h):
        return (wt[h] / n[h]) if h in n and n[h] else 0.0

    # BTN must NOT be inverted (premiums ~1, trash 0).
    btn_prem = sum(avg(btn_wt, btn_n, h) for h in ("AA", "KK", "AKs")) / 3
    btn_trash = sum(avg(btn_wt, btn_n, h) for h in ("72o", "32o", "K2o")) / 3
    btn_ok = btn_prem > 0.9 and btn_trash < 0.05
    print(f"    BTN premiums (AA/KK/AKs) avg={btn_prem:.2f}, trash (72o/32o/K2o) avg={btn_trash:.2f}{_flag(btn_ok)}")
    # BB calling range must be CAPPED (premiums 3-bet -> ~0) and bottomed (trash 0).
    bb_prem = sum(avg(bb_wt, bb_n, h) for h in ("AA", "KK", "QQ", "AKs", "AKo")) / 5
    bb_trash = sum(avg(bb_wt, bb_n, h) for h in ("72o", "32o", "T2o")) / 3
    bb_mid = sum(avg(bb_wt, bb_n, h) for h in ("QJs", "JTs", "76s", "22")) / 4
    bb_ok = bb_prem < 0.05 and bb_trash < 0.05 and bb_mid > 0.3
    print(f"    BB calling range CAPPED{_flag(bb_ok)}: premiums(AA/KK/QQ/AK) avg={bb_prem:.2f} (want ~0),"
          f" trash avg={bb_trash:.2f}, mid(QJs/JTs/76s/22) avg={bb_mid:.2f}")
    fails += (not btn_ok) + (not bb_ok)

    # -- did the SOLVE use the capped range? r:0 in-range == BB call range
    bb_range = {h for h, w in cur.execute(
        "SELECT hand, weight FROM preflop_ranges WHERE player='BB' AND weight>0")}
    bb_playable = {h for h in bb_range if not (h[:2] in board or h[2:] in board)}
    inrange_hands = {idx2hand[i] for i in inrange}
    used_ok = inrange_hands == bb_playable
    print(f"\n[5] SOLVE USED THE CAPPED RANGE{_flag(used_ok)}")
    print(f"    r:0 in-range combos = {len(inrange_hands)}; BB calling range (board-masked) = {len(bb_playable)}")
    print(f"    in r:0 not in range = {len(inrange_hands - bb_playable)}; in range not at r:0 = {len(bb_playable - inrange_hands)}")
    fails += not used_ok

    # -- coverage -------------------------------------------------------
    all_nodes = [r[0] for r in cur.execute("SELECT DISTINCT node FROM gto_postflop")]

    def street_of(node: str) -> str:
        cards = [p for p in node.split(":") if len(p) == 2 and p[0] in "23456789TJQKA" and p[1] in "cdhs"]
        return {0: "flop", 1: "turn", 2: "river"}.get(len(cards), "river+")

    by_street = Counter(street_of(n) for n in all_nodes)
    has_bet = defaultdict(bool)
    for node, action, fb in cur.execute("SELECT node, action, freq_blob FROM gto_postflop"):
        if action.startswith("BET_") and any(fb):
            has_bet[node] = True
    bet_by_street = Counter(street_of(n) for n in all_nodes if has_bet[n])
    print(f"\n[6] COVERAGE: {len(all_nodes)} nodes  {dict(by_street)}")
    for s in ("flop", "turn", "river"):
        t, b = by_street[s], bet_by_street[s]
        print(f"    {s}: {b}/{t} nodes offer a live bet ({(b / t * 100 if t else 0):.1f}%)")
    turn_ok = bet_by_street["turn"] > 0 and bet_by_street["river"] > 0
    print(f"    turn + river betting expanded{_flag(turn_ok)}")
    fails += not turn_ok

    # -- strategy sanity: c-bet + OOP lead frequency (range-weighted) ----
    def agg_strategy(node: str, weights: dict[str, float]):
        rows = cur.execute(
            "SELECT action, freq_blob FROM gto_postflop WHERE node=?", (node,)
        ).fetchall()
        f = {a: list(fb) for a, fb in rows}
        agg = defaultdict(float)
        tot = 0.0
        for i in idx2hand:
            s = sum(f[a][i] for a in f)
            if s == 0:
                continue
            w = weights.get(idx2hand[i], 0.0)
            if w <= 0:
                continue
            tot += w
            for a in f:
                agg[a] += w * (f[a][i] / s)
        return {a: agg[a] / tot for a in agg} if tot else {}, tot

    btn_full = {h: float(w) for h, w in cur.execute("SELECT hand, weight FROM preflop_ranges WHERE player='BTN'")}
    bb_full = {h: float(w) for h, w in cur.execute("SELECT hand, weight FROM preflop_ranges WHERE player='BB'")}
    cbet, _ = agg_strategy("r:0:c", btn_full)
    lead, _ = agg_strategy("r:0", bb_full)
    cbet_total = sum(v for a, v in cbet.items() if a.startswith("BET_"))
    lead_total = sum(v for a, v in lead.items() if a.startswith("BET_"))
    print("\n[7] STRATEGY SANITY (range-weighted)")
    print(f"    BTN c-bet at r:0:c = {cbet_total * 100:.0f}% (sizes: "
          + ", ".join(f"{a}={v*100:.0f}%" for a, v in sorted(cbet.items()) if v > 0.01) + ")")
    lead_warn = lead_total > 0.30
    print(f"    BB OOP lead at r:0 = {lead_total * 100:.0f}%{_flag(True, warn=lead_warn)}"
          f"  (textbook BTN-vs-BB SRP is ~0-15%; >30% = unusual, confirm with vendor)")

    print("\n" + "=" * 72)
    verdict = "CLEAN" if fails == 0 else f"{fails} HARD CHECK(S) FAILED"
    print(f"VERDICT: {verdict}"
          + ("  (see WARN lines for non-blocking items)" if fails == 0 else ""))
    print("=" * 72)
    con.close()
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/audit_postflop_db.py /path/to/solve.db")
    raise SystemExit(audit(sys.argv[1]))
