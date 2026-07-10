#!/usr/bin/env python
"""Re-runnable audit for a third-party SQLite ``.db`` postflop solve.

Pulls the REAL numbers from the file (never the metadata labels) and prints a
prioritized PASS / WARN / FAIL report covering the things that killed prior
vendor attempts: format integrity, range correctness (the dealbreaker), whether
the SOLVE actually used the shipped range, board masking, geometry, coverage,
and a strategy sanity pass (c-bet + OOP-lead frequencies).

    venv/bin/python scripts/audit_postflop_db.py /path/to/solve.db

Designed for the BTN-vs-BB family (OOP=BB first to act, IP=BTN), and
pot-type aware (July 2026): a single-raised pot expects a CAPPED BB calling
range, a 3-bet pot expects the OPPOSITE shape (BB 3-bettor holds the
premiums; BTN's call-vs-3bet range is the capped-ish one). Pot type is read
from the ``preflop_line`` metadata (falling back to the spot name). See
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


def _preflop_raise_count(meta: dict) -> int:
    """Raises in the preflop line: 1 = SRP, 2 = 3-bet pot, 3 = 4-bet pot.

    Counts the raise steps in the ``preflop_line`` metadata (e.g.
    "BTN open 3bb, BB 3bet 17bb, BTN call" -> 2). Files without the key
    (the pre-July-2026 exports) fall back to the spot name, else SRP.
    """
    line = meta.get("preflop_line", "")
    if line:
        return sum(
            1 for part in line.split(",")
            if any(t in part.lower() for t in ("open", "3bet", "3-bet", "4bet", "4-bet", "raise"))
        )
    spot = meta.get("spot", "")
    if "4BP" in spot or "4bet" in spot.lower():
        return 3
    if "3BP" in spot or "3bet" in spot.lower():
        return 2
    return 1


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
    n_raises = _preflop_raise_count(meta)
    pot_type = {1: "SRP", 2: "3-bet pot", 3: "4-bet pot"}.get(n_raises, f"{n_raises}-raise pot")
    print("\n[1] METADATA / GEOMETRY")
    for k in ("spot", "flop", "pot", "pot_bb", "eff_stack", "eff_stack_bb",
              "stack_bb", "ante", "rake", "game_format", "btn_open",
              "preflop_line", "oop_range", "ip_range",
              "bb_total_defense", "accuracy"):
        if k in meta:
            print(f"    {k:18s}= {meta[k]}")
    print(f"    pot type          = {pot_type} ({n_raises} preflop raise(s))")
    # Cross-check the chip geometry against the bb-denominated metadata when
    # both are present (the exact chips-per-bb identity the adapter relies on).
    pot, pot_bb = float(meta.get("pot", 0)), float(meta.get("pot_bb", 0) or 0)
    eff, eff_bb = float(meta.get("eff_stack", 0)), float(meta.get("eff_stack_bb", 0) or 0)
    if pot_bb and eff_bb:
        bb_from_pot, bb_from_eff = pot / pot_bb, eff / eff_bb
        geom_ok = abs(bb_from_pot - bb_from_eff) < 0.01
        print(f"    chips/bb agree{_flag(geom_ok)}: pot says {bb_from_pot:g}, eff_stack says {bb_from_eff:g}")
        fails += not geom_ok

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

    print(f"\n[4] RANGES (pulled from preflop_ranges, by 169-class; expectations for a {pot_type})")
    btn_wt, btn_n, btn_tot = range_summary("BTN")
    bb_wt, bb_n, bb_tot = range_summary("BB")

    def avg(wt, n, h):
        return (wt[h] / n[h]) if h in n and n[h] else 0.0

    btn_trash = sum(avg(btn_wt, btn_n, h) for h in ("72o", "32o", "K2o")) / 3
    bb_trash = sum(avg(bb_wt, bb_n, h) for h in ("72o", "32o", "T2o")) / 3
    if n_raises >= 2:  # 3-bet pot: BB 3-bet (uncapped), BTN call-vs-3bet (capped-ish)
        print(f"    BTN call-vs-3bet width = {btn_tot / 13.26:.1f}%   BB 3-bet width = {bb_tot / 13.26:.1f}%")
        # The 3-bettor holds the premiums; trash 3-bet bluffs exist but the
        # worst offsuit junk should be ~absent.
        bb_prem = sum(avg(bb_wt, bb_n, h) for h in ("AA", "KK", "AKs")) / 3
        bb_ok = bb_prem > 0.9 and bb_trash < 0.05
        print(f"    BB 3-bet range UNCAPPED{_flag(bb_ok)}: premiums (AA/KK/AKs) avg={bb_prem:.2f} (want ~1),"
              f" trash (72o/32o/T2o) avg={bb_trash:.2f}")
        # The caller flats the middle (premiums mostly 4-bet, junk folds). A
        # premium-heavy caller range usually means the WRONG file was shipped
        # (the full open range instead of call-vs-3bet) -> warn loudly.
        btn_prem = sum(avg(btn_wt, btn_n, h) for h in ("AA", "KK", "AKs", "AKo")) / 4
        btn_mid = sum(avg(btn_wt, btn_n, h) for h in ("KQs", "QJs", "JTs", "TT")) / 4
        btn_ok = btn_trash < 0.05 and btn_mid > 0.3
        prem_warn = btn_prem > 0.8
        print(f"    BTN call-vs-3bet CAPPED-ISH{_flag(btn_ok)}: trash avg={btn_trash:.2f},"
              f" mid(KQs/QJs/JTs/TT) avg={btn_mid:.2f}")
        print(f"    BTN premiums (AA/KK/AK) avg={btn_prem:.2f}{_flag(True, warn=prem_warn)}"
              f"  (mostly 4-bet -> well below 1; ~1 suggests the open range was shipped instead)")
        fails += (not bb_ok) + (not btn_ok)
    else:  # single-raised pot: BTN open (uncapped), BB call (capped)
        print(f"    BTN open width = {btn_tot / 13.26:.1f}%   BB call width = {bb_tot / 13.26:.1f}%")
        # BTN must NOT be inverted (premiums ~1, trash 0).
        btn_prem = sum(avg(btn_wt, btn_n, h) for h in ("AA", "KK", "AKs")) / 3
        btn_ok = btn_prem > 0.9 and btn_trash < 0.05
        print(f"    BTN premiums (AA/KK/AKs) avg={btn_prem:.2f}, trash (72o/32o/K2o) avg={btn_trash:.2f}{_flag(btn_ok)}")
        # BB calling range must be CAPPED (premiums 3-bet -> ~0) and bottomed (trash 0).
        bb_prem = sum(avg(bb_wt, bb_n, h) for h in ("AA", "KK", "QQ", "AKs", "AKo")) / 5
        bb_mid = sum(avg(bb_wt, bb_n, h) for h in ("QJs", "JTs", "76s", "22")) / 4
        bb_ok = bb_prem < 0.05 and bb_trash < 0.05 and bb_mid > 0.3
        print(f"    BB calling range CAPPED{_flag(bb_ok)}: premiums(AA/KK/QQ/AK) avg={bb_prem:.2f} (want ~0),"
              f" trash avg={bb_trash:.2f}, mid(QJs/JTs/76s/22) avg={bb_mid:.2f}")
        fails += (not btn_ok) + (not bb_ok)

    # -- did the SOLVE use the shipped OOP range? r:0 in-range == BB's range
    # (BB is OOP and acts first at r:0 in both pot types).
    bb_range = {h for h, w in cur.execute(
        "SELECT hand, weight FROM preflop_ranges WHERE player='BB' AND weight>0")}
    bb_playable = {h for h in bb_range if not (h[:2] in board or h[2:] in board)}
    inrange_hands = {idx2hand[i] for i in inrange}
    used_ok = inrange_hands == bb_playable
    bb_role = "3-bet" if n_raises >= 2 else "calling"
    print(f"\n[5] SOLVE USED THE SHIPPED BB {bb_role.upper()} RANGE{_flag(used_ok)}")
    print(f"    r:0 in-range combos = {len(inrange_hands)}; BB {bb_role} range (board-masked) = {len(bb_playable)}")
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
    ip_after_check, _ = agg_strategy("r:0:c", btn_full)
    oop_first, _ = agg_strategy("r:0", bb_full)
    ip_bet_total = sum(v for a, v in ip_after_check.items() if a.startswith("BET_"))
    oop_bet_total = sum(v for a, v in oop_first.items() if a.startswith("BET_"))
    print("\n[7] STRATEGY SANITY (range-weighted)")
    if n_raises >= 2:
        # 3-bet pot: BB (the 3-bettor) c-bets at r:0; a high frequency is normal.
        print(f"    BB (3-bettor) c-bet at r:0 = {oop_bet_total * 100:.0f}% (sizes: "
              + ", ".join(f"{a}={v*100:.0f}%" for a, v in sorted(oop_first.items()) if v > 0.01) + ")")
        print(f"    BTN stab after BB check at r:0:c = {ip_bet_total * 100:.0f}% (sizes: "
              + ", ".join(f"{a}={v*100:.0f}%" for a, v in sorted(ip_after_check.items()) if v > 0.01) + ")")
    else:
        print(f"    BTN c-bet at r:0:c = {ip_bet_total * 100:.0f}% (sizes: "
              + ", ".join(f"{a}={v*100:.0f}%" for a, v in sorted(ip_after_check.items()) if v > 0.01) + ")")
        lead_warn = oop_bet_total > 0.30
        print(f"    BB OOP lead at r:0 = {oop_bet_total * 100:.0f}%{_flag(True, warn=lead_warn)}"
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
