#!/usr/bin/env python
"""Re-runnable audit for a third-party SQLite ``.db`` postflop solve.

Pulls the REAL numbers from the file (never the metadata labels) and prints a
prioritized PASS / WARN / FAIL report covering the things that killed prior
vendor attempts: format integrity, range correctness (the dealbreaker), whether
the SOLVE actually used the shipped range, board masking, geometry, coverage,
a strategy sanity pass (c-bet + OOP-lead frequencies), and -- July 2026 -- an
EV-vs-strategy internal-consistency check on river facing-bet nodes (catches
exports whose frequency tables and EV tables describe two different solutions,
the defect that made the v7 river-barrel answers suspect).

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

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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


def _ev_vs_strategy_check(con, *, max_nodes: int = 24, max_combos_per_node: int = 60) -> int:
    """[8] EV-vs-strategy consistency on river facing-bet nodes.

    A river call is a pure showdown, so the file's own EV tables imply an exact
    equity for every hero combo: ``(EV_call - EV_fold + to_call) / (pot + to_call)``.
    That number must match the exact showdown equity of the same combo against
    the betting range reconstructed from the file's own frequency tables
    (preflop range x the conditional freq blobs along the line, board-masked).
    A systematic gap means the strategy tables and the EV tables describe two
    DIFFERENT solutions -- the July-2026 v7 defect (purified/late-iterate
    strategies exported alongside averaged EVs; bluff-catchers looked ~5-7
    equity points better than they really were, so mixed river calls were
    ~14bb blunders against the file's own EVs). This check would have caught
    that at intake.

    Two hard checks (measured baselines: v8 clean ~1-2 pts; v7 defective
    +5-7 pts on barrel lines, opposite-signed on check-check rivers):
    the pooled median gap (> 3.0 pts = FAIL, > 2.0 = WARN) and the share of
    nodes whose per-node median gap exceeds 3 pts (>= 3 nodes and >= 25% =
    FAIL) - the latter catches a non-converged export whose oppositely-biased
    lines net the pooled median to ~0.

    Returns the number of hard failures (0-2).
    """
    from pipeline.fact_extractor.equity import rank_hand
    from pipeline.postflop.adapters.sqlite_db import (
        SqliteDbAdapter,
        _node_street,
        _Solve,
        _split_cards,
        _stride_sample,
        derive_scenario,
    )

    s = _Solve(con)
    adapter = SqliteDbAdapter("<audit>", **derive_scenario(s.meta))
    bb = adapter._bb_chips(s)
    flop = tuple(_split_cards(s.meta["flop"]))
    board_set = set(flop)
    start_pot = float(s.meta["pot"])
    eff_flop = float(s.meta["eff_stack"])
    pre = {adapter.oop: s.preflop_weights("BB"), adapter.ip: s.preflop_weights("BTN")}
    base_reach = {
        side: [
            (pre[side].get(s.idx_to_hand[i], 0.0)
             if not (set(_split_cards(s.idx_to_hand[i])) & board_set) else 0.0)
            for i in range(s.n)
        ]
        for side in (adapter.oop, adapter.ip)
    }

    # Candidate nodes: river decision nodes offering a FOLD (i.e. facing a wager;
    # the to_call > 0 guard below re-confirms from the betting state, since
    # vendor action LABELS are unreliable).
    fold_nodes = {r[0] for r in con.execute(
        "SELECT DISTINCT node FROM gto_postflop WHERE action='FOLD'")}
    candidates = sorted(n for n in fold_nodes if _node_street(n) == "river")
    print("\n[8] EV-VS-STRATEGY CONSISTENCY (river facing-bet nodes)")
    if not candidates:
        print("    no river facing-bet nodes in this file - check skipped (flop/turn-only export)")
        return 0

    node_medians: list[tuple[str, float, int]] = []
    pooled: list[float] = []
    for nid in _stride_sample(candidates, max_nodes):
        node = adapter._build_node(
            s, nid, bb=bb, start_pot=start_pot, eff_flop=eff_flop,
            base_reach=base_reach, flop=flop,
        )
        if node is None or node.to_call_bb <= 0:
            continue
        acts = {a.db_name: a for a in s.actions(nid)}
        # The passive action facing a bet IS the call, whatever the vendor
        # labelled it (v8 stores it as CHECK, v7 as CALL - labels are unreliable,
        # the betting state above is the truth).
        passive = [acts[k] for k in ("CALL", "CHECK") if k in acts]
        if len(passive) != 1 or "FOLD" not in acts:
            continue
        call_act = passive[0]
        hero_is_oop = node.actor == adapter.oop
        call_ev = call_act.ev_oop if hero_is_oop else call_act.ev_ip
        fold_ev = acts["FOLD"].ev_oop if hero_is_oop else acts["FOLD"].ev_ip
        call_f, fold_f = call_act.freq, acts["FOLD"].freq
        to_call = node.to_call_bb * bb
        pot_now = node.pot_bb * bb  # includes the bet hero faces
        board = list(node.board)

        # Villain's betting range, ranked once per node.
        villain = [
            (set(_split_cards(h)), rank_hand(_split_cards(h) + board), w)
            for h, w in node.villain_range.items() if w > 0
        ]
        if not villain:
            continue

        biases: list[float] = []
        heroes = _stride_sample(
            sorted(h for h, w in node.hero_range.items() if w > 0),
            max_combos_per_node,
        )
        for h in heroes:
            i = s.hand_to_idx[h]
            if call_f[i] + fold_f[i] <= 0:
                continue
            cards = _split_cards(h)
            hero_rank = rank_hand(cards + board)
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
            implied = (call_ev[i] - fold_ev[i] + to_call) / (pot_now + to_call)
            biases.append(implied - exact)
        if biases:
            node_medians.append((nid, statistics.median(biases), len(biases)))
            pooled.extend(biases)

    if not pooled:
        print("    no scoreable (node, combo) samples - check skipped")
        return 0

    # Two prongs: a systematic one-direction bias shifts the POOLED median
    # (v7 3BP: +5.3), while a non-converged export can bias different lines in
    # OPPOSITE directions and net the pool to ~0 (v7 SRP Ts9s5d: +6 on barrel
    # lines, -4 on check-check rivers) - the NODE-share prong catches that.
    med = statistics.median(pooled)
    bad_nodes = sum(1 for _n, m, _k in node_medians if abs(m) > 0.03)
    pooled_ok = abs(med) <= 0.03
    pooled_warn = 0.02 < abs(med) <= 0.03
    share_ok = not (bad_nodes >= 3 and bad_nodes / len(node_medians) >= 0.25)
    share_warn = share_ok and bad_nodes >= 2
    print(f"    sampled {len(node_medians)} nodes / {len(pooled)} (node, combo) points")
    print(f"    pooled median implied-minus-exact equity = {med * 100:+.1f} pts"
          f"{_flag(pooled_ok, warn=pooled_warn)}  (FAIL > 3.0, WARN > 2.0)")
    print(f"    nodes with |median gap| > 3 pts = {bad_nodes}/{len(node_medians)}"
          f"{_flag(share_ok, warn=share_warn)}  (FAIL at >= 3 nodes and >= 25%;"
          f" catches opposite-direction bias that nets the pool to ~0)")
    worst = sorted(node_medians, key=lambda t: -abs(t[1]))[:5]
    for nid, m, k in worst:
        print(f"      {m * 100:+6.1f} pts  ({k:3d} combos)  {nid}")
    return (not pooled_ok) + (not share_ok)


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

    # -- EV vs strategy internal consistency (the July-2026 v7 defect) --
    fails += _ev_vs_strategy_check(con)

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
