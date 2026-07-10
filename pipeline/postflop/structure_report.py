"""Street-by-street betting-structure report for a vendor ``.db`` solve.

Answers "what options does this tree actually offer at each street?" so the
admin picker can show, per solve, the real action menus (bet sizes as pot
fractions, raise sizes, all-ins) at every street x actor x situation, plus
an auto-derived plain-English LIMITATIONS list (e.g. "after a river check,
BTN can never bet" or "the only raise facing a turn bet is all-in"). These
are properties of the TREE the vendor solved, not bugs in the solve; the
report makes them visible before a batch is generated from the file.

The scan walks every decision node's node-string (chip walk only; no
frequency/EV blobs are read), which takes seconds to tens of seconds on a
2 GB export, so the result is CACHED in a JSON sidecar next to the ``.db``
(``<name>.structure.json``), keyed by file size + mtime + report version.
The solves folder is gitignored, so sidecars ride along with their files.

Self-contained within ``pipeline.postflop`` (reads the vendor format via
the adapter's metadata helpers only).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.postflop.adapters.sqlite_db import derive_scenario, read_db_metadata

# Bump when the report content/derivation changes; stale sidecars recompute.
STRUCTURE_REPORT_VERSION = 1

_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_STREETS = ("flop", "turn", "river")
# How many distinct menus to list per (street, actor, context) row before
# collapsing the tail into an "other" count. The big turn/river groups hold
# at most ~5 distinct menus in practice, so this rarely truncates.
_MAX_MENUS_PER_ROW = 6

# Display order inside one context row: most common menu first.
_CONTEXT_ORDER = {
    "first to act": 0, "after a check": 1, "facing a bet": 2,
    "facing a raise": 3, "facing a 3-bet": 4,
}


def _context_label(n_bets: int, any_action: bool) -> str:
    if not any_action:
        return "first to act"
    if n_bets == 0:
        return "after a check"
    return {1: "facing a bet", 2: "facing a raise", 3: "facing a 3-bet"}.get(
        n_bets, f"facing {n_bets} raises"
    )


def _bb_chips(meta: dict) -> float:
    """Chips per bb, matching the adapter's preference order (exact bb-pair
    keys first, else 100 -- the report never needs the legacy btn_open path
    because it only scales display fractions)."""
    for chips_key, bb_key in (("pot", "pot_bb"), ("eff_stack", "eff_stack_bb")):
        try:
            chips = float(meta.get(chips_key, "") or 0)
            in_bb = float(meta.get(bb_key, "") or 0)
        except ValueError:
            continue
        if chips and in_bb:
            return round(chips / in_bb)
    return 100.0


def compute_structure_report(db_path: str | Path) -> dict:
    """Walk every decision node and aggregate the action menus.

    Returns a JSON-serializable dict::

        {"version": 1, "oop": "BB", "ip": "BTN",
         "start_pot_bb": 6.5, "eff_bb": 197.0,
         "streets": {street: [
             {"actor": pos, "context": label, "nodes": N,
              "menus": [{"nodes": n, "options": [...]}, ...],
              "other_menus_nodes": n_rest},
         ...]},
         "limitations": ["plain-English notes", ...]}
    """
    path = str(db_path)
    meta = read_db_metadata(path)
    scenario = derive_scenario(meta)
    oop = str(scenario.get("oop_position", "BB"))
    ip = str(scenario.get("ip_position", "BTN"))
    start_pot = float(meta["pot"])
    eff = float(meta["eff_stack"])
    bb = _bb_chips(meta)

    con = sqlite3.connect(path)
    try:
        node_actions: dict[str, list[str]] = defaultdict(list)
        for node, action in con.execute("SELECT node, action FROM gto_postflop"):
            node_actions[node].append(action)
    finally:
        con.close()

    # (street, actor, context) -> Counter of action-menu tuples, plus the
    # per-group count of nodes whose menu offers any bet/raise (kept during
    # aggregation so the limitation checks see EVERY node, not just the
    # displayed top menus).
    menus: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    bet_nodes: dict[tuple[str, str, str], int] = defaultdict(int)

    for node_id, actions in node_actions.items():
        walked = _walk_node(
            node_id, start_pot=start_pot, eff=eff, oop=oop, ip=ip,
        )
        if walked is None:
            continue
        street, actor, context, pot, invested, behind, to_call = walked
        menu = _menu_labels(
            sorted(set(actions)), to_call=to_call, pot=pot,
            invested=invested, behind=behind,
        )
        key = (street, actor, context)
        menus[key][tuple(menu)] += 1
        if any(o.startswith(("Bet", "Raise", "All-in")) for o in menu):
            bet_nodes[key] += 1

    streets: dict[str, list[dict]] = {}
    for street in _STREETS:
        rows = []
        for (st, actor, context), counter in menus.items():
            if st != street:
                continue
            ranked = counter.most_common()
            shown = ranked[:_MAX_MENUS_PER_ROW]
            rows.append({
                "actor": actor,
                "context": context,
                "nodes": sum(counter.values()),
                "nodes_with_bet": bet_nodes[(st, actor, context)],
                "menus": [{"nodes": n, "options": list(m)} for m, n in shown],
                "other_menus_nodes": sum(n for _, n in ranked[_MAX_MENUS_PER_ROW:]),
            })
        rows.sort(key=lambda r: (r["actor"] != oop, _CONTEXT_ORDER.get(r["context"], 9)))
        if rows:
            streets[street] = rows

    return {
        "version": STRUCTURE_REPORT_VERSION,
        "oop": oop,
        "ip": ip,
        "start_pot_bb": start_pot / bb,
        "eff_bb": eff / bb,
        "streets": streets,
        "limitations": _derive_limitations(streets, oop=oop, ip=ip),
    }


def _walk_node(node_id: str, *, start_pot: float, eff: float, oop: str, ip: str):
    """Chip-walk one node string. Returns ``(street, actor, context, pot,
    invested_of_actor, behind_of_actor, to_call)`` or None for a
    through-a-fold line / beyond-river node."""
    tokens = node_id.split(":")[2:]
    pot = start_pot
    invested = {0: 0.0, 1: 0.0}  # 0 = OOP, 1 = IP
    behind = {0: eff, 1: eff}
    street_idx = 0
    to_act = 0
    seg: list[str] = []  # action tokens on the current street
    for t in tokens:
        if _CARD_RE.match(t):
            street_idx += 1
            invested = {0: 0.0, 1: 0.0}
            to_act = 0
            seg = []
            continue
        if t == "f":
            return None  # decision nodes never sit beyond a fold
        if t.startswith("b"):
            size = float(t[1:])
            added = min(size - invested[to_act], behind[to_act])
            pot += added
            invested[to_act] += added
            behind[to_act] -= added
        elif t == "c":
            need = max(invested.values()) - invested[to_act]
            pot += need
            invested[to_act] += need
            behind[to_act] -= need
        seg.append(t)
        to_act = 1 - to_act
    if street_idx > 2:
        return None
    n_bets = sum(1 for t in seg if t.startswith("b"))
    context = _context_label(n_bets, any_action=bool(seg))
    to_call = max(invested.values()) - invested[to_act]
    actor = oop if to_act == 0 else ip
    return (
        _STREETS[street_idx], actor, context, pot,
        invested[to_act], behind[to_act], to_call,
    )


def _menu_labels(
    actions: list[str], *, to_call: float, pot: float,
    invested: float, behind: float,
) -> list[str]:
    """Human labels for one node's vendor actions, passive first then bets
    ascending (all-in last). Bets label as % of the pot BEFORE the wager;
    raises as a multiple of the bet faced."""
    passive: list[str] = []
    wagers: list[tuple[float, str]] = []
    facing = to_call > 0
    pot_before = pot - to_call
    for a in actions:
        if a == "FOLD":
            passive.append("Fold")
        elif a in ("CHECK", "CALL"):
            passive.append("Call" if facing else "Check")
        elif a.startswith("BET_"):
            size = float(a[4:])
            wager = size - invested
            all_in = wager >= behind - 1
            if facing:
                top = invested + to_call  # the wager being faced (street total)
                mult = size / top if top else 0.0
                label = "All-in raise" if all_in else f"Raise {mult:.1f}x"
            else:
                pct = round(size / pot_before * 100) if pot_before > 0 else 0
                label = f"All-in ({pct}% pot)" if all_in else f"Bet {pct}%"
            wagers.append((size, label))
    passive.sort(key=lambda s: ("Fold", "Check", "Call").index(s))
    return passive + [label for _, label in sorted(wagers)]


def _derive_limitations(streets: dict, *, oop: str, ip: str) -> list[str]:
    """Plain-English notes on what the tree structurally cannot ask.

    Derived from the aggregated menus, not hardcoded to any one vendor
    export, so a future tree that fixes a gap stops being flagged."""
    notes: list[str] = []

    def rows(street: str, actor: str, context: str) -> list[dict]:
        return [
            r for r in streets.get(street, [])
            if r["actor"] == actor and r["context"] == context
        ]

    # 1. River check-then-bet branch (the known v7 truncation).
    for row in rows("river", ip, "after a check"):
        if row["nodes_with_bet"] == 0:
            notes.append(
                f"After a river check, {ip} can never bet: all "
                f"{row['nodes']:,} such nodes are check-only. No river "
                "value-bet-after-check questions can come from this solve."
            )

    # 2. All-in-only raises, per street.
    for street in _STREETS:
        raise_labels = set()
        n_facing = 0
        for r in streets.get(street, []):
            if not r["context"].startswith("facing"):
                continue
            n_facing += r["nodes"]
            for m in r["menus"]:
                raise_labels.update(
                    o for o in m["options"] if o.startswith(("Raise", "All-in"))
                )
        sized = [x for x in raise_labels if x.startswith("Raise")]
        if n_facing and raise_labels and not sized:
            notes.append(
                f"Facing a {street} bet, the only raise in the tree is "
                "all-in (no sized raises)."
            )

    # 3. Single-size betting streets (e.g. the 67%-only turn barrels).
    for street in ("turn", "river"):
        sizes = set()
        n_first = 0
        n_first_with_bet = 0
        for r in streets.get(street, []):
            if r["context"] in ("first to act", "after a check"):
                n_first += r["nodes"]
                n_first_with_bet += r["nodes_with_bet"]
                for m in r["menus"]:
                    sizes.update(
                        o for o in m["options"] if o.startswith("Bet ")
                    )
        if len(sizes) == 1:
            notes.append(
                f"{street.capitalize()} bets come in one size only "
                f"({next(iter(sizes))})."
            )
        if n_first and n_first_with_bet < n_first / 2:
            notes.append(
                f"Most {street} first-in/after-check spots offer no bet at "
                f"all (a bet exists at {n_first_with_bet:,} of "
                f"{n_first:,} such nodes); the tree only expands betting on "
                "some lines."
            )
    return notes


# --- sidecar cache -----------------------------------------------------------
def _sidecar_path(db_path: str | Path) -> Path:
    return Path(db_path).with_suffix(".structure.json")


def load_structure_report(db_path: str | Path) -> dict | None:
    """The cached report, or None when absent/stale (file changed or the
    report version moved on). Never raises on a corrupt sidecar."""
    db = Path(db_path)
    sidecar = _sidecar_path(db)
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        stat = db.stat()
        if (
            payload.get("db_size") == stat.st_size
            and payload.get("db_mtime") == int(stat.st_mtime)
            and payload.get("report", {}).get("version") == STRUCTURE_REPORT_VERSION
        ):
            return payload["report"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def compute_and_cache_structure_report(db_path: str | Path) -> dict:
    """Compute the report and write the sidecar (atomic replace)."""
    db = Path(db_path)
    report = compute_structure_report(db)
    stat = db.stat()
    payload = {
        "db_size": stat.st_size,
        "db_mtime": int(stat.st_mtime),
        "report": report,
    }
    sidecar = _sidecar_path(db)
    tmp = sidecar.with_suffix(".structure.json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(sidecar)
    return report


__all__ = [
    "STRUCTURE_REPORT_VERSION",
    "compute_and_cache_structure_report",
    "compute_structure_report",
    "load_structure_report",
]
