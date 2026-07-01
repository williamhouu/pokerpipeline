"""Convert the flat 8-max 200bb preflop SQLite solve into an on-disk range pack.

The pipeline consumes preflop packs as a folder of range files parsed by a
filename grammar (see ``pipeline/preflop/pack.py``). This vendor solve ships as
a single flat SQLite table instead, so this one-time converter materialises it
into that on-disk shape -- one ``.rng`` file per (decision node, action) -- so
the entire existing pipeline (node enumerator, node cache, fact extractor, EV
column, admin dropdown, audit) consumes it unchanged.

Design decisions (verified against the DB, June 2026):

* **File CONTENT = Monker ``.rng`` format** (``<Hand>\\n<weight>;<ev>`` x169),
  so ``pipeline.preflop_ranges.parse_range_file`` reads it with no new content
  parser. ``weight`` is the JOINT reach-and-take probability the format
  requires; ``ev`` is the per-action EV in big blinds (``ev_units_per_bb=1.0``).
* **File NAME** encodes the full action line so the new ``gto_preflop_8max``
  filename grammar can rebuild the history. Tokens ``SEAT-CODE`` joined by
  ``_``; CODE is ``F`` / ``C`` / ``A`` / ``R<bb>``. The LAST token is the
  actor's action (= one action option at the node).
* **Conditional -> joint via reach.** The DB's ``range_pct`` is conditional
  (sums to 100 per hand at a node). The file format needs joint = reach x
  conditional, where reach is the product of the actor's OWN earlier-action
  frequencies along the line. This is also what makes reach-gating work: a hand
  the opener never opens gets joint 0 at the vs-3bet node and is excluded (the
  phantom pocket-pair rows handle themselves -- no special-casing).
* **vs_5bet is EXCLUDED** (the only under-converged layer; team decision
  2026-06-30). open / vs_open / vs_3bet / vs_4bet only.
* **UTG1 -> UTG+1** normalised; the rest of the pipeline never sees the dialect.

Usage::

    python scripts/convert_preflop_db_to_pack.py \\
        "~/Downloads/gto_preflop_8max_200bb (2).db" ranges/preflop_8max_200bb
"""
from __future__ import annotations

import sqlite3
import sys
from collections import deque
from pathlib import Path

# 8-max preflop action order (UTG first, BB last). DB dialect uses "UTG1".
ACTION_ORDER: tuple[str, ...] = ("UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB")
_NORMALISE = {"UTG1": "UTG+1"}

# Scenarios we keep (vs_5bet excluded -- under-converged, team call 2026-06-30).
KEEP_SCENARIOS: tuple[str, ...] = ("open", "vs_open", "vs_3bet", "vs_4bet")

# The 169 canonical hand-class labels must all be present in every file.
_N_HANDS = 169


def _norm(seat: str) -> str:
    return _NORMALISE.get(seat, seat)


def _parse_token(tok: str) -> tuple[str, float | None]:
    """A preflop_actions token -> (code, size). 'F'->fold, 'C'->call,
    'RAI'->all-in, 'R17'->raise to 17bb."""
    if tok == "F":
        return ("F", None)
    if tok == "C":
        return ("C", None)
    if tok in ("RAI", "A"):
        return ("A", None)
    if tok.startswith("R"):
        return ("R", float(tok[1:]))
    raise ValueError(f"unknown action token {tok!r}")


def _simulate_seats(preflop_actions: str) -> list[tuple[str, str, float | None]]:
    """Assign each token in a ``preflop_actions`` string to its acting seat by
    walking standard preflop action order (a raise re-opens action to every
    still-active non-raiser, starting after the raiser). Returns the ordered
    history ``[(seat, code, size), ...]`` and is validated by the caller against
    the node's declared actor (the next seat to act)."""
    if not preflop_actions:
        tokens: list[str] = []
    else:
        tokens = preflop_actions.split("-")
    folded: set[str] = set()
    need = deque(ACTION_ORDER)  # everyone acts in the first orbit (BB has option)
    history: list[tuple[str, str, float | None]] = []
    ti = 0
    while ti < len(tokens):
        if not need:
            raise ValueError("ran out of seats to act before tokens exhausted")
        seat = need.popleft()
        if seat in folded:
            continue
        code, size = _parse_token(tokens[ti])
        ti += 1
        history.append((seat, code, size))
        if code == "F":
            folded.add(seat)
        elif code in ("R", "A"):
            # Re-open: every active seat except the raiser must act again, in
            # order starting after the raiser.
            i = ACTION_ORDER.index(seat)
            rotated = ACTION_ORDER[i + 1:] + ACTION_ORDER[:i]
            need = deque(s for s in rotated if s not in folded and s != seat)
        # call ('C') does not re-open action
    return history


def _next_actor(preflop_actions: str) -> str:
    """The seat that acts AFTER the given history (the node's actor)."""
    if not preflop_actions:
        return ACTION_ORDER[0]
    tokens = preflop_actions.split("-")
    folded: set[str] = set()
    need = deque(ACTION_ORDER)
    ti = 0
    while ti < len(tokens):
        seat = need.popleft()
        if seat in folded:
            continue
        code, _ = _parse_token(tokens[ti])
        ti += 1
        if code == "F":
            folded.add(seat)
        elif code in ("R", "A"):
            i = ACTION_ORDER.index(seat)
            rotated = ACTION_ORDER[i + 1:] + ACTION_ORDER[:i]
            need = deque(s for s in rotated if s not in folded and s != seat)
    while need:
        seat = need.popleft()
        if seat not in folded:
            return seat
    raise ValueError("no actor after history")


def _code_for_action(action: str, raise_size: float | None) -> tuple[str, float | None]:
    """A gto_preflop ``action`` -> filename (code, size)."""
    if action == "fold":
        return ("F", None)
    if action == "call":
        return ("C", None)
    if action == "all_in":
        return ("A", None)
    if action.startswith("raise_"):
        return ("R", float(action.split("_", 1)[1]))
    raise ValueError(f"unknown action {action!r}")


def _token(seat: str, code: str, size: float | None) -> str:
    if code == "R":
        # 'g' formatting drops the trailing .0 (R3 not R3.0) but keeps R38.5.
        return f"{_norm(seat)}-R{size:g}"
    return f"{_norm(seat)}-{code}"


def _reach_by_hand(
    con: sqlite3.Connection, scenario: str, position: str, vs_position: str,
    raise_sizes: list[float],
) -> dict[str, float]:
    """The actor's reach probability per hand for this node = product of the
    actor's OWN earlier-action frequencies along the line. open/vs_open -> 1.0
    (the actor is dealt every hand and reaches regardless)."""
    if scenario in ("open", "vs_open"):
        return {}  # 1.0 for every hand (caller-side; sentinel = empty -> no scaling)
    cur = con.cursor()
    reach: dict[str, float] = {}
    if scenario == "vs_3bet":
        # Actor opened (raise to the open size = first raise in the line).
        open_size = raise_sizes[0]
        rows = cur.execute(
            "SELECT hand, frequency_pct FROM ev_hand_actions WHERE scenario='open' "
            "AND position=? AND action=?",
            (position, f"raise_{open_size:g}"),
        ).fetchall()
        for hand, pct in rows:
            reach[hand] = (pct or 0.0) / 100.0
    elif scenario == "vs_4bet":
        # Actor 3-bet (to the second raise size) facing vs_position's open.
        threebet_size = raise_sizes[1]
        rows = cur.execute(
            "SELECT hand, frequency_pct FROM ev_hand_actions WHERE scenario='vs_open' "
            "AND position=? AND vs_position=? AND action=?",
            (position, vs_position, f"raise_{threebet_size:g}"),
        ).fetchall()
        for hand, pct in rows:
            reach[hand] = (pct or 0.0) / 100.0
    return reach


def convert(db_path: Path, out_root: Path) -> None:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Map each (scenario, position, vs_position) -> its preflop_actions line.
    lines: dict[tuple[str, str, str], str] = {}
    for r in cur.execute(
        "SELECT scenario, position, COALESCE(vs_position,'') vp, preflop_actions "
        "FROM ev_raw_nodes"
    ):
        lines[(r["scenario"], r["position"], r["vp"])] = r["preflop_actions"] or ""

    spots = cur.execute(
        "SELECT DISTINCT scenario, position, COALESCE(vs_position,'') vp "
        "FROM ev_hand_actions WHERE scenario IN (%s)"
        % ",".join("?" * len(KEEP_SCENARIOS)),
        KEEP_SCENARIOS,
    ).fetchall()

    n_nodes = n_files = n_excluded_reach = 0
    for sp in spots:
        scenario, position, vp = sp["scenario"], sp["position"], sp["vp"]
        line = lines.get((scenario, position, vp), "")
        # Validate our reconstruction agrees with the DB's declared actor.
        actor = _next_actor(line)
        if actor != position:
            raise SystemExit(
                f"actor mismatch for {scenario}/{position}/{vp}: line {line!r} "
                f"-> {actor}, expected {position}"
            )
        history = _simulate_seats(line)
        raise_sizes = [sz for (_s, c, sz) in history if c == "R" and sz is not None]
        prefix_tokens = [_token(s, c, sz) for (s, c, sz) in history]

        reach = _reach_by_hand(con, scenario, position, vp, raise_sizes)

        # Per (hand, action): conditional freq + per-action EV, from the
        # COMPLETE ev_hand_actions grid (169 hands x every action; gto_preflop
        # is sparse -- only non-zero rows -- so it can't supply full files).
        rows = cur.execute(
            "SELECT hand, action, raise_size, frequency_pct, hand_ev AS ev "
            "FROM ev_hand_actions WHERE scenario=? AND position=? "
            "AND COALESCE(vs_position,'')=?",
            (scenario, position, vp),
        ).fetchall()

        # Group by action -> {hand: (joint_weight, ev)}.
        by_action: dict[tuple[str, float | None], dict[str, tuple[float, float]]] = {}
        for r in rows:
            code, size = _code_for_action(r["action"], r["raise_size"])
            cond = (r["frequency_pct"] or 0.0) / 100.0
            scale = reach.get(r["hand"], 1.0) if reach else 1.0
            joint = cond * scale
            ev = r["ev"] if r["ev"] is not None else 0.0
            by_action.setdefault((code, size), {})[r["hand"]] = (joint, ev)

        node_dir = out_root / _norm(position)
        node_dir.mkdir(parents=True, exist_ok=True)
        node_has_reach = False
        for (code, size), hands in by_action.items():
            if len(hands) != _N_HANDS:
                raise SystemExit(
                    f"{scenario}/{position}/{vp} action {code}{size}: "
                    f"{len(hands)} hands, expected {_N_HANDS}"
                )
            if any(w > 0 for w, _ev in hands.values()):
                node_has_reach = True
            fname = "_".join(prefix_tokens + [_token(position, code, size)]) + ".rng"
            # Monker .rng content: <Hand>\n<weight>;<ev>  x169.
            out_lines: list[str] = []
            for hand in sorted(hands):
                w, ev = hands[hand]
                out_lines.append(hand)
                out_lines.append(f"{w:.6g};{ev:.6g}")
            (node_dir / fname).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            n_files += 1
        if not node_has_reach:
            n_excluded_reach += 1
        n_nodes += 1

    con.close()
    print(
        f"Converted {n_nodes} nodes -> {n_files} .rng files under {out_root} "
        f"({n_excluded_reach} nodes had zero total reach -- phantom, auto-excluded "
        "downstream)."
    )


def main() -> None:
    if len(sys.argv) != 3:  # noqa: PLR2004
        print(__doc__)
        raise SystemExit(2)
    db_path = Path(sys.argv[1]).expanduser()
    out_root = Path(sys.argv[2]).expanduser()
    if not db_path.is_file():
        raise SystemExit(f"DB not found: {db_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    convert(db_path, out_root)


if __name__ == "__main__":
    main()
