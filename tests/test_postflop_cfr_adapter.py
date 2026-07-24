"""Unit tests for the PioSolver ``.cfr`` -> IR adapter, against a MOCK UPI client.

This Mac cannot run PioSolver (Windows only), so the adapter is exercised here
with a hand-built fake UPI client that returns canned-but-realistic responses
for a small flop tree (the same response shapes the real ``PioSolverClient``
emits). This verifies the mapping logic -- node walk, action-label derivation,
range/strategy/EV wiring, chips->bb -- end to end into the pipeline, WITHOUT a
solver. Full real-solver integration is covered only on a Windows host with
PioSolver Edge + a ``.cfr`` (see ``load_postflop_cfr``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.postflop.adapters.cfr_pio import (  # noqa: E402
    CfrPioAdapter,
    _actor_side,
    _node_tokens,
)
from pipeline.postflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.postflop.explanation_generator import (  # noqa: E402
    placeholder_explanation,
)
from pipeline.postflop.facts import extract_facts  # noqa: E402
from pipeline.postflop.format_writer import build_postflop_row  # noqa: E402
from pipeline.postflop.options import build_options  # noqa: E402
from pipeline.postflop.solve import PreflopStep, validate_solve  # noqa: E402
from pipeline.postflop.spot_sampler import sample_spot  # noqa: E402

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _hand_order() -> list[str]:
    """The canonical 1326-combo order (high card first), like ``show_hand_order``."""
    cards = [r + s for r in _RANKS for s in _SUITS]
    return [cards[j] + cards[i] for i in range(52) for j in range(i + 1, 52)]


_HAND_ORDER = _hand_order()
_IDX = {h: i for i, h in enumerate(_HAND_ORDER)}

# --- a small synthetic flop tree (BTN open, BB call, flop 2c Js 7s) ----------
# Geometry: bb = 10 chips, flop-entry pot 55 (5.5bb), 975 behind (97.5bb).
_OOP_RANGE = {"JhTd": 1.0, "9h8h": 1.0, "QdQc": 1.0}  # BB (acts first on the flop)
_IP_RANGE = {"AcKc": 1.0, "Th9h": 1.0, "5d5c": 1.0}  # BTN

# node -> (node_type, board, [children])
_TREE = {
    "r": ("ROOT", "2c Js 7s", ["r:0"]),
    "r:0": ("OOP_DEC", "2c Js 7s", ["r:0:c", "r:0:b22"]),
    "r:0:c": ("IP_DEC", "2c Js 7s", ["r:0:c:c", "r:0:c:b18"]),
    "r:0:b22": ("IP_DEC", "2c Js 7s", ["r:0:b22:f", "r:0:b22:c", "r:0:b22:b66"]),
    # terminals (no DEC, no children) -> the BFS stops here.
    "r:0:c:c": ("SPLIT_NODE", "2c Js 7s", []),
    "r:0:c:b18": ("SPLIT_NODE", "2c Js 7s", []),
    "r:0:b22:f": ("SPLIT_NODE", "2c Js 7s", []),
    "r:0:b22:c": ("SPLIT_NODE", "2c Js 7s", []),
    "r:0:b22:b66": ("SPLIT_NODE", "2c Js 7s", []),
}

# node -> {"OOP"/"IP": {combo: reach}}
_RANGES = {
    "r:0": {"OOP": _OOP_RANGE, "IP": _IP_RANGE},
    "r:0:c": {"OOP": _OOP_RANGE, "IP": _IP_RANGE},
    "r:0:b22": {"OOP": _OOP_RANGE, "IP": _IP_RANGE},
}

# node -> list of {combo: P(action|combo)} in CHILD order.
_STRATEGY = {
    "r:0": [  # children: Check, Bet 22
        {"JhTd": 0.3, "9h8h": 0.85, "QdQc": 0.22},
        {"JhTd": 0.7, "9h8h": 0.15, "QdQc": 0.78},
    ],
    "r:0:c": [  # children: Check, Bet 18
        {"AcKc": 0.4, "Th9h": 0.7, "5d5c": 0.8},
        {"AcKc": 0.6, "Th9h": 0.3, "5d5c": 0.2},
    ],
    "r:0:b22": [  # children: Fold, Call, Raise 66
        {"AcKc": 0.05, "Th9h": 0.2, "5d5c": 0.5},
        {"AcKc": 0.8, "Th9h": 0.6, "5d5c": 0.45},
        {"AcKc": 0.15, "Th9h": 0.2, "5d5c": 0.05},
    ],
}

# child node -> {combo: actor EV in CHIPS} (queried via calc_ev(actor_side, child)).
_EV = {
    "r:0:c": {"JhTd": 20.0, "9h8h": 9.0, "QdQc": 27.0},
    "r:0:b22": {"JhTd": 26.0, "9h8h": 3.0, "QdQc": 36.0},
    "r:0:c:c": {"AcKc": 30.0, "Th9h": 12.0, "5d5c": 15.0},
    "r:0:c:b18": {"AcKc": 33.0, "Th9h": 10.0, "5d5c": 9.0},
    "r:0:b22:f": {"AcKc": 0.0, "Th9h": 0.0, "5d5c": 0.0},
    "r:0:b22:c": {"AcKc": 40.0, "Th9h": 22.0, "5d5c": 18.0},
    "r:0:b22:b66": {"AcKc": 55.0, "Th9h": 30.0, "5d5c": 8.0},
}


def _row(values: dict[str, float], default: float) -> list[float]:
    out = [default] * 1326
    for combo, v in values.items():
        out[_IDX[combo]] = v
    return out


class _FakeUpi:
    """A canned UPI client over the small ``_TREE`` above."""

    def __init__(self) -> None:
        self.loaded: str | None = None

    def load_tree(self, cfr_path: str, mode: str = "auto", timeout: float = 0.0) -> None:
        self.loaded = cfr_path

    def show_node(self, node: str) -> dict:
        ntype, board, children = _TREE[node]
        return {"node_id": node, "node_type": ntype, "board": board,
                "pot": "0 0 55", "children": len(children)}

    def show_children(self, node: str) -> list[str]:
        return list(_TREE[node][2])

    def show_strategy(self, node: str) -> list[list[float]]:
        return [_row(r, 0.0) for r in _STRATEGY[node]]

    def show_range(self, player: str, node: str) -> list[float]:
        return _row(_RANGES[node][player], 0.0)

    def calc_ev(self, player: str, node: str) -> dict:
        if node not in _EV:
            raise RuntimeError(f"no EV for {node}")
        return {"ev": _row(_EV[node], float("nan")), "matchups": _row({}, 0.0)}

    def show_hand_order(self) -> list[str]:
        return list(_HAND_ORDER)

    def show_effective_stack(self) -> int:
        return 975


def _build():
    adapter = CfrPioAdapter(
        _FakeUpi(),
        cfr_path="fake.cfr",
        oop_position="BB",
        ip_position="BTN",
        bb_chips=10.0,
        starting_pot_chips=55.0,
        effective_stack_chips=975.0,
        preflop_summary=(PreflopStep("BTN", "open", to_bb=2.5), PreflopStep("BB", "call")),
        table_size=6,
        stakes="$0.50/$1",
        bb_in_dollars=1.0,
        source_reference="cfr-test/2cJs7s",
    )
    return adapter.build()


# --- small helpers ----------------------------------------------------------
def test_node_token_and_actor_helpers() -> None:
    assert _node_tokens("r:0") == []
    assert _node_tokens("r:0:c:b22") == ["c", "b22"]
    assert _actor_side("OOP_DEC") == "OOP"
    assert _actor_side("IP_DEC") == "IP"
    assert _actor_side("SPLIT_NODE") is None
    assert _actor_side("ROOT") is None


# --- the adapter ------------------------------------------------------------
def test_build_produces_valid_flop_solve() -> None:
    solve = _build()
    assert validate_solve(solve) == []
    assert solve.positions == ("BB", "BTN")
    assert solve.flop == ("2c", "Js", "7s")
    assert solve.effective_stack_bb == 97.5
    assert solve.starting_pot_bb == 5.5
    # 3 decision nodes (the two terminals branches stop the BFS).
    assert set(solve.nodes) == {"r:0", "r:0:c", "r:0:b22"}


def test_oop_lead_node_geometry_and_strategy() -> None:
    solve = _build()
    node = solve.nodes["r:0"]
    assert node.actor == "BB" and node.villain == "BTN"
    assert node.to_call_bb == 0.0  # first to act
    assert node.pot_bb == 5.5
    labels = {a.label for a in node.actions}
    assert labels == {"Check", "Bet 2bb"}  # 22 chips at 10/bb -> 2bb (bb label rule)
    # Per-combo strategy carried through faithfully.
    assert node.strategy["JhTd"]["Bet 2bb"] == pytest.approx(0.7)
    assert node.strategy["9h8h"]["Check"] == pytest.approx(0.85)
    # Both ranges present, keyed by combo.
    assert set(node.hero_range) == set(_OOP_RANGE)
    assert set(node.villain_range) == set(_IP_RANGE)


def test_facing_bet_node_derives_call_raise_and_price() -> None:
    solve = _build()
    node = solve.nodes["r:0:b22"]
    assert node.actor == "BTN"  # IP faces OOP's bet
    assert node.to_call_bb == pytest.approx(2.2)  # 22 chips / 10
    assert node.pot_bb == pytest.approx(7.7)  # 55 + 22
    labels = {a.label for a in node.actions}
    # 66 chips / 10 = 6.6bb, snapped to the 0.5bb display grid -> "Raise to 6.5bb".
    assert labels == {"Fold", "Call", "Raise to 6.5bb"}
    verbs = {a.label: a.verb for a in node.actions}
    assert verbs["Call"] == "call" and verbs["Raise to 6.5bb"] == "raise"


def test_per_action_evs_are_wired_and_converted_to_bb() -> None:
    solve = _build()
    node = solve.nodes["r:0"]
    # combo EVs are chips/bb: JhTd checks 20 chips -> 2.0bb, bets 26 -> 2.6bb.
    assert node.combo_evs["JhTd"]["Check"] == pytest.approx(2.0)
    assert node.combo_evs["JhTd"]["Bet 2bb"] == pytest.approx(2.6)
    # Range-aggregate action EV is reach-weighted over the 3 in-range combos.
    by_label = {a.label: a for a in node.actions}
    assert by_label["Check"].ev_bb is not None
    assert by_label["Check"].ev_bb == pytest.approx((20.0 + 9.0 + 27.0) / 3 / 10)


def test_missing_ev_degrades_gracefully() -> None:
    # include_evs=False -> no calc_ev calls; combo_evs empty, action ev_bb None.
    adapter = CfrPioAdapter(
        _FakeUpi(), cfr_path="fake.cfr", bb_chips=10.0,
        starting_pot_chips=55.0, effective_stack_chips=975.0, include_evs=False,
    )
    solve = adapter.build()
    node = solve.nodes["r:0"]
    assert node.combo_evs == {}
    assert all(a.ev_bb is None for a in node.actions)


def test_built_solve_runs_through_the_pipeline() -> None:
    # The whole point: a .cfr-sourced solve is pipeline-compatible with no
    # special-casing -- facts, options, difficulty, and a CSV row all build.
    solve = _build()
    node = solve.nodes["r:0"]
    spot = sample_spot(node, "JhTd")
    facts = extract_facts(spot, solve)
    assert 0.0 <= facts.hero_equity_vs_villain <= 1.0
    opts, correct = build_options(spot)
    assert correct in opts
    g = placeholder_explanation(facts, opts, correct)
    row = build_postflop_row(facts, g, solve, compute_difficulty(facts), 1)
    assert row["Cards on Table"] == "2-clubs, J-spades, 7-spades"
    assert row["User Seat"].startswith("BB-")


def test_strategy_rows_must_match_children_count() -> None:
    # A node whose strategy row count != child count is skipped (logged), not
    # mis-mapped -- guards against silently pairing the wrong action with a row.
    class _Broken(_FakeUpi):
        def show_strategy(self, node: str) -> list[list[float]]:
            rows = super().show_strategy(node)
            return rows[:-1] if node == "r:0:b22" else rows  # drop one row

    adapter = CfrPioAdapter(
        _Broken(), cfr_path="x", bb_chips=10.0,
        starting_pot_chips=55.0, effective_stack_chips=975.0,
    )
    solve = adapter.build()
    assert "r:0:b22" not in solve.nodes  # skipped
    assert "r:0" in solve.nodes  # the well-formed nodes still build


def test_starting_pot_falls_back_to_node_pot_line() -> None:
    # When the caller omits starting_pot_chips, the adapter reads the settled
    # pot from the root node's UPI pot line ("0 0 55" -> 55).
    adapter = CfrPioAdapter(
        _FakeUpi(), cfr_path="x", bb_chips=10.0, effective_stack_chips=975.0,
    )
    solve = adapter.build()
    assert solve.starting_pot_bb == 5.5
