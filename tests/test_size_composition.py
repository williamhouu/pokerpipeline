"""Per-size range composition (pipeline/postflop/facts.py, July 2026).

The solver-derived replacement for "the big bet is polarized" LLM theory:
each sized open bet's betting range (reach x P(size)) is classified in
Python into strong/medium/air shares, a resolved CHARACTER per size, and a
resolved cross-size COMPARISON verdict. These tests pin the aggregation
arithmetic, the verdict thresholds, the live-menu (artifact-strip) filter,
the >=2-sizes gate, and the SOLVER DATA rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.postflop.explanation_generator import (  # noqa: E402
    build_solver_data_block,
)
from pipeline.postflop.facts import (  # noqa: E402
    SizeComposition,
    _characterize_size,
    compare_size_compositions,
    compute_size_composition,
    extract_facts,
)
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.solve import NodeAction, PostflopNode  # noqa: E402
from pipeline.postflop.spot_sampler import sample_spot  # noqa: E402

BOARD = ("2c", "Js", "7s")


def _entry(*, pot_fraction, middle_pct, value_pct=0.4, air_pct=0.0) -> SizeComposition:
    return SizeComposition(
        label=f"Bet {pot_fraction:.0%}",
        to_bb=pot_fraction * 5.5,
        pot_fraction=pot_fraction,
        freq=0.3,
        value_pct=value_pct,
        middle_pct=middle_pct,
        air_pct=air_pct,
        draw_pct=0.0,
        character="mixed",
    )


def _node(actions, strategy, hero_range, node_id="n1") -> PostflopNode:
    return PostflopNode(
        node_id=node_id,
        street="flop",
        board=BOARD,
        actor="BTN",
        villain="BB",
        pot_bb=5.5,
        effective_stack_bb=97.5,
        actions=tuple(actions),
        strategy=strategy,
        hero_range=hero_range,
        villain_range={"KdQd": 1.0},
    )


# --- character verdict thresholds -------------------------------------------
def test_characterize_size_order_and_thresholds() -> None:
    # Dominant value names the range before the thin middle can call it polarized.
    assert _characterize_size(0.75, 0.10, 0.15) == "value_heavy"
    assert _characterize_size(0.10, 0.15, 0.75) == "bluff_heavy"
    # Thin middle with a real value+air split = polarized.
    assert _characterize_size(0.45, 0.20, 0.35) == "polarized"
    # Fat middle = merged.
    assert _characterize_size(0.25, 0.55, 0.20) == "merged"
    # In between = mixed.
    assert _characterize_size(0.35, 0.40, 0.25) == "mixed"


# --- cross-size comparison ---------------------------------------------------
def test_compare_bigger_more_polarized_and_reverse() -> None:
    small = _entry(pot_fraction=0.33, middle_pct=0.40)
    big = _entry(pot_fraction=0.75, middle_pct=0.10)
    assert compare_size_compositions((small, big)) == "bigger_more_polarized"
    assert compare_size_compositions((big, small)) == "bigger_more_polarized"
    # Reverse the middle shares -> the bigger size is the merged one.
    small2 = _entry(pot_fraction=0.33, middle_pct=0.10)
    big2 = _entry(pot_fraction=0.75, middle_pct=0.40)
    assert compare_size_compositions((small2, big2)) == "bigger_more_merged"


def test_compare_below_margin_is_similar_and_short_is_empty() -> None:
    a = _entry(pot_fraction=0.33, middle_pct=0.30)
    b = _entry(pot_fraction=0.75, middle_pct=0.25)  # 5pt < the 8pt margin
    assert compare_size_compositions((a, b)) == "similar"
    assert compare_size_compositions((a,)) == ""
    assert compare_size_compositions(()) == ""


# --- aggregation arithmetic ---------------------------------------------------
def test_composition_shares_are_reach_times_strategy_weighted() -> None:
    # Three hero combos with obvious buckets on 2c Js 7s:
    #   JdJh = set of jacks (strong made), Td7d = second pair (medium),
    #   9d8d = nine-high air.
    actions = [
        NodeAction(label="Check", verb="check", freq=0.4),
        NodeAction(label="Bet 2bb", verb="bet", freq=0.4, to_bb=1.8, pot_fraction=0.33),
        NodeAction(label="Bet 4bb", verb="bet", freq=0.2, to_bb=4.1, pot_fraction=0.75),
    ]
    strategy = {
        "JdJh": {"Bet 4bb": 0.5, "Bet 2bb": 0.5},
        "Td7d": {"Bet 2bb": 0.5, "Check": 0.5},
        "9d8d": {"Bet 4bb": 0.5, "Check": 0.5},
    }
    hero_range = {"JdJh": 1.0, "Td7d": 1.0, "9d8d": 1.0}
    entries = compute_size_composition(_node(actions, strategy, hero_range))
    by_label = {e.label: e for e in entries}
    assert set(by_label) == {"Bet 2bb", "Bet 4bb"}

    # Bet 4bb betting range: JdJh 0.5 + 9d8d 0.5 -> 50% value, 50% air, 0 medium.
    big = by_label["Bet 4bb"]
    assert abs(big.value_pct - 0.5) < 1e-9
    assert abs(big.air_pct - 0.5) < 1e-9
    assert abs(big.middle_pct - 0.0) < 1e-9
    assert big.character == "polarized"

    # Bet 2bb betting range: JdJh 0.5 + Td7d 0.5 -> 50% value, 50% medium.
    small = by_label["Bet 2bb"]
    assert abs(small.value_pct - 0.5) < 1e-9
    assert abs(small.middle_pct - 0.5) < 1e-9
    assert small.character == "merged"

    # freq = share of the whole range on this size: 1.0 of 3.0 total weight.
    assert abs(big.freq - 1.0 / 3.0) < 1e-9
    assert compare_size_compositions(entries) == "bigger_more_polarized"


def test_single_size_menu_computes_nothing() -> None:
    actions = [
        NodeAction(label="Check", verb="check", freq=0.5),
        NodeAction(label="Bet 2bb", verb="bet", freq=0.5, to_bb=1.8, pot_fraction=0.33),
    ]
    node = _node(actions, {"JdJh": {"Bet 2bb": 1.0}}, {"JdJh": 1.0})
    assert compute_size_composition(node) == ()


def test_sliver_size_is_omitted() -> None:
    # Bet 4bb carries only 1% of the sized-bet mass -> a sliver, omitted.
    actions = [
        NodeAction(label="Bet 2bb", verb="bet", freq=0.9, to_bb=1.8, pot_fraction=0.33),
        NodeAction(label="Bet 4bb", verb="bet", freq=0.1, to_bb=4.1, pot_fraction=0.75),
    ]
    strategy = {"JdJh": {"Bet 2bb": 0.99, "Bet 4bb": 0.01}}
    entries = compute_size_composition(_node(actions, strategy, {"JdJh": 1.0}))
    assert [e.label for e in entries] == ["Bet 2bb"]


def test_board_blocked_hero_combos_are_skipped() -> None:
    actions = [
        NodeAction(label="Bet 2bb", verb="bet", freq=0.5, to_bb=1.8, pot_fraction=0.33),
        NodeAction(label="Bet 4bb", verb="bet", freq=0.5, to_bb=4.1, pot_fraction=0.75),
    ]
    # JsJd shares the Js on the board -> not a real holding, must not count.
    strategy = {
        "JsJd": {"Bet 4bb": 1.0},
        "JdJh": {"Bet 2bb": 1.0},
        "9d8d": {"Bet 4bb": 1.0},
    }
    hero_range = {"JsJd": 1.0, "JdJh": 1.0, "9d8d": 1.0}
    entries = compute_size_composition(_node(actions, strategy, hero_range))
    big = next(e for e in entries if e.label == "Bet 4bb")
    assert abs(big.air_pct - 1.0) < 1e-9  # only 9d8d, the blocked set is gone


# --- extract_facts wiring -----------------------------------------------------
def test_extract_facts_populates_size_fields_on_the_cbet_node() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    node = solve.nodes["flop_ip_cbet"]
    spot = sample_spot(node, "AcJc")
    facts = extract_facts(spot, solve, equity_runouts=20)
    labels = [e.label for e in facts.size_compositions]
    assert labels == ["Bet 2bb", "Bet 4bb"]
    assert facts.size_comparison in (
        "bigger_more_polarized",
        "bigger_more_merged",
        "similar",
    )
    for e in facts.size_compositions:
        assert abs(e.value_pct + e.middle_pct + e.air_pct - 1.0) < 1e-9


def test_extract_facts_empty_on_facing_bet_node() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    node_id = next(
        nid for nid, n in solve.nodes.items() if n.is_facing_bet
    )
    node = solve.nodes[node_id]
    combo = next(iter(node.strategy))
    facts = extract_facts(sample_spot(node, combo), solve, equity_runouts=20)
    assert facts.size_compositions == ()
    assert facts.size_comparison == ""


def test_live_menu_filter_drops_stripped_size_keeps_survivor() -> None:
    # The artifact strip removes one of two sizes: the stripped size must not
    # render anywhere (the strip invariant), the surviving size's line stays
    # (its character is still true and useful), and with only one real size
    # there is no cross-size comparison.
    solve = btn_vs_bb_srp_2cJs7s()
    node = solve.nodes["flop_ip_cbet"]
    spot = sample_spot(node, "AcJc")
    object.__setattr__(spot, "artifact_labels", frozenset({"Bet 4bb"}))
    facts = extract_facts(spot, solve, equity_runouts=20)
    assert [e.label for e in facts.size_compositions] == ["Bet 2bb"]
    assert facts.size_comparison == ""


# --- SOLVER DATA rendering ----------------------------------------------------
def test_data_block_renders_size_lines_only_on_a_size_choice() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    node = solve.nodes["flop_ip_cbet"]
    facts = extract_facts(sample_spot(node, "AcJc"), solve, equity_runouts=20)
    block = build_solver_data_block(facts)
    assert "SIZE COMPOSITION" in block
    assert "SIZE COMPARISON" in block
    assert "Bet 2bb (33% pot):" in block
    assert "Bet 4bb (75% pot):" in block

    facing_id = next(nid for nid, n in solve.nodes.items() if n.is_facing_bet)
    node2 = solve.nodes[facing_id]
    combo = next(iter(node2.strategy))
    facts2 = extract_facts(sample_spot(node2, combo), solve, equity_runouts=20)
    assert "SIZE COMPOSITION" not in build_solver_data_block(facts2)


def test_memo_returns_identical_result_for_same_node() -> None:
    solve = btn_vs_bb_srp_2cJs7s()
    node = solve.nodes["flop_ip_cbet"]
    first = compute_size_composition(node)
    second = compute_size_composition(node)
    assert first == second
    assert first  # the fixture c-bet node has two real sizes
