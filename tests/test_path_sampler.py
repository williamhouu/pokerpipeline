"""Tests for pipeline.path_sampler.

Run directly (`python tests/test_path_sampler.py`) or under pytest.

The pure-function tests (pot / stack math, action labelling) always run. The
integration test loads the real BTN-vs-BB test solve through PioSolver Edge; it
skips cleanly when PioSolver or the .cfr is not available, so it never fails a
machine that simply lacks the solver.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.path_sampler import (                                   # noqa: E402
    PathSampler, _is_card, _sample_evenly, effective_stack, label_action,
    street_of, total_pot,
)
from pipeline.piosolver import PioSolverClient, find_piosolver         # noqa: E402

CFR = Path(__file__).resolve().parent.parent / "test_solves" / "btn_vs_bb_srp_2cJs7s.cfr"


# --- pure-function tests (no solver needed) ----------------------------------
def test_total_pot():
    assert total_pot("0 0 55 ") == 55.0
    assert total_pot("36 0 55 ") == 91.0          # OOP bet 36 into a 55 pot
    assert total_pot("975 975 55") == 2005.0      # both all-in for 975


def test_effective_stack():
    # Root: nothing invested -> full starting stack behind.
    assert effective_stack("0 0 55", 55, 975) == 975
    # OOP has bet 36 -> 36 less behind.
    assert effective_stack("36 0 55", 55, 975) == 939
    # OOP bet 36, IP raised to 102 -> the raiser is in for the most.
    assert effective_stack("36 102 55", 55, 975) == 873
    # Turn node: 72 collected into the carry, split evenly.
    assert effective_stack("0 0 127", 55, 975) == 939
    # Both all-in -> zero effective stack.
    assert effective_stack("975 512 55", 55, 975) == 0


def test_street_of():
    assert street_of(["2c", "Js", "7s"]) == "flop"
    assert street_of(["2c", "Js", "7s", "Kh"]) == "turn"
    assert street_of(["2c", "Js", "7s", "Kh", "3d"]) == "river"
    assert street_of([]) == "preflop"


def test_label_action():
    assert label_action("c", False) == ("check", False)
    assert label_action("c", True) == ("call", False)
    assert label_action("b36", False) == ("bet 36", True)
    assert label_action("b102", True) == ("raise 102", True)
    assert label_action("f", True) == ("fold", True)


def test_is_card_and_sample_evenly():
    assert _is_card("Kh") and _is_card("2c")
    assert not _is_card("b36") and not _is_card("c") and not _is_card("0")
    assert _sample_evenly([1, 2, 3], 6) == [1, 2, 3]          # fewer than the limit
    sampled = _sample_evenly(list(range(49)), 6)
    assert len(sampled) == 6 and sampled[0] == 0 and sampled == sorted(sampled)


# --- integration test (needs PioSolver Edge + the test solve) ----------------
def test_path_sampler_on_real_solve():
    exe = find_piosolver()
    if exe is None or not CFR.is_file():
        print("    (skipped -- PioSolver Edge or the test solve was not found)")
        return

    with PioSolverClient(exe) as client:
        client.load_tree(CFR)
        sampler = PathSampler(client, oop_position="BB", ip_position="BTN")
        nodes = list(sampler.enumerate_decision_nodes(max_chance_children=6))

        assert 150 <= len(nodes) <= 20000, f"enumerated {len(nodes)} decision nodes"

        depth = {"flop": 3, "turn": 4, "river": 5}
        for node in nodes:
            assert node.node_type in ("OOP_DEC", "IP_DEC"), node.node_type
            assert node.street in depth, node.street
            assert len(node.board) == depth[node.street], node.node_id
            assert node.pot > 0, node.node_id
            assert node.effective_stack >= 0, node.node_id
            assert {node.hero_position, node.villain_position} == {"BB", "BTN"}
            assert node.available_actions, node.node_id
            assert node.parent_node_id, node.node_id

        assert {n.street for n in nodes} == {"flop", "turn", "river"}

        # Build full solver context for a spread of the enumerated spots.
        sample = nodes[:: max(1, len(nodes) // 8)][:8]
        for node in sample:
            ctx = sampler.build_spot_context(node)
            assert ctx.hero_range, f"empty hero range at {node.node_id}"
            assert ctx.villain_range, f"empty villain range at {node.node_id}"
            assert ctx.actions, f"no actions at {node.node_id}"
            freqs = [a.frequency for a in ctx.actions]
            assert all(-0.01 <= f <= 1.01 for f in freqs), freqs
            assert abs(sum(freqs) - 1.0) < 0.05, f"strategy sums to {sum(freqs):.3f}"

        print(f"    enumerated {len(nodes)} decision nodes "
              f"(flop/turn/river), built context for {len(sample)}")


if __name__ == "__main__":
    suite = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in suite:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
