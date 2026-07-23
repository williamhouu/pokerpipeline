"""MTT bb-ante pack integration (July 2026).

The "PLO MTT (BB ANTE) 3.5x Open" MonkerViewer export differs from the cash
packs in three load-bearing ways, each pinned here:

  * a 1bb BB ante that is IN the pot from the first action (the sized-raise
    tokens only resolve to their advertised ~3.5bb with it) but NOT part of
    the BB's callable commitment -- and it comes off the BB's stack;
  * EV units of milli-BB (the cash packs use milli-SB) -- normalised to
    sb-units at the spot-sampler read seam via ``PloDecisionNode.ev_scale``;
  * tournament framing: Context names the ante, Cash/Tourney says
    Tournament, and the Live/Online venue column is blank.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.plo.action_history import (  # noqa: E402
    call_price,
    format_plo_context,
    resolve_pot_limit,
)
from pipeline.plo.hand_order import HAND_COUNT  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import (  # noqa: E402
    PloAction,
    PloActionType,
    discover_plo_pack,
    parse_node_path,
)
from pipeline.plo.spot_sampler import sample_plo_spot  # noqa: E402

R = PloActionType.RAISE
A = PloActionType.ALL_IN
C = PloActionType.CALL
F = PloActionType.FOLD


# --- pot geometry with the bb ante ------------------------------------------
def test_ante_joins_the_pot_and_open_resolves_to_3_5x():
    # First-in: pot = SB 0.5 + BB 1 + ante 1 = 2.5. The export's 40072 open
    # token then resolves to ~3.5bb -- the advertised "3.5x Open". Without
    # the ante the same token gives 2.8bb, which is how the calibration
    # proves the ante is really in the solve.
    actions, pot = resolve_pot_limit(
        (PloAction("LJ", R, 72),), stack_bb=10.0, ante_bb=1.0
    )
    assert abs(actions[0].to_bb - 3.52) < 0.01
    assert abs(pot - (2.5 + 3.52)) < 0.01
    no_ante, _ = resolve_pot_limit((PloAction("LJ", R, 72),), stack_bb=10.0)
    assert abs(no_ante[0].to_bb - 2.8) < 0.01


def test_bb_all_in_total_is_stack_minus_ante():
    # The BB posted the 1bb ante as dead money, so at a 10bb stack their
    # all-in bet totals 9bb; a non-blind seat's jam is the full 10bb.
    hist = (PloAction("LJ", R, 72), PloAction("HJ", F), PloAction("CO", F),
            PloAction("BU", F), PloAction("SB", F), PloAction("BB", A))
    actions, _pot = resolve_pot_limit(hist, stack_bb=10.0, ante_bb=1.0)
    assert actions[-1].to_bb == 9.0
    hist2 = (PloAction("LJ", A),)
    actions2, _ = resolve_pot_limit(hist2, stack_bb=10.0, ante_bb=1.0)
    assert actions2[0].to_bb == 10.0


def test_call_price_includes_the_ante_as_dead_money():
    # BB facing a 3.5bb open: pot = 2.5 (with ante) + 3.5 = 6.0; BB owes 2.5
    # more (blind counts toward the call, the ante does NOT).
    hist = (PloAction("LJ", R, 72), PloAction("HJ", F), PloAction("CO", F),
            PloAction("BU", F), PloAction("SB", F))
    pot, to_call = call_price(hist, "BB", stack_bb=10.0, ante_bb=1.0)
    assert abs(pot - 6.02) < 0.01
    assert abs(to_call - 2.52) < 0.01


# --- EV unit normalisation ---------------------------------------------------
def _write_rng(path: Path, p: float, ev: float) -> None:
    out = []
    for _ in range(HAND_COUNT):
        out.append("x")
        out.append(f"{p};{ev}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_ev_scale_normalises_milli_bb_to_sb_units(tmp_path):
    # Same file EV, read through a cash node (scale 1) vs an MTT node
    # (scale 2): the MTT read doubles it (bb -> sb units), so every
    # downstream EV consumer (gap, difficulty, action_ev_bb) stays unit-safe.
    root = tmp_path / "r"
    _write_rng(root / "0.rng", 0.3, -500)   # ev field: -0.5 units
    _write_rng(root / "40072.rng", 0.7, 1000)
    from pipeline.plo.node_enumerator import PloActionOption

    def _node(scale: float) -> PloDecisionNode:
        return PloDecisionNode(
            actor="LJ",
            history_before=(),
            actions=(
                PloActionOption(
                    action=PloAction("LJ", PloActionType.FOLD),
                    path=root / "0.rng",
                ),
                PloActionOption(
                    action=PloAction("LJ", R, 72), path=root / "40072.rng"
                ),
            ),
            history_stem="",
            table_size=6,
            ev_scale=scale,
        )

    cash = sample_plo_spot(_node(1.0), 0)
    mtt = sample_plo_spot(_node(2.0), 0)
    assert cash.ev_by_action["Raise 72%"] == 1.0   # milli -> sb units
    assert mtt.ev_by_action["Raise 72%"] == 2.0    # milli-bb x2 -> sb units


# --- framing -----------------------------------------------------------------
def test_tournament_context_names_the_ante():
    assert format_plo_context(
        stack_bb=10.0, game_format="tournament", ante_bb=1.0
    ) == "Tournament. 10bb effective stacks. Big blind ante 1bb."
    # Cash context byte-identical to before (no ante mention).
    assert format_plo_context() == (
        "$0.5/$1 Online PLO cash. $100 effective stacks."
    )


# --- pack discovery + end-to-end on a mini fake MTT pack ---------------------
def _mini_mtt_pack(tmp_path: Path):
    root = tmp_path / "ranges" / "Omaha" / "6-way" / "10bb(bb-ante)3.5x"
    _write_rng(root / "0.rng", 0.3, 0)
    _write_rng(root / "1.rng", 0.0, 0)
    _write_rng(root / "40072.rng", 0.7, 500)
    _write_rng(root / "40072.0.rng", 0.2, 0)      # HJ folds vs the open
    _write_rng(root / "40072.1.rng", 0.75, 400)   # HJ calls
    _write_rng(root / "40072.3.rng", 0.05, 100)   # HJ jams
    return discover_plo_pack(tmp_path)


def test_mtt_spec_matches_by_dir_signature(tmp_path):
    pack = _mini_mtt_pack(tmp_path)
    assert pack.pack_id == "plo_mtt_6max_10bb"
    assert pack.spec.ante_bb == 1.0
    assert pack.spec.game_format == "tournament"
    assert pack.spec.ev_in_bb is True
    assert pack.spec.stack_bb == 10.0


def test_mtt_batch_end_to_end(tmp_path):
    from pipeline.plo.batch import generate_plo_batch

    pack = _mini_mtt_pack(tmp_path)
    out = tmp_path / "batch.csv"
    result = generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False
    )
    assert result.questions_written == 1
    with out.open(encoding="utf-8-sig") as handle:
        row = next(iter(csv.DictReader(handle)))
    assert "opens to 3.5bb" in row["Question"]
    assert row["Context"] == (
        "Tournament. 10bb effective stacks. Big blind ante 1bb."
    )
    assert row["Cash/Tourney"] == "Tournament"
    assert row["Live or Online"] == ""
    anim = json.loads(row["animation_script"])
    assert anim["ante_bb"] == 1.0
    ante_events = [e for e in anim["events"] if e.get("ante")]
    assert len(ante_events) == 1
    assert ante_events[0]["seat"] == "BB"
    # BB stack after blind (1) + ante (1) at a 10bb stack.
    assert ante_events[0]["stack_bb"] == 8.0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["run_settings"]["ante_bb"] == 1.0
    assert meta["run_settings"]["game_format"] == "tournament"


def test_solver_data_situation_uses_ante_and_real_stack(tmp_path):
    """The LLM-facing SOLVER DATA 'situation' line must render history
    amounts with the pack's stack_bb + ante_bb (July 2026 regression).

    Left at its 100bb/no-ante defaults it contradicted the Question prose,
    the villain block, and the price block on MTT/short-stack packs ("opens
    to 3bb" where every other surface said 3.5bb; "moves all-in for 100bb"
    on a 10bb stack) -- and the Layer-7 claim checker then treated the WRONG
    situation numbers as truth and rewrote correct prose toward them.
    """
    from pipeline.plo.explanation_generator import build_solver_data
    from pipeline.plo.fact_extractor import extract_plo_facts
    from pipeline.plo.node_enumerator import enumerate_plo_nodes
    from pipeline.plo.options import build_options

    root = tmp_path / "ranges" / "Omaha" / "6-way" / "10bb(bb-ante)3.5x"
    _write_rng(root / "0.rng", 0.3, 0)
    _write_rng(root / "1.rng", 0.0, 0)
    _write_rng(root / "40072.rng", 0.7, 500)
    _write_rng(root / "40072.0.rng", 0.2, 0)      # HJ folds vs the open
    _write_rng(root / "40072.1.rng", 0.75, 400)   # HJ calls
    _write_rng(root / "40072.3.rng", 0.05, 100)   # HJ jams
    _write_rng(root / "40072.3.0.rng", 0.2, 0)    # CO folds vs the jam
    _write_rng(root / "40072.3.1.rng", 0.8, 200)  # CO calls
    pack = discover_plo_pack(tmp_path)
    nodes = {n.history_stem: n for n in enumerate_plo_nodes(pack)}

    def _solver_data(stem: str) -> dict:
        facts = extract_plo_facts(
            sample_plo_spot(nodes[stem], 0), pack, compute_equity=False
        )
        options, correct = build_options(facts)
        return build_solver_data(
            facts, options, correct,
            stack_bb=pack.spec.stack_bb, ante_bb=pack.spec.ante_bb,
        )

    # Facing the open: the ante-inflated 3.5bb size, same as the Question.
    sd = _solver_data("40072")
    assert "opens to 3.5bb" in sd["situation"]
    assert sd["villain"]["action"] == "opens to 3.5bb"
    # Facing the jam: the REAL 10bb stack, not the 100bb default.
    sd = _solver_data("40072.3")
    assert "moves all-in for 10bb" in sd["situation"]
    assert "100bb" not in sd["situation"]
    assert sd["villain"]["action"] == "moves all-in for 10bb"


def test_stack_depth_tags_use_the_pack_stack(tmp_path):
    """The short/standard/deep concept tags read the node's stamped stack_bb
    (July 2026 regression -- they were hardcoded 'always standard_stack' from
    the single-100bb-pack era, so every 10-25bb question shipped a wrong tag
    and the short_stack difficulty modifier never fired)."""
    from pipeline.plo.fact_extractor import extract_plo_facts
    from pipeline.plo.node_enumerator import enumerate_plo_nodes
    from pipeline.plo.spot_tags import compute_plo_concept_tags

    pack = _mini_mtt_pack(tmp_path)  # 10bb MTT pack
    node = next(n for n in enumerate_plo_nodes(pack) if n.history_stem == "")
    assert node.stack_bb == 10.0
    facts = extract_plo_facts(
        sample_plo_spot(node, 0), pack, compute_equity=False
    )
    tags = compute_plo_concept_tags(facts)
    assert "short_stack" in tags
    assert "standard_stack" not in tags


def test_grammar_decodes_the_mtt_tokens():
    # The export's observed token set: 0/1/3 + node-relative 40xxx opens.
    hist = parse_node_path("40072.1.0.0.0.3")
    assert hist[0].action is R and hist[0].raise_pct == 72
    assert hist[1].action is C
    assert hist[-1].action is A


def test_deep_mtt_75bb_depth_is_registered(tmp_path):
    """July 21 PM (team ask): the 75bb MTT depth is registered (the archive
    has no 80bb -- 75 is the nearest to the requested ~80; 50bb and the
    deep MTT 100bb deliberately skipped)."""
    from pipeline.plo.pack import discover_plo_pack

    root = tmp_path / "ranges" / "Omaha" / "6-way" / "75b(bb-ante)3.5x"
    root.mkdir(parents=True)
    _write_rng(root / "0.rng", 0.3, 0)
    pack = discover_plo_pack(tmp_path)
    assert pack.pack_id == "plo_mtt_6max_75bb"
    assert pack.spec.stack_bb == 75.0
    assert pack.spec.game_format == "tournament"
    assert pack.spec.ante_bb == 1.0
    assert pack.spec.ev_in_bb is True
