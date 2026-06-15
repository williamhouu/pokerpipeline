"""Tests for the Monker-export NLHE pack adapter.

Covers the three pieces that make a Monker pack readable by the preflop
pipeline:

  * ``pipeline.preflop.grammars.monker_nlhe`` -- the stem decoder (seat
    rotation, token vocabulary, canonical seat names).
  * ``pipeline.preflop_ranges.parse_monker_rng_file`` -- the two-line
    self-labeling 169-class content reader, and ``parse_range_file``'s
    suffix dispatch onto it.
  * ``pipeline.preflop.node_enumerator`` honoring ``pack.file_glob`` so
    `.rng` files group into decision nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop import grammars  # noqa: E402
from pipeline.preflop.grammars.monker_nlhe import parse  # noqa: E402
from pipeline.preflop.grammars.types import (  # noqa: E402
    MIN_RAISE_PCT,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import enumerate_nodes  # noqa: E402
from pipeline.preflop.pack import PreflopPack  # noqa: E402
from pipeline.preflop_ranges import (  # noqa: E402
    canonical_169_hand_classes,
    parse_monker_rng_file,
    parse_range_file,
)

NINE_MAX_ORDER = ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB")


def _pack(table_size: int = 9, root: Path | None = None) -> PreflopPack:
    """A fake Monker pack for the parser to tag results with."""
    return PreflopPack(
        pack_id="monker_test",
        root_path=root or Path("/tmp/fake"),
        grammar_name="monker_nlhe",
        table_size=table_size,
        stack_depth_bb=100,
        open_size_bb=4.0,
        file_glob="*.rng",
    )


# --- stem decoding: happy path ----------------------------------------------
def test_parse_root_open():
    r = parse(Path("/x/40120.rng"), _pack())
    assert r.actor == "UTG"
    assert r.actor_action == PreflopActionType.RAISE
    assert r.actor_raise_size_pct == 120.0
    assert len(r.action_history) == 1


def test_parse_root_fold():
    r = parse(Path("/x/0.rng"), _pack())
    assert r.actor == "UTG"
    assert r.actor_action == PreflopActionType.FOLD
    assert r.actor_raise_size_pct is None


def test_parse_sb_limp_first_in():
    """Seven folds then a call = the SB completing (the real pack has this)."""
    r = parse(Path("/x/0.0.0.0.0.0.0.1.rng"), _pack())
    assert r.actor == "SB"
    assert r.actor_action == PreflopActionType.CALL
    assert [a.position for a in r.action_history] == [
        "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB",
    ]
    assert all(
        a.action_type is PreflopActionType.FOLD
        for a in r.action_history[:-1]
    )


def test_parse_bb_raise_vs_limp():
    r = parse(Path("/x/0.0.0.0.0.0.0.1.40100.rng"), _pack())
    assert r.actor == "BB"
    assert r.actor_action == PreflopActionType.RAISE
    assert r.actor_raise_size_pct == 100.0


def test_parse_rotation_after_raise():
    """A caller/raiser rotates to the back and acts again on a re-raise."""
    # UTG opens, 7 folds, BB calls, UTG raises again (3-bet), BB calls.
    r = parse(Path("/x/40120.0.0.0.0.0.0.0.1.40090.1.rng"), _pack())
    positions = [a.position for a in r.action_history]
    assert positions == [
        "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB",
        "UTG", "BB",
    ]
    assert r.actor == "BB"
    assert r.actor_action == PreflopActionType.CALL
    utg_3bet = r.action_history[9]
    assert utg_3bet.action_type == PreflopActionType.RAISE
    assert utg_3bet.raise_size_pct == 90.0


def test_parse_all_in_leaves_action():
    """An all-in seat is out of the rotation; later tokens skip it."""
    # UTG opens, UTG+1 jams, UTG+2 calls the jam.
    r = parse(Path("/x/40120.3.1.rng"), _pack())
    positions = [a.position for a in r.action_history]
    assert positions == ["UTG", "UTG+1", "UTG+2"]
    assert r.action_history[1].action_type == PreflopActionType.ALL_IN
    assert r.actor == "UTG+2"
    assert r.actor_action == PreflopActionType.CALL


def test_parse_deep_limp_raise_war():
    """Real-pack deep stem: limp, raise, re-raise, jam, call."""
    r = parse(Path("/x/0.0.0.0.0.0.0.1.40100.40100.3.1.rng"), _pack())
    positions = [a.position for a in r.action_history]
    assert positions == [
        "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN",
        "SB", "BB", "SB", "BB", "SB",
    ]
    assert r.actor == "SB"
    assert r.actor_action == PreflopActionType.CALL
    assert r.action_history[10].action_type == PreflopActionType.ALL_IN


def test_parse_six_max_rotation():
    """table_size=6 rotates UTG/HJ/CO/BTN/SB/BB (no UTG+1/UTG+2/LJ)."""
    r = parse(Path("/x/0.0.0.0.1.rng"), _pack(table_size=6))
    assert [a.position for a in r.action_history] == [
        "UTG", "HJ", "CO", "BTN", "SB",
    ]
    assert r.actor == "SB"
    assert r.actor_action == PreflopActionType.CALL


# --- short-stack 6-max tokens: `5` (min-raise) and `14` (BB iso) -------------
def test_parse_min_raise_open_token_5():
    """`5` is the short-stack min-raise open; it carries the MIN_RAISE_PCT
    sentinel (no pot fraction -- sized relative to the bet downstream)."""
    r = parse(Path("/x/5.rng"), _pack(table_size=6))
    assert r.actor == "UTG"
    assert r.actor_action == PreflopActionType.RAISE
    assert r.actor_raise_size_pct == MIN_RAISE_PCT


def test_parse_min_raise_reopens_action():
    """A `5` min-raise rotates the raiser to the back (re-raise reopens it):
    UTG opens `5`, folds to BB, BB jams over it."""
    r = parse(Path("/x/5.0.0.0.0.3.rng"), _pack(table_size=6))
    positions = [a.position for a in r.action_history]
    assert positions == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    assert r.action_history[0].action_type == PreflopActionType.RAISE
    assert r.action_history[0].raise_size_pct == MIN_RAISE_PCT
    assert r.actor == "BB"
    assert r.actor_action == PreflopActionType.ALL_IN


def test_parse_bb_iso_token_14():
    """`14` = BB iso over a single SB limp, emitted as a 75%-pot raise
    (folds to SB, SB limps, BB raises)."""
    r = parse(Path("/x/0.0.0.0.1.14.rng"), _pack(table_size=6))
    assert r.actor == "BB"
    assert r.actor_action == PreflopActionType.RAISE
    assert r.actor_raise_size_pct == 75.0
    assert r.action_history[-2].action_type == PreflopActionType.CALL  # SB limp


def test_parse_via_registry_dispatch():
    """grammars.parse routes grammar_name='monker_nlhe' to this parser."""
    r = grammars.parse(Path("/x/40120.rng"), pack=_pack())
    assert r.pack_id == "monker_test"
    assert r.actor == "UTG"


def test_render_raise_size_token():
    """The sentinel renders 'min' (never '-1%'); real pcts render '<n>%'.

    The min-raise option label and the premise-realism gate's lookup label
    both go through this, so they must agree on the same string."""
    from pipeline.preflop.grammars.types import (  # noqa: PLC0415
        render_raise_size_token,
    )

    assert render_raise_size_token(MIN_RAISE_PCT) == "min"
    assert render_raise_size_token(75.0) == "75%"
    assert render_raise_size_token(120.0) == "120%"


def test_min_raise_node_id_and_label_render_min():
    """A min-raise leaks neither '-1%' into the option label nor the node id:
    the option reads 'Raise min', a min-raise in history reads '<seat>_min'."""
    from pipeline.preflop.grammars.types import ParsedAction  # noqa: PLC0415
    from pipeline.preflop.node_enumerator import (  # noqa: PLC0415
        PreflopActionOption,
        PreflopDecisionNode,
    )

    # Hero (BB) faces a UTG min-raise open: the option offered is a fold,
    # the history carries UTG's min-raise.
    r5 = parse(Path("/x/5.rng"), _pack(table_size=6))  # the UTG open file
    opt = PreflopActionOption(
        action_type=PreflopActionType.RAISE,
        raise_size_pct=MIN_RAISE_PCT,
        range_file=r5,
    )
    node = PreflopDecisionNode(
        pack_id="monker_test",
        actor="BB",
        history_before=(
            ParsedAction("UTG", PreflopActionType.RAISE, MIN_RAISE_PCT),
        ),
        actions=(opt,),
    )
    assert opt.label == "Raise min"
    assert node.node_id == "UTG_min_BB_decision"


# --- stem decoding: rejections ----------------------------------------------
@pytest.mark.parametrize(
    "stem",
    [
        "40",       # raise prefix with no percent digits
        "400",      # 0% raise is nonsense
        "2",        # unknown special token
        "15",       # only `14` is special, not the whole 1x family
        "abc",      # garbage
        "40120.x",  # garbage mid-chain
    ],
)
def test_parse_rejects_bad_tokens(stem: str):
    with pytest.raises(ValueError, match="monker_nlhe"):
        parse(Path(f"/x/{stem}.rng"), _pack())


def test_parse_rejects_more_actions_than_seats():
    # 9 folds empty the queue; a 10th action has no seat to act.
    with pytest.raises(ValueError, match="no seat left"):
        parse(Path(f"/x/{'.'.join(['0'] * 10)}.rng"), _pack())
    # Same at 6-max, two sizes smaller.
    with pytest.raises(ValueError, match="no seat left"):
        parse(Path(f"/x/{'.'.join(['0'] * 7)}.rng"), _pack(table_size=6))


# --- .rng content reading -----------------------------------------------------
def _write_rng(
    path: Path,
    overrides: dict[str, tuple[float, float]] | None = None,
    *,
    drop: str | None = None,
    relabel: tuple[str, str] | None = None,
) -> Path:
    """Write a structurally valid 169-class .rng (with optional corruption)."""
    overrides = overrides or {}
    lines: list[str] = []
    for cls in canonical_169_hand_classes():
        if cls == drop:
            continue
        label = cls
        if relabel and cls == relabel[0]:
            label = relabel[1]
        p, ev = overrides.get(cls, (0.0, 0.0))
        lines.append(label)
        lines.append(f"{p};{ev}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_monker_rng_file_roundtrip(tmp_path: Path):
    rng = _write_rng(
        tmp_path / "40120.rng",
        {"AA": (1.0, 8313.56), "A2s": (0.546, 0.0174), "KK": (1.0, 2039.79)},
    )
    out = parse_monker_rng_file(rng)
    assert len(out) == 169
    assert out["AA"] == (1.0, 8313.56)
    assert out["A2s"] == (0.546, 0.0174)
    assert out["72o"] == (0.0, 0.0)


def test_parse_range_file_dispatches_on_rng_suffix(tmp_path: Path):
    rng = _write_rng(tmp_path / "0.rng", {"72o": (0.97, -1.5), "AA": (0.0, 0.0)})
    weights = parse_range_file(rng)
    assert len(weights) == 169
    assert weights["72o"] == 0.97
    assert weights["AA"] == 0.0  # EVs are not in the weights view


def test_parse_monker_rng_empty_ev_field_is_zero(tmp_path: Path):
    rng = _write_rng(tmp_path / "1.rng")
    text = rng.read_text(encoding="utf-8").replace("0.0;0.0", "0.5;", 1)
    rng.write_text(text, encoding="utf-8")
    out = parse_monker_rng_file(rng)
    assert out["AA"] == (0.5, 0.0)


def test_parse_monker_rng_rejects_missing_class(tmp_path: Path):
    rng = _write_rng(tmp_path / "bad.rng", drop="J5s")
    with pytest.raises(ValueError, match="expected 338"):
        parse_monker_rng_file(rng)


def test_parse_monker_rng_rejects_unknown_label(tmp_path: Path):
    rng = _write_rng(tmp_path / "bad.rng", relabel=("J5s", "XX"))
    with pytest.raises(ValueError, match="canonical 169"):
        parse_monker_rng_file(rng)


def test_parse_monker_rng_bare_weight_line_means_ev_zero(tmp_path: Path):
    """The export drops the EV field on some deep nodes -- a bare weight
    line (no ';') is valid and parses as ev 0.0 (real-pack example:
    40120.40084.40046.3.1.1.1.1.0.rng)."""
    rng = _write_rng(tmp_path / "deep.rng")
    text = rng.read_text(encoding="utf-8").replace("0.0;0.0", "0.75", 1)
    rng.write_text(text, encoding="utf-8")
    out = parse_monker_rng_file(rng)
    assert out["AA"] == (0.75, 0.0)


def test_parse_monker_rng_rejects_duplicate_label(tmp_path: Path):
    rng = _write_rng(tmp_path / "bad.rng", relabel=("J5s", "J6s"))
    with pytest.raises(ValueError, match="duplicate hand label"):
        parse_monker_rng_file(rng)


# --- node enumeration over a synthetic mini-pack ------------------------------
def test_enumerate_nodes_groups_rng_siblings(tmp_path: Path):
    """Sibling .rng files (same history, different final token) = one node.

    Files are empty on purpose: enumeration must only parse names, never
    contents.
    """
    for stem in (
        "0",
        "40120",
        "40120.0.0.0.0.0.0.0.0",
        "40120.0.0.0.0.0.0.0.1",
        "40120.0.0.0.0.0.0.0.40084",
    ):
        (tmp_path / f"{stem}.rng").touch()

    nodes = enumerate_nodes([_pack(root=tmp_path)])
    assert len(nodes) == 2

    by_actor = {n.actor: n for n in nodes}
    root = by_actor["UTG"]
    assert root.history_before == ()
    assert [o.label for o in root.actions] == ["Fold", "Raise 120%"]

    bb = by_actor["BB"]
    assert len(bb.history_before) == 8
    # Options sort by backing filename: 0 (Fold) < 1 (Call) < 40084.
    assert [o.label for o in bb.actions] == ["Fold", "Call", "Raise 84%"]
    assert bb.node_id.startswith("UTG_120%")
    assert bb.node_id.endswith("BB_decision")


def test_enumerate_nodes_file_glob_excludes_other_extensions(tmp_path: Path):
    (tmp_path / "40120.rng").touch()
    (tmp_path / "stray_notes.txt").write_text("not a range", encoding="utf-8")
    nodes = enumerate_nodes([_pack(root=tmp_path)])
    assert len(nodes) == 1
    assert nodes[0].actions[0].range_file.path.suffix == ".rng"


def test_villain_range_path_is_stem_prefix(tmp_path: Path):
    """Monker villain range = the node stem truncated at the villain's token."""
    from pipeline.preflop.fact_extractor import (  # noqa: PLC0415
        construct_villain_range_path,
        identify_villain,
    )

    for stem in (
        "40120.0.0.0.0.0.0.0.0",
        "40120.0.0.0.0.0.0.0.1",
        "40120.0.0.0.0.0.0.0.40084",
    ):
        (tmp_path / f"{stem}.rng").touch()
    pack = _pack(root=tmp_path)
    (node,) = enumerate_nodes([pack])
    villain = identify_villain(node)
    assert villain is not None
    assert villain.position == "UTG"
    path = construct_villain_range_path(node, villain, pack)
    assert path == tmp_path / "40120.rng"
