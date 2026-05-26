"""Tests for pipeline.preflop.fact_extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.fact_extractor import (                              # noqa: E402
    VillainRangeStats,
    construct_villain_range_path,
    extract_facts,
    identify_villain,
)
from pipeline.preflop.grammars.types import (                              # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import (                              # noqa: E402
    PreflopActionOption,
    PreflopDecisionNode,
)
from pipeline.preflop.pack import PreflopPack, clear_registry              # noqa: E402
from pipeline.preflop.spot_sampler import sample_spot                       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_full_range(path: Path, weights: dict[str, float]) -> None:
    """Write a range file with all 169 entries, defaulting unlisted classes
    to 0.0. parse_range_file enforces 169-entry presence."""
    from pipeline.preflop_ranges import canonical_169_hand_classes
    path.parent.mkdir(parents=True, exist_ok=True)
    line = ",".join(
        f"{cls}:{weights.get(cls, 0.0)}"
        for cls in canonical_169_hand_classes()
    )
    path.write_text(line)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


# --- identify_villain ------------------------------------------------------
def _node(history: tuple[ParsedAction, ...]) -> PreflopDecisionNode:
    """Minimal node fixture; actions list is empty (irrelevant for villain id)."""
    return PreflopDecisionNode(
        pack_id="test", actor="BTN", history_before=history, actions=(),
    )


def test_identify_villain_returns_last_raiser():
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.RAISE, 77.0),
        ParsedAction("BTN", PreflopActionType.FOLD),
        ParsedAction("SB", PreflopActionType.FOLD),
    )
    villain = identify_villain(_node(history))
    assert villain is not None
    assert villain.position == "CO"
    assert villain.action_type is PreflopActionType.RAISE
    assert villain.raise_size_pct == 77.0


def test_identify_villain_picks_all_in_when_more_recent():
    """An AllIn is treated as villain action just like a raise."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.ALL_IN),
        ParsedAction("CO", PreflopActionType.FOLD),
    )
    villain = identify_villain(_node(history))
    assert villain is not None
    assert villain.position == "HJ"
    assert villain.action_type is PreflopActionType.ALL_IN


def test_identify_villain_none_for_no_aggression():
    """All folds before hero -> no villain (hero is first to raise)."""
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
    )
    assert identify_villain(_node(history)) is None


def test_identify_villain_none_for_empty_history():
    assert identify_villain(_node(())) is None


# --- construct_villain_range_path -----------------------------------------
def test_construct_villain_path_simple_open(tmp_path):
    """UTG opens, hero (BTN) decides. Villain range path = UTG/UTG_60%.txt."""
    pack = PreflopPack(
        pack_id="t", root_path=tmp_path, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
    )
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
    )
    villain = ParsedAction("UTG", PreflopActionType.RAISE, 60.0)
    path = construct_villain_range_path(_node(history), villain, pack)
    assert path == tmp_path / "UTG" / "UTG_60%.txt"


def test_construct_villain_path_3bet_pot(tmp_path):
    """UTG opens, BB 3-bets. Villain (BB) range path includes UTG's open."""
    pack = PreflopPack(
        pack_id="t", root_path=tmp_path, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
    )
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.FOLD),
        ParsedAction("SB", PreflopActionType.FOLD),
        ParsedAction("BB", PreflopActionType.RAISE, 155.0),
    )
    villain = ParsedAction("BB", PreflopActionType.RAISE, 155.0)
    path = construct_villain_range_path(_node(history), villain, pack)
    assert path == (
        tmp_path / "BB" /
        "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%.txt"
    )


def test_construct_villain_path_all_in(tmp_path):
    pack = PreflopPack(
        pack_id="t", root_path=tmp_path, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
    )
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.ALL_IN),
    )
    villain = ParsedAction("HJ", PreflopActionType.ALL_IN)
    path = construct_villain_range_path(_node(history), villain, pack)
    assert path == tmp_path / "HJ" / "UTG_60%_HJ_AI.txt"


def test_construct_villain_path_villain_not_in_history(tmp_path):
    """Defensive: a villain that's not actually in the history triggers
    ValueError rather than silently producing a wrong path."""
    pack = PreflopPack(
        pack_id="t", root_path=tmp_path, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
    )
    history = (ParsedAction("UTG", PreflopActionType.FOLD),)
    fake_villain = ParsedAction("BB", PreflopActionType.RAISE, 100.0)
    with pytest.raises(ValueError, match="not found in node history"):
        construct_villain_range_path(_node(history), fake_villain, pack)


# --- extract_facts: no-villain spot ----------------------------------------
def test_extract_facts_no_villain_returns_empty_villain(tmp_path):
    """Hero first-to-act (no prior raises) -> villain_stats is None."""
    pack = PreflopPack(
        pack_id="t", root_path=tmp_path, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
    )
    # Build a synthetic spot at a UTG-first-to-act node.
    _write_full_range(tmp_path / "UTG" / "UTG_60%.txt", {"AA": 1.0, "AKs": 1.0})
    _write_full_range(tmp_path / "UTG" / "UTG_Fold.txt", {})
    from pipeline.preflop.node_enumerator import enumerate_nodes
    nodes = enumerate_nodes([pack])
    spot = sample_spot(nodes[0], "AA")
    facts = extract_facts(spot, pack)
    assert facts.villain_stats is None
    assert facts.hero_equity_vs_villain is None


# --- extract_facts: villain range file missing -----------------------------
def test_extract_facts_missing_villain_file_returns_empty(tmp_path, caplog):
    """If we'd compute a villain path but the file doesn't exist, warn + skip."""
    pack = PreflopPack(
        pack_id="t", root_path=tmp_path, grammar_name="ryan_pack",
        table_size=6, stack_depth_bb=100, open_size_bb=2.5,
    )
    # Synth pack with BTN options but NO UTG file present.
    _write_full_range(
        tmp_path / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt", {},
    )
    _write_full_range(
        tmp_path / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Call.txt",
        {"AA": 1.0},
    )
    from pipeline.preflop.node_enumerator import enumerate_nodes
    nodes = enumerate_nodes([pack])
    spot = sample_spot(nodes[0], "AA")
    import logging
    with caplog.at_level(logging.WARNING, logger="pipeline.preflop.fact_extractor"):
        facts = extract_facts(spot, pack)
    assert facts.villain_stats is None
    assert facts.hero_equity_vs_villain is None
    assert any("villain range file missing" in r.message for r in caplog.records)


# --- integration: real pack ------------------------------------------------
def test_extract_facts_against_real_pack():
    """End-to-end: a real BTN-vs-UTG-open spot. Verifies equity numbers
    look like real GTO -- AA crushes, T9s gets crushed."""
    ranges = REPO_ROOT / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import discover_packs
    packs = discover_packs(ranges)
    if not packs:
        pytest.skip("Ryan pack not present")
    pack = packs[0]
    nodes = enumerate_nodes(packs)
    # Pick BTN facing exactly UTG open (no 3-bet).
    node = next(
        n for n in nodes
        if n.actor == "BTN"
        and len(n.history_before) == 3
        and n.history_before[0].position == "UTG"
        and n.history_before[0].action_type is PreflopActionType.RAISE
        and n.history_before[1].action_type is PreflopActionType.FOLD
        and n.history_before[2].action_type is PreflopActionType.FOLD
    )
    # AA equity vs UTG open range: dominantly high (75%+).
    facts_aa = extract_facts(sample_spot(node, "AA"), pack, equity_runouts=100)
    assert facts_aa.villain_stats is not None
    assert facts_aa.villain_stats.position == "UTG"
    assert facts_aa.hero_equity_vs_villain is not None
    assert facts_aa.hero_equity_vs_villain > 0.75
    # T9s equity vs UTG open range: weak (under 45%).
    facts_t9s = extract_facts(
        sample_spot(node, "T9s"), pack, equity_runouts=100,
    )
    assert facts_t9s.hero_equity_vs_villain is not None
    assert facts_t9s.hero_equity_vs_villain < 0.50
    # Top combo in UTG range should include premium hands.
    top_classes = {hc for hc, _w in facts_aa.villain_stats.top_combos}
    assert any(hc in top_classes for hc in ("AA", "KK", "QQ", "JJ", "AKs"))


def test_extract_facts_villain_range_stats_sanity():
    """Combo count / % look like real values, not garbage."""
    ranges = REPO_ROOT / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import discover_packs
    packs = discover_packs(ranges)
    if not packs:
        pytest.skip("Ryan pack not present")
    pack = packs[0]
    nodes = enumerate_nodes(packs)
    node = next(
        n for n in nodes
        if n.actor == "BTN"
        and len(n.history_before) == 3
        and n.history_before[0].action_type is PreflopActionType.RAISE
        and n.history_before[1].action_type is PreflopActionType.FOLD
        and n.history_before[2].action_type is PreflopActionType.FOLD
    )
    facts = extract_facts(sample_spot(node, "AA"), pack, equity_runouts=50)
    v = facts.villain_stats
    assert v is not None
    # UTG opens ~15% of hands at 100bb 6-max. Allow generous bounds because
    # the pack might have non-standard sizing.
    assert 1.0 < v.pct_of_dealt_hands < 50.0
    # weighted_combo_count is the sum across all 169 expanded to 1326.
    assert v.weighted_combo_count == pytest.approx(
        (v.pct_of_dealt_hands / 100.0) * 1326, rel=0.01
    )
