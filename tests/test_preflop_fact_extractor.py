"""Tests for pipeline.preflop.fact_extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.fact_extractor import (  # noqa: E402
    classify_archetype,
    compute_blockers,
    compute_villain_range_stats,
    construct_villain_range_path,
    extract_facts,
    identify_showdown_opponents,
    identify_villain,
)
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopDecisionNode,
)
from pipeline.preflop.pack import PreflopPack, clear_registry  # noqa: E402
from pipeline.preflop.spot_sampler import (
    PreflopSpot,  # noqa: E402, F811
    sample_spot,  # noqa: E402
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_full_range(path: Path, weights: dict[str, float]) -> None:
    """Write a range file with all 169 entries, defaulting unlisted classes
    to 0.0. parse_range_file enforces 169-entry presence."""
    from pipeline.preflop_ranges import canonical_169_hand_classes

    path.parent.mkdir(parents=True, exist_ok=True)
    line = ",".join(
        f"{cls}:{weights.get(cls, 0.0)}" for cls in canonical_169_hand_classes()
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
        pack_id="test",
        actor="BTN",
        history_before=history,
        actions=(),
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
def test_identify_showdown_opponents_multiway_allin():
    """Hero (BTN, the _node actor) faces a multi-way all-in. The opener
    raises then FOLDS to a jam (drops out); CO jams; HJ calls the jam
    (stays in); SB folds. The field = the still-committed seats, with each
    one's LAST action (so HJ's jam-call, not its earlier call)."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 120.0),
        ParsedAction("HJ", PreflopActionType.CALL),
        ParsedAction("CO", PreflopActionType.ALL_IN),
        ParsedAction("SB", PreflopActionType.FOLD),
        ParsedAction("UTG", PreflopActionType.FOLD),   # opener folds to the jam
        ParsedAction("HJ", PreflopActionType.CALL),    # HJ calls the jam
    )
    opps = identify_showdown_opponents(_node(history))
    by_pos = {o.position: o for o in opps}
    assert sorted(by_pos) == ["CO", "HJ"]              # UTG/SB folded out
    assert by_pos["CO"].action_type is PreflopActionType.ALL_IN
    assert by_pos["HJ"].action_type is PreflopActionType.CALL


def test_identify_showdown_opponents_excludes_hero_and_handles_headsup():
    """Hero's own prior action is never an opponent; a lone jammer = 1
    opponent (heads-up, so the multi-way path won't fire)."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 100.0),
        ParsedAction("BTN", PreflopActionType.CALL),   # hero (BTN) called earlier
        ParsedAction("UTG", PreflopActionType.ALL_IN),  # UTG jams over the call
    )
    opps = identify_showdown_opponents(_node(history))
    assert [o.position for o in opps] == ["UTG"]        # BTN (hero) excluded


def test_construct_villain_path_simple_open(tmp_path):
    """UTG opens, hero (BTN) decides. Villain range path = UTG/UTG_60%.txt."""
    pack = PreflopPack(
        pack_id="t",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
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
        pack_id="t",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
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
        tmp_path / "BB" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%.txt"
    )


def test_construct_villain_path_all_in(tmp_path):
    pack = PreflopPack(
        pack_id="t",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
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
        pack_id="t",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )
    history = (ParsedAction("UTG", PreflopActionType.FOLD),)
    fake_villain = ParsedAction("BB", PreflopActionType.RAISE, 100.0)
    with pytest.raises(ValueError, match="not found in node history"):
        construct_villain_range_path(_node(history), fake_villain, pack)


# --- extract_facts: no-villain spot ----------------------------------------
def test_extract_facts_no_villain_returns_empty_villain(tmp_path):
    """Hero first-to-act (no prior raises) -> villain_stats is None."""
    pack = PreflopPack(
        pack_id="t",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
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
        pack_id="t",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )
    # Synth pack with BTN options but NO UTG file present.
    _write_full_range(
        tmp_path / "BTN" / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt",
        {},
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
    # Enumerate only the pack under test: other registered packs (9-max,
    # the short-stack 6-max packs) can share node geometry and would
    # otherwise be picked by the predicate below with the wrong pack object.
    nodes = enumerate_nodes([pack])
    # Pick BTN facing exactly UTG open (no 3-bet).
    node = next(
        n
        for n in nodes
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
        sample_spot(node, "T9s"),
        pack,
        equity_runouts=100,
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
    # Enumerate only the pack under test: other registered packs (9-max,
    # the short-stack 6-max packs) can share node geometry and would
    # otherwise be picked by the predicate below with the wrong pack object.
    nodes = enumerate_nodes([pack])
    node = next(
        n
        for n in nodes
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


def test_most_common_combos_is_combo_weighted_not_weight_sorted(tmp_path):
    """`most_common_combos` ranks by combo SHARE (weight x combos), so a
    12-combo offsuit hand leads equal-weight pairs/suited -- while `top_combos`
    stays weight-sorted (premium-first). This is the fix for the data block
    reading like "the premium hands" instead of "what villain shows up with"."""
    # Tight jam range: all full weight, QQ partial. By combos: AKo(12) >
    # AA/KK(6) > AKs(4) > QQ at 0.5 (3). By weight: AA/KK/AKs/AKo tie at 1.0
    # (premium-first), QQ last.
    rng = {"AA": 1.0, "KK": 1.0, "QQ": 0.5, "AKs": 1.0, "AKo": 1.0}
    path = tmp_path / "jam.txt"
    _write_full_range(path, rng)
    villain = ParsedAction("BTN", PreflopActionType.ALL_IN)
    stats = compute_villain_range_stats(villain, path, top_n=5)

    most_common = [c for c, _w in stats.most_common_combos]
    top = [c for c, _w in stats.top_combos]
    # Most-common: AKo first (12 combos), QQ last (0.5*6 = 3 weighted combos).
    assert most_common[0] == "AKo"
    assert most_common[-1] == "QQ"
    # Weight-sorted top_combos buries AKo behind the equal-weight premiums and
    # is a genuinely different order -- proving the two stats aren't aliases.
    assert top[0] != "AKo"
    assert most_common != top


# --- chunk 2: compute_blockers --------------------------------------------
def test_compute_blockers_finds_pair_blockers():
    """Holding AsKh, hero blocks AA combos containing As (3 of 6) and KK
    combos containing Kh (3 of 6)."""
    hero_combo = "AsKh"
    # Simplified villain range: AA (all 6 combos), KK (all 6 combos).
    villain = {
        "AsAh": 1.0,
        "AsAd": 1.0,
        "AsAc": 1.0,
        "AhAd": 1.0,
        "AhAc": 1.0,
        "AdAc": 1.0,
        "KsKh": 1.0,
        "KsKd": 1.0,
        "KsKc": 1.0,
        "KhKd": 1.0,
        "KhKc": 1.0,
        "KdKc": 1.0,
    }
    blockers = compute_blockers(hero_combo, villain)
    # Hero has As: blocks the 3 AA combos containing As (AsAh, AsAd, AsAc).
    # Hero has Kh: blocks the 3 KK combos containing Kh (KsKh, KhKd, KhKc).
    assert blockers == {"AA": 3, "KK": 3}


def test_compute_blockers_skips_zero_weight():
    """Zero-weight combos don't count as blocked."""
    hero_combo = "AsKh"
    villain = {"AsAh": 0.0, "AsAd": 1.0}
    blockers = compute_blockers(hero_combo, villain)
    assert blockers == {"AA": 1}  # AsAd counted; AsAh skipped


def test_compute_blockers_no_overlap_empty():
    """If hero's cards don't overlap any villain combo, no blockers."""
    hero_combo = "2c3d"
    villain = {"AhAs": 1.0, "KhKs": 1.0}
    blockers = compute_blockers(hero_combo, villain)
    assert blockers == {}


# --- chunk 2: classify_archetype ------------------------------------------
def _spot_with(
    dominant_action: str,
    frequencies: dict[str, float],
    history: tuple[ParsedAction, ...] = (),
    *,
    actor: str = "BTN",
    hand_class: str = "AKo",
    combo: str = "AhKc",
) -> PreflopSpot:
    """Build a PreflopSpot with rigged values for archetype testing."""
    node = PreflopDecisionNode(
        pack_id="t",
        actor=actor,
        history_before=history,
        actions=(),
    )
    dom_freq = frequencies[dominant_action]
    return PreflopSpot(
        node=node,
        hero_hand_class=hand_class,
        hero_card_combo=combo,
        action_frequencies=frequencies,
        dominant_action=dominant_action,
        dominant_frequency=dom_freq,
    )


def test_archetype_unclassified_for_zero_presence():
    """Hand with ~0% total presence at the node -> unclassified."""
    spot = _spot_with("Fold", {"Fold": 0.0, "Call": 0.0})
    assert (
        classify_archetype(spot, villain=None, hero_equity_vs_villain=None)
        == "unclassified"
    )


def test_archetype_open_for_value_no_villain():
    """No villain + dominant Raise -> open_for_value."""
    spot = _spot_with("Raise 60%", {"Raise 60%": 1.0, "Fold": 0.0})
    assert (
        classify_archetype(spot, villain=None, hero_equity_vs_villain=None)
        == "open_for_value"
    )


def test_archetype_bb_check_in_limped_pot():
    """BB facing a limp (SB completed, no raise) -> bb_check, not open/fold.

    No raise in the history means identify_villain returns None; the BB's
    dominant 'Call' is really a check (nothing to call).
    """
    history = (ParsedAction("SB", PreflopActionType.CALL),)
    spot = _spot_with("Call", {"Call": 0.95, "Raise 100%": 0.05}, history, actor="BB")
    assert (
        classify_archetype(spot, villain=None, hero_equity_vs_villain=None)
        == "bb_check"
    )


def test_archetype_bb_check_is_bb_only():
    """A non-BB first-in with a dominant non-raise never gets bb_check."""
    spot = _spot_with("Fold", {"Fold": 1.0, "Raise 60%": 0.0}, actor="CO")
    assert (
        classify_archetype(spot, villain=None, hero_equity_vs_villain=None)
        != "bb_check"
    )


def test_archetype_sb_complete_first_in():
    """SB first-in with dominant Call = completing the half bet (a limp).

    The Monker 9-max pack offers the SB limp (the Pio 6-max pack didn't);
    labelling it open_for_value/fold_outranged would hand the LLM the
    wrong frame for a Call answer.
    """
    history = tuple(
        ParsedAction(pos, PreflopActionType.FOLD)
        for pos in ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN")
    )
    spot = _spot_with(
        "Call", {"Fold": 0.2, "Call": 0.6, "Raise 100%": 0.2}, history, actor="SB"
    )
    assert (
        classify_archetype(spot, villain=None, hero_equity_vs_villain=None)
        == "sb_complete"
    )


def test_archetype_sb_complete_is_sb_only():
    """A non-blind first-in with a dominant Call never gets sb_complete."""
    spot = _spot_with("Call", {"Call": 0.9, "Fold": 0.1}, actor="CO")
    assert (
        classify_archetype(spot, villain=None, hero_equity_vs_villain=None)
        != "sb_complete"
    )


def test_archetype_3bet_for_value():
    """Facing one raise, dominant Raise with high equity -> 3bet_for_value."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.FOLD),
        ParsedAction("CO", PreflopActionType.FOLD),
    )
    spot = _spot_with("Raise 77%", {"Raise 77%": 1.0, "Fold": 0.0}, history)
    villain = history[0]
    assert (
        classify_archetype(spot, villain, hero_equity_vs_villain=0.62)
        == "3bet_for_value"
    )


def test_archetype_3bet_as_bluff():
    """Facing one raise, dominant Raise with low equity -> 3bet_as_bluff."""
    history = (ParsedAction("UTG", PreflopActionType.RAISE, 60.0),)
    spot = _spot_with("Raise 77%", {"Raise 77%": 1.0, "Fold": 0.0}, history)
    assert (
        classify_archetype(spot, history[0], hero_equity_vs_villain=0.35)
        == "3bet_as_bluff"
    )


def test_archetype_4bet_for_value():
    """Facing two raises (open + 3-bet), dominant Raise -> 4bet."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.RAISE, 77.0),
        ParsedAction("CO", PreflopActionType.FOLD),
    )
    spot = _spot_with("Raise 50%", {"Raise 50%": 1.0, "Fold": 0.0}, history)
    assert (
        classify_archetype(spot, history[1], hero_equity_vs_villain=0.55)
        == "4bet_for_value"
    )


def test_archetype_squeeze_for_value():
    """Open + caller + hero raises = squeeze."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.CALL),
        ParsedAction("CO", PreflopActionType.FOLD),
    )
    spot = _spot_with("Raise 85%", {"Raise 85%": 1.0, "Fold": 0.0}, history)
    arch = classify_archetype(spot, history[0], hero_equity_vs_villain=0.58)
    assert arch == "squeeze_for_value"


def test_archetype_fold_dominated():
    """Dominant Fold with very low equity -> fold_dominated.

    A 'Call' action IS offered at the node (a normal facing-an-open fold), so
    the no-call guard does NOT fire and the equity split applies.
    """
    history = (ParsedAction("UTG", PreflopActionType.RAISE, 60.0),)
    spot = _spot_with("Fold", {"Fold": 1.0, "Call": 0.0}, history)
    arch = classify_archetype(spot, history[0], hero_equity_vs_villain=0.25)
    assert arch == "fold_dominated"


def test_archetype_fold_pot_odds_when_call_offered_and_speculative():
    """Dominant Fold, decent equity, a call offered, AND a SPECULATIVE hand
    (set-miner / implied-odds) -> fold_pot_odds. The pot-odds/set-mine frame
    only fits speculative hands (a pocket pair here)."""
    history = (ParsedAction("UTG", PreflopActionType.RAISE, 60.0),)
    spot = _spot_with(
        "Fold", {"Fold": 1.0, "Call": 0.0}, history, hand_class="55", combo="5h5c"
    )
    arch = classify_archetype(spot, history[0], hero_equity_vs_villain=0.47)
    assert arch == "fold_pot_odds"


def test_archetype_dominated_highcard_fold_not_pot_odds():
    """A DOMINATED high-card fold (A9o) with decent equity is NOT a set-miner,
    so it routes to fold_dominated -- NOT fold_pot_odds. Guards the June-2026
    fix: A9o folding to a 3-bet was wrongly getting the 'call and set-mine'
    exploit frame."""
    history = (ParsedAction("UTG", PreflopActionType.RAISE, 60.0),)
    spot = _spot_with(
        "Fold", {"Fold": 1.0, "Call": 0.0}, history, hand_class="A9o", combo="Ah9c"
    )
    arch = classify_archetype(spot, history[0], hero_equity_vs_villain=0.43)
    assert arch == "fold_dominated"


def test_archetype_fold_no_continue_when_no_call_offered():
    """The June 2026 A5o misfire: SB opens, BB 3-bets, hero SB faces a
    fold-or-4bet spot (NO call action). A dominant Fold here is NOT a pot-odds
    decision -- there is no price to call -- so it must route to
    fold_no_continue, never fold_pot_odds (whose 'price you'd need to call'
    frame produced a reversed equity-vs-price claim). The high equity (47.66%)
    would otherwise have selected fold_pot_odds."""
    history = (
        ParsedAction("BB", PreflopActionType.RAISE, 200.0),  # BB 3-bets the SB open
    )
    spot = _spot_with(
        "Fold",
        {"Fold": 0.84, "Raise 5x": 0.16},  # 4-bet-or-fold: NO 'Call' key
        history,
        actor="SB",
    )
    arch = classify_archetype(spot, history[0], hero_equity_vs_villain=0.4766)
    assert arch == "fold_no_continue"


def test_archetype_call_for_value():
    """Dominant Call with positive equity -> call_for_value."""
    history = (ParsedAction("UTG", PreflopActionType.RAISE, 60.0),)
    spot = _spot_with("Call", {"Call": 1.0, "Fold": 0.0}, history)
    arch = classify_archetype(spot, history[0], hero_equity_vs_villain=0.55)
    assert arch == "call_for_value"


def test_archetype_call_allin_facing_a_jam():
    """Calling an all-in -> call_allin (a pure pot-odds spot), NOT
    call_for_implied_odds -- there are no future streets to realize equity on,
    so the implied-odds / draw-chasing frame would be nonsense."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("CO", PreflopActionType.ALL_IN),
    )
    spot = _spot_with("Call", {"Call": 1.0, "Fold": 0.0}, history)
    # 40% equity would otherwise route to call_for_implied_odds; the all-in
    # in the history overrides that.
    arch = classify_archetype(spot, history[1], hero_equity_vs_villain=0.40)
    assert arch == "call_allin"


def test_archetype_call_for_implied_odds_when_no_jam():
    """A speculative call of a RAISE (no all-in) still -> call_for_implied_odds."""
    history = (ParsedAction("UTG", PreflopActionType.RAISE, 60.0),)
    spot = _spot_with("Call", {"Call": 1.0, "Fold": 0.0}, history)
    arch = classify_archetype(spot, history[0], hero_equity_vs_villain=0.40)
    assert arch == "call_for_implied_odds"


# --- H2: multi-way splits use FIELD equity, not heads-up --------------------
def test_archetype_multiway_call_uses_field_equity():
    """A multi-way CALL that's value vs the lone raiser (0.52) but behind the
    FIELD (0.41) is an implied-odds call, not a value call. H2: the split must
    use field equity on multi-way must-beat-everyone spots."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.CALL),
    )
    spot = _spot_with("Call", {"Call": 1.0, "Fold": 0.0}, history)
    arch = classify_archetype(
        spot, history[0],
        hero_equity_vs_villain=0.52,   # would be call_for_value heads-up
        hero_equity_vs_field=0.41,     # but behind the field
        is_multiway=True,
    )
    assert arch == "call_for_implied_odds"


def test_archetype_multiway_fold_uses_field_equity():
    """Heads-up the fold reads as fold_pot_odds (0.42 >= 0.40); vs the field
    (0.36) it's a domination fold. H2 routes the split to field equity."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.CALL),
    )
    spot = _spot_with("Fold", {"Fold": 1.0, "Call": 0.0}, history)
    arch = classify_archetype(
        spot, history[0],
        hero_equity_vs_villain=0.42,
        hero_equity_vs_field=0.36,
        is_multiway=True,
    )
    assert arch == "fold_dominated"


def test_archetype_multiway_squeeze_keeps_headsup_equity():
    """A RAISE (squeeze) isolates -- hero is usually heads-up vs the raiser
    when called -- so it KEEPS the heads-up number (0.55 -> value), NOT the
    field number (0.35, which would wrongly read as a bluff)."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.CALL),
    )
    spot = _spot_with("Raise 85%", {"Raise 85%": 1.0, "Fold": 0.0}, history)
    arch = classify_archetype(
        spot, history[0],
        hero_equity_vs_villain=0.55,   # value vs the raiser -> squeeze_for_value
        hero_equity_vs_field=0.35,     # ignored for a raise
        is_multiway=True,
    )
    assert arch == "squeeze_for_value"


def test_archetype_headsup_call_ignores_field_equity():
    """Heads-up (is_multiway=False), the heads-up number drives the split even
    if a field number is passed -- backward compatible."""
    history = (ParsedAction("UTG", PreflopActionType.RAISE, 60.0),)
    spot = _spot_with("Call", {"Call": 1.0, "Fold": 0.0}, history)
    arch = classify_archetype(
        spot, history[0],
        hero_equity_vs_villain=0.52,
        hero_equity_vs_field=0.30,     # present but ignored (heads-up)
        is_multiway=False,
    )
    assert arch == "call_for_value"


def test_archetype_all_in_for_value():
    """Dominant AllIn with positive equity -> all_in_for_value."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("HJ", PreflopActionType.RAISE, 77.0),
    )
    spot = _spot_with("AllIn", {"AllIn": 1.0, "Fold": 0.0}, history)
    arch = classify_archetype(spot, history[1], hero_equity_vs_villain=0.56)
    assert arch == "all_in_for_value"


# --- chunk 2: extract_facts populates new fields ---------------------------
def test_extract_facts_populates_chunk2_fields_real_pack():
    """End-to-end: hero_range_eq, blockers, archetype all populated."""
    ranges = REPO_ROOT / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import discover_packs

    packs = discover_packs(ranges)
    if not packs:
        pytest.skip("Ryan pack not present")
    pack = packs[0]
    # Enumerate only the pack under test: other registered packs (9-max,
    # the short-stack 6-max packs) can share node geometry and would
    # otherwise be picked by the predicate below with the wrong pack object.
    nodes = enumerate_nodes([pack])
    node = next(
        n
        for n in nodes
        if n.actor == "BTN"
        and len(n.history_before) == 3
        and n.history_before[0].action_type is PreflopActionType.RAISE
        and n.history_before[1].action_type is PreflopActionType.FOLD
        and n.history_before[2].action_type is PreflopActionType.FOLD
    )
    facts = extract_facts(sample_spot(node, "AA"), pack, equity_runouts=30)
    # range equity present
    assert facts.hero_range_equity_vs_villain is not None
    assert 0.0 <= facts.hero_range_equity_vs_villain <= 1.0
    # blockers: AA hero blocks some AA combos in villain's range
    assert "AA" in facts.blockers
    assert facts.blockers["AA"] >= 1
    # archetype: AA facing UTG open -> 3bet_for_value (assuming dominant is Raise)
    # OR call_for_value, depending on pack strategy. Just assert it's not empty.
    assert facts.archetype != ""
    assert facts.archetype != "unclassified"


def test_spot_rng_is_deterministic_per_spot():
    """Same spot -> same RNG stream; different combo -> different stream.

    The June 2026 fix for threshold instability: equity Monte-Carlo is
    seeded by (node_id, hand_class, combo), so archetype frames, equity
    tags, ev_gap, and difficulty can't flip between recomputations of
    the same spot.
    """
    from pipeline.preflop.fact_extractor import _spot_rng

    spot_a = _spot_with("Call", {"Call": 0.7, "Fold": 0.3})
    spot_b = _spot_with("Call", {"Call": 0.7, "Fold": 0.3})
    seq_a = [_spot_rng(spot_a).random() for _ in range(3)]
    seq_b = [_spot_rng(spot_b).random() for _ in range(3)]
    assert seq_a[0] == seq_b[0]  # identical identity -> identical stream

    import dataclasses
    spot_c = dataclasses.replace(spot_a, hero_card_combo="AsKd")
    assert _spot_rng(spot_c).random() != seq_a[0]


def test_equity_vs_range_reproducible_with_seeded_rng():
    import random

    from pipeline.preflop.fact_extractor import compute_hero_equity_vs_range

    villain = {"AhAd": 1.0, "KhKd": 1.0, "QsJs": 0.5}
    eq1 = compute_hero_equity_vs_range(
        "AsKs", villain, max_runouts=50, rng=random.Random("seed")
    )
    eq2 = compute_hero_equity_vs_range(
        "AsKs", villain, max_runouts=50, rng=random.Random("seed")
    )
    assert eq1 == eq2


def test_top_value_combos_restricts_blockers_to_real_value() -> None:
    """blocks_villain_top_value should count blocking villain's STRONG hands,
    not the bottom of the range (22 clipping A2s). top_value_combos keeps only
    the strongest ~35% by Chen strength, so compute_blockers counts only
    meaningful blocks."""
    from pipeline.preflop.fact_extractor import (
        combo_str_to_hand_class,
        compute_blockers,
        top_value_combos,
    )

    suits = "cdhs"

    def all_combos(hc: str) -> list[str]:
        if len(hc) == 2:  # pair -> 6 combos
            r = hc[0]
            return [r + a + r + b for i, a in enumerate(suits) for b in suits[i + 1:]]
        hi, lo, kind = hc[0], hc[1], hc[2]
        if kind == "s":  # suited -> 4
            return [hi + s + lo + s for s in suits]
        return [hi + a + lo + b for a in suits for b in suits if a != b]  # offsuit 12

    rng: dict[str, float] = {}
    for hc in ("AA", "KK", "QQ", "AKs", "A2s", "72o"):
        for combo in all_combos(hc):
            rng[combo] = 1.0

    top = top_value_combos(rng)
    top_classes = {combo_str_to_hand_class(c) for c in top}
    # Premiums are "top value"; the bottom of the range is not.
    assert "AA" in top_classes
    assert "A2s" not in top_classes
    assert "72o" not in top_classes
    # 22 blocks NONE of the top value (no A/K/Q) -> the tag stays off (the bug).
    assert compute_blockers("2c2d", top) == {}
    # An ace+king holder DOES block top value (AA + KK via the ace / king).
    blk = compute_blockers("AsKs", top)
    assert blk.get("AA", 0) > 0 and blk.get("KK", 0) > 0
