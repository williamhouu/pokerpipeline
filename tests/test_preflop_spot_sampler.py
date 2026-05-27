"""Tests for pipeline.preflop.spot_sampler."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.grammars.types import PreflopActionType  # noqa: E402
from pipeline.preflop.node_enumerator import (  # noqa: E402
    enumerate_nodes,
)
from pipeline.preflop.pack import PreflopPack, clear_registry  # noqa: E402
from pipeline.preflop.spot_sampler import (  # noqa: E402
    PreflopSpot,
    enumerate_spots_for_node,
    random_combo_for_class,
    representative_combo_for_class,
    sample_spot,
    sample_spots_across_nodes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


# ----- representative_combo_for_class -------------------------------------
def test_pair_combo_is_clubs_diamonds():
    assert representative_combo_for_class("AA") == "AcAd"
    assert representative_combo_for_class("22") == "2c2d"
    assert representative_combo_for_class("TT") == "TcTd"


def test_suited_combo_is_spades():
    assert representative_combo_for_class("AKs") == "AsKs"
    assert representative_combo_for_class("76s") == "7s6s"


def test_offsuit_combo_is_hearts_clubs():
    assert representative_combo_for_class("AKo") == "AhKc"
    assert representative_combo_for_class("72o") == "7h2c"


def test_representative_combo_rejects_malformed():
    with pytest.raises(ValueError):
        representative_combo_for_class("")
    with pytest.raises(ValueError):
        representative_combo_for_class("AKx")
    with pytest.raises(ValueError):
        representative_combo_for_class("AAA")


# ----- random_combo_for_class ----------------------------------------------
def test_random_pair_uses_distinct_suits():
    rng = random.Random(42)
    for _ in range(20):
        combo = random_combo_for_class("AA", rng=rng)
        # Format: rank + suit_a + rank + suit_b, both ranks = A, suits differ.
        assert combo[0] == "A" and combo[2] == "A"
        assert combo[1] != combo[3]


def test_random_suited_uses_same_suit():
    rng = random.Random(42)
    for _ in range(20):
        combo = random_combo_for_class("AKs", rng=rng)
        assert combo[0] == "A" and combo[2] == "K"
        assert combo[1] == combo[3]  # same suit


def test_random_offsuit_uses_different_suits():
    rng = random.Random(42)
    for _ in range(20):
        combo = random_combo_for_class("AKo", rng=rng)
        assert combo[0] == "A" and combo[2] == "K"
        assert combo[1] != combo[3]


# ----- sample_spot (synthetic pack) ---------------------------------------
def _make_node_pack(tmp_path: Path) -> PreflopPack:
    """Build a tmp pack with a known node: BTN faces a 60% open, can fold or
    call. AKo always calls; 72o always folds; AA always calls (artificial)."""
    btn = tmp_path / "BTN"
    btn.mkdir(parents=True)
    # BTN Call range: AA = 100%, AKo = 100%, 72o = 0%
    (btn / "UTG_60%_HJ_Fold_CO_Fold_BTN_Call.txt").write_text(
        ",".join(
            f"{cls}:1.0" if cls in ("AA", "AKo") else f"{cls}:0.0" for cls in _all_169()
        )
    )
    # BTN Fold range: AA = 0%, AKo = 0%, 72o = 100%
    (btn / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt").write_text(
        ",".join(f"{cls}:1.0" if cls == "72o" else f"{cls}:0.0" for cls in _all_169())
    )
    return PreflopPack(
        pack_id="synth",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )


def _all_169() -> list[str]:
    """The canonical 169 classes -- exported here so tests don't have to
    import the pipeline helper repeatedly."""
    from pipeline.preflop_ranges import canonical_169_hand_classes

    return canonical_169_hand_classes()


def test_sample_spot_basic_frequencies(tmp_path):
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "AKo")
    assert spot.hero_hand_class == "AKo"
    assert spot.hero_card_combo == "AhKc"
    # AKo's range file weight: Call=1.0, Fold=0.0
    assert spot.action_frequencies == {"Call": 1.0, "Fold": 0.0}
    assert spot.dominant_action == "Call"
    assert spot.dominant_frequency == 1.0


def test_sample_spot_fold_dominant(tmp_path):
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "72o")
    assert spot.dominant_action == "Fold"
    assert spot.dominant_frequency == 1.0


def test_sample_spot_zero_total_unknown_class(tmp_path):
    """A hand class not present in the range files yields all-zero freqs."""
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "JTs")
    assert spot.action_frequencies == {"Call": 0.0, "Fold": 0.0}
    # Total is zero; dominant just picks one (max of equal zeros).
    assert spot.dominant_frequency == 0.0


def test_sample_spot_explicit_combo_overrides(tmp_path):
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "AKo", combo="AdKs")
    assert spot.hero_card_combo == "AdKs"


def test_sample_spot_rejects_unknown_class(tmp_path):
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    with pytest.raises(ValueError, match="unknown hand_class"):
        sample_spot(node, "XX")


# ----- enumerate_spots_for_node -------------------------------------------
def test_enumerate_spots_yields_169_with_no_filter(tmp_path):
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spots = list(enumerate_spots_for_node(node, min_total_weight=0.0))
    assert len(spots) == 169


def test_enumerate_spots_filter_drops_zero_total(tmp_path):
    """With min_total_weight=0.5, only hands with >= 50% combined weight pass."""
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spots = list(enumerate_spots_for_node(node, min_total_weight=0.5))
    # AA, AKo (each Call=1.0) and 72o (Fold=1.0) all have total=1.0.
    # Other hands have total=0.0.
    classes = {s.hero_hand_class for s in spots}
    assert classes == {"AA", "AKo", "72o"}


def test_sample_spots_across_nodes_default_filter(tmp_path):
    """Default min_total_weight=0.01 includes hands with any presence."""
    pack = _make_node_pack(tmp_path)
    nodes = enumerate_nodes([pack])
    spots = list(sample_spots_across_nodes(nodes))
    # 3 hands have nonzero weight in our synthetic pack (AA, AKo, 72o).
    assert {s.hero_hand_class for s in spots} == {"AA", "AKo", "72o"}


# ----- normalisation: presence-weighted -> conditional --------------------
def _make_partial_presence_pack(tmp_path: Path) -> PreflopPack:
    """Synthetic pack where AKo reaches the node only 40% of the time
    (raw weights: Call=0.30, Fold=0.10), and 72o reaches 100% but folds
    purely (Fold=1.0). After normalisation:
      AKo: Call=0.75, Fold=0.25, presence=0.40
      72o: Fold=1.00, Call=0.00, presence=1.00
    """
    btn = tmp_path / "BTN"
    btn.mkdir(parents=True)
    (btn / "UTG_60%_HJ_Fold_CO_Fold_BTN_Call.txt").write_text(
        ",".join(f"{cls}:0.30" if cls == "AKo" else f"{cls}:0.0" for cls in _all_169())
    )
    (btn / "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold.txt").write_text(
        ",".join(
            f"{cls}:0.10"
            if cls == "AKo"
            else f"{cls}:1.0"
            if cls == "72o"
            else f"{cls}:0.0"
            for cls in _all_169()
        )
    )
    return PreflopPack(
        pack_id="synth_partial",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )


def test_sample_spot_normalises_partial_presence(tmp_path):
    """The core bug fix. AKo's raw weights are Call=0.30, Fold=0.10 --
    presence-weighted. The CONDITIONAL strategy at the decision (given
    AKo reaches it) is Call=75%, Fold=25%. sample_spot must return the
    conditional strategy in action_frequencies."""
    pack = _make_partial_presence_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "AKo")
    assert spot.action_frequencies["Call"] == pytest.approx(0.75)
    assert spot.action_frequencies["Fold"] == pytest.approx(0.25)
    # Conditional strategy sums to 1.0.
    assert sum(spot.action_frequencies.values()) == pytest.approx(1.0)
    # dominant_frequency reads from the normalised dict.
    assert spot.dominant_action == "Call"
    assert spot.dominant_frequency == pytest.approx(0.75)


def test_sample_spot_records_presence_unnormalised(tmp_path):
    """presence is the sum of the RAW (pre-normalisation) weights -- the
    'how often does this hand reach the node' signal that drives the
    worthiness filter and the difficulty rating's presence check."""
    pack = _make_partial_presence_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "AKo")
    assert spot.presence == pytest.approx(0.40)
    # 72o reaches the node fully (Fold=1.0, Call=0.0 -> raw sum 1.0).
    trash = sample_spot(node, "72o")
    assert trash.presence == pytest.approx(1.0)


def test_sample_spot_zero_presence_keeps_zeros(tmp_path):
    """A hand class never present in the pack: presence=0, all freqs=0.
    We don't divide by zero -- the values stay at 0 so structural
    consumers (canonicalisation, etc.) still see the action set."""
    pack = _make_partial_presence_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "JTs")
    assert spot.presence == 0.0
    assert spot.action_frequencies == {"Call": 0.0, "Fold": 0.0}
    assert spot.dominant_frequency == 0.0


def test_sample_spot_full_presence_unchanged_by_normalisation(tmp_path):
    """When raw weights already sum to 1.0, normalisation is a no-op --
    full-presence hands keep their existing frequencies."""
    pack = _make_node_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "AKo")
    # AKo's raw weights sum to 1.0 (Call=1.0, Fold=0.0).
    assert spot.presence == pytest.approx(1.0)
    assert spot.action_frequencies == {"Call": 1.0, "Fold": 0.0}


def test_preflop_spot_auto_computes_presence_from_freqs():
    """Test fixtures that construct PreflopSpot directly without passing
    presence get the sum of action_frequencies as presence, via the
    sentinel default + __post_init__. This keeps fixture tests working
    without forcing every test to know about the new field."""
    from pipeline.preflop.node_enumerator import PreflopDecisionNode

    node = PreflopDecisionNode(pack_id="t", actor="BTN", history_before=(), actions=())
    spot = PreflopSpot(
        node=node,
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies={"Call": 0.7, "Fold": 0.3},
        dominant_action="Call",
        dominant_frequency=0.7,
    )
    # Auto-computed from action_frequencies sum.
    assert spot.presence == pytest.approx(1.0)


def test_preflop_spot_respects_explicit_presence():
    """When sample_spot (or any caller) sets presence explicitly, the
    sentinel default is ignored and __post_init__ doesn't auto-compute."""
    from pipeline.preflop.node_enumerator import PreflopDecisionNode

    node = PreflopDecisionNode(pack_id="t", actor="BTN", history_before=(), actions=())
    spot = PreflopSpot(
        node=node,
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies={"Call": 0.75, "Fold": 0.25},  # normalised
        dominant_action="Call",
        dominant_frequency=0.75,
        presence=0.40,  # explicit
    )
    assert spot.presence == pytest.approx(0.40)


def test_enumerate_spots_filter_uses_presence_not_freq_sum(tmp_path):
    """The min_total_weight filter uses spot.presence (un-normalised).
    A hand with presence < threshold gets dropped regardless of its
    (always-near-1.0) conditional frequencies."""
    pack = _make_partial_presence_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    # AKo has presence=0.40, 72o has presence=1.0. Threshold 0.5
    # should drop AKo but keep 72o.
    spots = list(enumerate_spots_for_node(node, min_total_weight=0.5))
    classes = {s.hero_hand_class for s in spots}
    assert "AKo" not in classes
    assert "72o" in classes


def test_worthiness_filter_drops_pure_fold_after_normalisation(tmp_path):
    """The original bug: a hand that reaches a node 62% of the time but
    folds 100% of those times had raw 'Fold: 0.62' -- which falsely
    looked like a 62%-mixed strategy and slipped through the 55-95%
    worthiness window. After normalisation it's correctly Fold: 1.00
    -> dominant_frequency=1.0 -> fails the 95% ceiling -> dropped."""
    from pipeline.preflop.question_extractor import is_question_worthy

    pack = _make_partial_presence_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    # AKo: pre-fix raw was {Call:0.30, Fold:0.10}, dominant_freq=0.30
    #      post-fix conditional is {Call:0.75, Fold:0.25}, dominant_freq=0.75
    spot_akoff = sample_spot(node, "AKo")
    # 0.75 IS in [0.55, 0.95] window -> worthy.
    assert is_question_worthy(spot_akoff) is True
    # 72o: raw {Fold:1.0, Call:0.0}, conditional {Fold:1.0, Call:0.0},
    # dominant_freq=1.0 -> fails the 95% ceiling.
    spot_trash = sample_spot(node, "72o")
    assert is_question_worthy(spot_trash) is False


# ----- integration: real pack ---------------------------------------------
def test_sample_spot_against_real_pack():
    """End-to-end on the real Ryan pack. Sanity-check premium hands raise,
    trash hands fold, and the frequencies look like real GTO numbers."""
    ranges = REPO_ROOT / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    from pipeline.preflop.pack import discover_packs

    packs = discover_packs(ranges)
    if not packs:
        pytest.skip("Ryan pack not present under ranges/")
    nodes = enumerate_nodes(packs)
    # Pick a clear node: BTN facing UTG fold, HJ open 60%, CO fold (so BTN
    # is acting after a single open from HJ).
    node = next(
        n
        for n in nodes
        if n.actor == "BTN"
        and len(n.history_before) == 3
        and n.history_before[0].position == "UTG"
        and n.history_before[0].action_type is PreflopActionType.FOLD
        and n.history_before[1].position == "HJ"
        and n.history_before[1].action_type is PreflopActionType.RAISE
        and len(n.actions) >= 3
    )
    # AA at this node should be dominantly raising (3-bet) or calling --
    # NEVER folding. Folding AA preflop in cash never happens.
    aa = sample_spot(node, "AA")
    fold_freq = aa.action_frequencies.get("Fold", 0.0)
    assert fold_freq < 0.05, f"AA should almost never fold preflop, got {fold_freq:.2%}"
    # 72o should be dominantly folding facing an open.
    trash = sample_spot(node, "72o")
    assert trash.dominant_action == "Fold"
    assert trash.dominant_frequency > 0.9


def test_real_pack_question_worthy_count():
    """At a real complex node, count hands with dominant in [55%, 95%]."""
    ranges = REPO_ROOT / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    from pipeline.preflop.pack import discover_packs

    packs = discover_packs(ranges)
    if not packs:
        pytest.skip("Ryan pack not present under ranges/")
    nodes = enumerate_nodes(packs)
    # Mid-game node where we'd expect some genuinely mixed strategies.
    node = next(
        n
        for n in nodes
        if n.actor == "BTN" and len(n.actions) == 4 and len(n.history_before) >= 2
    )
    spots = list(enumerate_spots_for_node(node))
    worthy = [s for s in spots if 0.55 <= s.dominant_frequency <= 0.95]
    # A real complex BTN node should have at least a few mixed-strategy hands.
    assert len(worthy) >= 1, (
        f"expected some question-worthy hands at {node.node_id}, got 0"
    )
