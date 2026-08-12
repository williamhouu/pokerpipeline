"""Tests for the PLO preflop chart export (pipeline/plo/chart_export.py).

Covers: bucket precedence, the 3-way suits partition (and its parity with
hand_model's flush_suits), mix/aggregate largest-remainder behaviour, the
in-range rule, path-key connectivity and zero-reach pruning on a small
synthetic fixture pack, plus a real-pack smoke test on plo_6max_12bb (the
smallest pack on disk; skipped when the extraction folder is absent).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from pipeline.plo.chart_export import (
    CHART_BUCKETS,
    build_dev_charts,
    build_pack_class_chart,
    chart_bucket,
    chart_suits,
    dev_hand_class,
    dev_node_key,
    dev_seat_name,
    dev_suit_suffix,
    export_pack,
    replay_dev_node_key,
    safe_node_key,
    spec_for_pack_id,
    validate_dev_charts,
    validate_pack_export,
)
from pipeline.plo.chart_export_readme import README_MD
from pipeline.plo.hand_model import classify_plo_hand, flush_suits
from pipeline.plo.hand_order import HAND_COUNT, hand_order
from pipeline.plo.pack import PloPack, discover_plo_pack

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bucket(hand: str) -> str:
    return chart_bucket(classify_plo_hand(hand))


# --- buckets ---------------------------------------------------------------
class TestBucketPrecedence:
    def test_aaxx_wins_over_everything(self) -> None:
        assert _bucket("AhAsJdJc") == "AAxx"
        assert _bucket("AhAsKdKc") == "AAxx"  # AAKK: AAxx, not KKxx
        assert _bucket("AhAsAdKc") == "AAxx"  # trips of aces
        assert _bucket("AhAsKdQc") == "AAxx"

    def test_kkxx_after_aaxx(self) -> None:
        assert _bucket("KhKsQdQc") == "KKxx"  # KKQQ: KKxx, not QQ-TT
        assert _bucket("KhKs2d3c") == "KKxx"
        assert _bucket("AhKsKdQc") == "KKxx"  # one ace only

    def test_qq_tt_band(self) -> None:
        assert _bucket("QhQsJdTc") == "QQ-TT"
        assert _bucket("JhJs7d2c") == "QQ-TT"
        assert _bucket("ThTs9d9c") == "QQ-TT"  # highest pair T wins over Two pair
        assert _bucket("QhQsJdJc") == "QQ-TT"  # QQJJ: highest pair is Q

    def test_two_pair_is_remaining_double_paired(self) -> None:
        assert _bucket("9h9s8d8c") == "Two pair"
        assert _bucket("9h9s2d2c") == "Two pair"

    def test_low_pair_single_pair_nine_or_below(self) -> None:
        assert _bucket("8h8sKdQc") == "Low pair"
        assert _bucket("2h2s7dJc") == "Low pair"
        assert _bucket("9h9s9dKc") == "Low pair"  # low trips
        assert _bucket("5h5s5d5c") == "Low pair"  # low quads

    def test_rundowns(self) -> None:
        assert _bucket("JhTs9d8c") == "Rundown"
        assert _bucket("KhQsJdTc") == "Rundown"  # broadway perfect rundown
        assert _bucket("AhKsQdJc") == "Rundown"  # AKQJ is a rundown
        assert _bucket("Jh9s8d7c") == "Rundown"  # one-gapper, not all broadway
        assert _bucket("Ah2s3d4c") == "Rundown"  # ace plays low

    def test_broadway_is_the_one_gapper_broadways(self) -> None:
        assert _bucket("AhKsQdTc") == "Broadway"
        assert _bucket("AhKsJdTc") == "Broadway"
        assert _bucket("AhQsJdTc") == "Broadway"

    def test_dangler(self) -> None:
        assert _bucket("KhQsJd2c") == "Dangler"
        assert _bucket("AhKsQd2c") == "Dangler"  # trio A-K-Q + dead deuce
        assert _bucket("JhTs9d2c") == "Dangler"

    def test_other(self) -> None:
        assert _bucket("Jh9s8d6c") == "Other"  # two-gapper
        assert _bucket("KhQs7d2c") == "Other"  # two dead cards
        assert _bucket("QhTs8d6c") == "Other"  # evenly spread, no far card
        assert _bucket("Qh7s5d3c") == "Dangler"  # the queen is the dead card

    def test_every_class_gets_a_bucket(self) -> None:
        from pipeline.plo.hand_order import cards_at

        seen: set[str] = set()
        for i in range(0, HAND_COUNT, 97):
            seen.add(chart_bucket(classify_plo_hand(cards_at(i))))
        assert seen <= set(CHART_BUCKETS)


# --- suits -----------------------------------------------------------------
class TestSuits:
    @pytest.mark.parametrize(
        ("hand", "expected"),
        [
            ("AhKhAsKs", "double-suited"),
            ("AhKhQsJd", "single-suited"),
            ("AhKhQh2d", "single-suited"),  # three of a suit
            ("AhKhQhJh", "single-suited"),  # monotone
            ("AhKsQdJc", "rainbow"),
        ],
    )
    def test_partition(self, hand: str, expected: str) -> None:
        assert chart_suits(classify_plo_hand(hand)) == expected

    def test_matches_flush_suits_count(self) -> None:
        """The 3-way partition must equal counting hand_model flush suits."""
        from pipeline.plo.hand_order import cards_at

        by_count = {2: "double-suited", 1: "single-suited", 0: "rainbow"}
        for i in range(0, HAND_COUNT, 211):
            cards = cards_at(i)
            expected = by_count[len(flush_suits(cards))]
            assert chart_suits(classify_plo_hand(cards)) == expected


# --- fixture pack ----------------------------------------------------------
def _write_rng(path: Path, weights: dict[int, float]) -> None:
    """A minimal .rng: 2 lines per hand (pattern, then 'p;ev')."""
    lines = []
    for i in range(HAND_COUNT):
        p = weights.get(i, 0.0)
        lines.append("????")
        lines.append(f"{p};0.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def fixture_pack(tmp_path: Path) -> PloPack:
    """Tiny 6-max tree: LJ open node, HJ response node, one dead branch.

    Files (stems are dot-joined tokens; 40100 = pot raise):
      root LJ node: ``0`` (fold) + ``40100`` (open)
      after LJ folds, HJ node: ``0.0`` / ``0.40100`` (ALL-ZERO: pruned)
      after LJ opens, HJ node: ``40100.0`` / ``40100.1`` / ``40100.40100``
    Hand 0 opens pure; hand 1 mixes open/fold; hand 2 folds pure (still in
    range at the root). Facing the open, hand 0 mixes call/3-bet, hand 1
    calls pure.
    """
    _write_rng(tmp_path / "0.rng", {1: 0.4, 2: 1.0})
    _write_rng(tmp_path / "40100.rng", {0: 1.0, 1: 0.6})
    _write_rng(tmp_path / "0.0.rng", {})
    _write_rng(tmp_path / "0.40100.rng", {})
    _write_rng(tmp_path / "40100.0.rng", {2: 1.0})
    _write_rng(tmp_path / "40100.1.rng", {0: 0.25, 1: 1.0})
    _write_rng(tmp_path / "40100.40100.rng", {0: 0.75})
    return PloPack(root=tmp_path, label="fixture")


class TestFixtureExport:
    def test_export_tree_and_hands(self, fixture_pack: PloPack, tmp_path: Path) -> None:
        out = tmp_path / "out"
        stats = export_pack(fixture_pack, out)  # validates internally
        pack_dir = out / "plo_6max_100bb"
        index = json.loads((pack_dir / "index.json").read_text())

        # Tree: root + the HJ-facing-open node; the dead "0" branch pruned.
        assert set(index["nodes"]) == {"", "40100"}
        assert stats.nodes_pruned_zero_reach == 1
        assert index["format"] == "Cash"
        assert index["game"] == "PLO"
        assert "ante_bb" not in index

        root = index["nodes"][""]
        assert root["pos"] == "UTG"  # LJ displays as UTG at 6-max
        assert root["bl"] == 0
        assert root["pot_bb"] == 1.5
        # acts in ladder order; open resolves to 3.5bb via the shared walk
        assert [a["k"] for a in root["acts"]] == ["fold", "raise"]
        assert root["acts"][1]["to_bb"] == 3.5
        assert sum(root["aggregate"].values()) == 100

        child = index["nodes"]["40100"]
        assert child["pos"] == "HJ"
        assert child["bl"] == 1
        assert child["to_call_bb"] == 3.5
        assert child["pot_bb"] == 5.0
        # pot 3-bet over a 3.5bb open = 3.5 + (5 + 3.5) = 12bb
        raise_act = [a for a in child["acts"] if a["k"] == "raise"][-1]
        assert raise_act["to_bb"] == 12

        # Hands file: in-range rule + conditional mixes.
        with gzip.open(pack_dir / root["hands_file"], "rt") as fh:
            root_hands = json.load(fh)
        by_h = {e["h"]: e for e in root_hands["hands"]}
        order = hand_order()
        assert set(by_h) == {order[0], order[1], order[2]}
        assert by_h[order[0]]["m"] == {"Raise 100%": 100}
        assert by_h[order[1]]["m"] == {"Raise 100%": 60, "Fold": 40}
        assert by_h[order[2]]["m"] == {"Fold": 100}  # pure folds ARE in range

        with gzip.open(pack_dir / child["hands_file"], "rt") as fh:
            child_hands = json.load(fh)
        by_h2 = {e["h"]: e for e in child_hands["hands"]}
        # hand 2 folded the open away entirely; only hands 0 and 1 continue
        # to face... no: hand 2 IS in range facing the open? Hand 2 never
        # opens (p=0 in 40100.rng) but the RESPONSE node is HJ's, whose
        # reach is HJ's own prior actions (none). Hand 2 folds pure there.
        assert by_h2[order[2]]["m"] == {"Fold": 100}
        assert by_h2[order[0]]["m"] == {"Raise 100%": 75, "Call": 25}
        assert by_h2[order[1]]["m"] == {"Call": 100}
        # entry static fields present
        entry = by_h2[order[0]]
        assert entry["b"] in CHART_BUCKETS
        assert entry["s"] in {"double-suited", "single-suited", "rainbow"}
        assert set(entry["f"]) == {"pair", "suit", "conn"}
        assert entry["n"] >= 1

    def test_validation_catches_broken_connectivity(
        self, fixture_pack: PloPack, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        export_pack(fixture_pack, out)
        pack_dir = out / "plo_6max_100bb"
        index_path = pack_dir / "index.json"
        index = json.loads(index_path.read_text())
        # Corrupt: orphan key whose parent act token does not exist.
        index["nodes"]["40100.9"] = index["nodes"]["40100"]
        index_path.write_text(json.dumps(index))
        with pytest.raises(AssertionError, match="not an act of its parent"):
            validate_pack_export(pack_dir, fixture_pack.spec)

    def test_validation_catches_size_drift(
        self, fixture_pack: PloPack, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        export_pack(fixture_pack, out)
        pack_dir = out / "plo_6max_100bb"
        index_path = pack_dir / "index.json"
        index = json.loads(index_path.read_text())
        index["nodes"]["40100"]["acts"][-1]["to_bb"] = 11.5
        index_path.write_text(json.dumps(index))
        with pytest.raises(AssertionError, match="to_bb"):
            validate_pack_export(pack_dir, fixture_pack.spec)


# --- misc ------------------------------------------------------------------
class TestMisc:
    def test_safe_node_key(self) -> None:
        assert safe_node_key("") == "root"
        assert safe_node_key("40100.0.1") == "40100_0_1"

    def test_spec_for_pack_id(self) -> None:
        assert spec_for_pack_id("plo_mtt_6max_25bb").ante_bb == 1.0
        with pytest.raises(KeyError):
            spec_for_pack_id("plo_9max_200bb")

    def test_readme_has_no_em_dashes_and_no_vendor_name(self) -> None:
        assert "—" not in README_MD
        assert "onker" not in README_MD  # never name the solve vendor


# --- the developer per-class chart (plo-charts-data.json) -------------------
def _dev(hand: str) -> str:
    return dev_hand_class(classify_plo_hand(hand))


class TestDevClassifier:
    """The app developer's classifier (plo-charts.md section 6) -- NOT the
    per-hand export's bucket taxonomy; precedence differs on purpose."""

    def test_trips_first(self) -> None:
        assert _dev("AhAsAdKc") == "Trips"  # AAAK is Trips, not AAxx
        assert _dev("AhAsAdAc") == "Trips"
        assert _dev("9h9s9dKc") == "Trips"

    def test_two_pair_beats_pair_buckets(self) -> None:
        assert _dev("AhAsKdKc") == "Two pair"  # AAKK under HIS rules
        assert _dev("QhQsJdJc") == "Two pair"
        assert _dev("9h9s2d2c") == "Two pair"

    def test_single_pair_split(self) -> None:
        assert _dev("AhAsKdQc") == "AAxx"
        assert _dev("KhKs7d2c") == "KKxx"
        assert _dev("QhQsJdTc") == "QQ-TT"
        assert _dev("ThTs9d3c") == "QQ-TT"
        assert _dev("9h9sKdQc") == "Low pair"

    def test_rundown_span_rule(self) -> None:
        assert _dev("JhTs9d8c") == "Rundown"  # span 3
        assert _dev("AhKsQdJc") == "Rundown"  # span 3
        assert _dev("AhKsQdTc") == "Rundown"  # span 4 (all-broadway one-gap)
        assert _dev("Jh9s8d7c") == "Rundown"  # span 4
        assert _dev("Qh9s8d7c") == "Other"  # span 5: not a rundown; gap3rd4th=1

    def test_broadway_is_structurally_unreachable(self) -> None:
        # Every 4-distinct all-T+ hand has span <= 4, so Rundown always wins
        # under his precedence. Documented data flag for the developer.
        for hand in ("AhKsQdJc", "AhKsQdTc", "AhKsJdTc", "AhQsJdTc", "KhQsJdTc"):
            assert _dev(hand) == "Rundown"

    def test_dangler_gap_rule(self) -> None:
        assert _dev("KhQsJd2c") == "Dangler"  # 3rd J(11) - 4th 2 = 9 >= 5
        assert _dev("AhKsQd7c") == "Dangler"  # Q(12) - 7 = 5
        assert _dev("AhKsQd8c") == "Other"  # Q(12) - 8 = 4: not a dangler
        assert _dev("Ah2s3d4c") == "Other"  # no ace-low reading in his spec

    def test_suit_suffixes(self) -> None:
        assert dev_suit_suffix(classify_plo_hand("AhKhAsKs")) == "ds"
        assert dev_suit_suffix(classify_plo_hand("AhKhQsJd")) == "ss"
        assert dev_suit_suffix(classify_plo_hand("AhKhQh2d")) == "ss"
        assert dev_suit_suffix(classify_plo_hand("AhKsQdJc")) == "rbw"


class TestDevNodeKey:
    def test_codes_and_ladder(self) -> None:
        from pipeline.plo.pack import parse_node_path

        # LJ folds, HJ opens, CO calls, BU 3-bets, SB folds, BB folds,
        # HJ (opener) faces the 3-bet -> the doc's example shape.
        seats = ("LJ", "HJ", "CO", "BU", "SB", "BB")
        hist = parse_node_path("0.40100.1.40100.0.0.1", seats=seats)[:-1]
        key = dev_node_key(hist)
        assert key == "UTG:f.HJ:r.CO:c.BTN:3.SB:f.BB:f"
        assert (
            replay_dev_node_key(key, seats=("LJ", "HJ", "CO", "BU", "SB", "BB")) == "HJ"
        )

    def test_limp_vs_call(self) -> None:
        from pipeline.plo.pack import parse_node_path

        seats = ("LJ", "HJ", "CO", "BU", "SB", "BB")
        # LJ limps (call before any raise), HJ opens, LJ's limp stays "l".
        hist = parse_node_path("1.40100.0", seats=seats)[:-1]
        assert dev_node_key(hist) == "UTG:l.HJ:r"

    def test_raise_cap_omits_deep_nodes(self) -> None:
        from pipeline.plo.pack import parse_node_path

        seats = ("LJ", "HJ", "CO", "BU", "SB", "BB")
        # open, 3-bet, 4-bet, 5-bet: encodable (4 raises).
        hist4 = parse_node_path("40100.40100.0.0.0.0.40100.40100.0", seats=seats)[:-1]
        key4 = dev_node_key(hist4)
        assert key4 is not None
        assert key4.endswith(":5") or ":5." in key4
        # a 5th raise in the history -> not encodable -> omitted
        hist5 = parse_node_path("40100.40100.0.0.0.0.40100.40100.3.0", seats=seats)[:-1]
        assert dev_node_key(hist5) is None

    def test_replay_rejects_wrong_seat_order(self) -> None:
        with pytest.raises(AssertionError, match="out of order"):
            replay_dev_node_key(
                "HJ:f.UTG:f", seats=("LJ", "HJ", "CO", "BU", "SB", "BB")
            )


class TestNineMaxSeatMapping:
    """The developer's c9 seat list says MP where our 9-max pack says UTG+2
    (same seat, third to act) -- every dev-chart surface must apply it."""

    def test_dev_seat_name(self) -> None:
        assert dev_seat_name("UTG+2", table_size=9) == "MP"
        assert dev_seat_name("UTG", table_size=9) == "UTG"
        assert dev_seat_name("LJ", table_size=9) == "LJ"  # the REAL 9-max LJ
        assert dev_seat_name("BTN", table_size=9) == "BTN"
        # 6-max path is the untouched display_seat remap (LJ->UTG, BU->BTN).
        assert dev_seat_name("LJ", table_size=6) == "UTG"
        assert dev_seat_name("BU", table_size=6) == "BTN"

    def test_nine_max_node_with_utg2_actor_emits_mp(self) -> None:
        from pipeline.plo.pack import SEATS_9MAX, parse_node_path

        # UTG folds, UTG+1 folds, UTG+2 (his MP) opens; LJ next to act.
        hist = parse_node_path("0.0.2.0", seats=SEATS_9MAX)[:-1]
        key = dev_node_key(hist, table_size=9)
        assert key == "UTG:f.UTG+1:f.MP:r"
        assert replay_dev_node_key(key, seats=SEATS_9MAX) == "LJ"
        # And a node where UTG+2 IS the hero replays to MP.
        hero_hist = parse_node_path("0.0.0", seats=SEATS_9MAX)[:-1]
        assert replay_dev_node_key(
            dev_node_key(hero_hist, table_size=9), seats=SEATS_9MAX
        ) == "MP"

    def test_validate_dev_charts_replays_mp_keys(self) -> None:
        doc = {
            "version": 1,
            "charts": {
                "c9": {
                    "100": {"MP": {"UTG:f.UTG+1:f": {"AAxx": [50, 0, 50]}}}
                }
            },
        }
        assert validate_dev_charts(doc) == 1
        # The pack-internal seat name is NOT a legal c9 heroSeat.
        bad_seat = {
            "version": 1,
            "charts": {
                "c9": {
                    "100": {"UTG+2": {"UTG:f.UTG+1:f": {"AAxx": [50, 0, 50]}}}
                }
            },
        }
        with pytest.raises(AssertionError, match="illegal heroSeat"):
            validate_dev_charts(bad_seat)
        bad_triple = {
            "version": 1,
            "charts": {
                "c9": {
                    "100": {"MP": {"UTG:f.UTG+1:f": {"AAxx": [50, 0, 49]}}}
                }
            },
        }
        with pytest.raises(AssertionError, match="bad triple"):
            validate_dev_charts(bad_triple)


class TestDevChartFixture:
    def test_class_triples_and_skeleton(self, fixture_pack: PloPack) -> None:
        doc = build_dev_charts((fixture_pack,))
        charts = doc["charts"]
        assert doc["version"] == 1
        assert set(charts) == {"c6"}
        depth = charts["c6"]["100"]

        # Root: UTG first-in, key "". Hands 0/1/2 are AAAA (Trips|rbw, 1
        # combo), AA(2A) (Trips|ss, 12 combos), 2AAA (Trips|rbw, 4 combos).
        root = depth["UTG"][""]
        # Trips|rbw = AAAA raise 100% (1 combo) + 2AAA fold 100% (4 combos)
        assert root["Trips|rbw"] == [80, 0, 20]
        # Trips|ss = AA(2A): fold 40% / raise 60% of 12 combos
        assert root["Trips|ss"] == [40, 0, 60]
        # bare merge: fold 4+4.8=8.8, call 0, raise 1+7.2=8.2 of 17
        assert root["Trips"] == [52, 0, 48]
        for triple in root.values():
            assert sum(triple) == 100
            assert all(v >= 0 for v in triple)

        # HJ facing the open: nodeKey "UTG:r"; call mass lands in slot 1.
        facing = depth["HJ"]["UTG:r"]
        assert facing["Trips|rbw"] == [80, 5, 15]
        assert facing["Trips|ss"] == [0, 100, 0]
        # The dead "0" branch (zero reach) is never emitted.
        assert "UTG:f" not in depth.get("HJ", {})

    def test_every_suffixed_class_has_a_bare_merge(self, fixture_pack: PloPack) -> None:
        chart = build_pack_class_chart(fixture_pack)
        for node_entry in (e for seat in chart.values() for e in seat.values()):
            for key in node_entry:
                if "|" in key:
                    assert key.split("|")[0] in node_entry

    def test_clean_line_filter_keeps_the_tree_connected(self) -> None:
        """clean_line_node reads the HISTORY only, so it is monotone along a
        line: a passing node's parent always passes (no orphan chart nodes).
        Pinned on 9-max lines that FAIL under hero-inclusive counting."""
        from pipeline.plo.chart_export import clean_line_node
        from pipeline.plo.pack import SEATS_9MAX, parse_node_path

        class _N:  # minimal stand-in: clean_line_node reads history_before
            def __init__(self, hist):  # noqa: ANN001
                self.history_before = hist

        # UTG opens, UTG+1 calls, UTG+2 calls -> LJ deciding vs a 3-way pot:
        # 3 entrants in the LINE (hero-inclusive counting would say 4).
        hist = parse_node_path("2.1.1.0", seats=SEATS_9MAX)[:-1]
        assert clean_line_node(_N(hist))
        # Every prefix (the parent chain) passes too -- connectivity.
        for k in range(len(hist)):
            assert clean_line_node(_N(hist[:k]))
        # A 4th ENTRANT in the line fails; so does a 3rd raise.
        hist4 = parse_node_path("2.1.1.1.0", seats=SEATS_9MAX)[:-1]
        assert not clean_line_node(_N(hist4))
        hist3r = parse_node_path("2.2.0.0.0.0.0.0.2.0", seats=SEATS_9MAX)[:-1]
        assert not clean_line_node(_N(hist3r))


# --- real-pack smoke test --------------------------------------------------
@pytest.mark.skipif(
    not (REPO_ROOT / "plo12_ranges").exists(),
    reason="plo12_ranges pack not extracted on this machine",
)
def test_real_pack_smoke_plo12(tmp_path: Path) -> None:
    pack = discover_plo_pack(REPO_ROOT / "plo12_ranges")
    assert pack.pack_id == "plo_6max_12bb"
    out = tmp_path / "export"
    stats = export_pack(pack, out)  # full internal validation runs
    assert stats.nodes_written > 100
    index = json.loads((out / "plo_6max_12bb" / "index.json").read_text())
    assert index["stack_bb"] == 12
    assert index["format"] == "Cash"
    root = index["nodes"][""]
    with gzip.open(out / "plo_6max_12bb" / root["hands_file"], "rt") as fh:
        root_hands = json.load(fh)
    # first-in root: naturally every class with any strategy
    assert len(root_hands["hands"]) == HAND_COUNT
