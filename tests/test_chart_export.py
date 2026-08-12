"""Tests for the SolverPack chart exporter (pipeline/preflop/chart_export.py).

Pure/synthetic tests always run; the landmark tests against the real MTT
8-max packs skip gracefully when the extracted ``mtt8_*_ranges/`` sibling
dirs are absent (they are gitignored multi-GB extractions).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.preflop.chart_export import (
    CASH8_CHART_EXPORT_PACK_IDS,
    CASH_CHART_EXPORT_PACK_IDS,
    build_solver_pack,
    export_node,
    grid_hand,
    grid_index,
    largest_remainder_percentages,
    node_path_key,
    pack_export_id,
    rake_label,
    validate_solver_pack,
)
from pipeline.preflop.grammars.types import (
    ParsedAction,
    ParsedRangeFile,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import (
    PreflopActionOption,
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.pack import (
    KNOWN_PACK_SIGNATURES,
    PreflopPack,
    clear_registry,
    discover_packs,
)
from pipeline.preflop_ranges import canonical_169_hand_classes

REPO_ROOT = Path(__file__).resolve().parent.parent
RANGES_ROOT = REPO_ROOT / "ranges"


# --- grid index math ---------------------------------------------------------


def test_grid_hand_landmarks():
    assert grid_hand(0) == "AA"
    assert grid_hand(1) == "AKs"
    assert grid_hand(13) == "AKo"
    assert grid_hand(14) == "KK"
    assert grid_hand(168) == "22"
    assert grid_hand(10) == "A4s"
    assert grid_hand(130) == "A4o"  # r=10, c=0 -> offsuit twin of idx 10


def test_grid_hand_full_grid_is_the_169_classes():
    labels = {grid_hand(i) for i in range(169)}
    assert len(labels) == 169
    assert labels == set(canonical_169_hand_classes())


def test_grid_index_roundtrip():
    for idx in range(169):
        assert grid_index(grid_hand(idx)) == idx


def test_grid_hand_rejects_out_of_range():
    with pytest.raises(ValueError):
        grid_hand(169)
    with pytest.raises(ValueError):
        grid_hand(-1)


# --- largest-remainder percentages -------------------------------------------


def test_largest_remainder_sums_to_100():
    assert largest_remainder_percentages([1.0]) == [100]
    assert largest_remainder_percentages([0.64, 0.36]) == [64, 36]
    assert largest_remainder_percentages([0.5, 0.5]) == [50, 50]
    # normalises un-normalised input
    assert largest_remainder_percentages([2.0, 1.0, 1.0]) == [50, 25, 25]
    # thirds: remainders equal, exact allocation still sums to 100
    thirds = largest_remainder_percentages([1 / 3, 1 / 3, 1 / 3])
    assert sum(thirds) == 100
    assert sorted(thirds) == [33, 33, 34]
    # tie in remainder goes to the higher-frequency entry (house pattern)
    assert largest_remainder_percentages([0.665, 0.335]) == [67, 33]
    # zero entries stay zero
    assert largest_remainder_percentages([0.995, 0.005, 0.0]) == [100, 0, 0]


def test_largest_remainder_rejects_all_zero():
    with pytest.raises(ValueError):
        largest_remainder_percentages([0.0, 0.0])


# --- synthetic pack ----------------------------------------------------------

# Weights for the synthetic root node (UTG first-in at 15bb-style depth).
# acts land ordered: fold("0"), raise 2bb("5"), raise 4.5bb("40100"),
# allin("3") -- indices 0..3.
_ROOT_WEIGHTS = {
    "AA": {"0": 0.0, "5": 0.0, "40100": 1.0, "3": 0.0},
    "AKs": {"0": 0.0, "5": 0.0, "40100": 0.0, "3": 1.0},
    "KK": {"0": 0.0, "5": 0.5, "40100": 0.5, "3": 0.0},
    "A4s": {"0": 0.64, "5": 0.36, "40100": 0.0, "3": 0.0},
}
# UTG+1 after the min-raise open ("5."): AA pure call, KK mixed
# fold/call, everything else absent (pure fold via reach).
_CHILD_WEIGHTS = {
    "AA": {"0": 0.0, "1": 1.0},
    "KK": {"0": 0.7, "1": 0.3},
}


def _write_rng(path: Path, weights_for_token: dict[str, float]) -> None:
    lines = []
    for hand in canonical_169_hand_classes():
        lines.append(hand)
        lines.append(f"{weights_for_token.get(hand, 0.0)};0.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def synthetic_pack(tmp_path: Path) -> PreflopPack:
    root_tokens = ["0", "5", "40100", "3"]
    for token in root_tokens:
        default = 1.0 if token == "0" else 0.0  # unlisted hands pure-fold
        weights = {
            hand: mix.get(token, 0.0) for hand, mix in _ROOT_WEIGHTS.items()
        }
        _write_rng(
            tmp_path / f"{token}.rng",
            {
                hand: weights.get(hand, default)
                for hand in canonical_169_hand_classes()
            },
        )
    for token in ["0", "1"]:
        _write_rng(
            tmp_path / f"5.{token}.rng",
            {
                hand: mix.get(token, 0.0)
                for hand, mix in _CHILD_WEIGHTS.items()
            },
        )
    return PreflopPack(
        pack_id="test_mtt8_synth",
        root_path=tmp_path,
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=15,
        open_size_bb=2.0,
        file_glob="*.rng",
        size_round_bb=0.5,
        ante_bb=1.0,
        game_format="tournament",
    )


def test_synthetic_pack_export(synthetic_pack: PreflopPack):
    solver_pack = build_solver_pack(synthetic_pack)  # validates internally

    assert solver_pack["id"] == "nlh-8max-15bb"
    assert solver_pack["format"] == "MTT"
    assert solver_pack["table"] == "8-Max"
    assert solver_pack["stack"] == 15
    assert solver_pack["ante"] == 1
    assert solver_pack["blinds"] == [0.5, 1]
    assert solver_pack["order"] == [
        "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB",
    ]
    assert solver_pack["decisionNodes"] == 2
    assert set(solver_pack["nodes"]) == {"", "5"}

    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    assert root["bl"] == 0
    assert root["pot"] == 2.5  # SB 0.5 + BB 1 + ante 1
    assert root["bet"] == 1
    # act ordering: fold, raises ascending by to, allin last
    assert [(a["c"], a["k"], a["to"]) for a in root["acts"]] == [
        ("0", "fold", 0),
        ("5", "raise", 2),
        # 100% pot on (2.5 pot + 1 to-call) = raise to 4.5
        ("40100", "raise", 4.5),
        ("3", "allin", 15),
    ]
    # pure encoding: base-36 act indices; "." for mixed
    assert root["pure"][grid_index("AA")] == "2"  # pure 4.5bb raise
    assert root["pure"][grid_index("AKs")] == "3"  # pure jam
    assert root["pure"][grid_index("KK")] == "."
    assert root["pure"][grid_index("A4s")] == "."
    assert root["pure"][grid_index("72o")] == "0"  # unlisted -> pure fold
    assert root["mix"][str(grid_index("KK"))] == [0, 50, 50, 0]
    assert root["mix"][str(grid_index("A4s"))] == [64, 36, 0, 0]

    child = solver_pack["nodes"]["5"]
    assert child["pos"] == "UTG+1"
    assert child["bl"] == 1  # facing an open
    assert child["pot"] == 4.5  # SB 0.5 + BB 1 + ante 1 + open 2
    assert child["bet"] == 2
    assert [(a["k"], a["to"]) for a in child["acts"]] == [
        ("fold", 0),
        ("call", 2),  # call-TO = the current bet
    ]
    assert child["pure"][grid_index("AA")] == "1"
    assert child["mix"][str(grid_index("KK"))] == [70, 30]
    # hand absent from every file at the node -> pure fold
    assert child["pure"][grid_index("72o")] == "0"


def test_validate_catches_broken_tree(synthetic_pack: PreflopPack):
    solver_pack = build_solver_pack(synthetic_pack)
    # a key whose final token is not one of its parent's act c codes
    solver_pack["nodes"]["5.9"] = dict(solver_pack["nodes"]["5"])
    solver_pack["decisionNodes"] += 1
    with pytest.raises(AssertionError, match="not an act of its parent"):
        validate_solver_pack(solver_pack)


def test_validate_catches_bad_mix_sum(synthetic_pack: PreflopPack):
    solver_pack = build_solver_pack(synthetic_pack)
    key = str(grid_index("KK"))
    solver_pack["nodes"][""]["mix"][key] = [0, 50, 49, 0]
    with pytest.raises(AssertionError, match="sums to 99"):
        validate_solver_pack(solver_pack)


def test_node_path_key_rejects_mismatched_prefixes():
    def _rf(stem: str) -> ParsedRangeFile:
        return ParsedRangeFile(
            pack_id="x",
            path=Path(f"/tmp/{stem}.rng"),
            actor="UTG+1",
            actor_action=PreflopActionType.CALL,
            actor_raise_size_pct=None,
            action_history=(),
        )

    history = (
        ParsedAction(position="UTG", action_type=PreflopActionType.RAISE,
                     raise_size_pct=100.0),
    )
    node = PreflopDecisionNode(
        pack_id="x",
        actor="UTG+1",
        history_before=history,
        actions=(
            PreflopActionOption(PreflopActionType.CALL, None, _rf("0.1")),
            PreflopActionOption(PreflopActionType.CALL, None, _rf("5.1")),
        ),
    )
    with pytest.raises(AssertionError, match="stems disagree"):
        node_path_key(node)


# --- rake labels -------------------------------------------------------------


def _pack(**overrides) -> PreflopPack:
    base = dict(
        pack_id="x",
        root_path=Path("/tmp/none"),
        grammar_name="monker_nlhe",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )
    base.update(overrides)
    return PreflopPack(**base)


def test_rake_label_none_pct_and_cap():
    assert rake_label(_pack(rake_pct=None)) == "none"
    assert rake_label(_pack(rake_pct=0.04)) == "4%"
    assert (
        rake_label(
            _pack(rake_pct=0.05, description="rake 5%/0.5bb cap (notes).")
        )
        == "5% cap 0.5bb"
    )
    assert (
        rake_label(_pack(rake_pct=0.1, description="rake 10%/3bb cap"))
        == "10% cap 3bb"
    )


# --- synthetic 6-max CASH packs ----------------------------------------------


@pytest.fixture()
def synthetic_cash_pack(tmp_path: Path) -> PreflopPack:
    """A tiny Monker-grammar 6-max CASH pack: root fold/min-raise only."""
    for token, default in (("0", 1.0), ("5", 0.0)):
        weights = {
            hand: _ROOT_WEIGHTS.get(hand, {}).get(token, default)
            for hand in canonical_169_hand_classes()
        }
        _write_rng(tmp_path / f"{token}.rng", weights)
    return PreflopPack(
        pack_id="test_nlhe6_synth",
        root_path=tmp_path,
        grammar_name="monker_nlhe",
        table_size=6,
        stack_depth_bb=20,
        open_size_bb=2.0,
        file_glob="*.rng",
        size_round_bb=0.5,
        rake_pct=0.05,
        description="synthetic -- rake 5%/0.5bb cap.",
    )


def test_synthetic_cash_pack_header(synthetic_cash_pack: PreflopPack):
    solver_pack = build_solver_pack(synthetic_cash_pack)
    assert solver_pack["id"] == "nlh-6max-20bb"
    assert solver_pack["format"] == "Cash"
    assert solver_pack["table"] == "6-Max"
    assert solver_pack["stack"] == 20
    assert solver_pack["ante"] == 0
    assert solver_pack["rake"] == "5% cap 0.5bb"
    assert solver_pack["blinds"] == [0.5, 1]
    assert solver_pack["order"] == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    assert root["pot"] == 1.5  # SB 0.5 + BB 1, NO ante
    assert root["bet"] == 1
    assert [(a["c"], a["k"], a["to"]) for a in root["acts"]] == [
        ("0", "fold", 0),
        ("5", "raise", 2),  # the min-raise open
    ]


def test_cash_pack_with_nonzero_ante_is_rejected(
    synthetic_cash_pack: PreflopPack,
):
    from dataclasses import replace

    broken = replace(synthetic_cash_pack, ante_bb=1.0)
    with pytest.raises(AssertionError, match="ante_bb"):
        build_solver_pack(broken)


# --- synthetic ryan_pack (PioViewer grammar) ---------------------------------

# Per-node weights for the ryan fixture. Tokens are the pack's documented
# (pct, level) lookup entries: 60% open -> 2.5bb, 77% 3-bet -> 8bb.
_RYAN_ROOT = {  # UTG first-in
    "AA": {"Fold": 0.0, "60%": 1.0},
    "A4s": {"Fold": 0.64, "60%": 0.36},
}
_RYAN_HJ = {  # HJ facing the UTG open
    "AA": {"Fold": 0.0, "Call": 0.5, "77%": 0.5},
    "KK": {"Fold": 0.0, "Call": 1.0, "77%": 0.0},
}
_RYAN_CO = {  # CO facing the HJ 3-bet
    "AA": {"Fold": 0.0, "AI": 1.0},
}


def _write_txt(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = ",".join(
        f"{hand}:{weights.get(hand, 0.0)}"
        for hand in canonical_169_hand_classes()
    )
    path.write_text(line + "\n", encoding="utf-8")


def _ryan_weights(
    table: dict[str, dict[str, float]], token: str, default: float
) -> dict[str, float]:
    """Listed hands take their mix entry for ``token``; unlisted hands take
    ``default`` (1.0 on the Fold file, 0.0 elsewhere -> pure fold)."""
    return {
        hand: table[hand].get(token, 0.0) if hand in table else default
        for hand in canonical_169_hand_classes()
    }


@pytest.fixture()
def synthetic_ryan_pack(tmp_path: Path) -> PreflopPack:
    for token, default in (("Fold", 1.0), ("60%", 0.0)):
        _write_txt(
            tmp_path / "UTG" / f"UTG_{token}.txt",
            _ryan_weights(_RYAN_ROOT, token, default),
        )
    for token, default in (("Fold", 1.0), ("Call", 0.0), ("77%", 0.0)):
        _write_txt(
            tmp_path / "HJ" / f"UTG_60%_HJ_{token}.txt",
            _ryan_weights(_RYAN_HJ, token, default),
        )
    for token, default in (("Fold", 1.0), ("AI", 0.0)):
        _write_txt(
            tmp_path / "CO" / f"UTG_60%_HJ_77%_CO_{token}.txt",
            _ryan_weights(_RYAN_CO, token, default),
        )
    return PreflopPack(
        pack_id="test_ryan_synth",
        root_path=tmp_path,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
        file_glob="*.txt",
        size_round_bb=0.5,
        rake_pct=0.04,
        description="synthetic ryan -- rake 4%/0.3bb cap.",
    )


def test_synthetic_ryan_pack_export(synthetic_ryan_pack: PreflopPack):
    solver_pack = build_solver_pack(synthetic_ryan_pack)  # validates tree

    assert solver_pack["id"] == "nlh-6max-100bb"
    assert solver_pack["format"] == "Cash"
    assert solver_pack["table"] == "6-Max"
    assert solver_pack["ante"] == 0
    assert solver_pack["rake"] == "4% cap 0.3bb"
    assert solver_pack["order"] == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

    # ryan path keys: dot-joined ACTION tokens (positions dropped); the c
    # codes are the bare action tokens, unique among siblings.
    assert set(solver_pack["nodes"]) == {"", "60%", "60%.77%"}

    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    assert root["pot"] == 1.5
    assert [(a["c"], a["k"], a["to"]) for a in root["acts"]] == [
        ("Fold", "fold", 0),
        ("60%", "raise", 2.5),  # the documented 2.5x open
    ]
    assert root["pure"][grid_index("AA")] == "1"
    assert root["mix"][str(grid_index("A4s"))] == [64, 36]
    assert root["pure"][grid_index("72o")] == "0"

    hj = solver_pack["nodes"]["60%"]
    assert hj["pos"] == "HJ"
    assert hj["bl"] == 1
    assert hj["pot"] == 4  # 0.5 + 1 + 2.5
    assert hj["bet"] == 2.5
    assert [(a["c"], a["k"], a["to"]) for a in hj["acts"]] == [
        ("Fold", "fold", 0),
        ("Call", "call", 2.5),
        ("77%", "raise", 8),  # the documented 3-bet size
    ]
    assert hj["mix"][str(grid_index("AA"))] == [0, 50, 50]
    assert hj["pure"][grid_index("KK")] == "1"

    co = solver_pack["nodes"]["60%.77%"]
    assert co["pos"] == "CO"
    assert co["bl"] == 2
    assert co["pot"] == 12  # 0.5 + 1 + 2.5 + 8
    assert co["bet"] == 8
    assert [(a["c"], a["k"], a["to"]) for a in co["acts"]] == [
        ("Fold", "fold", 0),
        ("AI", "allin", 100),  # jam TO the full effective stack
    ]
    assert co["pure"][grid_index("AA")] == "1"


# --- synthetic gto_preflop_8max (8-max cash grammar) -------------------------

# UTG first-in at 100bb: AA pure-opens, KK mixes open/jam, A4s mixes
# fold/open, everything else pure fold.
_GTO8_ROOT = {
    "AA": {"F": 0.0, "R3": 1.0, "A": 0.0},
    "KK": {"F": 0.0, "R3": 0.5, "A": 0.5},
    "A4s": {"F": 0.64, "R3": 0.36, "A": 0.0},
}
# LJ facing the UTG open. The UTG+1 decision node is deliberately NOT
# materialised (its fold exists only as history) -- the forced-fold splice
# must contract it out of the tree keys.
_GTO8_LJ = {
    "AA": {"F": 0.0, "C": 0.0, "R7.5": 1.0},
    "KK": {"F": 0.0, "C": 1.0, "R7.5": 0.0},
}
# UTG facing the LJ 3-bet (five more unmaterialised folds behind).
_GTO8_UTG_V3B = {
    "AA": {"F": 0.0, "C": 1.0},
}


@pytest.fixture()
def synthetic_gto8_pack(tmp_path: Path) -> PreflopPack:
    for code, default in (("F", 1.0), ("R3", 0.0), ("A", 0.0)):
        _write_rng(
            tmp_path / f"UTG-{code}.rng",
            {
                hand: _GTO8_ROOT.get(hand, {}).get(code, default)
                for hand in canonical_169_hand_classes()
            },
        )
    lj_prefix = "UTG-R3_UTG+1-F_LJ"
    for code, default in (("F", 1.0), ("C", 0.0), ("R7.5", 0.0)):
        _write_rng(
            tmp_path / f"{lj_prefix}-{code}.rng",
            {
                hand: _GTO8_LJ.get(hand, {}).get(code, default)
                for hand in canonical_169_hand_classes()
            },
        )
    utg_prefix = "UTG-R3_UTG+1-F_LJ-R7.5_HJ-F_CO-F_BTN-F_SB-F_BB-F_UTG"
    for code, default in (("F", 1.0), ("C", 0.0)):
        _write_rng(
            tmp_path / f"{utg_prefix}-{code}.rng",
            {
                hand: _GTO8_UTG_V3B.get(hand, {}).get(code, default)
                for hand in canonical_169_hand_classes()
            },
        )
    return PreflopPack(
        pack_id="test_gto8_synth",
        root_path=tmp_path,
        grammar_name="gto_preflop_8max",
        table_size=8,
        stack_depth_bb=100,
        open_size_bb=3.0,
        file_glob="*.rng",
        size_round_bb=0.5,
        rake_pct=0.1,
    )


def test_synthetic_gto8_pack_export(synthetic_gto8_pack: PreflopPack):
    solver_pack = build_solver_pack(synthetic_gto8_pack)  # validates tree

    # NAMING RULE: 8-max CASH carries the -cash- infix (MTT owns the bare
    # nlh-8max-<depth>bb names).
    assert solver_pack["id"] == "nlh-8max-cash-100bb"
    assert solver_pack["format"] == "Cash"
    assert solver_pack["table"] == "8-Max"
    assert solver_pack["stack"] == 100
    assert solver_pack["ante"] == 0
    assert solver_pack["rake"] == "10%"
    assert solver_pack["order"] == [
        "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB",
    ]

    # Forced-fold splice: the UTG+1 node (and the five seats behind the
    # 3-bet) are unmaterialised, so their fold tokens contract out of the
    # keys; the fractional 3-bet's c code is dot-sanitized (R7.5 -> R7-5).
    assert set(solver_pack["nodes"]) == {"", "R3", "R3.R7-5"}

    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    assert root["bl"] == 0
    assert root["pot"] == 1.5  # no ante in a cash pot
    assert [(a["c"], a["k"], a["to"]) for a in root["acts"]] == [
        ("F", "fold", 0),
        ("R3", "raise", 3),  # the registered 3bb open
        ("A", "allin", 100),
    ]
    assert root["pure"][grid_index("AA")] == "1"
    assert root["mix"][str(grid_index("KK"))] == [0, 50, 50]
    assert root["mix"][str(grid_index("A4s"))] == [64, 36, 0]
    assert root["pure"][grid_index("72o")] == "0"

    lj = solver_pack["nodes"]["R3"]
    assert lj["pos"] == "LJ"
    assert lj["bl"] == 1
    assert lj["pot"] == 4.5  # 0.5 + 1 + 3
    assert lj["bet"] == 3
    assert [(a["c"], a["k"], a["to"]) for a in lj["acts"]] == [
        ("F", "fold", 0),
        ("C", "call", 3),
        ("R7-5", "raise", 7.5),
    ]
    assert lj["pure"][grid_index("AA")] == "2"
    assert lj["pure"][grid_index("KK")] == "1"

    utg = solver_pack["nodes"]["R3.R7-5"]
    assert utg["pos"] == "UTG"
    assert utg["bl"] == 2
    assert utg["pot"] == 12  # 0.5 + 1 + 3 + 7.5
    assert utg["bet"] == 7.5
    assert [(a["c"], a["k"], a["to"]) for a in utg["acts"]] == [
        ("F", "fold", 0),
        ("C", "call", 7.5),
    ]
    assert utg["pure"][grid_index("AA")] == "1"


def test_8max_cash_ids_never_collide_with_mtt():
    """NAMING-COLLISION GUARD: one chart id per (table, depth, format).

    The MTT exports own nlh-8max-<depth>bb; the cash trio must carry the
    -cash- infix at every shared depth, or files overwrite each other.
    """
    for depth in (100, 200, 300):
        cash = _pack(
            table_size=8, stack_depth_bb=depth, game_format="cash"
        )
        mtt = _pack(
            table_size=8,
            stack_depth_bb=depth,
            game_format="tournament",
            ante_bb=1.0,
        )
        assert pack_export_id(cash) == f"nlh-8max-cash-{depth}bb"
        assert pack_export_id(mtt) == f"nlh-8max-{depth}bb"
        assert pack_export_id(cash) != pack_export_id(mtt)
    # 6-max cash keeps its already-shipped bare name.
    assert pack_export_id(_pack(table_size=6)) == "nlh-6max-100bb"


# --- real-pack landmark tests ------------------------------------------------


def _mtt8_pack(depth: int) -> PreflopPack | None:
    """Discover (once per call, registry cleared) the real mtt8 pack, or None."""
    if not (REPO_ROOT / f"mtt8_{depth}bb_ranges").is_dir():
        return None
    clear_registry()
    packs = discover_packs(RANGES_ROOT, KNOWN_PACK_SIGNATURES)
    clear_registry()  # leave no global state behind for other test files
    return next(
        (p for p in packs if p.pack_id == f"monker_mtt8_{depth}bb"), None
    )


def _real_pack(pack_id: str) -> PreflopPack | None:
    """Discover a real registered pack by id (registry left clean)."""
    clear_registry()
    packs = discover_packs(RANGES_ROOT, KNOWN_PACK_SIGNATURES)
    clear_registry()
    return next((p for p in packs if p.pack_id == pack_id), None)


@pytest.mark.skipif(
    not (REPO_ROOT / "mtt8_15bb_ranges").is_dir(),
    reason="mtt8_15bb_ranges not extracted",
)
def test_15bb_landmarks():
    pack = _mtt8_pack(15)
    assert pack is not None
    solver_pack = build_solver_pack(pack)  # full validation runs inside

    assert pack_export_id(pack) == "nlh-8max-15bb"
    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    # Audited landmark: 15bb root = fold + min-raise to 2.0 + jam 15.0
    assert [(a["k"], a["to"]) for a in root["acts"]] == [
        ("fold", 0), ("raise", 2), ("allin", 15),
    ]
    raise_idx = next(
        i for i, a in enumerate(root["acts"]) if a["k"] == "raise"
    )
    assert root["pure"][grid_index("AA")] == str(raise_idx)  # AA pure-opens
    # Audited landmark: A4s opens mixed ~64/36 fold/raise
    a4s = root["mix"][str(grid_index("A4s"))]
    assert a4s == [64, 36, 0]


@pytest.mark.skipif(
    not (REPO_ROOT / "mtt8_10bb_ranges").is_dir(),
    reason="mtt8_10bb_ranges not extracted",
)
def test_10bb_is_jam_or_fold_everywhere():
    pack = _mtt8_pack(10)
    assert pack is not None
    solver_pack = build_solver_pack(pack)
    kinds = {
        a["k"] for node in solver_pack["nodes"].values() for a in node["acts"]
    }
    # Audited landmark: the 10bb tree offers NO plain raise anywhere.
    assert "raise" not in kinds
    assert kinds <= {"fold", "call", "allin"}
    # Every jam is TO the full 10bb effective stack.
    for node in solver_pack["nodes"].values():
        for a in node["acts"]:
            if a["k"] == "allin":
                assert a["to"] == 10


@pytest.mark.skipif(
    not (REPO_ROOT / "mtt8_75bb_ranges").is_dir(),
    reason="mtt8_75bb_ranges not extracted",
)
def test_75bb_hj_first_in_opens_2_5bb():
    pack = _mtt8_pack(75)
    assert pack is not None
    nodes = enumerate_nodes([pack])
    hj_rfi = next(
        n
        for n in nodes
        if n.actor == "HJ"
        and len(n.history_before) == 3
        and all(
            a.action_type is PreflopActionType.FOLD for a in n.history_before
        )
    )
    key, obj = export_node(hj_rfi, pack)
    assert key == "0.0.0"
    raise_tos = [a["to"] for a in obj["acts"] if a["k"] == "raise"]
    # Audited landmark: the 40043 open resolves to 2.5bb on the 0.5 grid.
    assert raise_tos == [2.5]


@pytest.mark.skipif(
    not (REPO_ROOT / "mtt8_15bb_ranges").is_dir(),
    reason="mtt8_15bb_ranges not extracted",
)
def test_cross_check_against_format_writer_action_mix():
    """The pure/mix encoding round-trips format_writer's per-hand mix.

    ``_action_mix_for_node`` is the CSV `ranges` column's builder (raw
    joint freqs per normalised label); the chart export must encode the
    SAME distribution, conditional on reach, within integer rounding.
    """
    from pipeline.preflop.format_writer import _action_mix_for_node

    pack = _mtt8_pack(15)
    assert pack is not None
    nodes = enumerate_nodes([pack])

    def _unique_kinds(n) -> bool:
        # _action_mix_for_node keys the chart by normalised label, so two
        # same-kind raises would collide there; sample label-unique nodes.
        key, obj = export_node(n, pack)
        kinds = [a["k"] for a in obj["acts"]]
        return len(set(kinds)) == len(kinds)

    root = next(n for n in nodes if not n.history_before)
    facing_open = next(
        n
        for n in nodes
        if len(n.history_before) == 1
        and len(n.actions) >= 3
        and _unique_kinds(n)
    )
    deeper = next(
        n for n in nodes if len(n.history_before) == 4 and _unique_kinds(n)
    )

    for node in (root, facing_open, deeper):
        key, obj = export_node(node, pack)
        chart = _action_mix_for_node(node, pack)
        # map each act index -> the format_writer label for that action
        labels = [act["k"] for act in obj["acts"]]
        for hand, entry in chart.items():
            idx = grid_index(hand)
            freqs = [entry.get(lbl, {}).get("freq", 0.0) for lbl in labels]
            total = sum(freqs)
            assert total > 0
            ch = obj["pure"][idx]
            if ch == ".":
                row = obj["mix"][str(idx)]
                for pct, freq in zip(row, freqs):
                    assert abs(pct - 100.0 * freq / total) <= 1.0
            else:
                act_i = int(ch, 36)
                assert freqs[act_i] == pytest.approx(total)


# --- real CASH-pack landmark tests -------------------------------------------


@pytest.mark.skipif(
    not (RANGES_ROOT / "ryan_preflop_tree").is_dir(),
    reason="ryan_preflop_tree not present",
)
def test_ryan_100bb_cash_landmarks():
    pack = _real_pack("ryan_preflop_tree_6max_100bb")
    assert pack is not None
    solver_pack = build_solver_pack(pack)  # key uniqueness + connectivity

    assert solver_pack["id"] == "nlh-6max-100bb"
    assert solver_pack["format"] == "Cash"
    assert solver_pack["table"] == "6-Max"
    assert solver_pack["stack"] == 100
    assert solver_pack["ante"] == 0
    assert solver_pack["rake"] == "4% cap 0.3bb"
    assert solver_pack["order"] == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    assert root["pot"] == 1.5  # no ante anywhere in a cash pot
    # Registered landmark: the 60% open token resolves to the 2.5bb open.
    assert [(a["c"], a["k"], a["to"]) for a in root["acts"]] == [
        ("Fold", "fold", 0),
        ("60%", "raise", 2.5),
    ]
    # Every jam in the tree is TO the full 100bb effective stack, and no
    # c code carries a "." (the tree-link key scheme forbids it).
    for node in solver_pack["nodes"].values():
        for a in node["acts"]:
            assert "." not in a["c"]
            if a["k"] == "allin":
                assert a["to"] == 100


@pytest.mark.skipif(
    not (REPO_ROOT / "nlhe6_ranges").is_dir(),
    reason="nlhe6_ranges not extracted",
)
@pytest.mark.parametrize("depth", [20, 30])
def test_nlhe6_shortstack_cash_landmarks(depth: int):
    pack = _real_pack(f"monker_nlhe_6max_{depth}bb")
    assert pack is not None
    solver_pack = build_solver_pack(pack)

    assert solver_pack["id"] == f"nlh-6max-{depth}bb"
    assert solver_pack["format"] == "Cash"
    assert solver_pack["table"] == "6-Max"
    assert solver_pack["ante"] == 0
    assert solver_pack["rake"] == "5% cap 0.5bb"
    assert solver_pack["order"] == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    assert root["pot"] == 1.5
    # Audited landmark: the `5` min-raise token opens to 2bb.
    assert [(a["k"], a["to"]) for a in root["acts"]] == [
        ("fold", 0),
        ("raise", 2),
    ]
    for node in solver_pack["nodes"].values():
        for a in node["acts"]:
            if a["k"] == "allin":
                assert a["to"] == depth


@pytest.mark.skipif(
    not (RANGES_ROOT / "preflop_8max_100bb_improved").is_dir(),
    reason="preflop_8max_100bb_improved not present",
)
@pytest.mark.parametrize("depth", [100, 200, 300])
def test_8max_cash_improved_landmarks(depth: int):
    pack = _real_pack(f"preflop_8max_{depth}bb_IMPROVED")
    assert pack is not None
    solver_pack = build_solver_pack(pack)  # connectivity + grid validation

    assert solver_pack["id"] == f"nlh-8max-cash-{depth}bb"
    assert solver_pack["format"] == "Cash"
    assert solver_pack["table"] == "8-Max"
    assert solver_pack["stack"] == depth
    assert solver_pack["ante"] == 0
    assert solver_pack["rake"] == "10%"  # registered rake_pct=0.1, no cap
    assert solver_pack["order"] == [
        "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB",
    ]
    # All three conversions share the same 91-node tree structure.
    assert solver_pack["decisionNodes"] == 91

    root = solver_pack["nodes"][""]
    assert root["pos"] == "UTG"
    assert root["pot"] == 1.5  # no ante anywhere in a cash pot
    # Registered landmark: the 3bb open (open_size_bb) at every depth.
    raise_tos = [a["to"] for a in root["acts"] if a["k"] == "raise"]
    assert raise_tos == [pack.open_size_bb] == [3.0]

    for node in solver_pack["nodes"].values():
        for a in node["acts"]:
            # the tree-link key scheme forbids "." in c codes (R38.5 must
            # have been sanitized to R38-5)
            assert "." not in a["c"]
            if a["k"] == "allin":
                assert a["to"] == depth

    # Forced-fold splice landmark: the opener's response to an SB 3-bet is
    # keyed WITHOUT the BB's unmaterialised forced fold, so the drill can
    # walk open -> 3-bet -> response.
    # (SB 3-bet sizes differ per depth, so match the key shape: 7 tokens =
    # BTN open + SB 3-bet with the BB's fold contracted out.)
    spliced_keys = [
        k
        for k in solver_pack["nodes"]
        if k.count(".") == 6 and k.startswith("F.F.F.F.F.R3.R")  # noqa: PLR2004
    ]
    assert spliced_keys, "expected the SB-3-bet splice node"
    for k in spliced_keys:
        assert solver_pack["nodes"][k]["pos"] == "BTN"
        assert solver_pack["nodes"][k]["bl"] == 2

    # vs_5bet was excluded at conversion: a bl-3 raise (the 5-bet) has no
    # child, which is legal (the walker stops) -- but every plain raise
    # BELOW bl 3 must have its response node inside the open->4-bet tree.
    nodes = solver_pack["nodes"]
    for key, node in nodes.items():
        for a in node["acts"]:
            if a["k"] != "raise":
                continue
            child = (key + "." + a["c"]) if key else a["c"]
            if node["bl"] < 3:  # noqa: PLR2004
                assert child in nodes, (
                    f"{key!r}: raise {a['c']} lacks its response node"
                )
            else:
                assert child not in nodes  # the excluded vs_5bet layer


def test_cash_export_pack_ids_are_registered():
    registered = {sig.pack_id for sig in KNOWN_PACK_SIGNATURES}
    assert set(CASH_CHART_EXPORT_PACK_IDS) <= registered
    assert set(CASH8_CHART_EXPORT_PACK_IDS) <= registered
