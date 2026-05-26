"""Tests for pipeline.preflop.pack -- the PreflopPack registry + discovery.

Run under pytest. Isolated from disk except for one integration test that
exercises discover_packs() against the real ranges/ folder if present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.pack import (                                       # noqa: E402
    KNOWN_PACK_SIGNATURES,
    PreflopPack,
    PreflopPackSignature,
    all_packs,
    clear_registry,
    discover_packs,
    get_pack,
    packs_by_table_size,
    register_pack,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_registry_between_tests():
    """Ensure each test starts with an empty registry."""
    clear_registry()
    yield
    clear_registry()


def _sample_pack(pack_id: str = "test_pack", table_size: int = 6) -> PreflopPack:
    return PreflopPack(
        pack_id=pack_id,
        root_path=Path("/tmp/fake-pack-root"),
        grammar_name="ryan_pack",
        table_size=table_size,
        stack_depth_bb=100,
        open_size_bb=2.5,
    )


# --- PreflopPack validation -------------------------------------------------
def test_preflop_pack_rejects_invalid_table_size():
    with pytest.raises(ValueError, match="table_size"):
        PreflopPack(
            pack_id="x", root_path=Path("/tmp"), grammar_name="ryan_pack",
            table_size=15, stack_depth_bb=100, open_size_bb=2.5,
        )


def test_preflop_pack_rejects_zero_stack():
    with pytest.raises(ValueError, match="stack_depth_bb"):
        PreflopPack(
            pack_id="x", root_path=Path("/tmp"), grammar_name="ryan_pack",
            table_size=6, stack_depth_bb=0, open_size_bb=2.5,
        )


def test_preflop_pack_rejects_bad_sb_ratio():
    with pytest.raises(ValueError, match="sb_to_bb_ratio"):
        PreflopPack(
            pack_id="x", root_path=Path("/tmp"), grammar_name="ryan_pack",
            table_size=6, stack_depth_bb=100, open_size_bb=2.5,
            sb_to_bb_ratio=1.5,
        )


# --- registry ---------------------------------------------------------------
def test_register_pack_round_trip():
    pack = _sample_pack()
    register_pack(pack)
    assert get_pack("test_pack") is pack
    assert pack in all_packs()


def test_register_pack_rejects_duplicate():
    register_pack(_sample_pack())
    with pytest.raises(ValueError, match="already registered"):
        register_pack(_sample_pack())


def test_get_pack_unknown_raises_with_known_list():
    register_pack(_sample_pack("alpha"))
    register_pack(_sample_pack("beta"))
    with pytest.raises(KeyError) as exc_info:
        get_pack("missing")
    assert "missing" in str(exc_info.value)
    assert "alpha" in str(exc_info.value)
    assert "beta" in str(exc_info.value)


def test_packs_by_table_size_filters():
    register_pack(_sample_pack("six_a", table_size=6))
    register_pack(_sample_pack("six_b", table_size=6))
    register_pack(_sample_pack("nine", table_size=9))
    six_max = packs_by_table_size(6)
    assert len(six_max) == 2
    assert {p.pack_id for p in six_max} == {"six_a", "six_b"}
    assert len(packs_by_table_size(9)) == 1
    assert packs_by_table_size(5) == ()


# --- discovery --------------------------------------------------------------
def test_discover_packs_finds_nothing_in_empty_dir(tmp_path):
    found = discover_packs(tmp_path)
    assert found == ()
    assert all_packs() == ()


def test_discover_packs_finds_a_pack(tmp_path):
    """Create a fake pack layout and verify discovery picks it up."""
    fake_pack_root = tmp_path / "ryan_preflop_tree" / "PioViewer - NLH 6max 100bb 2.5x Open"
    fake_pack_root.mkdir(parents=True)
    found = discover_packs(tmp_path)
    assert len(found) == 1
    assert found[0].pack_id == "ryan_preflop_tree_6max_100bb"
    assert found[0].root_path == fake_pack_root.resolve()
    # Registered too:
    assert get_pack("ryan_preflop_tree_6max_100bb") is found[0]


def test_discover_packs_respects_signature_override(tmp_path):
    """Custom signature list = fine-grained control in tests."""
    fake_root = tmp_path / "custom" / "data"
    fake_root.mkdir(parents=True)
    custom_sig = PreflopPackSignature(
        pack_id="custom_pack",
        relative_pack_root="custom/data",
        grammar_name="ryan_pack",
        table_size=9,
        stack_depth_bb=50,
        open_size_bb=2.0,
    )
    found = discover_packs(tmp_path, signatures=(custom_sig,))
    assert len(found) == 1
    assert found[0].pack_id == "custom_pack"
    assert found[0].table_size == 9


# --- integration: real ranges/ folder ---------------------------------------
def test_discover_packs_against_real_ranges_dir():
    """If the real ranges/ folder is present locally, discovery finds the
    Ryan 6-max pack. Skips cleanly if ranges/ isn't present (CI, fresh clone)."""
    ranges = REPO_ROOT / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    found = discover_packs(ranges)
    assert any(p.pack_id == "ryan_preflop_tree_6max_100bb" for p in found), \
        f"expected Ryan 6max pack in {[p.pack_id for p in found]}"


# --- KNOWN_PACK_SIGNATURES sanity check ------------------------------------
def test_known_pack_signatures_have_unique_ids():
    ids = [sig.pack_id for sig in KNOWN_PACK_SIGNATURES]
    assert len(ids) == len(set(ids)), f"duplicate pack_ids: {ids}"


def test_known_pack_signatures_have_known_grammars():
    """Every signature references a grammar that exists in the registry."""
    from pipeline.preflop.grammars import _REGISTRY
    for sig in KNOWN_PACK_SIGNATURES:
        assert sig.grammar_name in _REGISTRY, (
            f"signature {sig.pack_id!r} references unknown grammar "
            f"{sig.grammar_name!r}; known: {list(_REGISTRY)}"
        )
