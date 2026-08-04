"""Tests for the NLH MTT 8-max BB-ante packs (Aug 2026).

Two tiers:

  * Pure-logic tests (always run): the fixed-bb token sentinel, the ante's
    dead-money pot accounting in resolve_preflop_history, the open_limp
    archetype, and the tournament context line. These pin the ante
    conventions PROVEN by the intake audit (scripts/audit_mtt8_pack.py):
    ante IN the pot for Monker pot-relative sizing, never part of the bet
    level to match.
  * Pack-file tests (skipped when the gitignored ``mtt8_*_ranges/`` dirs
    are absent): discovery, real size resolution, the animation ante event.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.preflop.action_history import resolve_preflop_history
from pipeline.preflop.grammars.types import (
    MIN_RAISE_PCT,
    ParsedAction,
    PreflopActionType,
    decode_fixed_bb,
    encode_fixed_bb,
    render_raise_size_token,
)
from pipeline.preflop.pack import (
    KNOWN_PACK_SIGNATURES,
    PreflopPack,
    clear_registry,
    discover_packs,
    pack_allins_realistic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MTT_15_ROOT = REPO_ROOT / "mtt8_15bb_ranges"

needs_pack = pytest.mark.skipif(
    not MTT_15_ROOT.is_dir(), reason="mtt8 pack files not extracted"
)


def _pack(ante: float = 1.0, **kw) -> PreflopPack:
    defaults = dict(
        pack_id="test_mtt8",
        root_path=Path("/nonexistent"),
        grammar_name="monker_nlhe",
        table_size=8,
        stack_depth_bb=15,
        open_size_bb=2.0,
        file_glob="*.rng",
        size_round_bb=0.5,
        ante_bb=ante,
        game_format="tournament",
    )
    defaults.update(kw)
    return PreflopPack(**defaults)


def _folds(n: int) -> list[ParsedAction]:
    seats = ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB")
    return [
        ParsedAction(position=seats[i], action_type=PreflopActionType.FOLD)
        for i in range(n)
    ]


# --- fixed-bb sentinel -------------------------------------------------------
def test_fixed_bb_sentinel_roundtrip_and_rendering():
    enc = encode_fixed_bb(2.5)
    assert decode_fixed_bb(enc) == 2.5
    assert render_raise_size_token(enc) == "2.5bb"
    # No collision with the min-raise sentinel or real pot-% tokens.
    assert decode_fixed_bb(MIN_RAISE_PCT) is None
    assert decode_fixed_bb(60.0) is None
    assert decode_fixed_bb(None) is None
    # 1bb encodes away from MIN_RAISE_PCT (the collision that motivated
    # the -1000 base).
    assert encode_fixed_bb(1.0) != MIN_RAISE_PCT


# --- ante pot accounting -----------------------------------------------------
def test_ante_joins_pot_for_monker_pot_relative_sizing():
    """The 75/300bb `40043` open must resolve to 2.5bb (the EV-anchored
    value) -- which happens only with the 1bb ante in the pot."""
    pack = _pack(stack_depth_bb=75, open_size_bb=2.5)
    history = (
        ParsedAction(
            position="UTG",
            action_type=PreflopActionType.RAISE,
            raise_size_pct=43.0,
        ),
    )
    state = resolve_preflop_history(history, pack)
    assert state.sizes_bb[-1] == 2.5  # 1 + 0.43*(2.5+1) = 2.505 -> grid 2.5
    assert state.dead_bb == 1.0
    # pot = SB 0.5 + BB 1 + ante 1 + raise-to 2.5
    assert state.pot_bb == pytest.approx(5.0)
    # The ante never changes what a caller owes.
    assert state.call_cost_bb("BB") == pytest.approx(1.5)


def test_no_ante_resolution_is_unchanged_for_cash_packs():
    pack = _pack(ante=0.0, game_format="cash", stack_depth_bb=100)
    history = (
        ParsedAction(
            position="UTG",
            action_type=PreflopActionType.RAISE,
            raise_size_pct=43.0,
        ),
    )
    state = resolve_preflop_history(history, pack)
    # 1 + 0.43*(1.5+1) = 2.075 -> grid 2.0 (no ante in the pot)
    assert state.sizes_bb[-1] == 2.0
    assert state.dead_bb == 0.0
    assert state.pot_bb == pytest.approx(3.5)


def test_fixed_token_resolves_to_registered_bb_size():
    pack = _pack(fixed_raise_tokens_bb=(("14", 2.5),))
    history = tuple(_folds(6)) + (
        ParsedAction(
            position="SB",
            action_type=PreflopActionType.RAISE,
            raise_size_pct=encode_fixed_bb(2.5),
        ),
    )
    state = resolve_preflop_history(history, pack)
    assert state.sizes_bb[-1] == 2.5
    # pot = SB 2.5 + BB 1 + ante 1
    assert state.pot_bb == pytest.approx(4.5)


# --- allins realism ----------------------------------------------------------
def test_mtt_allins_realism_overrides():
    assert pack_allins_realistic(_pack(stack_depth_bb=15))  # <=40 default
    assert not pack_allins_realistic(_pack(stack_depth_bb=300))
    assert pack_allins_realistic(
        _pack(stack_depth_bb=50, allins_realistic=True)
    )
    assert pack_allins_realistic(
        _pack(stack_depth_bb=75, allins_realistic=True)
    )


# --- archetype ---------------------------------------------------------------
def test_open_limp_archetype_for_non_blind_first_in_call():
    from pipeline.preflop.fact_extractor import classify_archetype

    class _Node:
        actor = "BTN"
        history_before = tuple(_folds(5))

    class _Spot:
        node = _Node()
        action_frequencies = {"Fold": 0.4, "Call": 0.44, "AllIn": 0.16}
        dominant_action = "Call"
        hero_hand_class = "A5s"

    assert classify_archetype(_Spot(), None, None) == "open_limp"


def test_open_limp_has_guidance_ease_and_why_factor():
    from pipeline.preflop.difficulty import ARCHETYPE_BASE_EASE
    from pipeline.preflop.explanation_generator import (
        PREFLOP_ARCHETYPE_GUIDANCE,
    )
    from pipeline.preflop import why_factors

    assert "open_limp" in PREFLOP_ARCHETYPE_GUIDANCE
    assert "open_limp" in ARCHETYPE_BASE_EASE
    assert "open_limp" in why_factors._ARCHETYPE_REASON


# --- context column ----------------------------------------------------------
def test_context_column_shows_stack_and_ante():
    from pipeline.preflop.format_writer import _context_column

    ctx = _context_column(
        _pack(), stakes_bb_dollars=1.0, game_format="tournament",
        live_or_online="", display_in_bb=True,
    )
    assert ctx.startswith("Tournament")  # team ask Aug 2026: format leads
    assert "15bb" in ctx
    assert "1bb ante" in ctx
    # A cash pack's context is unchanged (no ante part).
    cash_ctx = _context_column(
        _pack(ante=0.0, game_format="cash", stack_depth_bb=100),
        stakes_bb_dollars=1.0, game_format="cash",
        live_or_online="Online", display_in_bb=False,
    )
    assert "ante" not in cash_ctx


# --- registration ------------------------------------------------------------
def test_all_seven_mtt_signatures_registered():
    ids = {s.pack_id for s in KNOWN_PACK_SIGNATURES}
    for depth in (10, 15, 20, 30, 50, 75, 300):
        assert f"monker_mtt8_{depth}bb" in ids
    for s in KNOWN_PACK_SIGNATURES:
        if s.pack_id.startswith("monker_mtt8"):
            assert s.ante_bb == 1.0
            assert s.game_format == "tournament"
            assert s.table_size == 8
            assert s.ev_units_per_bb == 2000.0
            assert s.rake_pct is None


# --- pack-file tests ---------------------------------------------------------
@needs_pack
def test_discovery_and_sb_open_size_from_real_pack():
    clear_registry()
    packs = discover_packs(REPO_ROOT / "ranges")
    mtt = {p.pack_id for p in packs if p.pack_id.startswith("monker_mtt8")}
    assert "monker_mtt8_15bb" in mtt

    from pipeline.preflop.grammars.monker_nlhe import parse
    from pipeline.preflop.pack import get_pack

    pack = get_pack("monker_mtt8_15bb")
    parsed = parse(MTT_15_ROOT / "0.0.0.0.0.0.14.rng", pack)
    state = resolve_preflop_history(parsed.action_history, pack)
    # The SB `14` open, EV-anchored at 2.5bb in the intake audit.
    assert state.sizes_bb[-1] == 2.5


@needs_pack
def test_animation_script_posts_the_ante():
    import json

    from pipeline.preflop.fact_extractor import extract_facts
    from pipeline.preflop.format_writer import _preflop_animation_script
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import get_pack
    from pipeline.preflop.spot_sampler import sample_spot

    clear_registry()
    discover_packs(REPO_ROOT / "ranges")
    pack = get_pack("monker_mtt8_15bb")
    nodes = enumerate_nodes([pack])
    node = next(
        n
        for n in nodes
        if n.actor == "BTN"
        and len(n.history_before) == 5
        and all(a.action_type.value == "Fold" for a in n.history_before)
    )
    facts = extract_facts(sample_spot(node, "A5s"), pack, equity_runouts=20)
    anim = json.loads(
        _preflop_animation_script(
            facts, pack, stakes_bb_dollars=1.0, game_format="tournament"
        )
    )
    ante_events = [
        e for e in anim["events"] if e.get("type") == "post" and e.get("ante")
    ]
    assert len(ante_events) == 1
    assert ante_events[0]["seat"] == "BB"
    assert ante_events[0]["amount_bb"] == 1.0
    # Timeline still ends at the decision.
    assert anim["events"][-1]["type"] == "decision"


# --- soft validators ---------------------------------------------------------
def test_ante_mention_allowed_on_ante_packs():
    """The no-ante-mention soft validator is gated on facts.ante_bb: MTT
    prose MUST be able to cite the 1bb ante (the unconditional version
    false-flagged 30/60 rows of the first MTT production batches)."""
    from pipeline.preflop.fact_extractor import PreflopFacts
    from pipeline.preflop.validators import soft_validate_no_ante_mention

    class _G:
        answer_explanation = "You pick up the blinds plus the 1bb ante."

    class _FactsAnte(PreflopFacts if False else object):  # duck-typed
        ante_bb = 1.0

    class _FactsCash:
        ante_bb = 0.0

    assert soft_validate_no_ante_mention(_G(), _FactsAnte()) == []
    assert soft_validate_no_ante_mention(_G(), _FactsCash())


# --- call-off merge ----------------------------------------------------------
def test_allin_merges_into_call_when_facing_full_stack_jam():
    """Facing a jam at effective-stack depth, Call and AllIn are the same
    chips -- the options must show ONE action (Aug 2026 user catch: SB JJ
    vs a 10bb jam offered both 'Call' and 'All-in')."""
    from types import SimpleNamespace

    from pipeline.preflop.grammars.types import PreflopActionType as T
    from pipeline.preflop.spot_sampler import PreflopSpot, merge_allin_call_off

    pack = _pack(stack_depth_bb=10)
    node = SimpleNamespace(
        actor="SB",
        node_id="test",
        pack_id="test_mtt8",
        history_before=tuple(_folds(3))[:0]
        + (
            ParsedAction(position="UTG", action_type=T.FOLD),
            ParsedAction(position="UTG+1", action_type=T.FOLD),
            ParsedAction(position="LJ", action_type=T.FOLD),
            ParsedAction(position="HJ", action_type=T.ALL_IN),
            ParsedAction(position="CO", action_type=T.FOLD),
            ParsedAction(position="BTN", action_type=T.FOLD),
        ),
        actions=(
            SimpleNamespace(label="Fold", action_type=T.FOLD),
            SimpleNamespace(label="Call", action_type=T.CALL),
            SimpleNamespace(label="AllIn", action_type=T.ALL_IN),
        ),
    )
    spot = PreflopSpot(
        node=node,
        hero_hand_class="JJ",
        hero_card_combo="JcJd",
        action_frequencies={"Fold": 0.1, "Call": 0.6, "AllIn": 0.3},
        dominant_action="Call",
        dominant_frequency=0.6,
        presence=1.0,
    )
    merged = merge_allin_call_off(spot, pack)
    assert "AllIn" not in merged.action_frequencies
    assert merged.action_frequencies["Call"] == pytest.approx(0.9)
    assert merged.dominant_action == "Call"
    assert merged.dominant_frequency == pytest.approx(0.9)


def test_allin_not_merged_when_a_real_raise_is_possible():
    """Facing a mere open (bet level far below stack), AllIn is a REAL
    distinct raise -- the merge must not touch it."""
    from types import SimpleNamespace

    from pipeline.preflop.grammars.types import PreflopActionType as T
    from pipeline.preflop.spot_sampler import PreflopSpot, merge_allin_call_off

    pack = _pack(stack_depth_bb=15)
    node = SimpleNamespace(
        actor="BB",
        node_id="test2",
        pack_id="test_mtt8",
        history_before=(
            ParsedAction(
                position="UTG", action_type=T.RAISE, raise_size_pct=MIN_RAISE_PCT
            ),
        )
        + tuple(
            ParsedAction(position=p, action_type=T.FOLD)
            for p in ("UTG+1", "LJ", "HJ", "CO", "BTN", "SB")
        ),
        actions=(
            SimpleNamespace(label="Fold", action_type=T.FOLD),
            SimpleNamespace(label="Call", action_type=T.CALL),
            SimpleNamespace(label="AllIn", action_type=T.ALL_IN),
        ),
    )
    spot = PreflopSpot(
        node=node,
        hero_hand_class="99",
        hero_card_combo="9c9d",
        action_frequencies={"Fold": 0.2, "Call": 0.5, "AllIn": 0.3},
        dominant_action="Call",
        dominant_frequency=0.5,
        presence=1.0,
    )
    merged = merge_allin_call_off(spot, pack)
    assert merged.action_frequencies == spot.action_frequencies


def test_rio_skill_never_fires_facing_an_all_in():
    """No postflop play means no implied odds in either direction -- the
    dominated+offsuit fallback path must not bypass the all-in exclusion
    (cross-check catch on a 10bb jam-call spot, Aug 2026)."""
    from pipeline.skill_tagger import SKILL_CATALOG, SkillContext

    rule = SKILL_CATALOG["Reverse Implied Odds"]
    base = dict(
        path="preflop",
        street="Preflop",
        archetype="fold_dominated",
        concept_tags=frozenset({"dominated", "unconnected_offsuit"}),
    )
    assert rule(SkillContext(**base, facing_all_in=False))
    assert not rule(SkillContext(**base, facing_all_in=True))
