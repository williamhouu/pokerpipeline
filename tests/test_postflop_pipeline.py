"""Tests for the postflop pipeline (pipeline.postflop.*).

Per-layer positive/negative coverage plus an end-to-end dry-run that produces
a CSV, and a mock-LLM real-run path. The synthetic fixture
(``btn_vs_bb_srp_2cJs7s``) drives everything, so nothing here needs a solver
file or an API key. Determinism is asserted explicitly (seeded equity + no
timestamps => byte-identical batches), which is the guard the brief wants
against silent regressions.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import (  # noqa: E402
    ExplanationValidationError,
    GeneratedExplanation,
)
from pipeline.postflop.action_history import (  # noqa: E402
    build_context_line,
    format_question,
)
from pipeline.postflop.app_table_format import (  # noqa: E402
    _seat_states,
    build_postflop_app_table_columns,
)
from pipeline.postflop.batch import generate_postflop_batch  # noqa: E402
from pipeline.postflop.claim_checker import (  # noqa: E402
    build_checker_user_prompt,
    check_postflop_claims,
    claim_check_to_json,
    parse_checker_response,
)
from pipeline.postflop.concept_tags import (  # noqa: E402
    PostflopTagInput,
    classify_postflop_archetype,
    compute_postflop_tags,
)
from pipeline.postflop.difficulty import compute_difficulty  # noqa: E402
from pipeline.postflop.explanation_generator import (  # noqa: E402
    build_solver_data_block,
    generate_postflop_explanation,
    placeholder_explanation,
)
from pipeline.postflop.facts import (  # noqa: E402
    _advantage_label,
    compute_currently_ahead,
    compute_range_advantage,
    extract_facts,
)
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.format_writer import (  # noqa: E402
    POSTFLOP_CSV_COLUMNS,
    POSTFLOP_ROW_COLUMNS,
    build_postflop_row,
)
from pipeline.postflop.options import build_options  # noqa: E402
from pipeline.postflop.question_extractor import evaluate_spot  # noqa: E402
from pipeline.postflop.reviser import revise_postflop_explanation  # noqa: E402
from pipeline.postflop.solve import validate_solve  # noqa: E402
from pipeline.postflop.spot_sampler import (  # noqa: E402
    enumerate_spots,
    sample_spot,
    spot_action_evs_bb,
    spot_ev_gap_bb,
)
from pipeline.postflop.validators import (  # noqa: E402
    run_postflop_audit_validators,
    run_postflop_soft_validators,
    soft_validate_equity_vs_data,
    soft_validate_verdict_vs_answer,
    validate_banned_phrases,
    validate_card_suit_consistency,
    validate_correct_answer,
    validate_no_garbled_card_glyphs,
)

SOLVE = btn_vs_bb_srp_2cJs7s()


def _spot(node_id: str, combo: str):
    return sample_spot(SOLVE.nodes[node_id], combo)


# --- solve / fixture --------------------------------------------------------
def test_fixture_validates_clean() -> None:
    assert validate_solve(SOLVE) == []


def test_fixture_has_the_four_nodes() -> None:
    assert set(SOLVE.nodes) == {
        "flop_ip_cbet", "flop_ip_facing_bet", "flop_oop_lead", "turn_oop"
    }


# --- spot sampler -----------------------------------------------------------
def test_sample_spot_normalises_and_finds_dominant() -> None:
    spot = _spot("flop_ip_cbet", "AcJc")
    assert abs(sum(spot.action_frequencies.values()) - 1.0) < 1e-9
    assert spot.dominant_action == "Bet 4bb"
    assert spot.dominant_verb == "bet"


def test_ev_gap_prefers_combo_evs() -> None:
    # AcJc combo EVs are 3.2 / 2.6 / 2.0 -> gap 0.6.
    assert spot_ev_gap_bb(_spot("flop_ip_cbet", "AcJc")) == pytest.approx(0.6)


def test_ev_gap_none_when_under_two_evs() -> None:
    # A combo with no combo_evs and a node whose actions lack ev_bb -> None.
    from pipeline.postflop.solve import NodeAction, PostflopNode
    node = PostflopNode(
        node_id="x", street="flop", board=("2c", "Js", "7s"), actor="BB",
        villain="BTN", pot_bb=5.5, effective_stack_bb=97.5,
        actions=(NodeAction("Check", "check", 0.7), NodeAction("Bet 33%", "bet", 0.3)),
        strategy={"AhKh": {"Check": 0.7, "Bet 33%": 0.3}},
        hero_range={"AhKh": 1.0}, villain_range={"QdQs": 1.0},
    )
    assert spot_ev_gap_bb(sample_spot(node, "AhKh")) is None


# --- question extractor -----------------------------------------------------
def test_worthy_spot_in_window() -> None:
    ev = evaluate_spot(_spot("flop_ip_cbet", "AcJc"))
    assert ev.is_worthy


def test_ev_gap_filter_is_off_by_default() -> None:
    # Ah5h: dominant Call 62% with a small EV gap (0.3bb). Pin the floor to
    # 0.55 so the 62% sits in-window (the default floor is now 0.65) -- this
    # test is about the EV axis: the EV filter is OFF by default (mirrors
    # preflop), so the spot is worthy.
    ev = evaluate_spot(_spot("flop_ip_facing_bet", "Ah5h"), min_frequency=0.55)
    assert ev.is_worthy


def test_optional_ev_gap_filter_drops_small_gap_when_enabled() -> None:
    # Same spot (floor pinned to 0.55 so 62% is in-window), but with the opt-in
    # filter at 0.5bb: now dropped on the EV axis (gap 0.3bb).
    ev = evaluate_spot(
        _spot("flop_ip_facing_bet", "Ah5h"), min_frequency=0.55, min_ev_gap_bb=0.5
    )
    assert not ev.is_worthy
    assert "EV gap" in ev.reason


def test_unworthy_outside_frequency_window() -> None:
    ev = evaluate_spot(_spot("flop_ip_cbet", "9c8c"), max_frequency=0.80)
    assert not ev.is_worthy  # 9c8c checks 85% > 80% cap


# --- facts ------------------------------------------------------------------
def test_facts_equity_is_deterministic() -> None:
    a = extract_facts(_spot("flop_ip_cbet", "AcJc"), SOLVE)
    b = extract_facts(_spot("flop_ip_cbet", "AcJc"), SOLVE)
    assert a.hero_equity_vs_villain == b.hero_equity_vs_villain


def test_facts_top_pair_is_high_equity() -> None:
    f = extract_facts(_spot("flop_ip_cbet", "AcJc"), SOLVE)
    assert f.made_hand == "top_pair_top_kicker"
    assert f.hero_equity_vs_villain > 0.65
    assert f.archetype == "value_bet"
    assert "c_bet_spot" in f.concept_tags


def test_facts_break_even_only_when_facing_bet() -> None:
    facing = extract_facts(_spot("flop_ip_facing_bet", "KsJd"), SOLVE)
    cbet = extract_facts(_spot("flop_ip_cbet", "AcJc"), SOLVE)
    assert facing.break_even_equity == pytest.approx(1.8 / (7.3 + 1.8))
    assert cbet.break_even_equity is None


def test_facts_hero_position_and_aggressor() -> None:
    cbet = extract_facts(_spot("flop_ip_cbet", "AcJc"), SOLVE)
    assert cbet.hero_in_position and cbet.hero_is_preflop_aggressor  # BTN
    turn = extract_facts(_spot("turn_oop", "Jh9c"), SOLVE)
    assert not turn.hero_in_position and not turn.hero_is_preflop_aggressor  # BB


# --- concept tags / archetype ----------------------------------------------
def _tag_input(**kw) -> PostflopTagInput:
    base = dict(
        street="flop", preflop_raise_count=1, n_players=2,
        hero_is_preflop_aggressor=True, hero_in_position=True,
        is_facing_bet=False, dominant_verb="bet", made_hand="overpair",
        draws=(), strength_bucket="strong", suit_distribution="two_tone",
        pair_status="unpaired", connectedness="disconnected", composite="semi_wet",
        hero_equity=0.7, break_even_equity=None,
    )
    base.update(kw)
    return PostflopTagInput(**base)


def test_archetype_value_bet_vs_bluff() -> None:
    assert classify_postflop_archetype(_tag_input(strength_bucket="strong")) == "value_bet"
    assert classify_postflop_archetype(
        _tag_input(strength_bucket="air", made_hand="no_pair_air")
    ) == "bluff"


def test_archetype_bluff_catch_facing_bet() -> None:
    arch = classify_postflop_archetype(
        _tag_input(dominant_verb="call", is_facing_bet=True, strength_bucket="medium")
    )
    assert arch == "bluff_catch"


def test_tags_fire_for_value_bet() -> None:
    tags = compute_postflop_tags(_tag_input(strength_bucket="strong"))
    assert "value_bet_spot" in tags and "c_bet_spot" in tags and "single_raised_pot" in tags


def test_adapter_node_street_and_sampling() -> None:
    from pipeline.postflop.adapters.sqlite_db import _node_street, _stride_sample

    assert _node_street("r:0:c") == "flop"  # no chance card
    assert _node_street("r:0:c:c:2c:c") == "turn"  # one chance card
    assert _node_street("r:0:c:c:2c:c:c:7h:c") == "river"  # two chance cards
    # stride sample: deterministic, evenly spaced, all when k >= len.
    items = [f"{i:03d}" for i in range(100)]
    s = _stride_sample(items, 10)
    assert len(s) == 10 and s[0] == "000" and s == sorted(s)
    assert _stride_sample(items, 500) == items
    assert _stride_sample(items, 0) == []


def test_discover_skips_backup_directories(tmp_path) -> None:
    """July 23 2026: a backup copy parked under the solves folder surfaced as
    a DUPLICATE picker entry with the same filename as the live file (no way
    to tell which is which -> silent generation from a stale solve).
    Discovery must skip backup-style and dot/underscore-prefixed subdirs."""
    from pipeline.postflop.adapters.sqlite_db import discover_db_solves

    (tmp_path / "live.db").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.db").write_bytes(b"x")  # normal subdir: kept
    for skipped in ("pre_july18_backup", "Archive", "old_solves", "_trash", ".hidden"):
        d = tmp_path / skipped
        d.mkdir()
        (d / "live.db").write_bytes(b"x")  # same name as the live file
    from pathlib import Path

    rel = sorted(
        str(Path(s.path).relative_to(tmp_path)) for s in discover_db_solves(str(tmp_path))
    )
    assert rel == ["live.db", str(Path("sub") / "nested.db")]


def test_per_street_node_caps() -> None:
    """July 23 2026 (checkdown-monotony fix): the node cap accepts a
    per-street dict, and the full-hand default samples the river much deeper
    (2500) than the flat 600 -- a shallow river sample stranded turn barrel
    lines without their river continuation, so the no-mid-hand-endings rule
    dropped them and batches over-rotated into checkdowns."""
    import inspect

    from pipeline.postflop.adapters.sqlite_db import (
        FULL_HAND_MAX_NODES_PER_STREET,
        _resolve_street_cap,
    )
    from pipeline.postflop.run import generate_full_hand_batch_from_db

    assert _resolve_street_cap(600, "river") == 600
    assert _resolve_street_cap(None, "river") is None
    assert _resolve_street_cap({"flop": 600, "river": 2500}, "river") == 2500
    assert _resolve_street_cap({"flop": 600}, "river") is None  # missing = uncapped
    assert FULL_HAND_MAX_NODES_PER_STREET["river"] >= 2500  # noqa: PLR2004
    assert FULL_HAND_MAX_NODES_PER_STREET["river"] > FULL_HAND_MAX_NODES_PER_STREET["turn"]
    # The full-hand loader defaults to the river-deep mapping.
    sig = inspect.signature(generate_full_hand_batch_from_db)
    assert sig.parameters["max_nodes_per_street"].default is FULL_HAND_MAX_NODES_PER_STREET


def test_prior_street_context() -> None:
    from pipeline.postflop.facts import _prior_street_context
    from pipeline.postflop.solve import PostflopStep

    # flop has no prior street.
    assert _prior_street_context((), "BTN", "flop") == (False, False)
    # turn after hero bet the flop -> (hero_bet_prev=True, checked_through=False).
    hist = (PostflopStep("flop", "BTN", "bet", to_bb=2.0),
            PostflopStep("flop", "BB", "call"))
    assert _prior_street_context(hist, "BTN", "turn") == (True, False)
    # turn after the flop checked through -> (False, True).
    checked = (PostflopStep("flop", "BB", "check"), PostflopStep("flop", "BTN", "check"))
    assert _prior_street_context(checked, "BTN", "turn") == (False, True)


def test_archetype_no_protection_on_river() -> None:
    # A medium hand betting a wet flop is protection; the SAME bet on the river
    # (no cards to come) is thin value, never protection.
    flop = _tag_input(strength_bucket="medium", composite="wet", street="flop")
    river = _tag_input(strength_bucket="medium", composite="wet", street="river")
    assert classify_postflop_archetype(flop) == "protection_bet"
    assert classify_postflop_archetype(river) == "value_bet"
    assert "protection_bet_spot" in compute_postflop_tags(flop)
    assert "protection_bet_spot" not in compute_postflop_tags(river)


def test_street_action_context_tags() -> None:
    # turn barrel: aggressor bet the flop, bets the turn again.
    barrel = _tag_input(street="turn", hero_is_preflop_aggressor=True,
                        hero_bet_prev_street=True)
    assert "turn_barrel" in compute_postflop_tags(barrel)
    assert "delayed_cbet" not in compute_postflop_tags(barrel)
    # delayed c-bet: aggressor checked the flop (checked through), bets the turn.
    delayed = _tag_input(street="turn", hero_is_preflop_aggressor=True,
                         hero_bet_prev_street=False, prev_street_checked_through=True)
    assert "delayed_cbet" in compute_postflop_tags(delayed)
    assert "turn_barrel" not in compute_postflop_tags(delayed)
    # probe: OOP non-aggressor leads after the aggressor declined the prior street.
    probe = _tag_input(street="turn", hero_is_preflop_aggressor=False,
                       hero_in_position=False, prev_street_checked_through=True)
    assert "probe_bet" in compute_postflop_tags(probe)
    # river bet tag + no flop-only c-bet/donk on later streets.
    river = _tag_input(street="river", hero_is_preflop_aggressor=True)
    assert "river_bet" in compute_postflop_tags(river)
    assert "c_bet_spot" not in compute_postflop_tags(river)
    assert "donk_bet_spot" not in compute_postflop_tags(
        _tag_input(street="turn", hero_is_preflop_aggressor=False, hero_in_position=False)
    )


# --- options ----------------------------------------------------------------
def test_options_multi_action_plain_labels() -> None:
    opts, correct = build_options(_spot("flop_ip_cbet", "AcJc"))
    assert opts == ["Check", "Bet 2bb", "Bet 4bb"]
    assert correct == "Bet 4bb" and correct in opts


def test_options_binary_gto_is_always_mostly_spectrum() -> None:
    # GTO style gives the 4-rung spectrum for a 2-action spot. TEAM RULE
    # (July 2026): spectrum options are size-free ("Mostly Bet", never
    # "Mostly Bet 33%") -- matching the preflop pipeline's canonical labels.
    opts, correct = build_options(_spot("flop_oop_lead", "7h6h"), style="gto")
    assert opts == ["Always Check", "Mostly Check", "Mostly Bet", "Always Bet"]
    assert correct == "Mostly Check" and correct in opts


def test_gto_spectrum_options_never_carry_a_bet_size() -> None:
    """TEAM RULE (July 2026): an Always/Mostly option never names a bet size.

    Sweep EVERY node of the fixture solve under style="gto": any option with
    an Always/Mostly prefix must contain no digits (no "53%", no "8.5bb").
    Plain-label fallbacks (3+ live verbs) and the sizing style keep sizes.
    """
    solve = btn_vs_bb_srp_2cJs7s()
    checked = 0
    for node in solve.nodes.values():
        for spot in enumerate_spots(node):
            opts, correct = build_options(spot, style="gto")
            assert correct in opts
            for opt in opts:
                if opt.startswith(("Always ", "Mostly ")):
                    assert not any(ch.isdigit() for ch in opt), (
                        f"sized spectrum option {opt!r} at {spot.node.node_id}"
                    )
                    checked += 1
    assert checked > 0  # the sweep actually exercised spectrum spots


def test_options_styles_basic_gto_auto() -> None:
    spot = _spot("flop_oop_lead", "7h6h")  # a clearly-dominant (>=80%) 2-action spot
    # basic = VERB-ONLY (July 22 2026 user rule: basic NEVER shows a bet
    # size -- Fold / Check / Call / Bet / Raise / All-in only).
    basic_opts, basic_correct = build_options(spot, style="basic")
    assert basic_opts == ["Check", "Bet"]
    for opt in basic_opts:
        assert not any(ch.isdigit() for ch in opt)
    # sizing = the old plain labels WITH sizes, as its own style.
    sizing_opts, sizing_correct = build_options(spot, style="sizing")
    assert sizing_opts == ["Check", "Bet 2bb"]
    assert sizing_correct == spot.dominant_action
    # auto picks basic here (dominant >= 80%), not the spectrum.
    assert spot.dominant_frequency >= 0.80  # noqa: PLR2004
    assert build_options(spot, style="auto") == (basic_opts, basic_correct)
    # gto forces the spectrum.
    assert build_options(spot, style="gto")[0][0] == "Always Check"
    # blend resolves deterministically to basic OR sizing, same pick per spot.
    blend = build_options(spot, style="blend")
    assert blend in ((basic_opts, basic_correct), (sizing_opts, sizing_correct))
    assert build_options(spot, style="blend") == blend
    # unknown style raises.
    import pytest as _pytest

    with _pytest.raises(ValueError, match="unknown answer style"):
        build_options(spot, style="nonsense")


def test_options_gto_collapses_multisize_to_check_vs_bet() -> None:
    # A multi-SIZE check+bet spot (3 actions, 2 verbs) collapses under gto to a
    # Check-vs-Bet spectrum -- the bet size is dropped from the option.
    from pipeline.postflop.options import frequencies_for_options
    spot = _spot("flop_ip_cbet", "AcJc")  # Check / Bet 2bb / Bet 4bb
    opts, correct = build_options(spot, style="gto")
    assert opts == ["Always Check", "Mostly Check", "Mostly Bet", "Always Bet"]
    assert correct in opts and correct.split()[-1] == "Bet"
    # neutral credit sums the bet sizes back into the "Bet" family.
    fo = frequencies_for_options(spot.action_frequencies, opts)
    assert fo["Bet"] == pytest.approx(
        sum(f for lbl, f in spot.action_frequencies.items() if lbl.startswith("Bet"))
    )
    assert fo["Check"] == pytest.approx(spot.action_frequencies.get("Check", 0.0))


def test_options_gto_keeps_plain_for_three_verb_spot() -> None:
    # Fold/Call/Raise = three action TYPES -> can't be a 2-rung spectrum -> plain.
    spot = _spot("flop_ip_facing_bet", "KsJd")
    opts, correct = build_options(spot, style="gto")
    assert opts == ["Fold", "Call", "Raise"]
    assert correct in opts


def test_options_correct_always_in_options() -> None:
    for node_id, node in SOLVE.nodes.items():
        for spot in enumerate_spots(node):
            for style in ("basic", "gto", "auto"):
                opts, correct = build_options(spot, style=style)
                assert correct in opts, (node_id, spot.hero_combo, style, correct, opts)


def test_options_gto_near_binary_collapse() -> None:
    # A 3-verb facing-bet spot whose third verb is a GTO sliver (<5%) collapses to
    # the two LIVE verbs' Always/Mostly spectrum, instead of falling back to plain
    # labels. The live pair is chosen by FREQUENCY, not aggression.
    from types import SimpleNamespace

    from pipeline.postflop.options import build_options
    from pipeline.postflop.solve import NodeAction
    acts = (
        NodeAction(label="Fold", verb="fold", freq=0.0),
        NodeAction(label="Call", verb="call", freq=0.0),
        NodeAction(label="Raise to 12bb", verb="raise", freq=0.0, to_bb=12.0, pot_fraction=1.0),
    )

    def _stub(freqs, dom, dom_freq):
        return SimpleNamespace(
            node=SimpleNamespace(actions=acts, node_id="x"),
            action_frequencies=freqs, dominant_action=dom, dominant_frequency=dom_freq,
            live_actions=acts,  # options read live_actions (artifact-strip)
        )

    # Fold 60 / Call 38 / Raise 2 -> Raise sliver dropped -> Fold-vs-Call.
    nb = _stub({"Fold": 0.60, "Call": 0.38, "Raise to 12bb": 0.02}, "Fold", 0.60)
    assert build_options(nb, style="gto") == (
        ["Always Fold", "Mostly Fold", "Mostly Call", "Always Call"], "Mostly Fold"
    )
    # Fold 3 / Call 50 / Raise 47 -> Fold (least aggressive) is the sliver, dropped
    # -> Call-vs-Raise (proves the pick is by frequency, not aggression order). The
    # raise option is size-free ("Raise", not "Raise to 12bb") -- team rule July 2026.
    cr = _stub({"Fold": 0.03, "Call": 0.50, "Raise to 12bb": 0.47}, "Call", 0.50)
    assert build_options(cr, style="gto") == (
        ["Always Call", "Mostly Call", "Mostly Raise", "Always Raise"],
        "Mostly Call",
    )
    # Genuinely 3-way (all >= 5%) -> NO collapse, plain labels (full labels kept).
    three = _stub({"Fold": 0.40, "Call": 0.35, "Raise to 12bb": 0.25}, "Fold", 0.40)
    assert build_options(three, style="gto")[0] == ["Fold", "Call", "Raise to 12bb"]


def test_facing_probe_tag_and_new_skill_rules() -> None:
    # facing_probe_spot = hero IS the aggressor, IP, prior street checked through,
    # now facing a turn/river lead (the mirror of probe_bet).
    from types import SimpleNamespace

    from pipeline.postflop.concept_tags import compute_postflop_tags
    from pipeline.postflop.skills import POSTFLOP_SKILL_RULES
    fires = _tag_input(
        is_facing_bet=True, street="turn", hero_is_preflop_aggressor=True,
        hero_in_position=True, prev_street_checked_through=True,
    )
    assert "facing_probe_spot" in compute_postflop_tags(fires)
    # Not a probe if the prior street was bet (not checked through).
    not_probe = _tag_input(
        is_facing_bet=True, street="turn", hero_is_preflop_aggressor=True,
        hero_in_position=True, prev_street_checked_through=False,
    )
    assert "facing_probe_spot" not in compute_postflop_tags(not_probe)

    probe_skill = POSTFLOP_SKILL_RULES["Facing a Probe Bet"]
    assert probe_skill(SimpleNamespace(concept_tags=["facing_probe_spot"]))
    assert not probe_skill(SimpleNamespace(concept_tags=["probe_bet"]))

    blockers = POSTFLOP_SKILL_RULES["Blockers & Card Removal"]

    def _b(effect, tags=(), archetype="value_bet"):
        return SimpleNamespace(
            blocker_effect=effect, concept_tags=list(tags), archetype=archetype
        )

    # Facing a bet (bluff-catch): blocking EITHER value or bluffs -> fires.
    assert blockers(_b("value", ["facing_bet_spot"]))
    assert blockers(_b("bluffs", ["facing_bet_spot"]))
    # Bluffing: only blocking their VALUE enables the bluff (the river-air
    # nut-flush-blocker case) -> fires.
    assert blockers(_b("value", archetype="bluff"))
    assert blockers(_b("value", ["bluff_spot"]))
    # Bluffing while blocking their BLUFFS is irrelevant to a bluff -> not tagged.
    assert not blockers(_b("bluffs", ["bluff_spot"]))
    # Neutral effect -> never.
    assert not blockers(_b("neutral", ["facing_bet_spot"]))
    # A blocker effect but a value bet / not facing a bet -> incidental, not tagged.
    assert not blockers(_b("value", ["c_bet_spot"], archetype="value_bet"))


# --- action history (multi-street) ------------------------------------------
def test_question_flop_cbet_shows_preflop_and_flop() -> None:
    q = format_question(_spot("flop_ip_cbet", "AcJc"), SOLVE)
    assert "You're on the Button with A♣️J♣️." in q
    assert "You open to 2.5bb and the Big Blind calls." in q
    assert "The flop is 2♣️ J♠️ 7♠️. The Big Blind checks." in q


def test_question_turn_shows_full_line() -> None:
    q = format_question(_spot("turn_oop", "Jh9c"), SOLVE)
    # The turn question MUST render preflop + flop + turn ahead of it.
    assert "The Button opens to 2.5bb and you call." in q
    # 1.8bb (33% of the 5.5bb pot) snaps to the 0.5bb display grid -> 2bb.
    assert "You check, the Button bets 2bb, and you call." in q
    assert "The turn is 2❤️." in q


def test_question_renders_dollars_when_not_in_bb() -> None:
    import dataclasses

    solve = dataclasses.replace(SOLVE, bb_in_dollars=2.0)
    spot = sample_spot(solve.nodes["flop_ip_cbet"], "AcJc")
    q = format_question(spot, solve, display_in_bb=False)
    assert "You open to $5" in q  # 2.5bb * $2
    assert "bb" not in q  # no big-blind amounts leak into dollar mode


def test_postflop_prompt_override(tmp_path, monkeypatch) -> None:
    from pipeline.postflop import explanation_generator as eg

    override = tmp_path / "postflop_system.txt"
    monkeypatch.setattr(eg, "_POSTFLOP_PROMPT_OVERRIDE_PATH", override)
    assert eg.load_postflop_system_prompt() == eg.POSTFLOP_SYSTEM_PROMPT  # no file
    override.write_text("CUSTOM POSTFLOP PROMPT")
    assert eg.load_postflop_system_prompt() == "CUSTOM POSTFLOP PROMPT"


def test_context_line() -> None:
    assert build_context_line(SOLVE) == "Online · $0.50/$1"


def test_context_line_states_stack_when_not_100bb() -> None:
    from dataclasses import replace  # noqa: PLC0415

    # Readers assume 100bb, so a non-100bb game must say so in the Context.
    assert "bb" not in build_context_line(SOLVE).replace("$", "")  # SOLVE is 100bb
    assert "200bb" in build_context_line(replace(SOLVE, effective_stack_bb=200.0))
    assert "40bb" in build_context_line(replace(SOLVE, effective_stack_bb=40.0))


def test_prior_street_node_picks_the_street_before() -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from pipeline.postflop.batch import _prior_street_node  # noqa: PLC0415

    nodes = {
        "r:0": SimpleNamespace(node_id="r:0", street="flop"),
        "r:0:b214": SimpleNamespace(node_id="r:0:b214", street="flop"),
        "r:0:b214:c:2c:c": SimpleNamespace(node_id="r:0:b214:c:2c:c", street="turn"),
    }
    solve = SimpleNamespace(nodes=nodes)
    # Turn question -> the deepest FLOP ancestor (the street before).
    assert _prior_street_node(nodes["r:0:b214:c:2c:c"], solve).node_id == "r:0:b214"
    # Flop question -> None (caller uses the shared preflop / flop-entry ranges).
    assert _prior_street_node(nodes["r:0:b214"], solve) is None


def test_context_line_appends_rake_when_present() -> None:
    from dataclasses import replace  # noqa: PLC0415

    # A solve solved with rake states the structure in the Context (the EVs
    # already bake it in; this is display so the framing is unambiguous).
    raked = replace(SOLVE, rake="8% cap 2bb")
    assert build_context_line(raked) == "Online · $0.50/$1 · 8% cap 2bb rake"
    # "none" / empty -> no rake suffix (unchanged framing).
    assert build_context_line(replace(SOLVE, rake="none")) == "Online · $0.50/$1"


# --- difficulty -------------------------------------------------------------
def test_difficulty_clear_spot_easier_than_close_spot() -> None:
    # 9c8c checks 85% (clear) vs AcJc bets 75% 70% (closer) -> 9c8c lower score.
    clear = compute_difficulty(extract_facts(_spot("flop_ip_cbet", "9c8c"), SOLVE))
    close = compute_difficulty(extract_facts(_spot("flop_ip_cbet", "AcJc"), SOLVE))
    assert clear.score < close.score
    assert 400 <= clear.score <= 3200


# --- validators -------------------------------------------------------------
def _facts_for(node_id="flop_ip_cbet", combo="AcJc"):
    return extract_facts(_spot(node_id, combo), SOLVE)


def _gen(prose: str, options=("Check", "Bet 33%", "Bet 75%", ""), correct="Bet 75%"):
    return GeneratedExplanation(*options, correct, prose)


def test_validate_correct_answer_rejects_off_option() -> None:
    g = _gen("Bet big.", correct="Bet 99%")
    assert not validate_correct_answer(g, _facts_for()).is_valid


def test_validate_banned_phrase_em_dash() -> None:
    g = _gen("Bet here — you have the best hand.")
    assert not validate_banned_phrases(g, _facts_for()).is_valid


def test_card_suit_allows_board_and_hero_cards() -> None:
    # Hero AcJc on board 2c Js 7s -- naming J♠️ (board) and A♣️ (hero) is fine.
    g = _gen("Your A♣️ is good and the J♠️ on board pairs you.")
    assert validate_card_suit_consistency(g, _facts_for()).is_valid


def test_card_suit_rejects_invented_card() -> None:
    g = _gen("The K❤️ changes everything.")  # not hero's, not on board
    res = validate_card_suit_consistency(g, _facts_for())
    assert not res.is_valid and "Kh" in res.error_message


def test_garbled_card_glyph_rejected() -> None:
    # A rank followed by a non-suit symbol (Taurus instead of a spade) is a
    # garble the suit-consistency check can't see (its regex needs a valid suit).
    facts = _facts_for()
    assert not validate_no_garbled_card_glyphs(_gen("The 9♉ is scary."), facts).is_valid
    assert not validate_no_garbled_card_glyphs(_gen("You hold T\U0001f535 here."), facts).is_valid
    # Real suit glyphs (with and without the variation selector) pass.
    assert validate_no_garbled_card_glyphs(_gen("The 9♠️ and K❤️."), facts).is_valid
    assert validate_no_garbled_card_glyphs(_gen("Bet 33% with top pair."), facts).is_valid
    # It is wired into the hard validator stack.
    assert not run_postflop_audit_validators(_gen("A 9♉ board."), facts).is_valid


def test_soft_verdict_flags_wrong_action() -> None:
    # Answer is "Bet 75%" (aggressive) but the verdict opens with "check".
    g = _gen("You should just check and give up.")
    warns = soft_validate_verdict_vs_answer(g, _facts_for())
    assert len(warns) == 1


def test_soft_verdict_clean_when_action_matches() -> None:
    g = _gen("Betting big is best for value here.")
    assert soft_validate_verdict_vs_answer(g, _facts_for()) == []


def test_soft_equity_flags_wrong_number() -> None:
    facts = _facts_for("flop_ip_cbet", "AcJc")  # ~79% equity
    g = _gen("You only have about 40% equity, so bet big.")
    warns = soft_validate_equity_vs_data(g, facts)
    assert len(warns) == 1


def test_soft_equity_accepts_break_even_price() -> None:
    # "you only need 20% equity to continue and you have 35%" cites the break-even
    # PRICE and hero equity -- both legitimate, neither invented. (June 2026:
    # this was the false positive that flagged half the real Call explanations.)
    facts = _facts_for("flop_ip_facing_bet", "Ah5h")
    assert facts.break_even_equity is not None
    be = round(facts.break_even_equity * 100)
    eq = round(facts.hero_equity_vs_villain * 100)
    g = _gen(
        f"You only need {be}% equity to continue and you have around {eq}%, so call."
    )
    assert soft_validate_equity_vs_data(g, facts) == []


def test_soft_equity_ignores_bet_size_percentages() -> None:
    # A bet-size % ("a 33% bet") sitting next to the word "equity" must NOT be
    # read as an invented equity number (June 2026: this FP fired on the
    # factor-list postflop prompt, where prose routinely puts bet sizes beside
    # "equity"). Only a real equity figure that matches nothing should flag.
    facts = _facts_for("flop_ip_cbet", "AcJc")  # high equity, no break-even
    eq = round(facts.hero_equity_vs_villain * 100)
    g = _gen(
        f"A small 33% bet protects your equity here. You hold about {eq}% "
        "against the range, so betting denies the overcards a free card."
    )
    assert soft_validate_equity_vs_data(g, facts) == []


def test_soft_equity_ignores_sizing_list_percentages() -> None:
    # "Sizing up to 50% or 67%" near the word "equity" is a list of bet sizes,
    # not an equity claim (the size context sits BEFORE the figure).
    facts = _facts_for("flop_ip_cbet", "AcJc")
    g = _gen("Sizing up to 50% or 67% just gets raised off your equity here.")
    assert soft_validate_equity_vs_data(g, facts) == []


def test_soft_equity_ignores_derived_gap() -> None:
    # "the missing 2%" is the gap between equity and the break-even price (a
    # correct derived number), not an invented equity figure.
    facts = _facts_for("flop_ip_facing_bet", "Ah5h")
    g = _gen("Position helps, but it does not manufacture the missing 2% of equity.")
    assert soft_validate_equity_vs_data(g, facts) == []


def test_soft_equity_ignores_action_frequencies() -> None:
    # Solver frequencies ("folding 66% of the time", "mix at 66% check") near the
    # word "equity" are not equity figures.
    facts = _facts_for("flop_ip_cbet", "AcJc")
    g1 = _gen("The draw has real equity, but folding 66% of the time is the baseline.")
    g2 = _gen("Your hand has equity; the solver does mix at 66% check here.")
    assert soft_validate_equity_vs_data(g1, facts) == []
    assert soft_validate_equity_vs_data(g2, facts) == []


def test_placeholder_passes_hard_validators() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    g = placeholder_explanation(facts, opts, correct)
    assert run_postflop_audit_validators(g, facts).is_valid
    assert run_postflop_soft_validators(g, facts) == []


def test_solver_data_block_has_key_facts() -> None:
    block = build_solver_data_block(_facts_for())
    assert "HERO EQUITY" in block and "CORRECT ACTION: Bet 4bb" in block
    assert "STRATEGIC FRAME (value_bet)" in block


# --- Layer 6 with a mock client ---------------------------------------------
class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Usage:
    input_tokens = 100
    output_tokens = 50


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]
        self.usage = _Usage()


class _MockMessages:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    def create(self, **kwargs):  # noqa: ARG002
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return _Resp(text)


class _MockClient:
    def __init__(self, texts: list[str]) -> None:
        self.messages = _MockMessages(texts)


class _CapturingMessages:
    """Like _MockMessages but records the last user-message content, so a test
    can assert what actually reached the prompt (e.g. the action-history line)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0
        self.last_user = ""

    def create(self, **kwargs):
        msgs = kwargs.get("messages") or []
        if msgs:
            self.last_user = msgs[-1].get("content", "")
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return _Resp(text)


class _CapturingClient:
    def __init__(self, texts: list[str]) -> None:
        self.messages = _CapturingMessages(texts)


def test_generate_with_mock_client_succeeds() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    client = _MockClient(["Betting big is the play. You have top pair top kicker."])
    g = generate_postflop_explanation(facts, opts, correct, SOLVE, client=client)
    assert g.correct_answer == correct
    assert "Betting big" in g.answer_explanation


def test_generate_retries_then_succeeds() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    # First response has an em dash (hard fail), second is clean.
    client = _MockClient([
        "Bet big — value.",
        "Bet big for value here with the best hand.",
    ])
    g = generate_postflop_explanation(facts, opts, correct, SOLVE, client=client)
    assert client.messages.calls == 2
    assert "—" not in g.answer_explanation


def test_generate_raises_after_two_failures() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    client = _MockClient(["Bad — prose.", "Still bad; nope."])  # both fail
    with pytest.raises(ExplanationValidationError):
        generate_postflop_explanation(facts, opts, correct, SOLVE, client=client)


# --- format writer ----------------------------------------------------------
def test_postflop_schema_prefix_matches_preflop() -> None:
    """The postflop CSV's leading columns are byte-identical (name + order) to
    the preflop CSV, so the app reads ONE layout for both paths; postflop only
    adds extra columns AFTER that shared prefix. Guards the June-2026 unification
    against drift."""
    from pipeline.preflop.format_writer import PREFLOP_CSV_COLUMNS  # noqa: PLC0415

    pre = list(PREFLOP_CSV_COLUMNS)
    post = list(POSTFLOP_CSV_COLUMNS)
    assert post[: len(pre)] == pre, "shared prefix drifted from the preflop schema"
    # July 2026 standalone declutter: the postflop extras (play-through tags +
    # diagnostics) are all trimmed from the standalone CSV, so the two shipped
    # layouts are now IDENTICAL -- the app reads one schema for both paths.
    assert post == pre, "standalone postflop schema should equal the preflop one"
    # The four shared-schema classification columns are now part of the prefix.
    for col in ("Preflop Pot Type", "Pot Participant", "Stack Depth", "exploit_notes"):
        assert col in pre and col in post


def test_build_row_has_all_columns() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    g = placeholder_explanation(facts, opts, correct)
    row = build_postflop_row(facts, g, SOLVE, compute_difficulty(facts), 1)
    # Row dicts carry the SUPERSET; the writer trims to the shipped schema.
    assert set(row) == set(POSTFLOP_ROW_COLUMNS)
    # Cards on Table is now the app's rank-suitword board token (was emoji),
    # so the CSV feeds the app's board renderer directly.
    assert row["Cards on Table"] == "2-clubs, J-spades, 7-spades"
    assert row["Hand Stage"] == "Flop"


def test_round_to_half_bb_grid() -> None:
    from pipeline.bb_display import round_to_half_bb  # noqa: PLC0415

    assert round_to_half_bb(2.14) == 2.0
    assert round_to_half_bb(4.36) == 4.5
    assert round_to_half_bb(7.8) == 8.0
    assert round_to_half_bb(2.5) == 2.5  # no-op on the grid
    assert round_to_half_bb(100.0) == 100.0
    assert round_to_half_bb(0.0) == 0.0


def test_exact_amount_str_never_snaps_to_the_display_grid() -> None:
    """The math panel's equation amounts are EXACT (a self-consistent
    equation beats a pretty one, team July 2026): 2.14bb stays 2.14bb."""
    from pipeline.bb_display import exact_amount_str  # noqa: PLC0415

    assert exact_amount_str(2.14) == "2.14bb"
    assert exact_amount_str(9.5) == "9.5bb"
    assert exact_amount_str(2.0) == "2bb"
    assert exact_amount_str(3.0, display_in_bb=False, bb_in_dollars=2.0) == "$6"
    assert exact_amount_str(3.75, display_in_bb=False, bb_in_dollars=2.0) == "$7.50"
    # No dollar rate -> falls back to bb even when dollars were requested.
    assert exact_amount_str(3.0, display_in_bb=False, bb_in_dollars=None) == "3bb"


def test_postflop_stat_notes_equation_and_hand_name() -> None:
    """Format B pot-odds equation + E1 hand naming on the postflop panel,
    driven by a REAL fixture spot so the numbers come from the node."""
    from pipeline.postflop.stat_notes import build_stat_notes  # noqa: PLC0415

    spot = _spot("flop_ip_facing_bet", "Ah5h")
    facts = extract_facts(spot, SOLVE, equity_runouts=40)
    notes = {n.key: n for n in build_stat_notes(facts)}
    po = notes["pot_odds"]
    assert po.note.startswith("call ") and "÷ (pot " in po.note
    assert po.note.endswith(f"= {po.value}.")
    # The equation reproduces its own percentage from the printed numbers.
    import re as _re

    nums = [float(x) for x in _re.findall(r"([\d.]+)bb", po.note)]
    call, pot = nums[0], nums[1]
    assert f"{call / (pot + call):.0%}" == po.value
    eq = notes["hero_equity"]
    assert eq.note.startswith("Your hand, A") and "5" in eq.note.split(",")[1]


# --- app table-state tokens (the Runout chip/seat/board renderer) -----------
def test_app_table_cbet_spot_tokens_bb() -> None:
    # BTN to act after BB checks (c-bet decision). Hero has not acted this
    # street -> bare User Seat; the villain's check renders "-0BB-check".
    facts = extract_facts(_spot("flop_ip_cbet", "AcJc"), SOLVE)
    t = build_postflop_app_table_columns(facts, SOLVE, display_in_bb=True)
    assert t["user_seat"] == "BTN-97.5BB"
    assert t["user_cards"] == "A-clubs, J-clubs"
    assert t["cards_on_table"] == "2-clubs, J-spades, 7-spades"
    assert t["seats"] == "BB-97.5BB-0BB-check"
    assert t["pot"] == "5.5BB"
    assert t["default_stack"] == "100BB"
    assert t["table_size"] == "6"


def test_app_table_facing_bet_tokens_dollars() -> None:
    # BTN faces BB's 1.8bb bet. Dollars (bb_in_dollars=1.0): remaining keeps
    # cents (95.7 -> $95.7), the villain shows the bet it made this street.
    facts = extract_facts(_spot("flop_ip_facing_bet", "KsJd"), SOLVE)
    t = build_postflop_app_table_columns(facts, SOLVE, display_in_bb=False)
    assert t["user_seat"] == "BTN-$97.5"
    assert t["seats"] == "BB-$95.7-$1.8-bet"
    assert t["pot"] == "$7.3"  # 5.5 + the unmatched 1.8 bet


def test_app_table_turn_multistreet_and_stack_invariant() -> None:
    # Turn, BB to act after flop check/bet 1.8/call. Both players invested 2.5
    # (preflop) + 1.8 (flop) = 4.3, so each has 95.7bb behind. The DISPLAY tokens
    # snap to the 0.5bb grid (95.7 -> 95.5, 9.1 -> 9), but the underlying
    # _seat_states float stays EXACT (95.7) -- proving the rounding is display-only
    # AND cross-checking that the per-player betting walk matches the IR's
    # aggregate min-stack-behind. BTN has not acted on the turn -> a bare seat.
    facts = extract_facts(_spot("turn_oop", "Jh9c"), SOLVE)
    t = build_postflop_app_table_columns(facts, SOLVE, display_in_bb=True)
    assert t["user_seat"] == "BB-95.5BB"
    assert t["seats"] == "BTN-95.5BB"
    assert t["cards_on_table"] == "2-clubs, J-spades, 7-spades, 2-hearts"
    assert t["pot"] == "9BB"
    states = _seat_states(facts, SOLVE)
    behind = min(s["remaining"] for s in states.values())  # exact float, unrounded
    assert behind == pytest.approx(95.7)  # display rounds, the math does not
    assert abs(behind - facts.spot.node.effective_stack_bb) < 1e-6


def test_app_table_lead_spot_both_seats_bare() -> None:
    # BB first to act on the flop, BTN yet to act this street: neither seat has
    # chips in front, so both render bare POS-$remaining.
    facts = extract_facts(_spot("flop_oop_lead", "7h6h"), SOLVE)
    t = build_postflop_app_table_columns(facts, SOLVE, display_in_bb=True)
    assert t["user_seat"] == "BB-97.5BB"
    assert t["seats"] == "BTN-97.5BB"


def test_app_table_dollar_conversion_uses_bb_in_dollars() -> None:
    # A $2/bb solve scales every amount by bb_in_dollars (97.5bb -> $195).
    from dataclasses import replace  # noqa: PLC0415

    solve2 = replace(SOLVE, bb_in_dollars=2.0)
    facts = extract_facts(_spot("flop_ip_facing_bet", "KsJd"), solve2)
    t = build_postflop_app_table_columns(facts, solve2, display_in_bb=False)
    assert t["user_seat"] == "BTN-$195"
    assert t["seats"] == "BB-$191.4-$3.6-bet"
    assert t["pot"] == "$14.6"


def test_per_action_ev_column_replaces_the_gap() -> None:
    # The CSV now carries the full per-action EV list, not the single gap.
    assert "action_ev_bb" in POSTFLOP_CSV_COLUMNS
    assert "ev_gap_bb" not in POSTFLOP_CSV_COLUMNS
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    g = placeholder_explanation(facts, opts, correct)
    row = build_postflop_row(facts, g, SOLVE, compute_difficulty(facts), 1)
    cell = row["action_ev_bb"]
    # One "Label: +X.XX" entry per action the hand can take, signed, bb to 2dp.
    assert cell  # the fixture exposes EVs
    parts = cell.split(", ")
    assert len(parts) == len(facts.spot.action_frequencies)
    assert all(": " in p and ("+" in p or "-" in p) for p in parts)
    # First listed is the most-frequent action (ordered like action_frequencies).
    assert parts[0].rsplit(": ", 1)[0] == facts.spot.dominant_action


def test_spot_action_evs_prefers_per_combo_and_matches_the_gap() -> None:
    spot = _spot("flop_ip_cbet", "AcJc")
    evs = spot_action_evs_bb(spot)
    assert evs is not None
    assert set(evs) <= {a.label for a in spot.node.actions}
    # The gap between the best two of these equals spot_ev_gap_bb (same source).
    top2 = sorted(evs.values(), reverse=True)[:2]
    assert top2[0] - top2[1] == pytest.approx(spot_ev_gap_bb(spot))


# --- end-to-end batch -------------------------------------------------------
def test_batch_dry_run_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "postflop.csv"
    result = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=20, dry_run=True
    )
    # 6 of the 7 fixture combos sit in the 65-99% frequency window (Ah5h at
    # 62% is now below the 0.65 floor); the EV-gap filter is off by default,
    # so those 6 are worthy.
    assert result.questions_written == 6
    assert result.failures == []
    assert out.exists() and result.meta_path is not None and result.meta_path.exists()
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert all(r["Correct Answer"] for r in rows)
    # Every row's correct answer is among its options.
    for r in rows:
        opts = [r[f"option {i}"] for i in (1, 2, 3, 4) if r[f"option {i}"]]
        assert r["Correct Answer"] in opts


def test_batch_meta_carries_range_snapshots(tmp_path: Path) -> None:
    """The meta carries flop-entry ('preflop') ranges per player + each
    question's current-street ranges, both 169-class keyed by position -- the
    data the Review page renders as visual range grids for all players."""
    import json

    out = tmp_path / "postflop.csv"
    result = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=20, dry_run=True
    )
    assert result.meta_path is not None
    meta = json.loads(result.meta_path.read_text())
    positions = set(SOLVE.positions)  # {BB, BTN}

    pre = meta["preflop_ranges"]
    assert set(pre) == positions
    for snap in pre.values():
        assert len(snap) == 169
        assert all(0.0 <= w <= 1.0 for w in snap.values())

    assert meta["questions"]
    for q in meta["questions"]:
        sr = q["street_ranges"]
        assert set(sr) == positions
        assert all(len(snap) == 169 for snap in sr.values())
        # Plus per-player action strategy grids (action-coloured chart data):
        # {position: {action_label: {hand_class: weight}}}, the actor + (when it
        # has acted) the villain.
        assert q["street_actor"] in positions
        strat = q["street_strategy"]
        assert set(strat) <= positions and strat  # one or both players
        for pos_strat in strat.values():
            assert pos_strat  # at least one action
            for snap in pos_strat.values():
                assert len(snap) == 169
                assert all(w >= 0.0 for w in snap.values())


def test_batch_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    generate_postflop_batch(solve=SOLVE, output_path=a, total_questions=20,
                            dry_run=True, write_meta=False)
    generate_postflop_batch(solve=SOLVE, output_path=b, total_questions=20,
                            dry_run=True, write_meta=False)
    assert a.read_bytes() == b.read_bytes()


def test_batch_real_path_with_mock_client(tmp_path: Path) -> None:
    out = tmp_path / "postflop_real.csv"
    client = _MockClient(["This is the play for value with a strong hand."])
    result = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=3, client=client,
    )
    assert result.questions_written == 3
    assert not result.dry_run
    assert result.total_output_tokens > 0


def test_batch_counts_prompt_cache_tokens(tmp_path: Path) -> None:
    """THE USAGE RULE (July 2026): with the shared call seam prompt-caching
    the system prompt, `usage.input_tokens` is only the UNCACHED remainder --
    the batch result must carry cache_creation/cache_read totals or the
    lifetime spend ledger silently undercounts."""

    class _CachedUsage:
        input_tokens = 100
        output_tokens = 50
        cache_creation_input_tokens = 1200
        cache_read_input_tokens = 300

    class _CachedResp:
        def __init__(self, text: str) -> None:
            self.content = [_Block(text)]
            self.usage = _CachedUsage()

    class _CachedMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):  # noqa: ARG002
            self.calls += 1
            return _CachedResp("This is the play for value with a strong hand.")

    from types import SimpleNamespace  # noqa: PLC0415

    client = SimpleNamespace(messages=_CachedMessages())
    out = tmp_path / "postflop_cached.csv"
    result = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=2, client=client,
    )
    n = client.messages.calls
    assert n >= 2
    assert result.total_input_tokens == 100 * n
    assert result.total_output_tokens == 50 * n
    assert result.total_cache_creation_tokens == 1200 * n
    assert result.total_cache_read_tokens == 300 * n


# --- ⚡ parallel LLM workers (July 2026, ported from the PLO batch) ----------
# llm_workers > 1 runs each question's LLM chain on a worker thread while ALL
# deterministic work (facts, options, difficulty, row building) stays on the
# main thread in draw order, with strictly in-order commits -- so the CSV and
# meta question records must come out IDENTICAL to a sequential run, and the
# usage totals must stay exact under concurrency (THE USAGE RULE).

def test_parallel_workers_match_sequential_output(tmp_path: Path) -> None:
    out_seq = tmp_path / "seq.csv"
    out_par = tmp_path / "par.csv"
    r1 = generate_postflop_batch(
        solve=SOLVE, output_path=out_seq, total_questions=4,
        client=_MockClient(["This is the play for value with a strong hand."]),
        llm_workers=1,
    )
    r3 = generate_postflop_batch(
        solve=SOLVE, output_path=out_par, total_questions=4,
        client=_MockClient(["This is the play for value with a strong hand."]),
        llm_workers=3,
    )
    assert r1.questions_written == r3.questions_written == 4  # noqa: PLR2004
    assert out_seq.read_bytes() == out_par.read_bytes()
    meta_seq = json.loads(r1.meta_path.read_text())
    meta_par = json.loads(r3.meta_path.read_text())
    assert meta_seq["questions"] == meta_par["questions"]
    assert meta_seq["counters"] == meta_par["counters"]
    assert meta_par["run_settings"]["llm_workers"] == 3  # noqa: PLR2004


def test_parallel_usage_totals_stay_exact(tmp_path: Path) -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    class _FixedUsage:
        input_tokens = 100
        output_tokens = 40
        cache_creation_input_tokens = 7
        cache_read_input_tokens = 3

    class _UsageResp:
        def __init__(self) -> None:
            self.content = [
                _Block("This is the play for value with a strong hand.")
            ]
            self.usage = _FixedUsage()

    class _UsageMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):  # noqa: ARG002
            self.calls += 1
            return _UsageResp()

    client = SimpleNamespace(messages=_UsageMessages())
    result = generate_postflop_batch(
        solve=SOLVE, output_path=tmp_path / "usage.csv", total_questions=4,
        client=client, llm_workers=4,
    )
    n = client.messages.calls
    assert result.questions_written == 4  # noqa: PLR2004
    assert result.total_input_tokens == 100 * n
    assert result.total_output_tokens == 40 * n
    assert result.total_cache_creation_tokens == 7 * n
    assert result.total_cache_read_tokens == 3 * n


def test_parallel_failures_recorded_and_batch_continues(tmp_path: Path) -> None:
    """Chains that fail validation (here: an em dash, twice) land in
    ``failures`` and free their pipeline slot -- the run terminates with the
    worthy pool exhausted rather than hanging or crashing."""
    bad = "This play — the banned em dash — fails the hard validators."
    result = generate_postflop_batch(
        solve=SOLVE, output_path=tmp_path / "fail.csv", total_questions=3,
        client=_MockClient([bad]), llm_workers=3,
    )
    assert result.questions_written == 0
    assert result.questions_attempted == result.worthy_spots_available
    assert len(result.failures) == result.questions_attempted


def test_compare_mechanism_two_batches_share_identical_spots(tmp_path: Path) -> None:
    # The postflop Compare page's core invariant: two batches on the SAME solve +
    # the SAME deterministic selector see byte-identical spots (no shared RNG seed
    # needed), so an A/B isolates the prompt. Mirrors what
    # run.compare_postflop_batches_from_db does, minus the .db load + the LLM.
    import csv as _csv  # noqa: PLC0415

    from admin_panel import compare as _cmp  # noqa: PLC0415 -- pure logic, no streamlit
    from pipeline.postflop.facts import preflop_aggressor  # noqa: PLC0415
    from pipeline.postflop.spot_selection import make_spot_selector  # noqa: PLC0415

    selector = make_spot_selector(
        diversify=True, aggressor=preflop_aggressor(SOLVE), ip_position=SOLVE.ip_position
    )
    a, b = tmp_path / "cmp_A.csv", tmp_path / "cmp_B.csv"
    for out in (a, b):
        generate_postflop_batch(
            solve=SOLVE, output_path=out, total_questions=4, dry_run=True,
            write_meta=False, spot_selector=selector,
        )
    rows_a = [{str(k): str(v) for k, v in r.items()}
              for r in _csv.DictReader(a.open(encoding="utf-8-sig"))]
    rows_b = [{str(k): str(v) for k, v in r.items()}
              for r in _csv.DictReader(b.open(encoding="utf-8-sig"))]
    # The node reference (old solver_reference value) lives in Notes' Node:
    # field now; the postflop join keys on the full ref (node+combo unique).
    from pipeline.provenance import node_reference_from_notes

    _ref = lambda r: node_reference_from_notes(r["Notes"])  # noqa: E731
    # Identical spot set, in the same order.
    assert [_ref(r) for r in rows_a] == [_ref(r) for r in rows_b]
    assert all(_ref(r) for r in rows_a)  # refs are present, not blank
    # The node-aware join pairs every spot.
    pairs = _cmp.join_by_spot(rows_a, rows_b, key_fn=_ref)
    assert len(pairs) == len(rows_a) > 0


def test_batch_rejects_malformed_solve(tmp_path: Path) -> None:
    import dataclasses
    bad = dataclasses.replace(SOLVE, flop=("2c", "Js"))  # 2-card flop
    with pytest.raises(ValueError, match="malformed"):
        generate_postflop_batch(
            solve=bad, output_path=tmp_path / "x.csv", total_questions=1, dry_run=True
        )


# --- #2: range-vs-range advantage (the "who is ahead here" facts) ------------
def test_advantage_label_margins() -> None:
    # 'hero' / 'villain' / 'even' off the difference and a margin.
    assert _advantage_label(0.60, 0.40, 0.02) == "hero"
    assert _advantage_label(0.40, 0.60, 0.02) == "villain"
    assert _advantage_label(0.505, 0.495, 0.02) == "even"  # inside the margin


def test_range_advantage_fields_populated_and_valid() -> None:
    facts = _facts_for()
    assert 0.0 <= facts.hero_range_equity <= 1.0
    assert facts.range_advantage in ("hero", "villain", "even")
    assert facts.nut_advantage in ("hero", "villain", "even")
    assert 0.0 <= facts.hero_nut_share <= 1.0
    assert 0.0 <= facts.villain_nut_share <= 1.0


def test_range_advantage_is_node_level_and_deterministic() -> None:
    # Same node -> identical range stats regardless of which hero combo, and
    # byte-identical across calls (fixed seed) -- the audit relies on this.
    node = SOLVE.nodes["flop_ip_cbet"]
    a = compute_range_advantage(node)
    b = compute_range_advantage(node)
    assert a == b
    # node-level: a second combo at the SAME node yields the same range equity.
    f1 = extract_facts(sample_spot(node, "AcJc"), SOLVE, equity_runouts=60)
    f2 = extract_facts(sample_spot(node, "9c8c"), SOLVE, equity_runouts=60)
    assert f1.hero_range_equity == f2.hero_range_equity


def test_range_advantage_in_data_block() -> None:
    block = build_solver_data_block(_facts_for())
    assert "RANGE ADVANTAGE:" in block
    assert "NUT ADVANTAGE:" in block


def test_range_advantage_concept_tags() -> None:
    base = dict(
        street="flop", preflop_raise_count=1, n_players=2,
        hero_is_preflop_aggressor=True, hero_in_position=True, is_facing_bet=False,
        dominant_verb="bet", made_hand="top_pair_top_kicker", draws=(),
        strength_bucket="strong", suit_distribution="rainbow", pair_status="unpaired",
        connectedness="disconnected", composite="dry", hero_equity=0.7,
        break_even_equity=None,
    )
    hero = PostflopTagInput(**base, range_advantage="hero", nut_advantage="hero")
    villain = PostflopTagInput(**base, range_advantage="villain", nut_advantage="villain")
    even = PostflopTagInput(**base, range_advantage="even", nut_advantage="even")
    assert "range_advantage" in compute_postflop_tags(hero)
    assert "nut_advantage" in compute_postflop_tags(hero)
    assert "range_disadvantage" in compute_postflop_tags(villain)
    assert "nut_disadvantage" in compute_postflop_tags(villain)
    even_tags = compute_postflop_tags(even)
    assert not any("advantage" in t for t in even_tags)


# --- currently-ahead (showdown equity composition) --------------------------
def test_currently_ahead_beats_air_loses_to_pairs() -> None:
    board = ["Kd", "7s", "3s"]
    hero = ["2d", "2c"]  # a pair of deuces
    villain = {
        "AhQh": 1.0,  # ace-high -> hero (pair of 2s) is AHEAD
        "JcTd": 1.0,  # jack-high -> AHEAD
        "KhQc": 1.0,  # pair of kings -> hero is BEHIND
    }
    ahead, behind, tied = compute_currently_ahead(hero, villain, board)
    assert ahead == pytest.approx(2 / 3)  # beats the two unpaired hands
    assert behind == pytest.approx(1 / 3)  # behind the pair of kings


def test_currently_ahead_excludes_blocked_and_is_exact() -> None:
    board = ["Kd", "7s", "3s"]
    hero = ["2d", "2c"]
    # A combo sharing the 2d (hero) or 3s (board) is not a real villain holding.
    villain = {"2dAh": 1.0, "3sQh": 1.0, "AcQc": 1.0}
    ahead, behind, tied = compute_currently_ahead(hero, villain, board)
    # Only AcQc is live (ace-high) -> hero ahead of 100% of the live range.
    assert ahead == 1.0 and behind == 0.0
    # Exact + deterministic (no runouts): identical on every call.
    assert compute_currently_ahead(hero, villain, board) == (ahead, behind, tied)
    # Empty / all-blocked range -> (0, 0), never a divide-by-zero.
    assert compute_currently_ahead(hero, {"2d2h": 1.0}, board) == (0.0, 0.0, 0.0)


def test_currently_ahead_in_facts_and_data_block() -> None:
    # The set on the c-bet node beats nearly everything; the line is in the block.
    facts = extract_facts(_spot("flop_ip_cbet", "QdQh"), SOLVE)
    assert 0.0 <= facts.currently_ahead_pct <= 1.0
    assert facts.currently_ahead_pct > 0.7  # an overpair beats most of BB's range
    block = build_solver_data_block(facts)
    assert "CURRENTLY AHEAD:" in block


def test_range_equity_column_present() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    row = build_postflop_row(
        facts, placeholder_explanation(facts, opts, correct), SOLVE,
        compute_difficulty(facts), 1,
    )
    assert "range_equity" in POSTFLOP_ROW_COLUMNS  # superset; CSV drops it
    assert "range_equity" not in POSTFLOP_CSV_COLUMNS
    assert row["range_equity"].endswith("%")


# --- #1: claim checker + reviser --------------------------------------------
def test_claim_check_parse_clean_flagged_and_failopen() -> None:
    assert parse_checker_response('{"issues": []}').passed
    flagged = parse_checker_response(
        '{"issues": [{"claim": "X", "problem": "Y"}]}'
    )
    assert not flagged.passed and flagged.issues[0].claim == "X"
    # Fails OPEN: unparseable -> clean pass (a checker hiccup never blocks a row).
    assert parse_checker_response("not json at all").passed


def test_claim_check_to_json_roundtrips() -> None:
    res = parse_checker_response('{"issues": [{"claim": "a", "problem": "b"}]}')
    assert json.loads(claim_check_to_json(res)) == [{"claim": "a", "problem": "b"}]
    assert claim_check_to_json(parse_checker_response('{"issues": []}')) == "[]"


def test_check_postflop_claims_with_mock() -> None:
    client = _MockClient(['{"issues": [{"claim": "p", "problem": "wrong"}]}'])
    res = check_postflop_claims("some prose", "SOLVER DATA block", client)
    assert not res.passed and res.issues[0].problem == "wrong"


def test_checker_user_prompt_embeds_the_action_line() -> None:
    # The line is the source of truth for the action sequence; embedding it
    # (ahead of the data block) is what stops the checker false-flagging a true
    # earlier-street reference (a donk-lead / check-raise) as invented.
    line = "The Big Blind bets 2bb, and you call. The turn is 2 of clubs."
    prompt = build_checker_user_prompt("some prose", "DATA", line)
    assert "QUESTION" in prompt and line in prompt
    assert prompt.index(line) < prompt.index("SOLVER DATA")
    # Blank / whitespace question -> no QUESTION header (older direct callers).
    assert "QUESTION" not in build_checker_user_prompt("prose", "DATA")
    assert "QUESTION" not in build_checker_user_prompt("prose", "DATA", "   ")


def test_check_postflop_claims_threads_question_through() -> None:
    cap = _CapturingClient(['{"issues": []}'])
    check_postflop_claims(
        "prose", "DATA", cap, question="The Big Blind led the flop for 2bb."
    )
    assert "The Big Blind led the flop for 2bb." in cap.messages.last_user


def test_reviser_fixes_and_keeps_options() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    original = GeneratedExplanation(
        *(opts + ["", "", "", ""])[:4], correct,
        "Bet big. You have a confusing claim here.",
    )
    fixed_prose = "Bet big for value. Your strong hand gets called by worse."
    client = _MockClient([fixed_prose])
    res = revise_postflop_explanation(
        original, facts, issues=["confusing claim -- unclear"], client=client,
    )
    assert res.changed
    assert res.explanation.answer_explanation == fixed_prose
    # options + correct are re-attached verbatim (the reviser cannot change them).
    assert res.explanation.options() == original.options()
    assert res.explanation.correct_answer == correct


def test_reviser_discards_rewrite_that_breaks_a_validator() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    original = GeneratedExplanation(
        *(opts + ["", "", "", ""])[:4], correct, "Bet big for value here.",
    )
    # BOTH attempts break a hard rule (em dash): the corrective retry runs
    # once, then the rewrite is discarded and the original ships (July 2026:
    # one retry, never an unbounded loop, second failure = old behavior).
    client = _CapturingClient(["Bet big — for value.", "Bet big — again."])
    res = revise_postflop_explanation(
        original, facts, issues=["x -- y"], client=client,
    )
    assert not res.changed
    assert res.explanation.answer_explanation == "Bet big for value here."  # original kept
    assert res.rejected_reason  # records why it was discarded (the LAST reason)
    # The retry was CORRECTIVE: the 2nd call carries the rejected text + rule.
    assert "WAS REJECTED" in client.messages.last_user
    assert "Bet big — for value." in client.messages.last_user


def test_reviser_corrective_retry_recovers_after_hard_reject() -> None:
    """The user-requested July 2026 upgrade: a rewrite that breaks a hard rule
    gets ONE corrective retry (the exact validator error + rejected text fed
    back) instead of shipping the flagged original unfixed."""
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    original = GeneratedExplanation(
        *(opts + ["", "", "", ""])[:4], correct, "Bet big for value here.",
    )
    good = "Bet big for value. Worse hands still pay you off."
    client = _CapturingClient(["Bet big — for value.", good])
    res = revise_postflop_explanation(
        original, facts, issues=["x -- y"], client=client,
    )
    assert res.changed
    assert res.explanation.answer_explanation == good
    assert "WAS REJECTED" in client.messages.last_user


def test_reviser_noop_without_issues_or_client() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    g = GeneratedExplanation(*(opts + ["", "", "", ""])[:4], correct, "Bet for value.")
    # No issues -> no-op (no call made).
    assert not revise_postflop_explanation(
        g, facts, issues=[], client=_MockClient(["x"]),
    ).changed
    # No client -> no-op (the library's no-op-without-client contract).
    assert not revise_postflop_explanation(
        g, facts, issues=["a -- b"], client=None,
    ).changed


def test_reviser_threads_question_through() -> None:
    # The reviser must see the line too, so it keeps a correct line reference
    # instead of deleting it to satisfy a false "invented line" flag.
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    original = GeneratedExplanation(
        *(opts + ["", "", "", ""])[:4], correct, "Bet for value here.",
    )
    cap = _CapturingClient(["Bet for value with the best hand here."])
    revise_postflop_explanation(
        original, facts, issues=["x -- y"], client=cap,
        question="The Big Blind led the flop and you called.",
    )
    assert "The Big Blind led the flop and you called." in cap.messages.last_user


# --- #1: batch lifecycle (claim check / auto-fix) ---------------------------
def _system_text(kw: dict) -> str:
    """The system prompt TEXT from a captured messages.create kwargs dict.

    The shared call seam wraps a string system into a cache-controlled block
    list (July 2026), so content-routing mocks must unwrap before sniffing."""
    system = kw.get("system", "")
    if isinstance(system, list):
        return "".join(b.get("text", "") for b in system)
    return str(system)


class _LifecycleMessages:
    """Content-aware mock: a claim-check call (system has 'poker editor') flags
    the ORIGINAL prose and clears the REVISED; a revise call (user has 'AUDIT
    ISSUES TO FIX') returns the rewrite; everything else is generation."""

    def __init__(self, gen: str, revised: str, *, flag: bool) -> None:
        self.gen, self.revised, self.flag = gen, revised, flag
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        system = _system_text(kw)
        user = kw["messages"][0]["content"]
        if "poker editor" in system:  # claim-check call
            if not self.flag or self.revised in user:
                return _Resp('{"issues": []}')
            return _Resp('{"issues": [{"claim": "vague", "problem": "unclear line"}]}')
        if "AUDIT ISSUES TO FIX" in user:  # revise call
            return _Resp(self.revised)
        return _Resp(self.gen)


class _LifecycleClient:
    def __init__(self, gen: str, revised: str, *, flag: bool) -> None:
        self.messages = _LifecycleMessages(gen, revised, flag=flag)


def test_batch_claim_check_only_flags(tmp_path: Path) -> None:
    out = tmp_path / "cc.csv"
    client = _LifecycleClient(
        "Check here to control the pot.", "unused", flag=True,
    )
    result = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=3, client=client,
        run_claim_checker=True, equity_runouts=60,
    )
    meta = json.loads(result.meta_path.read_text())
    assert meta["run_settings"]["run_claim_checker"] is True
    assert meta["counters"]["claim_flagged_rows"] >= 1
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # Every checked row carries a non-empty claim_check cell (flag JSON).
    assert all(r["claim_check"] for r in rows)
    assert any(r["validation_status"] == "flagged" for r in rows)


def test_batch_revise_pass_fixes_and_records(tmp_path: Path) -> None:
    out = tmp_path / "rev.csv"
    gen = "Check here to control the pot."
    revised = "Check to keep the pot small and realize your equity cheaply."
    client = _LifecycleClient(gen, revised, flag=True)
    result = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=3, client=client,
        revise_pass=True, final_audit=True, equity_runouts=60,
    )
    meta = json.loads(result.meta_path.read_text())
    c = meta["counters"]
    assert c["revise_flagged"] >= 1
    assert c["revise_fixed"] == c["revise_flagged"]  # mock always produces a valid fix
    # The shipped explanation is the REWRITE, and the revise record is captured.
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert any(revised in r["Answer Explanation"] for r in rows)
    fixed = [q for q in meta["questions"] if (q.get("revise") or {}).get("status") == "fixed"]
    assert fixed and fixed[0]["revise"]["revised_explanation"] == revised
    assert "final_audit_issues" in fixed[0]["revise"]


# --- difficulty: 3-axis + trap-aware ----------------------------------------
def test_difficulty_has_three_axes() -> None:
    d = compute_difficulty(_facts_for())
    assert 0.0 <= d.easy_freq <= 1.0
    assert 0.0 <= d.easy_concept <= 1.0
    assert 0.0 <= d.easy_hand <= 1.0
    assert 400 <= d.score <= 3200  # noqa: PLR2004


def test_difficulty_ev_gap_is_diagnostic_not_scored() -> None:
    # easy_ev is reported but does NOT move the score (postflop drops EV from the
    # blend, like PLO). Same spot, different EV gap -> identical score.
    import dataclasses
    base = _facts_for()
    lo = compute_difficulty(dataclasses.replace(base, ev_gap_bb=0.0))
    hi = compute_difficulty(dataclasses.replace(base, ev_gap_bb=3.0))
    assert lo.score == hi.score
    assert lo.easy_ev != hi.easy_ev  # the diagnostic still reflects the gap


def test_difficulty_bluff_catch_harder_than_value_bet() -> None:
    # The concept axis makes a bluff-catch (hinges on villain) harder than a
    # clear value bet, at comparable frequency.
    vb = compute_difficulty(_facts_for("flop_ip_cbet", "QdQh"))  # value_bet
    bc = compute_difficulty(_facts_for("flop_ip_facing_bet", "KsJd"))  # bluff_catch
    assert bc.easy_concept < vb.easy_concept


def test_trap_difficulty_floors_counterintuitive_fold() -> None:
    import dataclasses

    from pipeline.trap_grading import TRAP_FLOOR_MAX, TRAP_FLOOR_MIN

    base = _facts_for("flop_ip_facing_bet", "KsJd")  # facing a bet (has a price)
    # Force a PURE fold whose equity clearly CLEARS the price -> a "trap".
    # The 30pt equity-vs-price margin saturates the graded floor.
    trap = dataclasses.replace(
        base, dominant_verb="fold", dominant_action="Fold",
        dominant_frequency=1.0, hero_equity_vs_villain=0.60,
        break_even_equity=0.30, n_players=2,
    )
    off = compute_difficulty(trap, apply_trap_bump=False)
    on = compute_difficulty(trap, apply_trap_bump=True)
    assert not off.trap_bump_applied and off.score < TRAP_FLOOR_MIN
    assert on.trap_bump_applied and on.score == TRAP_FLOOR_MAX


def test_trap_difficulty_grades_by_contradiction_size() -> None:
    """Graded floor (July 2026): a mild postflop trap rates lower than a
    severe one instead of both pinning to a flat 2400."""
    import dataclasses

    from pipeline.trap_grading import TRAP_FLOOR_MIN, graded_trap_floor

    base = _facts_for("flop_ip_facing_bet", "KsJd")
    def _trap(eq: float) -> object:
        return dataclasses.replace(
            base, dominant_verb="fold", dominant_action="Fold",
            dominant_frequency=1.0, hero_equity_vs_villain=eq,
            break_even_equity=0.30, n_players=2,
        )
    mild = compute_difficulty(_trap(0.36), apply_trap_bump=True)
    severe = compute_difficulty(_trap(0.50), apply_trap_bump=True)
    assert mild.trap_bump_applied and severe.trap_bump_applied
    assert TRAP_FLOOR_MIN <= mild.score < severe.score
    assert severe.score == graded_trap_floor(0.20)


def test_trap_difficulty_skips_non_facing_bet() -> None:
    import dataclasses
    # No price (not facing a bet) -> never a trap, even with the flag on.
    base = dataclasses.replace(
        _facts_for("flop_ip_cbet", "AcJc"), break_even_equity=None,
    )
    assert not compute_difficulty(base, apply_trap_bump=True).trap_bump_applied


def test_batch_trap_difficulty_counter(tmp_path: Path) -> None:
    out = tmp_path / "trap.csv"
    res = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=20, dry_run=True,
        trap_difficulty=True,
    )
    meta = json.loads(res.meta_path.read_text())
    assert meta["run_settings"]["trap_difficulty"] is True
    assert "trap_floored" in meta["counters"]


# --- skills (postflop -> the app's catalog) ---------------------------------
def test_postflop_skill_names_are_valid_catalog_keys() -> None:
    # Every postflop skill must be a real app-catalog name (so the CSV column
    # the app consumes matches), guarding against drift.
    from pipeline.postflop.skills import POSTFLOP_SKILL_RULES
    from pipeline.skill_tagger import SKILL_CATALOG
    assert set(POSTFLOP_SKILL_RULES) <= set(SKILL_CATALOG)


def test_postflop_skills_fire_on_cbet_and_facing_bet() -> None:
    from pipeline.postflop.skills import compute_postflop_skills
    cbet = compute_postflop_skills(_facts_for("flop_ip_cbet", "QdQh"))
    assert "C-Betting" in cbet and "Value Betting" in cbet
    facing = compute_postflop_skills(_facts_for("flop_ip_facing_bet", "KsJd"))
    assert "Pot Odds" in facing and "Bluff Catching" in facing


def test_postflop_skills_strict_count() -> None:
    # Strict tagging: a typical spot gets a handful of skills, not a dozen.
    from pipeline.postflop.skills import compute_postflop_skills
    for nid, combo in (("flop_ip_cbet", "AcJc"), ("flop_oop_lead", "7h6h")):
        n = len(compute_postflop_skills(_facts_for(nid, combo)))
        assert 1 <= n <= 6  # noqa: PLR2004


def test_postflop_blockers_skill_tracks_blocker_effect() -> None:
    # The Blockers skill fires when blocker_effect is non-neutral AND the spot is
    # one where card removal drives the decision: facing a bet (bluff-catch) OR
    # bluffing (the blocker enables the bet) -- not every spot that happens to block.
    from pipeline.postflop.skills import compute_postflop_skills
    for nid in ("flop_ip_cbet", "flop_ip_facing_bet", "flop_oop_lead"):
        for spot in enumerate_spots(SOLVE.nodes[nid]):
            facts = extract_facts(spot, SOLVE, equity_runouts=40)
            fired = "Blockers & Card Removal" in compute_postflop_skills(facts)
            expect = (
                "facing_bet_spot" in facts.concept_tags
                and facts.blocker_effect in ("value", "bluffs")
            ) or (
                facts.blocker_effect == "value"
                and (
                    "bluff_spot" in facts.concept_tags
                    or facts.archetype in ("bluff", "bluff_raise")
                )
            )
            assert fired == expect


def test_faces_check_raise_line_detects_xr_only() -> None:
    # Raise == a second b<size> token (this solve has no `r` token), so a
    # check-raise faced is the line check, bet, raise => [c, b, b].
    from pipeline.postflop.skills import _faces_check_raise_line
    assert _faces_check_raise_line("r:0:c:b214:b436")  # flop: check, bet, raise
    # later street: only tokens after the last chance card count
    assert _faces_check_raise_line("r:0:b214:c:2h:c:b722:b19700")
    # NOT a check-raise:
    assert not _faces_check_raise_line("r:0:c:b214")          # c-bet faced (no raise)
    assert not _faces_check_raise_line("r:0:b214:b436")       # donk lead + raise (no check)
    assert not _faces_check_raise_line("r:0:c:b214:b436:b871")  # re-raise war (4 tokens)
    assert not _faces_check_raise_line("r:0:c")               # just a check
    assert not _faces_check_raise_line("flop_ip_cbet")        # fixture id, no tokens


def test_facing_check_raise_skill_rule() -> None:
    from types import SimpleNamespace

    from pipeline.postflop.skills import POSTFLOP_SKILL_RULES
    xr = POSTFLOP_SKILL_RULES["Facing a Check-Raise"]

    def f(node_id: str, facing: bool):
        node = SimpleNamespace(node_id=node_id, is_facing_bet=facing)
        return SimpleNamespace(spot=SimpleNamespace(node=node))

    assert xr(f("r:0:c:b214:b436", True))
    assert not xr(f("r:0:c:b214:b436", False))  # guard: must be facing a bet
    assert not xr(f("r:0:c:b214", True))        # c-bet faced, not a check-raise


def test_reverse_implied_odds_classifier() -> None:
    from types import SimpleNamespace

    from pipeline.postflop.skills import _reverse_implied_odds

    def f(made: str, draws=(), archetype: str = "bluff_catch", tags=("facing_bet_spot",),
          facing_all_in: bool = False):
        history = (
            [SimpleNamespace(all_in=True)] if facing_all_in
            else [SimpleNamespace(all_in=False)]
        )
        return SimpleNamespace(
            made_hand=made, draws=tuple(draws), archetype=archetype,
            concept_tags=list(tags),
            spot=SimpleNamespace(node=SimpleNamespace(history=history)),
        )

    # A dominated made hand FACING A BET (pays off the better hand) -> RIO.
    assert _reverse_implied_odds(f("top_pair_weak_kicker"))
    assert _reverse_implied_odds(f("flush_weak"))
    assert _reverse_implied_odds(f("straight_weak"))
    # A non-nut flush DRAW carries RIO in itself, even when not facing a bet.
    assert _reverse_implied_odds(f("ace_high", ["flush_draw_weak"], tags=()))
    # A dominated made hand NOT facing a bet (e.g. betting it thin) -> not the lesson.
    assert not _reverse_implied_odds(f("top_pair_weak_kicker", tags=()))
    # Strong / nutted -> not RIO.
    assert not _reverse_implied_odds(f("top_pair_top_kicker"))
    assert not _reverse_implied_odds(f("flush_nut"))
    assert not _reverse_implied_odds(f("ace_high", ["flush_draw_nut"]))
    # Disjoint from Implied Odds: a clean drawing call never fires RIO.
    assert not _reverse_implied_odds(
        f("ace_high", ["flush_draw_weak"], archetype="call_drawing")
    )
    # Facing an ALL-IN: no future betting, no implied odds either way (the
    # preflop rule, ported July 2026 after the full-hand cross-check caught
    # a weak flush facing a river jam tagged RIO). Both branches gated.
    assert not _reverse_implied_odds(f("flush_weak", facing_all_in=True))
    assert not _reverse_implied_odds(
        f("ace_high", ["flush_draw_weak"], tags=(), facing_all_in=True)
    )


def test_mdf_skill_fires_on_bubble_defender_vs_wide_bet() -> None:
    # MDF, done properly: fires on a BORDERLINE defender (equity ~ the calling
    # price) against a bet you must defend WIDE (MDF >= 50%), not a strength bucket.
    from types import SimpleNamespace

    from pipeline.postflop.skills import POSTFLOP_SKILL_RULES
    mdf = POSTFLOP_SKILL_RULES["Minimum Defense Frequency (MDF)"]

    def f(tags, verb, equity, break_even, pot=4.0, to_call=1.0):
        # default geometry: a 33% pot bet (pot_before 3, bet 1) -> MDF 0.75, price 0.20.
        return SimpleNamespace(
            concept_tags=tags, dominant_verb=verb,
            hero_equity_vs_villain=equity, break_even_equity=break_even,
            pot_bb=pot, to_call_bb=to_call,
        )

    # Bubble defender (equity ~ the 0.20 price) vs a 33% bet (defend wide) -> fires.
    assert mdf(f(["facing_bet_spot"], "call", 0.22, 0.20))
    assert mdf(f(["facing_bet_spot"], "fold", 0.15, 0.20))
    # Equity far above the price (a clear call, not the bubble) -> no MDF.
    assert not mdf(f(["facing_bet_spot"], "call", 0.55, 0.20))
    # An overbet (pot_before 4, bet 6 -> MDF 0.40 < 0.50): pot-odds, not defend-wide.
    assert not mdf(f(["facing_bet_spot"], "call", 0.375, 0.375, pot=10.0, to_call=6.0))
    # Not facing a bet -> no MDF.
    assert not mdf(f(["c_bet_spot"], "bet", 0.22, 0.20))


def test_skills_column_in_csv() -> None:
    facts = _facts_for("flop_ip_cbet", "QdQh")
    opts, correct = build_options(facts.spot)
    row = build_postflop_row(
        facts, placeholder_explanation(facts, opts, correct), SOLVE,
        compute_difficulty(facts), 1,
    )
    assert "skills" in POSTFLOP_CSV_COLUMNS
    assert "C-Betting" in row["skills"]


# --- #1: blocker value/bluff decomposition ----------------------------------
def test_blocker_decomposition_blocks_value() -> None:
    from pipeline.postflop.facts import compute_blocker_decomposition
    board = ["Qs", "Jd", "9s"]
    villain = {"QcQh": 1.0, "4c3c": 1.0}  # a set (value) + 4-high (bluff)
    # Hero holds Qc -> removes the value set, not the bluff.
    v_pct, b_pct, effect = compute_blocker_decomposition(["Qc", "Td"], villain, board)
    assert effect == "value" and v_pct > b_pct


def test_blocker_decomposition_blocks_bluffs() -> None:
    from pipeline.postflop.facts import compute_blocker_decomposition
    board = ["Qs", "Jd", "9s"]
    villain = {"QcQh": 1.0, "4c3c": 1.0}
    # Hero holds 4c -> removes the bluff, not the value.
    _v, _b, effect = compute_blocker_decomposition(["4c", "Td"], villain, board)
    assert effect == "bluffs"


def test_blocker_decomposition_neutral_when_no_removal() -> None:
    from pipeline.postflop.facts import compute_blocker_decomposition
    board = ["Qs", "Jd", "9s"]
    villain = {"QcQh": 1.0, "4c3c": 1.0}
    assert compute_blocker_decomposition(["2d", "2h"], villain, board)[2] == "neutral"


def test_blocker_fields_on_facts_and_tags() -> None:
    facts = _facts_for()
    assert facts.blocker_effect in ("value", "bluffs", "neutral")
    assert 0.0 <= facts.blocked_value_pct <= 1.0
    # Tag fires from the verdict.
    base = dict(
        street="flop", preflop_raise_count=1, n_players=2,
        hero_is_preflop_aggressor=False, hero_in_position=False, is_facing_bet=True,
        dominant_verb="call", made_hand="second_pair", draws=(),
        strength_bucket="medium", suit_distribution="rainbow", pair_status="unpaired",
        connectedness="disconnected", composite="dry", hero_equity=0.5,
        break_even_equity=0.3,
    )
    assert "blocks_value" in compute_postflop_tags(
        PostflopTagInput(**base, blocker_effect="value")
    )
    assert "blocks_bluffs" in compute_postflop_tags(
        PostflopTagInput(**base, blocker_effect="bluffs")
    )


def test_blocker_data_block_only_when_effect() -> None:
    import dataclasses
    facts = _facts_for()
    with_effect = dataclasses.replace(
        facts, blocker_effect="value", blocked_value_pct=0.2, blocked_bluff_pct=0.02,
    )
    neutral = dataclasses.replace(facts, blocker_effect="neutral")
    assert "BLOCKERS:" in build_solver_data_block(with_effect)
    assert "BLOCKERS:" not in build_solver_data_block(neutral)


def test_soft_blocker_direction_flags_reversal() -> None:
    import dataclasses

    from pipeline.postflop.validators import soft_validate_blocker_direction
    facts = dataclasses.replace(_facts_for(), blocker_effect="value")
    opts, correct = build_options(facts.spot)
    # prose claims blocking BLUFFS, fact says VALUE -> reversed -> flag.
    bad = GeneratedExplanation(
        *(opts + ["", "", "", ""])[:4], correct,
        "Call here. You block a lot of their bluffs, so calling is easy.",
    )
    assert soft_validate_blocker_direction(bad, facts)
    # prose consistent with the fact -> no flag.
    good = GeneratedExplanation(
        *(opts + ["", "", "", ""])[:4], correct,
        "Call here. You block their value hands, so they are bluffing more often.",
    )
    assert not soft_validate_blocker_direction(good, facts)


def test_soft_raw_percent_size_flags_bare_percent_command() -> None:
    """TEAM RULE (July 2026): sizes read in natural language, never as a raw
    solver label echo ("You should bet 53%"). Flag-only; "53% of the pot"
    and natural phrasings pass."""
    from pipeline.postflop.validators import soft_validate_raw_percent_size

    facts = _facts_for()
    opts, correct = build_options(facts.spot)

    def _expl(prose: str) -> GeneratedExplanation:
        return GeneratedExplanation(*(opts + ["", "", "", ""])[:4], correct, prose)

    assert soft_validate_raw_percent_size(
        _expl("You should bet 53% here. The hand is strong."), facts,
    )
    assert soft_validate_raw_percent_size(
        _expl("The best play is raising 120% to pressure their range."), facts,
    )
    for clean in (
        "You should bet about half the pot (4bb) here.",
        "Betting 53% of the pot gets value from worse.",
        "You have 53% equity, so a small third-pot bet is best.",
        "The best play is to check.",
    ):
        assert not soft_validate_raw_percent_size(_expl(clean), facts), clean


# --- #2: curation filters (hand-strength + decision-type) --------------------
def test_spot_strength_bucket_and_decision_type() -> None:
    from pipeline.postflop.facts import preflop_aggressor
    from pipeline.postflop.spot_selection import (
        STRENGTH_BUCKETS,
        spot_decision_type,
        spot_strength_bucket,
    )
    agg, ip = preflop_aggressor(SOLVE), SOLVE.ip_position
    # A facing-bet node -> "Facing a bet".
    fb = sample_spot(SOLVE.nodes["flop_ip_facing_bet"], "KsJd")
    assert spot_decision_type(fb, aggressor=agg, ip_position=ip) == "Facing a bet"
    assert spot_strength_bucket(fb) in STRENGTH_BUCKETS


def test_spot_selector_filters_by_strength() -> None:
    from pipeline.postflop.facts import preflop_aggressor
    from pipeline.postflop.spot_selection import (
        make_spot_selector,
        spot_strength_bucket,
    )
    worthy = [s for nid in SOLVE.nodes for s in enumerate_spots(SOLVE.nodes[nid])]
    sel = make_spot_selector(
        strength_buckets=["strong"],
        aggressor=preflop_aggressor(SOLVE), ip_position=SOLVE.ip_position,
    )
    out = sel(worthy)
    assert out  # the fixture has some 'strong' hands
    assert all(spot_strength_bucket(s) == "strong" for s in out)


def test_spot_selector_filters_by_decision_type() -> None:
    from pipeline.postflop.facts import preflop_aggressor
    from pipeline.postflop.spot_selection import make_spot_selector, spot_decision_type
    agg, ip = preflop_aggressor(SOLVE), SOLVE.ip_position
    worthy = [s for nid in SOLVE.nodes for s in enumerate_spots(SOLVE.nodes[nid])]
    sel = make_spot_selector(decision_types=["Facing a bet"], aggressor=agg, ip_position=ip)
    out = sel(worthy)
    assert all(
        spot_decision_type(s, aggressor=agg, ip_position=ip) == "Facing a bet"
        for s in out
    )


# --- #3: solve-quality / node-reach gate ------------------------------------
def test_quality_gate_passes_healthy_fixture_node() -> None:
    from pipeline.postflop.quality import node_quality_issue
    # Every fixture node is well-reached -> none flagged (the batch relies on
    # this -- the default gate must not nuke the fixture).
    for nid in SOLVE.nodes:
        assert node_quality_issue(SOLVE.nodes[nid]) is None


def test_quality_gate_flags_low_reach_and_uniform() -> None:
    from types import SimpleNamespace

    from pipeline.postflop.quality import node_quality_issue
    # Too few hero combos reach -> flagged.
    thin = SimpleNamespace(
        hero_range={f"c{i}": 1.0 for i in range(3)},
        villain_range={f"v{i}": 1.0 for i in range(20)},
        strategy={},
    )
    assert node_quality_issue(thin) is not None
    # A uniform NON-PURE mix across all combos -> flagged (untrained default).
    uniform = SimpleNamespace(
        hero_range={f"c{i}": 1.0 for i in range(20)},
        villain_range={f"v{i}": 1.0 for i in range(20)},
        strategy={f"c{i}": {"Bet 75%": 0.5, "Check": 0.5} for i in range(20)},
    )
    assert node_quality_issue(uniform) is not None
    # A uniform PURE action is legitimate (a clear c-bet) -> NOT flagged.
    pure = SimpleNamespace(
        hero_range={f"c{i}": 1.0 for i in range(20)},
        villain_range={f"v{i}": 1.0 for i in range(20)},
        strategy={f"c{i}": {"Bet 75%": 1.0} for i in range(20)},
    )
    assert node_quality_issue(pure) is None
    # Hand-specific (varied) mixes -> NOT flagged.
    varied = SimpleNamespace(
        hero_range={f"c{i}": 1.0 for i in range(20)},
        villain_range={f"v{i}": 1.0 for i in range(20)},
        strategy={
            f"c{i}": {"Bet 75%": (i % 10) / 10, "Check": 1 - (i % 10) / 10}
            for i in range(20)
        },
    )
    assert node_quality_issue(varied) is None


def test_collect_worthy_quality_gate_counts_skips(tmp_path: Path) -> None:
    from pipeline.postflop.batch import _collect_worthy
    on, skipped_on, _premise_on, _art_on = _collect_worthy(
        SOLVE, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None,
        quality_gate=True,
    )
    off, skipped_off, _premise_off, _art_off = _collect_worthy(
        SOLVE, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None,
        quality_gate=False,
    )
    # Fixture is clean: gate on or off yields the same worthy set, 0 skipped.
    assert skipped_off == 0
    assert len(on) == len(off)


# --- premise-realism gate (the port of preflop's premise gate) --------------
def _premise_node(node_id, hero_total, villain_total):
    from types import SimpleNamespace
    return SimpleNamespace(
        node_id=node_id,
        hero_range={"_": float(hero_total)},
        villain_range={"_": float(villain_total)},
    )


def test_line_premise_min_freq_reads_action_frequencies() -> None:
    # Line r:0 (BB checks) -> r:0:c (BTN bets) -> r:0:c:b50 (hero=BB faces it).
    # The action freq from parent->child = sum(child.villain_range)/sum(parent.hero_range):
    # BB check = 80/100 = 0.8; BTN bet = 60/100 = 0.6; min = 0.6.
    from types import SimpleNamespace

    from pipeline.postflop.premise import line_premise_min_freq
    nodes = {
        "r:0": _premise_node("r:0", 100, 100),
        "r:0:c": _premise_node("r:0:c", 100, 80),       # BB's checked reach mass
        "r:0:c:b50": _premise_node("r:0:c:b50", 50, 60),  # BTN's betting reach mass
    }
    solve = SimpleNamespace(nodes=nodes)
    assert abs(line_premise_min_freq(nodes["r:0:c:b50"], solve) - 0.6) < 1e-9


def test_line_premise_min_freq_none_for_first_to_act() -> None:
    # A first-to-act node (no prior action on the line) -> None, so the gate passes.
    from types import SimpleNamespace

    from pipeline.postflop.premise import line_premise_min_freq
    nodes = {"r:0": _premise_node("r:0", 100, 100)}
    assert line_premise_min_freq(nodes["r:0"], SimpleNamespace(nodes=nodes)) is None


def test_line_premise_min_freq_skips_chance_and_closed_prefixes() -> None:
    # r:0:b20:c (street closed by the call) is NOT a decision node -> absent from
    # solve.nodes -> skipped, and the chain still pairs across the chance card.
    # BTN call (across the turn) = 40/100 = 0.4; BB turn check = 95/100 = 0.95; min = 0.4.
    from types import SimpleNamespace

    from pipeline.postflop.premise import line_premise_min_freq
    nodes = {
        "r:0:b20": _premise_node("r:0:b20", 100, 100),
        "r:0:b20:c:2c": _premise_node("r:0:b20:c:2c", 100, 40),
        "r:0:b20:c:2c:c": _premise_node("r:0:b20:c:2c:c", 50, 95),
    }
    solve = SimpleNamespace(nodes=nodes)
    assert abs(line_premise_min_freq(nodes["r:0:b20:c:2c:c"], solve) - 0.4) < 1e-9


def test_premise_gate_in_collect_worthy_filters_and_counts() -> None:
    # The fixture's node ids carry no betting line, so line_premise_min_freq is
    # None everywhere -> the gate is a no-op (nothing dropped) regardless of the
    # threshold. Guards the wiring + the 4-tuple return.
    from pipeline.postflop.batch import _collect_worthy
    worthy, _lq, premise_skipped, _art = _collect_worthy(
        SOLVE, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None,
        quality_gate=True, min_premise_freq=0.5,
    )
    assert premise_skipped == 0 and len(worthy) > 0


def test_batch_quality_gate_counter(tmp_path: Path) -> None:
    out = tmp_path / "q.csv"
    res = generate_postflop_batch(
        solve=SOLVE, output_path=out, total_questions=20, dry_run=True,
        quality_gate=True,
    )
    meta = json.loads(res.meta_path.read_text())
    assert meta["run_settings"]["quality_gate"] is True
    assert "low_quality_nodes_skipped" in meta["counters"]


def test_chat_context_column_postflop() -> None:
    facts = _facts_for("flop_ip_facing_bet", "KsJd")
    opts, correct = build_options(facts.spot)
    row = build_postflop_row(
        facts, placeholder_explanation(facts, opts, correct), SOLVE,
        compute_difficulty(facts), 1,
    )
    assert "chat_context" in POSTFLOP_CSV_COLUMNS
    ctx = json.loads(row["chat_context"])
    assert ctx["pipeline"] == "postflop"
    # The chatbot-specific additions are present + grounded.
    assert ctx["full_strategy"] and "frequency_pct" in ctx["full_strategy"][0]
    assert ctx["recommended_action"] == correct
    assert ctx["villain"]["seat"] == facts.villain_position
    assert ctx["guardrails"]


def test_options_gto_pure_spot_secondary_is_second_best_by_ev() -> None:
    """STANDING RULE (July 2026, mirrors preflop): on a PURE 3-verb spot the
    alternatives all sit at ~0%, so the spectrum pairs the dominant verb
    with the SECOND-BEST verb BY EV -- the genuinely most tempting mistake
    -- not a fixed fallback. No EVs -> old plain-labels fallback."""
    from types import SimpleNamespace

    from pipeline.postflop.options import build_options
    from pipeline.postflop.solve import NodeAction

    def _stub(evs_by_label):
        acts = (
            NodeAction(label="Fold", verb="fold", freq=0.0,
                       ev_bb=evs_by_label.get("Fold")),
            NodeAction(label="Call", verb="call", freq=1.0,
                       ev_bb=evs_by_label.get("Call")),
            NodeAction(label="Raise to 12bb", verb="raise", freq=0.0,
                       to_bb=12.0, pot_fraction=1.0,
                       ev_bb=evs_by_label.get("Raise to 12bb")),
        )
        return SimpleNamespace(
            node=SimpleNamespace(actions=acts, node_id="x", combo_evs={}),
            hero_combo="AsQs",
            action_frequencies={"Fold": 0.0, "Call": 1.0, "Raise to 12bb": 0.0},
            dominant_action="Call", dominant_frequency=1.0,
            live_actions=acts, artifact_labels=frozenset(),
        )

    # Raising (+1.95) beats folding (0.00) -> pair Call with the raise. The
    # spectrum option is size-free ("Raise") -- team rule July 2026.
    spot = _stub({"Call": 2.87, "Fold": 0.0, "Raise to 12bb": 1.95})
    assert build_options(spot, style="gto") == (
        ["Always Call", "Mostly Call", "Mostly Raise", "Always Raise"],
        "Always Call",
    )
    # EV ranking is symmetric: when raising is -EV, Fold is second-best.
    spot = _stub({"Call": 1.10, "Fold": 0.0, "Raise to 12bb": -0.55})
    assert build_options(spot, style="gto") == (
        ["Always Fold", "Mostly Fold", "Mostly Call", "Always Call"],
        "Always Call",
    )
    # No EVs at all -> the old plain-labels fallback.
    spot = _stub({})
    assert build_options(spot, style="gto")[0] == [
        "Fold", "Call", "Raise to 12bb",
    ]


def test_action_ev_display_drops_unreasonable_all_in() -> None:
    """Deep-stack display rule (team, July 2026, ported from preflop): a jam
    the solver never takes (<2% freq) that is clearly dominated (1bb+ below
    best) is dropped from the action_ev_bb display -- it is noise and it
    squashes the EV chart's scale. A LIVE or near-even jam always stays."""
    from pipeline.postflop.format_writer import _format_action_evs

    evs = {"Check": 1.18, "Bet 33%": 1.18, "Bet 67%": 1.16,
           "Bet 120%": 0.76, "All-in": -2.93}
    freqs = {"Check": 0.62, "Bet 33%": 0.25, "Bet 67%": 0.09,
             "Bet 120%": 0.04, "All-in": 0.0}
    out = _format_action_evs(evs, freqs)
    assert "All-in" not in out
    assert "Check: +1.18" in out and "Bet 120%: +0.76" in out
    # The solver actually jams sometimes -> keep it.
    assert "All-in" in _format_action_evs(evs, dict(freqs, **{"All-in": 0.10}))
    # Near-even jam (short-stack rivers) -> keep it.
    assert "All-in" in _format_action_evs(
        dict(evs, **{"All-in": 0.60}), freqs
    )
    # No all-in at the node -> untouched.
    no_jam = {k: v for k, v in evs.items() if k != "All-in"}
    assert _format_action_evs(no_jam, freqs).count(":") == 4


def test_spr_note_writes_out_the_equation() -> None:
    """SPR subtext = the labeled equation from the SAME numbers node.spr
    divides (team, July 2026, like pot odds): stack / pot reproduces the
    printed ratio exactly."""
    from pipeline.postflop.stat_notes import build_stat_notes

    spot = _spot("flop_ip_facing_bet", "Ah5h")
    facts = extract_facts(spot, SOLVE, equity_runouts=40)
    note = next(n for n in build_stat_notes(facts) if n.key == "spr")
    assert note.note.startswith("stack ") and "÷ pot " in note.note
    assert f"= {facts.spr:.1f}." in note.note
    import re as _re

    stack, pot = (float(x) for x in _re.findall(r"([\d.]+)bb", note.note)[:2])
    assert f"{stack / pot:.1f}" == f"{facts.spr:.1f}"


def test_currently_tied_chop_share_and_tie_clause() -> None:
    """The chop fact (team, July 2026): ahead+behind+tied sum to 1; a
    chop-heavy spot surfaces the tie share in the data block line and the
    math panel, so "drawing dead" prose can't survive on a chopping hand."""
    from pipeline.postflop.explanation_generator import _tie_clause
    from pipeline.postflop.facts import compute_currently_ahead
    from pipeline.postflop.stat_notes import _currently_ahead_note
    from types import SimpleNamespace

    # Hero plays the board pair vs identical holdings -> pure chop.
    board = ["Ah", "Kd", "Kc", "7s", "7d"]
    ahead, behind, tied = compute_currently_ahead(
        ["2c", "3c"], {"2h3h": 1.0}, board,
    )
    assert (ahead, behind, tied) == (0.0, 0.0, 1.0)
    # Data-block clause fires at/above 5%, silent below.
    assert "chops the pot" in _tie_clause(SimpleNamespace(currently_tied_pct=0.4))
    assert _tie_clause(SimpleNamespace(currently_tied_pct=0.03)) == ""
    # Panel note names the chop share and what a chop is.
    note = _currently_ahead_note(0.0, True, "BTN's", tied=0.47)
    assert "tie (chop the pot) with about 47%" in note.note
    assert "does not win the pot" in note.note
    assert "chop" not in _currently_ahead_note(0.5, True, "BTN's", tied=0.01).note


def test_action_ev_display_normalizes_fold_to_zero() -> None:
    """EV DISPLAY NORMALIZATION (team, July 2026): sources measure EV from
    different zero points (monker: BB fold = -1.00 from hand start). The
    displayed column always shifts so Fold reads 0; gaps are unchanged."""
    from pipeline.postflop.format_writer import _format_action_evs

    evs = {"Fold": -5.72, "Call": -9.88}
    freqs = {"Fold": 0.93, "Call": 0.07}
    assert _format_action_evs(evs, freqs) == "Fold: +0.00, Call: -4.16"
    # No fold at the node -> untouched.
    assert _format_action_evs({"Check": 1.18}, {"Check": 1.0}) == "Check: +1.18"


def test_validate_hero_hand_name_and_impossible_hands():
    """The two July-2026 hard validators built from Layer-7 audit history:
    hero's hand must be NAMED per the solver's classification (the top
    residual rewrite error: "top pair" for second pair on A-K-9), and prose
    may not cite hands the visible cards make impossible ("99" on a
    9-paired board)."""
    from types import SimpleNamespace

    from pipeline.explanation_generator import GeneratedExplanation
    from pipeline.postflop.validators import (
        validate_hero_hand_name,
        validate_no_impossible_hands,
    )

    def gen(text):
        return GeneratedExplanation(
            option_1="Always Check", option_2="Mostly Check",
            option_3="Mostly Bet", option_4="Always Bet",
            correct_answer="Mostly Bet", answer_explanation=text,
        )

    facts = SimpleNamespace(
        made_hand="second_pair",
        board=("As", "Kd", "9h"),
        spot=SimpleNamespace(hero_combo="KsTs"),
    )
    # the exact observed regression: rewrite calls second pair "top pair"
    bad = validate_hero_hand_name(gen("You have top pair with a decent kicker."), facts)
    assert not bad.is_valid and "second_pair" in bad.error_message
    assert validate_hero_hand_name(gen("You have second pair here."), facts).is_valid
    # villain context is never flagged
    assert validate_hero_hand_name(
        gen("You beat their top pair rarely."), facts
    ).is_valid
    # unlisted names are never flagged
    assert validate_hero_hand_name(gen("Your king-high misses."), facts).is_valid

    # impossible-hand check: THREE 9s visible (paired board + hero 9) ->
    # "99" is dead; two visible leaves the last two, still dealable.
    facts2 = SimpleNamespace(
        made_hand="trips",
        board=("9s", "9d", "Th"),
        spot=SimpleNamespace(hero_combo="9h8s"),
    )
    bad = validate_no_impossible_hands(gen("Pairs like 99 keep calling."), facts2)
    assert not bad.is_valid and "99" in bad.error_message
    assert validate_no_impossible_hands(gen("Pairs like 88 keep calling."), facts2).is_valid
    # two 9s visible only: 99 still possible (never flag)
    facts3 = SimpleNamespace(
        made_hand="second_pair",
        board=("9s", "9d", "Th"),
        spot=SimpleNamespace(hero_combo="KsQs"),
    )
    assert validate_no_impossible_hands(gen("99 turned quads."), facts3).is_valid
    # unpaired token: "A9" with one ace left is fine; dead only at zero aces
    facts4 = SimpleNamespace(
        made_hand="two_pair",
        board=("Ah", "Ad", "9c"),
        spot=SimpleNamespace(hero_combo="AsAc"),
    )
    bad = validate_no_impossible_hands(gen("A9 is in their range."), facts4)
    assert not bad.is_valid


def test_layer7_checker_calls_report_usage():
    """Spend-logger rule (July 2026): the Layer-7 checker calls (the gate's
    best-of-2 + the final audit) must report usage like generation and the
    reviser do -- they used to burn tokens the lifetime ledger never saw."""
    from types import SimpleNamespace

    from pipeline.explanation_generator import GeneratedExplanation
    from pipeline.postflop.layer7 import run_layer7_audit

    usage = SimpleNamespace(input_tokens=900, output_tokens=30)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"issues": []}')],
        usage=usage,
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    seen: list[object] = []
    expl = GeneratedExplanation(
        option_1="Fold", option_2="Call", option_3="", option_4="",
        correct_answer="Call", answer_explanation="The best play is to call.",
    )
    out = run_layer7_audit(
        expl, facts=SimpleNamespace(),  # facts only feed the reviser path
        solver_data_block="STREET: river", question_text="q",
        node_id="r:0", client=client, model="test-model", temperature=0.0,
        max_tokens=512, system_prompt=None, checker_prompt="check",
        run_claim_checker=True, revise_pass=False, final_audit=False,
        usage_callback=seen.append,
    )
    assert out.claim_check_json == "[]"
    assert len(seen) == 1 and seen[0].input_tokens == 900

    # The revise gate runs best-of-2: with a clean gate, exactly 2 checker
    # calls report usage (no reviser call happens).
    seen.clear()
    run_layer7_audit(
        expl, facts=SimpleNamespace(),
        solver_data_block="STREET: river", question_text="q",
        node_id="r:0", client=client, model="test-model", temperature=0.0,
        max_tokens=512, system_prompt=None, checker_prompt="check",
        run_claim_checker=False, revise_pass=True, final_audit=True,
        usage_callback=seen.append,
    )
    assert len(seen) == 2


def test_validate_no_list_formatting_postflop() -> None:
    """The postflop twin of the preflop list-formatting guard: a rewrite that
    restructures prose into bullets is hard-rejected (and the reviser's
    corrective retry then reformats it as sentences)."""
    from pipeline.postflop.validators import validate_no_list_formatting

    facts = _facts_for()

    def gen(text):
        return GeneratedExplanation("Check", "Bet 75%", "", "", "Bet 75%", text)

    live_sample = (
        "Call. Here's why:\n"
        "- You need 26% to continue and you have 23%.\n"
        "- BB's line is uncapped on bluffs.\n"
    )
    assert not validate_no_list_formatting(gen(live_sample), facts).is_valid
    assert validate_no_list_formatting(
        gen("Bet 75% for value. Worse hands call, and 5-4 suited folds out."),
        facts,
    ).is_valid


def test_solve_quality_flags_name_the_difficult_files() -> None:
    """July 22 2026 (user ask): the picker must flag known-problem solves
    with a plain-English reason. July 23 2026: the v7 family is EXONERATED
    (the "inconsistency" was the intake check's own per-street misread of
    the cumulative bet tokens) -- v7 files are informational notes now, with
    per-file residual/quirk notes for the two named SRP files; only the v6
    trial stays a warn; unknown files carry no flag."""
    from pipeline.postflop.adapters.sqlite_db import solve_quality_flag

    # The one v7 file with a (small) real residual names it specifically.
    sev, text = solve_quality_flag("BTN_vs_BB_SRP_200bb_Kd7s3s_v7.db")
    assert sev == "info"
    assert "residual" in text and "Production-usable" in text
    # The generic v7 entry records the retraction.
    sev7, text7 = solve_quality_flag("BTN_vs_BB_3BP_200bb_AsKd9h_v7.db")
    assert sev7 == "info"
    assert "retracted" in text7
    # Ts9s5d keeps its flop-lead quirk note.
    sev_t, text_t = solve_quality_flag("BTN_vs_BB_SRP_200bb_Ts9s5d_v7.db")
    assert sev_t == "info"
    assert "leads" in text_t or "donk" in text_t
    sev8, text8 = solve_quality_flag("BTN_vs_BB_SRP_100bb_QsJd9s_v8.db")
    assert sev8 == "info"
    assert "donk" in text8
    assert solve_quality_flag("BTN_vs_BB_SRP_100bb_trial_v6.db")[0] == "warn"
    assert solve_quality_flag("Fresh_export_v9.db") is None


def test_context_rake_strips_internal_chip_units() -> None:
    """July 22 2026 (user ask): the vendor metadata's "(300 chips)" is the
    same cap in the solver's internal currency and means nothing to a
    reader -- the Context and the picker both show the bb figure only."""
    from pipeline.postflop.action_history import display_rake

    assert display_rake("10% cap 3bb (300 chips)") == "10% cap 3bb"
    assert display_rake("8% cap 2bb") == "8% cap 2bb"
    assert display_rake("none") == "none"


# --- second rewrite round vs final-audit flags (July 2026, strict-clean) ----
class _TwoRoundMessages:
    """Content-aware mock for the second-rewrite lifecycle: the claim checker
    flags the ORIGINAL and the FIRST rewrite, and clears only the SECOND;
    revise calls return rewrite 1 then rewrite 2."""

    def __init__(self, gen: str, revised1: str, revised2: str) -> None:
        self.gen, self.revised1, self.revised2 = gen, revised1, revised2
        self.revise_calls = 0

    def create(self, **kw):
        system = _system_text(kw)
        user = kw["messages"][0]["content"]
        if "poker editor" in system:  # claim-check call
            if self.revised2 in user:
                return _Resp('{"issues": []}')
            return _Resp(
                '{"issues": [{"claim": "vague", "problem": "unclear line"}]}'
            )
        if "AUDIT ISSUES TO FIX" in user:  # revise call
            self.revise_calls += 1
            return _Resp(self.revised1 if self.revise_calls == 1 else self.revised2)
        return _Resp(self.gen)


class _TwoRoundClient:
    def __init__(self, gen: str, revised1: str, revised2: str) -> None:
        self.messages = _TwoRoundMessages(gen, revised1, revised2)


def _run_layer7(client, **flags):
    from pipeline.postflop.claim_checker import (
        POSTFLOP_CHECKER_SYSTEM_PROMPT,
    )
    from pipeline.postflop.layer7 import run_layer7_audit

    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    explanation = GeneratedExplanation(
        *(opts + ["", "", "", ""])[:4], correct, "Check to control the pot.",
    )
    return run_layer7_audit(
        explanation, facts,
        solver_data_block=build_solver_data_block(facts),
        question_text="You checked. Villain bets.",
        node_id=facts.spot.node.node_id, client=client, model="test-model",
        temperature=0.0, max_tokens=64, system_prompt=None,
        checker_prompt=POSTFLOP_CHECKER_SYSTEM_PROMPT,
        run_claim_checker=False, revise_pass=True, final_audit=True,
        **flags,
    )


def test_layer7_second_rewrite_converts_flagged_to_clean() -> None:
    """The 🧼 lever: a rewrite the final audit still flags gets ONE more
    revise round against those flags, and the re-audited clean rewrite ships
    with no remaining issues (so strict-clean stops churning the hand)."""
    r1 = "Check to keep the pot small on this flop."
    r2 = "Check to keep the pot small and realize your equity cheaply."
    out = _run_layer7(_TwoRoundClient("gen", r1, r2), second_rewrite=True)
    assert out.explanation.answer_explanation == r2
    assert out.final_audit_issues == []
    assert out.remaining_issues == []
    assert out.second_rewrite_attempted == 1
    assert out.second_rewrite_fixed == 1
    rec = out.revise_record
    assert rec["status"] == "fixed"
    assert rec["second_rewrite"]["status"] == "fixed"
    assert rec["second_rewrite"]["issues_before"]  # what round 2 targeted
    assert rec["revised_explanation"] == r2
    assert rec["final_audit_issues"] == []
    # And the claim_check CSV cell reflects the CLEAN final state.
    assert out.claim_check_json.strip() in ("[]", '{"issues": []}') or (
        "vague" not in out.claim_check_json
    )


def test_layer7_second_rewrite_off_keeps_round_one_flags() -> None:
    """Default (flag-only final audit, pre-July-22 behaviour): the first
    rewrite ships still flagged; no second revise call is made."""
    r1 = "Check to keep the pot small on this flop."
    r2 = "Check to keep the pot small and realize your equity cheaply."
    client = _TwoRoundClient("gen", r1, r2)
    out = _run_layer7(client, second_rewrite=False)
    assert out.explanation.answer_explanation == r1
    assert out.final_audit_issues  # still flagged
    assert out.second_rewrite_attempted == 0
    assert client.messages.revise_calls == 1  # never asked for rewrite 2
    assert "second_rewrite" not in out.revise_record


def test_layer7_second_rewrite_discarded_keeps_round_one_text() -> None:
    """A second rewrite that breaks a hard rule is discarded (after the
    reviser's own corrective retry): round 1's text AND flags are kept, and
    the discard is recorded."""
    r1 = "Check to keep the pot small on this flop."
    bad = "Check — to keep the pot small."  # em dash = hard-rule violation
    out = _run_layer7(_TwoRoundClient("gen", r1, bad), second_rewrite=True)
    assert out.explanation.answer_explanation == r1
    assert out.final_audit_issues  # round 1's flags survive
    assert out.second_rewrite_attempted == 1
    assert out.second_rewrite_fixed == 0
    second = out.revise_record["second_rewrite"]
    assert second["status"] == "discarded"
    assert second["rejected_reason"]


def test_bet_labels_state_bb_amounts_never_pot_percent() -> None:
    """TEAM STANDING RULE (July 23 2026): answer options that carry a bet
    size state the amount in BIG BLINDS ("Bet 2bb"), never the percentage
    of the pot ("Bet 33%"). Enforced at the adapter label (labels are keys
    everywhere: options, action_frequencies, SOLVER DATA, neutral credit,
    showdown matching), so every surface inherits it. The exact pot
    fraction still rides in NodeAction.pot_fraction for the data block."""
    import re

    from pipeline.postflop.fixtures import (
        btn_vs_bb_full_hand_2cJs7s,
        btn_vs_bb_srp_2cJs7s,
    )
    from pipeline.postflop.options import build_options
    from pipeline.postflop.spot_sampler import sample_spot

    pct = re.compile(r"\d+\s*%")
    for solve in (btn_vs_bb_srp_2cJs7s(), btn_vs_bb_full_hand_2cJs7s()):
        for node in solve.nodes.values():
            for action in node.actions:
                assert not pct.search(action.label), (
                    f"{node.node_id}: label {action.label!r} carries a pot "
                    "percentage -- bet labels must state bb amounts"
                )
            for combo in node.strategy:
                spot = sample_spot(node, combo)
                for style in ("basic", "sizing", "gto", "auto", "blend"):
                    options, correct = build_options(spot, style=style)
                    for o in options + [correct]:
                        assert not pct.search(o), (
                            f"{style} option {o!r} carries a pot percentage"
                        )


def test_range_advantage_node_cache_is_transparent() -> None:
    """The per-node memo must return byte-identical values to a fresh call.

    INVARIANT (facts.py _RANGE_ADV_CACHE): caching may never change a value --
    batches and the re-verifiers depend on compute_range_advantage being a
    pure function of the node. A recycled id() must not alias a different
    node (the node object is pinned in the cache entry).
    """
    from pipeline.postflop import facts as facts_mod
    from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s

    solve = btn_vs_bb_srp_2cJs7s()
    node = next(iter(solve.nodes.values()))
    facts_mod._RANGE_ADV_CACHE.clear()
    fresh = facts_mod.compute_range_advantage(node)
    assert id(node) in facts_mod._RANGE_ADV_CACHE
    cached = facts_mod.compute_range_advantage(node)
    assert cached == fresh
    # The pinned object guards against id-reuse aliasing.
    pinned, value = facts_mod._RANGE_ADV_CACHE[id(node)]
    assert pinned is node and value == fresh
    # A stale entry whose pinned object differs is ignored, not returned.
    facts_mod._RANGE_ADV_CACHE[id(node)] = (object(), ("bogus",) * 5)
    assert facts_mod.compute_range_advantage(node) == fresh
    facts_mod._RANGE_ADV_CACHE.clear()
