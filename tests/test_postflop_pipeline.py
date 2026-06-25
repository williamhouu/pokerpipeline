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
from pipeline.postflop.batch import generate_postflop_batch  # noqa: E402
from pipeline.postflop.claim_checker import (  # noqa: E402
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
    compute_range_advantage,
    extract_facts,
)
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.format_writer import (  # noqa: E402
    POSTFLOP_CSV_COLUMNS,
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
    assert spot.dominant_action == "Bet 75%"
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
    assert opts == ["Check", "Bet 33%", "Bet 75%"]
    assert correct == "Bet 75%" and correct in opts


def test_options_binary_gto_is_always_mostly_spectrum() -> None:
    # GTO style gives the 4-rung spectrum for a 2-action spot.
    opts, correct = build_options(_spot("flop_oop_lead", "7h6h"), style="gto")
    assert opts == ["Always Check", "Mostly Check", "Mostly Bet 33%", "Always Bet 33%"]
    assert correct == "Mostly Check" and correct in opts


def test_options_styles_basic_gto_auto() -> None:
    spot = _spot("flop_oop_lead", "7h6h")  # a clearly-dominant (>=80%) 2-action spot
    # basic = plain labels.
    basic_opts, basic_correct = build_options(spot, style="basic")
    assert basic_opts == ["Check", "Bet 33%"]
    assert basic_correct == spot.dominant_action
    # auto picks basic here (dominant >= 80%), not the spectrum.
    assert spot.dominant_frequency >= 0.80  # noqa: PLR2004
    assert build_options(spot, style="auto") == (basic_opts, basic_correct)
    # gto forces the spectrum.
    assert build_options(spot, style="gto")[0][0] == "Always Check"
    # unknown style raises.
    import pytest as _pytest

    with _pytest.raises(ValueError, match="unknown answer style"):
        build_options(spot, style="nonsense")


def test_options_gto_collapses_multisize_to_check_vs_bet() -> None:
    # A multi-SIZE check+bet spot (3 actions, 2 verbs) collapses under gto to a
    # Check-vs-Bet spectrum -- the bet size is dropped from the option.
    from pipeline.postflop.options import frequencies_for_options
    spot = _spot("flop_ip_cbet", "AcJc")  # Check / Bet 33% / Bet 75%
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
    assert "You check, the Button bets 1.8bb, and you call." in q
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
    assert "HERO EQUITY" in block and "CORRECT ACTION: Bet 75%" in block
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
def test_build_row_has_all_columns() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    g = placeholder_explanation(facts, opts, correct)
    row = build_postflop_row(facts, g, SOLVE, compute_difficulty(facts), 1)
    assert set(row) == set(POSTFLOP_CSV_COLUMNS)
    assert row["Cards on Table"] == "2♣️ J♠️ 7♠️"
    assert row["Hand Stage"] == "Flop"


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


def test_range_equity_column_present() -> None:
    facts = _facts_for()
    opts, correct = build_options(facts.spot)
    row = build_postflop_row(
        facts, placeholder_explanation(facts, opts, correct), SOLVE,
        compute_difficulty(facts), 1,
    )
    assert "range_equity" in POSTFLOP_CSV_COLUMNS
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
    # The rewrite has an em dash (a hard-validator failure) -> discarded.
    client = _MockClient(["Bet big — for value."])
    res = revise_postflop_explanation(
        original, facts, issues=["x -- y"], client=client,
    )
    assert not res.changed
    assert res.explanation.answer_explanation == "Bet big for value here."  # original kept
    assert res.rejected_reason  # records why it was discarded


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


# --- #1: batch lifecycle (claim check / auto-fix) ---------------------------
class _LifecycleMessages:
    """Content-aware mock: a claim-check call (system has 'poker editor') flags
    the ORIGINAL prose and clears the REVISED; a revise call (user has 'AUDIT
    ISSUES TO FIX') returns the rewrite; everything else is generation."""

    def __init__(self, gen: str, revised: str, *, flag: bool) -> None:
        self.gen, self.revised, self.flag = gen, revised, flag
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        system = kw.get("system", "")
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
    base = _facts_for("flop_ip_facing_bet", "KsJd")  # facing a bet (has a price)
    # Force a PURE fold whose equity clearly CLEARS the price -> a "trap".
    trap = dataclasses.replace(
        base, dominant_verb="fold", dominant_action="Fold",
        dominant_frequency=1.0, hero_equity_vs_villain=0.60,
        break_even_equity=0.30, n_players=2,
    )
    off = compute_difficulty(trap, apply_trap_bump=False)
    on = compute_difficulty(trap, apply_trap_bump=True)
    assert not off.trap_bump_applied and off.score < 2400  # noqa: PLR2004
    assert on.trap_bump_applied and on.score >= 2400  # noqa: PLR2004


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


def test_postflop_skills_no_blockers_skill() -> None:
    # Postflop ships no blocker data, so the Blockers skill must never fire.
    from pipeline.postflop.skills import compute_postflop_skills
    for nid in ("flop_ip_cbet", "flop_ip_facing_bet", "flop_oop_lead"):
        for spot in enumerate_spots(SOLVE.nodes[nid]):
            facts = extract_facts(spot, SOLVE, equity_runouts=40)
            assert "Blockers & Card Removal" not in compute_postflop_skills(facts)


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
    on, skipped_on = _collect_worthy(
        SOLVE, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None,
        quality_gate=True,
    )
    off, skipped_off = _collect_worthy(
        SOLVE, min_frequency=0.65, max_frequency=0.99, min_ev_gap_bb=None,
        quality_gate=False,
    )
    # Fixture is clean: gate on or off yields the same worthy set, 0 skipped.
    assert skipped_off == 0
    assert len(on) == len(off)


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
