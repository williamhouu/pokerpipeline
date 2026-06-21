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
from pipeline.postflop.facts import extract_facts  # noqa: E402
from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s  # noqa: E402
from pipeline.postflop.format_writer import (  # noqa: E402
    POSTFLOP_CSV_COLUMNS,
    build_postflop_row,
)
from pipeline.postflop.options import build_options  # noqa: E402
from pipeline.postflop.question_extractor import evaluate_spot  # noqa: E402
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
