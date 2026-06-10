"""Tests for pipeline.plo.explanation_generator (Layer 6, mock client)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import ExplanationValidationError  # noqa: E402
from pipeline.plo.explanation_generator import (  # noqa: E402
    VOICE_RULES_PLO,
    build_plo_system_prompt,
    build_solver_data,
    generate_plo_answer_explanation,
)
from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.gold_examples import load_plo_gold_examples  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType  # noqa: E402
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

CARDS = ("As", "Ks", "Ah", "Kh")


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [type("C", (), {"text": text})()]
        self.usage = None


class _Messages:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def create(self, **_kw: object) -> _Resp:
        return _Resp(self._texts.pop(0))


class _MockClient:
    def __init__(self, *texts: str) -> None:
        self.messages = _Messages(list(texts))


def _json(prose: str) -> str:
    return json.dumps({"answer_explanation": prose})


def _facts() -> PloFacts:
    node = PloDecisionNode(
        actor="HJ",
        history_before=(PloAction("LJ", PloActionType.RAISE, 100),),
        actions=(),
        history_stem="40100",
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=CARDS,
        action_frequencies={"Call": 0.7, "Raise 100%": 0.3},
        ev_by_action={"Call": 2.0, "Raise 100%": 1.0, "Fold": -7.0},
        presence=1.0,
    )
    return PloFacts(
        spot=spot,
        hand_class=classify_plo_hand(CARDS),
        archetype="3bet_for_value",
        villain_stats=PloVillainStats(seat="LJ", action_label="Raise 100%", weighted_combo_count=1.0, pct_of_dealt_hands=18.0),
        hero_equity_vs_villain=0.55,
        hero_range_equity_vs_villain=0.42,
    )


# --- prompt + data block --------------------------------------------------
def test_ships_without_examples_by_default():
    assert load_plo_gold_examples() == ()
    prompt = build_plo_system_prompt()
    assert "EXAMPLES" not in prompt


def test_system_prompt_has_rules_and_bans_em_dash():
    prompt = build_plo_system_prompt()
    assert "VOICE RULES" in prompt
    assert "Never use an em dash" in prompt
    assert "BANNED PHRASES" in prompt
    assert len(VOICE_RULES_PLO) == 13  # noqa: PLR2004
    # The clean-final-draft rule (kills the self-correction artifact).
    assert any("clean, final draft" in r for r in VOICE_RULES_PLO)
    # The preflop blocker-framing rule (card removal, not flush math).
    assert any("blockers as preflop CARD REMOVAL" in r for r in VOICE_RULES_PLO)
    # Paragraphs + say-only-what-drives-the-spot.
    assert any("short paragraphs" in r for r in VOICE_RULES_PLO)
    assert any("only what actually drives" in r for r in VOICE_RULES_PLO)


def test_examples_seam_injects_when_provided():
    prompt = build_plo_system_prompt(
        examples=({"question": "Q1", "answer_explanation": "A1"},)
    )
    assert "EXAMPLES" in prompt
    assert "A1" in prompt


def test_solver_data_has_the_facts():
    data = build_solver_data(_facts(), ["Fold", "Call", "3-bet"], "Call")
    assert data["correct_action"] == "Call"
    assert data["your_hand_equity_vs_villain_range_pct"] == 55  # noqa: PLR2004
    assert data["villain"]["seat"] == "UTG"  # display code (pack seat is LJ)
    assert "3bet_for_value" in data["strategic_frame"]
    assert data["your_hand"]  # emoji cards


def test_solver_data_situation_includes_folds():
    # The LLM sees only this prose (no table render), so the folds must be in
    # it -- otherwise "facing a squeeze after the opener folded" (heads-up)
    # is indistinguishable from "opener still in" (multiway).
    node = PloDecisionNode(
        actor="SB",
        history_before=(
            PloAction("HJ", PloActionType.RAISE, 100),
            PloAction("SB", PloActionType.CALL, None),
            PloAction("BB", PloActionType.RAISE, 100),
            PloAction("HJ", PloActionType.FOLD, None),
        ),
        actions=(),
        history_stem="x",
    )
    spot = PloSpot(
        node=node,
        hero_index=0,
        hero_label="x",
        hero_cards=CARDS,
        action_frequencies={"Fold": 0.95, "Call": 0.05},
        presence=1.0,
    )
    facts = PloFacts(
        spot=spot, hand_class=classify_plo_hand(CARDS), archetype="fold_pot_odds"
    )
    data = build_solver_data(facts, ["Fold", "Call"], "Fold")
    assert "The Hijack folds." in data["situation"]


# --- generation -----------------------------------------------------------
def test_generation_returns_prose():
    client = _MockClient(_json("You should 3-bet here. Strong double-suited aces in position."))
    result = generate_plo_answer_explanation(_facts(), ["Fold", "Call", "3-bet"], "3-bet", client=client, examples=())
    assert result.correct_answer == "3-bet"
    assert "3-bet" in result.answer_explanation
    assert (result.option_1, result.option_2, result.option_3) == ("Fold", "Call", "3-bet")


def test_em_dash_and_semicolon_are_guaranteed_out():
    # Even when the model emits them, the deterministic strip removes them.
    client = _MockClient(_json("Call here. It plays well in position — double-suited; take a flop."))
    result = generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "Call", client=client, examples=())
    assert "—" not in result.answer_explanation
    assert ";" not in result.answer_explanation
    assert "–" not in result.answer_explanation


def test_banned_phrase_triggers_a_retry():
    # First attempt uses a banned phrase; the clean retry is accepted.
    client = _MockClient(
        _json("Call. We should leverage our position."),  # 'leverage' is banned
        _json("Call. Your position lets you realize this hand's equity."),
    )
    result = generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "Call", client=client, examples=())
    assert "leverage" not in result.answer_explanation.lower()


def test_correct_answer_must_be_an_option():
    with pytest.raises(ValueError, match="not in options"):
        generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "3-bet", client=_MockClient(), examples=())


def test_unparseable_response_raises_after_retries():
    client = _MockClient("not json at all", "still not json")
    with pytest.raises(ExplanationValidationError):
        generate_plo_answer_explanation(_facts(), ["Fold", "Call"], "Call", client=client, examples=(), max_retries=1)


# --- system_prompt override (the prompt-library / Compare seam) ------------
class _CapturingMessages:
    """Records the ``system`` kwarg of each create() call."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.systems: list[str] = []

    def create(self, **kw: object) -> _Resp:
        self.systems.append(str(kw.get("system", "")))
        return _Resp(self.text)


class _CapturingClient:
    def __init__(self, text: str) -> None:
        self.messages = _CapturingMessages(text)


def test_system_prompt_override_is_used_verbatim():
    client = _CapturingClient(_json("Call. It plays well in position."))
    custom = "CUSTOM PLO SYSTEM PROMPT -- edited in the admin panel."
    generate_plo_answer_explanation(
        _facts(), ["Fold", "Call"], "Call",
        client=client, examples=(), system_prompt=custom,
    )
    assert client.messages.systems == [custom]


def test_no_override_uses_built_in_system_prompt():
    client = _CapturingClient(_json("Call. It plays well in position."))
    generate_plo_answer_explanation(
        _facts(), ["Fold", "Call"], "Call", client=client, examples=(),
    )
    assert client.messages.systems == [build_plo_system_prompt(examples=())]


# --- card-fabrication audit (the wrong-suit guard) -------------------------
def test_fabricated_card_triggers_a_retry():
    # Hand is A♠ K♠ A♥ K♥; the first attempt invents a K♦ (wrong suit).
    client = _MockClient(
        _json("Call. Your K♦ blocks his aces and the shape plays well."),
        _json("Call. Your kings block his aces and the shape plays well."),
    )
    result = generate_plo_answer_explanation(
        _facts(), ["Fold", "Call"], "Call", client=client, examples=()
    )
    assert "♦" not in result.answer_explanation  # the invented suit is gone
    assert "kings" in result.answer_explanation.lower()


def test_fabricated_card_after_all_retries_raises():
    client = _MockClient(
        _json("Call. Your K♦ is great."),  # K♦ not in hand
        _json("Call. Your Q♣ is great."),  # Q♣ also not in hand
    )
    with pytest.raises(ExplanationValidationError, match="not in your hand"):
        generate_plo_answer_explanation(
            _facts(), ["Fold", "Call"], "Call", client=client, examples=(), max_retries=1
        )


def test_real_held_card_mention_passes():
    # A♠ IS in the hand -> not flagged.
    client = _MockClient(_json("Call. Your A♠ is the nut blocker here."))
    result = generate_plo_answer_explanation(
        _facts(), ["Fold", "Call"], "Call", client=client, examples=()
    )
    assert "A♠" in result.answer_explanation


# --- include_skills (the Compare A/B seam) ---------------------------------
def test_include_skills_adds_field_only_when_on():
    f = _facts()
    assert "skills_this_spot_tests" not in build_solver_data(f, ["Fold", "Call"], "Call")
    data = build_solver_data(f, ["Fold", "Call"], "Call", include_skills=True)
    assert isinstance(data.get("skills_this_spot_tests"), list)
    assert data["skills_this_spot_tests"]  # non-empty on a real spot


def test_villain_action_is_a_bb_size_not_a_percent():
    # _facts() villain is the LJ opener -> a clear 'opens to 3.5bb', never the
    # internal 'Raise 100%' which reads like a frequency.
    data = build_solver_data(_facts(), ["Fold", "Call"], "Call")
    action = data["villain"]["action"]
    assert "100%" not in action
    assert action == "opens to 3.5bb"


def test_extract_text_skips_thinking_blocks():
    # Fable 5 / adaptive thinking: the response starts with thinking block(s)
    # (no .text attribute) before the text block. _extract_text must scan,
    # not take content[0].
    from types import SimpleNamespace

    from pipeline.plo.explanation_generator import _extract_text

    thinking = SimpleNamespace(type="thinking", thinking="...")
    text = SimpleNamespace(type="text", text='{"answer_explanation": "Call."}')
    response = SimpleNamespace(content=[thinking, text])
    assert _extract_text(response) == '{"answer_explanation": "Call."}'


def test_solver_data_carries_card_redundancy_for_trips_only():
    # AAA9 (the reported miscount hand) -> the deterministic fact is in the
    # data block; a normal AAKK hand -> the key is absent entirely.
    facts = _facts()
    data = build_solver_data(facts, ["Fold", "Call"], "Fold")
    assert "card_redundancy" not in data  # AKAK: one pair per rank only

    import dataclasses

    trips_spot = dataclasses.replace(facts.spot, hero_cards=("9c", "Ad", "Ah", "As"))
    trips_facts = dataclasses.replace(facts, spot=trips_spot)
    data = build_solver_data(trips_facts, ["Fold", "Call"], "Fold")
    assert "ONE of the three is redundant" in data["card_redundancy"]


# --- ev_note (pure spots only; mixed spots carry NO EV number at all) -------
def _facts_with(freqs, evs, history=None):
    node = PloDecisionNode(
        actor="HJ",
        history_before=history
        if history is not None
        else (PloAction("LJ", PloActionType.RAISE, 100),),
        actions=(),
        history_stem="40100",
    )
    spot = PloSpot(
        node=node, hero_index=0, hero_label="x", hero_cards=CARDS,
        action_frequencies=freqs, ev_by_action=evs, presence=1.0,
    )
    return PloFacts(spot=spot, hand_class=classify_plo_hand(CARDS), archetype="")


def test_mixed_spot_carries_no_ev_at_all():
    # Even a 99/1 spot: a genuine mix means the actions are ~equal in EV, so
    # neither the raw gap nor an ev_note belongs in the block.
    data = build_solver_data(
        _facts_with({"Call": 0.99, "Fold": 0.01}, {"Call": 2.0, "Fold": 1.0}),
        ["Fold", "Call"],
        "Call",
    )
    assert "ev_gap_bb" not in data
    assert "ev_note" not in data


def test_pure_spot_with_three_actions_gets_best_alternative_note():
    # Pure call; alternatives are a 3-bet (best alt) and a fold. Loss =
    # (2.0 - 1.0) sb / 2 = 0.5bb.
    data = build_solver_data(
        _facts_with(
            {"Call": 1.0, "Raise 100%": 0.0, "Fold": 0.0},
            {"Call": 2.0, "Raise 100%": 1.0, "Fold": -7.0},
        ),
        ["Fold", "Call", "3-bet"],
        "Call",
    )
    assert data["ev_note"] == (
        "even the best alternative, 3-betting, loses about 0.5bb per hand, "
        "and the other options lose more."
    )


def test_pure_spot_with_two_actions_says_only_alternative():
    # An open node offers just fold/raise -> "the only alternative" is right.
    data = build_solver_data(
        _facts_with(
            {"Raise 100%": 1.0, "Fold": 0.0},
            {"Raise 100%": 3.0, "Fold": 0.0},
            history=(),
        ),
        ["Fold", "Raise"],
        "Raise",
    )
    assert data["ev_note"] == (
        "the only alternative, folding, loses about 1.5bb per hand."
    )


def test_pure_spot_with_tiny_gap_gets_no_note():
    # 0.1sb = 0.05bb gap: stating it would be fake precision.
    data = build_solver_data(
        _facts_with({"Call": 1.0, "Fold": 0.0}, {"Call": 1.0, "Fold": 0.9}),
        ["Fold", "Call"],
        "Call",
    )
    assert "ev_note" not in data


def test_unconverged_noise_gets_no_note():
    # An "alternative" with HIGHER EV than the pure action is solver noise.
    data = build_solver_data(
        _facts_with({"Call": 1.0, "Fold": 0.0}, {"Call": 1.0, "Fold": 5.0}),
        ["Fold", "Call"],
        "Call",
    )
    assert "ev_note" not in data


# --- suit_redundancy in the data block + the shape-claim audit ---------------
def test_solver_data_carries_suit_redundancy_for_three_suited():
    import dataclasses

    facts = _facts()
    spot = dataclasses.replace(facts.spot, hero_cards=("Jc", "Td", "Qd", "Kd"))
    facts3 = dataclasses.replace(
        facts, spot=spot, hand_class=classify_plo_hand(("Jc", "Td", "Qd", "Kd"))
    )
    data = build_solver_data(facts3, ["Fold", "Call"], "Fold")
    assert "third diamond is a dead card" in data["suit_redundancy"]
    # The double-suited fixture hand carries no such key.
    assert "suit_redundancy" not in build_solver_data(_facts(), ["Fold", "Call"], "Fold")


def test_shape_claim_audit_flags_invented_dangler():
    from pipeline.plo.explanation_generator import _shape_claim_errors

    hand = classify_plo_hand(("Jc", "Td", "Qd", "Kd"))  # rundown, NO dangler
    assert _shape_claim_errors(
        "Fold. The K on top is largely a dangler because it overlaps what "
        "you already cover.",
        hand,
    ) == ["dangler"]
    # Negated mention is fine.
    assert _shape_claim_errors("Fold. Your hand has no dangler.", hand) == []
    # True claims are fine.
    assert _shape_claim_errors("Fold. Your three-suited hand is weak.", hand) == []


def test_shape_claim_audit_ignores_villain_sentences():
    from pipeline.plo.explanation_generator import _shape_claim_errors

    rainbow = classify_plo_hand(("As", "Kh", "Qd", "Jc"))
    # Villain-range prose may say double-suited freely.
    assert _shape_claim_errors(
        "Fold. His range is full of double-suited hands and big pairs.",
        rainbow,
    ) == []
    # But a hero claim of double-suited on a rainbow hand is flagged.
    assert _shape_claim_errors(
        "Call. Your double-suited shape plays well here.", rainbow
    ) == ["double-suited"]


def test_nut_flush_claim_without_suited_ace_is_flagged():
    from pipeline.plo.explanation_generator import _shape_claim_errors

    no_ace = classify_plo_hand(("Kd", "Qd", "Jc", "Tc"))  # no suited ace
    assert _shape_claim_errors(
        "Call. You have the nut flush draw to fall back on.", no_ace
    ) == ["nut-flush claim without a suited ace"]
    # Blocker prose is about removal, not making it -- the claim regex is
    # verb-anchored, so "blocks the nut flush" never matches even in a pure
    # hero sentence with no villain reference.
    assert _shape_claim_errors(
        "4-bet. Your bare ace blocks the nut flush and adds fold equity.",
        classify_plo_hand(("Ah", "Kd", "Qc", "Js")),
    ) == []
    # Negation is fine, and a real suited ace may claim it.
    assert _shape_claim_errors("Fold. You do not have the nut flush draw.", no_ace) == []
    suited_ace = classify_plo_hand(("As", "Ks", "Qd", "Jd"))
    assert _shape_claim_errors(
        "Call. Your nut flush potential carries the hand.", suited_ace
    ) == []


def test_made_flush_preflop_tense_is_flagged():
    from pipeline.plo.explanation_generator import _shape_claim_errors

    ds = classify_plo_hand(("7c", "Tc", "9d", "Jd"))
    # The reported prose: a flush stated as already made.
    assert _shape_claim_errors(
        "Call. The diamonds give you a real flush, while the clubs are a "
        "small caution.",
        ds,
    ) == ["made flush stated preflop"]
    # Draw / potential phrasing stays legal.
    assert _shape_claim_errors(
        "Call. The diamonds give you a J-high flush draw and position helps.",
        ds,
    ) == []
    assert _shape_claim_errors(
        "Call. You can make at best a J-high flush, so play the suits with "
        "care.",
        ds,
    ) == []


def test_invented_dangler_triggers_a_retry():
    # CARDS is double-suited AAKK -- no dangler. First attempt invents one;
    # the clean retry is accepted.
    client = _MockClient(
        _json("Call. Your hand has a dangler that weakens it badly."),
        _json("Call. Your double-suited aces play well in position."),
    )
    result = generate_plo_answer_explanation(
        _facts(), ["Fold", "Call"], "Call", client=client, examples=()
    )
    assert "dangler" not in result.answer_explanation
