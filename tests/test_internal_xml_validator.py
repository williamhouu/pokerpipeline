"""The internal-XML-tag hard validator (all three pipelines, July 28 2026).

Guards the documented Opus 5 thinking-disabled failure mode: the model can
occasionally leak internal tags like ``<thinking>`` into visible prose. Our
production config runs Opus 5 with thinking disabled (see
``test_call_messages_create``), so the leak must be caught deterministically
and fed to generation's corrective retry. Enforced as a validator, not a
prompt rule -- naming thinking tags in prompts is documented to make leakage
worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.explanation_generator import (  # noqa: E402
    GeneratedExplanation,
    find_internal_xml_tag,
)


def _gen(prose: str) -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1="Fold",
        option_2="Call",
        option_3="",
        option_4="",
        correct_answer="Call",
        answer_explanation=prose,
    )


# --- shared detector ---------------------------------------------------------
def test_detector_flags_thinking_tags() -> None:
    assert find_internal_xml_tag("ok <thinking>hmm</thinking> ok") == "<thinking>"
    assert find_internal_xml_tag("</reflection>") == "</reflection>"
    assert find_internal_xml_tag("<internal-note>x") == "<internal-note>"


def test_detector_allows_normal_poker_prose() -> None:
    for prose in (
        "You should call. Your equity is 57% against a 30% price.",
        "pot odds < 30% and equity > price",
        "bet < raise amount > here",
        "A 3-bet to 9bb makes sense with K❤️J❤️ on 8❤️ 6❤️ 5♠️.",
        "",
    ):
        assert find_internal_xml_tag(prose) is None, prose


# --- per-pipeline validators (thin wrappers over the shared detector) --------
def test_postflop_validator_rejects_and_passes() -> None:
    from pipeline.postflop.validators import validate_no_internal_xml

    bad = validate_no_internal_xml(_gen("Call. <thinking>because</thinking>"), None)
    assert not bad.is_valid and "<thinking>" in bad.error_message
    assert validate_no_internal_xml(_gen("Call. The price is 30%."), None).is_valid


def test_preflop_validator_rejects_and_passes() -> None:
    from pipeline.preflop.validators import validate_no_internal_xml

    bad = validate_no_internal_xml(_gen("Fold. <scratchpad>x</scratchpad>"), None)
    assert not bad.is_valid
    assert validate_no_internal_xml(_gen("Fold. Dominated too often."), None).is_valid


def test_plo_validator_rejects_and_passes() -> None:
    from pipeline.plo.validators import validate_no_internal_xml

    bad = validate_no_internal_xml(_gen("Raise. <thinking>pot it</thinking>"), None)
    assert not bad.is_valid
    assert validate_no_internal_xml(_gen("Raise. Double-suited broadways."), None).is_valid


# --- wired into each hard stack (a leak must reject generation) --------------
def test_postflop_stack_includes_the_check() -> None:
    import pipeline.postflop.validators as m

    assert m.validate_no_internal_xml.__name__ in {
        c.__name__
        for c in (
            m.validate_correct_answer,
            m.validate_banned_phrases,
            m.validate_no_list_formatting,
            m.validate_no_internal_xml,
        )
    }
    # The real proof: the runner rejects a leaked tag end-to-end.
    from pipeline.postflop.fixtures import btn_vs_bb_srp_2cJs7s
    from pipeline.postflop.facts import extract_facts
    from pipeline.postflop.spot_sampler import sample_spot

    solve = btn_vs_bb_srp_2cJs7s()
    node = solve.nodes["flop_ip_cbet"]
    facts = extract_facts(sample_spot(node, "AcJc"), solve, equity_runouts=10)
    gen = GeneratedExplanation(
        option_1="Check", option_2="Bet", option_3="", option_4="",
        correct_answer="Bet",
        answer_explanation="Bet for value. <thinking>obvious</thinking>",
    )
    result = m.run_postflop_audit_validators(gen, facts)
    assert not result.is_valid and "internal tag" in result.error_message
