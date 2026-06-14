"""Layer 7 claim checker (the second-pass strategy audit)."""
from __future__ import annotations

from types import SimpleNamespace

from pipeline.preflop.claim_checker import (
    build_checker_user_prompt,
    check_explanation_claims,
    parse_checker_response,
)


def _client(text: str) -> object:
    """Mock Anthropic client whose messages.create returns a raw string
    (_extract_text tolerates str)."""
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: text))


def test_clean_response_passes():
    r = parse_checker_response('{"issues": []}')
    assert r.passed is True
    assert r.issues == ()


def test_issues_parsed_and_fail():
    r = parse_checker_response(
        '{"issues": [{"claim": "AK pays off your set", '
        '"problem": "Sets are paid by overpairs, not unpaired AK."}]}'
    )
    assert r.passed is False
    assert len(r.issues) == 1
    assert r.issues[0].claim == "AK pays off your set"
    assert "overpairs" in r.issues[0].problem


def test_code_fence_tolerated():
    r = parse_checker_response('```json\n{"issues": []}\n```')
    assert r.passed is True


def test_malformed_fails_open():
    # A checker malfunction must never block a good explanation.
    r = parse_checker_response("not json at all")
    assert r.passed is True
    assert r.issues == ()
    assert r.raw == "not json at all"


def test_blank_issue_entries_dropped():
    r = parse_checker_response('{"issues": [{"claim": "", "problem": ""}]}')
    assert r.passed is True  # empty entry is not a real issue


def test_user_prompt_contains_data_and_explanation():
    prompt = build_checker_user_prompt(
        "You should fold.", {"hand_class": "A9s", "archetype": "fold_dominated"}
    )
    assert "ANSWER EXPLANATION:" in prompt and "You should fold." in prompt
    assert "SOLVER DATA:" in prompt and "A9s" in prompt


def test_end_to_end_with_mock_client():
    clean = check_explanation_claims(
        "This is a clear fold.", {"hand_class": "A9s"}, _client('{"issues": []}')
    )
    assert clean.passed is True

    flagged = check_explanation_claims(
        "Your AK pays off your set.",
        {"hand_class": "A9s"},
        _client('{"issues": [{"claim": "AK pays off your set", "problem": "wrong"}]}'),
    )
    assert flagged.passed is False
    assert flagged.issues[0].claim == "AK pays off your set"
