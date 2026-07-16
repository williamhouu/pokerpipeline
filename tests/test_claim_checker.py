"""Layer 7 claim checker (the second-pass strategy audit)."""
from __future__ import annotations

from types import SimpleNamespace

from pipeline.preflop.claim_checker import (
    CHECKER_SYSTEM_PROMPT,
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


def test_checker_prompt_covers_overcertain_conditional_outcomes():
    """The checker must look for conditional/future outcomes stated as certain
    (June 2026: it missed 'you would be playing a 3-bet pot multiway' on a fold
    where HJ was still to act). Pin the rule + the fields it keys on so it
    can't silently regress."""
    p = CHECKER_SYSTEM_PROMPT
    assert "your_call_or_fold_closes_the_action" in p
    assert "still_to_act_after_you" in p
    assert "multiway" in p
    assert "certain" in p.lower()


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


def test_none_client_lazily_builds_from_env(monkeypatch) -> None:
    """The admin panel drives the batch with client=None, so the checker must
    build its own client from ANTHROPIC_API_KEY.

    Regression: it used to call the API with a None client, which threw and
    was swallowed by the batch's try/except -- so run_claim_checker=True
    silently produced an empty claim_check column (no verdict in the UI).
    """
    import anthropic

    created: dict[str, str] = {}

    def _fake_anthropic(*, api_key: str) -> object:
        created["api_key"] = api_key
        return _client('{"issues": []}')

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic)

    result = check_explanation_claims(
        "This is a clear fold.", {"hand_class": "A9s"}, None
    )
    assert result.passed is True
    # It actually constructed a client from the env (rather than dying on None).
    assert created["api_key"] == "test-key-123"


def test_checker_reports_usage_to_callback():
    """Spend-logger rule (July 2026): every LLM call site MUST report usage.
    The checker used to burn tokens invisibly -- generation + reviser were
    counted, the gate / best-of-2 / final-audit calls were not, so audited
    batches logged roughly half their real spend."""
    usage = SimpleNamespace(
        input_tokens=1234, output_tokens=56,
        cache_creation_input_tokens=7, cache_read_input_tokens=8,
    )
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"issues": []}')],
        usage=usage,
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    calls: list[tuple] = []
    check_explanation_claims(
        "prose", {"k": "v"}, client, model="test-model",
        usage_callback=lambda *a: calls.append(a),
    )
    assert calls == [("test-model", 1234, 56, 7, 8)]


def test_sanity_checker_reports_usage_to_callback():
    from pipeline.preflop.sanity_checker import check_solver_data_sanity

    usage = SimpleNamespace(input_tokens=100, output_tokens=10)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"issues": []}')],
        usage=usage,
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    calls: list[tuple] = []
    check_solver_data_sanity(
        {"k": "v"}, client, model="test-model",
        usage_callback=lambda *a: calls.append(a),
    )
    assert calls == [("test-model", 100, 10, 0, 0)]
