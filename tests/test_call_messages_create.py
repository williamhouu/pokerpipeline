"""Tests for pipeline.explanation_generator.call_messages_create.

The wrapper exists because Anthropic deprecated `temperature` on Claude
Opus 4.x -- including it in the API call returns a 400 error and aborts
the batch. The wrapper drops `temperature` for models on the deny list
and passes everything through unchanged for others (Sonnet etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import (  # noqa: E402
    MODELS_WITHOUT_TEMPERATURE,
    call_messages_create,
)


def _record_only_client() -> tuple[SimpleNamespace, list[dict]]:
    """Mock client whose .messages.create just records its kwargs."""
    calls: list[dict] = []

    def create(**kwargs) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return client, calls


def test_temperature_dropped_for_opus_4_7() -> None:
    client, calls = _record_only_client()
    call_messages_create(
        client,
        model="claude-opus-4-7",
        max_tokens=1500,
        temperature=0.3,
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert len(calls) == 1
    assert "temperature" not in calls[0], (
        f"temperature leaked through for Opus: {calls[0]}"
    )
    # Other kwargs still present.
    assert calls[0]["model"] == "claude-opus-4-7"
    assert calls[0]["max_tokens"] == 1500
    assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_temperature_kept_for_sonnet() -> None:
    client, calls = _record_only_client()
    call_messages_create(
        client,
        model="claude-sonnet-4-6",
        max_tokens=1500,
        temperature=0.3,
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.3


def test_temperature_dropped_for_opus_4_5() -> None:
    """Both Opus 4.x ids land in MODELS_WITHOUT_TEMPERATURE."""
    client, calls = _record_only_client()
    call_messages_create(
        client,
        model="claude-opus-4-5",
        max_tokens=1500,
        temperature=0.7,
        system=[],
        messages=[{"role": "user", "content": "x"}],
    )
    assert "temperature" not in calls[0]


def test_temperature_kept_for_unknown_model() -> None:
    """Default behavior for any future model id we haven't seen: assume
    it accepts temperature. The user will see a fast failure if not and
    we add to the deny list."""
    client, calls = _record_only_client()
    call_messages_create(
        client,
        model="claude-future-model-9-2",
        max_tokens=1500,
        temperature=0.5,
        system=[],
        messages=[{"role": "user", "content": "x"}],
    )
    assert calls[0]["temperature"] == 0.5


def test_deny_list_contains_known_opus_versions() -> None:
    """Sanity: the deny list explicitly enumerates the Opus 4.x ids the
    user might pick from the admin panel."""
    assert "claude-opus-4-7" in MODELS_WITHOUT_TEMPERATURE


def test_return_value_passes_through() -> None:
    """The wrapper returns whatever client.messages.create returns,
    unchanged."""
    sentinel = SimpleNamespace(content=[SimpleNamespace(text="payload")])

    def create(**_kwargs) -> SimpleNamespace:
        return sentinel

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    result = call_messages_create(
        client,
        model="claude-sonnet-4-6",
        max_tokens=10,
        temperature=0.1,
        system=[],
        messages=[],
    )
    assert result is sentinel


def test_opus_and_sonnet_get_no_thinking_or_output_config() -> None:
    """Opus/Sonnet calls are unchanged: no thinking/output_config keys (an
    explicit thinking param would change behavior; omitting is valid
    everywhere)."""
    client, calls = _record_only_client()
    for model in ("claude-opus-4-7", "claude-sonnet-4-6"):
        call_messages_create(
            client,
            model=model,
            max_tokens=1500,
            temperature=0.3,
            system=[],
            messages=[{"role": "user", "content": "x"}],
        )
    for kw in calls:
        assert "thinking" not in kw
        assert "output_config" not in kw
        assert kw["max_tokens"] == 1500  # noqa: PLR2004


def test_opus_5_drops_temperature_and_disables_thinking() -> None:
    """Opus 5 (production default July 28 2026): temperature must be dropped
    (400s otherwise) AND thinking must be explicitly disabled -- Opus 5 turns
    thinking ON when the param is omitted, unlike Opus 4.x, which would spend
    billed thinking tokens inside max_tokens and risk truncated prose.
    INVARIANT: generation on Opus 5 behaves like the validated Opus 4.7
    setup (no thinking)."""
    client, calls = _record_only_client()
    call_messages_create(
        client,
        model="claude-opus-5",
        max_tokens=1500,
        temperature=0.3,
        system=[],
        messages=[{"role": "user", "content": "x"}],
    )
    assert "temperature" not in calls[0]
    assert calls[0]["thinking"] == {"type": "disabled"}


def test_opus_4_x_still_omits_thinking_param() -> None:
    """On Opus 4.x, omitting `thinking` already means no thinking -- sending
    an explicit disable is unnecessary; the param must stay absent."""
    client, calls = _record_only_client()
    call_messages_create(
        client,
        model="claude-opus-4-7",
        max_tokens=1500,
        temperature=0.3,
        system=[],
        messages=[{"role": "user", "content": "x"}],
    )
    assert "thinking" not in calls[0]


def test_string_system_is_wrapped_as_cached_block() -> None:
    """PROMPT CACHING (July 2026): a plain-string system prompt is wrapped
    into a single cache-controlled text block, so every call site passing a
    string (postflop generation, claim checkers, revisers, PLO) gets prompt
    caching for free. INVARIANT: the text reaches the API verbatim."""
    client, calls = _record_only_client()
    call_messages_create(
        client,
        model="claude-opus-5",
        max_tokens=700,
        temperature=0.3,
        system="You are the postflop explanation writer.",
        messages=[{"role": "user", "content": "x"}],
    )
    assert calls[0]["system"] == [
        {
            "type": "text",
            "text": "You are the postflop explanation writer.",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_block_list_system_passes_through_unwrapped() -> None:
    """A system that is already a block list (the preflop generator's
    two-breakpoint payload) must pass through UNTOUCHED -- double-wrapping
    would corrupt the payload and add a phantom cache breakpoint."""
    client, calls = _record_only_client()
    payload = [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
    ]
    call_messages_create(
        client,
        model="claude-opus-5",
        max_tokens=700,
        temperature=0.3,
        system=payload,
        messages=[{"role": "user", "content": "x"}],
    )
    assert calls[0]["system"] is payload


def test_default_model_is_opus_5_and_fully_configured() -> None:
    """The production default is Opus 5 and it appears on BOTH compatibility
    lists (temperature dropped, thinking disabled) -- a default model missing
    from either list would 400 or silently burn thinking tokens."""
    from pipeline.explanation_generator import (
        DEFAULT_MODEL,
        MODELS_THINKING_ON_BY_DEFAULT,
    )

    assert DEFAULT_MODEL == "claude-opus-5"
    assert DEFAULT_MODEL in MODELS_WITHOUT_TEMPERATURE
    assert DEFAULT_MODEL in MODELS_THINKING_ON_BY_DEFAULT
