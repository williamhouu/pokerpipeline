"""Tests for pipeline.preflop.reviser (the opt-in self-revision pass).

Focus: the revision is bounded -- it rewrites only the prose, keeps the
options + correct_answer verbatim, and a rewrite that fails the deterministic
hard validators is DISCARDED (the original is kept).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import GeneratedExplanation  # noqa: E402
from pipeline.preflop.fact_extractor import PreflopFacts  # noqa: E402
from pipeline.preflop.node_enumerator import PreflopDecisionNode  # noqa: E402
from pipeline.preflop.reviser import revise_explanation  # noqa: E402
from pipeline.preflop.spot_sampler import PreflopSpot  # noqa: E402


def _facts() -> PreflopFacts:
    """A first-to-act open spot (no villain -> minimal validator surface)."""
    spot = PreflopSpot(
        node=PreflopDecisionNode(pack_id="t", actor="BTN", history_before=(), actions=()),
        hero_hand_class="AKo",
        hero_card_combo="AhKc",
        action_frequencies={"Fold": 0.0, "Raise 60%": 1.0},
        dominant_action="Raise 60%",
        dominant_frequency=1.0,
    )
    return PreflopFacts(spot=spot, villain_stats=None, archetype="open_for_value")


def _gen(prose: str = "The best play is to raise. AKo opens from the Button.") -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1="Always raise", option_2="", option_3="", option_4="",
        correct_answer="Always raise", answer_explanation=prose,
    )


def _mock_client(responses: list[str]) -> SimpleNamespace:
    calls: list[dict[str, Any]] = []
    queue = list(responses)

    def create(*, model, max_tokens, system, messages, temperature=None, **_extra):
        calls.append({"system": system, "messages": messages})
        return SimpleNamespace(content=[SimpleNamespace(text=queue.pop(0))])

    return SimpleNamespace(messages=SimpleNamespace(create=create), _calls=calls)


def test_revise_noop_without_client() -> None:
    res = revise_explanation(_gen(), _facts(), issues=["x -- y"], client=None)
    assert not res.changed and res.explanation.answer_explanation == _gen().answer_explanation


def test_revise_noop_without_issues() -> None:
    client = _mock_client([])
    res = revise_explanation(_gen(), _facts(), issues=[], client=client)
    assert not res.changed
    assert client._calls == []  # no API call when there's nothing to fix


def test_revise_applies_clean_rewrite() -> None:
    new = "The best play is to raise. AKo is a premium opening hand from the Button."
    client = _mock_client([json.dumps({"answer_explanation": new})])
    res = revise_explanation(_gen(), _facts(), issues=["wording -- unclear"], client=client)
    assert res.changed
    assert res.explanation.answer_explanation == new
    # options + correct_answer are untouched by the rewrite.
    assert res.explanation.correct_answer == "Always raise"
    assert res.explanation.options() == ["Always raise"]


def test_revise_rejects_rewrite_that_breaks_a_validator() -> None:
    # A semicolon is a banned phrase -> the rewrite fails re-validation. Both
    # attempts fail here (the corrective retry runs once, bounded), so the
    # ORIGINAL explanation is kept -- exactly the pre-retry behavior.
    bad = "The best play is to raise; AKo is strong."
    bad2 = "The best play is to raise; AKo is premium."
    client = _mock_client([
        json.dumps({"answer_explanation": bad}),
        json.dumps({"answer_explanation": bad2}),
    ])
    original = _gen()
    res = revise_explanation(original, _facts(), issues=["x -- y"], client=client)
    assert not res.changed
    assert res.explanation.answer_explanation == original.answer_explanation
    assert res.rejected_reason  # records why the rewrite was thrown out
    assert len(client._calls) == 2  # exactly ONE corrective retry, no loop
    # The retry was CORRECTIVE: the rejected text + broken rule were fed back.
    second_user = client._calls[1]["messages"][0]["content"]
    assert "WAS REJECTED" in second_user and bad in second_user


def test_revise_corrective_retry_recovers_after_hard_reject() -> None:
    """July 2026 upgrade (user-requested): a rewrite that breaks a hard rule
    gets ONE corrective retry with the exact validator error fed back,
    instead of shipping the flagged original unfixed."""
    bad = "The best play is to raise; AKo is strong."
    good = "The best play is to raise. AKo is a premium opening hand."
    client = _mock_client([
        json.dumps({"answer_explanation": bad}),
        json.dumps({"answer_explanation": good}),
    ])
    res = revise_explanation(_gen(), _facts(), issues=["x -- y"], client=client)
    assert res.changed
    assert res.explanation.answer_explanation == good
    assert len(client._calls) == 2


def test_reviser_instruction_has_minimal_edit_rules() -> None:
    """The July 2026 port of the postflop MINIMAL-EDIT rules: the screenshot
    failure (a rewrite INVENTING a blocker claim on an open spot) came from
    the preflop reviser re-deriving unflagged content. Pin the rules so they
    can't silently drop out of the instruction."""
    from pipeline.preflop.reviser import _REVISER_INSTRUCTION

    p = _REVISER_INSTRUCTION
    assert "MINIMAL EDIT" in p
    assert "VERBATIM" in p
    assert "re-derive" in p
    assert "INTRODUCE" in p  # never add a claim the original did not make


def test_revise_noop_when_rewrite_is_identical() -> None:
    same = _gen().answer_explanation
    client = _mock_client([json.dumps({"answer_explanation": same})])
    res = revise_explanation(_gen(), _facts(), issues=["x -- y"], client=client)
    assert not res.changed


def test_revise_reports_usage_with_correct_arg_convention() -> None:
    """Regression: the reviser must call usage_callback with the generator's
    5-arg convention (model, input, output, cache_creation, cache_read). It
    previously called ``usage_callback(response.usage)`` -- one arg -- which
    raised a TypeError on every REAL response (the test mocks had no ``.usage``
    so this went unnoticed), and the broad ``except`` swallowed it, silently
    failing EVERY rewrite. The mock here carries ``.usage`` like the SDK, and
    the strict 5-arg callback would crash under the old one-arg call."""
    new = "The best play is to raise. AKo is a premium Button open."

    def create(*, model, max_tokens, system, messages, temperature=None, **_extra):
        return SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps({"answer_explanation": new}))],
            usage=SimpleNamespace(
                input_tokens=11, output_tokens=22,
                cache_creation_input_tokens=3, cache_read_input_tokens=4,
            ),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    seen: list[tuple] = []

    def usage_cb(model, input_tokens, output_tokens, cache_creation, cache_read):
        seen.append((model, input_tokens, output_tokens, cache_creation, cache_read))

    res = revise_explanation(
        _gen(), _facts(), issues=["wording -- unclear"], client=client,
        model="claude-x", usage_callback=usage_cb,
    )
    assert res.changed  # the rewrite did NOT crash on the usage callback
    assert res.explanation.answer_explanation == new
    assert seen == [("claude-x", 11, 22, 3, 4)]


def test_revise_ignores_options_the_model_emits() -> None:
    # Even if the model tries to emit different options/correct_answer, we only
    # read answer_explanation and re-attach the originals.
    payload = (
        '{"option_1": "Always fold", "correct_answer": "Always fold", '
        '"answer_explanation": "The best play is to raise. AKo opens for value."}'
    )
    client = _mock_client([payload])
    res = revise_explanation(_gen(), _facts(), issues=["x -- y"], client=client)
    assert res.explanation.correct_answer == "Always raise"  # NOT "Always fold"
    assert res.explanation.option_1 == "Always raise"
