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
    # A semicolon is a banned phrase -> the rewrite fails re-validation and is
    # discarded, so the ORIGINAL explanation is kept.
    bad = "The best play is to raise; AKo is strong."
    client = _mock_client([json.dumps({"answer_explanation": bad})])
    original = _gen()
    res = revise_explanation(original, _facts(), issues=["x -- y"], client=client)
    assert not res.changed
    assert res.explanation.answer_explanation == original.answer_explanation
    assert res.rejected_reason  # records why the rewrite was thrown out


def test_revise_noop_when_rewrite_is_identical() -> None:
    same = _gen().answer_explanation
    client = _mock_client([json.dumps({"answer_explanation": same})])
    res = revise_explanation(_gen(), _facts(), issues=["x -- y"], client=client)
    assert not res.changed


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
