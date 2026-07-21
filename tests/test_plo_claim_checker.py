"""The PLO Layer-7 claim checker (July 2026 port of the NLHE checker).

Flag-only, fails open, opt-in on the batch. These tests pin the parsing
contract, the fail-open rule, and the batch wiring (claim_check column +
validation_status + meta counter) with a mock client -- no API key.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.plo.claim_checker import (
    check_plo_explanation_claims,
    claim_check_to_json,
    parse_checker_response,
    parse_claim_check,
)


class _Messages:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        text = self._texts.pop(0) if self._texts else '{"issues": []}'
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _MockClient:
    def __init__(self, *texts: str) -> None:
        self.messages = _Messages(list(texts))


def test_parse_flags_and_fail_open() -> None:
    flagged = parse_checker_response(
        '{"issues": [{"claim": "you flip vs AKQJ", '
        '"problem": "equity pinned to one named hand, not the range"}]}'
    )
    assert not flagged.passed and flagged.issues[0].claim == "you flip vs AKQJ"
    # Unparseable output = clean pass (fails OPEN), raw kept for debugging.
    garbled = parse_checker_response("Sorry, I cannot")
    assert garbled.passed and garbled.raw == "Sorry, I cannot"
    # Serialization round-trip: "[]" when clean (distinguishable from "").
    assert claim_check_to_json(garbled) == "[]"
    assert parse_claim_check("") == []
    assert parse_claim_check(claim_check_to_json(flagged))[0]["claim"] == (
        "you flip vs AKQJ"
    )


def test_parse_drops_self_retracted_issues() -> None:
    """An entry whose problem text talks itself out of the flag ("...Not a
    real issue.") is checker noise, not a finding (July 2026: one shipped on
    a live MTT batch and flagged a clean row). Genuine findings that merely
    QUOTE soft language ("says the price is fine but...") must survive."""
    retracted = parse_checker_response(
        '{"issues": [{"claim": "UTG had limped in front", '
        '"problem": "misorders the action, but the phrasing is acceptable. '
        'Not a real issue."}]}'
    )
    assert retracted.passed and retracted.issues == ()
    kept = parse_checker_response(
        '{"issues": [{"claim": "the price is fine", '
        '"problem": "says the price is fine but 28% equity is below the '
        '33% break-even"}]}'
    )
    assert not kept.passed and len(kept.issues) == 1


def test_checker_call_binds_data_block_and_prose() -> None:
    client = _MockClient('{"issues": []}')
    res = check_plo_explanation_claims(
        "The best play is to call.",
        {"your_hand": "Ac Ad Kc Kd", "action_strategy": {"Call": "70%"}},
        client,
        model="test-model",
    )
    assert res.passed
    sent = client.messages.calls[0]["messages"][0]["content"]
    assert "SOLVER DATA" in sent and "Ac Ad Kc Kd" in sent
    assert "The best play is to call." in sent


def test_batch_wiring_populates_claim_check_and_flags(tmp_path: Path) -> None:
    """run_claim_checker=True: the checker's verdict lands in the claim_check
    column, a flagged row ships validation_status=flagged, and the meta
    counter counts it. Uses the fixture pack + a mock client that first
    writes the explanation, then returns one flag."""
    from tests.test_plo_batch import _clean_hj_pack

    from pipeline.plo.batch import generate_plo_batch

    pack = _clean_hj_pack(tmp_path)
    client = _MockClient(
        json.dumps({"answer_explanation": "The best play is to call."}),
        '{"issues": [{"claim": "bad claim", "problem": "wrong"}]}',
    )
    out = tmp_path / "b.csv"
    res = generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False,
        generate_explanations=True, explanation_client=client,
        run_claim_checker=True,
    )
    assert res.questions_written == 1
    row = next(csv.DictReader(out.open(encoding="utf-8-sig")))
    issues = parse_claim_check(row["claim_check"])
    assert issues and issues[0]["claim"] == "bad claim"
    meta = json.loads(res.meta_path.read_text())
    # validation_status moved from the CSV to the meta record (July 16).
    assert meta["questions"][0]["validation_status"] == "flagged"
    assert meta["counters"]["claim_flagged_rows"] == 1
    assert meta["questions"][0]["claim_check_issues"][0]["claim"] == "bad claim"
    assert meta["run_settings"]["run_claim_checker"] is True


def test_plo_checker_reports_usage_to_callback() -> None:
    """Spend-logger rule (July 2026): the checker's tokens must reach the
    admin cost readout via the PLO 5-arg usage callback."""
    usage = SimpleNamespace(input_tokens=500, output_tokens=40)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"issues": []}')],
        usage=usage,
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    calls: list[tuple] = []
    check_plo_explanation_claims(
        "prose", {"k": "v"}, client, model="test-model",
        usage_callback=lambda *a: calls.append(a),
    )
    assert calls == [("test-model", 500, 40, 0, 0)]
