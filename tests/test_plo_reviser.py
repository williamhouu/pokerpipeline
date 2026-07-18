"""Tests for pipeline.plo.reviser + the batch revise lifecycle (mock client)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.explanation_generator import GeneratedExplanation  # noqa: E402
from pipeline.plo.fact_extractor import PloFacts, PloVillainStats  # noqa: E402
from pipeline.plo.hand_model import classify_plo_hand  # noqa: E402
from pipeline.plo.node_enumerator import PloDecisionNode  # noqa: E402
from pipeline.plo.pack import PloAction, PloActionType, PloPack  # noqa: E402
from pipeline.plo.reviser import (  # noqa: E402
    _REVISER_INSTRUCTION,
    revise_plo_explanation,
)
from pipeline.plo.spot_sampler import PloSpot  # noqa: E402

CARDS = ("As", "Ks", "Ah", "Kh")


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [type("C", (), {"text": text})()]
        self.usage = None


class _Messages:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls: list[dict] = []

    def create(self, **kw: object) -> _Resp:
        self.calls.append(kw)
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
        villain_stats=PloVillainStats(
            seat="LJ", action_label="Raise 100%",
            weighted_combo_count=1.0, pct_of_dealt_hands=18.0,
        ),
    )


def _gen(prose: str) -> GeneratedExplanation:
    return GeneratedExplanation(
        option_1="Always Call",
        option_2="Mostly Call",
        option_3="Mostly Fold",
        option_4="Always Fold",
        correct_answer="Mostly Call",
        answer_explanation=prose,
    )


ORIGINAL = (
    "Calling is the main play, though you will be out of position for the "
    "rest of the hand."
)
FIXED = (
    "Calling is the main play, and you act after the opener on every street."
)
ISSUES = ["you will be out of position -- hero is in position vs the LJ"]


# --- the instruction (pinned, like the NLHE reviser) --------------------------
def test_reviser_instruction_has_minimal_edit_rules():
    """The July-2026 minimal-edit mandate ships in the PLO reviser from day
    one: verbatim-except-flagged, never re-derive unflagged content, never
    introduce a new claim."""
    assert "MINIMAL EDIT" in _REVISER_INSTRUCTION
    assert "VERBATIM" in _REVISER_INSTRUCTION
    assert "Never re-derive" in _REVISER_INSTRUCTION
    assert "Never INTRODUCE a new claim" in _REVISER_INSTRUCTION
    # And it may only emit the prose -- options/answer are solver-locked.
    assert "answer_explanation" in _REVISER_INSTRUCTION


# --- happy path ----------------------------------------------------------------
def test_validated_rewrite_ships_and_preserves_options():
    client = _MockClient(_json(FIXED))
    result = revise_plo_explanation(
        _gen(ORIGINAL), _facts(), issues=ISSUES, client=client
    )
    assert result.changed
    assert result.explanation.answer_explanation == FIXED
    # Options + correct answer are re-attached verbatim, never model-authored.
    assert result.explanation.correct_answer == "Mostly Call"
    assert result.explanation.options() == [
        "Always Call", "Mostly Call", "Mostly Fold", "Always Fold",
    ]


def test_no_issues_or_no_client_is_a_noop():
    untouched = _gen(ORIGINAL)
    assert not revise_plo_explanation(
        untouched, _facts(), issues=[], client=_MockClient()
    ).changed
    assert not revise_plo_explanation(
        untouched, _facts(), issues=ISSUES, client=None
    ).changed


def test_unchanged_rewrite_reports_not_changed():
    client = _MockClient(_json(ORIGINAL))
    result = revise_plo_explanation(
        _gen(ORIGINAL), _facts(), issues=ISSUES, client=client
    )
    assert not result.changed
    assert result.rejected_reason == ""


# --- the deterministic floor + corrective retry ---------------------------------
def test_invalid_rewrite_gets_one_corrective_retry_then_ships():
    """A rewrite that breaks a hard rule (list formatting) is fed back with
    the exact validator error for ONE more attempt -- the second, clean
    rewrite ships."""
    bad = "Call. Here's why:\n- you dominate both suits\n- the price is good"
    client = _MockClient(_json(bad), _json(FIXED))
    result = revise_plo_explanation(
        _gen(ORIGINAL), _facts(), issues=ISSUES, client=client
    )
    assert result.changed
    assert result.explanation.answer_explanation == FIXED
    assert len(client.messages.calls) == 2
    retry_user = client.messages.calls[1]["messages"][0]["content"]
    assert "WAS REJECTED" in retry_user
    assert "list" in retry_user  # the exact rule it broke is fed back


def test_two_invalid_rewrites_keep_the_original():
    bad1 = "Call. Here's why:\n- suits\n- price"
    bad2 = "Your Q♠ dominates, so call."  # fabricated card
    original = _gen(ORIGINAL)
    client = _MockClient(_json(bad1), _json(bad2))
    result = revise_plo_explanation(
        original, _facts(), issues=ISSUES, client=client
    )
    assert not result.changed
    assert result.explanation.answer_explanation == ORIGINAL
    assert result.rejected_reason  # the LAST rejection is recorded
    assert result.revised_text == bad2
    assert len(client.messages.calls) == 2  # bounded, never a third attempt


def test_call_failure_keeps_the_original():
    class _Boom:
        class messages:  # noqa: N801
            @staticmethod
            def create(**_kw: object) -> None:
                raise OSError("api down")

    result = revise_plo_explanation(
        _gen(ORIGINAL), _facts(), issues=ISSUES, client=_Boom()
    )
    assert not result.changed
    assert "revision call failed" in result.rejected_reason


def test_usage_callback_gets_the_five_arg_convention():
    seen: list[tuple] = []
    client = _MockClient(_json(FIXED))
    revise_plo_explanation(
        _gen(ORIGINAL), _facts(), issues=ISSUES, client=client,
        usage_callback=lambda *a: seen.append(a),
    )
    assert len(seen) == 1
    model, in_t, out_t, cc, cr = seen[0]
    assert isinstance(model, str)
    assert all(isinstance(v, int) for v in (in_t, out_t, cc, cr))


# --- batch lifecycle (gate best-of-2 -> revise -> final audit) -------------------
def _write_rng(path: Path, p: float) -> None:
    from pipeline.plo.hand_order import HAND_COUNT

    out = []
    for _ in range(HAND_COUNT):
        out.append("x")
        out.append(f"{p};0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _clean_hj_pack(tmp_path: Path) -> PloPack:
    root = tmp_path / "pack"
    # The LJ open range file too, so extract_plo_facts resolves villain_stats
    # (seat LJ) and the position facts/validators see hero HJ as In Position.
    # (The LJ node itself is never question-worthy: pure 100% frequency.)
    _write_rng(root / "40100.rng", 1.0)
    _write_rng(root / "40100.0.rng", 0.0)
    _write_rng(root / "40100.1.rng", 0.7)
    _write_rng(root / "40100.40100.rng", 0.3)
    return PloPack(root=root, label="test")


_GATE_FLAG = json.dumps({"issues": [
    {"claim": "you will be out of position", "problem": "hero is in position"},
]})
_GATE_CLEAN = json.dumps({"issues": []})


def test_batch_revise_lifecycle_fixes_and_records(tmp_path):
    """End-to-end through generate_plo_batch with revise_pass + final_audit:
    gate best-of-2 flags -> reviser rewrites -> final audit clean. The CSV
    ships the rewrite; the meta records the full lifecycle + counters."""
    from pipeline.plo.batch import generate_plo_batch

    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    # Call order: 1 generate, 2+3 gate best-of-2, 4 reviser, 5 final audit.
    client = _MockClient(
        _json(ORIGINAL), _GATE_FLAG, _GATE_FLAG, _json(FIXED), _GATE_CLEAN,
    )
    result = generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False,
        generate_explanations=True, explanation_client=client,
        run_claim_checker=False, revise_pass=True, final_audit=True,
    )
    assert result.explanations_written == 1
    assert len(client.messages.calls) == 5

    import csv as _csv

    with out.open(encoding="utf-8-sig") as handle:
        row = next(iter(_csv.DictReader(handle)))
    assert row["Answer Explanation"] == FIXED  # the rewrite shipped
    assert row["claim_check"] == "[]"          # final audit ran, clean
    assert "validation_status" not in row      # dropped from the CSV (July 16)

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    # The lifecycle status lives in the meta record now.
    assert meta["questions"][0]["validation_status"] == "draft"
    assert meta["run_settings"]["revise_pass"] is True
    assert meta["run_settings"]["final_audit"] is True
    counters = meta["counters"]
    assert counters["revise_flagged"] == 1
    assert counters["revise_fixed"] == 1
    assert counters["revise_discarded"] == 0
    record = meta["questions"][0]
    assert record["revise"]["status"] == "fixed"
    assert record["revise"]["original_explanation"] == ORIGINAL
    assert record["revise"]["revised_explanation"] == FIXED
    assert record["revise"]["final_audit_issues"] == []


def test_batch_revise_clean_gate_ships_original(tmp_path):
    from pipeline.plo.batch import generate_plo_batch

    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    # 1 generate, 2+3 gate best-of-2 both clean -> no reviser call.
    client = _MockClient(_json(FIXED), _GATE_CLEAN, _GATE_CLEAN)
    generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False,
        generate_explanations=True, explanation_client=client,
        revise_pass=True, final_audit=True,
    )
    assert len(client.messages.calls) == 3
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["counters"]["revise_flagged"] == 0
    assert meta["questions"][0]["revise"]["status"] == "clean"


def test_batch_gate_unions_the_two_passes(tmp_path):
    """A flag seen by EITHER gate pass reaches the reviser (the best-of-2
    union) -- a flaky single-pass miss can't hand out a lucky clean."""
    from pipeline.plo.batch import generate_plo_batch

    pack = _clean_hj_pack(tmp_path)
    out = tmp_path / "batch.csv"
    # Pass 1 misses (clean), pass 2 flags -> still revised.
    client = _MockClient(
        _json(ORIGINAL), _GATE_CLEAN, _GATE_FLAG, _json(FIXED), _GATE_CLEAN,
    )
    generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False,
        generate_explanations=True, explanation_client=client,
        revise_pass=True, final_audit=True,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["counters"]["revise_fixed"] == 1


def test_batch_soft_validator_flags_position_reversal(tmp_path):
    """A shipped explanation with a hero-bound position reversal gets
    validation_status=flagged + a validator_warnings meta record, even with
    Layer-7 off entirely (the soft validators are always on)."""
    from pipeline.plo.batch import generate_plo_batch

    pack = _clean_hj_pack(tmp_path)  # hero HJ vs LJ open -> hero In Position
    out = tmp_path / "batch.csv"
    client = _MockClient(_json(ORIGINAL))  # ORIGINAL claims hero is OOP
    generate_plo_batch(
        pack, output_path=out, total_questions=1, seed=0, compute_equity=False,
        generate_explanations=True, explanation_client=client,
    )
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["questions"][0]["validation_status"] == "flagged"
    assert meta["counters"]["soft_flagged_rows"] == 1
    warnings = meta["questions"][0]["validator_warnings"]
    assert warnings and "In Position" in warnings[0]
