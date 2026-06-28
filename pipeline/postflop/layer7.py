"""Shared Layer-7 audit/revise pass for ONE postflop explanation.

Extracted verbatim from the per-spot loop in :mod:`pipeline.postflop.batch` so
both the independent-spot driver AND the full-hand (play-through) driver run the
*identical* audit -- a full-hand leg gets the same QA as a standalone spot.

Two opt-in flows share one "gate" claim-check call:

* ``run_claim_checker`` (no revise): the gate FLAGS suspect prose (flag only).
* ``revise_pass``: if the gate flags, a 3rd call REWRITES the prose, re-validated
  by the hard validators (a rewrite that breaks a rule is DISCARDED, the original
  kept). ``final_audit`` is a 4th call that claim-checks the kept rewrite.

The gate runs best-of-N (``_REVISE_GATE_PASSES``, union of issues) under the
revise flow so a flaky single pass can't let a real issue slip; the flag-only
path stays one call. Returns a :class:`Layer7Outcome` carrying the (possibly
revised) explanation, the ``claim_check`` JSON for the CSV column, the issues
that REMAIN on the shipped prose (they drive the ``flagged`` status), the
``revise`` record for the Review page, and per-question counter deltas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pipeline.explanation_generator import GeneratedExplanation
from pipeline.postflop.claim_checker import (
    ClaimCheckResult,
    ClaimIssue,
    check_postflop_claims,
    claim_check_to_json,
)
from pipeline.postflop.facts import PostflopFacts
from pipeline.postflop.reviser import revise_postflop_explanation

logger = logging.getLogger(__name__)

# The revise gate runs the claim checker this many times and UNIONs the issues.
# The checker is non-deterministic even at temperature 0 (a single pass can miss
# a real issue), so best-of-N makes the gate reliable. Only the (opt-in, paid)
# revise pass pays for the extra calls -- the flag-only path stays one call.
_REVISE_GATE_PASSES = 2


def _safe_claim_check(
    prose: str, solver_data_block: str, client: object, *,
    model: str, system_prompt: str, node_id: str, question: str = "",
) -> ClaimCheckResult | None:
    """Run the claim checker, wrapped so a checker failure never drops a row."""
    try:
        return check_postflop_claims(
            prose, solver_data_block, client,
            question=question, model=model, system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("layer7: claim checker failed for %s: %s", node_id, exc)
        return None


def _gate_check_best_of(
    prose: str, solver_data_block: str, client: object, *,
    model: str, system_prompt: str, node_id: str, passes: int, question: str = "",
) -> ClaimCheckResult | None:
    """Run the claim-check gate ``passes`` times and union the issues (deduped by
    claim text). ``passed`` is False if any pass flagged anything; ``None`` if
    every pass errored."""
    merged: list[ClaimIssue] = []
    seen: set[str] = set()
    any_ran = False
    for _ in range(max(1, passes)):
        cc = _safe_claim_check(
            prose, solver_data_block, client,
            model=model, system_prompt=system_prompt, node_id=node_id,
            question=question,
        )
        if cc is None:
            continue
        any_ran = True
        for issue in cc.issues:
            key = issue.claim.strip().lower()
            if key not in seen:
                seen.add(key)
                merged.append(issue)
    if not any_ran:
        return None
    return ClaimCheckResult(passed=not merged, issues=tuple(merged))


@dataclass
class Layer7Outcome:
    """Result of the Layer-7 pass for one explanation."""

    explanation: GeneratedExplanation  # possibly the revised one
    claim_check_json: str = ""
    claim_issues: list[str] = field(default_factory=list)
    final_audit_issues: list[str] = field(default_factory=list)
    revise_record: dict | None = None
    remaining_issues: list[str] = field(default_factory=list)
    # per-question counter deltas (0 unless the relevant flow flagged/fixed).
    claim_flagged: int = 0
    revise_flagged: int = 0
    revise_fixed: int = 0
    revise_discarded: int = 0
    revise_unchanged: int = 0


def run_layer7_audit(
    explanation: GeneratedExplanation,
    facts: PostflopFacts,
    *,
    solver_data_block: str,
    question_text: str,
    node_id: str,
    client: object,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None,
    checker_prompt: str,
    run_claim_checker: bool,
    revise_pass: bool,
    final_audit: bool,
    usage_callback=None,
) -> Layer7Outcome:
    """Run the opt-in Layer-7 audit/revise pass on one explanation.

    Mirrors the preflop lifecycle. The caller has already produced ``explanation``
    via the real LLM (never a placeholder). Returns a :class:`Layer7Outcome`;
    ``remaining_issues`` are the issues still present on the shipped prose (they
    set the ``flagged`` status upstream).
    """
    out = Layer7Outcome(explanation=explanation)
    if not (run_claim_checker or revise_pass):
        return out

    cc = (
        _gate_check_best_of(
            explanation.answer_explanation, solver_data_block, client,
            model=model, system_prompt=checker_prompt, node_id=node_id,
            passes=_REVISE_GATE_PASSES, question=question_text,
        )
        if revise_pass
        else _safe_claim_check(
            explanation.answer_explanation, solver_data_block, client,
            model=model, system_prompt=checker_prompt, node_id=node_id,
            question=question_text,
        )
    )
    gate_issues = [f"{i.claim} -- {i.problem}" for i in cc.issues] if cc is not None else []

    if revise_pass:
        if not gate_issues:
            out.revise_record = {"status": "clean", "gate_issues": []}
        else:
            out.revise_flagged = 1
            original_prose = explanation.answer_explanation
            try:
                rev = revise_postflop_explanation(
                    explanation, facts, issues=gate_issues,
                    client=client, model=model, temperature=temperature,
                    max_tokens=max_tokens, question=question_text,
                    system_prompt=system_prompt, usage_callback=usage_callback,
                )
            except Exception as exc:  # noqa: BLE001 - never drop a row
                logger.warning("layer7: reviser failed for %s: %s", node_id, exc)
                rev = None
            if rev is not None and rev.changed:
                out.revise_fixed = 1
                out.explanation = rev.explanation  # ship the rewrite
                out.revise_record = {
                    "status": "fixed",
                    "gate_issues": gate_issues,
                    "original_explanation": original_prose,
                    "revised_explanation": rev.explanation.answer_explanation,
                }
                if final_audit:  # 4th call: claim-check the rewrite
                    cc4 = _safe_claim_check(
                        out.explanation.answer_explanation, solver_data_block,
                        client, model=model, system_prompt=checker_prompt,
                        node_id=node_id, question=question_text,
                    )
                    if cc4 is not None:
                        out.final_audit_issues = [
                            f"{i.claim} -- {i.problem}" for i in cc4.issues
                        ]
                        out.claim_check_json = claim_check_to_json(cc4)
                    out.revise_record["final_audit_issues"] = out.final_audit_issues
            else:
                reason = (
                    getattr(rev, "rejected_reason", "") if rev
                    else "the reviser call failed"
                )
                attempt = getattr(rev, "revised_text", "") if rev else ""
                if reason:
                    out.revise_discarded = 1
                    rstatus = "discarded"
                else:
                    out.revise_unchanged = 1
                    rstatus = "unchanged"
                out.revise_record = {
                    "status": rstatus,
                    "gate_issues": gate_issues,
                    "rejected_reason": reason,
                    "attempted_rewrite": attempt,
                    "original_explanation": original_prose,
                }
                if cc is not None:  # gate findings describe the shipped original
                    out.claim_check_json = claim_check_to_json(cc)
    elif run_claim_checker:
        if cc is not None:
            out.claim_check_json = claim_check_to_json(cc)
        out.claim_issues = gate_issues
        if gate_issues:
            out.claim_flagged = 1

    # Issues that REMAIN on the shipped explanation drive the "flagged" status.
    remaining = list(out.claim_issues) + list(out.final_audit_issues)
    if out.revise_record is not None and out.revise_record["status"] in (
        "discarded", "unchanged",
    ):
        remaining += list(out.revise_record.get("gate_issues") or [])
    out.remaining_issues = remaining
    return out


__all__ = ["Layer7Outcome", "run_layer7_audit"]
