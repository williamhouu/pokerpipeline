"""PLO Layer-7 self-revision: an OPT-IN pass that REWRITES a flagged explanation.

The PLO port of :mod:`pipeline.preflop.reviser` (July 2026), kept
self-contained like every PLO module. Where the claim checker
(:mod:`pipeline.plo.claim_checker`) only FLAGS suspect prose, this takes the
explanation plus the checker's issues and produces a corrected version, then
re-runs the deterministic hard validators on it.

The rewrite is bounded HARD, so an LLM can only polish wording within ground
truth and can never ship a fabrication the validators catch:

* it may rewrite ONLY the ``answer_explanation`` prose;
* ``option_1..option_4`` and ``correct_answer`` are solver-derived and
  re-attached verbatim -- the reviser literally cannot change the recommended
  action or the answer set, because we ignore anything it emits for those;
* the rewrite is re-validated by
  :func:`pipeline.plo.validators.run_plo_audit_validators` (banned phrases,
  list formatting, fabricated cards, hand-shape claims -- the SAME floor the
  generation retry loop enforces). A rewrite that breaks a hard rule gets ONE
  corrective retry with the exact validator error fed back (the pattern
  Layer-6 generation uses); a second failure keeps the original.

Carries the two July-2026 NLHE reviser upgrades from day one: the MINIMAL-EDIT
mandate (verbatim except flagged sentences; never introduce a claim the
original didn't make) and the bounded corrective retry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pipeline.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    GeneratedExplanation,
    call_messages_create,
    prompt_sanctions_lists,
)
from pipeline.plo.explanation_generator import (
    UsageCallback,
    _extract_text,
    _extract_usage,
    _parse,
    build_plo_system_prompt,
    build_solver_data,
)
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.validators import run_plo_audit_validators

logger = logging.getLogger(__name__)

# Total rewrite attempts: the first pass plus ONE corrective retry that feeds
# the hard-validator rejection back. Bounded -- the discard path is rare, so
# the extra call is cheap, but never loop unbounded.
_REVISE_ATTEMPTS = 2

# The editor's mandate. Kept terse; the VOICE RULES it must still obey come
# from the (reused) generation system prompt, so they never drift out of sync.
_REVISER_INSTRUCTION = (
    "You are an editor fixing ONE Pot-Limit Omaha answer explanation you "
    "wrote earlier. An automated audit flagged the problems listed below. "
    "REWRITE the explanation so each valid problem is gone, keeping the "
    "verdict and everything that is already correct.\n"
    "HOW TO USE THE FLAGS:\n"
    "- Treat each flag as a real error and rewrite the offending sentence to "
    "remove it. These audits are usually right: a reversed in-position / "
    "out-of-position claim, a hand-shape word the hand doesn't have, a "
    "backwards equity-or-price framing, an action mix stated wrong.\n"
    "- The SOLVER DATA above is ground truth -- use it to fix correctly. The "
    "position field settles any in-position / out-of-position wording; "
    "`your_hand_shape`, `card_redundancy`, and `suit_redundancy` settle any "
    "shape claim; `action_strategy` settles any mix claim; the equity fields "
    "settle any ahead/behind claim.\n"
    "- Keep a flagged sentence as-is ONLY if the SOLVER DATA shows it is "
    "already correct (a false flag). Do NOT return the explanation unchanged "
    "when any flag is valid -- rewrite the sentences that are wrong. A no-op "
    "response is only acceptable when every flag is demonstrably false.\n"
    "MINIMAL EDIT, NOT A REWRITE:\n"
    "- Reproduce the original explanation VERBATIM and change ONLY the "
    "specific sentences the flags identify. Do not rephrase, reorder, "
    "shorten, expand, or restyle any unflagged sentence -- copying it "
    "character-for-character is the requirement, not a suggestion.\n"
    "- Never re-derive anything in an unflagged sentence: hand shapes, "
    "percentages, and comparisons there are already correct and must "
    "survive untouched. Most rewrite regressions come from 'improving' "
    "text nobody flagged.\n"
    "- Never INTRODUCE a new claim the original did not make (a shape "
    "feature, a suit-dominance read, an equity number): if it was not in "
    "the original and no flag asks for it, it does not belong in the fix. "
    "When fixing a flagged sentence, keep its length and role.\n"
    "HARD CONSTRAINTS:\n"
    "- Do NOT change the recommended action, the verdict, any number, or the "
    "four options. Those are fixed by the solver and given to you above. "
    "Only the wording of the explanation may change.\n"
    "- Obey every VOICE RULE in your instructions (no em dash, no semicolon, "
    "no banned phrases, no bulleted or numbered lists, suit emojis only for "
    "hero's own four cards, second person).\n"
    'Output a single JSON object with exactly one key, "answer_explanation", '
    "whose value is the corrected explanation prose. No other keys, no text "
    "outside the JSON."
)


@dataclass(frozen=True)
class PloReviseResult:
    """Outcome of one revision pass."""

    explanation: GeneratedExplanation  # revised iff it passed re-validation, else the original
    changed: bool                      # True iff a validated rewrite replaced the original
    revised_text: str = ""             # the rewrite (kept even when rejected, for the meta/debug)
    rejected_reason: str = ""          # why a rewrite was discarded (validation failure / parse error)


def _rebuild(original: GeneratedExplanation, prose: str) -> GeneratedExplanation:
    """A copy of ``original`` with ONLY the explanation prose replaced.

    Built field-by-field (not from the model's output) so the options and
    correct_answer are guaranteed identical to the solver-derived ones.
    """
    return GeneratedExplanation(
        option_1=original.option_1,
        option_2=original.option_2,
        option_3=original.option_3,
        option_4=original.option_4,
        correct_answer=original.correct_answer,
        answer_explanation=prose,
    )


def revise_plo_explanation(
    explanation: GeneratedExplanation,
    facts: PloFacts,
    *,
    issues: list[str],
    client: Any,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str | None = None,
    include_skills: bool = False,
    usage_callback: UsageCallback | None = None,
) -> PloReviseResult:
    """One LLM pass that rewrites a flagged explanation, then re-validates it.

    Args:
        explanation: the explanation to fix (its options + correct_answer are
            kept verbatim; only the prose is rewritten).
        facts: the PLO fact block (ground truth + the re-validation input).
        issues: the audit findings to fix (the claim checker's
            ``"<claim> -- <problem>"`` strings). Empty -> no-op.
        client: an Anthropic client. ``None`` -> no-op (returns the original;
            the batch resolves ONE shared client up front, per the
            revise-pass client=None lesson).
        system_prompt: the generation system prompt to reuse, so the reviser
            obeys the same voice rules. ``None`` -> the built-in PLO prompt.
        include_skills: mirrors the generation call, so the SOLVER DATA block
            the reviser sees is the one the draft was written from.

    Returns:
        A :class:`PloReviseResult`. ``changed`` is True only when a rewrite
        both differed from the original AND passed the deterministic hard
        validators; otherwise the original explanation is returned unchanged.
    """
    if client is None or not issues:
        return PloReviseResult(explanation=explanation, changed=False)

    system = (
        system_prompt if system_prompt is not None else build_plo_system_prompt()
    )
    options = [o for o in explanation.options() if o]  # the fixed answer set
    solver_data = build_solver_data(
        facts, options, explanation.correct_answer, include_skills=include_skills
    )
    user = (
        "SOLVER DATA:\n"
        f"{json.dumps(solver_data, indent=2, default=str)}\n\n"
        f"OPTIONS (fixed, do not change): {options}\n"
        f"CORRECT ANSWER (fixed, do not change): {explanation.correct_answer}\n\n"
        f"YOUR EARLIER EXPLANATION:\n{explanation.answer_explanation}\n\n"
        "AUDIT ISSUES TO FIX:\n"
        + "\n".join(f"- {i}" for i in issues)
        + "\n\n"
        + _REVISER_INSTRUCTION
    )

    # CORRECTIVE RETRY: a rewrite that breaks a hard rule is not discarded
    # outright -- the rejection (the exact validator error plus the rejected
    # text) is fed back for ONE more attempt. A second failure keeps the
    # original (flagged, with the last rejection recorded).
    corrective = ""
    last_text = ""
    last_reason = ""
    for _attempt in range(_REVISE_ATTEMPTS):
        try:
            response = call_messages_create(
                client,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user + corrective}],
            )
            if usage_callback is not None:
                # The PLO 5-arg convention (model, in, out, cache_creation,
                # cache_read) -- every LLM call site MUST report usage (the
                # July-2026 spend-logger standing rule).
                usage_callback(model, *_extract_usage(response))
            revised_text = _parse(_extract_text(response))
        except Exception as exc:  # noqa: BLE001 - a reviser hiccup must never drop the row
            logger.warning("plo reviser: call/parse failed: %s", exc)
            return PloReviseResult(
                explanation=explanation, changed=False,
                rejected_reason=f"revision call failed: {exc}",
            )

        if revised_text.strip() == explanation.answer_explanation.strip():
            return PloReviseResult(
                explanation=explanation, changed=False, revised_text=revised_text
            )

        revised = _rebuild(explanation, revised_text)
        # The deterministic floor: a rewrite that breaks a hard rule never
        # ships. Factor-list prompts sanction "- " lines, so the no-list rule
        # is skipped for them (else every faithful rewrite would bounce).
        result = run_plo_audit_validators(
            revised, facts,
            allow_list_formatting=prompt_sanctions_lists(system),
        )
        if result.is_valid:
            return PloReviseResult(
                explanation=revised, changed=True, revised_text=revised_text
            )
        last_text, last_reason = revised_text, result.error_message
        corrective = (
            "\n\nYOUR PREVIOUS REWRITE (below) WAS REJECTED by a deterministic "
            "hard rule and discarded:\n"
            f"{revised_text}\n\n"
            "THE RULE IT BROKE:\n"
            f"{result.error_message}\n\n"
            "Produce a NEW rewrite of the original explanation that fixes the "
            "audit issues WITHOUT breaking this rule. Change only the flagged "
            "sentences and introduce no claim the original did not make."
        )

    return PloReviseResult(
        explanation=explanation,
        changed=False,
        revised_text=last_text,
        rejected_reason=last_reason,
    )


__all__ = ["PloReviseResult", "revise_plo_explanation"]
