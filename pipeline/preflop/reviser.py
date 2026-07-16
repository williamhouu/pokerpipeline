"""Layer-7 self-revision: an OPT-IN pass that REWRITES a flagged explanation.

Where the claim checker (:mod:`pipeline.preflop.claim_checker`) only FLAGS
suspect prose, this takes the explanation plus the checker's issues and produces
a corrected version, then re-runs the deterministic hard validators on it.

The rewrite is bounded HARD, so an LLM can only polish wording within ground
truth and can never ship a fabrication the validators catch:

* it may rewrite ONLY the ``answer_explanation`` prose;
* ``option_1..option_4`` and ``correct_answer`` are solver-derived and re-attached
  verbatim -- the reviser literally cannot change the recommended action or the
  answer set, because we ignore anything it emits for those fields;
* the rewrite is re-validated by :func:`run_preflop_audit_validators`. If it
  breaks a hard rule (a banned phrase, an invented blocker, an equity-vs-price
  reversal, ...), the rewrite is DISCARDED and the original (which already
  passed) is kept.

This is the brief's "biggest risk" (LLM judgement creeping back in) deliberately
relaxed for PROSE ONLY and backstopped by the deterministic floor. It costs one
extra LLM call, so it is opt-in (a batch flag), like the claim checker. The
batch keeps BOTH versions in ``meta.json`` so a reviewer can see what changed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pipeline.explanation_generator import (
    GeneratedExplanation,
    _extract_text,
    call_messages_create,
)
from pipeline.preflop.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    UsageCallback,
    _extract_usage,
    _parse_explanation_only_response,
    _trim_facts_for_prompt,
    load_preflop_system_prompt,
)
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.validators import run_preflop_audit_validators

logger = logging.getLogger(__name__)

# Total rewrite attempts: the first pass plus ONE corrective retry that feeds
# the hard-validator rejection back (July 2026). Bounded -- the discard path
# is rare, so the extra call is cheap, but never loop unbounded.
_REVISE_ATTEMPTS = 2

# The editor's mandate. Kept terse; the VOICE RULES it must still obey come from
# the (reused) generation system prompt, so they never drift out of sync.
_REVISER_INSTRUCTION = (
    "You are an editor fixing ONE preflop answer explanation you wrote earlier. "
    "An automated audit flagged the problems listed below. REWRITE the "
    "explanation so each valid problem is gone, keeping the verdict and "
    "everything that is already correct.\n"
    "HOW TO USE THE FLAGS:\n"
    "- Treat each flag as a real error and rewrite the offending sentence to "
    "remove it. These audits are usually right: a reversed position, a "
    "miscounted blocker, a backwards equity-or-price claim.\n"
    "- The SOLVER DATA above is ground truth -- use it to fix correctly. The "
    "`hero_position` field settles any in-position / out-of-position wording; "
    "the blocker counts settle any 'you block X' claim; the equities settle any "
    "ahead/behind claim.\n"
    "- Keep a flagged sentence as-is ONLY if the SOLVER DATA shows it is already "
    "correct (a false flag). Do NOT return the explanation unchanged when any "
    "flag is valid -- rewrite the sentences that are wrong. A no-op response is "
    "only acceptable when every flag is demonstrably false.\n"
    "MINIMAL EDIT, NOT A REWRITE:\n"
    "- Reproduce the original explanation VERBATIM and change ONLY the "
    "specific sentences the flags identify. Do not rephrase, reorder, "
    "shorten, expand, or restyle any unflagged sentence -- copying it "
    "character-for-character is the requirement, not a suggestion.\n"
    "- Never re-derive anything in an unflagged sentence: hand names, "
    "percentages, and comparisons there are already correct and must "
    "survive untouched. Most rewrite regressions come from 'improving' "
    "text nobody flagged.\n"
    "- Never INTRODUCE a new claim the original did not make (a blocker "
    "effect, a domination read, an equity number): if it was not in the "
    "original and no flag asks for it, it does not belong in the fix. When "
    "fixing a flagged sentence, keep its length and role.\n"
    "HARD CONSTRAINTS:\n"
    "- Do NOT change the recommended action, the verdict, any number, or the "
    "four options. Those are fixed by the solver and given to you above. Only "
    "the wording of the explanation may change.\n"
    "- Obey every VOICE RULE in your instructions (no em dash, no semicolon, no "
    "banned phrases, suit emojis for specific cards, second person, the "
    "archetype frame, blocker/position/equity rules).\n"
    'Output a single JSON object with exactly one key, "answer_explanation", '
    "whose value is the corrected explanation prose. No other keys, no text "
    "outside the JSON."
)


@dataclass(frozen=True)
class ReviseResult:
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


def revise_explanation(
    explanation: GeneratedExplanation,
    facts: PreflopFacts,
    *,
    issues: list[str],
    client: Any,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str | None = None,
    usage_callback: UsageCallback | None = None,
) -> ReviseResult:
    """One LLM pass that rewrites a flagged explanation, then re-validates it.

    Args:
        explanation: the explanation to fix (its options + correct_answer are
            kept verbatim; only the prose is rewritten).
        facts: the Layer-5 data block (ground truth + the re-validation input).
        issues: the audit findings to fix (e.g. the claim checker's
            ``"<claim> -- <problem>"`` strings). Empty -> no-op.
        client: an Anthropic client. ``None`` -> no-op (returns the original).
        system_prompt: the generation system prompt to reuse, so the reviser
            obeys the same voice rules. ``None`` -> the active preflop prompt.

    Returns:
        A :class:`ReviseResult`. ``changed`` is True only when a rewrite both
        differed from the original AND passed the deterministic hard validators;
        otherwise the original explanation is returned unchanged.
    """
    if client is None or not issues:
        return ReviseResult(explanation=explanation, changed=False)

    system = (
        system_prompt if system_prompt is not None else load_preflop_system_prompt()
    )
    options = [o for o in explanation.options() if o]  # the fixed answer set
    user = (
        "SOLVER DATA:\n"
        f"{json.dumps(_trim_facts_for_prompt(facts), indent=2, default=str)}\n\n"
        f"OPTIONS (fixed, do not change): {options}\n"
        f"CORRECT ANSWER (fixed, do not change): {explanation.correct_answer}\n\n"
        f"YOUR EARLIER EXPLANATION:\n{explanation.answer_explanation}\n\n"
        "AUDIT ISSUES TO FIX:\n"
        + "\n".join(f"- {i}" for i in issues)
        + "\n\n"
        + _REVISER_INSTRUCTION
    )

    # CORRECTIVE RETRY (July 2026): a rewrite that breaks a hard rule used to
    # be discarded outright, shipping the WORST questions (flagged AND failed
    # auto-fix) as unfixed originals. Now the rejection -- the exact validator
    # error plus the rejected text -- is fed back for ONE more attempt (the
    # same corrective-retry pattern Layer-6 generation uses). Bounded to
    # _REVISE_ATTEMPTS total; a second failure keeps today's behavior exactly
    # (original ships, flagged, with the last rejection recorded).
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
                # Same 5-arg convention the generator uses: (model, input,
                # output, cache_creation, cache_read). Calling it with the raw
                # usage object raised a TypeError that the except below
                # swallowed, so EVERY rewrite silently failed
                # (regression-tested in test_reviser.py).
                usage_callback(model, *_extract_usage(response))
            revised_text = _parse_explanation_only_response(_extract_text(response))
        except Exception as exc:  # noqa: BLE001 - a reviser hiccup must never drop the row
            logger.warning("reviser: call/parse failed: %s", exc)
            return ReviseResult(
                explanation=explanation, changed=False,
                rejected_reason=f"revision call failed: {exc}",
            )

        if revised_text.strip() == explanation.answer_explanation.strip():
            return ReviseResult(
                explanation=explanation, changed=False, revised_text=revised_text
            )

        revised = _rebuild(explanation, revised_text)
        # The deterministic floor: a rewrite that breaks a hard rule never ships.
        result = run_preflop_audit_validators(revised, facts)
        if result.is_valid:
            return ReviseResult(
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

    return ReviseResult(
        explanation=explanation,
        changed=False,
        revised_text=last_text,
        rejected_reason=last_reason,
    )


__all__ = ["ReviseResult", "revise_explanation"]
