"""Layer-7 self-revision (postflop): an OPT-IN pass that REWRITES flagged prose.

The postflop analogue of :mod:`pipeline.preflop.reviser`. Where the claim
checker (:mod:`pipeline.postflop.claim_checker`) only FLAGS suspect prose, this
takes the explanation plus the checker's issues and produces a corrected
version, then re-runs the deterministic postflop hard validators on it.

The rewrite is bounded HARD, so the LLM can only polish wording within ground
truth and can never ship a fabrication the validators catch:

* it may rewrite ONLY the ``answer_explanation`` prose;
* ``option_1..option_4`` and ``correct_answer`` are solver-derived and
  re-attached verbatim -- the reviser literally cannot change the recommended
  action or the answer set, because anything it emits for those fields is
  ignored;
* the rewrite is re-validated by :func:`run_postflop_audit_validators`. If it
  breaks a hard rule (a banned phrase, an invented/garbled card), the rewrite
  is DISCARDED and the original (which already passed) is kept.

One extra LLM call, so it is opt-in (a batch flag), like the claim checker. The
batch keeps BOTH versions in ``meta.json`` so a reviewer can see what changed.
Postflop's generator returns plain prose (no JSON contract), so the reviser
asks for plain prose back too -- simpler than preflop's JSON-only response.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pipeline.explanation_generator import (
    GeneratedExplanation,
    _extract_text,
    call_messages_create,
)
from pipeline.postflop.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    build_solver_data_block,
    load_postflop_system_prompt,
)
from pipeline.postflop.facts import PostflopFacts
from pipeline.postflop.validators import run_postflop_audit_validators

logger = logging.getLogger(__name__)

UsageCallback = Callable[[object], None]

# Total rewrite attempts: the first pass plus ONE corrective retry that feeds
# the hard-validator rejection back (July 2026). Bounded -- the discard path
# is rare, so the extra call is cheap, but never loop unbounded.
_REVISE_ATTEMPTS = 2

# The editor's mandate. Kept terse; the VOICE RULES it must still obey come from
# the (reused) generation system prompt, so they never drift out of sync.
_REVISER_INSTRUCTION = (
    "You are an editor fixing ONE postflop answer explanation you wrote "
    "earlier. An automated audit flagged the problems listed below. REWRITE the "
    "explanation so each valid problem is gone, keeping the verdict and "
    "everything that is already correct.\n"
    "HOW TO USE THE FLAGS:\n"
    "- Treat each flag as a real error and rewrite the offending sentence to "
    "remove it. These audits are usually right: a range advantage given to the "
    "wrong player, a mislabeled draw, a backwards equity-or-price claim, a made "
    "hand called something it is not.\n"
    "- The SOLVER DATA above is ground truth. The RANGE ADVANTAGE / NUT "
    "ADVANTAGE lines settle who is ahead; the DRAWS line settles the draw type; "
    "HERO EQUITY and the FACING A BET break-even settle any ahead/behind or "
    "price claim; the BOARD line settles the cards.\n"
    "- The QUESTION above gives the exact action history (who bet, checked, "
    "called, raised, or led on each street). It is the source of truth for the "
    "LINE: if a flag calls a true line reference 'invented' (e.g. doubting a "
    "donk-lead or a check-raise that the QUESTION actually shows), that is a "
    "FALSE flag -- KEEP the correct statement, do not delete it to satisfy the "
    "flag.\n"
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
    "- When fixing a flagged sentence, keep its length and role; name "
    "hero's hand EXACTLY as the HAND CLASS line does.\n"
    "HARD CONSTRAINTS:\n"
    "- Do NOT change the recommended action, the verdict, any number, the bet "
    "size, or the four options. Those are fixed by the solver and given to you "
    "above. Only the wording of the explanation may change.\n"
    "- Obey every VOICE RULE in your instructions (no em dash, no semicolon, no "
    "banned phrases, suit emojis for specific cards, second person, the "
    "strategic frame, the range/draw/equity rules).\n"
    "Output ONLY the corrected explanation prose: no preamble, no headings, no "
    "JSON, no option labels."
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


def revise_postflop_explanation(
    explanation: GeneratedExplanation,
    facts: PostflopFacts,
    *,
    issues: list[str],
    client: object,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    question: str = "",
    system_prompt: str | None = None,
    usage_callback: UsageCallback | None = None,
) -> ReviseResult:
    """One LLM pass that rewrites a flagged postflop explanation, then
    re-validates it.

    Args:
        explanation: the explanation to fix (its options + correct_answer are
            kept verbatim; only the prose is rewritten).
        facts: the Layer-5 data block (ground truth + the re-validation input).
        issues: the audit findings to fix (the claim checker's
            ``"<claim> -- <problem>"`` strings). Empty -> no-op.
        question: the action-history narrative (the line). Passed so the reviser
            sees the same truth the writer did and keeps correct line references
            instead of deleting them to satisfy a false "invented line" flag.
        client: an Anthropic client. ``None`` -> no-op (returns the original).
        system_prompt: the generation system prompt to reuse, so the reviser
            obeys the same voice rules. ``None`` -> the active postflop prompt.

    Returns:
        A :class:`ReviseResult`. ``changed`` is True only when a rewrite both
        differed from the original AND passed the deterministic hard validators;
        otherwise the original explanation is returned unchanged.
    """
    if client is None or not issues:
        return ReviseResult(explanation=explanation, changed=False)

    system = (
        system_prompt if system_prompt is not None else load_postflop_system_prompt()
    )
    options = [o for o in explanation.options() if o]  # the fixed answer set
    head = (
        f"QUESTION (the spot + full action history):\n{question}\n\n"
        if question.strip()
        else ""
    )
    user = (
        f"{head}"
        "SOLVER DATA:\n"
        f"{build_solver_data_block(facts)}\n\n"
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
            if usage_callback is not None and getattr(response, "usage", None) is not None:
                usage_callback(response.usage)
            revised_text = _extract_text(response).strip()
        except Exception as exc:  # noqa: BLE001 - a reviser hiccup must never drop the row
            logger.warning("postflop reviser: call/parse failed: %s", exc)
            return ReviseResult(
                explanation=explanation,
                changed=False,
                rejected_reason=f"revision call failed: {exc}",
            )

        if not revised_text or revised_text == explanation.answer_explanation.strip():
            return ReviseResult(
                explanation=explanation, changed=False, revised_text=revised_text
            )

        revised = _rebuild(explanation, revised_text)
        # The deterministic floor: a rewrite that breaks a hard rule never ships.
        result = run_postflop_audit_validators(revised, facts)
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


__all__ = ["ReviseResult", "revise_postflop_explanation"]
