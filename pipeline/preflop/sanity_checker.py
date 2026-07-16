"""Layer 7 sanity audit -- an LLM pass over the SOLVER DATA block itself.

WHAT IT IS (plain terms): every other check in the pipeline verifies
INTERNAL consistency -- prose against the data block, CSV against a rebuild.
None of them can notice when a deterministic FACT is itself wrong, because
they all treat the data block as ground truth (that is how the BvB
position bug and the empty-domination bug shipped: every layer faithfully
agreed with a wrong fact). This pass is the one place an LLM is ALLOWED to
bring outside poker knowledge: it reads ONLY the solver-data facts (never
the prose) and asks "does anything here contradict basic poker?".

WHY IT IS FLAG-ONLY AND CLEARLY SEPARATED: the project's founding premise
is that LLMs are confidently wrong about poker, so this checker's opinions
are HYPOTHESES for a human reviewer, never gates. It must not touch
generation, rewrite anything, or reject a row -- it appends warnings to the
question's meta record for the Review page. It FAILS OPEN (an errored or
unparseable call = no flags) and is opt-in per batch (one extra LLM call
per question).

Scope discipline: the prompt restricts it to BASIC, high-confidence facts a
competent player states on sight (postflop action order, domination
direction, equity plausibility bands, pot-odds arithmetic, action-label /
raise-level consistency) -- NOT strategy opinions, which is where LLM poker
judgement gets unreliable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pipeline.explanation_generator import (
    DEFAULT_MODEL,
    _extract_text,
    call_messages_create,
)

_SANITY_MAX_TOKENS = 1024

# v2 prompt (July 2026). Calibrated on a real batch where v1 produced 7
# flags, ALL false positives -- several with the checker's own poker wrong.
# v2 embeds the deterministic REFERENCE RULES (so the checker computes from
# them instead of trusting its memory) and each v1 misfire as an explicit
# do-not-flag counter-example.
SANITY_SYSTEM_PROMPT = (
    "You are a strong No-Limit Hold'em player auditing a block of "
    "machine-generated FACTS about one preflop spot (positions, ranges, "
    "equity, pot odds, domination lists, action history). Unlike every "
    "other check in this pipeline, you SHOULD use poker knowledge -- but "
    "compute it from the REFERENCE RULES below, never from memory. Your "
    "job is to catch a fact that is simply wrong about poker. You never "
    "see or judge any explanation prose -- only the facts.\n\n"
    "REFERENCE RULES (authoritative -- reason FROM these):\n"
    "  * Postflop action order at a ring table: SB, BB, UTG, UTG+1, UTG+2, "
    "LJ, HJ, CO, BTN. The LATER seat in this list is in position. This "
    "compares hero to THE VILLAIN IN THE POT only -- seats that already "
    "folded or are still to act preflop are irrelevant. So: blind-vs-blind "
    "means the SB acts first postflop and the BB is in position; LJ IS in "
    "position against a UTG+1 open even though five seats are behind LJ "
    "preflop.\n"
    "  * Break-even to call = amount_to_call / (pot_after_your_call). "
    "Recompute it from the stated pot and call amounts before flagging; "
    "different seats face different prices (a cold-caller of a 3bb open "
    "with blinds posted faces 3/(3+3+1.5) = 40%, while the BB discount "
    "makes it ~31% -- BOTH are correct in their own spots). Only flag when "
    "YOUR arithmetic from the stated amounts disagrees by 3+ points.\n"
    "  * Gappers count the SKIPPED ranks: KQ = connector, KJ = one-gapper, "
    "KT = two-gapper, AT = three-gapper (skips K, Q, J).\n"
    "  * Same ranks suited vs offsuit (A4s vs A4o) is ~52/48 and mostly "
    "chops -- a coinflip classification is CORRECT, not domination.\n"
    "  * Domination lists cap at ~6 entries and only name hands the "
    "villain actually holds, so a MISSING entry is never an error -- only "
    "a wrong-DIRECTION entry, or an empty dominated_by while the villain's "
    "listed likely hands obviously dominate the hero.\n\n"
    "Check ONLY these categories, and flag ONLY high-confidence errors:\n"
    "  (a) POSITION: the stated in/out-of-position value contradicts the "
    "reference seat order (hero vs the villain in the pot).\n"
    "  (b) DOMINATION DIRECTION: a listed entry points the wrong way, or "
    "an empty dominated_by is contradicted by the likely-hands list.\n"
    "  (c) EQUITY PLAUSIBILITY: a heads-up equity number 10+ points off "
    "what the matchup implies. Mind the WHOLE villain range: against a "
    "3-bet range that includes bluff hands, a dominated broadway at ~43% "
    "is plausible; judge against the listed likely hands, not an imagined "
    "premium-only range.\n"
    "  (d) ARITHMETIC: a break-even you recomputed from the stated amounts "
    "that disagrees by 3+ points; frequencies not summing to ~100%; a "
    "dominant action that is not the most frequent.\n"
    "  (e) ACTION HISTORY: labels inconsistent with the raise count (a "
    "'3-bet' that is actually a 4-bet), an all-in described as leaving "
    "postflop play, players listed as still-to-act who already folded.\n\n"
    "DO NOT FLAG (each of these was a real false positive):\n"
    "  * 'BB in position in a blind-vs-blind pot' -- that is CORRECT (the "
    "SB acts first postflop).\n"
    "  * 'LJ in position against a UTG+1 open' -- CORRECT (LJ acts later "
    "postflop; players behind preflop don't matter).\n"
    "  * 'break_even 0.40 facing a single 3bb open' -- CORRECT for a "
    "cold-caller (3/7.5); do not apply the BB's discounted price to other "
    "seats.\n"
    "  * 'AT labeled three_gapper' -- CORRECT (skips K, Q, J).\n"
    "  * 'A4s listed as a coinflip vs hero's A4o' -- CORRECT (same ranks "
    "mostly chop).\n"
    "  * A plausible-but-surprising equity vs a WIDE or bluff-containing "
    "range.\n\n"
    "BE VERY CONSERVATIVE. You are the fallible layer here: every flag "
    "sends a human to re-check a deterministic computation, so a false "
    "flag is expensive. If a value is merely surprising, strategy-"
    "dependent, or within noise, OMIT it. Do NOT comment on strategy, "
    "frequencies being tight or loose, ranges being good or bad, or "
    "anything outside the five categories.\n\n"
    "Return a single JSON object: {\"issues\": [{\"fact\": \"<the field or "
    "quoted value>\", \"problem\": \"<one sentence: what basic poker fact it "
    "contradicts>\"}]}. If nothing contradicts basic poker, return "
    "{\"issues\": []}. Output only the JSON, no preamble, no code fence."
)


@dataclass(frozen=True)
class SanityIssue:
    """One flagged fact from the sanity audit."""

    fact: str      # the field / value the checker is challenging
    problem: str   # the basic poker fact it appears to contradict


@dataclass(frozen=True)
class SanityCheckResult:
    """The sanity audit's verdict for one spot's data block."""

    passed: bool
    issues: tuple[SanityIssue, ...]
    raw: str = ""


def build_sanity_user_prompt(solver_data: dict[str, Any]) -> str:
    """The per-call user message: the data block alone (no prose)."""
    return (
        "SOLVER DATA (machine-generated facts to audit):\n"
        f"{json.dumps(solver_data, indent=2, default=str)}\n"
    )


def parse_sanity_response(text: str) -> SanityCheckResult:
    """Parse the checker's JSON. Fails OPEN: unparseable output becomes a
    clean pass (raw kept for debugging) so a malfunction never flags rows."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
        raw_issues = payload.get("issues", [])
        issues = tuple(
            SanityIssue(
                fact=str(i.get("fact", "")),
                problem=str(i.get("problem", "")),
            )
            for i in raw_issues
            if isinstance(i, dict)
        )
        return SanityCheckResult(passed=not issues, issues=issues, raw=text)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return SanityCheckResult(passed=True, issues=(), raw=text)


def check_solver_data_sanity(
    solver_data: dict[str, Any],
    client: Any,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    usage_callback: Any = None,
) -> SanityCheckResult:
    """Audit one spot's SOLVER DATA block against basic poker knowledge.

    One LLM call; flag-only; fails open. Mirrors the claim checker's
    client handling (self-creates from ANTHROPIC_API_KEY when handed None,
    the admin panel's pattern). ``usage_callback`` is the preflop 5-arg
    convention (spend-logger rule, July 2026: every LLM call site MUST
    report usage or the lifetime ledger under-counts)."""
    if client is None:
        import os  # noqa: PLC0415

        from anthropic import Anthropic  # noqa: PLC0415

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = call_messages_create(
        client,
        model=model,
        max_tokens=_SANITY_MAX_TOKENS,
        temperature=temperature,
        system=SANITY_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_sanity_user_prompt(solver_data)}
        ],
    )
    if usage_callback is not None:
        from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
            _extract_usage,
        )

        usage_callback(model, *_extract_usage(response))
    return parse_sanity_response(_extract_text(response))


def _normalize_fact(fact: str) -> str:
    """Loose key for matching the same challenged fact across passes:
    lowercase, keep only the part before a ':' (the field name), strip."""
    return fact.split(":", 1)[0].strip().lower()


def consensus_issues(
    first: SanityCheckResult, second: SanityCheckResult
) -> tuple[SanityIssue, ...]:
    """Issues BOTH passes raised about the same fact (July 2026).

    The checker is non-deterministic even at temperature 0, and v1
    calibration showed single-pass flags are FP-heavy. Requiring two
    independent passes to challenge the same fact (matched on the field
    name, since wording varies) cuts one-off hallucinated flags while a
    genuinely wrong fact keeps getting challenged. The first pass's
    wording is kept.
    """
    second_keys = {_normalize_fact(i.fact) for i in second.issues}
    return tuple(
        i for i in first.issues if _normalize_fact(i.fact) in second_keys
    )


__all__ = [
    "SANITY_SYSTEM_PROMPT",
    "SanityCheckResult",
    "SanityIssue",
    "build_sanity_user_prompt",
    "check_solver_data_sanity",
    "consensus_issues",
    "parse_sanity_response",
]
