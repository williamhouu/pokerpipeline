"""Per-question chatbot context (the ``chat_context`` CSV column).

The app's per-question AI chatbot lets a user, after answering, ask anything
about THAT spot. To keep it accurate it must be fed the full deterministic
truth, never left to invent poker facts. This module assembles that truth into
ONE self-contained JSON blob per question, with a single schema shared across
the preflop, postflop, and PLO pipelines, so the app consumes one column the
same way regardless of question type.

It carries strictly MORE than the generation "SOLVER DATA" block, because a
chatbot user asks comparative + hypothetical questions the explanation never
addressed:

* ``full_strategy`` -- EVERY action's frequency AND EV (not just the recommended
  one), so "why not raise?" is answerable with the real numbers;
* ``villain`` -- the opponent's range summary + likely holdings, so "what could
  they have?" is grounded;
* ``also_acceptable`` -- the neutral-credit alternatives, so "was my line ok?"
  is accurate;
* ``coaching_answer`` -- the explanation already shown, so the chat is consistent;
* ``guardrails`` -- a baked-in instruction not to invent numbers (a safety net
  on top of whatever system prompt the app wraps this in).

This is a pure formatter (like :mod:`pipeline.neutral_credit`) -- cards in,
facts in, JSON out -- so every pipeline can build it from data it already has
without coupling to another pipeline's internals. Deterministic: the same
inputs always produce the same string (byte-identical CSV is preserved).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# The chatbot's grounding rule, baked into the data so accuracy survives even a
# thin app-side system prompt. The app SHOULD also enforce this, but defence in
# depth is cheap here.
CHATBOT_GUARDRAILS = (
    "All numbers and facts above are solver-derived and exact for THIS hand and "
    "spot. Answer only from this data. Never invent or estimate equities, "
    "frequencies, combos, blockers, or cards beyond what is given. You may "
    "reason about and explain the data and general strategy, but if the user "
    "asks about a different hand, sizing, or runout that is not in this data, "
    "say it is outside this spot's solved data rather than guessing a number."
)


@dataclass(frozen=True)
class StrategyEntry:
    """One available action's frequency (+ EV when the solve exposes it)."""

    action: str
    frequency_pct: float
    ev_bb: float | None = None


def _strategy_payload(strategy: list[StrategyEntry]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in sorted(strategy, key=lambda s: -s.frequency_pct):
        item: dict[str, Any] = {
            "action": e.action,
            "frequency_pct": round(e.frequency_pct, 1),
        }
        if e.ev_bb is not None:
            item["ev_bb"] = round(e.ev_bb, 2)
        out.append(item)
    return out


def build_chat_context(
    *,
    pipeline: str,                       # "preflop" | "postflop" | "plo"
    situation: str,                      # action history / how the spot was reached
    hero_hand: str,                      # hero's cards, emoji form
    hand_summary: str,                   # hand class / shape (+ made hand / draws)
    recommended_action: str,
    also_acceptable: list[str],          # neutral-credit options ("" excluded)
    full_strategy: list[StrategyEntry],  # ALL actions, freq (+ EV)
    key_facts: dict[str, Any],           # equity, pot odds, spr, advantages, blockers, ...
    villain: dict[str, Any] | None,      # seat, action, range %, likely hands
    strategic_frame: str,                # archetype + one-line intent
    concept_tags: list[str],
    skills_tested: list[str],
    difficulty: int | str,
    coaching_answer: str,                # the explanation already shown to the user
) -> str:
    """Assemble the per-question chatbot context as a compact JSON string.

    One schema for all three pipelines (see the module docstring). Pure +
    deterministic; ``ensure_ascii=False`` keeps suit emojis readable.
    """
    payload = {
        "pipeline": pipeline,
        "situation": situation,
        "hero_hand": hero_hand,
        "hand_summary": hand_summary,
        "recommended_action": recommended_action,
        "also_acceptable": [a for a in also_acceptable if a],
        "full_strategy": _strategy_payload(full_strategy),
        "key_facts": {k: v for k, v in key_facts.items() if v not in (None, "")},
        "villain": villain or {},
        "strategic_frame": strategic_frame,
        "concept_tags": list(concept_tags),
        "skills_tested": list(skills_tested),
        "difficulty": difficulty,
        "coaching_answer": coaching_answer,
        "guardrails": CHATBOT_GUARDRAILS,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = ["CHATBOT_GUARDRAILS", "StrategyEntry", "build_chat_context"]
