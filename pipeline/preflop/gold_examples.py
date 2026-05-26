"""Preflop-only filter on top of the shared gold-example pool.

The existing ``pipeline.explanation_generator.load_gold_examples()`` reads
ALL gold examples from ``docs/output_format_examples.xlsx`` Sheet 2 -- a
mix of preflop, flop, turn, and river. For preflop-question generation
Layer 6 only wants the preflop subset; otherwise the LLM gets postflop
voice cues that don't apply (board-texture talk, equity-vs-runout math,
etc.).

This module wraps the existing loader and filters by ``Hand Stage``
(case-insensitive: the xlsx has ``preflop`` and ``Preflop`` both).
Result is cached so a batch run doesn't re-read the xlsx per question.

Usage::

    from pipeline.preflop.gold_examples import load_preflop_gold_examples
    examples = load_preflop_gold_examples()
    # Pass to Layer 6 in place of the postflop pool.
"""

from __future__ import annotations

from functools import lru_cache

from pipeline.explanation_generator import load_gold_examples

# Hand Stage values that count as "preflop". The xlsx has inconsistent
# casing across rows (some `preflop`, some `Preflop`) -- normalise.
_PREFLOP_STAGE_VALUES = {"preflop"}


def _is_preflop(example: dict) -> bool:
    """True if the example's Hand Stage normalises to 'preflop'."""
    stage = example.get("Hand Stage")
    if not isinstance(stage, str):
        return False
    return stage.strip().lower() in _PREFLOP_STAGE_VALUES


@lru_cache(maxsize=1)
def load_preflop_gold_examples(
    path: str | None = None,
) -> tuple[dict, ...]:
    """Load the preflop subset of the gold-example pool.

    Args:
        path: Optional override for the xlsx path (mirrors the postflop
            loader's API). None = use the default docs path.

    Returns:
        Tuple of preflop example dicts in the order they appear in the
        xlsx. Each dict has the same fields as the postflop loader
        (Hand Stage, Question, option 1-4, Correct Answer, Answer
        Explanation, Preflop Pot Type, Pot Participant, Stack Depth).
    """
    pool = load_gold_examples(path)
    return tuple(ex for ex in pool if _is_preflop(ex))
