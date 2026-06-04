"""PLO gold-example pool for Layer 6 few-shot prompting -- EMPTY by design.

The PLO explanation generator ships WITHOUT few-shot examples: the (vetted NLHE)
voice rules carry the voice, and shipping no examples avoids templating every one
of 10k questions into a single recognizable mold. This module is the seam for
adding examples later, with zero generator changes -- populate
:data:`PLO_GOLD_EXAMPLES` and the generator picks them up.

Each example is a dict ``{"question": ..., "answer_explanation": ...}``. When you
do add them, use 3-5 DELIBERATELY DIVERSE ones (different openings + structures)
so they anchor the voice without imposing one template. The best source is the
pipeline's own strongest outputs after a graded batch (bootstrap), or a handful
of hand-vetted PLO explanations.
"""

from __future__ import annotations

from typing import Any

PLO_GOLD_EXAMPLES: tuple[dict[str, Any], ...] = ()


def load_plo_gold_examples() -> tuple[dict[str, Any], ...]:
    """The PLO few-shot examples (empty until :data:`PLO_GOLD_EXAMPLES` is filled)."""
    return PLO_GOLD_EXAMPLES


__all__ = ["PLO_GOLD_EXAMPLES", "load_plo_gold_examples"]
