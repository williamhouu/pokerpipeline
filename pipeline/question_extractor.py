"""Layer 4: Question Extractor.

Decides whether a decision spot is worth turning into a training question, and
rates how hard it is.

Per docs/engineering_brief.docx, "Layer 4: Question Extractor", a spot makes a
good question when BOTH filters pass:

  * the solver's highest-frequency action sits between 55% and 95% -- dominant
    enough that there is a clear answer to teach, not so dominant the answer is
    obvious;
  * the EV gap between the best action and the second-best is at least 0.5bb --
    big enough that choosing the wrong answer costs real money, not solver noise.

Difficulty is the brief's MVP formula, mapping the top action's frequency onto
a 500-3000 scale (500 = easiest, 3000 = hardest).
"""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.fact_extractor.spot_data import SpotData

# Brief thresholds -- starting values, to be tuned against the gold pool.
MIN_TOP_FREQUENCY = 0.55          # below: no clear best answer to teach
MAX_TOP_FREQUENCY = 0.95          # above: the answer is too obvious
MIN_EV_GAP_BB = 0.5               # below: overlaps with solver noise

_DIFFICULTY_CEILING = 3000        # hardest
_DIFFICULTY_FLOOR = 500           # easiest


def top_action_frequency(spot_data: SpotData) -> float:
    """Frequency of the solver's most-played action at the spot (0 if unknown).

    Read from the range-aggregate strategy -- the solver's strategy averaged
    over hero's whole range, which is what "dominant but not pure" describes.
    """
    strategy = spot_data.decision_data.range_aggregate_strategy
    return max(strategy.values()) if strategy else 0.0


def difficulty_score(spot_data: SpotData) -> int:
    """The spot's difficulty on the brief's 500-3000 scale.

    Brief MVP formula: 3000 - ((freq - 0.55) / 0.40) * 2500, where `freq` is the
    top action's frequency. A barely-dominant 55% spot scores 3000 (hardest); a
    near-pure 95% spot scores 500 (easiest). Clamped to the 500-3000 range.
    """
    frequency = top_action_frequency(spot_data)
    score = 3000 - ((frequency - 0.55) / 0.40) * 2500
    return round(max(_DIFFICULTY_FLOOR, min(_DIFFICULTY_CEILING, score)))


def is_question_worthy(spot_data: SpotData, *,
                       min_frequency: float = MIN_TOP_FREQUENCY,
                       max_frequency: float = MAX_TOP_FREQUENCY,
                       min_ev_gap_bb: float = MIN_EV_GAP_BB) -> bool:
    """Whether a spot passes both of the brief's question-worthiness filters.

    Filter 1 (frequency) is inclusive at both ends; filter 2 (EV gap) is "at
    least", so a gap of exactly the threshold passes.
    """
    frequency = top_action_frequency(spot_data)
    if not min_frequency <= frequency <= max_frequency:
        return False
    return spot_data.decision_data.ev_gap_bb >= min_ev_gap_bb


@dataclass
class QuestionEvaluation:
    """The Question Extractor's verdict on a spot."""

    is_worthy: bool
    top_action_frequency: float
    ev_gap_bb: float
    difficulty_score: int


def evaluate_spot(spot_data: SpotData) -> QuestionEvaluation:
    """Run both filters and the difficulty rating, returning the full verdict."""
    return QuestionEvaluation(
        is_worthy=is_question_worthy(spot_data),
        top_action_frequency=top_action_frequency(spot_data),
        ev_gap_bb=spot_data.decision_data.ev_gap_bb,
        difficulty_score=difficulty_score(spot_data),
    )
