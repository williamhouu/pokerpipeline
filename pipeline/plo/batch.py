"""PLO batch orchestrator -- sample worthy spots and write a question CSV.

generate_plo_batch ties the deterministic pipeline together: enumerate nodes ->
sample worthy spots -> extract facts -> deterministic options + difficulty ->
build_plo_row -> write_plo_csv. It produces a complete CSV today, with every
column populated except ``Answer Explanation`` (Layer 6, the LLM prose, isn't
built yet -- when it lands the loop just fills that column).

Like the admin preview, it defaults to the verified-CLEAN lines (<= 2 prior
raises, <= 3 players): random node sampling otherwise over-weights Monker's deep
multiway 4-bet+/jam tail, which is largely unconverged. Pass ``None`` for both
caps to include it.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.explanation_generator import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ExplanationValidationError,
)
from pipeline.plo.difficulty import compute_plo_difficulty
from pipeline.plo.explanation_generator import (
    UsageCallback,
    generate_plo_answer_explanation,
)
from pipeline.plo.fact_extractor import extract_plo_facts
from pipeline.plo.format_writer import build_plo_row, write_plo_csv
from pipeline.plo.hand_order import HAND_COUNT
from pipeline.plo.node_enumerator import (
    PloDecisionNode,
    enumerate_plo_nodes,
    plo_active_player_count,
    plo_node_action_context,
)
from pipeline.plo.options import build_options
from pipeline.plo.pack import PloActionType, PloPack
from pipeline.plo.question_extractor import (
    MAX_TOP_FREQUENCY,
    MIN_TOP_FREQUENCY,
    is_question_worthy,
)
from pipeline.plo.spot_sampler import PloSpot, sample_plo_spot

logger = logging.getLogger(__name__)

_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
_MIN_PRESENCE = 0.5
_WORTHY_TRIES = 800  # random hands to try per node when hunting


@dataclass(frozen=True)
class PloBatchResult:
    """Outcome of a PLO batch run."""

    output_path: Path
    questions_written: int
    questions_requested: int
    nodes_scanned: int
    explanations_written: int = 0
    explanations_failed: int = 0
    difficulty_filtered_out: int = 0
    ev_gap_filtered_out: int = 0

    @property
    def shortfall(self) -> int:
        """How many fewer questions were written than requested."""
        return max(0, self.questions_requested - self.questions_written)


def _first_worthy_spot(
    node: PloDecisionNode,
    rng: random.Random,
    *,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
) -> PloSpot | None:
    for index in rng.sample(range(HAND_COUNT), k=min(_WORTHY_TRIES, HAND_COUNT)):
        spot = sample_plo_spot(node, index)
        if spot.presence >= _MIN_PRESENCE and is_question_worthy(
            spot, min_frequency=min_frequency, max_frequency=max_frequency
        ):
            return spot
    return None


def generate_plo_batch(
    pack: PloPack,
    *,
    output_path: Path | str,
    total_questions: int = 30,
    hero_positions: list[str] | None = None,
    max_prior_raises: int | None = 2,
    max_active_players: int | None = 3,
    action_contexts: list[str] | None = None,
    player_counts: list[int] | None = None,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_ev_gap_bb: float | None = None,
    compute_equity: bool = True,
    answer_style: str = "auto",
    seed: int = 0,
    stakes_bb_dollars: float = 1.0,
    game_format: str = "cash",
    display_in_bb: bool = False,
    stack_bb: float = 100.0,
    pack_label: str = "plo_6max_100bb",
    min_difficulty: int = 0,
    max_difficulty: int = 10_000,
    generate_explanations: bool = False,
    explanation_client: Any = None,
    explanation_model: str = DEFAULT_MODEL,
    explanation_temperature: float = DEFAULT_TEMPERATURE,
    explanation_system_prompt: str | None = None,
    explanation_include_skills: bool = False,
    usage_callback: UsageCallback | None = None,
) -> PloBatchResult:
    """Generate up to ``total_questions`` PLO question rows and write the CSV.

    One worthy spot per node (for variety) across a shuffled, filtered node set.
    Equity is computed per kept spot when ``compute_equity`` (the ~1s/spot cost;
    it only enriches the equity/range concept tags).

    Node filters (all optional, all AND-combined): ``hero_positions``,
    ``action_contexts`` (the :data:`PLO_ACTION_CONTEXTS` buckets), and
    ``player_counts`` mirror the NLHE Generate page; ``max_prior_raises`` /
    ``max_active_players`` are the coarse clean-line caps. Spot filters:
    ``min_frequency`` / ``max_frequency`` set the worthiness window, and
    ``min_ev_gap_bb`` drops near-coinflip spots (reported in
    ``ev_gap_filtered_out``).

    When ``generate_explanations`` is True, Layer 6 (the LLM) fills the
    ``Answer Explanation`` column per spot; otherwise it is left blank (the
    deterministic path, no API key needed). A failed explanation never drops the
    question -- the row still ships with a blank explanation and is counted in
    ``explanations_failed``. ``explanation_system_prompt`` (when set) overrides
    the built-in Layer 6 system prompt verbatim -- the admin prompt library +
    Compare page pass an edited prompt here.
    """
    nodes = enumerate_plo_nodes(pack)
    rng = random.Random(seed)
    ctx_set = set(action_contexts) if action_contexts else None
    pc_set = set(player_counts) if player_counts else None
    candidates = [
        n
        for n in nodes
        if (hero_positions is None or n.actor in hero_positions)
        and (
            max_prior_raises is None
            or sum(1 for a in n.history_before if a.action in _AGGRESSIVE) <= max_prior_raises
        )
        and (
            max_active_players is None
            or plo_active_player_count(n) <= max_active_players
        )
        and (ctx_set is None or plo_node_action_context(n) in ctx_set)
        and (pc_set is None or plo_active_player_count(n) in pc_set)
    ]
    rng.shuffle(candidates)

    rows: list[dict[str, str]] = []
    scanned = 0
    explanations_written = 0
    explanations_failed = 0
    difficulty_filtered_out = 0
    ev_gap_filtered_out = 0
    for node in candidates:
        if len(rows) >= total_questions:
            break
        scanned += 1
        spot = _first_worthy_spot(
            node, rng, min_frequency=min_frequency, max_frequency=max_frequency
        )
        if spot is None:
            continue
        facts = extract_plo_facts(
            spot, pack, compute_equity=compute_equity, rng=random.Random(seed)
        )
        # EV-gap quality gate (PLO has a real ev_gap on every spot, raises
        # included), applied BEFORE the paid LLM call -- mirrors the NLHE gate.
        if (
            min_ev_gap_bb is not None
            and facts.ev_gap_bb is not None
            and facts.ev_gap_bb < min_ev_gap_bb
        ):
            ev_gap_filtered_out += 1
            continue
        # Difficulty-band filter BEFORE the (paid) LLM call, so out-of-band
        # spots cost no API spend -- the same gate the NLHE Generate page uses.
        difficulty = compute_plo_difficulty(facts)
        if not min_difficulty <= difficulty.score <= max_difficulty:
            difficulty_filtered_out += 1
            continue
        options, correct = build_options(facts, style=answer_style)

        explanation = ""
        if generate_explanations:
            try:
                generated = generate_plo_answer_explanation(
                    facts,
                    list(options),
                    correct,
                    client=explanation_client,
                    system_prompt=explanation_system_prompt,
                    model=explanation_model,
                    temperature=explanation_temperature,
                    include_skills=explanation_include_skills,
                    usage_callback=usage_callback,
                )
                explanation = generated.answer_explanation
                explanations_written += 1
            except (ExplanationValidationError, OSError, KeyError) as exc:
                explanations_failed += 1
                logger.warning("Layer 6 failed for a spot, shipping blank: %s", exc)

        rows.append(
            build_plo_row(
                facts,
                difficulty=difficulty,
                options=options,
                correct_answer=correct,
                explanation=explanation,
                number=len(rows) + 1,
                pack_label=pack_label,
                stakes_bb_dollars=stakes_bb_dollars,
                game_format=game_format,
                display_in_bb=display_in_bb,
                stack_bb=stack_bb,
            )
        )

    output_path = Path(output_path)
    write_plo_csv(rows, output_path)
    return PloBatchResult(
        output_path=output_path,
        questions_written=len(rows),
        questions_requested=total_questions,
        nodes_scanned=scanned,
        explanations_written=explanations_written,
        explanations_failed=explanations_failed,
        difficulty_filtered_out=difficulty_filtered_out,
        ev_gap_filtered_out=ev_gap_filtered_out,
    )


__all__ = ["PloBatchResult", "generate_plo_batch"]
