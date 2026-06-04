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

import random
from dataclasses import dataclass
from pathlib import Path

from pipeline.plo.difficulty import compute_plo_difficulty
from pipeline.plo.fact_extractor import extract_plo_facts
from pipeline.plo.format_writer import build_plo_row, write_plo_csv
from pipeline.plo.hand_order import HAND_COUNT
from pipeline.plo.node_enumerator import PloDecisionNode, enumerate_plo_nodes
from pipeline.plo.options import build_options
from pipeline.plo.pack import PloActionType, PloPack
from pipeline.plo.question_extractor import is_question_worthy
from pipeline.plo.spot_sampler import PloSpot, sample_plo_spot

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

    @property
    def shortfall(self) -> int:
        """How many fewer questions were written than requested."""
        return max(0, self.questions_requested - self.questions_written)


def _active_players(node: PloDecisionNode) -> int:
    seats = {a.seat for a in node.history_before if a.action is not PloActionType.FOLD}
    seats.add(node.actor)
    return len(seats)


def _first_worthy_spot(node: PloDecisionNode, rng: random.Random) -> PloSpot | None:
    for index in rng.sample(range(HAND_COUNT), k=min(_WORTHY_TRIES, HAND_COUNT)):
        spot = sample_plo_spot(node, index)
        if spot.presence >= _MIN_PRESENCE and is_question_worthy(spot):
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
    compute_equity: bool = True,
    answer_style: str = "auto",
    seed: int = 0,
    stakes_bb_dollars: float = 1.0,
    game_format: str = "cash",
    display_in_bb: bool = False,
    stack_bb: float = 100.0,
    pack_label: str = "plo_6max_100bb",
) -> PloBatchResult:
    """Generate up to ``total_questions`` PLO question rows and write the CSV.

    One worthy spot per node (for variety) across a shuffled, filtered node set.
    Equity is computed per kept spot when ``compute_equity`` (the ~1s/spot cost;
    it only enriches the equity/range concept tags). The explanation column is
    left blank until Layer 6 exists.
    """
    nodes = enumerate_plo_nodes(pack)
    rng = random.Random(seed)
    candidates = [
        n
        for n in nodes
        if (hero_positions is None or n.actor in hero_positions)
        and (
            max_prior_raises is None
            or sum(1 for a in n.history_before if a.action in _AGGRESSIVE) <= max_prior_raises
        )
        and (max_active_players is None or _active_players(n) <= max_active_players)
    ]
    rng.shuffle(candidates)

    rows: list[dict[str, str]] = []
    scanned = 0
    for node in candidates:
        if len(rows) >= total_questions:
            break
        scanned += 1
        spot = _first_worthy_spot(node, rng)
        if spot is None:
            continue
        facts = extract_plo_facts(
            spot, pack, compute_equity=compute_equity, rng=random.Random(seed)
        )
        options, correct = build_options(facts, style=answer_style)
        rows.append(
            build_plo_row(
                facts,
                difficulty=compute_plo_difficulty(facts),
                options=options,
                correct_answer=correct,
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
    )


__all__ = ["PloBatchResult", "generate_plo_batch"]
