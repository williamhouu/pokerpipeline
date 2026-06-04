"""Read-only PLO pipeline preview for the admin panel.

Runs real PLO decision spots from the pack through the deterministic pipeline
(worthiness gate -> facts -> options -> difficulty / concept tags / skills) and
returns a flat, displayable row per spot, so a reviewer can eyeball that the PLO
work is producing sensible questions.

It is strictly READ-ONLY: no LLM (Layer 6 / explanations are not built yet) and
no CSV writing. The ``answer_explanation`` and the final CSV layout are
deliberately absent here -- this is a sanity-check surface, not the output path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from pipeline.plo.action_history import format_plo_action_history
from pipeline.plo.difficulty import compute_plo_difficulty
from pipeline.plo.fact_extractor import extract_plo_facts
from pipeline.plo.hand_order import HAND_COUNT
from pipeline.plo.node_enumerator import (
    PloDecisionNode,
    enumerate_plo_nodes,
    plo_active_player_count,
    plo_node_action_context,
)
from pipeline.plo.options import build_options, canonicalize_strategy
from pipeline.plo.pack import PloActionType, PloPack, discover_plo_pack
from pipeline.plo.position import hero_relative_position
from pipeline.plo.question_extractor import (
    MAX_TOP_FREQUENCY,
    MIN_TOP_FREQUENCY,
    is_question_worthy,
)
from pipeline.plo.skill_tagger import compute_plo_skills
from pipeline.plo.spot_sampler import PloSpot, sample_plo_spot
from pipeline.plo.spot_tags import compute_plo_concept_tags

_SUIT_EMOJI = {"s": "♠️", "h": "❤️", "d": "♦️", "c": "♣️"}
_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
_MIN_PRESENCE = 0.5  # a real spot the hero actually reaches with this hand
_DEFAULT_WORTHY_TRIES = 600  # random hands to try per node when hunting


def format_cards(cards: tuple[str, ...]) -> str:
    """``('As','Ks','Ah','Kh')`` -> ``'A♠️ K♠️ A❤️ K❤️'``."""
    return " ".join(c[0] + _SUIT_EMOJI[c[1]] for c in cards)


@dataclass(frozen=True)
class PloPreviewRow:
    """One previewed PLO spot, flattened for display."""

    hand_label: str
    cards: str
    position: str
    relative_position: str
    action_line: str
    options: tuple[str, ...]
    correct_answer: str
    action_frequencies: tuple[tuple[str, float], ...]  # (label, freq), desc
    archetype: str
    dominant_action: str
    dominant_freq: float
    ev_gap_bb: float | None
    equity: float | None
    difficulty: int
    concept_tags: tuple[str, ...]
    skills: tuple[str, ...]


def load_pack_and_nodes(pack_dir: Path) -> tuple[PloPack, tuple[PloDecisionNode, ...]]:
    """Discover the PLO pack under ``pack_dir`` and enumerate its nodes."""
    pack = discover_plo_pack(pack_dir)
    return pack, enumerate_plo_nodes(pack)


def _first_worthy_spot(
    node: PloDecisionNode,
    rng: random.Random,
    *,
    tries: int = _DEFAULT_WORTHY_TRIES,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
) -> PloSpot | None:
    """A worthy spot at this node, found by random-sampling hands (cheap --
    worthiness reads only the spot, no facts/equity)."""
    for index in rng.sample(range(HAND_COUNT), k=min(tries, HAND_COUNT)):
        spot = sample_plo_spot(node, index)
        if spot.presence >= _MIN_PRESENCE and is_question_worthy(
            spot, min_frequency=min_frequency, max_frequency=max_frequency
        ):
            return spot
    return None


def build_preview_rows(
    pack: PloPack,
    nodes: tuple[PloDecisionNode, ...],
    *,
    count: int = 8,
    seed: int = 0,
    hero_positions: list[str] | None = None,
    max_prior_raises: int | None = 2,
    max_active_players: int | None = 3,
    action_contexts: list[str] | None = None,
    player_counts: list[int] | None = None,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_ev_gap_bb: float | None = None,
    compute_equity: bool = False,
    answer_style: str = "auto",
    display_in_bb: bool = True,
    max_nodes_scanned: int = 80,
) -> list[PloPreviewRow]:
    """Sample up to ``count`` worthy spots and run them through the pipeline.

    Picks one worthy spot per node (for variety) across a shuffled, filtered
    node set. The node filters mirror :func:`pipeline.plo.batch.generate_plo_batch`
    (and the NLHE Generate page): ``hero_positions``, ``action_contexts`` (the
    :data:`~pipeline.plo.node_enumerator.PLO_ACTION_CONTEXTS` buckets),
    ``player_counts``, plus the coarse ``max_prior_raises`` / ``max_active_players``
    clean-line caps. The spot filters are the worthiness window
    (``min_frequency`` / ``max_frequency``) and the ``min_ev_gap_bb`` quality gate.

    Equity is only computed for the kept spots, and only when ``compute_equity``
    is True (the one ~1s/spot cost). ``action_line`` is the REAL Question prose
    (:func:`~pipeline.plo.action_history.format_plo_action_history`) so the
    preview matches the CSV exactly.
    """
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

    rows: list[PloPreviewRow] = []
    for node in candidates[:max_nodes_scanned]:
        spot = _first_worthy_spot(
            node, rng, min_frequency=min_frequency, max_frequency=max_frequency
        )
        if spot is None:
            continue
        facts = extract_plo_facts(
            spot, pack, compute_equity=compute_equity, rng=random.Random(seed)
        )
        if (
            min_ev_gap_bb is not None
            and facts.ev_gap_bb is not None
            and facts.ev_gap_bb < min_ev_gap_bb
        ):
            continue
        options, correct = build_options(facts, style=answer_style)
        difficulty = compute_plo_difficulty(facts)
        freqs = tuple(
            (label, freq)
            for label, freq in sorted(
                canonicalize_strategy(facts).items(), key=lambda kv: -kv[1]
            )
            if freq > 0
        )
        rows.append(
            PloPreviewRow(
                hand_label=spot.hero_label,
                cards=format_cards(spot.hero_cards),
                position=node.actor,
                relative_position=hero_relative_position(facts),
                action_line=format_plo_action_history(
                    facts, display_in_bb=display_in_bb
                ),
                options=tuple(options),
                correct_answer=correct,
                action_frequencies=freqs,
                archetype=facts.archetype,
                dominant_action=spot.dominant_action,
                dominant_freq=spot.dominant_frequency,
                ev_gap_bb=facts.ev_gap_bb,
                equity=facts.hero_equity_vs_villain,
                difficulty=difficulty.score,
                concept_tags=tuple(compute_plo_concept_tags(facts)),
                skills=tuple(compute_plo_skills(facts)),
            )
        )
        if len(rows) >= count:
            break
    return rows


__all__ = [
    "PloPreviewRow",
    "build_preview_rows",
    "format_cards",
    "load_pack_and_nodes",
]
