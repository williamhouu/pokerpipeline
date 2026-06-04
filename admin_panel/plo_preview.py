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

from pipeline.plo.difficulty import compute_plo_difficulty
from pipeline.plo.fact_extractor import extract_plo_facts
from pipeline.plo.hand_order import HAND_COUNT
from pipeline.plo.node_enumerator import PloDecisionNode, enumerate_plo_nodes
from pipeline.plo.options import build_options
from pipeline.plo.pack import PloActionType, PloPack, discover_plo_pack
from pipeline.plo.position import hero_relative_position
from pipeline.plo.question_extractor import is_question_worthy
from pipeline.plo.skill_tagger import compute_plo_skills
from pipeline.plo.spot_sampler import PloSpot, sample_plo_spot
from pipeline.plo.spot_tags import compute_plo_concept_tags

_SUIT_EMOJI = {"s": "♠️", "h": "❤️", "d": "♦️", "c": "♣️"}
_SEAT_FULL = {
    "LJ": "Lojack",
    "HJ": "Hijack",
    "CO": "Cutoff",
    "BU": "Button",
    "SB": "Small Blind",
    "BB": "Big Blind",
}
_LEVEL_VERB = {1: "opens", 2: "3-bets", 3: "4-bets", 4: "5-bets"}
_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
_MIN_PRESENCE = 0.5  # a real spot the hero actually reaches with this hand
_DEFAULT_WORTHY_TRIES = 600  # random hands to try per node when hunting


def format_cards(cards: tuple[str, ...]) -> str:
    """``('As','Ks','Ah','Kh')`` -> ``'A♠️ K♠️ A❤️ K❤️'``."""
    return " ".join(c[0] + _SUIT_EMOJI[c[1]] for c in cards)


def action_line(node: PloDecisionNode) -> str:
    """A readable one-line action history for the node (folds dropped)."""
    parts: list[str] = []
    level = 0
    for a in node.history_before:
        if a.action is PloActionType.FOLD:
            continue
        if a.action in _AGGRESSIVE:
            level += 1
            verb = "jams" if a.action is PloActionType.ALL_IN else _LEVEL_VERB.get(level, "raises")
        else:
            verb = "calls"
        parts.append(f"the {_SEAT_FULL[a.seat]} {verb}")
    seat = _SEAT_FULL[node.actor]
    if not parts:
        return f"You're first to act in the {seat}."
    history = ", ".join(parts)
    return f"{history[0].upper()}{history[1:]} — you're in the {seat}."


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
    node: PloDecisionNode, rng: random.Random, *, tries: int = _DEFAULT_WORTHY_TRIES
) -> PloSpot | None:
    """A worthy spot at this node, found by random-sampling hands (cheap --
    worthiness reads only the spot, no facts/equity)."""
    for index in rng.sample(range(HAND_COUNT), k=min(tries, HAND_COUNT)):
        spot = sample_plo_spot(node, index)
        if spot.presence >= _MIN_PRESENCE and is_question_worthy(spot):
            return spot
    return None


def _active_players(node: PloDecisionNode) -> int:
    """Players still in the pot at the node (unique non-folders + the actor)."""
    seats = {a.seat for a in node.history_before if a.action is not PloActionType.FOLD}
    seats.add(node.actor)
    return len(seats)


def build_preview_rows(
    pack: PloPack,
    nodes: tuple[PloDecisionNode, ...],
    *,
    count: int = 8,
    seed: int = 0,
    hero_positions: list[str] | None = None,
    max_prior_raises: int | None = 2,
    max_active_players: int | None = 3,
    compute_equity: bool = False,
    answer_style: str = "auto",
    max_nodes_scanned: int = 80,
) -> list[PloPreviewRow]:
    """Sample up to ``count`` worthy spots and run them through the pipeline.

    Picks one worthy spot per node (for variety) across a shuffled, filtered
    node set. Equity is only computed for the kept spots, and only when
    ``compute_equity`` is True (the one ~1s/spot cost).

    The default filters (``max_prior_raises=2``, ``max_active_players=3``) keep
    the preview on the verified-clean lines -- opens, single-raised pots, and
    heads-up/3-way 3-bet pots. Monker's deep multiway 4-bet+/jam tail is largely
    UNCONVERGED (absurd EV gaps, inverted ranges), so it is excluded by default;
    pass ``None`` for both to include it (clearly a noisier sample).
    """
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

    rows: list[PloPreviewRow] = []
    for node in candidates[:max_nodes_scanned]:
        spot = _first_worthy_spot(node, rng)
        if spot is None:
            continue
        facts = extract_plo_facts(
            spot, pack, compute_equity=compute_equity, rng=random.Random(seed)
        )
        options, correct = build_options(facts, style=answer_style)
        difficulty = compute_plo_difficulty(facts)
        rows.append(
            PloPreviewRow(
                hand_label=spot.hero_label,
                cards=format_cards(spot.hero_cards),
                position=node.actor,
                relative_position=hero_relative_position(facts),
                action_line=action_line(node),
                options=tuple(options),
                correct_answer=correct,
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
    "action_line",
    "build_preview_rows",
    "format_cards",
    "load_pack_and_nodes",
]
