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
import threading
from dataclasses import dataclass
from pathlib import Path

from pipeline.plo.action_history import format_plo_action_history
from pipeline.plo.difficulty import compute_plo_difficulty
from pipeline.plo.fact_extractor import extract_plo_facts
from pipeline.plo.hand_order import HAND_COUNT
from pipeline.plo.node_enumerator import (
    PloDecisionNode,
    enumerate_plo_nodes,
    plo_node_action_context,
    plo_pot_entrant_count,
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


# Serializes the heavy node walk: enumerate_plo_nodes is lru_cached but NOT
# locked, so a page visit racing the background warmer would compute the
# 15-25s walk twice (double CPU + a transient double copy in RAM). The
# second caller waits here, then hits the warm cache instantly.
_ENUM_LOCK = threading.Lock()


def load_pack_and_nodes(pack_dir: Path) -> tuple[PloPack, tuple[PloDecisionNode, ...]]:
    """Discover the PLO pack under ``pack_dir`` and enumerate its nodes."""
    pack = discover_plo_pack(pack_dir)
    with _ENUM_LOCK:
        return pack, enumerate_plo_nodes(pack)


# --- non-blocking pack loading (July 2026 perf fix) --------------------------
# Enumerating the 9-max pack (327k files -> 160k nodes) takes ~15-25s cold,
# and it used to run INSIDE the Generate page's render, freezing the whole
# panel on the first visit after every restart. These helpers move the walk
# to a daemon thread: the page renders immediately (filters editable, seats
# come from the cheap discover_plo_pack), and the result -- nodes plus the
# precomputed filter meta -- appears when ready. INVARIANT: nothing in here
# may touch Streamlit (st.*) from the worker thread; the thread computes,
# the render loop polls via request_pack_load.
PloPackLoad = tuple[
    PloPack,
    tuple[PloDecisionNode, ...],
    tuple[tuple[str, str, int], ...],  # plo_filter_meta triples
]
_BG_LOCK = threading.Lock()
_BG_RESULTS: dict[str, "PloPackLoad | Exception"] = {}
_BG_THREADS: dict[str, threading.Thread] = {}


def _bg_load(key: str, pack_dir: Path) -> None:
    from pipeline.plo.node_enumerator import plo_filter_meta  # noqa: PLC0415

    result: PloPackLoad | Exception
    try:
        pack, nodes = load_pack_and_nodes(pack_dir)
        result = (pack, nodes, plo_filter_meta(nodes))
    except Exception as exc:  # noqa: BLE001 -- surfaced to the page via request_pack_load
        result = exc
    with _BG_LOCK:
        _BG_RESULTS[key] = result
        _BG_THREADS.pop(key, None)


def request_pack_load(pack_dir: Path | str) -> PloPackLoad | None:
    """Non-blocking pack load: the result when ready, else ``None``.

    First call for a pack dir starts a daemon thread and returns ``None``
    immediately; later calls return ``None`` until the walk finishes, then
    the ``(pack, nodes, filter_meta)`` tuple forever after (the underlying
    ``enumerate_plo_nodes`` lru_cache holds the data, so this adds no second
    copy). A failed load raises its exception ONCE and clears the slot, so
    the next call retries (e.g. after the user extracts a missing pack).
    Also the boot-time warmer: call it at panel startup and the pack is
    usually ready before anyone opens a PLO page.
    """
    key = str(pack_dir)
    with _BG_LOCK:
        got = _BG_RESULTS.get(key)
        if isinstance(got, Exception):
            del _BG_RESULTS[key]
            raise got
        if got is not None:
            return got
        if key not in _BG_THREADS:
            worker = threading.Thread(
                target=_bg_load,
                args=(key, Path(pack_dir)),
                daemon=True,
                name=f"plo-pack-load:{key}",
            )
            _BG_THREADS[key] = worker
            worker.start()
    return None


def _first_worthy_spot(
    node: PloDecisionNode,
    rng: random.Random,
    *,
    tries: int = _DEFAULT_WORTHY_TRIES,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    exclude_ambiguous_band: bool = False,
) -> PloSpot | None:
    """A worthy spot at this node, found by random-sampling hands (cheap --
    worthiness reads only the spot, no facts/equity)."""
    for index in rng.sample(range(HAND_COUNT), k=min(tries, HAND_COUNT)):
        spot = sample_plo_spot(node, index)
        if spot.presence >= _MIN_PRESENCE and is_question_worthy(
            spot,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            exclude_ambiguous_band=exclude_ambiguous_band,
        ):
            return spot
    return None


def build_preview_rows(
    pack: PloPack,
    nodes: tuple[PloDecisionNode, ...],
    *,
    count: int = 8,
    seed: int | None = 0,
    hero_positions: list[str] | None = None,
    max_prior_raises: int | None = 2,
    max_active_players: int | None = 3,
    action_contexts: list[str] | None = None,
    player_counts: list[int] | None = None,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    exclude_ambiguous_band: bool = False,
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
        # Entrant counting, matching generate_plo_batch (July 16 2026 fix).
        and (
            max_active_players is None
            or plo_pot_entrant_count(n) <= max_active_players
        )
        and (ctx_set is None or plo_node_action_context(n) in ctx_set)
        and (pc_set is None or plo_pot_entrant_count(n) in pc_set)
    ]
    rng.shuffle(candidates)

    rows: list[PloPreviewRow] = []
    for node in candidates[:max_nodes_scanned]:
        spot = _first_worthy_spot(
            node,
            rng,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            exclude_ambiguous_band=exclude_ambiguous_band,
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
                # INVARIANT (July 2026): history amounts MUST get stack_bb +
                # ante_bb from the pack spec, or MTT/short-stack previews show
                # ante-blind sizes and 100bb jams (the build_solver_data bug).
                action_line=format_plo_action_history(
                    facts, display_in_bb=display_in_bb,
                    stack_bb=pack.spec.stack_bb, ante_bb=pack.spec.ante_bb,
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
