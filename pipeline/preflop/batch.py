"""End-to-end preflop batch orchestrator.

Glues the eight pipeline layers together for the preflop path:

  discover_packs -> enumerate_nodes
                -> filter (hero position + action context)
                -> enumerate spots (per node x 169 hand classes)
                -> evaluate_spot (worthiness gate: presence + freq window)
                -> sample N spots
                -> extract_facts (per-spot: equity, villain stats, archetype)
                -> generate_preflop_explanation (Layer 6: LLM)
                -> build_preflop_row + write_preflop_csv (Layer 8)

The public entry point is :func:`generate_preflop_batch`. It takes the
choices the admin panel (or a CLI) exposes -- pack, position filters,
action-context filters, frequency window, total question count, stake
config, model, output path -- and returns a :class:`BatchResult` with
stats and any per-spot failures.

Design notes:

* **One bad spot doesn't kill the batch.** Per-spot exceptions are
  caught and recorded in :attr:`BatchResult.failures`; the orchestrator
  proceeds to the next spot. Per the brief: "<15% should still need
  human review after one regen", but the FIRST review batch surfaces
  the failure modes that drive Phase B's validator stack -- so we want
  to see every failure rather than crash on it.

* **Dry-run support.** ``dry_run=True`` skips the LLM call and emits a
  placeholder explanation per spot, so the orchestrator can be
  exercised without burning API tokens. The CSV produced under
  dry-run is structurally valid but the option strings / answer
  explanation are placeholders.

* **Progress callback.** Optional ``progress_callback`` is invoked
  before each spot's LLM call with ``(message, current_index,
  total)``. The admin panel uses it to update a Streamlit progress
  bar; CLI callers can pass a tqdm-shaped callable or None.

* **Deterministic sampling.** Pass ``random_seed`` for reproducible
  batches (tests, debugging). Default None = nondeterministic.

* **No solver dependency.** The whole orchestrator runs on the
  preflop range pack alone -- no PioSolver Edge / .cfr file needed.
  This is what unblocks generation on the Mac development box.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    GeneratedExplanation,
)
from pipeline.preflop.ev_engine import compute_ev_gap_bb
from pipeline.preflop.explanation_generator import (
    generate_preflop_answer_explanation,
)
from pipeline.preflop.fact_extractor import (
    DEFAULT_EQUITY_RUNOUTS,
    extract_facts,
)
from pipeline.preflop.format_writer import write_preflop_csv
from pipeline.preflop.grammars.types import PreflopActionType
from pipeline.preflop.node_enumerator import (
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.options import build_options
from pipeline.preflop.pack import PreflopPack
from pipeline.preflop.question_extractor import (
    MAX_TOP_FREQUENCY,
    MIN_PRESENCE,
    MIN_TOP_FREQUENCY,
    PreflopQuestionEvaluation,
    difficulty_score,
    evaluate_spot,
)
from pipeline.preflop.spot_sampler import (
    PreflopSpot,
    enumerate_spots_for_node,
)

logger = logging.getLogger(__name__)

# Action-context labels, matching the admin panel's filter dropdown. The
# orchestrator and the UI must share the same vocabulary so the UI
# multiselect strings round-trip cleanly to the filter logic.
ACTION_CONTEXTS: tuple[str, ...] = (
    "Opening",
    "Facing single raise",
    "Facing 3-bet",
    "Facing 4-bet+",
    "After call(s)",
)


# --- progress callback signature --------------------------------------------
# Callable[[message: str, current_index: int, total: int], None]. The admin
# panel passes a Streamlit-shaped callback that updates a progress bar +
# status text; CLI callers can pass tqdm-shaped functions or None.
ProgressCallback = Callable[[str, int, int], None]


# --- result dataclass -------------------------------------------------------
@dataclass(frozen=True)
class BatchResult:
    """Outcome of one preflop batch generation run."""

    # Where the CSV was written (None if no rows were produced).
    output_path: Path | None

    # How many questions actually made it to the CSV. <= questions_attempted.
    questions_written: int

    # How many spots were attempted (sampled + sent to Layer 6). Failures
    # count here too -- they just don't make it to the CSV.
    questions_attempted: int

    # Per-spot error messages. Each entry is a single-line string of the
    # form "<node_id> / <hand_class>: <error>". Empty list when every
    # attempt succeeded.
    failures: list[str] = field(default_factory=list)

    # How many worthy spots were found before the random-sample step.
    # Useful UI feedback: "found 1,200 worthy spots, sampled 30 of them".
    worthy_spots_available: int = 0

    # How many nodes survived the position + action-context filter.
    # Sanity-check companion to worthy_spots_available.
    nodes_after_filter: int = 0


# --- node filtering ---------------------------------------------------------
def node_action_context(node: PreflopDecisionNode) -> str:
    """Categorize one node by the action hero is facing.

    Public so the admin panel and the orchestrator share one source of
    truth. Returns one of :data:`ACTION_CONTEXTS`.

      * "Opening"             -- no prior raise in history
      * "After call(s)"       -- at least one prior raise AND at least
                                  one call afterwards (squeeze spot)
      * "Facing single raise" -- exactly one prior raise
      * "Facing 3-bet"        -- exactly two prior raises
      * "Facing 4-bet+"       -- three or more prior raises
    """
    n_raises = sum(
        1
        for a in node.history_before
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )
    n_calls = sum(
        1 for a in node.history_before if a.action_type is PreflopActionType.CALL
    )
    if n_raises == 0:
        return "Opening"
    if n_calls > 0:
        return "After call(s)"
    if n_raises == 1:
        return "Facing single raise"
    if n_raises == 2:
        return "Facing 3-bet"
    return "Facing 4-bet+"


def filter_nodes(
    nodes: Iterable[PreflopDecisionNode],
    *,
    hero_positions: Iterable[str] | None,
    action_contexts: Iterable[str] | None,
) -> list[PreflopDecisionNode]:
    """Apply the UI position + action-context filters to a node iterable.

    None / empty filter = "include everything"; mirrors the admin panel
    behavior where an empty multiselect means "no filter on this axis".
    """
    pos_set = set(hero_positions) if hero_positions else None
    ctx_set = set(action_contexts) if action_contexts else None
    out: list[PreflopDecisionNode] = []
    for node in nodes:
        if pos_set is not None and node.actor not in pos_set:
            continue
        if ctx_set is not None and node_action_context(node) not in ctx_set:
            continue
        out.append(node)
    return out


# --- worthy-spot collection -------------------------------------------------
def collect_worthy_spots(
    nodes: Iterable[PreflopDecisionNode],
    *,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    min_presence: float = MIN_PRESENCE,
) -> list[tuple[PreflopSpot, PreflopQuestionEvaluation]]:
    """Enumerate every (node, hand class) spot and keep the worthy ones.

    A spot is "worthy" when ``evaluate_spot`` says so -- i.e. the hand
    actually reaches the node (presence filter) AND the dominant
    action's frequency lies inside the configured window. Trivial spots
    (100% one action) are filtered out by the max_frequency ceiling.

    Returns a list of ``(spot, evaluation)`` pairs so callers can keep
    the precomputed difficulty score without re-running ``evaluate_spot``
    later.
    """
    out: list[tuple[PreflopSpot, PreflopQuestionEvaluation]] = []
    for node in nodes:
        for spot in enumerate_spots_for_node(node, min_total_weight=min_presence):
            evaluation = evaluate_spot(
                spot,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                min_presence=min_presence,
            )
            if evaluation.is_worthy:
                out.append((spot, evaluation))
    return out


# --- the entry point --------------------------------------------------------
def generate_preflop_batch(
    *,
    pack: PreflopPack,
    output_path: Path | str,
    total_questions: int,
    hero_positions: Iterable[str] | None = None,
    action_contexts: Iterable[str] | None = None,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    answer_style: str = "auto",
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    equity_runouts: int = DEFAULT_EQUITY_RUNOUTS,
    dry_run: bool = False,
    client: object | None = None,
    progress_callback: ProgressCallback | None = None,
    random_seed: int | None = None,
) -> BatchResult:
    """Generate a batch of preflop questions and write them to a CSV.

    Args:
        pack: The preflop range pack to draw spots from.
        output_path: Where to write the resulting CSV. Parent dirs are
            created if missing. Pass None-equivalent (empty string) is
            not supported; the orchestrator always writes a file when
            at least one spot succeeds.
        total_questions: Target batch size. The orchestrator samples up
            to this many worthy spots; if fewer worthy spots exist after
            filters, it produces whatever it has.
        hero_positions: Optional whitelist of hero seats (UTG, HJ, CO,
            BTN, SB, BB). None / empty = all positions.
        action_contexts: Optional whitelist of action contexts
            (see :data:`ACTION_CONTEXTS`). None / empty = all contexts.
        min_frequency, max_frequency: Frequency window for the
            worthiness filter. Defaults to the brief's 55-95% sweet
            spot.
        stakes_bb_dollars: BB size in dollars, forwarded to the row
            builder. Default 0.50 = Tier 1.
        live_or_online, game_format: Cosmetic columns; forwarded.
        model: Anthropic model id for Layer 6. Defaults to the postflop
            production model (Opus 4.7).
        temperature, max_tokens: Layer 6 sampling controls.
        equity_runouts: Per-villain-combo equity sample count for
            ``extract_facts``. Default 200. Lower for faster batches at
            lower equity-number precision.
        dry_run: When True, skip the Anthropic API call and emit a
            placeholder explanation per spot. CSV is still produced.
        client: Optional Anthropic-shaped client (tests + non-default
            authentication). When None, Layer 6 builds one from
            ``ANTHROPIC_API_KEY`` lazily.
        progress_callback: Optional progress reporter. Called with
            ``(message, current_index, total)`` before each LLM call.
        random_seed: Optional RNG seed for reproducible sampling.

    Returns:
        BatchResult with the output path, success/attempt counts,
        per-spot failure messages, and intermediate-stage counts.
    """
    out_path = Path(output_path)
    rng = random.Random(random_seed)

    # 1. Enumerate the pack's nodes once.
    nodes = enumerate_nodes([pack])

    # 2. Filter by position + action context.
    filtered_nodes = filter_nodes(
        nodes,
        hero_positions=hero_positions,
        action_contexts=action_contexts,
    )

    # 3. Collect worthy spots (presence + freq window).
    worthy = collect_worthy_spots(
        filtered_nodes,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )

    # 4. Sample up to total_questions.
    sample_size = min(total_questions, len(worthy))
    if sample_size <= 0:
        return BatchResult(
            output_path=None,
            questions_written=0,
            questions_attempted=0,
            failures=[],
            worthy_spots_available=len(worthy),
            nodes_after_filter=len(filtered_nodes),
        )
    sampled = rng.sample(worthy, sample_size)

    # 5. Per-spot: extract facts, generate explanation, collect row triples.
    rows: list[tuple] = []
    failures: list[str] = []
    # ``_evaluation`` carried the pre-facts freq-only difficulty; we
    # discard it now and recompute below with EV gap once facts are
    # available. Kept in the tuple so collect_worthy_spots's return
    # shape stays stable.
    for index, (spot, _evaluation) in enumerate(sampled):
        if progress_callback is not None:
            progress_callback(
                f"Generating question {index + 1}/{sample_size} "
                f"({spot.node.actor} / {spot.hero_hand_class})",
                index,
                sample_size,
            )
        try:
            facts = extract_facts(spot, pack, equity_runouts=equity_runouts)
            # Deterministic option selection -- pure Python, no LLM.
            options, correct = build_options(facts, style=answer_style)
            if dry_run:
                explanation = _placeholder_explanation(options, correct)
            else:
                explanation = generate_preflop_answer_explanation(
                    facts,
                    options,
                    correct,
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            # Re-score difficulty with the EV gap now that facts are
            # available. evaluation.difficulty_score from the cheap
            # pre-facts filter was freq-only; this refines it.
            # compute_ev_gap_bb returns None when raises are in the
            # top-2 (v1 limitation) -- difficulty_score handles that
            # fallback to freq-only internally.
            ev_gap = compute_ev_gap_bb(facts, pack)
            enriched_difficulty = difficulty_score(spot, ev_gap_bb=ev_gap)
            rows.append((facts, explanation, enriched_difficulty))
        except Exception as exc:  # noqa: BLE001
            # Catch-all on purpose: one bad spot must not abort the batch.
            failures.append(f"{spot.node.node_id} / {spot.hero_hand_class}: {exc}")
            logger.warning(
                "batch: spot %s / %s failed: %s",
                spot.node.node_id,
                spot.hero_hand_class,
                exc,
            )

    # 6. Write CSV if we produced anything. write_preflop_csv creates
    # parent dirs as needed.
    written = 0
    final_out: Path | None = None
    if rows:
        written = write_preflop_csv(
            out_path,
            rows,
            pack=pack,
            stakes_bb_dollars=stakes_bb_dollars,
            live_or_online=live_or_online,
            game_format=game_format,
        )
        final_out = out_path

    return BatchResult(
        output_path=final_out,
        questions_written=written,
        questions_attempted=sample_size,
        failures=failures,
        worthy_spots_available=len(worthy),
        nodes_after_filter=len(filtered_nodes),
    )


# --- internal helpers -------------------------------------------------------
def _placeholder_explanation(
    options: list[str],
    correct_answer: str,
) -> GeneratedExplanation:
    """A stub GeneratedExplanation for dry-run mode.

    Uses the deterministic option set + correct_answer (so the row is
    structurally consistent with whatever ``answer_style`` the caller
    chose). Only the answer_explanation prose is a placeholder; the
    options + correct answer are real.
    """
    padded = (list(options) + ["", "", "", ""])[:4]
    return GeneratedExplanation(
        option_1=padded[0],
        option_2=padded[1],
        option_3=padded[2],
        option_4=padded[3],
        correct_answer=correct_answer,
        answer_explanation="[dry-run placeholder; rerun without dry-run for real prose]",
    )


__all__ = [
    "ACTION_CONTEXTS",
    "BatchResult",
    "ProgressCallback",
    "collect_worthy_spots",
    "filter_nodes",
    "generate_preflop_batch",
    "node_action_context",
]
