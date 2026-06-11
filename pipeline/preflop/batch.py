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

import hashlib
import json
import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from pipeline.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ExplanationValidationError,
    GeneratedExplanation,
)
from pipeline.preflop.difficulty import (
    compute_difficulty,
)
from pipeline.preflop.ev_engine import (
    compute_break_even_equity,
    compute_ev_gap_bb,
)
from pipeline.preflop.explanation_generator import (
    build_explanation_prompt_parts,
    build_shared_prompt_parts,
    generate_preflop_answer_explanation,
)
from pipeline.preflop.fact_extractor import (
    DEFAULT_EQUITY_RUNOUTS,
    extract_facts,
)
from pipeline.preflop.format_writer import (
    format_preflop_question,
    write_preflop_csv,
)
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
    evaluate_spot,
)
from pipeline.preflop.spot_sampler import (
    PreflopSpot,
    enumerate_spots_for_node,
)

logger = logging.getLogger(__name__)

# Difficulty-score band defaults: the full hard-clamped range the 4-axis
# algorithm can emit (see pipeline.preflop.difficulty). "Mixed" preset =
# this full band; the narrower presets (Easy / Medium / Hard) pass a
# sub-band so the generator keeps only spots whose COMPUTED rating lands
# in the requested tier -- the freq window alone no longer determines
# difficulty now that it's one of four axes.
DIFFICULTY_MIN: int = 400
DIFFICULTY_MAX: int = 3200

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
class PreflopFailure:
    """Full context of one spot that didn't make it to the CSV.

    Built by the batch loop when generation throws. Carries enough data
    that a reviewer can read it and decide: was the LLM wrong, or did
    the validator misfire? In particular:

    * ``question_text`` / ``options`` / ``correct_answer`` -- the
      deterministic inputs the LLM was working from. These are
      identical to what the CSV would have shown if the spot had passed.
    * ``failed_explanation`` -- the prose the LLM actually wrote on its
      last attempt. Empty when no LLM response parsed cleanly (e.g.
      a pre-call ValueError on bad inputs).
    * ``error_message`` -- the validator failure or exception message
      that routed this spot to review.

    Rendered as a one-line summary by ``__str__`` for log compatibility
    with the old ``list[str]`` shape; the admin panel's UI uses the
    structured fields directly.
    """

    node_id: str
    hand_class: str
    hero_position: str
    archetype: str  # may be "" if facts never built (e.g. pre-call error)
    question_text: str
    context_text: str
    options: list[str] = field(default_factory=list)
    correct_answer: str = ""
    failed_explanation: str = ""
    action_frequencies: dict[str, float] = field(default_factory=dict)
    error_message: str = ""

    def __str__(self) -> str:
        """Backward-compat one-line form for logs / old print loops."""
        return f"{self.node_id} / {self.hand_class}: {self.error_message}"


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

    # Per-spot failures with full context (question / options / correct /
    # the LLM's failed explanation / error). Empty list when every
    # attempt succeeded. PreflopFailure.__str__ provides the legacy
    # one-line form when the admin panel's log mode prints them.
    failures: list[PreflopFailure] = field(default_factory=list)

    # How many worthy spots were found before the random-sample step.
    # Useful UI feedback: "found 1,200 worthy spots, sampled 30 of them".
    worthy_spots_available: int = 0

    # How many nodes survived the position + action-context filter.
    # Sanity-check companion to worthy_spots_available.
    nodes_after_filter: int = 0

    # How many worthy spots were rejected by the difficulty-band / min-EV
    # filters (they passed the freq worthiness gate but their computed
    # rating fell outside the requested tier). UI feedback: a narrow
    # preset that starves the batch shows up here.
    difficulty_filtered_out: int = 0

    # How many worthy spots were skipped by the convergence guard --
    # unconverged solver nodes (AA folding preflop / premium-pair inversions),
    # typically the near-zero-reach multiway 5-bet/jam lines. UI feedback.
    noise_filtered_out: int = 0

    # How many questions the caller asked for (the total_questions arg). The
    # run writes fewer when the worthy pool runs out after filtering; the UI
    # compares this to questions_written to explain any shortfall.
    requested_questions: int = 0

    # Token-usage totals across every Anthropic call in the batch.
    # Populated only when the LLM is actually invoked (dry-run leaves
    # these at 0). The admin panel multiplies these by per-model
    # rates to produce a $ cost; the pipeline itself doesn't know
    # about pricing.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    # Model id the batch ran on, e.g. "claude-opus-4-7". Empty for
    # dry-runs or when no LLM calls happened. Stored here (rather
    # than computed per-call) because the whole batch uses one model.
    model_used: str = ""

    # Display name of the prompt this batch ran on (the prompt-workshop
    # label, e.g. "Concise voice v2"). Empty when the caller didn't tag
    # the run. The actual prompt text + sha live in the meta sidecar.
    prompt_name: str = ""
    # Path to the <stem>.meta.json sidecar written next to the CSV --
    # holds the prompt snapshot + per-question inputs for the inspector.
    # None when no rows were produced (no CSV, no meta).
    meta_path: Path | None = None


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


def active_player_count(node: PreflopDecisionNode) -> int:
    """Players still in the pot at this decision (incl. hero).

    Node-level twin of ``concept_tags._non_fold_actor_count`` (keep in
    sync): a position counts only if its LAST action is a non-fold, so a
    player who entered the pot and then folded (opened, then folded to a
    squeeze) is out. Hero always counts. 2 = heads-up, 3 = three-way, etc.
    Used by the player-count filter so the user can ask for, say, only
    3- or 4-way spots instead of the deep bloodbaths.
    """
    last_action: dict[str, PreflopActionType] = {}
    for a in node.history_before:
        last_action[a.position] = a.action_type
    positions = {
        position
        for position, action_type in last_action.items()
        if action_type is not PreflopActionType.FOLD
    }
    positions.add(node.actor)
    return len(positions)


def filter_nodes(
    nodes: Iterable[PreflopDecisionNode],
    *,
    hero_positions: Iterable[str] | None,
    action_contexts: Iterable[str] | None,
    player_counts: Iterable[int] | None = None,
) -> list[PreflopDecisionNode]:
    """Apply the UI position / action-context / player-count filters.

    None / empty filter = "include everything"; mirrors the admin panel
    behavior where an empty multiselect means "no filter on this axis".
    ``player_counts`` keeps only nodes with that many players still in the
    pot (e.g. ``{2, 3}`` for heads-up + three-way only).
    """
    pos_set = set(hero_positions) if hero_positions else None
    ctx_set = set(action_contexts) if action_contexts else None
    count_set = set(player_counts) if player_counts else None
    out: list[PreflopDecisionNode] = []
    for node in nodes:
        if pos_set is not None and node.actor not in pos_set:
            continue
        if ctx_set is not None and node_action_context(node) not in ctx_set:
            continue
        if count_set is not None and active_player_count(node) not in count_set:
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
    exclude_ambiguous_band: bool = False,
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
                exclude_ambiguous_band=exclude_ambiguous_band,
            )
            if evaluation.is_worthy:
                out.append((spot, evaluation))
    return out


# --- convergence guard ------------------------------------------------------
# AA must CONTINUE (call/raise/jam) ~100% of the time preflop in cash (no
# ICM) -- it never folds and is never under-allocated. So an AA continue
# frequency below ~1 marks an unconverged node (the canary). Premium pairs
# must also continue in strength order facing the same action.
_AA_CONTINUE_NOISE_TOL = 0.02
_PREMIUM_MONOTONIC_TOL = 0.10
# Below this PRESENCE (joint reach mass summed across every option, fold
# included) a premium effectively never reaches the node, so its
# conditional strategy is meaningless noise -- skip the check rather than
# divide by ~0. Same idea as the spot sampler's presence filter.
_CANARY_MIN_PRESENCE = 0.005


def node_is_unconverged(node: PreflopDecisionNode) -> bool:
    """True if a node's solver strategy is detectably non-GTO (unconverged).

    These show up on near-zero-reach lines (deep multiway limp/jam pots)
    the solver never converged, and they produce nonsense questions. Two
    clean tells, both computed as CONDITIONAL frequencies (continue mass
    divided by the hand's presence at the node):

    * **AA canary** -- given AA reaches this node at all, it must continue
      ~100% of the time: AA never folds preflop in cash. A conditional
      continue meaningfully below 1 means the node is garbage.
    * **Premium-pair monotonicity** -- facing the same action AA must
      continue at least as often as KK, and KK at least as often as QQ; an
      inversion (a stronger pair continuing LESS) is unconverged.

    The normalisation matters: range files store JOINT weights (reach x
    action), so at any node hero reached by an earlier call/limp/raise,
    AA's raw mass is far below 1 simply because AA mostly took a different
    earlier action. The pre-June-2026 version compared the RAW sum to ~1
    and therefore flagged ~80% of perfectly converged hero-acted-before
    nodes on the 9-max Monker pack (silently over-filtering any
    "After call(s)" generation on both packs). When AA's presence is below
    :data:`_CANARY_MIN_PRESENCE` there is nothing to judge -- the node is
    left to the worthiness/presence filters.

    Deliberately does NOT flag KK/QQ folding on its own -- those can rarely be
    legitimate folds in extreme multiway AA-heavy all-in spots, so they aren't
    a clean signal.
    """
    non_fold = [
        o for o in node.actions if o.action_type is not PreflopActionType.FOLD
    ]
    if not non_fold:
        return False  # nothing but fold -> no continue strategy to check
    from pipeline.preflop_ranges import parse_range_file  # noqa: PLC0415

    # Per-premium: presence (all options) and continue mass (non-fold).
    presence = {"AA": 0.0, "KK": 0.0, "QQ": 0.0}
    cont = {"AA": 0.0, "KK": 0.0, "QQ": 0.0}
    for opt in node.actions:
        try:
            weights = parse_range_file(opt.range_file.path)
        except (OSError, ValueError):
            return False  # unreadable -> don't guess; leave it to other filters
        is_continue = opt.action_type is not PreflopActionType.FOLD
        for hand in presence:
            w = weights.get(hand, 0.0)
            presence[hand] += w
            if is_continue:
                cont[hand] += w

    if presence["AA"] < _CANARY_MIN_PRESENCE:
        return False  # AA effectively never here; nothing to judge
    cond = {
        hand: (cont[hand] / presence[hand] if presence[hand] > 0 else 1.0)
        for hand in presence
    }

    if cond["AA"] < 1.0 - _AA_CONTINUE_NOISE_TOL:
        return True
    return (
        presence["KK"] >= _CANARY_MIN_PRESENCE
        and cond["AA"] < cond["KK"] - _PREMIUM_MONOTONIC_TOL
    ) or (
        presence["KK"] >= _CANARY_MIN_PRESENCE
        and presence["QQ"] >= _CANARY_MIN_PRESENCE
        and cond["KK"] < cond["QQ"] - _PREMIUM_MONOTONIC_TOL
    )


# --- the entry point --------------------------------------------------------
def generate_preflop_batch(
    *,
    pack: PreflopPack,
    output_path: Path | str,
    total_questions: int,
    hero_positions: Iterable[str] | None = None,
    action_contexts: Iterable[str] | None = None,
    player_counts: Iterable[int] | None = None,
    min_frequency: float = MIN_TOP_FREQUENCY,
    max_frequency: float = MAX_TOP_FREQUENCY,
    exclude_ambiguous_band: bool = False,
    min_difficulty: int = DIFFICULTY_MIN,
    max_difficulty: int = DIFFICULTY_MAX,
    min_ev_gap_bb: float | None = None,
    answer_style: str = "auto",
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    game_format: str = "cash",
    display_in_bb: bool = False,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str | None = None,
    prompt_name: str = "",
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
            spot. This is a question-WORTHINESS gate (is the decision
            teachable at all), distinct from the difficulty band below.
        min_difficulty, max_difficulty: Keep only spots whose COMPUTED
            4-axis difficulty rating lands in [min, max]. Defaults to the
            full band (DIFFICULTY_MIN..DIFFICULTY_MAX = no filter). The
            admin panel's Easy/Medium/Hard presets pass sub-bands here.
            The score is computed before the (paid) LLM call so out-of-
            band spots cost no tokens.
        min_ev_gap_bb: Optional quality gate -- drop spots whose EV gap
            to the second-best action is below this (bb). None = off.
            Only applies to spots the EV engine could score (call/fold);
            raise-involved spots (ev_gap None) always pass so the gate
            doesn't wipe out the raise-decision population.
        stakes_bb_dollars: BB size in dollars, forwarded to the row
            builder. Default 0.50 = Tier 1.
        live_or_online, game_format: Cosmetic columns; forwarded.
        display_in_bb: When True, render all amounts (Question prose,
            User Seat / Seats / Pot / Default Stack, Context stack) in
            big blinds instead of dollars, even for a cash game. The
            Cash/Tourney column still reflects game_format. Wired to the
            admin "Display amounts as: Big blinds" toggle.
        model: Anthropic model id for Layer 6. Defaults to the postflop
            production model (Opus 4.7).
        temperature, max_tokens: Layer 6 sampling controls.
        system_prompt: Optional system-prompt text to run this batch on
            (the prompt-workshop UI passes a specific named prompt here).
            None = use the active override-or-default prompt.
        prompt_name: Display label for the prompt, recorded in the meta
            sidecar so outputs can be tagged with which prompt produced
            them. Cosmetic; doesn't affect generation.
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

    # 2. Filter by position + action context + player count.
    filtered_nodes = filter_nodes(
        nodes,
        hero_positions=hero_positions,
        action_contexts=action_contexts,
        player_counts=player_counts,
    )

    # 3. Collect worthy spots (presence + freq window).
    worthy = collect_worthy_spots(
        filtered_nodes,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        exclude_ambiguous_band=exclude_ambiguous_band,
    )

    # 4. Nothing worthy -> empty result.
    if not worthy:
        return BatchResult(
            output_path=None,
            questions_written=0,
            questions_attempted=0,
            failures=[],
            worthy_spots_available=0,
            nodes_after_filter=len(filtered_nodes),
            requested_questions=total_questions,
        )
    # Randomise order so the difficulty-band / EV-gap filter doesn't bias
    # toward whichever nodes happen to enumerate first; we then walk the
    # shuffled pool until total_questions rows are collected.
    rng.shuffle(worthy)

    # 5. Per-spot: extract facts, compute the 4-axis difficulty, apply the
    #    difficulty-band + min-EV-gap filters BEFORE the (paid) LLM call,
    #    then generate. Stop once total_questions rows are collected.
    rows: list[tuple] = []
    # Per-spot prompt inputs (framing / options / correct / solver-data /
    # live block), captured in row order for the meta sidecar + inspector.
    prompt_records: list[dict[str, object]] = []
    failures: list[PreflopFailure] = []
    # Cosmetic progress denominator: we can never produce more rows than
    # there are worthy spots, so cap the bar's total at that upper bound
    # (the difficulty/EV filters may end the run below it).
    progress_total = min(total_questions, len(worthy))
    # Token totals -- summed by ``_record_usage`` which Layer 6 calls
    # once per successful API request (including retries).
    usage_totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

    def _record_usage(
        _model_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation: int,
        cache_read: int,
    ) -> None:
        usage_totals["input"] += input_tokens
        usage_totals["output"] += output_tokens
        usage_totals["cache_creation"] += cache_creation
        usage_totals["cache_read"] += cache_read

    # Tally of spots that passed the worthiness gate but were rejected by
    # the difficulty-band / EV-gap filters (useful UI feedback when a
    # narrow preset starves the batch). ``attempted`` counts only spots
    # we committed an LLM call to (post-filter).
    attempted = 0
    difficulty_filtered_out = 0
    # Spots skipped by the convergence guard (unconverged solver nodes --
    # AA folding preflop / premium-pair inversions). Cached per node so each
    # node is only checked once even though many hands share it.
    noise_filtered_out = 0
    _node_noise: dict[str, bool] = {}
    # ``_evaluation`` carried the pre-facts freq-only difficulty; we
    # discard it and recompute the canonical 4-axis rating below.
    for spot, _evaluation in worthy:
        if len(rows) >= total_questions:
            break
        # --- convergence guard (skip unconverged solver nodes) -----------
        # Near-zero-reach multiway jam lines the solver never converged
        # produce nonsense (AA folding a jam, premium inversions). Skip the
        # whole node before any equity sim / LLM spend. Cached per node.
        nid = spot.node.node_id
        is_bad = _node_noise.get(nid)
        if is_bad is None:
            is_bad = node_is_unconverged(spot.node)
            _node_noise[nid] = is_bad
        if is_bad:
            noise_filtered_out += 1
            continue
        # --- facts + canonical difficulty (no API spend yet) -------------
        # compute_difficulty blends freq + EV gap + archetype/concept +
        # hand class (with EV-weight redistribution when the EV engine
        # couldn't score the spot, typical for raise-involved spots).
        # A failure here is rare (equity / extract issues) but must not
        # abort the batch.
        try:
            facts: object = extract_facts(
                spot, pack, equity_runouts=equity_runouts)
            # Enrich with the pot-odds break-even equity (needs the pack's
            # chip geometry) so Layer 6 can cite it instead of computing
            # pot odds itself. None on spots with no bet to call.
            facts = replace(
                facts, break_even_equity=compute_break_even_equity(facts, pack)
            )
            ev_gap = compute_ev_gap_bb(facts, pack)
            difficulty = compute_difficulty(facts, ev_gap_bb=ev_gap)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                _build_failure(
                    spot, None, [], "", exc,
                    pack=pack, stakes_bb_dollars=stakes_bb_dollars,
                    live_or_online=live_or_online, game_format=game_format,
                )
            )
            logger.warning(
                "batch: spot %s / %s facts/difficulty failed: %s",
                spot.node.node_id, spot.hero_hand_class, exc,
            )
            continue
        # --- difficulty-band + min-EV-gap filter -------------------------
        if not (min_difficulty <= difficulty.score <= max_difficulty):
            difficulty_filtered_out += 1
            continue
        # EV gate only when the EV engine scored the spot; raise-involved
        # spots (ev_gap None) pass unconditionally.
        if (min_ev_gap_bb is not None and ev_gap is not None
                and ev_gap < min_ev_gap_bb):
            difficulty_filtered_out += 1
            continue

        # --- generation (the paid step) ----------------------------------
        attempted += 1
        if progress_callback is not None:
            progress_callback(
                f"Generating question {len(rows) + 1}/{progress_total} "
                f"({spot.node.actor} / {spot.hero_hand_class})",
                len(rows),
                progress_total,
            )
        options: list[str] = []
        correct: str = ""
        try:
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
                    usage_callback=_record_usage,
                    system_prompt=system_prompt,
                )
            rows.append((facts, explanation, difficulty))
            # Record the exact (deterministic) inputs that produced this row,
            # in the same order rows are written to the CSV. Cheap + no API:
            # gold examples are lru-cached and the parts are pure functions.
            prompt_records.append(
                _prompt_record(
                    spot,
                    build_explanation_prompt_parts(
                        facts, options, correct, system_prompt=system_prompt
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Catch-all on purpose: one bad spot must not abort the batch.
            failures.append(
                _build_failure(
                    spot, facts, options, correct, exc,
                    pack=pack, stakes_bb_dollars=stakes_bb_dollars,
                    live_or_online=live_or_online, game_format=game_format,
                )
            )
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
    meta_path: Path | None = None
    if rows:
        written = write_preflop_csv(
            out_path,
            rows,
            pack=pack,
            stakes_bb_dollars=stakes_bb_dollars,
            live_or_online=live_or_online,
            game_format=game_format,
            display_in_bb=display_in_bb,
        )
        final_out = out_path
        # Sidecar metadata: which prompt produced these rows + the exact
        # per-spot inputs, in CSV row order. Powers output->prompt tagging
        # and the prompt inspector. Written for dry-runs too (the inputs
        # are real even when the explanations are placeholders).
        shared = build_shared_prompt_parts(system_prompt=system_prompt)
        meta = _build_batch_meta(
            prompt_name=prompt_name,
            system_prompt=shared["system_prompt"],
            gold_block=shared["gold_block"],
            model=model,
            temperature=temperature,
            seed=random_seed,
            dry_run=dry_run,
            prompt_records=prompt_records,
            pack=pack,
        )
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    return BatchResult(
        output_path=final_out,
        questions_written=written,
        questions_attempted=attempted,
        failures=failures,
        worthy_spots_available=len(worthy),
        nodes_after_filter=len(filtered_nodes),
        difficulty_filtered_out=difficulty_filtered_out,
        noise_filtered_out=noise_filtered_out,
        requested_questions=total_questions,
        total_input_tokens=usage_totals["input"],
        total_output_tokens=usage_totals["output"],
        total_cache_creation_tokens=usage_totals["cache_creation"],
        total_cache_read_tokens=usage_totals["cache_read"],
        # Only record the model id when we actually called it. Dry-run
        # leaves usage at 0 -- model_used "" signals "no LLM ran".
        model_used=model if not dry_run else "",
        prompt_name=prompt_name,
        meta_path=meta_path,
    )


# --- internal helpers -------------------------------------------------------
def _prompt_record(spot: PreflopSpot, parts: dict[str, object]) -> dict[str, object]:
    """One per-question record for the meta sidecar.

    ``parts`` comes from
    :func:`pipeline.preflop.explanation_generator.build_explanation_prompt_parts`.
    We keep only the per-spot VARYING pieces (the shared system prompt +
    gold block are stored once at the batch level) plus the node / hand id
    so the inspector can label each row.
    """
    from pipeline.preflop.app_table_format import _format_user_cards

    return {
        "node_id": spot.node.node_id,
        # user_cards is the Review/inspector join key (matches the CSV's
        # "User Cards" column); hand_class kept for reference. (June 2026:
        # hand_class was dropped from the CSV, so user_cards is now the key.)
        "user_cards": _format_user_cards(spot.hero_card_combo),
        "hand_class": spot.hero_hand_class,
        "framing": parts["framing"],
        "options": parts["options"],
        "correct_answer": parts["correct_answer"],
        "solver_data": parts["solver_data"],
        "live_block": parts["live_block"],
    }


def _build_batch_meta(
    *,
    prompt_name: str,
    system_prompt: str,
    gold_block: str,
    model: str,
    temperature: float,
    seed: int | None,
    dry_run: bool,
    prompt_records: list[dict[str, object]],
    pack: PreflopPack | None = None,
) -> dict[str, object]:
    """Assemble the ``<stem>.meta.json`` payload for one batch.

    Records the prompt SNAPSHOT (name + text + sha) so a later edit or
    rename of that prompt can't make this batch's provenance ambiguous,
    the run settings (model / temperature / seed), the source pack (so
    Review/Compare can tell a 9-max batch from a 6-max one), and the
    per-question inputs in CSV row order.
    """
    return {
        "prompt_name": prompt_name,
        "prompt_sha": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "prompt_text": system_prompt,
        "gold_block": gold_block,
        "model": "" if dry_run else model,
        "temperature": temperature,
        "seed": seed,
        "dry_run": dry_run,
        "pack_id": pack.pack_id if pack else "",
        "table_size": pack.table_size if pack else None,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "questions": prompt_records,
    }


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


def _build_failure(
    spot: PreflopSpot,
    facts: object | None,
    options: list[str],
    correct: str,
    exc: BaseException,
    *,
    pack: PreflopPack,
    stakes_bb_dollars: float,
    live_or_online: str,
    game_format: str,
) -> PreflopFailure:
    """Build a PreflopFailure with as much context as available.

    Some failure modes fire BEFORE facts are extracted (e.g. equity
    sampler bugs) -- in those cases we have only ``spot`` and the
    exception. Other failures fire AFTER an LLM call -- the
    :class:`ExplanationValidationError` carries the last-attempt text
    + parsed candidate, which we pull into the failure record so the
    reviewer can see what the LLM actually wrote.

    Renders the question text best-effort (skips on extract_facts
    failures since the renderer needs facts).
    """
    # Lazy import to avoid a cycle with format_writer at module load.
    question_text = ""
    context_text = ""
    archetype = ""
    action_frequencies: dict[str, float] = {}
    if facts is not None:
        try:
            question_text = format_preflop_question(
                facts,
                pack=pack,
                stakes_bb_dollars=stakes_bb_dollars,
                live_or_online=live_or_online,
                game_format=game_format,
            )
        except Exception:  # noqa: BLE001
            question_text = "(question render failed)"
        archetype = getattr(facts, "archetype", "") or ""
        action_frequencies = dict(
            getattr(getattr(facts, "spot", None), "action_frequencies", {})
            or {}
        )

    # If the exception is an ExplanationValidationError we attached
    # context on (the last LLM attempt's parsed candidate), pull it
    # for the failed_explanation field.
    failed_explanation = ""
    if isinstance(exc, ExplanationValidationError):
        candidate = getattr(exc, "last_attempt_candidate", None)
        if candidate is not None:
            failed_explanation = candidate.answer_explanation or ""
        elif getattr(exc, "last_attempt_text", ""):
            # Parsing failed before we got a candidate -- show the raw
            # text so reviewers can see what came out of the model.
            failed_explanation = (
                f"(raw text, did not parse) {exc.last_attempt_text}"
            )

    return PreflopFailure(
        node_id=spot.node.node_id,
        hand_class=spot.hero_hand_class,
        hero_position=spot.node.actor,
        archetype=archetype,
        question_text=question_text,
        context_text=context_text,
        options=list(options),
        correct_answer=correct,
        failed_explanation=failed_explanation,
        action_frequencies=action_frequencies,
        error_message=str(exc),
    )


__all__ = [
    "ACTION_CONTEXTS",
    "BatchResult",
    "PreflopFailure",
    "ProgressCallback",
    "collect_worthy_spots",
    "filter_nodes",
    "generate_preflop_batch",
    "node_action_context",
]
