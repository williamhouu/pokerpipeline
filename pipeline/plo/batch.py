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

import json
import logging
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from pipeline.explanation_generator import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ExplanationValidationError,
)
from pipeline.plo.action_history import fold_ev_consistency_issue
from pipeline.plo.balanced_select import (
    BalanceAttrs,
    answer_verb,
    balance_report,
    balanced_order,
    difficulty_band,
    hand_shape_family,
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
    PLO_ACTION_CONTEXTS,
    PloDecisionNode,
    enumerate_plo_nodes,
    plo_pot_entrant_count,
    plo_node_action_context,
)
from pipeline.plo.options import build_options
from pipeline.plo.pack import PloActionType, PloPack
from pipeline.plo.position import position_bucket
from pipeline.plo.question_extractor import (
    MAX_TOP_FREQUENCY,
    MIN_TOP_FREQUENCY,
    is_question_worthy,
)
from pipeline.plo.reviser import revise_plo_explanation
from pipeline.plo.validators import run_plo_soft_validators
from pipeline.plo.spot_sampler import (
    PloSpot,
    sample_plo_spot,
    strip_artifact_allins,
)

logger = logging.getLogger(__name__)

_AGGRESSIVE = {PloActionType.RAISE, PloActionType.MIN_RAISE, PloActionType.ALL_IN}
_MIN_PRESENCE = 0.5
# Abort the batch after this many explanation failures IN A ROW (resets on
# success): occasional failures backfill, a systemic one must not burn spend
# across the whole spot pool.
_MAX_CONSECUTIVE_FAILURES = 5
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
    # Human-readable reason per failed explanation (e.g. the validation error),
    # so the UI can show WHY a row shipped blank instead of just a count.
    explanation_failure_reasons: tuple[str, ...] = ()
    # ARTIFACT-STRIP (July 2026): deep-stack spots skipped because their real
    # strategy mixes the artifact All-in at >= 5% (never askable).
    artifact_material_spots_skipped: int = 0
    # The .meta.json sidecar written next to the CSV (July 2026) -- the batch
    # re-verifier's input.
    meta_path: Path | None = None
    # True when a graceful stop ended the batch before total_questions (the
    # CSV/meta contain everything committed; the shortfall is intentional).
    stopped_early: bool = False
    # Token spend across EVERY LLM pass of this batch (generation + the
    # Layer-7 gate/reviser/final audit), accumulated internally so the totals
    # survive a process boundary (July 2026: PLO generation runs as a
    # background subprocess job; the admin's spend logger reads these off the
    # pickled result, like the NLHE BatchResult). "" model = no LLM ran.
    model_used: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

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
    exclude_ambiguous_band: bool = False,
    exclude_indices: set[int] | frozenset[int] = frozenset(),
    allow_allin_answers: bool = True,
    counters: dict[str, int] | None = None,
) -> PloSpot | None:
    """A worthy spot at this node, skipping hands already drawn this batch
    (``exclude_indices``) so repeat visits to a node yield a NEW hand.

    ``allow_allin_answers=False`` (deep stacks, where the pack's All-in
    branches are tree artifacts) routes every sampled hand through the
    ARTIFACT-STRIP (July 2026, :func:`pipeline.plo.spot_sampler.
    strip_artifact_allins`): trace jam mass is stripped + renormalised
    BEFORE the worthiness window, so the stripped mix drives worthiness,
    options, qualifiers, EV gap, and difficulty; a MATERIAL jam mix (>= 5%:
    the solver genuinely wants a line we refuse to show) is skipped outright
    and counted into ``counters["artifact_material_spots_skipped"]`` when a
    dict is given. Subsumes the old dominant-is-all-in check."""
    for index in rng.sample(range(HAND_COUNT), k=min(_WORTHY_TRIES, HAND_COUNT)):
        if index in exclude_indices:
            continue
        spot = sample_plo_spot(node, index)
        if not allow_allin_answers:
            spot = strip_artifact_allins(spot)
            if spot.artifact_material:
                if counters is not None:
                    counters["artifact_material_spots_skipped"] = (
                        counters.get("artifact_material_spots_skipped", 0) + 1
                    )
                continue
        if spot.presence >= _MIN_PRESENCE and is_question_worthy(
            spot,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            exclude_ambiguous_band=exclude_ambiguous_band,
        ):
            return spot
    return None


def _balance_attrs_for(
    spot: PloSpot, facts: Any, difficulty_score: float, pack: PloPack
) -> BalanceAttrs:
    """The five balance-axis values for one spot (see balanced_select)."""
    hand_class = facts.hand_class
    return BalanceAttrs(
        difficulty_band=difficulty_band(difficulty_score),
        action_context=plo_node_action_context(spot.node),
        answer_verb=answer_verb(spot.dominant_action),
        position=position_bucket(spot.node.actor, table_size=pack.table_size),
        hand_shape=hand_shape_family(
            hand_class.pair_pattern, hand_class.suit_pattern
        ),
        node_id=spot.node.node_id,
    )


def _gate_check_best_of(
    explanation_text: str,
    solver_data: dict[str, Any],
    client: Any,
    *,
    model: str,
    system_prompt: str | None,
    usage_callback: UsageCallback | None,
    passes: int,
) -> list[dict[str, str]]:
    """Run the claim checker ``passes`` times and UNION the issues.

    The checker is non-deterministic even at temperature 0, so a single pass
    can miss a real issue and hand a flagged question a "lucky clean" -- the
    same reason the NLHE revise gate runs best-of-2. Issues are deduped by
    claim text; each pass fails open (an errored call contributes nothing).
    """
    from pipeline.plo.claim_checker import (  # noqa: PLC0415
        PLO_CHECKER_SYSTEM_PROMPT,
        check_plo_explanation_claims,
    )

    merged: dict[str, dict[str, str]] = {}
    for _ in range(passes):
        try:
            result = check_plo_explanation_claims(
                explanation_text,
                solver_data,
                client,
                model=model,
                system_prompt=system_prompt or PLO_CHECKER_SYSTEM_PROMPT,
                usage_callback=usage_callback,
            )
        except Exception as exc:  # noqa: BLE001 - the gate fails open
            logger.warning("plo claim gate call failed (fails open): %s", exc)
            continue
        for issue in result.issues:
            key = issue.claim.strip().lower() or issue.problem.strip().lower()
            merged.setdefault(key, {"claim": issue.claim, "problem": issue.problem})
    return list(merged.values())


# Best-of-N gate passes when the auto-fix is deciding whether to rewrite
# (mirrors pipeline.preflop/postflop; the flag-only path stays one call).
_REVISE_GATE_PASSES = 2


def _issues_json(issue_dicts: list[dict[str, str]]) -> str:
    """Serialize issue dicts for the ``claim_check`` column ("[]" = clean)."""
    return json.dumps(
        [{"claim": d.get("claim", ""), "problem": d.get("problem", "")}
         for d in issue_dicts],
        separators=(",", ":"),
    )


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
    exclude_ambiguous_band: bool = False,
    min_ev_gap_bb: float | None = None,
    diversify: bool = False,
    balanced: bool = False,
    compute_equity: bool = True,
    answer_style: str = "auto",
    seed: int | None = 0,
    stakes_bb_dollars: float = 1.0,
    game_format: str | None = None,
    display_in_bb: bool = False,
    stack_bb: float | None = None,
    pack_label: str | None = None,
    min_difficulty: int = 0,
    max_difficulty: int = 10_000,
    generate_explanations: bool = False,
    explanation_client: Any = None,
    explanation_model: str = DEFAULT_MODEL,
    explanation_temperature: float = DEFAULT_TEMPERATURE,
    explanation_system_prompt: str | None = None,
    explanation_include_skills: bool = False,
    run_claim_checker: bool = False,
    revise_pass: bool = False,
    final_audit: bool = False,
    claim_checker_prompt: str | None = None,
    usage_callback: UsageCallback | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
    llm_workers: int = 1,
) -> PloBatchResult:
    """Generate up to ``total_questions`` PLO question rows and write the CSV.

    Spots are drawn round-robin across a shuffled, filtered node set: one new
    hand per node per pass, so situations spread first and a node contributes
    multiple (always different) hands only when the batch is bigger than the
    node pool.

    ``balanced=True`` (🎛️ Fully balanced, July 2026) pre-draws a capped
    candidate pool, scores it cheaply, and greedy-orders it for marginal
    balance across difficulty band, situation, correct-answer verb
    (fold/call/raise -- read from the dominant action, so basic and GTO
    option styles balance identically), position, and hand shape; achieved
    vs target lands in the meta ``balance_report``. Supersedes
    ``diversify`` (situation is one of its axes).
    Equity is computed per kept spot when ``compute_equity`` (the ~1s/spot cost;
    it only enriches the equity/range concept tags).

    Node filters (all optional, all AND-combined): ``hero_positions``,
    ``action_contexts`` (the :data:`PLO_ACTION_CONTEXTS` buckets), and
    ``player_counts`` mirror the NLHE Generate page; ``max_prior_raises`` /
    ``max_active_players`` are the coarse clean-line caps. Spot filters:
    ``min_frequency`` / ``max_frequency`` set the worthiness window, and
    ``exclude_ambiguous_band`` punches a hole at 90-95% within it (the
    "mostly-feels-like-always" trap) without capping the ceiling, so a 100%
    max still admits the pure 95-100% spots. ``min_ev_gap_bb`` drops
    near-coinflip spots (reported in ``ev_gap_filtered_out``).

    ``seed`` controls spot sampling: an int reproduces the identical batch
    (same nodes, hands, order); ``None`` seeds from fresh OS entropy so every
    run draws different spots -- the admin Generate page passes ``None``
    unless the user pins a test set.

    When ``generate_explanations`` is True, Layer 6 (the LLM) fills the
    ``Answer Explanation`` column per spot; otherwise it is left blank (the
    deterministic path, no API key needed). A failed explanation DROPS the
    question from the CSV (a blank-explanation row is not shippable) and is
    counted in ``explanations_failed`` with its reason; the round-robin keeps
    drawing, so the batch backfills toward ``total_questions`` with other
    spots. ``explanation_system_prompt`` (when set) overrides the built-in
    Layer 6 system prompt verbatim -- the admin prompt library + Compare page
    pass an edited prompt here.

    ``llm_workers`` (⚡ July 2026, user ask): run up to N questions' LLM
    chains CONCURRENTLY (generation + gate + reviser + final audit are one
    sequential chain PER question, but different questions are independent
    and the time is ~all API latency, so N workers cut wall-clock ~N-fold at
    identical cost). Everything deterministic stays in the MAIN thread in
    draw order -- facts, gates, difficulty, row building, the incremental
    commit -- and results are committed strictly in submission order, so the
    CSV/meta row order and every deterministic column are IDENTICAL to a
    sequential run and the byte-exact re-verifier is unaffected. 1 = the
    classic sequential path, byte-for-byte unchanged.
    """
    # Resolve a concrete seed up front (July 2026): ``None`` used to mean
    # "irreproducible OS entropy", which made a batch impossible to re-verify.
    # Now None draws a fresh random seed ONCE and records it in the meta
    # sidecar, so every batch stays different run-to-run AND byte-exactly
    # rebuildable by scripts/audit_plo_batch.py. An explicit int behaves as
    # before (pinned test sets).
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    # Stack depth drives the pot-limit walk, the all-in-realism gate
    # (allins_ok <= 40bb), and the Context. Default it to the PACK's registered
    # depth (July 2026, short-stack packs): the 100bb/9-max packs are 100bb so
    # this is a no-op for them, but a 12bb/20bb pack MUST generate at its own
    # depth or the geometry (and the artifact-jam gate) would be wrong. An
    # explicit caller value still wins.
    if stack_bb is None:
        stack_bb = pack.spec.stack_bb
    # Tournament framing + the bb ante come from the pack registration (July
    # 2026, MTT packs): the ante is load-bearing for EVERY pot number (prose,
    # price, animation, app tokens), so it can never be a caller guess. An
    # explicit game_format still wins (tests / the Compare page).
    if game_format is None:
        game_format = pack.spec.game_format
    ante_bb = pack.spec.ante_bb
    # Which pack this batch generates from: the registered pack_id unless the
    # caller overrides the label (legacy callers passed a literal string; the
    # id and the label were the same value for the original 6-max pack).
    if pack_label is None:
        pack_label = pack.pack_id
    # Resolve ONE shared client up front for every LLM pass (generation, the
    # claim-check gate, the reviser, the final audit). The reviser treats
    # ``None`` as a no-op by contract, so leaving this to each callee's lazy
    # creation would silently disable the auto-fix -- the exact revise-pass
    # client=None bug NLHE shipped once (June 2026). One client also avoids
    # re-authing per call.
    if generate_explanations and explanation_client is None:
        import os  # noqa: PLC0415

        from anthropic import Anthropic  # noqa: PLC0415

        explanation_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # Accumulate token spend INTERNALLY (July 2026) so the result carries the
    # batch's totals across a process boundary (background jobs). Every LLM
    # call site already reports through usage_callback (THE USAGE RULE), so
    # wrapping it here counts everything; a caller-supplied callback still
    # gets every event, unchanged.
    _usage_totals = {"in": 0, "out": 0, "cc": 0, "cr": 0}
    _model_seen: list[str] = [""]
    _caller_usage_cb = usage_callback
    # ⚡ llm_workers > 1 reports usage from worker threads; the lock keeps the
    # totals exact (THE USAGE RULE: an uncounted call is invisible spend).
    _usage_lock = threading.Lock()

    def _accumulate_usage(
        mdl: str, in_t: int, out_t: int, cc: int, cr: int
    ) -> None:
        with _usage_lock:
            _usage_totals["in"] += in_t
            _usage_totals["out"] += out_t
            _usage_totals["cc"] += cc
            _usage_totals["cr"] += cr
            _model_seen[0] = mdl
        if _caller_usage_cb is not None:
            _caller_usage_cb(mdl, in_t, out_t, cc, cr)

    usage_callback = _accumulate_usage
    nodes = enumerate_plo_nodes(pack)
    rng = random.Random(seed)
    ctx_set = set(action_contexts) if action_contexts else None
    pc_set = set(player_counts) if player_counts else None
    # Artifact all-ins (July 2026, team standing rule): at deep stacks the
    # all-in branches are TREE artifacts, not real lines -- never build a
    # question on a line through one, nor one whose correct answer is one
    # (the per-spot check below). Realistic short-stack jams (<= 40bb)
    # stay allowed.
    allins_ok = float(stack_bb) <= 40.0  # noqa: PLR2004
    candidates = [
        n
        for n in nodes
        if (hero_positions is None or n.actor in hero_positions)
        and (
            allins_ok
            or not any(a.action is PloActionType.ALL_IN for a in n.history_before)
        )
        and (
            max_prior_raises is None
            or sum(1 for a in n.history_before if a.action in _AGGRESSIVE) <= max_prior_raises
        )
        # Player filters count POT ENTRANTS (anyone who voluntarily put
        # chips in, folded-since or not), NOT currently-live seats (July 16
        # 2026 fix): live-counting read a collapsed 6-entrant squeeze pot
        # as "heads-up", so caller-heavy monsters passed a 1-2 player
        # filter AND the clean-lines cap. INVARIANT: the Generate page's
        # node count reads the same function via plo_filter_meta -- keep
        # them identical or the count lies.
        and (
            max_active_players is None
            or plo_pot_entrant_count(n) <= max_active_players
        )
        and (ctx_set is None or plo_node_action_context(n) in ctx_set)
        and (pc_set is None or plo_pot_entrant_count(n) in pc_set)
    ]
    rng.shuffle(candidates)
    # Balanced action mix (July 2026, from the first 9-max audit): a raw
    # shuffle draws ~all facing-3-bet/squeeze spots, because deep-raise nodes
    # vastly outnumber opens and single-raise defends in the worthy pool.
    # Diversify interleaves the shuffled candidates ROUND-ROBIN across the
    # action-context buckets, so the round-robin node draw below visits
    # "Opening", "Facing single raise", "Facing 3-bet", ... alternately and a
    # fill-to-N batch spreads across situations first. Deterministic: bucket
    # membership is a pure node fact and the within-bucket order is the
    # seeded shuffle's. (The postflop --diversify analogue.)
    if diversify:
        buckets: dict[str, list[PloDecisionNode]] = {}
        for n in candidates:
            buckets.setdefault(plo_node_action_context(n), []).append(n)
        bucket_order = [c for c in PLO_ACTION_CONTEXTS if c in buckets]
        interleaved: list[PloDecisionNode] = []
        while len(interleaved) < len(candidates):
            for ctx in bucket_order:
                if buckets[ctx]:
                    interleaved.append(buckets[ctx].pop(0))
        candidates = interleaved

    rows: list[dict[str, str]] = []
    question_records: list[dict[str, Any]] = []
    strip_counters: dict[str, int] = {}
    scanned = 0
    claim_flagged_rows = 0
    # Auto-fix lifecycle tallies (only move when revise_pass is on):
    # flagged = fixed + discarded + unchanged.
    revise_flagged = 0
    revise_fixed = 0
    revise_discarded = 0
    revise_unchanged = 0
    soft_flagged_rows = 0
    explanations_written = 0
    explanations_failed = 0
    explanation_failure_reasons: list[str] = []
    difficulty_filtered_out = 0
    ev_gap_filtered_out = 0
    # Round-robin passes over the (shuffled) nodes: each pass draws at most
    # one NEW hand per node, so a node contributes its 2nd question only
    # after every node has had the chance to contribute a 1st. A batch
    # larger than the node pool tops up with extra hands per node instead
    # of capping at one-question-per-node; a batch smaller than the pool
    # still spreads across distinct situations. ``drawn`` tracks the hands
    # already sampled per node this batch (including filtered-out ones, so
    # a rejected hand isn't redrawn forever); the loop ends when a full
    # pass draws nothing new.
    drawn: dict[str, set[int]] = {}
    # Circuit breaker for the drop-and-backfill failure handling: a systemic
    # problem (API outage, broken prompt edit) would otherwise burn API spend
    # across the ENTIRE spot pool chasing total_questions. Scattered failures
    # backfill fine; this many in a row with no success aborts the batch.
    consecutive_failures = 0
    aborted = False
    # Graceful stop (July 2026): when ``stop_check`` reports True (checked
    # between spots, so the in-flight question always finishes), the batch
    # ends early and ships what it has -- same clean write path as a full run.
    stopped_early = False

    # INCREMENTAL COMMIT (July 2026): the CSV + meta are (re)written after
    # EVERY committed question, atomically (tmp + replace so a concurrent
    # Review-page read never sees a torn file). A crash / kill / power loss
    # loses at most the question in flight -- the old end-only write lost the
    # whole batch. ``complete=False`` marks an in-flight (or interrupted)
    # sidecar; the FINAL flush flips it True, producing byte-identical CSV
    # content to the old single write.
    output_path = Path(output_path)
    meta_path = output_path.with_suffix(".meta.json")

    def _flush(*, complete: bool) -> None:
        csv_tmp = output_path.with_name(output_path.name + ".tmp")
        write_plo_csv(rows, csv_tmp)
        csv_tmp.replace(output_path)
        meta = {
            "pack_label": pack_label,
            # The registered pack this batch generated from (July 2026,
            # multi-pack era): the re-verifier resolves the pack folder
            # from this id.
            "pack_id": pack.pack_id,
            "table_size": pack.table_size,
            # False while the batch is still writing (or if it died mid-run);
            # True on the final flush of a finished OR gracefully-stopped run.
            "complete": complete,
            "run_settings": {
                "seed": seed,
                "total_questions": total_questions,
                "hero_positions": hero_positions,
                "max_prior_raises": max_prior_raises,
                "max_active_players": max_active_players,
                "action_contexts": action_contexts,
                "player_counts": player_counts,
                "min_frequency": min_frequency,
                "max_frequency": max_frequency,
                "exclude_ambiguous_band": exclude_ambiguous_band,
                "min_ev_gap_bb": min_ev_gap_bb,
                "diversify": diversify,
                "balanced": balanced,
                "compute_equity": compute_equity,
                "answer_style": answer_style,
                "stakes_bb_dollars": stakes_bb_dollars,
                "game_format": game_format,
                "ante_bb": ante_bb,
                "display_in_bb": display_in_bb,
                "stack_bb": stack_bb,
                "min_difficulty": min_difficulty,
                "max_difficulty": max_difficulty,
                "generate_explanations": generate_explanations,
                "model": explanation_model if generate_explanations else "",
                "run_claim_checker": run_claim_checker,
                "revise_pass": revise_pass,
                "final_audit": final_audit,
                # ⚡ concurrent LLM chains (July 2026); 1 = sequential.
                # Selection, row order, and every deterministic column are
                # worker-count-independent (commits are in submission order),
                # so the re-verifier needs no knowledge of this.
                "llm_workers": llm_workers,
            },
            "counters": {
                "nodes_scanned": scanned,
                "questions_written": len(rows),
                "difficulty_filtered_out": difficulty_filtered_out,
                "ev_gap_filtered_out": ev_gap_filtered_out,
                "explanations_written": explanations_written,
                "explanations_failed": explanations_failed,
                # Layer-7: rows whose SHIPPED text carries >= 1 claim flag.
                "claim_flagged_rows": claim_flagged_rows,
                # Auto-fix lifecycle (revise_pass batches):
                # flagged = fixed + discarded + unchanged.
                "revise_flagged": revise_flagged,
                "revise_fixed": revise_fixed,
                "revise_discarded": revise_discarded,
                "revise_unchanged": revise_unchanged,
                # Deterministic soft validators (position wording, v1).
                "soft_flagged_rows": soft_flagged_rows,
                # Deep-stack spots whose real strategy mixes the artifact
                # All-in at >= 5% -- silenced (never asked); trace dust was
                # stripped + renormalised (per-question artifact_stripped
                # records).
                "artifact_material_spots_skipped": strip_counters.get(
                    "artifact_material_spots_skipped", 0
                ),
                # True when a graceful stop ended the batch before
                # total_questions (the shortfall is intentional).
                "stopped_early": stopped_early,
            },
            "questions": question_records,
        }
        if balanced:
            # Achieved-vs-target per axis for the SHIPPED rows, against the
            # scored pool -- the honest-shortfall record the done panel and
            # Review read (see balanced_select.balance_report).
            meta["balance_report"] = balance_report(
                selected_attrs, balance_pool_attrs
            )
        meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
        meta_tmp.write_text(json.dumps(meta, indent=2, default=str))
        meta_tmp.replace(meta_path)

    def _drawn_spots():
        """Round-robin worthy-spot draw over the (shuffled) nodes.

        Each pass draws at most one NEW hand per node, so a node contributes
        its 2nd question only after every node has had the chance to
        contribute a 1st; ends when a full pass draws nothing new. This is
        the old inline draw moved verbatim into a generator so the balanced
        pre-pass and the default path share ONE draw order (and stay
        byte-deterministic).
        """
        nonlocal scanned
        while True:
            drew = False
            for node in candidates:
                scanned += 1
                spot = _first_worthy_spot(
                    node,
                    rng,
                    min_frequency=min_frequency,
                    max_frequency=max_frequency,
                    exclude_ambiguous_band=exclude_ambiguous_band,
                    exclude_indices=drawn.setdefault(node.node_id, set()),
                    allow_allin_answers=allins_ok,
                    counters=strip_counters,
                )
                if spot is None:
                    continue
                drawn[node.node_id].add(spot.hero_index)
                drew = True
                yield spot
            if not drew:
                return

    # --- 🎛️ Fully balanced mode (July 2026): pre-draw a capped pool, score
    # it CHEAPLY (facts without the equity sim -- PLO difficulty has no
    # equity axis), gate it, then greedy-order it for marginal balance
    # across difficulty band / situation / answer verb / position / hand
    # shape (pipeline/plo/balanced_select.py). Generation then consumes the
    # ordered pool front-to-back, so backfill after a failed question stays
    # balanced too. Supersedes ``diversify`` (situation is one of its axes).
    # The committed rows' attrs are recomputed from the REAL facts, so the
    # meta ``balance_report`` is honest even if an estimate drifted.
    balance_pool_attrs: list[BalanceAttrs] = []
    selected_attrs: list[BalanceAttrs] = []
    if balanced:
        pool_cap = min(max(12 * total_questions, 120), 600)
        pool_spots: list[PloSpot] = []
        for pool_spot in _drawn_spots():
            pool_facts = extract_plo_facts(
                pool_spot, pack, compute_equity=False, rng=random.Random(seed)
            )
            if (
                min_ev_gap_bb is not None
                and pool_facts.ev_gap_bb is not None
                and pool_facts.ev_gap_bb < min_ev_gap_bb
            ):
                ev_gap_filtered_out += 1
                continue
            pool_diff = compute_plo_difficulty(pool_facts)
            if not min_difficulty <= pool_diff.score <= max_difficulty:
                difficulty_filtered_out += 1
                continue
            pool_spots.append(pool_spot)
            balance_pool_attrs.append(
                _balance_attrs_for(pool_spot, pool_facts, pool_diff.score, pack)
            )
            if len(pool_spots) >= pool_cap:
                break
        order = balanced_order(balance_pool_attrs, total_questions)
        spot_iter = iter([pool_spots[i] for i in order])
    else:
        spot_iter = _drawn_spots()

    # ⚡ One question's WHOLE LLM chain (Layer 6 + the opt-in Layer-7 gate /
    # reviser / final audit + the deterministic soft validators), free of any
    # shared-state mutation so it can run on a worker thread (llm_workers).
    # Returns an "ok" result carrying the shipped pieces plus counter DELTAS
    # -- applied at commit time, in submission order, on the main thread --
    # or a "failed" result with the reason. Deltas accumulated BEFORE a
    # mid-chain exception are still returned, mirroring the sequential
    # counter semantics exactly (e.g. explanations_written ticks the moment
    # generation succeeds, even if a later stage raises).
    def _run_llm_chain(spot, facts, options, correct) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": True, "explanation": "", "claim_check_cell": "",
            "claim_issues": [], "revise_record": None, "soft_warnings": [],
            "deltas": {},
        }
        if not generate_explanations:
            return out
        deltas: dict[str, int] = {}
        out["deltas"] = deltas

        def _bump(key: str) -> None:
            deltas[key] = deltas.get(key, 0) + 1

        node = spot.node
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
                stack_bb=stack_bb,
                ante_bb=ante_bb,
                usage_callback=usage_callback,
            )
            _bump("explanations_written")
            claim_check_cell = ""
            claim_issues: list[dict[str, str]] = []
            revise_record: dict[str, Any] | None = None
            # --- Layer-7 LLM audit / revise passes (opt-in extra LLM
            # calls; July 2026 NLHE-parity port). Two flows share one
            # "gate" check:
            #   * run_claim_checker (no revise): FLAG-ONLY, one call ->
            #     claim_check column + meta record, prose untouched.
            #   * revise_pass: the gate (best-of-2, issues UNIONed)
            #     DECIDES whether to rewrite. If it flags, the reviser
            #     rewrites the prose (minimal-edit, one corrective
            #     retry), re-validated by the deterministic hard
            #     validators -- a rewrite that breaks a rule is
            #     DISCARDED and the original ships flagged. final_audit
            #     re-checks the KEPT rewrite. Lifecycle -> `revise` in
            #     the meta record for the PLO Review page.
            if run_claim_checker or revise_pass:
                from pipeline.plo.explanation_generator import (  # noqa: PLC0415
                    build_solver_data,
                )

                solver_data = build_solver_data(
                    facts, list(options), correct,
                    include_skills=explanation_include_skills,
                    stack_bb=stack_bb, ante_bb=ante_bb,
                )
                gate_dicts = _gate_check_best_of(
                    generated.answer_explanation,
                    solver_data,
                    explanation_client,
                    model=explanation_model,
                    system_prompt=claim_checker_prompt,
                    usage_callback=usage_callback,
                    passes=_REVISE_GATE_PASSES if revise_pass else 1,
                )
                gate_strs = [
                    f"{d['claim']} -- {d['problem']}" for d in gate_dicts
                ]
                if revise_pass:
                    if not gate_dicts:
                        revise_record = {"status": "clean", "gate_issues": []}
                    else:
                        _bump("revise_flagged")
                        original_prose = generated.answer_explanation
                        try:
                            rev = revise_plo_explanation(
                                generated, facts, issues=gate_strs,
                                client=explanation_client,
                                model=explanation_model,
                                temperature=explanation_temperature,
                                system_prompt=explanation_system_prompt,
                                include_skills=explanation_include_skills,
                                stack_bb=stack_bb,
                                ante_bb=ante_bb,
                                usage_callback=usage_callback,
                            )
                        except Exception as exc:  # noqa: BLE001 - never drop a row
                            logger.warning(
                                "plo batch: reviser failed for %s: %s",
                                node.node_id, exc,
                            )
                            rev = None
                        if rev is not None and rev.changed:
                            _bump("revise_fixed")
                            generated = rev.explanation  # ship the rewrite
                            revise_record = {
                                "status": "fixed",
                                "gate_issues": gate_strs,
                                "original_explanation": original_prose,
                                "revised_explanation":
                                    rev.explanation.answer_explanation,
                            }
                            if final_audit:  # re-check the KEPT rewrite
                                fa_dicts = _gate_check_best_of(
                                    generated.answer_explanation,
                                    solver_data,
                                    explanation_client,
                                    model=explanation_model,
                                    system_prompt=claim_checker_prompt,
                                    usage_callback=usage_callback,
                                    passes=1,
                                )
                                claim_check_cell = _issues_json(fa_dicts)
                                claim_issues = fa_dicts
                                revise_record["final_audit_issues"] = [
                                    f"{d['claim']} -- {d['problem']}"
                                    for d in fa_dicts
                                ]
                        else:
                            # UNCHANGED or DISCARDED: the ORIGINAL ships,
                            # so the gate's issues stay on it -- record why.
                            reason = (
                                getattr(rev, "rejected_reason", "") if rev
                                else "the reviser call failed"
                            )
                            attempt = (
                                getattr(rev, "revised_text", "") if rev else ""
                            )
                            if reason:
                                _bump("revise_discarded")
                                status = "discarded"
                            else:
                                _bump("revise_unchanged")
                                status = "unchanged"
                            revise_record = {
                                "status": status,
                                "gate_issues": gate_strs,
                                "rejected_reason": reason,
                                "attempted_rewrite": attempt,
                                "original_explanation": original_prose,
                            }
                            claim_check_cell = _issues_json(gate_dicts)
                            claim_issues = gate_dicts
                else:
                    # Plain claim checker: one pass, flag only.
                    claim_check_cell = _issues_json(gate_dicts)
                    claim_issues = gate_dicts
                if claim_issues:
                    _bump("claim_flagged_rows")
            # Soft validators (deterministic, flag-not-reject) run on
            # the FINAL (possibly revised) prose -- v1 is the
            # position-wording check, the PLO checker's #1 live catch.
            soft_warnings = run_plo_soft_validators(generated, facts)
            if soft_warnings:
                _bump("soft_flagged_rows")
            out.update(
                explanation=generated.answer_explanation,
                claim_check_cell=claim_check_cell,
                claim_issues=claim_issues,
                revise_record=revise_record,
                soft_warnings=soft_warnings,
            )
            return out
        except (ExplanationValidationError, OSError, KeyError) as exc:
            cards = " ".join(spot.hero_cards)
            out.update(
                ok=False,
                failure_reason=f"{type(exc).__name__} ({cards}): {exc}",
            )
            return out

    def _commit_result(spot, facts, difficulty, options, correct, res) -> None:
        """Apply one chain result on the MAIN thread, in submission order:
        counters, the failure path (incl. the consecutive-failure abort),
        the fold-EV tripwire, the row + meta record, progress, and the
        incremental flush. This is the ONLY place shared batch state moves,
        so llm_workers > 1 needs no locking beyond the usage totals."""
        nonlocal aborted, consecutive_failures, explanations_failed
        nonlocal explanations_written, claim_flagged_rows, soft_flagged_rows
        nonlocal revise_flagged, revise_fixed, revise_discarded, revise_unchanged
        node = spot.node
        d = res.get("deltas") or {}
        explanations_written += d.get("explanations_written", 0)
        claim_flagged_rows += d.get("claim_flagged_rows", 0)
        soft_flagged_rows += d.get("soft_flagged_rows", 0)
        revise_flagged += d.get("revise_flagged", 0)
        revise_fixed += d.get("revise_fixed", 0)
        revise_discarded += d.get("revise_discarded", 0)
        revise_unchanged += d.get("revise_unchanged", 0)
        if not res["ok"]:
            # A failed explanation DROPS the question from the CSV --
            # a blank-explanation row is not shippable. The failure is
            # still counted + reported (the Generate page's failure
            # expander), and the driver keeps drawing, so the batch
            # backfills toward total_questions with other spots
            # instead of shipping a hole.
            explanations_failed += 1
            # Sequential parity: generation success reset the streak
            # BEFORE a later stage failed, so such a failure lands at 1.
            consecutive_failures = (
                1 if d.get("explanations_written") else consecutive_failures + 1
            )
            explanation_failure_reasons.append(res["failure_reason"])
            logger.warning(
                "Layer 6 failed for a spot, dropping the question: %s",
                res["failure_reason"],
            )
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "Aborting batch: %d consecutive explanation "
                    "failures (systemic problem, not bad luck).",
                    consecutive_failures,
                )
                aborted = True
            return
        if generate_explanations:
            consecutive_failures = 0
        explanation = res["explanation"]
        claim_check_cell = res["claim_check_cell"]
        claim_issues = res["claim_issues"]
        revise_record = res["revise_record"]
        soft_warnings = list(res["soft_warnings"])

        # FOLD-EV TRIPWIRE (July 22 2026, from the min-raise size bug):
        # the pack's own fold EV must equal minus hero's resolved
        # invested amount -- a mismatch means a DISPLAYED bet size on
        # this line is wrong. Deterministic, runs on every question
        # (dry runs included); flags-not-rejects like the other soft
        # validators, and a size-resolution bug flags EVERY affected
        # row, so it cannot ship quietly.
        fold_issue = fold_ev_consistency_issue(
            spot, stack_bb=stack_bb, ante_bb=ante_bb
        )
        if fold_issue:
            if not soft_warnings:
                soft_flagged_rows += 1
            soft_warnings.append(fold_issue)

        row = build_plo_row(
            facts,
            difficulty=difficulty,
            options=options,
            correct_answer=correct,
            explanation=explanation,
            number=len(rows) + 1,
            pack_label=pack_label,
            pack=pack,
            stakes_bb_dollars=stakes_bb_dollars,
            game_format=game_format,
            ante_bb=ante_bb,
            display_in_bb=display_in_bb,
            stack_bb=stack_bb,
            # Flagged when the SHIPPED text still carries claim flags
            # (flag-only, unresolved gate issues, or final-audit hits) OR
            # a deterministic soft validator warned -- same rule as NLHE.
            validation_status=(
                "flagged" if (claim_issues or soft_warnings) else "draft"
            ),
        )
        # Layer-7 verdict (""=not checked, "[]"=checked clean, else the
        # JSON issue list) -- same convention as the NLHE claim_check.
        row["claim_check"] = claim_check_cell
        rows.append(row)
        record: dict[str, Any] = {
            "number": len(rows),
            "node_id": node.node_id,
            "actor": node.actor,
            # The .rng hand index -- the re-verifier's join key (User
            # Cards is only a representative rendering of the class).
            "hero_index": spot.hero_index,
            "hero_label": spot.hero_label,
            "correct_answer": correct,
            "options": list(options),
            "difficulty": difficulty.score,
            # Lifecycle status lives HERE since the CSV column was
            # dropped (July 16 declutter): draft, or flagged when the
            # shipped text carries claim flags / soft warnings.
            "validation_status": row["validation_status"],
        }
        # Artifact-strip transparency (July 2026), as in the NLHE metas.
        if spot.stripped_artifact_freq > 0:
            record["artifact_stripped"] = {
                "labels": ["All-in"],
                "freq": round(spot.stripped_artifact_freq, 4),
            }
        if claim_issues:
            record["claim_check_issues"] = claim_issues
        if revise_record is not None:
            record["revise"] = revise_record
        if soft_warnings:
            record["validator_warnings"] = soft_warnings
        question_records.append(record)
        if balanced:
            # Recomputed from the REAL facts (not the pre-pass estimate)
            # so the meta balance_report describes what actually shipped.
            selected_attrs.append(
                _balance_attrs_for(spot, facts, difficulty.score, pack)
            )
        # Live progress for the admin page's inline bar (how far along
        # this batch is). Called after each question is committed.
        if progress_callback is not None:
            progress_callback(len(rows), total_questions)
        # INCREMENTAL COMMIT: everything committed so far is now on disk;
        # a crash from here loses only the next question.
        _flush(complete=False)

    def _prepare_spot(spot):
        """Deterministic pre-LLM work, ALWAYS on the main thread in draw
        order (the facts RNG + the filter counters must never depend on
        thread scheduling). Returns None when a pre-LLM gate filters the
        spot, else the tuple the chain + commit need."""
        nonlocal ev_gap_filtered_out, difficulty_filtered_out
        facts = extract_plo_facts(
            spot, pack, compute_equity=compute_equity, rng=random.Random(seed)
        )
        # EV-gap quality gate (PLO has a real ev_gap on every spot, raises
        # included), applied BEFORE the paid LLM call -- mirrors the NLHE
        # gate.
        if (
            min_ev_gap_bb is not None
            and facts.ev_gap_bb is not None
            and facts.ev_gap_bb < min_ev_gap_bb
        ):
            ev_gap_filtered_out += 1
            return None
        # Difficulty-band filter BEFORE the (paid) LLM call, so out-of-band
        # spots cost no API spend -- the same gate the NLHE Generate page
        # uses.
        difficulty = compute_plo_difficulty(facts)
        if not min_difficulty <= difficulty.score <= max_difficulty:
            difficulty_filtered_out += 1
            return None
        options, correct = build_options(facts, style=answer_style)
        return facts, difficulty, options, correct

    workers = max(1, int(llm_workers or 1))
    if workers == 1:
        # The classic sequential path -- behaviour (and dry-run bytes)
        # identical to before the ⚡ refactor.
        while len(rows) < total_questions and not aborted and not stopped_early:
            drew_this_pass = False
            for spot in spot_iter:
                if len(rows) >= total_questions or aborted:
                    break
                # Graceful stop: checked between spots, never mid-question, so
                # "finish the current question, keep everything committed".
                if stop_check is not None and stop_check():
                    stopped_early = True
                    break
                drew_this_pass = True
                prepared = _prepare_spot(spot)
                if prepared is None:
                    continue
                facts, difficulty, options, correct = prepared
                _commit_result(
                    spot, facts, difficulty, options, correct,
                    _run_llm_chain(spot, facts, options, correct),
                )
            if not drew_this_pass:
                break
    else:
        # ⚡ Bounded pipeline: keep up to ``workers`` chains in flight, never
        # more than the questions still needed; commit strictly in
        # SUBMISSION order (head-of-line), so row order + numbering match a
        # sequential run over the same draws. Graceful stop finishes every
        # chain already in flight and keeps it (the plural of "finish the
        # current question"); the consecutive-failure abort discards
        # not-yet-committed results exactly like the sequential path never
        # would have drawn them (their spend is still counted -- the usage
        # callback fires from the worker at call time).
        inflight: deque[tuple[Any, ...]] = deque()
        exhausted = False
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="plo-llm"
        ) as pool_exec:
            while True:
                while (
                    not exhausted and not stopped_early and not aborted
                    and len(inflight) < workers
                    and len(rows) + len(inflight) < total_questions
                ):
                    if stop_check is not None and stop_check():
                        stopped_early = True
                        break
                    spot = next(spot_iter, None)
                    if spot is None:
                        exhausted = True
                        break
                    prepared = _prepare_spot(spot)
                    if prepared is None:
                        continue
                    facts, difficulty, options, correct = prepared
                    inflight.append((
                        spot, facts, difficulty, options, correct,
                        pool_exec.submit(
                            _run_llm_chain, spot, facts, options, correct
                        ),
                    ))
                if aborted or not inflight:
                    break
                spot, facts, difficulty, options, correct, fut = (
                    inflight.popleft()
                )
                _commit_result(spot, facts, difficulty, options, correct,
                               fut.result())
            # Graceful stop: commit everything already in flight (paid for
            # and wanted). An abort discards instead -- same as sequential,
            # which would never have drawn past the failure streak.
            while inflight and not aborted and len(rows) < total_questions:
                spot, facts, difficulty, options, correct, fut = (
                    inflight.popleft()
                )
                _commit_result(spot, facts, difficulty, options, correct,
                               fut.result())
            inflight.clear()


    # Final flush: same content, complete=True. A gracefully-stopped batch is
    # COMPLETE (it ended cleanly, just early); only a crash leaves False.
    _flush(complete=True)

    return PloBatchResult(
        output_path=output_path,
        questions_written=len(rows),
        questions_requested=total_questions,
        nodes_scanned=scanned,
        explanations_written=explanations_written,
        explanations_failed=explanations_failed,
        explanation_failure_reasons=tuple(explanation_failure_reasons),
        difficulty_filtered_out=difficulty_filtered_out,
        ev_gap_filtered_out=ev_gap_filtered_out,
        artifact_material_spots_skipped=strip_counters.get(
            "artifact_material_spots_skipped", 0
        ),
        meta_path=meta_path,
        stopped_early=stopped_early,
        model_used=_model_seen[0],
        total_input_tokens=_usage_totals["in"],
        total_output_tokens=_usage_totals["out"],
        total_cache_creation_tokens=_usage_totals["cc"],
        total_cache_read_tokens=_usage_totals["cr"],
    )


__all__ = ["PloBatchResult", "generate_plo_batch"]
