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
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from pipeline.explanation_generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ExplanationValidationError,
    GeneratedExplanation,
)
from pipeline.preflop.claim_checker import (
    CHECKER_SYSTEM_PROMPT,
    check_explanation_claims,
    claim_check_to_json,
)
from pipeline.preflop.difficulty import (
    compute_difficulty,
)
from pipeline.preflop.ev_engine import (
    compute_break_even_equity,
    compute_ev_gap_bb,
)
from pipeline.preflop.explanation_generator import (
    _trim_facts_for_prompt,
    build_explanation_prompt_parts,
    build_shared_prompt_parts,
    generate_preflop_answer_explanation,
)
from pipeline.preflop.fact_extractor import (
    DEFAULT_EQUITY_RUNOUTS,
    construct_villain_range_path,
    extract_facts,
    identify_villain,
)
from pipeline.preflop.format_writer import (
    build_preflop_row,
    format_preflop_question,
    write_preflop_csv,
)
from pipeline.preflop.grammars.types import (
    PreflopActionType,
    render_raise_size_token,
)
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
from pipeline.preflop.reviser import revise_explanation
from pipeline.preflop.spot_sampler import (
    PreflopSpot,
    _cached_parse_range_file,
    enumerate_spots_for_node,
    sample_spot,
)
from pipeline.preflop.validators import run_preflop_soft_validators

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
    # "After call(s)" was split (June 2026): one live flat-caller produces
    # tame, well-solved spots (squeeze opportunities, a single overcall);
    # two or more is the multiway territory that gets noisy / unsolved.
    "After one call",
    "After multiple calls",
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

    # The COMPLETE CSV row (all columns) this spot would have produced, built
    # from the LLM's rejected-but-parsed explanation + the spot's facts. This
    # is what lets the Review page "keep the exact explanation" and promote a
    # human-review spot into the batch with one click -- no re-generation, no
    # re-sampling. None when the row couldn't be built (a pre-LLM failure with
    # no facts, or an attempt whose prose never parsed into a candidate), in
    # which case the spot is viewable on Review but not one-click promotable.
    row: dict[str, str] | None = None

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

    # How many worthy spots were skipped by the EV-coherence guard
    # (:func:`spot_mix_incoherent`): a "mixed" hand whose meaningfully-played
    # actions are several bb apart in the solver's OWN per-action EV, i.e. an
    # unconverged-node noise mix node_is_unconverged can't catch. 0 on
    # weight-only packs (no per-action EVs to judge). UI feedback.
    incoherent_mix_filtered_out: int = 0

    # Premise-realism gate counts (June 2026): spots skipped because the
    # villain takes their whole line too rarely (rare_line) or because
    # hero's own earlier actions in the story are near-never solver
    # actions with this hand (rare_premise). UI feedback.
    rare_line_filtered_out: int = 0
    rare_premise_filtered_out: int = 0

    # How many WRITTEN rows a soft validator warned on (their CSV
    # validation_status is "flagged"; the warning text lives in the meta
    # sidecar's question record). Soft warnings never reject a row --
    # they queue it for human review. (June 2026 round-2 audit.)
    soft_flagged_rows: int = 0

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

    The RAISE LEVEL wins: a pot that was opened, flat-called, then 3-bet is
    "Facing 3-bet" (not an after-call bucket) -- the highest raise hero faces
    is the intuitive label, and a flat earlier in the line doesn't change it.
    The "After ... call(s)" buckets are reserved for a SINGLE open that went
    multiway via flats (the squeeze / overcall texture). How multiway any
    bucket is, is the player-count filter's job, not this one's.

      * "Opening"              -- no prior raise in history
      * "Facing 4-bet+"        -- three or more raises (callers, if any, ignored)
      * "Facing 3-bet"         -- exactly two raises (callers, if any, ignored)
      * "After one call"       -- a SINGLE open + at most one live flat-caller
                                   (an overcall / single-caller squeeze spot)
      * "After multiple calls" -- a SINGLE open + two or more live flat-callers
      * "Facing single raise"  -- a single open, no flat-callers
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
    # Raise level takes precedence over any earlier flat: a 3-bet / 4-bet pot
    # is labeled by the raise hero faces even when someone called along the
    # way (June 2026 -- the after-call buckets used to absorb these, so a 4-bet
    # pot with one early caller silently showed up under "After one call").
    if n_raises >= 3:  # noqa: PLR2004
        return "Facing 4-bet+"
    if n_raises == 2:  # noqa: PLR2004
        return "Facing 3-bet"
    # Exactly one raise (a single open). Flat-callers make it a squeeze /
    # overcall spot; split by the number of LIVE flat-callers -- distinct
    # non-hero positions whose LAST action is a call (hero's own call and a
    # caller who later folded don't count; raw n_calls over-counts both).
    if n_calls > 0:
        last: dict[str, PreflopActionType] = {}
        for a in node.history_before:
            last[a.position] = a.action_type
        live_callers = sum(
            1
            for pos, t in last.items()
            if t is PreflopActionType.CALL and pos != node.actor
        )
        return "After one call" if live_callers <= 1 else "After multiple calls"
    return "Facing single raise"


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

# Uniform-default tell: a node the solver never refined leaves many hands at
# an equal split across every action (1/N each -- e.g. fold/call/all-in all
# ~0.333). A hand "reaches" the node if its weight summed over every option
# exceeds _UNIFORM_REACH_EPS; it's "uniform-default" if its per-action
# weights are within _UNIFORM_TOL of each other. A node is rejected when more
# than _UNIFORM_DEFAULT_MAX_FRAC of its reaching hands are uniform-default.
# Measured (June 2026): the four raise-ladder buckets (Opening / Facing
# single raise / 3-bet / 4-bet+) are uniformly 0% here, so a 30% bar has zero
# false-positive risk on well-solved nodes; only the deep "after calls"
# multiway tail trips it (the spot that motivated this read ~36%).
_UNIFORM_REACH_EPS = 0.01
_UNIFORM_TOL = 0.005
_UNIFORM_DEFAULT_MAX_FRAC = 0.30


def node_is_unconverged(node: PreflopDecisionNode) -> bool:
    """True if a node's solver strategy is detectably non-GTO (unconverged).

    These show up on near-zero-reach lines (deep multiway limp/jam pots)
    the solver never converged, and they produce nonsense questions. Three
    clean tells:

    * **AA canary** -- given AA reaches this node at all, it must continue
      ~100% of the time: AA never folds preflop in cash. A conditional
      continue (continue mass / presence) meaningfully below 1 means the
      node is garbage.
    * **Premium-pair monotonicity** -- facing the same action AA must
      continue at least as often as KK, and KK at least as often as QQ; an
      inversion (a stronger pair continuing LESS) is unconverged.
    * **Uniform default** -- too large a share of the reaching range is split
      equally across every action (1/N each), which is what the solver
      leaves on a node it never refined. Catches deep multiway nodes where
      AA isn't even present to canary on (the original case: a 4-way squeeze
      line ~36% uniform-default). See :data:`_UNIFORM_DEFAULT_MAX_FRAC`.

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

    # One parse of every option file feeds both tells: per-premium
    # presence/continue mass (the AA canary) and the full per-hand action mix
    # (the uniform-default check).
    presence = {"AA": 0.0, "KK": 0.0, "QQ": 0.0}
    cont = {"AA": 0.0, "KK": 0.0, "QQ": 0.0}
    mixes: dict[str, list[float]] = {}
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
        for hand, w in weights.items():
            mixes.setdefault(hand, []).append(w)

    # Uniform-default tell: too much of the reaching range is an equal split
    # across every action (the solver's never-refined default). Checked
    # before the AA-presence gate below, because these deep multiway nodes
    # often have AA essentially absent yet are still garbage. Needs >= 2
    # actions to have a split to judge.
    if len(node.actions) >= 2:
        reaching = [m for m in mixes.values() if sum(m) > _UNIFORM_REACH_EPS]
        if reaching:
            uniform = sum(1 for m in reaching if max(m) - min(m) < _UNIFORM_TOL)
            if uniform / len(reaching) >= _UNIFORM_DEFAULT_MAX_FRAC:
                return True

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


# --- EV-coherence guard (a hand-level convergence tell node_is_unconverged
#     can't see) ------------------------------------------------------------
# GTO mixing IS indifference: a hand splits its action between two plays only
# because they're ~equal in EV. So when the solver's OWN per-action EVs say a
# meaningfully-played action is several bb worse than the best action the hand
# also takes, the "mix" isn't a real mixed strategy -- it's an unconverged
# node. This is the noise that produced 82o "calling" an all-in 26% of the
# time at a ~15bb EV deficit: node_is_unconverged passed it because the tell is
# hand-specific (one hand's frequencies), not the node-wide uniform-default
# pattern that function looks for. Needs the pack's per-action EVs, so it is a
# no-op on weight-only packs (the PioViewer Ryan pack).
_MIX_MIN_ACTION_FREQ = 0.10   # a <10% sliver isn't a "mix" -- ignore it
_MIX_MAX_EV_SPREAD_BB = 3.0   # bb spread among mixed actions above which it's noise


def spot_mix_incoherent(facts: object, pack: PreflopPack) -> bool:
    """True when a mixed spot's meaningfully-played actions are far from
    EV-indifferent per the solver's own per-action EVs.

    Among the actions hero takes at least :data:`_MIX_MIN_ACTION_FREQ` of the
    time, if the spread between the best and worst per-action EV exceeds
    :data:`_MIX_MAX_EV_SPREAD_BB`, the spot is flagged: a converged solver only
    mixes near-indifferent actions, so a large spread means the node never
    refined this hand. See the module comment above.

    Fails OPEN -- returns False (keep the spot) when the pack ships no
    per-action EVs, when fewer than two actions clear the frequency floor (a
    near-pure spot, not a mix), or on any read error. A missing signal must
    never drop an otherwise-good spot.
    """
    from pipeline.preflop.format_writer import action_evs_bb  # noqa: PLC0415
    from pipeline.preflop.options import canonicalize_strategy  # noqa: PLC0415

    try:
        evs = action_evs_bb(facts, pack)  # type: ignore[arg-type]
    except (OSError, ValueError):
        return False
    if not evs:
        return False
    freqs = canonicalize_strategy(facts)  # type: ignore[arg-type]
    taken = [
        label
        for label, freq in freqs.items()
        if freq >= _MIX_MIN_ACTION_FREQ and label in evs
    ]
    if len(taken) < 2:  # noqa: PLR2004 -- need two played actions to have a "mix"
        return False
    spread = max(evs[label] for label in taken) - min(evs[label] for label in taken)
    return spread > _MIX_MAX_EV_SPREAD_BB


def ev_gap_from_action_evs(facts: object, pack: PreflopPack) -> float | None:
    """EV gap (bb) between the dominant and 2nd-most-frequent action, taken
    from the SOLVER'S OWN per-action EVs rather than the analytic ev_engine.

    This is the accurate gap -- just the solver's EV for the action hero plays
    most often minus its EV for the next-most-played action. The analytic
    :func:`compute_ev_gap_bb` is realization-blind and wildly overstates the
    gap on multiway 3-bet/4-bet pots (e.g. 12.55bb when the solver's own EVs
    are 0.11bb apart -- which is exactly why those spots mix near 50/50). That
    bad number used to feed difficulty's EV axis and rate close multiway
    decisions as Medium when they should be Hard.

    Returns None when the pack ships no per-action EVs (the EV-less Ryan pack)
    or fewer than two played actions carry an EV; the caller then falls back to
    the analytic engine. Same source the EV-coherence gate uses, so the two
    stay consistent.
    """
    from pipeline.preflop.format_writer import action_evs_bb  # noqa: PLC0415
    from pipeline.preflop.options import canonicalize_strategy  # noqa: PLC0415

    try:
        evs = action_evs_bb(facts, pack)  # type: ignore[arg-type]
    except (OSError, ValueError):
        return None
    if not evs:
        return None
    freqs = canonicalize_strategy(facts)  # type: ignore[arg-type]
    ranked = sorted(
        ((freq, label) for label, freq in freqs.items() if label in evs),
        reverse=True,
    )
    if len(ranked) < 2:  # noqa: PLR2004 -- need a dominant + a runner-up to have a gap
        return None
    top, second = ranked[0][1], ranked[1][1]
    return abs(evs[top] - evs[second])


def _villain_line_pct(
    node: PreflopDecisionNode, pack: PreflopPack
) -> float | None:
    """How often (as % of dealt hands) the primary villain takes their line.

    June 2026 audit finding: spots were generated around villain lines the
    solver takes ~0.0% of the time -- a question built on a ghost. This is
    the same number as ``villain_stats.pct_of_dealt_hands``, computed
    cheaply (one cached range-file read) BEFORE any equity sim or LLM
    spend so the batch can gate on it. ``None`` when there is no villain
    (open spots) or the file is unreadable -- the gate then passes.
    """
    villain = identify_villain(node)
    if villain is None:
        return None
    try:
        path = construct_villain_range_path(node, villain, pack)
        if not path.is_file():
            return None
        weights = _cached_parse_range_file(str(path))
    except (OSError, ValueError):
        return None
    total = sum(
        w * (6 if len(h) == 2 else 4 if h.endswith("s") else 12)
        for h, w in weights.items()
    )
    return 100.0 * total / 1326.0


def _incoming_villain_line_pct(
    node: PreflopDecisionNode,
    node_index: dict[tuple[str, tuple], PreflopDecisionNode],
    pack: PreflopPack,
) -> float | None:
    """% of dealt hands the action hero is FACING actually occurs.

    Generalises :func:`_villain_line_pct` past the most-recent raiser to
    also cover a LIMPED pot. ``identify_villain`` only sees raises and
    all-ins, so a completer (e.g. the SB limping to the BB) had no line
    measured. On the 9-max pack the SB completes ~0.1% of the time, so
    those bb-vs-limp spots (labelled "Opening", since there is no raise)
    slipped the premise gate and produced a question built on an action
    the villain almost never takes (June 2026 review finding). Here:

      * a raise / all-in incoming action -> :func:`_villain_line_pct`
        (cheap range-file mass, unchanged);
      * a CALL (limp) incoming action -> the completer's combo-weighted
        call frequency at the node where they completed;
      * no voluntary villain action (hero opens unraised) -> ``None`` so
        the gate passes.
    """
    if identify_villain(node) is not None:
        return _villain_line_pct(node, pack)
    # No raiser: find the most-recent completer (limp) hero is facing.
    for i in range(len(node.history_before) - 1, -1, -1):
        action = node.history_before[i]
        if action.action_type is PreflopActionType.FOLD:
            continue
        if action.action_type is not PreflopActionType.CALL:
            return None
        limper_node = node_index.get(
            (action.position, node.history_before[:i])
        )
        if limper_node is None:
            return None
        call_mass = sum(
            sp.presence
            * sp.action_frequencies.get("Call", 0.0)
            * (
                6 if len(sp.hero_hand_class) == 2
                else 4 if sp.hero_hand_class.endswith("s")
                else 12
            )
            for sp in enumerate_spots_for_node(limper_node)
        )
        return 100.0 * call_mass / 1326.0
    return None


def _hero_premise_min_freq(
    spot: PreflopSpot,
    node_index: dict[tuple[str, tuple], PreflopDecisionNode],
) -> float | None:
    """The weakest link in hero's own story: min frequency of hero's hand
    across HERO'S earlier actions in the line.

    June 2026 audit finding: a question premised on "you opened K6s from
    the LJ" -- an action the solver takes ~1% of the time -- asks the
    player to own a decision a good player wouldn't have made. For each
    prior action by hero, look up the node where hero took it and read
    this hand's conditional frequency for that exact action; return the
    minimum. ``None`` when hero has no prior actions (clean spots) or a
    lookup fails -- the gate then passes.
    """
    node = spot.node
    min_freq: float | None = None
    for i, a in enumerate(node.history_before):
        if a.position != node.actor:
            continue
        prior = node_index.get((node.actor, node.history_before[:i]))
        if prior is None:
            continue
        if a.action_type is PreflopActionType.RAISE:
            label = f"Raise {render_raise_size_token(a.raise_size_pct)}"
        else:
            label = a.action_type.value
        freq = sample_spot(prior, spot.hero_hand_class).action_frequencies.get(label)
        if freq is None:
            continue
        min_freq = freq if min_freq is None else min(min_freq, freq)
    return min_freq


# Dominant-action frequency at/above which a near-pure spot is DISPLAYED as a
# clean 100% spot (see the snap step in the per-spot loop). The last few percent
# of a 96-99% mix is GTO balancing noise, not a teachable alternative, so for a
# training tool "always X" beats "X 97% / Y 3%". The real frequencies are kept
# on the meta record for the Review/Compare note. The function default is
# 1.0 = OFF (tests/scripts see no snapping); the admin panel opts in at 0.96.
# When snapped, difficulty gets full EV credit so the spot reads easy on the EV
# axis too (consistent with easy_freq=1.0 from the pure dominant action).
# TODO(PLO): apply the same snap to the PLO path (pipeline/plo) once NLHE is
# validated -- PLO worthy spots are mixed by definition, so the threshold/UX
# there needs its own thought.
_PURE_DISPLAY_EV_GAP_BB = 3.0  # = difficulty's full-credit EV ceiling


def _snap_facts_to_pure(
    facts: object, spot: PreflopSpot, threshold: float
) -> tuple[object, dict[str, float] | None]:
    """Display a near-pure spot as a clean 100% spot.

    When hero's dominant action is at/above ``threshold`` (and not already a
    pure 100%), return a facts copy with the dominant action snapped to 100%
    (others 0%) and ``ev_gap_bb`` cleared -- so the options, the SOLVER DATA
    block, the CSV freqs, and the prose all read "always X" with no mention of
    the leftover few-percent mix -- plus the REAL frequencies (for the
    Review/Compare note). Returns ``(facts, None)`` unchanged when the spot
    isn't near-pure enough. ``facts`` is typed ``object`` to match the batch
    loop's local (extract_facts is annotated loosely there); the correctly
    typed ``spot`` drives the decision + rebuild.
    """
    if not (1.0 > spot.dominant_frequency >= threshold):
        return facts, None
    real_freqs = dict(spot.action_frequencies)
    pure_freqs = {
        label: (1.0 if label == spot.dominant_action else 0.0)
        for label in spot.action_frequencies
    }
    snapped = replace(
        facts,
        spot=replace(spot, action_frequencies=pure_freqs, dominant_frequency=1.0),
        ev_gap_bb=None,
    )
    return snapped, real_freqs


def _safe_claim_check(
    prose: str, facts: object, client: object, *,
    model: str, system_prompt: str, node_id: str,
) -> object | None:
    """Run the claim checker, wrapped so a checker failure never drops a row.

    Returns the ``ClaimCheckResult`` or ``None`` on error. Shared by the
    flag-only path and the revise pass (initial check + final audit)."""
    try:
        return check_explanation_claims(
            prose, _trim_facts_for_prompt(facts), client,
            model=model, system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("batch: claim checker failed for %s: %s", node_id, exc)
        return None


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
    min_villain_line_pct: float | None = 0.25,
    min_hero_premise_freq: float | None = 0.05,
    pure_snap_threshold: float = 1.0,
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
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    revise_pass: bool = False,
    final_audit: bool = False,
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
    # Per-row validation_status overrides, positionally aligned with
    # ``rows``: None = auto_approved, "flagged" = a soft validator warned
    # (warning text goes into the row's meta record below).
    row_statuses: list[str | None] = []
    # Per-row claim-checker output (Layer 7, opt-in), positionally aligned with
    # ``rows``: "" = checker not run, "[]" = run and clean, JSON list = flagged.
    claim_checks: list[str] = []
    # Per-row real solver mix for snapped-to-pure spots (None otherwise), aligned
    # with ``rows``: the CSV Notes column records it behind the 100% display.
    snapped_freqs_list: list[dict[str, float] | None] = []
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
    incoherent_mix_filtered_out = 0
    _node_noise: dict[str, bool] = {}
    # Premise-realism gates (June 2026 audit): spots built on a villain
    # line the solver ~never takes, or on hero prior actions the solver
    # ~never takes with this hand. Both checked from cached range files
    # BEFORE any equity sim / LLM spend.
    rare_line_filtered_out = 0
    rare_premise_filtered_out = 0
    # Self-revision (auto-fix) outcome tallies. Only move when revise_pass is on:
    # how many SHIPPED rows the claim-check gate flagged, and how the auto-fix
    # resolved them. fixed + discarded + unchanged == flagged.
    revise_flagged = 0
    revise_fixed = 0
    revise_discarded = 0
    revise_unchanged = 0
    _node_villain_pct: dict[str, float | None] = {}
    node_index: dict[tuple[str, tuple], PreflopDecisionNode] = {
        (n.actor, n.history_before): n for n in nodes
    }
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
        # --- premise-realism gates (cheap, pre-equity, pre-LLM) ----------
        if min_villain_line_pct is not None:
            if nid in _node_villain_pct:
                vpct = _node_villain_pct[nid]
            else:
                vpct = _incoming_villain_line_pct(spot.node, node_index, pack)
                _node_villain_pct[nid] = vpct
            if vpct is not None and vpct < min_villain_line_pct:
                rare_line_filtered_out += 1
                continue
        if min_hero_premise_freq is not None:
            premise = _hero_premise_min_freq(spot, node_index)
            if premise is not None and premise < min_hero_premise_freq:
                rare_premise_filtered_out += 1
                continue
        # --- facts + canonical difficulty (no API spend yet) -------------
        # compute_difficulty blends freq + EV gap + archetype/concept +
        # hand class (with EV-weight redistribution when the EV engine
        # couldn't score the spot, typical for raise-involved spots).
        # A failure here is rare (equity / extract issues) but must not
        # abort the batch.
        snapped_real_freqs: dict[str, float] | None = None
        try:
            facts: object = extract_facts(
                spot, pack, equity_runouts=equity_runouts)
            # Enrich with the pot-odds break-even equity (needs the pack's
            # chip geometry) so Layer 6 can cite it instead of computing
            # pot odds itself. None on spots with no bet to call.
            facts = replace(
                facts, break_even_equity=compute_break_even_equity(facts, pack)
            )
            # Prefer the solver's OWN per-action EVs (accurate on multiway
            # pots, and defined for raise-involved spots too); fall back to the
            # analytic engine only for EV-less packs (the Ryan pack). This is
            # the same EV source the coherence gate uses, so they agree, and it
            # stops the analytic gap's multiway overstatement from skewing
            # difficulty's EV axis.
            ev_gap = ev_gap_from_action_evs(facts, pack)
            if ev_gap is None:
                ev_gap = compute_ev_gap_bb(facts, pack)
            # Carry the gap on the facts so Layer 6's data block can cite the
            # "cost of the mistake" (None for raise-involved spots -> omitted).
            facts = replace(facts, ev_gap_bb=ev_gap)
            # --- snap near-pure spots to a clean pure display ----------------
            # A 96-99% dominant action is shown as 100% so the question reads as
            # "always X" everywhere the user + the LLM see it (options, the
            # SOLVER DATA block, the CSV freqs); the real mix is kept for the
            # Review/Compare note. See _snap_facts_to_pure.
            facts, snapped_real_freqs = _snap_facts_to_pure(
                facts, spot, pure_snap_threshold
            )
            if snapped_real_freqs is not None:
                ev_gap = _PURE_DISPLAY_EV_GAP_BB  # easy on the EV axis too
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
        # --- EV-coherence guard (skip incoherent "mixed" spots) ----------
        # A mix whose meaningfully-played actions are far apart in the solver's
        # own per-action EV is unconverged-node noise, not a real mixed
        # strategy (e.g. 82o "calling" a shove at a 15bb EV deficit). Pre-LLM,
        # so no spend is wasted. No-op on packs without per-action EVs.
        if spot_mix_incoherent(facts, pack):
            incoherent_mix_filtered_out += 1
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
            # --- Layer-7 LLM audit / revise passes (opt-in extra LLM calls) --
            # Two flows share one "gate" call (the 2nd LLM pass):
            #   * run_claim_checker (no revise): the gate just FLAGS prose, flag
            #     only -> `claim_check_issues` (the regular checker).
            #   * revise_pass: the gate DECIDES whether to rewrite. If it flags,
            #     a 3rd call REWRITES the prose, re-validated by the hard
            #     validators (a rewrite that breaks a rule is DISCARDED and the
            #     original kept). final_audit is a 4th call that claim-checks the
            #     KEPT rewrite. The full lifecycle is recorded in `revise` so the
            #     Review page can show what each call did and flag it distinctly.
            claim_check_json = ""
            claim_issues: list[str] = []          # regular checker (2nd call), flag-only
            final_audit_issues: list[str] = []    # 4th call: claim-check on the rewrite
            revise_record: dict[str, object] | None = None
            checker_prompt = claim_checker_prompt or CHECKER_SYSTEM_PROMPT
            if not dry_run and (run_claim_checker or revise_pass):
                cc = _safe_claim_check(
                    explanation.answer_explanation, facts, client,
                    model=model, system_prompt=checker_prompt,
                    node_id=spot.node.node_id,
                )
                gate_issues = (
                    [f"{i.claim} -- {i.problem}" for i in cc.issues]
                    if cc is not None else []
                )
                if revise_pass:
                    if not gate_issues:
                        revise_record = {"status": "clean", "gate_issues": []}
                    else:
                        revise_flagged += 1
                        original_prose = explanation.answer_explanation
                        try:
                            rev = revise_explanation(
                                explanation, facts, issues=gate_issues,
                                client=client, model=model, temperature=temperature,
                                max_tokens=max_tokens, system_prompt=system_prompt,
                                usage_callback=_record_usage,
                            )
                        except Exception as exc:  # noqa: BLE001 - never drop a row
                            logger.warning(
                                "batch: reviser failed for %s: %s",
                                spot.node.node_id, exc,
                            )
                            rev = None
                        if rev is not None and rev.changed:
                            revise_fixed += 1
                            explanation = rev.explanation  # ship the rewrite
                            revise_record = {
                                "status": "fixed",
                                "gate_issues": gate_issues,
                                "original_explanation": original_prose,
                                "revised_explanation": rev.explanation.answer_explanation,
                            }
                            if final_audit:  # 4th call: claim-check the rewrite
                                cc4 = _safe_claim_check(
                                    explanation.answer_explanation, facts, client,
                                    model=model, system_prompt=checker_prompt,
                                    node_id=spot.node.node_id,
                                )
                                if cc4 is not None:
                                    final_audit_issues = [
                                        f"{i.claim} -- {i.problem}" for i in cc4.issues
                                    ]
                                    claim_check_json = claim_check_to_json(cc4)
                                revise_record["final_audit_issues"] = final_audit_issues
                        else:
                            # The rewrite came back UNCHANGED or was DISCARDED by
                            # the hard validators. The ORIGINAL ships, so the
                            # gate's issues are unresolved -- record exactly why.
                            reason = (
                                getattr(rev, "rejected_reason", "") if rev
                                else "the reviser call failed"
                            )
                            attempt = getattr(rev, "revised_text", "") if rev else ""
                            if reason:
                                revise_discarded += 1
                                status = "discarded"
                            else:
                                revise_unchanged += 1
                                status = "unchanged"
                            revise_record = {
                                "status": status,
                                "gate_issues": gate_issues,
                                "rejected_reason": reason,
                                "attempted_rewrite": attempt,
                                "original_explanation": original_prose,
                            }
                            if cc is not None:  # gate findings describe the shipped original
                                claim_check_json = claim_check_to_json(cc)
                elif run_claim_checker:
                    # Plain claim checker: one 2nd LLM pass, flag only.
                    if cc is not None:
                        claim_check_json = claim_check_to_json(cc)
                    claim_issues = gate_issues

            # Issues that REMAIN on the shipped explanation drive the "flagged"
            # status: the regular checker's, the 4th-call audit's, or the gate
            # issues an auto-fix did NOT resolve (discarded / unchanged).
            remaining_issues = list(claim_issues) + list(final_audit_issues)
            if revise_record is not None and revise_record["status"] in (
                "discarded", "unchanged",
            ):
                remaining_issues += list(revise_record.get("gate_issues") or [])

            # Soft validators run on the FINAL (possibly revised) explanation:
            # never reject, only queue for review. Skipped on dry-runs.
            soft_warnings: list[str] = (
                [] if dry_run
                else run_preflop_soft_validators(explanation, facts)
            )

            # Append the parallel per-row lists together so they stay aligned.
            rows.append((facts, explanation, difficulty))
            claim_checks.append(claim_check_json)
            row_statuses.append(
                "flagged" if (soft_warnings or remaining_issues) else None
            )
            snapped_freqs_list.append(snapped_real_freqs)
            # Record the exact (deterministic) inputs that produced this row,
            # in the same order rows are written to the CSV.
            record = _prompt_record(
                spot,
                build_explanation_prompt_parts(
                    facts, options, correct, system_prompt=system_prompt
                ),
            )
            if soft_warnings:
                record["validator_warnings"] = soft_warnings
            if claim_issues:
                record["claim_check_issues"] = claim_issues
            if revise_record is not None:
                record["revise"] = revise_record
            if snapped_real_freqs is not None:
                # Real solver mix, kept for the Review/Compare "shown as pure"
                # note (the question itself displays 100%).
                record["snapped_actual_frequencies"] = snapped_real_freqs
            prompt_records.append(record)
        except Exception as exc:  # noqa: BLE001
            # Catch-all on purpose: one bad spot must not abort the batch.
            failures.append(
                _build_failure(
                    spot, facts, options, correct, exc,
                    pack=pack, stakes_bb_dollars=stakes_bb_dollars,
                    live_or_online=live_or_online, game_format=game_format,
                    difficulty=difficulty, display_in_bb=display_in_bb,
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
            row_statuses=row_statuses,
            claim_checks=claim_checks,
            snapped_freqs=snapped_freqs_list,
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
            run_settings={
                "answer_style": answer_style,
                "hero_positions": list(hero_positions) if hero_positions else None,
                "action_contexts": list(action_contexts) if action_contexts else None,
                "player_counts": list(player_counts) if player_counts else None,
                "min_frequency": min_frequency,
                "max_frequency": max_frequency,
                "min_difficulty": min_difficulty,
                "max_difficulty": max_difficulty,
                "min_ev_gap_bb": min_ev_gap_bb,
                "min_villain_line_pct": min_villain_line_pct,
                "min_hero_premise_freq": min_hero_premise_freq,
                "pure_snap_threshold": pure_snap_threshold,
                "equity_runouts": equity_runouts,
                "run_claim_checker": run_claim_checker,
                "revise_pass": revise_pass,
                "final_audit": final_audit,
                "display_in_bb": display_in_bb,
                "stakes_bb_dollars": stakes_bb_dollars,
                "live_or_online": live_or_online,
            },
            # Outcome counters (June 2026 round-2 audit, finding #6: the
            # gate settings were recorded but the skip COUNTS were UI-only,
            # so an audit couldn't confirm the gates fired from the meta
            # alone). run_settings = inputs; counters = what happened.
            counters={
                "worthy_spots_available": len(worthy),
                "nodes_after_filter": len(filtered_nodes),
                "difficulty_filtered_out": difficulty_filtered_out,
                "noise_filtered_out": noise_filtered_out,
                "incoherent_mix_filtered_out": incoherent_mix_filtered_out,
                "rare_line_filtered_out": rare_line_filtered_out,
                "rare_premise_filtered_out": rare_premise_filtered_out,
                "questions_attempted": attempted,
                "questions_written": written,
                "soft_flagged_rows": sum(1 for s in row_statuses if s),
                # Self-revision (4-call audit & auto-fix) outcomes. Only nonzero
                # when revise_pass was on. flagged = fixed + discarded + unchanged.
                "revise_flagged": revise_flagged,
                "revise_fixed": revise_fixed,
                "revise_discarded": revise_discarded,
                "revise_unchanged": revise_unchanged,
            },
            failures=failures,
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
        incoherent_mix_filtered_out=incoherent_mix_filtered_out,
        rare_line_filtered_out=rare_line_filtered_out,
        rare_premise_filtered_out=rare_premise_filtered_out,
        soft_flagged_rows=sum(1 for s in row_statuses if s),
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
    run_settings: dict[str, object] | None = None,
    counters: dict[str, object] | None = None,
    failures: list[PreflopFailure] | None = None,
) -> dict[str, object]:
    """Assemble the ``<stem>.meta.json`` payload for one batch.

    Records the prompt SNAPSHOT (name + text + sha) so a later edit or
    rename of that prompt can't make this batch's provenance ambiguous,
    the run settings (model / temperature / seed), the source pack (so
    Review/Compare can tell a 9-max batch from a 6-max one), the outcome
    counters (what the gates/filters actually skipped -- the audit's
    proof the gates ran), and the per-question inputs in CSV row order.
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
        "run_settings": run_settings or {},
        "counters": counters or {},
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "questions": prompt_records,
        # Spots that didn't make the CSV (validation rejected after the retry
        # budget). Persisted with full context + the rebuilt row so the Review
        # page can show them and promote good ones -- they used to live only
        # in the live Generate result and vanish on navigation.
        "failures": [asdict(f) for f in (failures or [])],
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
    difficulty: object | None = None,
    display_in_bb: bool = False,
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
    candidate = None
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

    # Build the COMPLETE CSV row keeping the rejected explanation, so the
    # Review page can promote it with one click ("keep the exact
    # explanation"). Needs the parsed candidate (a structurally valid
    # GeneratedExplanation -- it passed the format/correct-answer checks and
    # tripped a CONTENT validator), the facts, and the difficulty. number=0 is
    # a placeholder -- the promote step assigns the real No on append.
    # validation_status="needs_review" stamps its provenance. Wrapped so a
    # row-build hiccup can never lose the failure record itself.
    row: dict[str, str] | None = None
    if candidate is not None and facts is not None and difficulty is not None:
        try:
            row = build_preflop_row(
                facts,  # type: ignore[arg-type]
                candidate,
                pack=pack,
                difficulty=difficulty,  # type: ignore[arg-type]
                number=0,
                stakes_bb_dollars=stakes_bb_dollars,
                live_or_online=live_or_online,
                game_format=game_format,
                display_in_bb=display_in_bb,
                validation_status="needs_review",
            )
        except Exception:  # noqa: BLE001
            row = None

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
        row=row,
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
