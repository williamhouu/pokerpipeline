"""Streamlit admin panel for the poker question pipeline.

Run from repo root:
    venv/bin/streamlit run admin_panel/app.py

Six pages selected via the sidebar:

  * Files     -- live disk scan of solves/ and ranges/; per-scenario status
                 indicators. (Upload widgets are visual placeholders --
                 packs / solves are managed on disk for now.)
  * Generate  -- batch configurator for the preflop and postflop paths.
                 Preflop is fully wired: pack -> filters -> deterministic
                 options -> Layer 6 (Anthropic) -> Layer 8 (CSV writer).
                 Runs on a background thread (see admin_panel.jobs) so
                 sidebar / tab switches don't abandon in-flight batches.
                 Postflop path button is disabled until Pio solves land.
  * Review    -- per-question reader for a chosen batch: the full
                 Question, options (correct highlighted), the whole
                 explanation, and the solver frequencies, one card at a
                 time. Light grade buttons (approve / needs-review /
                 reject + note) save to a sidecar <batch>.review.json via
                 admin_panel.review -- the CSV is never mutated. Ranges
                 are tucked into a small expander at the end.
  * History   -- table of every CSV under test_output/preflop_batches/,
                 multi-row select with confirmed bulk delete, per-file
                 preview + download. Each batch lands in its own
                 timestamped file so history persists across sessions.
  * Browse    -- read-only view of test_output/tier1_consolidated.csv
                 (the legacy 70-question demo dataset, for reference).
  * Prompt    -- live editor for the preflop Layer 6 system prompt.
                 Saves to admin_panel/prompts/preflop_system.txt;
                 :func:`pipeline.preflop.explanation_generator.load_preflop_system_prompt`
                 reads it on every batch.
  * Skills    -- catalog browser for the 42 user-facing skill rules
                 from pipeline.skill_tagger. Per-skill status (fires
                 today / awaits postflop / TODO) + trigger description
                 + source code.

Sidebar carries: a live "Job: X/Y" indicator that auto-refreshes
while a batch runs, a "Lifetime API spend" metric summed from
test_output/usage_log.jsonl, and disk-status indicators for ranges
and solves.

Cost tracking: every real-API batch's token usage is captured from
the Anthropic SDK response (see :mod:`admin_panel.usage`), priced
against MODELS_WITHOUT_TEMPERATURE-aware rate cards, and appended to
the JSONL log. The Generate result UI and the sidebar widget both
read from that data.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from admin_panel.prompt_library import PromptLibrary
    from pipeline.plo.fact_extractor import PloFacts
    from pipeline.plo.node_enumerator import PloDecisionNode
    from pipeline.plo.pack import PloPack

# Add the repo root to sys.path so `from pipeline...` imports work when
# Streamlit invokes this script directly. Streamlit's default sys.path
# only includes the script's parent dir (admin_panel/), not the cwd or
# repo root -- pipeline/ lives one level up. Mirrors the same trick
# used in scripts/* and tests/*.
_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

# Load .env from the repo root so ANTHROPIC_API_KEY (and any future
# secrets) are available to Layer 6 when the user clicks Generate. .env
# is gitignored. python-dotenv silently no-ops if the file is missing,
# which is fine -- the admin panel UI still renders without an API key;
# only the actual generation call would fail later.
#
# override=True because some shells / parent processes pre-set
# ANTHROPIC_API_KEY to an empty string, which load_dotenv treats as
# "already set" and refuses to overwrite by default. With override the
# .env value always wins, which is what we want for local dev.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT_FOR_IMPORTS / ".env", override=True)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# Background job registry runs long batches on a thread so Streamlit
# reruns (sidebar clicks, tab switches) don't kill them mid-flight.
# Usage helper computes per-batch $ cost from token totals and appends
# a JSONL log so a lifetime spend total survives admin-panel restarts.
from admin_panel import (  # noqa: E402
    compare,
    gen_settings,
    jobs,
    range_view,
    review,
    usage,
)

# Imports from the pipeline (safe at module load -- these touch no I/O and
# don't require a PioSolver binary or API key to import).
from pipeline.action_history import preflop_order  # noqa: E402
from pipeline.explanation_generator import DEFAULT_TEMPERATURE  # noqa: E402
from pipeline.fact_extractor.hand_class import STRENGTH_BUCKETS  # noqa: E402
from pipeline.preflop.batch import (  # noqa: E402
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    BatchResult,
    active_player_count,
    generate_preflop_batch,
    node_action_context,
)
from pipeline.preflop.fact_extractor import PreflopFacts  # noqa: E402
from pipeline.preflop.node_cache import (  # noqa: E402
    descriptor_to_node,
    load_descriptors,
    load_metadata,
)
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopDecisionNode,
)
from pipeline.preflop.options import ANSWER_STYLE_FROM_RADIO_LABEL  # noqa: E402
from pipeline.preflop.pack import (  # noqa: E402
    PreflopPack,
    discover_packs,
)
from pipeline.preflop.pack import (  # noqa: E402
    clear_registry as clear_preflop_registry,
)
from pipeline.scenario_config import COMMON_STAKE_LEVELS_BB_DOLLARS  # noqa: E402

# Map admin-panel model-radio labels to Anthropic API model identifiers.
# Display strings stay human-readable; the API call needs the ID string.
# Opus 4.7 first = the default everywhere (June 2026) and the production
# model for every batch. Sonnet 4.6 is the cheap/fast option for iterating
# on prompts before committing to a real batch.
_MODEL_LABEL_TO_API: dict[str, str] = {
    "Opus 4.7 (high fidelity, the default)": "claude-opus-4-7",
    "Sonnet 4.6 (cheapest, fastest)": "claude-sonnet-4-6",
}

# Where preflop generation writes its CSV output. Sibling of test_output/
# tier1_consolidated.csv but kept in its own subdir so the Browse page's
# existing data isn't accidentally shadowed.
PREFLOP_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "test_output" / "preflop_batches"
)
# Where postflop generation writes (mirror of pipeline.postflop.run.
# POSTFLOP_OUTPUT_DIR; same path, recomputed here like PREFLOP_OUTPUT_DIR).
POSTFLOP_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "test_output" / "postflop_batches"
)

# JSONL log of every completed real-API batch. Sibling of the CSVs
# the History tab manages. Gitignored. The sidebar's "Lifetime spend"
# widget + the History page's totals read from this file.
USAGE_LOG_PATH = (
    Path(__file__).resolve().parent.parent / "test_output" / "usage_log.jsonl"
)


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, mtime: float, *, as_str: bool = False) -> pd.DataFrame:
    """A batch CSV parsed once per (path, mtime) instead of on every rerun.

    Streamlit re-runs the whole page script on every interaction, so the
    Review / Compare / Browse pages were re-reading and re-parsing their CSV
    on each keystroke or click. This keys on the file's ``mtime`` (passed in,
    not read here) so an edit -- a Review save bumps the mtime -- misses the
    cache and re-reads fresh, while plain navigation reuses the parse.
    ``cache_data`` returns a COPY each call, so callers may mutate the frame
    without corrupting the cache.

    ``mtime`` MUST NOT be renamed with a leading underscore: ``st.cache_data``
    EXCLUDES underscore-prefixed parameters from the cache key. This function
    originally took ``_mtime`` (meant as "passed in, not read here"), which
    silently made the cache key (path, as_str) only -- the first parse of a
    batch was served for the whole session, so every Review edit LOOKED
    unsaved on navigate-back even though the CSV on disk was correct. That
    stale-display was the true root of the recurring "my edit vanished" bug
    (2026-07-04; reproduced + pinned by tests/test_review_autosave.py's
    cache test and the scripts/verify_review_editor_e2e.py AppTest).

    ``as_str``: load every cell as a string with ``""`` for blanks (the
    Compare pages need that; the Review page coerces per-cell via ``_cell``).
    """
    if as_str:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource
def _logged_job_ids() -> set[str]:
    """Job ids already appended to the usage log -- for at-most-once logging.

    MUST be a ``cache_resource`` singleton, NOT a bare module-level set.
    Streamlit re-executes app.py's module body on every *full* rerun (any
    widget toggle), so ``_LOGGED_JOB_IDS = set()`` as a plain global reset to
    empty on each interaction -- and ``_maybe_log_completed_job`` then
    re-appended the still-displayed completed batch's cost every time, which
    is the "lifetime spend jumps a dollar when I change difficulty" bug.
    ``cache_resource`` lives in the server process and survives reruns -- the
    same persistence the jobs registry (an *imported* module) gets for free.
    It resets on a server restart, which is correct: the completed job is
    gone then too, so there's nothing left to re-log.
    """
    return set()

# Where the prompt-editor page writes Layer 6 system-prompt overrides. The
# pipeline checks for this file on every generation call -- a saved edit
# takes effect on the next batch without an admin-panel restart. Mirror
# of pipeline.preflop.explanation_generator._PROMPT_OVERRIDE_PATH; kept in
# sync so the editor and the loader agree on the path. Gitignored so
# experimental prompts don't leak into commits.
PREFLOP_PROMPT_OVERRIDE_PATH = (
    Path(__file__).resolve().parent / "prompts" / "preflop_system.txt"
)
# Same pattern for the postflop system prompt (mirror of
# pipeline.postflop.explanation_generator._POSTFLOP_PROMPT_OVERRIDE_PATH).
POSTFLOP_PROMPT_OVERRIDE_PATH = (
    Path(__file__).resolve().parent / "prompts" / "postflop_system.txt"
)
# Postflop prompt LIBRARY (June 2026) -- named postflop system prompts, the
# postflop analog of the preflop library. Reuses the generic PromptLibrary class
# pointed at its own dir; the ACTIVE entry is mirrored into
# POSTFLOP_PROMPT_OVERRIDE_PATH so load_postflop_system_prompt() (the Generate
# subprocess child, the CLI, the Compare prefill) reads it with no extra wiring.
POSTFLOP_LIBRARY_DIR = (
    Path(__file__).resolve().parent / "prompts" / "postflop_library"
)

# --- repo paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SOLVES_DIR = REPO_ROOT / "solves"
RANGES_DIR = REPO_ROOT / "ranges"
# On-disk cache of enumerated pack nodes (pipeline.preflop.node_cache).
# Gitignored; rebuilt automatically when a pack's files change. Turns the
# ~6s cold pack-walk into a ~90ms descriptor load on every panel restart.
NODE_CACHE_DIR = REPO_ROOT / ".node_cache"
RANGES_SUBDIR = (
    RANGES_DIR / "ryan_preflop_tree" / "PioViewer - NLH 6max 100bb 2.5x Open"
)
TIER1_CSV = REPO_ROOT / "test_output" / "tier1_consolidated.csv"

POSITION_FOLDERS = ("BB", "BTN", "CO", "HJ", "SB", "UTG")
EXPECTED_RANGE_COUNTS = {
    "BB": 3815,
    "BTN": 3649,
    "CO": 3473,
    "HJ": 3293,
    "SB": 2999,
    "UTG": 2977,
}

# The 9-max Monker pack is flat (no position folders): one .rng per
# node-action. Count verified in the June 2026 extraction audit
# (docs/nlhe9_pack_notes.md).
NLHE9_PACK_ID = "monker_nlhe_9max_100bb"
EXPECTED_NLHE9_RANGE_COUNT = 93_235

# Shared "Action faced" tooltip. The bucket rule isn't obvious: it's the
# HIGHEST raise hero faces, so a 3-bet / 4-bet pot keeps that label even when
# someone flat-called along the way -- the "After ... call(s)" buckets are
# specifically a single open that picked up flat-callers (squeeze / overcall).
# Multiway-ness is the separate "Players in the pot" filter's job. (Before the
# June 2026 reorder, any line with a caller was lumped into "After one call",
# so picking that bucket silently pulled in 4-bet pots.)
ACTION_FACED_HELP = (
    "The bucket is the HIGHEST raise you face: a 3-bet or 4-bet pot stays "
    "'Facing 3-bet' / 'Facing 4-bet+' even if someone flat-called earlier in "
    "the hand. 'After one/multiple call(s)' means a SINGLE open that picked up "
    "flat-callers (a squeeze / overcall spot). Empty = all. Use 'Players in "
    "the pot' to control how multiway any bucket is."
)

# Board-texture filter options. The composite descriptor values are produced
# by pipeline/fact_extractor/board_texture.py:_composite() -- six categories
# spanning the dryness ladder. Suit + pair status are separate axes the user
# can layer on for tighter filtering.
BOARD_COMPOSITES = (
    "dry",
    "static",
    "semi_wet",
    "wet",
    "very_wet",
    "dynamic",
)
BOARD_SUIT_DISTRIBUTIONS = ("rainbow", "two_tone", "monotone", "flush_on_board")
BOARD_PAIR_STATUSES = ("unpaired", "paired", "trips_on_board")


# --- registered scenario metadata ------------------------------------------
# Hand-extracted from pipeline/scenario_spec.py + the templates/ folder
# (14 Tier-1 scenarios). Hard-coded here so this preview runs without
# importing the pipeline (which would otherwise need a working PioSolver
# binary on PATH for some imports). Will be replaced with a proper import
# once the real generation backend is wired in.
@dataclass
class ScenarioMeta:
    name: str
    format: str  # "Cash" / "MTT"
    stack_bb: int
    table_size: int  # 6 / 9 / etc.
    pot_type: str  # "SRP" / "3-bet pot" / "4-bet pot"
    preflop_action: str  # human-readable summary


SCENARIOS: list[ScenarioMeta] = [
    # --- 5 SRP ---
    ScenarioMeta(
        "Cash6max_100bb_BTN_open_BB_call",
        "Cash",
        100,
        6,
        "SRP",
        "BTN opens 2.5bb → BB calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_BTN_open_SB_call",
        "Cash",
        100,
        6,
        "SRP",
        "BTN opens 2.5bb → SB calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_CO_open_BB_call",
        "Cash",
        100,
        6,
        "SRP",
        "CO opens 2.5bb → BB calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_HJ_open_BB_call",
        "Cash",
        100,
        6,
        "SRP",
        "HJ opens 2.5bb → BB calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_SB_open_BB_call",
        "Cash",
        100,
        6,
        "SRP",
        "SB opens 3bb → BB calls",
    ),
    # --- 5 3-bet pots ---
    ScenarioMeta(
        "Cash6max_100bb_BTN_open_BB_3bet_BTN_call",
        "Cash",
        100,
        6,
        "3-bet pot",
        "BTN opens → BB 3-bets → BTN calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_BTN_open_SB_3bet_BTN_call",
        "Cash",
        100,
        6,
        "3-bet pot",
        "BTN opens → SB 3-bets → BTN calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_CO_open_BTN_3bet_CO_call",
        "Cash",
        100,
        6,
        "3-bet pot",
        "CO opens → BTN 3-bets → CO calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_HJ_open_BB_3bet_HJ_call",
        "Cash",
        100,
        6,
        "3-bet pot",
        "HJ opens → BB 3-bets → HJ calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_UTG_open_BB_3bet_UTG_call",
        "Cash",
        100,
        6,
        "3-bet pot",
        "UTG opens → BB 3-bets → UTG calls",
    ),
    # --- 4 4-bet pots ---
    ScenarioMeta(
        "Cash6max_100bb_BTN_open_BB_3bet_BTN_4bet_BB_call",
        "Cash",
        100,
        6,
        "4-bet pot",
        "BTN opens → BB 3-bets → BTN 4-bets → BB calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_CO_open_BTN_3bet_CO_4bet_BTN_call",
        "Cash",
        100,
        6,
        "4-bet pot",
        "CO opens → BTN 3-bets → CO 4-bets → BTN calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_HJ_open_BB_3bet_HJ_4bet_BB_call",
        "Cash",
        100,
        6,
        "4-bet pot",
        "HJ opens → BB 3-bets → HJ 4-bets → BB calls",
    ),
    ScenarioMeta(
        "Cash6max_100bb_UTG_open_BB_3bet_UTG_4bet_BB_call",
        "Cash",
        100,
        6,
        "4-bet pot",
        "UTG opens → BB 3-bets → UTG 4-bets → BB calls",
    ),
]


# --- disk-state detection (the live magic) ---------------------------------
def count_cfrs(scenario_name: str) -> int:
    """Return how many .cfr files exist for this scenario, or 0 if missing."""
    folder = SOLVES_DIR / scenario_name
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("*.cfr")))


# --- preflop pack helpers (cached, Streamlit reruns the script on every
# interaction so without cache we'd re-scan + re-register packs each time) -----
@st.cache_resource
def _cached_preflop_packs() -> tuple[PreflopPack, ...]:
    """Discover and register the preflop packs once per Streamlit session."""
    clear_preflop_registry()
    return discover_packs(RANGES_DIR)


@st.cache_resource
def _cached_pack_descriptors(pack_id: str) -> tuple[tuple, int]:
    """ONE pack's compact node descriptors (+ range-file count) from the
    on-disk node cache, built once if absent.

    Descriptors are plain strings/floats (see
    :mod:`pipeline.preflop.node_cache`): ~90ms to load for the 9-max pack
    versus ~6s to re-parse all 93k range files. Everything downstream --
    the fast metadata views and full-node reconstruction -- derives from
    this single source.
    """
    packs = [p for p in _cached_preflop_packs() if p.pack_id == pack_id]
    if not packs:
        return (), 0
    return load_descriptors(packs[0], NODE_CACHE_DIR)


def _meta_derive(node: PreflopDecisionNode) -> tuple[str, str, int]:
    """Per-node metadata stored in the fast metadata cache: hero actor,
    action context, and player count. Runs once per node at cache BUILD
    time on full walked nodes, so it reuses the real ``node_action_context``
    / ``active_player_count`` (values match the full-node path exactly). If
    this logic changes, bump ``node_cache.META_CACHE_VERSION``."""
    return (node.actor, node_action_context(node), active_player_count(node))


@st.cache_resource
def _cached_pack_metadata(pack_id: str) -> tuple[tuple, int]:
    """ONE pack's precomputed per-node metadata (+ file count) from the
    small on-disk metadata cache.

    ``rows`` is ``[(actor, context, player_count), ...]`` -- everything the
    list / filter / count views need WITHOUT materialising a single node.
    Tens of ms to load (the ~1 MB meta.pkl), versus the ~700ms rebuild or
    ~6s parse-walk it replaces. The sidebar file-count, Generate, Compare,
    and the filter recount all read this.
    """
    packs = [p for p in _cached_preflop_packs() if p.pack_id == pack_id]
    if not packs:
        return (), 0
    return load_metadata(packs[0], NODE_CACHE_DIR, _meta_derive)


@st.cache_resource
def _cached_preflop_nodes_by_actor(
    pack_id: str,
) -> dict[str, tuple[PreflopDecisionNode, ...]]:
    """FULL nodes for ONE pack, grouped by hero position.

    Reconstructed from the compact on-disk cache rather than re-parsed from
    93k files (~2s vs ~6s, byte-identical to the parse-walk). Only the paths
    that need a complete node -- the Range viewer and the prompt-preview
    sampler -- pull this; the list / filter / count views use the much
    cheaper :func:`_cached_pack_metadata` instead. Per-pack (cached per
    pack_id) so the 6-max and 9-max trees never mix.
    """
    descriptors, _ = _cached_pack_descriptors(pack_id)
    by_actor: dict[str, list[PreflopDecisionNode]] = defaultdict(list)
    for descr in descriptors:
        node = descriptor_to_node(descr, pack_id)
        by_actor[node.actor].append(node)
    return {actor: tuple(nodes) for actor, nodes in by_actor.items()}


# Where the NLHE Generate page persists its pack choice (hidden file so the
# batch-dir *.csv globs never see it). Same mechanism as the PLO page's
# settings snapshot.
PREFLOP_GEN_SETTINGS_PATH = PREFLOP_OUTPUT_DIR / ".preflop_generate_settings.json"


@st.cache_resource
def _cached_node_filter_meta(
    pack_id: str,
) -> dict[str, tuple[tuple[str, int], ...]]:
    """Per-actor (action_context, player_count) tuples for live filter counts.

    The Generate page recomputes "how many nodes match these filters" on
    EVERY widget interaction; filtering these precomputed tuples takes
    milliseconds. Grouped from :func:`_cached_pack_metadata` (precomputed on
    disk), so even the first build is just a group-by, never a node walk.
    Also the source for the Generate node count + position list and the
    Compare seat list.
    """
    rows, _ = _cached_pack_metadata(pack_id)
    by_actor: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for actor, ctx, players in rows:
        by_actor[actor].append((ctx, players))
    return {actor: tuple(metas) for actor, metas in by_actor.items()}


@st.cache_resource
def _cached_ranges_index(
    pack_id: str,
) -> tuple[
    dict[str, PreflopDecisionNode],
    dict[str, tuple[str, ...]],
    dict[str, str],
]:
    """Prebuilt lookup tables for the Range viewer.

    Returns ``(node_by_id, sorted_ids_by_actor, display_label_by_id)``.
    Building these per rerun meant constructing + sorting 44k node-id
    strings and recomputing every option's action context on each click --
    the viewer's 10-second lag on the 9-max pack. Built once per pack.
    """
    by_actor = _cached_preflop_nodes_by_actor(pack_id)
    node_by_id: dict[str, PreflopDecisionNode] = {}
    ids_by_actor: dict[str, tuple[str, ...]] = {}
    labels: dict[str, str] = {}
    for actor, nodes in by_actor.items():
        ordered = sorted(nodes, key=lambda n: n.node_id)
        ids_by_actor[actor] = tuple(n.node_id for n in ordered)
        for n in ordered:
            node_by_id[n.node_id] = n
            labels[n.node_id] = f"{node_action_context(n)} · {n.node_id}"
    return node_by_id, ids_by_actor, labels


@st.cache_resource
def _cached_ranges_by_hist(
    pack_id: str,
) -> dict[tuple, PreflopDecisionNode]:
    """``history_before`` → decision node, for the Range viewer's tree walk.

    A full action history uniquely identifies whose turn it is, so this maps
    every history to its node. The action-tree navigator finds the child of
    any (node, action) in O(1): append the chosen action to the node's history
    and look it up. The empty tuple ``()`` keys the opening node (first to
    act). Built once per pack (cache_resource), like
    :func:`_cached_ranges_index`.
    """
    node_by_id, _ids, _labels = _cached_ranges_index(pack_id)
    return {n.history_before: n for n in node_by_id.values()}


def _pack_display_framing(pack: PreflopPack) -> tuple[str, float]:
    """Pack-aware ``(venue, stakes_bb_dollars)`` display default.

    Venue/stakes are cosmetic -- every solver number is in bb -- but the
    framing should match the pack's shape. The Monker packs
    (``monker_nlhe``: 9-handed / short-stack, capped-rake) and the 8-max 200bb
    pack (``gto_preflop_8max``: a live deep-cash structure -- the DB's own
    gametype is ``Cash8mLiveGeneral``) read as Live $1/$2 games; the 6-max Ryan
    pack keeps the original Online $0.25/$0.50 framing. Both the Generate page
    (as the widget default) and the Compare page (which has no venue widget) call
    this, so a pack frames identically wherever it's generated -- the
    source-of-truth that keeps the two paths from drifting (they did: Compare
    used to hardcode Online).
    """
    is_live = pack.grammar_name in ("monker_nlhe", "gto_preflop_8max")
    return ("Live", 2.00) if is_live else ("Online", 0.50)


def _select_preflop_pack(widget_key: str) -> PreflopPack | None:
    """Render the pack selector and return the chosen pack.

    With a single discovered pack there is no choice to make -- show a
    caption and return it. The widget's session value persists across
    pages that share ``widget_key``; the Generate page additionally
    snapshots it to disk on launch.
    """
    packs = _cached_preflop_packs()
    if not packs:
        return None

    def _label(pack_id: str) -> str:
        p = next(pk for pk in packs if pk.pack_id == pack_id)
        return f"{p.table_size}-max · {p.stack_depth_bb}bb · {p.pack_id}"

    if len(packs) == 1:
        st.caption(f"Range pack: **{_label(packs[0].pack_id)}**")
        return packs[0]
    ids = [p.pack_id for p in packs]
    choice = st.selectbox(
        "Range pack",
        options=ids,
        format_func=_label,
        key=widget_key,
        help=(
            "Which preflop solve the batch samples from. 6-max = Ryan's "
            "PioViewer pack (rake 4%/0.3bb); 9-max = the Monker pack "
            "(rake 10%/3bb -- visibly tighter ranges)."
        ),
    )
    return next(p for p in packs if p.pack_id == choice)


@st.cache_resource(show_spinner="Building a sample spot for the prompt preview…")
def _preview_sample_spot() -> tuple[PreflopFacts, list[str], str] | None:
    """One representative (facts, options, correct_answer) for the prompt
    preview. Cached for the session (extract_facts runs equity sims).

    Picks a facing-a-raise spot so the SOLVER DATA block is rich (villain
    stats + equities + blockers + concept tags) -- i.e. shows the most.
    """
    from dataclasses import replace  # noqa: PLC0415

    from pipeline.preflop.ev_engine import compute_break_even_equity  # noqa: PLC0415
    from pipeline.preflop.fact_extractor import extract_facts  # noqa: PLC0415
    from pipeline.preflop.grammars.types import PreflopActionType  # noqa: PLC0415
    from pipeline.preflop.options import build_options  # noqa: PLC0415
    from pipeline.preflop.spot_sampler import (  # noqa: PLC0415
        enumerate_spots_for_node,
    )

    packs = _cached_preflop_packs()
    if not packs:
        return None
    pack = packs[0]
    raise_types = (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    for nodes in _cached_preflop_nodes_by_actor(pack.pack_id).values():
        for node in nodes:
            if not any(a.action_type in raise_types for a in node.history_before):
                continue
            for spot in enumerate_spots_for_node(node, min_total_weight=0.05):
                try:
                    facts = extract_facts(spot, pack, equity_runouts=80)
                    facts = replace(
                        facts,
                        break_even_equity=compute_break_even_equity(facts, pack),
                    )
                    options, correct = build_options(facts, style="auto")
                except Exception:  # noqa: BLE001
                    continue
                return facts, options, correct
    return None


@st.cache_resource
def ranges_pack_status() -> tuple[bool, dict[str, int]]:
    """Return (is_complete, per_position_counts) for the 6-max ranges pack.

    Computed ONCE per session (was ``cache_data(ttl=60)``). The sidebar
    renders this on EVERY rerun of every page, and the underlying glob
    stats ~20k files. With a 60s TTL, any click more than a minute after
    the last one re-globbed all 20k files on the spot -- so reviewing at a
    human pace made every next/prev pay the glob (June 2026 slowness
    report). The 6-max pack is stable within a session; after a manual
    re-extraction, restart the panel (or clear caches) to re-check. Same
    once-per-session pattern as :func:`_monker_pack_file_count`.
    """
    counts = {}
    for pos in POSITION_FOLDERS:
        folder = RANGES_SUBDIR / pos
        counts[pos] = len(list(folder.glob("*.txt"))) if folder.is_dir() else 0
    is_complete = all(
        counts[pos] == EXPECTED_RANGE_COUNTS[pos] for pos in POSITION_FOLDERS
    )
    return is_complete, counts


def _monker_pack_file_count(pack_id: str) -> int:
    """A pack's range-file count, served from the small metadata cache so the
    sidebar never re-globs 93k files (the old rglob was ~650ms on every cold
    render) and never loads the 18 MB descriptors just for a count."""
    return _cached_pack_metadata(pack_id)[1]


# --- page: Files -----------------------------------------------------------
def render_files_page() -> None:
    st.title("Files")
    st.caption(
        "Manage the solves and ranges the pipeline reads from. "
        "Drag a zip into the upload areas to add new ones."
    )

    # Ranges section
    st.subheader("📂 Ranges (preflop pack)")
    ranges_ok, ranges_counts = ranges_pack_status()
    total_files = sum(ranges_counts.values())
    if ranges_ok:
        st.success(
            f"✅ Ranges pack is complete: {total_files:,} files across "
            "6 position folders."
        )
    elif total_files == 0:
        st.error(
            "❌ No ranges pack uploaded yet. Pipeline cannot generate "
            "questions without preflop ranges."
        )
    else:
        st.warning(
            f"⚠️ Ranges pack is partial: {total_files:,} files present, expected 20,206."
        )

    ranges_df = pd.DataFrame(
        [
            {
                "Position": pos,
                "Files present": f"{ranges_counts[pos]:,}",
                "Expected": f"{EXPECTED_RANGE_COUNTS[pos]:,}",
                "Status": "✅"
                if ranges_counts[pos] == EXPECTED_RANGE_COUNTS[pos]
                else ("⚠️" if ranges_counts[pos] > 0 else "❌"),
            }
            for pos in POSITION_FOLDERS
        ]
    )
    st.dataframe(ranges_df, hide_index=True, use_container_width=True)
    st.file_uploader(
        "Replace ranges pack (drop a zip)",
        type=["zip"],
        key="ranges_upload",
        disabled=True,
        help="Upload not wired in this preview. The ranges/ folder is "
        "managed manually for now.",
    )

    # 9-max Monker pack (flat .rng dir under nlhe9_ranges/, gitignored).
    st.subheader("📂 Ranges (NLHE 9-max Monker pack)")
    nlhe9_count = _monker_pack_file_count(NLHE9_PACK_ID)
    if nlhe9_count == EXPECTED_NLHE9_RANGE_COUNT:
        st.success(
            f"✅ 9-max pack is complete: {nlhe9_count:,} .rng node files "
            "(44,058 decision nodes)."
        )
    elif nlhe9_count == 0:
        st.error(
            "❌ 9-max pack not found. Extract it to "
            "`nlhe9_ranges/ranges/Hold'em/9-way/100bb[10p-3bb]/` (the "
            "canonical 3.4GB .mkr lives in the team Dropbox; see "
            "docs/nlhe9_pack_notes.md)."
        )
    else:
        st.warning(
            f"⚠️ 9-max pack is partial: {nlhe9_count:,} files present, "
            f"expected {EXPECTED_NLHE9_RANGE_COUNT:,}. Re-extract from the "
            "Dropbox .mkr export."
        )

    st.divider()

    # Solves section
    st.subheader("📂 Solves (PioSolver .cfr files)")
    rows = []
    total_cfrs = 0
    for s in SCENARIOS:
        n = count_cfrs(s.name)
        total_cfrs += n
        if n == 25:
            status = "✅"
            note = "complete (25/25 flops)"
        elif n == 0:
            status = "❌"
            note = "missing"
        else:
            status = "⚠️"
            note = f"partial ({n}/25 flops)"
        rows.append(
            {
                "Scenario": s.name,
                "Pot type": s.pot_type,
                ".cfrs": f"{n}/25",
                "Status": status,
                "Note": note,
            }
        )

    if total_cfrs == 0:
        st.error(
            "❌ No solves uploaded yet. Waiting on William to share the "
            "current `solves/` folder via Google Drive."
        )
    elif total_cfrs == len(SCENARIOS) * 25:
        st.success(
            f"✅ All scenarios solved: {total_cfrs} .cfr files across "
            f"{len(SCENARIOS)} scenarios."
        )
    else:
        st.warning(
            f"⚠️ Partial solves: {total_cfrs} .cfr files across "
            f"{len(SCENARIOS)} scenarios."
        )

    solves_df = pd.DataFrame(rows)
    st.dataframe(solves_df, hide_index=True, use_container_width=True)
    st.file_uploader(
        "Add solves (drop per-scenario zips)",
        type=["zip"],
        key="solves_upload",
        disabled=True,
        accept_multiple_files=True,
        help="Upload not wired in this preview. solves/ is managed manually for now.",
    )


# --- page: Generate --------------------------------------------------------
def render_generate_page() -> None:
    st.title("Generate questions")

    # --- MODE TOGGLE: Postflop vs Preflop ---
    # Preflop uses pipeline/preflop/* (no Pio solves needed; reads
    # ranges/ryan_preflop_tree/ instead). Postflop generates from third-party
    # .db solves picked in solves/postflop/ (each solve self-describes).
    mode = st.radio(
        "Mode",
        options=["Postflop", "Preflop"],
        index=1,  # default to Preflop
        horizontal=True,
        help=(
            "Postflop generates from a specific `.db` solve you pick (drop them "
            "in `solves/postflop/`). Preflop uses the preflop range packs. Both "
            "run end-to-end."
        ),
        key="generate_mode",
    )
    st.divider()

    if mode == "Preflop":
        _render_generate_page_preflop()
        return

    # --- POSTFLOP PATH: pick a specific .db solve and generate from it ---
    _render_generate_page_postflop()


# Folder the postflop solve picker scans by default (gitignored; the user drops
# their .db solves here, or points the picker elsewhere).
_POSTFLOP_SOLVES_DIR = Path(__file__).resolve().parent.parent / "solves" / "postflop"


def _render_generate_page_postflop() -> None:
    """Solve-centric postflop Generate.

    Each ``.db`` is ONE scenario on ONE flop, and it describes itself (table
    size, positions, stack, format) from its own metadata. So the UX is a
    *picker*, not a filter cascade: choose the solve, set a few per-solve knobs
    (how many, whose decisions, variety), and run a real subprocess batch. The
    old board-texture / scenario-cascade UI assumed a large ``.cfr`` library
    that doesn't exist yet -- with a handful of specific solves, picking one at a
    time is the right tool. (Cross-solve texture filters become useful later, at
    library scale.)
    """
    from pipeline.postflop.adapters.sqlite_db import discover_db_solves  # noqa: PLC0415
    from pipeline.postflop.options import (  # noqa: PLC0415
        ANSWER_STYLE_FROM_RADIO_LABEL,
    )
    from pipeline.postflop.run import (  # noqa: PLC0415
        POSTFLOP_OUTPUT_DIR,
        generate_full_hand_batch_from_db,
        generate_postflop_batch_from_db,
        generate_preflop_entry_batch_from_db,
    )

    _render_postflop_job_panel()

    st.caption(
        "Generate questions from one postflop solve. Every solve carries its own "
        "scenario (table size, positions, stack, flop) — pick it and go."
    )

    # 1. Solves folder + discovery (recursive scan; each solve self-describes).
    folder = st.text_input(
        "Solves folder",
        value=str(_POSTFLOP_SOLVES_DIR),
        key="postflop_solves_dir",
        help=(
            "Folder holding your `.db` postflop solves (scanned recursively). "
            "Drop solve files here, or point this at wherever you keep them."
        ),
    )
    summaries = discover_db_solves(folder)
    usable = [s for s in summaries if s.ok]
    broken = [s for s in summaries if not s.ok]

    if not usable:
        st.info(
            f"No readable `.db` solves found in `{folder}`. Drop your solve "
            "files there (each is one scenario on one flop), or change the "
            "folder above."
        )
        if broken:
            with st.expander(f"⚠️ {len(broken)} file(s) couldn't be read"):
                for b in broken:
                    st.caption(f"`{Path(b.path).name}` — {b.error}")
        return

    # 2. Pick a solve (labelled by its own metadata).
    st.subheader("1. Pick a solve")
    by_path = {s.path: s for s in usable}
    picked = st.selectbox(
        f"{len(usable)} solve(s) found",
        options=[s.path for s in usable],
        format_func=lambda p: f"{Path(p).name}   —   {by_path[p].label}",
        key="postflop_pick_solve",
    )
    solve = by_path[picked]
    if broken:
        st.caption(f"({len(broken)} other file(s) in the folder couldn't be read.)")

    # 3. Solve details -- the "what am I generating from" confirmation.
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Table", f"{solve.table_size}-max" if solve.table_size else "—")
        cols[1].metric("Stack", f"{solve.stack_bb:g}bb" if solve.stack_bb else "—")
        cols[2].metric("Matchup", f"{solve.ip_position} vs {solve.oop_position}")
        cols[3].metric("Flop", solve.flop_pretty)
        # Structure row -- the knobs that DIFFER across solves (game type, rake,
        # ante). Surfaced explicitly because down the line many solves will mix
        # structures and the framing must be unambiguous when you pick one.
        s = st.columns(3)
        s[0].metric("Game", "Tournament" if solve.game_format == "tournament" else "Cash")
        s[1].metric("Rake", solve.rake_pretty)
        s[2].metric("Ante", solve.ante_pretty)
        meta_bits = [solve.spot] if solve.spot else []
        if solve.solve_date:
            meta_bits.append(f"solved {solve.solve_date}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

    st.divider()

    # 3b. Question MODE -- independent spots (today), full-hand play-throughs, or
    # standalone preflop-entry questions. The mode steers which run.py driver the
    # GENERATE button calls + a couple of mode-only controls below.
    pf_mode_label = st.radio(
        "Question mode",
        options=[
            "Independent spots",
            "Full hands (preflop → river)",
            "Preflop only (entry decisions)",
        ],
        index=0,
        horizontal=True,
        key="postflop_question_mode",
        help=(
            "**Independent spots** — one standalone question per decision (today's "
            "behaviour). **Full hands** — linked preflop→river sequences a user "
            "plays street by street (grouped by `hand_id`, ordered by "
            "`sequence_index`); the count below is the number of HANDS. **Preflop "
            "only** — standalone preflop-entry questions read from this solve's "
            "flop-entry frequencies (its call/open mix)."
        ),
    )
    pf_mode = (
        "full_hand" if pf_mode_label.startswith("Full")
        else "preflop" if pf_mode_label.startswith("Preflop")
        else "spots"
    )
    pf_include_villain = False
    if pf_mode == "full_hand":
        pf_include_villain = st.checkbox(
            "Also ask the villain's decisions (flip the frame)",
            value=False,
            key="postflop_include_villain",
            help=(
                "For each hand, also emit the OTHER player's decision line on the "
                "same runout as a SEPARATE hand — so one tree yields ~8+ questions "
                "(both perspectives). Off = hero-only."
            ),
        )
        st.caption(
            "Full-hand depth = the deepest *worthy* decision on each line "
            "(widen the frequency window in Advanced for deeper hands)."
        )
    elif pf_mode == "preflop":
        st.caption(
            "Preflop-entry questions come from the solve's flop-entry range "
            "frequencies (a solve INPUT, not a solved decision — no EVs). Honest "
            "for the entry that started this hand; not a substitute for the "
            "preflop range-pack pipeline."
        )

    st.divider()

    # 4. What to ask.
    st.subheader("2. What to ask")

    is_spots = pf_mode == "spots"
    is_full = pf_mode == "full_hand"
    is_preflop = pf_mode == "preflop"

    # Per-mode DEFAULTS: every variable the GENERATE button reads must exist even
    # when its widget is hidden for the current mode (so a hidden control never
    # NameErrors). The relevant driver simply doesn't receive the irrelevant ones.
    streets = ("flop", "turn", "river")  # full-hand always spans the hand
    diversify = False
    strength_filter: list[str] = []
    decision_filter: list[str] = []
    trap_difficulty = False
    use_ev_gap = False
    min_ev_gap: float | None = None
    max_nodes: int = 600
    quality_gate = True
    min_premise_freq: float | None = None
    pf_run_claim_checker = False
    pf_revise_pass = False
    pf_final_audit = False
    pf_claim_checker_prompt: str | None = None

    col1, col2 = st.columns(2)
    with col1:
        if is_full:
            total = st.number_input(
                "Number of HANDS",
                min_value=1,
                max_value=2000,
                value=10,
                step=1,
                key="postflop_total",
                help="Each HAND becomes SEVERAL linked questions (a preflop entry "
                "plus one per street the hero acts on), so the CSV row count is "
                "much higher than the hand count — see the estimate below.",
            )
        else:
            total = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=5000,
                value=20,
                step=5,
                key="postflop_total",
            )
    with col2:
        hero_help = (
            "Which player's entry decision to ask about. The caller's "
            "defend is the interesting mix; the opener mostly just opens."
            if is_preflop else
            "Which player's decisions each hand follows. Both = a mix of hands "
            "from each perspective." if is_full else
            "Generate spots where this player is to act. Both = a mix."
        )
        heroes = st.multiselect(
            "Whose decisions to ask about",
            options=[solve.ip_position, solve.oop_position],
            default=[solve.ip_position, solve.oop_position],
            key="postflop_heroes",
            help=hero_help,
        )

    if is_full:
        n = int(total)
        st.info(
            f"🃏 **Full-hand mode counts HANDS, not questions.** One hand is a "
            f"preflop → river play-through: a preflop-entry question plus the "
            f"hero's decision on each street, all linked by `hand_id` and played "
            f"in order. **{n} hands ≈ {n * 4}–{n * 6} questions** in the CSV "
            f"(depends how deep each line runs)."
            + ("  Villain-frame is on, so expect roughly double — each tree also "
               "yields the villain's line." if pf_include_villain else "")
        )
    elif is_preflop:
        st.caption(
            "Preflop-entry questions: one per hand the chosen seat enters with "
            "(no streets, no play-through linkage)."
        )

    # Streets — ONLY spot mode lets you pick. A full-hand play-through always
    # spans the whole hand (preflop→river); a preflop-entry question has no
    # streets. Hiding the picker in those modes is why it no longer appears.
    if is_spots:
        streets = st.multiselect(
            "Streets to ask about",
            options=["flop", "turn", "river"],
            default=["flop", "turn", "river"],
            key="postflop_streets",
            help=(
                "Which streets to generate questions from. The solve covers the whole "
                "tree off its one flop, so turn/river questions branch from that flop. "
                "Turn/river have huge node counts — they're down-sampled (see advanced)."
            ),
        )

    # Variety / curation filters — spot mode only (full-hand assembles whole
    # connected lines; preflop-entry has no postflop curation to apply).
    if is_spots:
        diversify = st.toggle(
            "Vary the decision types (recommended)",
            value=True,
            key="postflop_diversify",
            help=(
                "Round-robin across streets and decision types (c-bet / barrel / "
                "probe / facing-bet / check / river bet) so a fill-to-N batch isn't "
                "dominated by one street or type."
            ),
        )
        # Curation filters -- the postflop analog of preflop's hand-strength +
        # action-faced filters. Applied BEFORE the equity sim (no wasted spend).
        from pipeline.postflop.spot_selection import (  # noqa: PLC0415
            DECISION_TYPES,
            STRENGTH_BUCKETS,
        )

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            strength_filter = st.multiselect(
                "Hero hand strength (filter)",
                options=list(STRENGTH_BUCKETS),
                default=[],
                key="postflop_strength_filter",
                help="Keep only spots where hero's made-hand bucket is one of these. "
                "Empty = all. The postflop analog of the preflop hand-strength filter.",
            )
        with fcol2:
            decision_filter = st.multiselect(
                "Decision type (filter)",
                options=list(DECISION_TYPES),
                default=[],
                key="postflop_decision_filter",
                help="Keep only these decision situations (analog of preflop's "
                "'action faced'). Empty = all. Situation-based, so it never leaks the "
                "answer.",
            )

    # Answer option style — all modes.
    style_label = st.radio(
        "Answer option style",
        options=list(ANSWER_STYLE_FROM_RADIO_LABEL),
        index=1,  # GTO (always/mostly) — matches the preflop default the team uses
        horizontal=True,
        key="postflop_style",
        help=(
            "Basic = plain action labels (Check / Bet 33% / Bet 75%). "
            "GTO = the Always/Mostly spectrum on 2-action spots (plain on 3+ "
            "sizes). Auto = plain when one action clearly dominates, else GTO."
        ),
    )
    answer_style = ANSWER_STYLE_FROM_RADIO_LABEL[style_label]

    # Full-hand only (July 2026): the hand-level difficulty band + the
    # razor's-edge toggle for the PACK-BACKED preflop leg, and a note that
    # the preflop leg auto-upgrades when a preflop pack matches the solve.
    fh_razor_difficulty = False
    fh_band = None
    if is_full:
        st.caption(
            "🃏 The preflop leg is built from the closest-matching preflop "
            "range pack automatically (same table size, stack and open "
            "size), giving it real EVs, ranges and the math panel. When no "
            "pack matches, it falls back to the entry-derived question."
        )
        fh_razor_difficulty = st.checkbox(
            "🔪 Razor's-edge difficulty on the preflop leg",
            value=False,
            key="fullhand_razor_difficulty",
            help="Applies the preflop razor's-edge rating (range-boundary "
            "hands rate 2000-2600) to pack-backed preflop legs. Same rule "
            "as the preflop Generate page; see its ℹ️ for details.",
        )
        fh_preset = st.radio(
            "Hand difficulty (the hand's HARDEST decision)",
            options=["Mixed", "Easy", "Medium", "Hard", "Custom"],
            index=0,
            horizontal=True,
            key="fullhand_difficulty_preset",
            help="Filters PLAY-THROUGHS by hand_difficulty = the max of the "
            "legs' ratings (a hand demands what its hardest decision "
            "demands; easy connective legs don't dilute it). Every leg "
            "also keeps its own per-question rating for the app.",
        )
        _fh_bands = {
            "Easy": (400, 1300), "Medium": (1300, 2100),
            "Hard": (2100, 3200), "Mixed": None, "Custom": (400, 3200),
        }
        fh_band = _fh_bands[fh_preset]
        if fh_preset == "Custom":
            fh_band = st.slider(
                "Hand difficulty band", min_value=400, max_value=3200,
                value=(400, 3200), step=50,
                key="fullhand_difficulty_slider",
            )

    # Trap-aware difficulty + the difficulty/skills explainers apply to POSTFLOP
    # legs (spots + full-hand). Preflop-entry uses a frequency-only difficulty,
    # so they don't apply there.
    if not is_preflop:
        trap_difficulty = st.checkbox(
            "🪤 Trap-aware difficulty",
            value=False,
            key="postflop_trap_difficulty",
            help="Floor counterintuitive PURE spots to a GRADED Medium-to-Hard "
            "rating (1800 to 2900, scaled by how far equity sits on the wrong "
            "side of the price). A 'trap' is where the solver's action "
            "contradicts the equity-vs-price baseline (folds a hand whose "
            "equity clears the price, or continues one clearly below it). "
            "Score only -- never changes the answer/options/prose. Off by "
            "default; recommended for Medium/Hard batches. Heads-up "
            "facing-a-bet spots only.",
        )
        with st.popover("ℹ️ How is postflop difficulty calculated?"):
            from pipeline.postflop.difficulty import (  # noqa: PLC0415
                W_CONCEPT,
                W_FREQ,
                W_HAND,
            )
            from pipeline.trap_grading import (  # noqa: PLC0415
                TRAP_FLOOR_MAX,
                TRAP_FLOOR_MIN,
            )

            st.markdown(
                f"A **3-axis** weighted ease score (like preflop/PLO), mapped to "
                f"`3000 - ease*2500` clipped to [400, 3200]:\n\n"
                f"- **Frequency** (weight {W_FREQ:.0%}) — how dominant the top action "
                "is. A near-coin-flip (55%) is hard, a pure (100%) action is easy.\n"
                f"- **Concept** (weight {W_CONCEPT:.0%}) — how hard the strategic "
                "frame is. A value bet is easy, a thin bluff-catch / trap-check is "
                "hard.\n"
                f"- **Hand class** (weight {W_HAND:.0%}) — U-shaped: premium made "
                "hands and clear air are easy, the medium/marginal middle is hard.\n\n"
                "The **EV gap is NOT scored** — a worthy postflop spot mixes at ~0 EV "
                "gap by construction, so it adds no signal (it's kept only as the "
                "`easy_ev` diagnostic column).\n\n"
                f"**🪤 Trap-aware (opt-in):** floors a counterintuitive pure spot to "
                f"a GRADED {TRAP_FLOOR_MIN}-{TRAP_FLOOR_MAX} rating (scaled by how "
                "far equity sits on the wrong side of the price), so a deceptive "
                "but clear-cut spot rates Medium-to-Hard. Otherwise a pure spot "
                "can't exceed ~Medium."
            )

        _render_postflop_skills_explainer()

    with st.expander("Worthiness window + filters (advanced)"):
        # The frequency window applies to EVERY mode: it gates which spots are
        # worthy (spots), which decisions seed a hand (full-hand), and which
        # entry hands are a real mix (preflop).
        if is_full:
            freq_help = (
                "A street decision must sit in this band to SEED a hand (so every "
                "hand has at least one genuine decision). Widen the low end for "
                "deeper hands that reach the river more often."
            )
        elif is_preflop:
            freq_help = (
                "Keep entry hands the seat plays at a genuinely-mixed frequency "
                "(a real defend), not pure folds/calls."
            )
        else:
            freq_help = (
                "Keep spots where the top action sits in this band. Below 65% "
                "reads as 'no clear answer'; 99%+ is a gimme. Mirrors preflop."
            )
        freq_lo, freq_hi = st.slider(
            "Solver frequency window (%) — how dominant the best action is",
            min_value=50,
            max_value=100,
            value=(65, 99),
            key="postflop_freq",
            help=freq_help,
        )
        # EV-gap / node-cap / quality / premise gates operate on POSTFLOP nodes
        # (spots + full-hand). Preflop-entry reads the flop-entry ranges directly
        # (it never walks the node tree), so none of these apply there.
        if not is_preflop:
            use_ev_gap = st.checkbox(
                "Also require a minimum EV gap to the 2nd-best action",
                value=False,
                key="postflop_use_evgap",
                help="Off by default (real solves mix at ~0 EV gap). On = a quality gate.",
            )
            min_ev_gap = (
                st.number_input(
                    "Min EV gap (bb)", min_value=0.0, value=0.5, step=0.1, key="postflop_evgap"
                )
                if use_ev_gap
                else None
            )
            max_nodes = st.number_input(
                "Max nodes per street (turn/river down-sampling)",
                min_value=50,
                max_value=20000,
                value=600,
                step=50,
                key="postflop_maxnodes",
                help=(
                    "Turn (~2k) and river (~130k) have far too many nodes to build "
                    "all of. Each street is down-sampled to this many representative "
                    "nodes — plenty for a batch. Flop (~25) is always built in full."
                ),
            )
            quality_gate = st.checkbox(
                "Skip low-quality / barely-reached nodes (recommended)",
                value=True,
                key="postflop_quality_gate",
                help="The convergence guard: skip whole nodes that are barely "
                "reached (only a handful of combos get there) or look untrained "
                "(nearly every hand plays one identical mixed strategy). Matters most "
                "for third-party solves and down-sampled turn/river nodes. Count of "
                "skipped nodes is in the batch meta.",
            )
            premise_pct = st.number_input(
                "Premise-realism gate — min line-action frequency (%)",
                min_value=0.0,
                max_value=20.0,
                value=0.5,
                step=0.25,
                key="postflop_premise_pct",
                help="Realism check on the action HISTORY: drop a spot whose line "
                "includes a PRIOR action (by either player) the solver takes below "
                "this often — a question built on a line someone almost never takes "
                "(e.g. a 0.1% turn overbet-jam, a never-used bet size). 0.5% is "
                "permissive (drops only clear ghost lines); raise it for stricter "
                "realism. Set 0 to disable. Skipped-node count is in the batch meta "
                "(premise_filtered_nodes).",
            )
            min_premise_freq = (premise_pct / 100.0) if premise_pct > 0 else None
        else:
            st.caption(
                "EV-gap, node-cap, quality and premise gates apply to postflop "
                "node questions — preflop-entry reads the flop-entry ranges "
                "directly, so they don't apply here."
            )

    with st.expander("Presentation — display amounts / stakes / venue"):
        st.caption(
            "The solve fixes the strategy; these only change how amounts render "
            "in the question text. The solve doesn't carry stakes."
        )
        display_choice = st.radio(
            "Display amounts as",
            options=["Big blinds", "Dollars"],
            index=0,
            horizontal=True,
            key="postflop_display",
            help="How the pot / stack / bets render in the question prose.",
        )
        display_in_bb = display_choice == "Big blinds"
        default_live_idx = 0 if (solve.table_size or 9) >= 9 else 1  # noqa: PLR2004
        live = st.radio(
            "Venue", ["Live", "Online"], index=default_live_idx, horizontal=True,
            key="postflop_live",
        )
        stakes = st.text_input("Stakes (display)", value="$1/$2", key="postflop_stakes")
        bb_dollars = st.number_input(
            "Big blind in $ (used when displaying dollars)",
            min_value=0.01, value=2.0, step=0.5, key="postflop_bbdollars",
        )

    # Layer-7 LLM audit (claim checker + optional auto-fix), mirroring preflop.
    # Runs on independent spots AND the full-hand POSTFLOP legs (the preflop-entry
    # leg is skipped -- the checker prompt is postflop-specific). Preflop-only
    # mode has no postflop legs, so the controls stay hidden there.
    if is_spots or is_full:
      with st.expander("Layer 7 — claim checker & auto-fix (advanced)"):
        if is_full:
            st.caption(
                "In full-hand mode this audits the POSTFLOP legs of each hand "
                "(flop/turn/river); the preflop-entry leg is not audited."
            )
        st.caption(
            "A second LLM pass that AUDITS each finished explanation against the "
            "solver data and flags confusing or wrong poker claims (range "
            "advantage to the wrong player, a mislabeled draw, a backwards "
            "equity-vs-price line). Real runs only; adds API calls."
        )
        # ONE mutually-exclusive choice. The three modes are exclusive in the
        # batch code: the auto-fix pass runs the claim check ITSELF as its gate,
        # so a separate "flag only" on top of it does nothing. A radio makes the
        # do-nothing combination impossible to set.
        pf_layer7_mode = st.radio(
            "Layer 7 mode",
            options=["Off", "Flag only", "Audit & auto-fix"],
            index=2,  # default to Audit & auto-fix (per the user's request)
            horizontal=True,
            key="postflop_layer7_mode",
            help="Off = no AI audit. Flag only = one extra LLM call per question "
            "that FLAGS suspect claims (never rewrites); flags show on the Postflop "
            "Review page. Audit & auto-fix = when a question is flagged, a further "
            "LLM pass rewrites the prose to fix it, re-checked by the deterministic "
            "hard validators (a rewrite that breaks a rule is discarded, the "
            "original kept). Only the prose changes; the action, numbers, and "
            "options stay solver-locked. (Auto-fix already includes the claim "
            "check, so there's no separate 'flag only' to add on top.)",
        )
        pf_run_claim_checker = pf_layer7_mode == "Flag only"
        pf_revise_pass = pf_layer7_mode == "Audit & auto-fix"
        if pf_revise_pass:
            pf_final_audit = st.checkbox(
                "Final audit after the fix",
                value=True,
                key="postflop_final_audit",
                help="Re-runs the claim checker on the rewritten explanation as a "
                "last check (flag only; never triggers another rewrite).",
            )
        if pf_run_claim_checker or pf_revise_pass:
            ck_key = "postflop_claim_checker_prompt"
            if ck_key not in st.session_state:
                st.session_state[ck_key] = _load_postflop_claim_checker_prompt()
            with st.expander("Claim-checker prompt (editable)"):
                edited = st.text_area(
                    "System prompt the postflop claim checker runs with",
                    height=320,
                    key=ck_key,
                )
                if edited.strip() and edited != _load_postflop_claim_checker_prompt():
                    _save_postflop_claim_checker_prompt(edited)
                    st.caption("Saved.")
            pf_claim_checker_prompt = st.session_state[ck_key]

    st.divider()

    # 5. Model + output + run.
    st.subheader("3. Generate")
    col1, col2 = st.columns(2)
    with col1:
        model_label = st.radio(
            "Model",
            options=list(_MODEL_LABEL_TO_API),
            index=0,
            key="postflop_model",
            help="Opus for production; Sonnet for cheap iteration.",
        )
    with col2:
        dry_run = st.toggle(
            "Dry run (no API spend)",
            value=False,
            key="postflop_dryrun",
            help=(
                "Generates real options, facts, EVs, and scenario with "
                "placeholder explanation text. Free — good for a structural check."
            ),
        )
    out_name = st.text_input(
        "Output filename",
        value="",
        placeholder=f"postflop_{solve.flop}_<timestamp>.csv",
        key="postflop_filename",
        help="Leave blank for an auto-timestamped name. Saved under "
        "test_output/postflop_batches/.",
    )
    if is_preflop:
        st.caption(
            "Preflop-entry explanations use the **preflop-entry** prompt (separate "
            "from the postflop system prompt) — edit it on the **Prompt** page "
            "(Postflop mode, bottom). Dry-run uses placeholder prose."
        )
    else:
        st.caption(
            "Explanations use the **postflop** system prompt — edit it on the "
            "**Prompt** page (Postflop mode). Dry-run uses placeholder prose."
            + ("  (Full-hand mode applies it to each postflop leg; the preflop leg "
               "uses the separate preflop-entry prompt — also on the Prompt page.)"
               if is_full else "")
        )

    # A one-line summary of exactly what GENERATE will do in this mode.
    if is_full:
        st.caption(
            f"▶ Will build **{int(total)} full hand(s)** "
            f"(≈ {int(total) * 4}–{int(total) * 6} linked questions)"
            + (", both perspectives" if pf_include_villain else "")
            + "."
        )
    elif is_preflop:
        st.caption(f"▶ Will build up to **{int(total)} preflop-entry question(s)**.")
    else:
        st.caption(f"▶ Will build up to **{int(total)} independent question(s)**.")

    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "No `ANTHROPIC_API_KEY` set — the run falls back to dry-run "
            "(placeholder prose)."
        )

    busy = jobs.has_active_job()
    # Streets are irrelevant to preflop-entry questions; only require them for the
    # postflop spot / full-hand modes.
    needs_streets = pf_mode != "preflop"
    if not heroes or (needs_streets and not streets):
        st.button("GENERATE", disabled=True, type="primary", use_container_width=True)
        missing = "player" if not heroes else "street"
        st.caption(f"Pick at least one {missing} above.")
    elif st.button("GENERATE", disabled=busy, type="primary", use_container_width=True):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        POSTFLOP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = out_name.strip() or f"postflop_{solve.flop}_{stamp}.csv"
        if not fname.endswith(".csv"):
            fname += ".csv"
        out_path = POSTFLOP_OUTPUT_DIR / fname
        try:
            if pf_mode == "full_hand":
                jobs.start_subprocess_job(
                    generate_full_hand_batch_from_db,
                    label=f"Postflop full hands: {Path(picked).name} ({int(total)} hands)",
                    db_path=picked,
                    output_path=str(out_path),
                    total_hands=int(total),
                    heroes=tuple(heroes),
                    streets=tuple(streets),
                    max_nodes_per_street=int(max_nodes),
                    include_villain=pf_include_villain,
                    quality_gate=quality_gate,
                    min_premise_freq=min_premise_freq,
                    answer_style=answer_style,
                    display_in_bb=display_in_bb,
                    stakes=stakes,
                    live_or_online=live,
                    bb_in_dollars=float(bb_dollars),
                    model=_MODEL_LABEL_TO_API[model_label],
                    dry_run=dry_run,
                    min_frequency=freq_lo / 100.0,
                    max_frequency=freq_hi / 100.0,
                    min_ev_gap_bb=min_ev_gap,
                    trap_difficulty=trap_difficulty,
                    razor_difficulty=fh_razor_difficulty,
                    min_hand_difficulty=(fh_band[0] if fh_band else None),
                    max_hand_difficulty=(fh_band[1] if fh_band else None),
                    run_claim_checker=pf_run_claim_checker,
                    claim_checker_prompt=pf_claim_checker_prompt,
                    revise_pass=pf_revise_pass,
                    final_audit=pf_final_audit,
                )
            elif pf_mode == "preflop":
                jobs.start_subprocess_job(
                    generate_preflop_entry_batch_from_db,
                    label=f"Preflop entry: {Path(picked).name} ({int(total)} q)",
                    db_path=picked,
                    output_path=str(out_path),
                    total_questions=int(total),
                    heroes=tuple(heroes),
                    answer_style=answer_style,
                    display_in_bb=display_in_bb,
                    stakes=stakes,
                    live_or_online=live,
                    bb_in_dollars=float(bb_dollars),
                    model=_MODEL_LABEL_TO_API[model_label],
                    dry_run=dry_run,
                    min_frequency=freq_lo / 100.0,
                    max_frequency=freq_hi / 100.0,
                )
            else:
                jobs.start_subprocess_job(
                    generate_postflop_batch_from_db,
                    label=f"Postflop: {Path(picked).name} ({int(total)} q)",
                    db_path=picked,
                    output_path=str(out_path),
                    total_questions=int(total),
                    heroes=tuple(heroes),
                    streets=tuple(streets),
                    max_nodes_per_street=int(max_nodes),
                    diversify=diversify,
                    strength_buckets=tuple(strength_filter),
                    decision_types=tuple(decision_filter),
                    quality_gate=quality_gate,
                    min_premise_freq=min_premise_freq,
                    answer_style=answer_style,
                    display_in_bb=display_in_bb,
                    stakes=stakes,
                    live_or_online=live,
                    bb_in_dollars=float(bb_dollars),
                    model=_MODEL_LABEL_TO_API[model_label],
                    dry_run=dry_run,
                    min_frequency=freq_lo / 100.0,
                    max_frequency=freq_hi / 100.0,
                    min_ev_gap_bb=min_ev_gap,
                    trap_difficulty=trap_difficulty,
                    run_claim_checker=pf_run_claim_checker,
                    claim_checker_prompt=pf_claim_checker_prompt,
                    revise_pass=pf_revise_pass,
                    final_audit=pf_final_audit,
                )
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


def _render_postflop_skills_explainer() -> None:
    """A plain-English dropdown of EXACTLY how each postflop skill is tagged.

    Reads the rules' own explainers (pipeline/postflop/skills.py) so this can
    never drift from the code that actually tags. Grouped by the app's catalog
    sections; also lists the skills deliberately left untagged and why."""
    from pipeline.postflop.skills import (  # noqa: PLC0415
        POSTFLOP_SKILL_EXPLAINERS,
        POSTFLOP_SKILL_RULES,
        POSTFLOP_SKILLS_NOT_TAGGED,
    )

    with st.expander("📋 How each postflop skill is tagged (plain English)"):
        st.caption(
            "Every skill in the `skills` CSV column is set by a deterministic "
            "Python rule (never the LLM). Here is exactly what triggers each one."
        )
        st.markdown(f"**Tagged today ({len(POSTFLOP_SKILL_RULES)} skills):**")
        for name in POSTFLOP_SKILL_RULES:
            why = POSTFLOP_SKILL_EXPLAINERS.get(name, "")
            st.markdown(f"- **{name}** — {why}")
        st.markdown("**Not tagged on postflop (no clean signal yet):**")
        for name, why in POSTFLOP_SKILLS_NOT_TAGGED.items():
            st.caption(f"- {name} — {why}")
        st.caption(
            "Preflop-only skills (3-Betting, Squeezing, Blind Defense, …) never "
            "apply to a postflop decision and are not listed."
        )


def _render_postflop_job_panel() -> None:
    """Top-of-page panel for the current/last POSTFLOP batch (mirrors the
    preflop one, but renders a ``PostflopBatchResult``). Only shows postflop
    jobs (filtered by the label prefix), so a preflop job doesn't render here."""
    from pipeline.postflop.batch import PostflopBatchResult  # noqa: PLC0415

    job = jobs.get_current_job()
    # Match all three Generate-page job kinds (independent spots, full hands,
    # preflop entry) but NOT "PostflopCompare:" (the Compare page has its own panel).
    if job is None or not str(job.label).startswith(
        ("Postflop:", "Postflop full hands:", "Preflop entry:")
    ):
        return
    is_full_hand_job = str(job.label).startswith("Postflop full hands:")
    with st.container(border=True):
        if job.is_active:
            _render_active_job_progress()
        elif job.status is jobs.JobStatus.COMPLETED:
            st.markdown(f"**✅ Last batch:** {job.label}")
            if isinstance(job.result, PostflopBatchResult):
                # Log token spend to the lifetime usage log exactly once. This
                # was previously NEVER done for ANY postflop batch (only the
                # preflop BatchResult path logged), so the sidebar "Lifetime API
                # spend" never moved for postflop runs.
                _maybe_log_completed_postflop_job(job, job.result)
                if is_full_hand_job:
                    st.caption(
                        f"**{job.result.requested_questions} hands → "
                        f"{job.result.questions_written} linked questions** "
                        "(grouped by hand_id in the CSV / Postflop Review)."
                    )
                _render_postflop_result_ui(job.result)
            else:
                st.warning("Job finished but returned no PostflopBatchResult.")
            st.caption(f"Finished in {job.elapsed_seconds:.0f}s.")
            if st.button("Hide last result", key=f"clear_pf_{job.id}"):
                jobs.clear_current_job()
                st.rerun()
        elif job.status is jobs.JobStatus.CANCELLED:
            st.warning(f"**⛔ Batch cancelled:** {job.label}")
            if st.button("Dismiss", key=f"dismiss_pf_cancel_{job.id}"):
                jobs.clear_current_job()
                st.rerun()
        else:  # FAILED
            st.error(f"**❌ Job failed:** {job.label}")
            with st.expander("Traceback"):
                st.code(job.error or "(no traceback captured)")
            if st.button("Dismiss failure", key=f"dismiss_pf_{job.id}"):
                jobs.clear_current_job()
                st.rerun()
    st.divider()


def _render_postflop_result_ui(result: Any) -> None:
    """Success summary + CSV download + a preview of the first few questions
    (options marked ✅ correct / 😐 neutral, plus the per-action EVs)."""
    import csv as _csv  # noqa: PLC0415

    cols = st.columns(4)
    cols[0].metric("Written", result.questions_written)
    cols[1].metric("Worthy spots", result.worthy_spots_available)
    cols[2].metric("Flagged", result.soft_flagged_rows)
    cols[3].metric("Failed", len(result.failures))
    if not result.dry_run and (result.total_input_tokens or result.total_output_tokens):
        st.caption(
            f"Tokens: {result.total_input_tokens:,} in / "
            f"{result.total_output_tokens:,} out · model {result.model_used}"
        )
    out = Path(result.output_path)
    st.caption(f"CSV: `{out}`")
    try:
        st.download_button(
            "⬇️ Download CSV",
            data=out.read_bytes(),
            file_name=out.name,
            mime="text/csv",
            key=f"dl_pf_{out.name}",
        )
        with out.open(encoding="utf-8-sig") as fh:
            rows = list(_csv.DictReader(fh))
    except OSError:
        rows = []

    if rows:
        with st.expander(
            f"Preview ({min(3, len(rows))} of {len(rows)} questions)", expanded=True
        ):
            for r in rows[:3]:
                st.markdown(f"**{r.get('User Seat', '')}** · {r.get('Cards on Table', '')}")
                st.caption(r.get("Question", ""))
                correct = r.get("Correct Answer", "")
                neutral = {
                    x.strip()
                    for x in (r.get("neutral_credit", "") or "").split(",")
                    if x.strip()
                }
                for i in (1, 2, 3, 4):
                    opt = r.get(f"option {i}", "")
                    if not opt:
                        continue
                    mark = "✅" if opt == correct else ("😐" if opt in neutral else "▫️")
                    st.markdown(f"{mark} {opt}")
                if r.get("action_ev_bb"):
                    st.caption(f"EVs: {r['action_ev_bb']}")
                st.divider()
    if result.failures:
        with st.expander(f"⚠️ {len(result.failures)} failed spot(s)"):
            for f in result.failures[:10]:
                st.caption(
                    f"{f.get('node_id')} / {f.get('hero_combo')}: "
                    f"{f.get('error_message', '')}"
                )


# NLHE Generate-page widgets whose selections persist across tab-switches /
# panel restarts (the PLO page already does this; the NLHE page didn't, so
# every restart silently reset style + filters to defaults -- which is how a
# batch ends up on Basic style + all-player-counts when GTO was picked).
_PREFLOP_GEN_SAVED_KEYS: tuple[str, ...] = (
    "preflop_gen_pack",
    "preflop_gen_positions",
    "preflop_gen_contexts",
    "preflop_gen_player_counts",
    "preflop_answer_style",
)


def _seed_preflop_generate_settings(
    *,
    positions_available: list[str],
    context_options: list[str],
    count_options: list[int],
) -> None:
    """Re-seed the NLHE Generate filter/style widgets from the last run's
    snapshot. Only fills keys ABSENT from session state (live edits always
    win), and sanitizes every saved value against the CURRENT options so a
    stale file can never crash a widget. Mirror of
    :func:`_seed_plo_generate_settings`.
    """
    saved = gen_settings.load_settings(PREFLOP_GEN_SETTINGS_PATH)

    def _subset(value: object, options: list, default: list) -> list:
        if not isinstance(value, list):
            return default
        return [x for x in value if x in options]  # empty = a real "any" choice

    def _choice(value: object, options: list, default: str) -> str:
        return value if isinstance(value, str) and value in options else default

    style_labels = list(ANSWER_STYLE_FROM_RADIO_LABEL.keys())
    restored: dict[str, object] = {
        "preflop_gen_positions": _subset(
            saved.get("preflop_gen_positions"),
            positions_available,
            positions_available,
        ),
        "preflop_gen_contexts": _subset(
            saved.get("preflop_gen_contexts"),
            context_options,
            ["Opening", "Facing single raise", "Facing 3-bet"],
        ),
        # Default to clean spots: open (1) / heads-up (2) / three-way (3).
        # Deep multiway is opt-in (it floods otherwise). Capped to pack size.
        "preflop_gen_player_counts": _subset(
            saved.get("preflop_gen_player_counts"),
            count_options,
            [c for c in count_options if c <= 3],  # noqa: PLR2004
        ),
        "preflop_answer_style": _choice(
            saved.get("preflop_answer_style"), style_labels, style_labels[0]
        ),
    }
    for key, value in restored.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- page: Generate (Preflop mode) ----------------------------------------
def _render_generate_page_preflop() -> None:
    """Preflop-mode Generate page.

    Uses pipeline/preflop/ backend. No Pio solves needed -- reads Ryan's
    preflop pack from ranges/. Filters are different from postflop:
    hero position (not table seat), action faced (opening / facing-raise
    / 3-bet / 4-bet / after-calls), hand-class strength buckets. Board
    texture filter is hidden (no board preflop).
    """
    st.caption(
        "Preflop generation reads from your uploaded preflop pack. "
        "Filters narrow which decision spots get sampled."
    )

    # Authoritative plain-English reference for the WHOLE generation pipeline.
    # Kept here (top of the page, collapsed) as the single place to look up
    # what every step/gate does -- the per-control tooltips are quick hints,
    # this is the full explanation.
    with st.expander(
        "📖 How question generation works — full plain-English reference"
    ):
        st.markdown(
            "Every batch runs the same pipeline. Each step either **picks** "
            "spots or **skips** them; whatever survives gets an explanation "
            "written. How many spots each step skipped always shows in the "
            "result summary after a run.\n\n"

            "**1. Pick the candidate spots (the filters below).**\n"
            "- *Hero positions* — which seats you want to be in.\n"
            "- *Action faced* — the bucket is the **highest raise you face**. A "
            "3-bet / 4-bet pot stays 'Facing 3-bet' / 'Facing 4-bet+' even if "
            "someone flat-called earlier in the hand; 'After one/multiple "
            "call(s)' means a **single open** that picked up flat-callers (a "
            "squeeze / overcall). Multiway-ness is the next filter's job.\n"
            "- *Players in the pot* — how many are still in at your decision "
            "(heads-up vs multiway).\n\n"

            "**2. Worthiness window — is it a real decision?** Keep a spot only "
            "if the solver takes its best action between the low and high % you "
            "set (default 55–99%). Below ~55% there's no clear best answer; at "
            "100% it's trivial. The 90–95% band can be punched out (the 'mostly "
            "that reads as always' trap).\n\n"

            "**3. Premise-realism gates — is the STORY natural?** Two checks, "
            "both tunable, both run before any LLM spend:\n"
            "- *Min villain line frequency* — skip if the OPPONENT's whole line "
            "is something the solver almost never does (a question built on a "
            "line that essentially never happens).\n"
            "- *Min hero premise frequency* — skip if one of YOUR OWN earlier "
            "actions in the story is a play the solver almost never makes with "
            "this hand. Example: 'you call from UTG+2 with AKs' when the solver "
            "3-bets AKs there 95% of the time and flats only 5% — at a 15% gate "
            "that spot is skipped.\n\n"

            "**4. Difficulty band.** Score each spot 400–3200 (four axes: "
            "frequency, EV gap, concepts, hand class — see the 'How is "
            "Difficulty calculated?' popover under *Difficulty*) and keep only "
            "the ones inside the tier you picked (Easy / Medium / Hard / "
            "Mixed). The **EV gap** here = the solver's OWN EV for the action "
            "you take most often minus its EV for the next-most-played action "
            "(from the per-action EVs the Monker packs ship). A near-coinflip "
            "decision the solver mixes 57/42 has a tiny gap (~0.1bb) → rated "
            "Hard; a clear decision has a big gap → easier. Only EV-less packs "
            "(the Ryan pack) fall back to the old approximate gap formula.\n\n"

            "**5. Unconverged-node guard (always on, not tunable).** Drop solver "
            "nodes the solve never refined: AA folding preflop, premium-pair "
            "inversions, or more than 30% of the reaching hands split equally "
            "across every action (the solver's untouched default). This is the "
            "near-zero-reach multiway 5-bet / jam tail.\n\n"

            "**6. EV-coherence guard (always on, not tunable).** A solver only "
            "**mixes** between actions that are about **equal in EV** — that's "
            "what 'indifference' means: if one play were clearly better, it "
            "would just always take it. So this guard looks at the actions your "
            "hand plays at least **10%** of the time, reads the solver's OWN "
            "per-action EV for each, and if the best and worst of those are more "
            "than about **3bb apart**, the 'mix' isn't a real strategy — it's an "
            "unconverged node, so the spot is dropped. Example: a hand that "
            "'calls' an all-in 26% of the time while calling is ~15bb worse than "
            "folding per the solver's own numbers. (Needs per-action EVs, so it "
            "only runs on the Monker packs; the EV-less Ryan pack is "
            "unaffected.) This catches hand-specific noise the unconverged-node "
            "guard above can't see.\n\n"

            "**7. Write the explanation (the only paid step).** Layer 6 turns "
            "the solved facts into prose. If **Layer 7 mode** is 'Flag only' or "
            "'Audit & auto-fix', a second LLM pass then audits that prose against "
            "the solver data — flagging any poker claim that contradicts the data, "
            "is poker-wrong, is invented, contradicts another sentence, or "
            "disagrees with the action mix (e.g. calling a spot '3-bet or fold' "
            "when the hand also calls a real share). It only flags for review, "
            "never rejects; 'Audit & auto-fix' then rewrites the flagged prose."
        )

    # Active/last background job panel. Persists across tab switches
    # since the job runs on its own thread.
    _render_preflop_job_panel()

    # --- 1. Pack info ---
    st.subheader("1. Preflop pack")
    packs = _cached_preflop_packs()
    if not packs:
        st.error(
            "❌ No preflop pack loaded. Upload a pack on the Files page "
            "first. (Ryan's PioViewer pack belongs under "
            "`ranges/ryan_preflop_tree/`; the 9-max Monker pack under "
            "`nlhe9_ranges/`.)"
        )
        return
    # Restore the last-used pack BEFORE the widget exists (same seeding
    # pattern as the PLO page); a stale/unknown saved id is ignored.
    if "preflop_gen_pack" not in st.session_state:
        saved = gen_settings.load_settings(PREFLOP_GEN_SETTINGS_PATH)
        saved_pack = saved.get("preflop_gen_pack")
        if saved_pack in {p.pack_id for p in packs}:
            st.session_state["preflop_gen_pack"] = saved_pack
    pack = _select_preflop_pack("preflop_gen_pack")
    assert pack is not None  # guarded by the empty-packs return above
    st.success(
        f"✅ **{pack.pack_id}** · {pack.table_size}-max · "
        f"{pack.stack_depth_bb}bb · {pack.open_size_bb}x open · "
        f"{pack.description}"
    )

    # Metadata only (context + player count per node, grouped by actor) --
    # the Generate page never needs full node objects, so it stays off the
    # reconstruction path and reads the precomputed on-disk metadata cache.
    node_meta = _cached_node_filter_meta(pack.pack_id)
    total_nodes = sum(len(rows) for rows in node_meta.values())
    st.caption(
        f"Walked the pack: **{total_nodes:,} preflop decision nodes** enumerated."
    )

    st.divider()

    # --- 2. Hero context ---
    st.subheader("2. Hero context")
    # Compute the option lists FIRST, then seed last-used selections into
    # session state BEFORE the widgets render, so the user's choices stick
    # across tab-switches / panel restarts (the widgets below carry `key=` and
    # NO `default=` -- they read the seeded session state).
    seat_order = preflop_order(pack.table_size)  # UTG first; reads like a table
    positions_available = [p for p in seat_order if p in node_meta]
    positions_available += sorted(
        p for p in node_meta if p not in seat_order
    )  # defensive: never hide a position the pack actually has
    context_options = [
        "Opening",
        "Facing single raise",
        "Facing 3-bet",
        "Facing 4-bet+",
        "After one call",
        "After multiple calls",
    ]
    count_options = list(range(1, pack.table_size + 1))
    _seed_preflop_generate_settings(
        positions_available=positions_available,
        context_options=context_options,
        count_options=count_options,
    )
    col1, col2 = st.columns(2)
    with col1:
        hero_positions = st.multiselect(
            "Hero positions",
            options=positions_available,
            key="preflop_gen_positions",
            help="Which seats hero is in. Empty = include all positions.",
        )
    with col2:
        action_contexts = st.multiselect(
            "Action faced",
            options=context_options,
            key="preflop_gen_contexts",
            help=ACTION_FACED_HELP,
        )
        player_counts = st.multiselect(
            "Players in the pot",
            options=count_options,
            key="preflop_gen_player_counts",
            format_func=lambda n: (
                "1 (open)" if n == 1 else "2 (heads-up)" if n == 2 else f"{n}-way"
            ),
            help="How many players are still in at hero's decision. Defaults to "
                 "1-3 (open / heads-up / three-way) for clean spots; add 4+ for "
                 "deep multiway. Empty = include all.",
        )

    # Filter the node catalog by all three filters; show a live count.
    # Uses the precomputed per-node (context, players) tuples -- walking
    # the real node objects here ran on every widget click and took
    # seconds at the 9-max pack's 44k-node scale.
    _count_set = set(player_counts) if player_counts else None
    _ctx_set = set(action_contexts) if action_contexts else None
    filter_meta = _cached_node_filter_meta(pack.pack_id)
    n_filtered = sum(
        1
        for actor in hero_positions
        for ctx, players in filter_meta.get(actor, ())
        if (_ctx_set is None or ctx in _ctx_set)
        and (_count_set is None or players in _count_set)
    )
    st.caption(
        f"**{n_filtered:,}** decision nodes match these filters "
        f"(of {total_nodes:,} total)."
    )

    st.divider()

    # --- 3. Content filters (hand class strength only; board texture N/A preflop) ---
    st.subheader("3. Content filters")
    st.caption("Optional. Empty = include every hand class that reaches the node.")
    _hand_classes = st.multiselect(
        "Hero hand strength buckets",
        options=list(STRENGTH_BUCKETS),
        default=[],
        help=(
            "premium / strong / medium / vulnerable / marginal / air. "
            "Sourced from pipeline/fact_extractor/hand_class.py."
        ),
    )
    st.caption("_(Board-texture filter is postflop-only; hidden in preflop mode.)_")

    st.divider()

    # --- 4. Difficulty ---
    st.subheader("4. Difficulty")
    st.caption(
        "Presets target a band of the COMPUTED difficulty rating -- the "
        "4-axis score (frequency, EV gap, archetype/concept, hand class), "
        "not frequency alone. The generator keeps only spots whose rating "
        "lands in the band, so a preset reliably yields that tier."
    )
    with st.popover("ℹ️  How is the Difficulty Rating calculated?"):
        _render_difficulty_explainer()
    preset = st.radio(
        "Preset",
        options=["Easy", "Medium", "Hard", "Mixed", "Custom"],
        index=3,  # default Mixed for preflop
        horizontal=True,
        key="preflop_difficulty_preset",
    )
    # Difficulty-RATING bands (the score runs ~400-3200). Mixed = full
    # range. These thirds split the brief's 500-3000 MVP band; tune here.
    difficulty_bands = {
        "Easy":   (400, 1300),
        "Medium": (1300, 2100),
        "Hard":   (2100, 3200),
        "Mixed":  (DIFFICULTY_MIN, DIFFICULTY_MAX),
        "Custom": (400, 3200),
    }
    # Always show the slider so the band is visible; for the fixed presets
    # it's read-only (disabled) and snaps to that preset's band, for Custom
    # it's editable. The key varies by preset so switching presets re-seeds
    # the slider to the new band (a keyed Streamlit widget otherwise ignores
    # a changed `value=` once its session-state entry exists).
    preset_low, preset_high = difficulty_bands[preset]
    band_low, band_high = st.slider(
        "Difficulty rating band",
        min_value=DIFFICULTY_MIN,
        max_value=DIFFICULTY_MAX,
        value=(preset_low, preset_high),
        step=50,
        disabled=(preset != "Custom"),
        key=f"preflop_difficulty_slider_{preset}",
        help=(
            "The computed-rating band this batch keeps. Presets snap it to "
            "a tier; pick Custom to set your own range."
        ),
    )

    # Trap-aware difficulty (opt-in NEW method). Lets pure/clear-cut but
    # counterintuitive spots ("Always fold" a hand that looks like a call)
    # rate Hard, so the Hard band isn't only close mixes. Full explanation
    # in the popover; off by default = original behaviour.
    trap_difficulty = st.checkbox(
        "🪤 Trap-aware difficulty — rate counterintuitive spots as Hard (NEW)",
        value=False,
        key="preflop_trap_difficulty",
        help=(
            "OFF = the original 4-axis rating (Hard = close-mix spots only). "
            "ON = ALSO rate \"trap\" spots Medium-to-Hard even when the answer "
            "is pure/clear-cut (a hand the solver folds 100% despite enough "
            "equity to call), GRADED 1800-2900 by how contradictory the pot "
            "odds look. This is what makes a \"Medium/Hard + 100% frequency\" "
            "batch return questions. Read the ℹ️ below before using."
        ),
    )
    with st.popover("ℹ️  What is Trap-aware difficulty? (read before using)"):
        _render_trap_difficulty_explainer()

    # Razor's-edge difficulty (opt-in, July 2026). A second kind of
    # pure-but-hard spot: the hand sits ON a range boundary (its grid
    # neighbor does the opposite at the same node), so the skill tested is
    # knowing exactly where the cutoff is. Deterministic; score-only.
    razor_difficulty = st.checkbox(
        "🔪 Razor's-edge difficulty — rate range-BOUNDARY hands Medium/Hard (NEW)",
        value=False,
        key="preflop_razor_difficulty",
        help=(
            "OFF = boundary position has no effect on the rating. ON = a "
            "hand whose grid NEIGHBOR does the opposite at the same spot "
            "(ATo always folds where AJo always calls; A5s 3-bets where "
            "A4s folds) is rated 2000-2600, graded by how many neighbors "
            "oppose it. Works at any frequency and is the other way to get "
            "Medium/Hard questions from 100% spots. Read the ℹ️ below."
        ),
    )
    with st.popover("ℹ️  What is Razor's-edge difficulty?"):
        _render_razor_difficulty_explainer()

    # Situation diversification. The worthy pool is walked in order until N
    # rows are collected, so a fill-to-N batch (especially Hard trap-aware)
    # collapses onto the most common situation -- every question an ace-high
    # fold facing a blind 3-bet. ON round-robins the pool across
    # seat / action-context / dominant-action / hand-category buckets so the
    # batch spreads across positions, hands, and actions. Default ON.
    diversify = st.checkbox(
        "🎲 Diversify situations — spread the batch across seats, hands & actions",
        value=True,
        key="preflop_diversify",
        help=(
            "ON (default) = round-robin the eligible spots across situation "
            "buckets, so a batch isn't a run of near-identical spots (e.g. all "
            "ace-high folds vs a blind 3-bet). Especially important with "
            "Trap-aware on. OFF = the old flat-random fill. Never changes any "
            "answer or difficulty — only WHICH worthy spots are picked."
        ),
    )

    # Advanced: question-worthiness window + EV-gap quality gate. These
    # are SEPARATE from difficulty -- worthiness decides whether a spot is
    # teachable at all; the EV gate drops near-coinflip call/fold spots.
    with st.expander(
        "Advanced filters (worthiness window · EV-gap gate)", expanded=True
    ):
        st.caption(
            "The frequency worthiness window gates whether a decision is "
            "teachable at all (the brief's 55-95% sweet spot). The EV-gap "
            "gate is a quality filter on call/fold spots."
        )
        freq_low, freq_high = st.slider(
            "Solver frequency worthiness window (%)",
            min_value=50,
            max_value=100,
            value=(65, 99),
            key="preflop_worthiness_slider",
            help="Below 65% = no clear best answer to teach; 100% = trivial.",
        )
        exclude_ambiguous_band = st.checkbox(
            "Exclude ambiguous 90–95% band",
            value=False,
            key="preflop_exclude_ambiguous_band",
            help=(
                "An OPTIONAL extra hole below the near-pure band: also drop the "
                "90–95% spots. These are clearly \"Mostly X\" but getting close to "
                "pure. Off by default -- neutral credit covers an \"Always X\" pick "
                "and 90–95% spots still teach a real \"mostly\" lesson. Check to "
                "tighten the window to 65–90%."
            ),
        )
        exclude_near_pure_band = st.checkbox(
            "Exclude ambiguous 95–99% band (recommended)",
            value=True,
            key="preflop_exclude_near_pure_band",
            help=(
                "Spots where the solver takes one action 95–99% of the time are "
                "nearly pure. The correct answer is \"Mostly X\" (\"Always X\" is "
                "reserved for a literal 100%, since we no longer round), but at "
                "this frequency it reads like \"Always X\", so the Mostly-vs-Always "
                "distinction is hair-splitting. On by default: punches a hole at "
                "95–99% so \"Mostly X\" questions land on genuinely mixed spots. A "
                "literal 100% spot is NOT in the band, so a genuine \"Always X\" "
                "spot still qualifies. (Neutral credit already gives an \"Always "
                "X\" pick partial credit, so these aren't unfair -- just fuzzy.) "
                "Uncheck to let near-pure spots in."
            ),
        )
        min_ev_gap = st.slider(
            "Minimum EV gap (bb) — 0 = off",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            key="preflop_min_ev_gap",
            help=(
                "Drops call/fold spots whose EV gap to the 2nd-best action "
                "is below this. Raise spots (no computed EV) always pass."
            ),
        )
        # Premise-realism gates (June 2026 audit). Both default ON.
        st.markdown("**Realism gates** — drop questions whose SETUP almost never happens")
        st.caption(
            "Each value is a **minimum frequency, in % of hands** — a spot is "
            "skipped when its setup falls below it. **Higher = stricter** (only "
            "common, natural spots get through); **lower = looser**; **0 = off** "
            "(allow even near-impossible setups). They run before any LLM spend; "
            "skipped spots are counted as “premise-realism gate” in the result "
            "summary. Defaults (0.25 / 30) are tuned — leave them unless you "
            "specifically want more (lower) or fewer (higher) edge-case spots."
        )
        min_villain_pct = st.number_input(
            "Min villain line frequency  (the opponent's action you're facing)",
            min_value=0.0, max_value=10.0, value=0.25, step=0.05,
            key="preflop_min_villain_pct",
            help=(
                "How often the OPPONENT actually takes the action you're facing, "
                "as a % of all dealt hands. At 0.25 it drops, say, a 'you face a "
                "3-bet' question when the solver 3-bets under 0.25% of hands -- "
                "you'd be quizzed on a line that basically never occurs (a "
                "'ghost'). Raise it to demand commoner opponent actions; 0 lets "
                "any line through. Open spots (no villain action yet) always pass."
            ),
        )
        st.caption(
            "↳ Is the **opponent's move you're facing** (a 3-bet, a jam, a limp) "
            "common enough to be worth a question? Default **0.25%** of hands."
        )
        min_premise_pct = st.number_input(
            "Min hero premise frequency  (a play YOU supposedly made earlier)",
            min_value=0.0, max_value=50.0, value=30.0, step=1.0,
            key="preflop_min_premise_pct",
            help=(
                "Looks at YOUR OWN earlier actions in the hand and skips the spot "
                "if the rarest one is below this %. Example: 'you flat-call AKs "
                "from UTG+2' is a weak premise if the solver 3-bets AKs there 95% "
                "and flats only 5% -- at 30% that spot is skipped, because it "
                "asks you to own a play a strong player almost never makes. "
                "Higher = only natural lines (note: a hand the solver takes that "
                "prior action <30% of the time gets filtered); 0 = off. Only "
                "applies when you ACTED before this decision -- a clean first "
                "decision (just deciding to open or call) always passes."
            ),
        )
        st.caption(
            "↳ Is the **earlier action that put you here** one the solver actually "
            "makes with this hand, so the backstory is believable? Default **30%**. "
            "Only affects spots where you acted before the decision."
        )
    # The two always-on guards (unconverged-node + EV-coherence) are documented
    # in the "📖 How question generation works" reference at the top of the
    # page (steps 5 & 6) -- kept there as the single source rather than
    # duplicated here, so there's one place to maintain.

    # Visible settings summary -- exactly what THIS batch will use. Promoted
    # to a prominent info box so the numbers are obvious the moment you pick
    # a preset (the preset moves the difficulty band; worthiness + EV-gap are
    # separate gates, shown here so all three are always visible).
    _ev_txt = "off" if min_ev_gap == 0.0 else f"≥ {min_ev_gap:.2f} bb"
    _excluded_bands = []
    if exclude_near_pure_band and freq_high > 95:  # noqa: PLR2004
        _excluded_bands.append("95–99%")
    if exclude_ambiguous_band and freq_high > 90:  # noqa: PLR2004
        _excluded_bands.append("90–95%")
    _band_note = (
        "  ·  " + " + ".join(_excluded_bands) + " excluded" if _excluded_bands else ""
    )
    st.info(
        f"**Numbers in effect for this batch** — difficulty rating "
        f"**{band_low}–{band_high}**  ·  worthiness frequency "
        f"**{freq_low}–{freq_high}%**{_band_note}  ·  EV-gap gate "
        f"**{_ev_txt}**.  "
        "Presets move the difficulty band; the worthiness window + EV-gap "
        "are separate gates you set in Advanced filters above."
    )

    # Structural-emptiness warning (July 2026). The difficulty rating is
    # bounded above by a pure function of the dominant frequency, so a band
    # whose floor exceeds that ceiling matches ZERO spots no matter the pack.
    # The classic foot-gun this catches: "1500+ difficulty at 100% frequency"
    # (pure spots cap near 1125, measured ~875) used to grind through HOURS
    # of equity sims before reporting an empty batch. The decision logic is
    # classify_band_reachability (pure, browserlessly tested in
    # tests/test_preflop_difficulty.py); this block only renders its verdict.
    from pipeline.preflop.difficulty import (  # noqa: PLC0415
        classify_band_reachability,
        max_achievable_difficulty,
    )
    from pipeline.trap_grading import (  # noqa: PLC0415
        TRAP_FLOOR_MAX,
        TRAP_FLOOR_MIN,
    )
    from pipeline.preflop.razor_edge import (  # noqa: PLC0415
        RAZOR_FLOOR_BY_COUNT,
        RAZOR_FLOOR_MAX,
    )
    _razor_min = min(RAZOR_FLOOR_BY_COUNT.values())
    _reach = classify_band_reachability(
        band_low, band_high, freq_low / 100.0,
        trap_difficulty=bool(trap_difficulty),
        razor_difficulty=bool(razor_difficulty),
    )
    _natural_ceiling = max_achievable_difficulty(freq_low / 100.0)
    if _reach == "empty":
        _fix_bits = []
        if not trap_difficulty and (
            band_low <= TRAP_FLOOR_MAX and band_high >= TRAP_FLOOR_MIN
        ):
            _fix_bits.append(
                f"**🪤 Trap-aware** (traps rate {TRAP_FLOOR_MIN}-{TRAP_FLOOR_MAX})"
            )
        if not razor_difficulty and (
            band_low <= RAZOR_FLOOR_MAX and band_high >= _razor_min
        ):
            _fix_bits.append(
                f"**🔪 Razor's-edge** (boundary hands rate "
                f"{_razor_min}-{RAZOR_FLOOR_MAX})"
            )
        if _fix_bits:
            _fix_hint = (
                " Turning on " + " or ".join(_fix_bits) + " would fix this."
            )
        elif trap_difficulty or razor_difficulty:
            _fix_hint = (
                " The special difficulty modes you enabled rate spots "
                f"between {min(TRAP_FLOOR_MIN, _razor_min)} and "
                f"{max(TRAP_FLOOR_MAX, RAZOR_FLOOR_MAX)}, entirely outside "
                "your band. Widen the band to overlap that range."
            )
        else:
            _fix_hint = (
                " Lower the difficulty band floor or lower the minimum "
                "worthiness frequency."
            )
        st.error(
            f"🚫 **This combination can produce 0 questions, guaranteed.** "
            f"With a minimum worthiness frequency of **{freq_low}%**, the "
            f"highest difficulty rating any spot can score is about "
            f"**{_natural_ceiling}**, but your band starts at "
            f"**{band_low}**. High-frequency (clear-cut) spots always rate "
            f"easy on the frequency and EV axes, so they can never reach "
            f"the Medium or Hard bands on their own.{_fix_hint}"
        )
    elif _reach == "special_only":
        _modes = []
        if trap_difficulty:
            _modes.append(
                f"counterintuitive traps (rated {TRAP_FLOOR_MIN}-"
                f"{TRAP_FLOOR_MAX} by pot-odds contradiction)"
            )
        if razor_difficulty:
            _modes.append(
                f"range-boundary hands (rated {_razor_min}-{RAZOR_FLOOR_MAX} "
                f"by opposing neighbors)"
            )
        st.warning(
            f"⚠️ **Special-rated spots only.** At a minimum worthiness "
            f"frequency of **{freq_low}%**, regular spots top out around "
            f"**{_natural_ceiling}** difficulty, below your band. Only "
            + " and ".join(_modes)
            + " qualify, a smaller pool than usual, so expect a slower "
            "fill and turn on 🎲 Diversify."
        )
    elif freq_low >= 100 and band_low > 875 and not (  # noqa: PLR2004
        trap_difficulty or razor_difficulty
    ):
        # The theoretical ceiling (1125) needs a worst-case archetype AND
        # hand class at once; measured on the 8-max packs, real 100%-pure
        # spots top out near 875. Warn on the gap the math can't rule out.
        st.warning(
            f"⚠️ **Very few candidates likely.** At a 100%-only frequency "
            f"window, measured difficulty on the 8-max packs tops out near "
            f"**875** (theoretical ceiling {_natural_ceiling}). A band "
            f"starting at {band_low} will match few or no spots. For "
            f"Medium/Hard questions at 100% frequency, turn on 🪤 "
            f"Trap-aware or 🔪 Razor's-edge difficulty."
        )

    st.divider()

    # --- 5. Answer option style (reuse) ---
    st.subheader("5. Answer option style")
    answer_style = st.radio(
        "Style",
        options=list(ANSWER_STYLE_FROM_RADIO_LABEL.keys()),
        # No index= -- the value is seeded into session state by
        # _seed_preflop_generate_settings (persists across restarts); falls
        # back to the first option (Basic) only if seeding ever didn't run.
        key="preflop_answer_style",
        help=(
            "**Basic** -- bare action labels (Fold / Call / Raise 60%). "
            "**GTO** -- Always / Mostly template that surfaces mixed "
            "strategies. **Sizing** -- not relevant for preflop (raise "
            "sizes are already in the action labels); falls back to Basic. "
            "**Auto-pick** -- chooses Basic for dominant-action spots, "
            "GTO for mixed spots."
        ),
    )
    # Options + correct_answer are now computed deterministically in
    # pipeline.preflop.options (no LLM involved for the picking).
    answer_style_canonical = ANSWER_STYLE_FROM_RADIO_LABEL[answer_style]

    st.divider()

    # --- 6. Batch size ---
    st.subheader("6. How many questions")
    total = st.number_input(
        "Total questions in this batch",
        min_value=1,
        max_value=10_000,
        value=10,
        step=5,
        key="preflop_total",
        help=(
            "Will be spread evenly across the matching scenarios + hand "
            "classes, with diversity-stratified sampling."
        ),
    )

    st.divider()

    # --- 7. Output options ---
    st.subheader("7. Output")
    col1, col2 = st.columns(2)
    with col1:
        _currency = st.radio(
            "Display amounts as",
            options=["Dollars ($1.25)", "Big blinds (2.5 bb)"],
            index=1,  # bb is more common for preflop discussion
            horizontal=True,
            key="preflop_currency",
        )
    with col2:
        _out_filename = st.text_input(
            "Output filename (prefix)",
            value="preflop_batch",
            key="preflop_out_filename",
            help=(
                "A timestamp (`_YYYYMMDD_HHMMSS.csv`) is appended at "
                "write time so every batch lands in its own file -- "
                "the History tab keeps them all."
            ),
        )

    # Stakes + venue are DISPLAY framing only -- every solver number is in
    # bb, so the strategy is identical at any stake. Defaults are
    # pack-aware: the 9-max Monker pack (9-handed, 4x opens, 10%/3bb rake
    # = a $1/$2-style live structure) frames as a Live $1/$2 game; the
    # 6-max pack keeps the original Online $0.25/$0.50 framing. Widgets
    # are keyed per pack so switching packs re-applies the matching
    # default without fighting your last manual choice.
    def _stake_label_preflop(bb_dollars: float) -> str:
        sb = bb_dollars / 2
        sb_str = f"${sb:.2f}".rstrip("0").rstrip(".") if sb < 1 else f"${sb:g}"
        bb_str = (
            f"${bb_dollars:.2f}".rstrip("0").rstrip(".")
            if bb_dollars < 1
            else f"${bb_dollars:g}"
        )
        return f"{sb_str}/{bb_str}"

    _default_venue, _stake_default = _pack_display_framing(pack)
    col3, col4 = st.columns(2)
    with col3:
        _stake_bb = st.selectbox(
            "Stakes (rendered in output)",
            options=list(COMMON_STAKE_LEVELS_BB_DOLLARS),
            index=list(COMMON_STAKE_LEVELS_BB_DOLLARS).index(_stake_default),
            format_func=_stake_label_preflop,
            key=f"preflop_stakes_{pack.pack_id}",
            help=(
                "Cosmetic: dollar amounts in the Question/Seats/POT scale "
                "to this stake; the underlying solve is stake-independent "
                "(all math in bb). The 9-max pack's rake (10% capped 3bb) "
                "matches a $1/$2 live cap of $6, so $1/$2 reads most "
                "coherent there."
            ),
        )
    with col4:
        _venue = st.radio(
            "Venue (Live or Online column)",
            options=["Online", "Live"],
            index=["Online", "Live"].index(_default_venue),
            horizontal=True,
            key=f"preflop_venue_{pack.pack_id}",
            help=(
                "Cosmetic framing for the CSV's 'Live or Online' + Context "
                "columns. The 9-max pack (9-handed, 4x opens, heavy capped "
                "rake) is shaped like a live low-stakes game."
            ),
        )

    st.divider()

    # --- 8. Model + API ---
    st.subheader("8. Model + API settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        _model = st.radio(
            "Model",
            options=list(_MODEL_LABEL_TO_API),
            index=0,  # Opus 4.7 -- the default model
            key="preflop_model",
        )
    # (A "Questions per API call" selector used to sit here -- it was never
    # wired to anything. Generation is one question per API call by design:
    # per-spot validators + retries need surgical failures, and prompt
    # caching already amortizes the big system prompt across calls.)
    with col2:
        _dry_run = st.toggle(
            "Dry run (no API calls)",
            key="preflop_dry_run",
        )

    # Cost estimate (rough per-question by model tier).
    cost_per_q = 0.15 if "Opus" in _model else 0.08
    est_cost = total * cost_per_q
    st.info(
        f"**Estimated**: {total} questions · ~${est_cost:.2f} · "
        f"difficulty {band_low}-{band_high} · {n_filtered:,} "
        f"nodes available"
    )

    st.divider()

    # --- 9. Prompt (which system prompt this batch runs on) ---
    st.subheader("9. Prompt")
    from admin_panel.prompt_library import PromptLibrary  # noqa: PLC0415
    from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
        build_preflop_system_prompt,
    )

    _plib = PromptLibrary()
    _plib.ensure_seeded(
        build_preflop_system_prompt, legacy_override=PREFLOP_PROMPT_OVERRIDE_PATH
    )
    _prompt_entries = _plib.list()
    _active_slug = _plib.active_slug()
    _prompt_text: str | None = None
    _prompt_name = ""
    if not _prompt_entries:
        st.caption("No prompts in the library; the built-in default will be used.")
    else:
        _pslugs = [e.slug for e in _prompt_entries]
        _pnames = {e.slug: e.name for e in _prompt_entries}
        if st.session_state.get("gen_prompt_select") not in _pslugs:
            st.session_state["gen_prompt_select"] = _active_slug or _pslugs[0]
        _chosen = st.selectbox(
            "Run this batch with prompt",
            options=_pslugs,
            format_func=lambda s: (
                f"{_pnames[s]}  ★ active" if s == _active_slug else _pnames[s]
            ),
            key="gen_prompt_select",
            help=(
                "Defaults to the ★ active prompt. Pick another to A/B it -- the "
                "batch and every output row are tagged with this prompt's name. "
                "Manage prompts on the Prompt library page."
            ),
        )
        _entry = _plib.get(_chosen)
        _prompt_text = _entry.text
        _prompt_name = _entry.name
        _not_active = "" if _chosen == _active_slug else "  ·  (not the active prompt)"
        st.caption(f"**{_prompt_name}** · {len(_prompt_text):,} chars{_not_active}")

    _cmp1, _cmp2 = st.columns(2)
    with _cmp1:
        _pin_seed = st.toggle(
            "Use the same hands every run",
            key="gen_pin_seed",
            help=(
                "Normally each batch draws a fresh random set of hands, so two "
                "runs aren't comparable. Turn this on to reuse the SAME hands "
                "every run (chosen by the seed number below). Then if you change "
                "the prompt and run again, any difference in the output comes "
                "from the PROMPT, not from getting different hands — like giving "
                "two students the identical test."
            ),
        )
    with _cmp2:
        _deterministic = st.toggle(
            "Make the wording repeatable",
            key="gen_temp0",
            help=(
                "The AI has a 'creativity' setting (called temperature). Left "
                "alone it's slightly random, so the same hand can come out worded "
                "differently each run. Turn this on to set it to zero: the same "
                "prompt on the same hand gives the exact same answer every time. "
                "Handy when comparing prompts so you're judging the prompt, not "
                "random wording luck. Leave it off for real batches — a little "
                "variety reads more naturally."
            ),
        )
    _seed_input = st.number_input(
        "Test-set seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        key="gen_seed_val",
        disabled=not _pin_seed,
        help="Same seed + same filters = the same sampled spots.",
    )
    _seed_val = int(_seed_input) if _pin_seed else None
    _temp_val = 0.0 if _deterministic else DEFAULT_TEMPERATURE

    st.divider()

    # --- 10. Generate button (kicks off a BACKGROUND job) ---
    # --- Layer-7 LLM audit (opt-in) -----------------------------------------
    # ONE mutually-exclusive choice. The three modes are exclusive in the batch
    # code: the auto-fix pass runs the claim check ITSELF as its gate, so a
    # separate "flag only" on top of it does nothing. A radio makes the
    # do-nothing combination impossible to set.
    layer7_mode = st.radio(
        "Layer 7 mode",
        options=["Off", "Flag only", "Audit & auto-fix"],
        index=2,  # default to Audit & auto-fix (per the user's request)
        horizontal=True,
        key="preflop_layer7_mode",
        help="Off = no AI audit. Flag only = one extra LLM call per question that "
        "FLAGS suspect claims (never rewrites); flags show under the explanation in "
        "Review and Compare. Audit & auto-fix = when a question is flagged, a "
        "further LLM pass rewrites the prose to fix it, re-checked by the "
        "deterministic hard validators (a rewrite that breaks a rule is discarded, "
        "the original kept). Only the prose changes; the action, numbers, and four "
        "options stay solver-locked. (Auto-fix already includes the claim check, so "
        "there's no separate 'flag only' to add on top.)",
    )
    run_claim_checker = layer7_mode == "Flag only"
    revise_pass = layer7_mode == "Audit & auto-fix"
    final_audit = False
    if revise_pass:
        final_audit = st.checkbox(
            "Final audit after the fix",
            value=True,
            key="preflop_final_audit",
            help="Re-runs the claim checker on the rewritten explanation as a last "
            "check -- it only flags for review, it never triggers another rewrite.",
        )
    with st.popover("ℹ️  How checking & validation works"):
        st.markdown(
            "Two kinds of checks run on every question, and **only one uses "
            "the AI**.\n\n"
            "**1. Gates — before any AI call (all deterministic).** A spot is "
            "dropped if it isn't worth a question: top-action frequency outside "
            "the 55–95% window, difficulty outside the chosen band, EV gap too "
            "small, an unconverged solver node, or a near-zero-frequency "
            "premise (a 'ghost' line). No prose is involved.\n\n"
            "**2. Hard validators — after the explanation is written "
            "(deterministic).** If the prose breaks a rule it is rejected and "
            "the model rewrites it once. Checks: invented options, banned "
            "phrases (em dash / semicolon), a suit emoji for a card you don't "
            "hold, a 'blocks X' claim not in the data, wrong preflop "
            "terminology (a raiser 'cold-calling', a 'squeeze' with no caller). "
            "If it still fails, the spot is routed to human review.\n\n"
            "**3. Soft validators — flag only (deterministic).** Tuned for "
            "precision (a flag should mean a human should look). Four checks: "
            "(a) prose calls hero in/out of position contradicting the computed "
            "position; (b) a cited equity / pot-odds % that contradicts the "
            "data block (heads-up spots only); (c) the opening verdict's action "
            "(fold / call / raise) conflicts with the spot's answer; (d) prose "
            "claims villain holds a hand that's at 0% in their range. Never "
            "rejects.\n\n"
            "**4. Claim checker — the ONLY AI step (opt-in, flag only).** A "
            "second LLM pass reads the explanation plus the solver data and "
            "flags claims that are confusing, misleading, or wrong. One extra "
            "API call per question. It fails open, so it can never block a "
            "good explanation.\n\n"
            "**Order:** gates → write the explanation (hard validators + one "
            "retry) → soft validators (flag) → claim checker (flag, if on).\n\n"
            "**Heads-up on the Review tab:** the orange *'Flagged by a soft "
            "validator'* badge shows BOTH the deterministic soft warnings AND "
            "the AI claim-checker's notes together, so an eloquent, "
            "reasoning-style flag there is the claim checker (AI), not a "
            "deterministic validator."
        )
    claim_checker_prompt: str | None = None
    if run_claim_checker or revise_pass:  # the revise pass uses the checker as its gate
        ck_key = "preflop_claim_checker_prompt"
        if ck_key not in st.session_state:
            st.session_state[ck_key] = _load_claim_checker_prompt()
        with st.expander("Claim-checker prompt (editable)"):
            edited = st.text_area(
                "System prompt the claim checker runs with",
                height=320,
                key=ck_key,
            )
            if edited.strip() and edited != _load_claim_checker_prompt():
                _save_claim_checker_prompt(edited)
                st.caption("Saved.")
        claim_checker_prompt = st.session_state[ck_key]

    # Sanity audit (opt-in, July 2026): the ONE LLM pass allowed to use its
    # own poker knowledge, pointed at the SOLVER DATA facts (never prose).
    # Flag-only; independent of the Layer 7 prose modes above.
    run_sanity_audit = st.checkbox(
        "🩺 Sanity audit — AI checks the solver FACTS against basic poker (NEW, flag only)",
        value=False,
        key="preflop_run_sanity_audit",
        help=(
            "One extra AI call per question that reads ONLY the solver-data "
            "facts (positions, equity, pot odds, domination lists, action "
            "history) and flags anything that contradicts basic poker. It "
            "never touches the prose or blocks a question; flags appear in "
            "Review as hypotheses for YOU to judge. This is the check that "
            "would have caught the blind-vs-blind position bug and the "
            "empty domination list. Read the ℹ️ below."
        ),
    )
    with st.popover("ℹ️  What is the Sanity audit? (and why it can be wrong)"):
        _render_sanity_audit_explainer()

    # --- Fast test mode: fewer equity run-outs (default OFF) ----------------
    # ON = 200 run-outs (≈2x faster, for iterating); OFF = 400 (most accurate,
    # for the real questions you'll ship). Defaults OFF (accurate) per the
    # user's request -- flip it ON only when iterating.
    fast_test_mode = st.toggle(
        "⚡ Fast test mode (fewer equity run-outs)",
        value=False,
        key="preflop_fast_test_mode",
        help="ON = ~2x faster generation by dealing 200 equity run-outs per "
        "spot instead of 400. The equity % each question shows gets slightly "
        "noisier (about ±1 point); that number also feeds the difficulty score "
        "and a few equity-based tags, but it stays inside the quality bar (the "
        "number-checker only rejects equity off by >3%). Leave it ON while "
        "testing. Turn it OFF for the FINAL questions you ship to use the full "
        "400 run-outs (most accurate equity).",
    )
    equity_runouts = 200 if fast_test_mode else 400

    # Inputs ready when: at least one position selected AND at least one
    # action context AND the filters match >= 1 node AND total > 0.
    # Disabled while another job is in flight -- the active-job panel
    # rendered above shows that one.
    can_generate = (
        bool(hero_positions)
        and bool(action_contexts)
        and n_filtered > 0
        and total > 0
    )
    job_active = jobs.has_active_job()

    if job_active:
        st.button(
            "GENERATE BATCH  (a job is already running -- see panel above)",
            disabled=True,
            type="primary",
            use_container_width=True,
            key="preflop_generate_btn_busy",
        )
    elif not can_generate:
        st.button(
            "GENERATE BATCH",
            disabled=True,
            type="primary",
            use_container_width=True,
            key="preflop_generate_btn_disabled",
        )
        st.caption(
            "Pick at least one hero position, one action context, "
            "and set a target above. The button activates once filters "
            "match at least one node."
        )
    elif st.button(
        "GENERATE BATCH",
        type="primary",
        use_container_width=True,
        key="preflop_generate_btn",
    ):
        # Snapshot the pack choice so the next session (or a panel
        # restart) regenerates against the same tree by default.
        # Snapshot this batch's pack + filters + style so the page re-seeds
        # them next time (across tab-switches / panel restarts) -- not just the
        # pack. Mirrors the PLO page.
        gen_settings.save_settings(
            PREFLOP_GEN_SETTINGS_PATH,
            {k: st.session_state.get(k) for k in _PREFLOP_GEN_SAVED_KEYS},
        )
        _start_preflop_job(
            pack=pack,
            hero_positions=list(hero_positions),
            action_contexts=list(action_contexts),
            player_counts=list(player_counts),
            freq_min=freq_low / 100.0,
            freq_max=freq_high / 100.0,
            exclude_ambiguous_band=exclude_ambiguous_band,
            exclude_near_pure_band=exclude_near_pure_band,
            min_difficulty=int(band_low),
            max_difficulty=int(band_high),
            trap_difficulty=bool(trap_difficulty),
            razor_difficulty=bool(razor_difficulty),
            diversify=bool(diversify),
            min_ev_gap_bb=(None if min_ev_gap == 0.0 else float(min_ev_gap)),
            min_villain_line_pct=(
                None if min_villain_pct == 0.0 else float(min_villain_pct)
            ),
            min_hero_premise_freq=(
                None if min_premise_pct == 0.0 else float(min_premise_pct) / 100.0
            ),
            display_in_bb=_currency.startswith("Big blinds"),
            stakes_bb_dollars=float(_stake_bb),
            live_or_online=_venue,
            total_questions=int(total),
            output_filename=_out_filename,
            model_label=_model,
            dry_run=bool(_dry_run),
            answer_style=answer_style_canonical,
            system_prompt=_prompt_text,
            prompt_name=_prompt_name,
            random_seed=_seed_val,
            temperature=_temp_val,
            run_claim_checker=run_claim_checker,
            claim_checker_prompt=claim_checker_prompt,
            revise_pass=revise_pass,
            final_audit=final_audit,
            run_sanity_audit=bool(run_sanity_audit),
            equity_runouts=equity_runouts,
        )


def _start_preflop_job(  # noqa: PLR0913 -- thin UI->batch parameter pass-through
    *,
    pack: PreflopPack,
    hero_positions: list[str],
    action_contexts: list[str],
    player_counts: list[int],
    freq_min: float,
    freq_max: float,
    exclude_ambiguous_band: bool,
    exclude_near_pure_band: bool,
    min_difficulty: int,
    max_difficulty: int,
    trap_difficulty: bool,
    razor_difficulty: bool,
    diversify: bool,
    min_ev_gap_bb: float | None,
    min_villain_line_pct: float | None,
    min_hero_premise_freq: float | None,
    display_in_bb: bool,
    total_questions: int,
    output_filename: str,
    model_label: str,
    dry_run: bool,
    answer_style: str,
    stakes_bb_dollars: float = 0.50,
    live_or_online: str = "Online",
    system_prompt: str | None = None,
    prompt_name: str = "",
    random_seed: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    run_claim_checker: bool = False,
    claim_checker_prompt: str | None = None,
    revise_pass: bool = False,
    final_audit: bool = False,
    run_sanity_audit: bool = False,
    equity_runouts: int = 400,
) -> None:
    """Kick off a preflop batch on a background thread and rerun.

    Before this refactor the batch ran inline in the Streamlit script
    thread, so any rerun (sidebar click, tab switch, button press)
    abandoned the in-flight work. Now we spawn the batch via
    :mod:`admin_panel.jobs`; progress + result are rendered by
    :func:`_render_preflop_job_panel`, which the page calls at its top
    and which polls the job state every second.
    """
    import os  # noqa: PLC0415

    # Fail loudly + early if dry-run is off but no API key is loaded.
    # Layer 6's lazy Anthropic() constructor would surface the same
    # error mid-batch, but catching it here saves the user from a
    # half-attempted run.
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "❌ ANTHROPIC_API_KEY is not set. Add it to `.env` at the "
            "repo root (the admin panel auto-loads .env), or enable "
            "**Dry run** above to test the pipeline without API calls."
        )
        return

    PREFLOP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Always append a timestamp suffix so each batch lands in its own
    # file -- prevents the History tab from showing only the latest
    # batch because every run overwrote `preflop_batch.csv`. The user
    # input becomes a prefix; trailing `.csv` is stripped first so we
    # don't end up with `name.csv_20260527_2310.csv`.
    stem = output_filename.removesuffix(".csv").strip() or "preflop_batch"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PREFLOP_OUTPUT_DIR / f"{stem}_{timestamp}.csv"

    model_api = _MODEL_LABEL_TO_API.get(model_label, model_label)

    label = (
        f"{total_questions} preflop questions"
        + (" (dry-run)" if dry_run else f" · {model_label.split()[0]}")
        + f" → {output_path.name}"
    )

    try:
        # Subprocess (not thread): batch generation runs in its own
        # interpreter so it never contends with the Streamlit UI for the
        # GIL. All kwargs below are picklable; the child re-creates the
        # Anthropic client itself from ANTHROPIC_API_KEY in the env.
        jobs.start_subprocess_job(
            generate_preflop_batch,
            label=label,
            pack=pack,
            output_path=output_path,
            total_questions=total_questions,
            hero_positions=hero_positions,
            action_contexts=action_contexts,
            player_counts=player_counts,
            min_frequency=freq_min,
            max_frequency=freq_max,
            exclude_ambiguous_band=exclude_ambiguous_band,
            exclude_near_pure_band=exclude_near_pure_band,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
            trap_difficulty=trap_difficulty,
            razor_difficulty=razor_difficulty,
            diversify=diversify,
            min_ev_gap_bb=min_ev_gap_bb,
            min_villain_line_pct=min_villain_line_pct,
            min_hero_premise_freq=min_hero_premise_freq,
            display_in_bb=display_in_bb,
            stakes_bb_dollars=stakes_bb_dollars,
            live_or_online=live_or_online,
            answer_style=answer_style,
            model=model_api,
            temperature=temperature,
            system_prompt=system_prompt,
            prompt_name=prompt_name,
            dry_run=dry_run,
            random_seed=random_seed,
            run_claim_checker=run_claim_checker,
            claim_checker_prompt=claim_checker_prompt,
            revise_pass=revise_pass,
            final_audit=final_audit,
            run_sanity_audit=run_sanity_audit,
            equity_runouts=equity_runouts,
        )
    except RuntimeError as exc:
        # Another job is already running. The button-disable check
        # normally prevents this, but races are possible if two browser
        # tabs hit Generate at the same time.
        st.error(f"⚠️ Could not start batch: {exc}")
        return

    # Re-run immediately so the job panel takes over the next render.
    st.rerun()


def _render_razor_difficulty_explainer() -> None:
    """Popover content for the "Razor's-edge difficulty" toggle.

    Reads the graded floors live from pipeline.preflop.razor_edge so the
    numbers shown are always the ones the algorithm uses.
    """
    from pipeline.preflop.razor_edge import (  # noqa: PLC0415
        RAZOR_FLOOR_BY_COUNT,
        RAZOR_FLOOR_MAX,
    )

    _one = RAZOR_FLOOR_BY_COUNT.get(1)
    _two = RAZOR_FLOOR_BY_COUNT.get(2)
    st.markdown(
        f"""
### 🔪 Razor's-edge difficulty

**The problem it fixes.** A hand the solver plays 100% one way is rated
Easy by the normal formula, because the ACTION is clear-cut. But some of
those hands are exactly what studying is for: the hand sits **right on the
edge of the range**, and the skill being tested is knowing where the
cutoff is. "ATo always folds to this 3-bet while AJo always calls" is
trivial for the solver and genuinely hard for a person.

**How it works (deterministic, no LLM, never touches the answer).** For
each spot, it looks up the hand's GRID NEIGHBORS at the same decision:
one kicker up and down (ATo → AJo / A9o), the suited-offsuit twin
(ATo → ATs), and for pockets the adjacent pairs (TT → JJ / 99). If a
neighbor that actually reaches this spot takes a DIFFERENT action
(fold vs call vs raise; two raise sizes don't count), the hand is a
boundary hand and its difficulty is floored:

- **1 opposing neighbor → {_one}** (a plain cutoff hand, upper-Medium)
- **2 opposing neighbors → {_two}** (boxed in from two directions, Hard)
- **3+ opposing neighbors → {RAZOR_FLOOR_MAX}** (an "island" doing what
  none of its neighbors do, usually a blocker story, deep-Hard)

**Impact on the questions you get**
- ON + a Medium or Hard band → "where exactly is the line?" questions,
  including pure 100% spots (the other route besides 🪤 Trap-aware, which
  catches pot-odds contradictions instead; a spot that is both keeps the
  higher rating).
- It changes **difficulty scores only** — never the answer, options,
  prose, or which spots are worthy.
- Expect boundary hands to be a meaningful slice of any range (edges are
  everywhere), so the pool is larger than the trap pool.
"""
    )


def _render_sanity_audit_explainer() -> None:
    """Popover content for the "Sanity audit" toggle."""
    st.markdown(
        """
### 🩺 Sanity audit

**What it is.** One extra AI call per question that reads ONLY the
solver-data facts (positions, equity, pot odds, domination lists, action
history) — never the explanation prose — and asks: *does anything here
contradict basic poker?* Example catches it is designed for: a
blind-vs-blind spot labeled "in position" for the small blind, or an
empty "hands that dominate you" list when the villain's likely hands
obviously include better aces. Both of those were real bugs that every
other check missed.

**Why every other check misses this kind of bug.** All the other
validators verify CONSISTENCY: the prose against the data block, the CSV
against a deterministic rebuild. When a deterministic FACT is itself
wrong, every layer faithfully agrees with it. This audit is the one
place an AI is allowed to bring OUTSIDE poker knowledge, which is the
check a human reviewer's eyes normally perform.

**Why it is flag-only — and why its flags can be WRONG.** This whole
pipeline exists because AIs are confidently wrong about poker. So this
checker's opinions are treated as **hypotheses for you to judge**, never
as gates: it cannot rewrite anything, reject a question, or change a
score. It is prompted to flag only high-confidence basics (action order,
domination direction, arithmetic, implausible equity) and to stay silent
on anything strategic. A flag means "a human should look", not "this is
wrong". (Calibration honesty: the first version's flags on a clean batch
were ALL false positives, so v2 embeds the exact reference rules and
those misfires as do-not-flag examples, and a flag now ships only when
**two independent passes challenge the same fact**.)

**The deterministic sibling that never cries wolf.** The fact categories
this audit fumbles most (position from seats, domination direction,
difficulty bands, frequency sums) are ALSO verified by a zero-AI
cross-check that runs automatically on every batch and shows a 🔬 badge
in Review. Trust the 🔬 findings outright; treat 🩺 flags as pointers.

**Where flags show up.** The question is marked *flagged* and the notes
appear under it on the Review page, alongside any claim-checker notes.

**Cost.** One extra API call per question (a second confirm call only on
the rows the first pass flags).
"""
    )


def _render_trap_difficulty_explainer() -> None:
    """Popover content for the "Trap-aware difficulty" toggle.

    Reads the graded-floor constants live from ``pipeline.trap_grading`` so
    the numbers shown are always the ones the algorithm uses.
    """
    from pipeline.trap_grading import (  # noqa: PLC0415
        TRAP_FLOOR_MAX,
        TRAP_FLOOR_MIN,
        TRAP_MARGIN_AT_MAX,
        TRAP_MARGIN_AT_MIN,
    )

    st.markdown(
        f"""
### 🪤 Trap-aware difficulty

**The problem it fixes.** Normally a question is rated **Hard** when the
solver's top action is a *close mix* (e.g. 60/40) — i.e. when it barely
matters what you pick. Two side effects fall out of that: every Hard question
ends up a near-coinflip with **~no EV difference**, and a **pure** spot
("Always fold", 100%) can *never* be rated above ~Easy (a pure action looks
"easy" by that measure). So a "Medium/Hard **and** 100% frequency" batch
returns **nothing**.

**What this adds.** A *second kind* of hard question — a **trap**: a spot where
the solver's clear answer is the **opposite** of what the hand looks like it
should do. Example: you hold enough equity to call the price, but the solver
folds 100% (domination / reverse implied odds). The answer is clear-cut and the
EV difference is **large** — but a human reads it wrong. Hard for a *person*,
not for the solver.

**How it works (deterministic — no LLM, never touches the answer).** For each
spot facing a bet, it compares the solver's main action to a simple **pot-odds
baseline** ("would a player call, based on equity vs. the price?"). When they
**disagree by a clear margin** — fold despite enough equity, or call/3-bet
despite too little — the spot is a trap and its difficulty is **floored to a
GRADED value between {TRAP_FLOOR_MIN} and {TRAP_FLOOR_MAX}**, scaled by HOW FAR
equity sits on the wrong side of the price: a mild contradiction
({TRAP_MARGIN_AT_MIN:.0%} of equity) rates {TRAP_FLOOR_MIN} (upper-Medium), an
extreme one ({TRAP_MARGIN_AT_MAX:.0%}+) rates {TRAP_FLOOR_MAX} (deep-Hard).
The typical trap on the 8-max packs (~16 points of contradiction) rates ~2430.
Opening spots (no price to call) are never traps. Non-trap spots score exactly
as before. (Graded July 2026; the old version pinned every trap to a flat
2400.)

**Impact on the questions you get**
- ON + the **Hard** band → strongly counterintuitive questions with a **single
  clear correct answer and a real EV gap**, *including* pure "Always X" spots.
- ON + the **Medium** band → also picks up *mildly* counterintuitive traps
  (small equity-vs-price contradictions). This is what makes "Medium/Hard at
  100%" actually produce questions.
- It changes **difficulty scores only** — never the answer, options, prose, or
  which spots are "worthy".
- **Safety net:** a trap that's a *fold-with-equity* is also auto-flagged for
  human review (the same check that catches broken-solve folds like the Monker
  QQ artifact), so a bad solve can't silently ship as a "hard" question.

**Old vs. new — which to use**
- **Off:** good general default. Hard = genuinely close decisions.
- **On:** use when you specifically want *counterintuitive* questions (the
  "trap" study category) — or when a Medium/Hard batch keeps coming back as
  all-mixes. **Recommended ON for Medium and Hard batches; leave OFF for
  Easy** (the Easy band sits below every graded floor, so it has no effect
  there).
"""
    )


def _render_difficulty_explainer() -> None:
    """Popover content for "How is Difficulty calculated?".

    Reads constants directly from :mod:`pipeline.preflop.difficulty`
    so the popover never drifts from the actual formula -- the
    weight / table / bump-rule values you see here ARE the values
    the algorithm uses. Edit ``pipeline/preflop/difficulty.py`` to
    tune; this popover updates automatically.
    """
    # Lazy imports: read live values from the difficulty module so the
    # popover never lies about what the algorithm is doing.
    from pipeline.preflop.difficulty import (  # noqa: PLC0415
        ARCHETYPE_BASE_EASE,
        BUMP_RULES,
        CONCEPT_TAG_MODIFIERS,
        HAND_CLASS_EASE,
        W_CONCEPT,
        W_EV,
        W_FREQ,
        W_HAND,
    )

    # --- overview ---
    st.markdown(
        f"""
**Difficulty Rating** runs roughly **500 (easiest) → 3000 (hardest)**
on the brief's MVP Elo-style scale. Soft bounds at **[400, 3200]**
absorb rare outliers without losing information.

The score blends **four independent axes**, each a number in [0, 1]
where 1 = "easy on this dimension" and 0 = "hard". The four are
combined as a weighted average and (optionally) adjusted by **bump
rules**:

```text
easy = ({W_FREQ:.2f} × easy_freq)
     + ({W_EV:.2f} × easy_ev)         [redistributed when ev unavailable]
     + ({W_CONCEPT:.2f} × easy_concept)
     + ({W_HAND:.2f} × easy_hand)

easy += sum(bump.easy_delta for bump in BUMP_RULES if bump matches)

difficulty = round(clip(3000 − easy × 2500, 400, 3200))
```

Result: an integer in roughly [500, 3000] (with rare ±100-200 outliers
in [400, 3200]). The four per-axis values are surfaced as the
diagnostic CSV columns **`easy_freq`, `easy_ev`, `easy_concept`,
`easy_hand`** (each in [0, 1] -- **1 = easy** on that axis, **0 =
hard**) so a reviewer can see exactly which axis made a spot easy or
hard. Each axis below names its column. (Bump deltas still nudge the
score internally, but `difficulty_bumps` is no longer a CSV column --
dropped June 2026 since the bump table ships empty.)
"""
    )

    # --- axis 1: freq ---
    st.markdown(
        f"""
### Axis 1 — Top action frequency · CSV `easy_freq` &nbsp;·&nbsp; weight **{W_FREQ:.0%}**

How dominant the correct answer is in Pio's strategy. Lower freq =
more genuinely mixed = harder to identify the "right" answer.

```text
easy_freq = clip((dominant_freq − 0.55) / 0.45, 0, 1)
```

| dominant_freq | easy_freq | contribution to difficulty |
|---|---|---|
| 55% (worthiness floor) | 0.00 | +1000 (hardest end) |
| 66% | 0.244 | +756 |
| 75% | 0.444 | +556 |
| 85% | 0.667 | +333 |
| 95% (Always threshold) | 0.889 | +111 |
| 100% (pure strategy) | 1.00 | +0 (easiest end) |
"""
    )

    # --- axis 2: ev ---
    st.markdown(
        f"""
### Axis 2 — EV gap · CSV `easy_ev` &nbsp;·&nbsp; weight **{W_EV:.0%}**

The chip cost (in bb) of picking the second-best action over the
correct one. Bigger gap = clearer "right answer" = easier spot.

```text
easy_ev = clip(ev_gap_bb / 3.0, 0, 1)
```

3 bb is the "fully easy" cap -- beyond that the spot is already
trivial. When the EV engine can't compute the gap (raise-involved
spots in v1, since we need postflop solves to score raise EVs), the
EV weight (**{W_EV:.0%}**) is **redistributed proportionally** across
the other three axes rather than being treated as a neutral 0.5.
This stops raise spots from being artificially pulled toward
mid-difficulty.

| ev_gap_bb | easy_ev | contribution |
|---|---|---|
| 0.0 (coin-flip) | 0.00 | +750 (hardest) |
| 0.5 | 0.167 | +625 |
| 1.5 | 0.500 | +375 |
| 3.0+ (capped) | 1.00 | +0 (easiest) |
| unavailable | — | weight redistributes |
"""
    )

    # --- axis 3: concept ---
    st.markdown(
        f"""
### Axis 3 — Concept · CSV `easy_concept` &nbsp;·&nbsp; weight **{W_CONCEPT:.0%}**

How strategically complex the spot's CONTEXT is. Built from two
pieces: the spot's `archetype` (one of 16 from
`pipeline.preflop.fact_extractor.classify_archetype`) gives a base
value, then `concept_tag` modifiers add or subtract.

```text
easy_concept = ARCHETYPE_BASE_EASE[archetype]
             + sum(CONCEPT_TAG_MODIFIERS[tag] for each firing tag)
             clipped to [0.05, 1.0]
```

**Archetype base-ease table (live values):**
"""
    )
    arch_df = pd.DataFrame(
        sorted(
            ARCHETYPE_BASE_EASE.items(), key=lambda kv: -kv[1]
        ),
        columns=["Archetype", "easy_concept base"],
    )
    st.dataframe(arch_df, hide_index=True, use_container_width=True)

    st.markdown("**Concept-tag modifiers (live values):**")
    mod_df = pd.DataFrame(
        [
            {
                "Concept tag (must fire)": tag,
                "easy_concept delta": f"{delta:+.2f}",
                "Why": _CONCEPT_MOD_RATIONALE.get(tag, ""),
            }
            for tag, delta in CONCEPT_TAG_MODIFIERS.items()
        ],
        columns=["Concept tag (must fire)", "easy_concept delta", "Why"],
    )
    st.dataframe(mod_df, hide_index=True, use_container_width=True)

    # --- axis 4: hand ---
    st.markdown(
        f"""
### Axis 4 — Hand class · CSV `easy_hand` &nbsp;·&nbsp; weight **{W_HAND:.0%}**

The hero's hand class shapes how obvious the right action is.
**U-shaped**: extreme hands (premium pairs OR clear trash) are easy;
marginal hands (small pairs, suited connectors, suited aces) are
hard because the right action requires real strategic reasoning.

```text
easy_hand = HAND_CLASS_EASE[matched_tag]  # first match wins
            default 0.55 when no tag matches
```

**Hand class ease table (live values):**
"""
    )
    hand_df = pd.DataFrame(
        [
            {
                "Hand class tag": tag,
                "easy_hand": f"{ease:.2f}",
                "Examples": _HAND_TAG_EXAMPLES.get(tag, ""),
            }
            for tag, ease in HAND_CLASS_EASE.items()
        ],
        columns=["Hand class tag", "easy_hand", "Examples"],
    )
    st.dataframe(hand_df, hide_index=True, use_container_width=True)
    st.caption(
        "_Hands that don't match any of these tags (e.g. KQo -- broadway "
        "offsuit but not premium_unpaired or unconnected_offsuit) fall "
        "back to **0.55**._"
    )

    # --- bump rules ---
    st.markdown(
        """
### Bump rules

Signed additive deltas applied AFTER the weighted axis sum. Each rule
captures a known spot pattern the linear axis blend mis-scores --
typically synergies between axes (e.g. mixed strategy AND advanced
concept). Each bump's `easy_delta` is small (±0.05) so the axes
remain the dominant signal.

```text
easy += sum(rule.easy_delta for rule in BUMP_RULES if rule.predicate(facts, ev))
```
"""
    )
    if not BUMP_RULES:
        st.info(
            "**Currently no bump rules are active.** The table is "
            "intentionally empty at v1 ship -- bumps get added in "
            "`pipeline/preflop/difficulty.py:BUMP_RULES` as observed "
            "batches show specific spots the axis blend mis-scores."
        )
    else:
        bumps_df = pd.DataFrame(
            [
                {
                    "Name": rule.name,
                    "Delta": f"{rule.easy_delta:+.2f}",
                    "Trigger": rule.description,
                }
                for rule in BUMP_RULES
            ],
            columns=["Name", "Delta", "Trigger"],
        )
        st.dataframe(bumps_df, hide_index=True, use_container_width=True)

    # --- worked examples ---
    st.markdown(
        """
### Worked examples (real spots from user batches)

These show how the four axes combine on representative spots.
"""
    )
    examples_df = pd.DataFrame(
        [
            {
                "Spot": "BTN open AA",
                "easy_freq": 1.00, "easy_ev": 1.00,
                "easy_concept": 1.00, "easy_hand": 1.00,
                "easy_blend": 1.00, "difficulty": 500,
                "Reading": "Pure value open with the nuts. Easiest.",
            },
            {
                "Spot": "BB defend KQs vs BTN open",
                "easy_freq": 0.78, "easy_ev": 0.50,
                "easy_concept": 0.60, "easy_hand": 0.65,
                "easy_blend": 0.612, "difficulty": 1470,
                "Reading": "Standard defending spot, easy hand class.",
            },
            {
                "Spot": "BTN 33 vs BB 3-bet (user's spot #3)",
                "easy_freq": 0.24, "easy_ev": 0.46,
                "easy_concept": 0.70, "easy_hand": 0.45,
                "easy_blend": 0.420, "difficulty": 1950,
                "Reading": "Small pair facing 3-bet -- marginal hand, costly mistake.",
            },
            {
                "Spot": "Mixed 3-bet bluff with A5s, low EV gap",
                "easy_freq": 0.00, "easy_ev": 0.10,
                "easy_concept": 0.45, "easy_hand": 0.40,
                "easy_blend": 0.165, "difficulty": 2588,
                "Reading": "Close decision, complex archetype, marginal hand. Hard.",
            },
            {
                "Spot": "5-bet pot small pair, near-coinflip",
                "easy_freq": 0.05, "easy_ev": 0.00,
                "easy_concept": 0.10, "easy_hand": 0.45,
                "easy_blend": 0.085, "difficulty": 2788,
                "Reading": "Hardest tier: 5-bet pot + marginal hand + mixed strategy.",
            },
        ]
    )
    st.dataframe(examples_df, hide_index=True, use_container_width=True)

    # --- design notes ---
    st.markdown(
        """
### Notes

- **Pure spots can now reach high difficulty.** Pre-redesign, an
  "Always X" spot was capped at ~1668 difficulty regardless of context.
  Now a pure spot with a hard concept (5-bet pot) and marginal hand
  can score 1900+ -- the concept and hand axes lift it.
- **Mixed spots can reach low difficulty.** Conversely, a mixed-strategy
  spot with a premium hand and routine concept can score in the
  Easy tier when warranted.
- **The filter slider above is freq-only.** Per-spot equity (needed
  for EV gap) is expensive; the slider filters BEFORE we compute
  facts. The Difficulty Rating in the CSV uses all four axes.
- **To tune**, edit `pipeline/preflop/difficulty.py`:
  - Weights: `W_FREQ`, `W_EV`, `W_CONCEPT`, `W_HAND` (must sum to 1.0)
  - Concept: `ARCHETYPE_BASE_EASE`, `CONCEPT_TAG_MODIFIERS`
  - Hand: `HAND_CLASS_EASE`
  - Bumps: `BUMP_RULES` (add `BumpRule(name=..., easy_delta=..., predicate=...)`)
  - Bounds: `_LINEAR_FLOOR`, `_LINEAR_CEILING`, `_HARD_FLOOR`, `_HARD_CEILING`

  After editing, run a new batch from the Generate page -- changes
  take effect immediately (the values shown above are read live from
  the module).
"""
    )


# Rationale strings shown alongside concept-tag modifiers in the
# popover's table. Hand-curated; update if a modifier is added /
# changed in CONCEPT_TAG_MODIFIERS.
_CONCEPT_MOD_RATIONALE: dict[str, str] = {
    "multiway_pot": (
        "3+ players still in the pot -- more variables, harder to read"
    ),
    "short_stack": (
        "ICM + push/fold dynamics on top of the base archetype"
    ),
    "deep_stack": (
        "100bb cash is the standard simpler case (positive bump)"
    ),
}

# Example hand classes per hand-tag, shown alongside HAND_CLASS_EASE
# in the popover. Hand-curated; update if a tag is added / renamed.
_HAND_TAG_EXAMPLES: dict[str, str] = {
    "premium_pair":         "AA, KK, QQ",
    "premium_unpaired":     "AK, AQ (suited or off)",
    "unconnected_offsuit":  "73o, 84o, K3o (clear folds)",
    "suited_broadway":      "KQs, KJs, KTs, QJs, QTs, JTs",
    "medium_pair":          "JJ, TT, 99",
    "small_pair":           "88-22",
    "suited_ace":           "A2s-AJs (non-premium Axs)",
    "suited_connector":     "98s, 87s, 76s, 65s, 54s",
}


def _job_done_rerun_once(job) -> None:
    """From inside a ticking fragment: the moment the job flips to done,
    trigger ONE full-app rerun so the static result panel (rendered
    OUTSIDE any fragment) appears without user interaction. Guarded per
    job id -- the rerun must not loop."""
    flag = f"_job_done_rerun_{job.id}"
    if not st.session_state.get(flag):
        st.session_state[flag] = True
        st.rerun(scope="app")


@st.fragment(run_every=1.0)
def _render_active_job_progress() -> None:
    """The ONLY auto-refreshing part of the job panel: live progress.

    Mounted exclusively while a job is active, so the once-a-second
    ticking stops the moment the batch finishes. (June 2026 fix: the
    whole panel -- including the completed-batch UI with its CSV
    re-read, download button, and preview -- used to live inside the
    ticking fragment, so a finished batch kept the app busy every
    second forever; that was the post-batch "everything is laggy,
    spinner never stops" report.)
    """
    job = jobs.get_current_job()
    if job is None:
        return
    if not job.is_active:
        _job_done_rerun_once(job)
        return
    st.markdown(f"**🔄 Running:** {job.label}")
    progress = job.progress
    if progress.total > 0:
        pct = min(1.0, (progress.current + 1) / progress.total)
        st.progress(pct, text=progress.message or "Starting…")
    else:
        st.text(progress.message or "Starting…")
    st.caption(
        f"job id `{job.id}` · elapsed {job.elapsed_seconds:.0f}s · "
        "this keeps running if you switch tabs."
    )
    if job.cancellable and st.button(
        "⛔ Cancel batch", key=f"cancel_job_{job.id}"
    ):
        if jobs.request_cancel_current_job():
            st.warning("Cancelling… the batch will stop in a moment.")


def _render_preflop_job_panel() -> None:
    """Top-of-page panel showing the current (or last) background job.

    Three visual states:

    * **Active** (PENDING / RUNNING)  -- live progress via the ticking
      fragment above (the user can keep configuring the NEXT batch).
    * **Completed** -- success + download + preview, rendered as plain
      static content on normal reruns only.
    * **Failed** -- error + traceback in an expander.

    A "Clear" button hides a done/failed job from the panel; this is
    UX-only (the registry slot is freed so the next batch can start
    without an extra step).
    """
    job = jobs.get_current_job()
    if job is None:
        return

    with st.container(border=True):
        if job.is_active:
            _render_active_job_progress()
        elif job.status is jobs.JobStatus.COMPLETED:
            st.markdown(f"**✅ Last batch:** {job.label}")
            if isinstance(job.result, BatchResult):
                # Append to the usage log exactly once -- module-level
                # dedupe across browser sessions + reruns.
                _maybe_log_completed_job(job, job.result)
                _render_preflop_result_ui(job.result)
            else:
                st.warning("Job finished but returned no BatchResult.")
            st.caption(
                f"Finished in {job.elapsed_seconds:.0f}s. "
                "Configure and click GENERATE BATCH below to run another."
            )
            if st.button("Hide last result", key=f"clear_job_{job.id}"):
                jobs.clear_current_job()
                st.rerun()
        elif job.status is jobs.JobStatus.CANCELLED:
            st.warning(f"**⛔ Batch cancelled:** {job.label}")
            st.caption(
                f"Stopped after {job.elapsed_seconds:.0f}s. The batch writes "
                "its CSV only when it finishes, so a cancelled run saves "
                "nothing — start a new batch when ready."
            )
            if st.button("Dismiss", key=f"dismiss_cancel_{job.id}"):
                jobs.clear_current_job()
                st.rerun()
        else:  # FAILED
            st.error(f"**❌ Job failed:** {job.label}")
            with st.expander("Traceback"):
                st.code(job.error or "(no traceback captured)")
            if st.button("Dismiss failure", key=f"dismiss_job_{job.id}"):
                jobs.clear_current_job()
                st.rerun()

    st.divider()


def _log_batch_result_usage(result: BatchResult) -> None:
    """Append one batch's usage to the lifetime JSONL log.

    Dry-runs (``model_used == ""``) are skipped. Used by the job-completion
    hook AND the (inline) NLHE Compare page, whose two batches previously
    never reached the lifetime total.
    """
    if not result.model_used:
        return
    cost = usage.compute_cost_usd(
        model=result.model_used,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        cache_creation_tokens=result.total_cache_creation_tokens,
        cache_read_tokens=result.total_cache_read_tokens,
    )
    usage.append_log_entry(
        USAGE_LOG_PATH,
        model=result.model_used,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        cache_creation_tokens=result.total_cache_creation_tokens,
        cache_read_tokens=result.total_cache_read_tokens,
        cost_usd=cost,
        questions_written=result.questions_written,
        output_filename=result.output_path.name if result.output_path else "",
    )


def _maybe_log_completed_job(job: jobs.Job[BatchResult], result: BatchResult) -> None:
    """Append this job's usage to the JSONL log -- exactly once per process.

    Idempotent: tracks logged job ids in a module-level set so the
    fragment's per-second re-render doesn't duplicate the entry.
    """
    logged = _logged_job_ids()
    if job.id in logged:
        return
    logged.add(job.id)
    _log_batch_result_usage(result)


def _maybe_log_completed_postflop_job(job: jobs.Job[Any], result: Any) -> None:
    """Append a POSTFLOP batch's token spend to the lifetime usage log, once.

    The postflop ``PostflopBatchResult`` has no cache-token fields (postflop
    doesn't prompt-cache the way preflop does), so they log as 0. Dry-runs and
    zero-token results are skipped. Idempotent via the same module-level
    logged-id set as the preflop path."""
    logged = _logged_job_ids()
    if job.id in logged:
        return
    logged.add(job.id)
    model = getattr(result, "model_used", "") or ""
    # dry-run model_used is "(dry-run placeholder)" -> not a real model, skip.
    if getattr(result, "dry_run", False) or model.startswith("("):
        return
    in_tok = int(getattr(result, "total_input_tokens", 0) or 0)
    out_tok = int(getattr(result, "total_output_tokens", 0) or 0)
    if not (in_tok or out_tok):
        return
    cost = usage.compute_cost_usd(model=model, input_tokens=in_tok, output_tokens=out_tok)
    out_path = getattr(result, "output_path", None)
    usage.append_log_entry(
        USAGE_LOG_PATH,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        cost_usd=cost,
        questions_written=int(getattr(result, "questions_written", 0) or 0),
        output_filename=out_path.name if out_path else "",
    )


def _render_preflop_failures(failures: list) -> None:
    """Render per-spot failures with full question/options/explanation context.

    Each failure is its own expander -- the header is a one-line
    summary (position + hand + short error category), the body shows
    the deterministic context (Context line, full Question prose, all
    four options with the correct one marked) plus the LLM's actual
    last attempt and the validator's error message.

    Failures with no LLM output (pre-call errors) skip the
    "LLM's attempted explanation" block.
    """
    n = len(failures)
    st.warning(
        f"⚠️  **{n} spot{'s' if n != 1 else ''} routed to human review** "
        "(failed validation after retry budget exhausted)."
    )

    for i, failure in enumerate(failures, start=1):
        # Short error category for the expander header (first ~80 chars
        # after stripping the boilerplate prefix).
        err = (failure.error_message or "").strip()
        if "last error:" in err:
            err_short = err.split("last error:", 1)[1].strip()
        else:
            err_short = err
        err_short = err_short.replace("\n", " ")[:90]

        header_archetype = (
            f"· {failure.archetype}" if failure.archetype else ""
        )
        with st.expander(
            f"**#{i}**  {failure.hero_position} {failure.hand_class}  "
            f"{header_archetype}  ·  _{err_short}_"
        ):
            # --- Question side (deterministic) ---
            if failure.question_text:
                st.markdown("**Question:**")
                st.text(failure.question_text)
            else:
                st.caption("_(question render not available -- pre-LLM failure)_")

            # --- Options + correct ---
            if failure.options:
                st.markdown("**Options:**")
                for j, opt in enumerate(failure.options, start=1):
                    if not opt:
                        continue
                    is_correct = opt == failure.correct_answer
                    marker = "✅" if is_correct else "·"
                    st.text(f"  {marker} option {j}: {opt}")
                if failure.correct_answer:
                    st.caption(
                        f"_Correct answer (deterministic): "
                        f"`{failure.correct_answer}`_"
                    )

            # --- Action frequencies (so reviewer can sanity-check the
            #     validator's "Pio freq" claims in the error) ---
            if failure.action_frequencies:
                freqs = ", ".join(
                    f"{label}: {freq:.0%}"
                    for label, freq in sorted(
                        failure.action_frequencies.items(),
                        key=lambda kv: -kv[1],
                    )
                    if freq > 0
                )
                if freqs:
                    st.caption(f"_Pio strategy at this node: {freqs}_")

            # --- LLM's last attempt ---
            if failure.failed_explanation:
                st.markdown("**LLM's attempted explanation (last retry):**")
                st.info(failure.failed_explanation)

            # --- Validator / exception message ---
            st.markdown("**Why it was rejected:**")
            st.error(failure.error_message)

            # --- Debug footer ---
            st.caption(f"_node_id: `{failure.node_id}`_")


def _render_review_failures(
    csv_path: Path, present_keys: set[tuple[str, str]]
) -> None:
    """Routed-to-human-review queue on the Review page.

    Lists the spots a validator rejected during generation (persisted in the
    batch's ``.meta.json`` -- they never reached the CSV), each with the full
    deterministic context, the LLM's rejected explanation, and the rejection
    reason. A one-click **Approve into batch** appends the spot KEEPING that
    exact explanation (see :func:`review.promote_failure`) and marks it
    approved. ``present_keys`` = the (solver_reference, User Cards) already in
    the CSV, so an already-promoted spot reads as added, not offered again.
    """
    failures = review.load_failures(csv_path)
    if not failures:
        return
    n = len(failures)
    with st.expander(
        f"⚠️ Routed to human review — {n} spot{'s' if n != 1 else ''} "
        "(rejected during generation, not in the CSV)",
        expanded=False,
    ):
        st.caption(
            "A validator rejected these during generation, so they never made "
            "the CSV. Read each one -- if the explanation is good (validators "
            "do misfire), **Approve into batch** appends it KEEPING this exact "
            "text and marks it approved, so it flows into the approved pool."
        )
        for i, f in enumerate(failures):
            pos = str(f.get("hero_position", ""))
            hc = str(f.get("hand_class", ""))
            err = str(f.get("error_message", "")).strip()
            err_short = (
                err.split("last error:", 1)[-1].strip().replace("\n", " ")[:90]
            )
            with st.container(border=True):
                st.markdown(f"**#{i + 1}**  {pos} {hc}  ·  _{err_short}_")
                qtext = str(f.get("question_text", ""))
                if qtext:
                    st.text(qtext)
                opts = f.get("options") or []
                correct = str(f.get("correct_answer", ""))
                if isinstance(opts, list):
                    for j, opt in enumerate(opts, start=1):
                        if opt:
                            mark = "✅" if opt == correct else "·"
                            st.text(f"  {mark} option {j}: {opt}")
                freqs = f.get("action_frequencies") or {}
                if isinstance(freqs, dict) and freqs:
                    pretty = ", ".join(
                        f"{k}: {float(v):.0%}"
                        for k, v in sorted(freqs.items(), key=lambda kv: -kv[1])
                        if float(v) > 0
                    )
                    if pretty:
                        st.caption(f"_Solver strategy at this node: {pretty}_")
                fe = str(f.get("failed_explanation", ""))
                if fe:
                    st.markdown("**LLM's explanation (this is what gets kept):**")
                    st.info(fe)
                st.markdown("**Why it was rejected:**")
                st.error(err)

                row = f.get("row")
                row_dict = row if isinstance(row, dict) else {}
                row_key = (
                    str(row_dict.get("solver_reference", "")),
                    str(row_dict.get("User Cards", "")),
                )
                if not row_dict:
                    st.caption(
                        "_No rebuilt row for this one (a pre-LLM failure), so "
                        "it can't be one-click added -- re-generate it on the "
                        "Generate page if you want it._"
                    )
                elif row_key in present_keys:
                    st.success("✅ Already added to this batch.")
                elif st.button(
                    "➕ Approve into batch (keep this explanation)",
                    key=f"promote::{csv_path.name}::{i}",
                    type="primary",
                ):
                    ok, msg = review.promote_failure(csv_path, f)
                    if ok:
                        st.success(f"Added: {msg}")
                        st.rerun()
                    else:
                        st.warning(msg)


def _render_preflop_result_ui(result: BatchResult) -> None:
    """Render the per-batch result UI: cost, summary, failures, download, preview.

    Extracted from the old synchronous ``_run_preflop_generation`` so
    both the job-panel and any future "browse a prior batch" entry can
    share the same view. Pure render -- no side effects beyond Streamlit
    output and reading the CSV from disk.
    """
    # Cost ticker first so it sits next to the success banner. Dry-runs
    # have ``model_used=""`` and skip this block (no API was called).
    if result.model_used:
        cost = usage.compute_cost_usd(
            model=result.model_used,
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            cache_creation_tokens=result.total_cache_creation_tokens,
            cache_read_tokens=result.total_cache_read_tokens,
        )
        cache_note = ""
        if result.total_cache_read_tokens or result.total_cache_creation_tokens:
            cache_note = (
                f" · cache {result.total_cache_read_tokens:,} read / "
                f"{result.total_cache_creation_tokens:,} write"
            )
        st.markdown(
            f"💰 **This batch:** {usage.format_cost(cost)} · "
            f"{result.total_input_tokens:,} input / "
            f"{result.total_output_tokens:,} output tokens"
            f"{cache_note} · `{result.model_used}`"
        )

    if result.questions_written == 0:
        st.warning(
            f"No questions produced. "
            f"Nodes after filter: **{result.nodes_after_filter}**, "
            f"worthy spots available: **{result.worthy_spots_available}**, "
            f"rejected by difficulty/EV filters: "
            f"**{result.difficulty_filtered_out}**, "
            f"skipped as unconverged nodes: **{result.noise_filtered_out}**, "
            f"skipped as incoherent mixes: "
            f"**{result.incoherent_mix_filtered_out}**, "
            f"failures: **{len(result.failures)}**. "
            "Try a wider difficulty band, the Mixed preset, or a wider "
            "worthiness window."
            + (
                " — Note: a **Hard** band with a **near-pure worthiness window** "
                "matches almost nothing unless **🪤 Trap-aware difficulty** is ON "
                "(pure spots score Easy otherwise), so nearly all the "
                f"{result.difficulty_filtered_out} difficulty rejections are that. "
                "Enable trap-aware, or widen the band."
                if result.difficulty_filtered_out > max(1, result.worthy_spots_available // 2)
                else ""
            )
        )
    else:
        _band_note = (
            f", {result.difficulty_filtered_out} rejected by difficulty/EV "
            f"filters" if result.difficulty_filtered_out else ""
        )
        _noise_note = (
            f", {result.noise_filtered_out} skipped as unconverged solver "
            f"nodes" if result.noise_filtered_out else ""
        )
        _mix_note = (
            f", {result.incoherent_mix_filtered_out} skipped as incoherent "
            f"mixes" if result.incoherent_mix_filtered_out else ""
        )
        st.success(
            f"Wrote **{result.questions_written}** questions to "
            f"`{result.output_path}` "
            f"(attempted {result.questions_attempted}, "
            f"{result.worthy_spots_available} worthy spots available"
            f"{_band_note}{_noise_note}{_mix_note})."
        )
        # When the run delivered fewer than requested, say so plainly and
        # break down WHERE the shortfall went -- pre-LLM filter rejections
        # (no spend) vs post-LLM validation failures -- so it's never a
        # mystery why "asked for 15, got 11".
        _short = result.requested_questions - result.questions_written
        if result.requested_questions and _short > 0:
            _why_bits = []
            if result.difficulty_filtered_out:
                _why_bits.append(
                    f"**{result.difficulty_filtered_out}** of the "
                    f"**{result.worthy_spots_available}** worthy spots were "
                    "rejected by your difficulty-band / EV-gap filters "
                    "(before any LLM call -- no spend)"
                )
            if result.noise_filtered_out:
                _why_bits.append(
                    f"**{result.noise_filtered_out}** were skipped as "
                    "unconverged solver nodes (AA folding a jam / premium "
                    "inversions -- the noisy multiway tail)"
                )
            if result.incoherent_mix_filtered_out:
                _why_bits.append(
                    f"**{result.incoherent_mix_filtered_out}** were skipped as "
                    "incoherent mixes (a meaningfully-played action several bb "
                    "worse than the best per the solver's own EVs -- "
                    "unconverged-node noise, not a real mixed strategy)"
                )
            if result.rare_line_filtered_out:
                _why_bits.append(
                    f"**{result.rare_line_filtered_out}** were skipped because "
                    "the villain's whole line is near-never taken (min villain "
                    "line frequency gate)"
                )
            if result.rare_premise_filtered_out:
                _why_bits.append(
                    f"**{result.rare_premise_filtered_out}** were skipped "
                    "because hero's own earlier actions in the story are "
                    "near-never solver plays (premise-realism gate)"
                )
            if result.failures:
                _why_bits.append(
                    f"**{len(result.failures)}** failed validation after the "
                    "retry budget (shown below)"
                )
            _why = "; ".join(_why_bits) or "the worthy-spot pool was exhausted"
            st.warning(
                f"⚠️ You asked for **{result.requested_questions}** but "
                f"**{result.questions_written}** qualified. {_why}. To get "
                "more: widen the difficulty band (or use **Mixed**), lower "
                "the **min EV gap**, or broaden the position / action-context "
                "filters."
            )

    if result.failures:
        _render_preflop_failures(result.failures)

    if result.output_path is None or not result.output_path.is_file():
        return

    # Download button: utf-8-sig keeps the BOM intact so Excel picks
    # up the encoding right. The key includes (filename, count, size)
    # to force a fresh widget per batch -- without that, two batches
    # with the same filename could collide.
    csv_bytes = result.output_path.read_bytes()
    download_key = (
        f"download_{result.output_path.name}_"
        f"{result.questions_written}_{len(csv_bytes)}"
    )
    st.download_button(
        label=f"📥 Download {result.output_path.name}",
        data=csv_bytes,
        file_name=result.output_path.name,
        mime="text/csv",
        use_container_width=True,
        key=download_key,
    )
    # In-place preview: first ~20 rows of the most useful columns.
    df = pd.read_csv(result.output_path, encoding="utf-8-sig")
    preview_cols = [
        "No",
        "User Seat",
        "User Cards",
        "Hand Stage",
        "Context",
        "Question",
        "option 1",
        "option 2",
        "option 3",
        "option 4",
        "Correct Answer",
        "Answer Explanation",
        "Difficulty Rating",
        "action_frequencies",
        "ev_gap_bb",
        "skills",
    ]
    present_cols = [c for c in preview_cols if c in df.columns]
    st.caption(
        f"File: {result.output_path.name} · "
        f"{len(csv_bytes):,} bytes · {result.questions_written} questions · "
        f"{len(df.columns)} columns (the in-page preview below shows only "
        f"{len(present_cols)} of them; download for the full CSV)."
    )
    st.dataframe(
        df[present_cols].head(20), hide_index=True, use_container_width=True
    )


# --- page: History ---------------------------------------------------------
def _scan_preflop_outputs() -> pd.DataFrame:
    """Return a one-row-per-CSV table for the History page.

    Columns:
      * ``filename`` -- the file name (no path).
      * ``modified`` -- mtime as ``YYYY-MM-DD HH:MM`` (sortable + readable).
      * ``size_kb`` -- size in KB, rounded.
      * ``questions`` -- row count (newlines minus 1 for the header).
      * ``_path``    -- absolute path; prefixed with ``_`` so the UI table
                        can drop it from display while we still use it
                        for deletes + downloads.

    Empty (no CSVs) returns an empty DataFrame with the right columns.
    Newest-modified first -- the most useful default.
    """
    if not PREFLOP_OUTPUT_DIR.is_dir():
        return pd.DataFrame(
            columns=["filename", "modified", "size_kb", "questions", "_path"]
        )

    rows: list[dict[str, object]] = []
    for path in PREFLOP_OUTPUT_DIR.glob("*.csv"):
        stat = path.stat()
        try:
            # Quick row count: newlines - 1 for the header row. Cheap
            # even on multi-MB files since we just count bytes.
            with path.open("rb") as fh:
                row_count = max(0, sum(1 for _ in fh) - 1)
        except OSError:
            row_count = 0
        rows.append(
            {
                "filename": path.name,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "size_kb": round(stat.st_size / 1024, 1),
                "questions": row_count,
                "_path": str(path),
                # Sort key kept as a float so we sort properly even
                # within the same minute display.
                "_mtime": stat.st_mtime,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["filename", "modified", "size_kb", "questions", "_path"]
        )

    df = pd.DataFrame(rows).sort_values("_mtime", ascending=False)
    return df.drop(columns=["_mtime"]).reset_index(drop=True)


def render_history_page() -> None:
    """List past preflop CSVs, with multi-select bulk delete + per-file preview.

    Where everything ships to: ``test_output/preflop_batches/``. The
    Generate page writes there; this page is the catalog + cleanup tool.
    Delete is hard-delete (no trash); a confirmation step prevents
    one-click accidents.
    """
    st.title("History — past preflop CSVs")
    st.caption(
        f"All CSVs in `{PREFLOP_OUTPUT_DIR.relative_to(REPO_ROOT)}/`. "
        "Newest first. Select rows to preview, download, or delete."
    )

    df = _scan_preflop_outputs()
    if df.empty:
        st.info(
            "No CSVs here yet. Run a batch from the Generate page and "
            "it will land in this folder."
        )
        return

    st.caption(
        f"**{len(df)}** file"
        + ("s" if len(df) != 1 else "")
        + f" · total {df['size_kb'].sum():.1f} KB · "
        f"{int(df['questions'].sum()):,} questions across all files."
    )

    # selection_mode="multi-row" lets the user shift+click / cmd+click
    # rows. on_select="rerun" makes the click trigger a script rerun so
    # we can read event.selection.rows below.
    event = st.dataframe(
        df.drop(columns=["_path"]),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "filename": st.column_config.TextColumn("File"),
            "modified": st.column_config.TextColumn("Modified"),
            "size_kb": st.column_config.NumberColumn(
                "Size (KB)", format="%.1f"
            ),
            "questions": st.column_config.NumberColumn("Questions"),
        },
        key="history_table",
    )

    # The streamlit stub types this as DataframeState which is missing
    # the .selection attribute, but at runtime it's an AttrDict with
    # selection.rows = list[int] of the highlighted row indices.
    selected_indices: list[int] = list(event.selection.rows)  # type: ignore[attr-defined]
    selected_paths: list[Path] = [
        Path(str(df.iloc[i]["_path"])) for i in selected_indices
    ]
    n_selected = len(selected_paths)

    st.divider()

    # --- Action bar ---
    col_del, col_meta = st.columns([1, 3])

    # Two-step delete: first click flips a session-state flag, second
    # click (the confirm) actually unlinks. Cancel button clears the
    # flag. Stops a misclick from wiping the catalog.
    confirming = bool(st.session_state.get("history_confirming_delete", False))

    with col_del:
        if confirming and n_selected > 0:
            st.error(f"⚠️ Permanently delete **{n_selected}** file(s)?")
            conf_col1, conf_col2 = st.columns(2)
            with conf_col1:
                if st.button(
                    f"Yes, delete {n_selected}",
                    type="primary",
                    use_container_width=True,
                    key="history_delete_confirm",
                ):
                    deleted = 0
                    for p in selected_paths:
                        try:
                            p.unlink(missing_ok=True)
                            deleted += 1
                        except OSError as exc:
                            st.warning(f"Couldn't delete `{p.name}`: {exc}")
                    st.session_state["history_confirming_delete"] = False
                    # Clear the table selection so a stale "X selected"
                    # banner doesn't show after the rerun.
                    if "history_table" in st.session_state:
                        del st.session_state["history_table"]
                    st.success(f"Deleted {deleted} file(s).")
                    st.rerun()
            with conf_col2:
                if st.button(
                    "Cancel",
                    use_container_width=True,
                    key="history_delete_cancel",
                ):
                    st.session_state["history_confirming_delete"] = False
                    st.rerun()
        else:
            if st.button(
                f"🗑 Delete {n_selected} selected"
                if n_selected
                else "🗑 Delete selected",
                disabled=(n_selected == 0),
                use_container_width=True,
                key="history_delete_btn",
            ):
                st.session_state["history_confirming_delete"] = True
                st.rerun()

    with col_meta:
        if n_selected == 0:
            st.caption(
                "_Tip: click a row to select it. Shift-click or "
                "cmd-click to multi-select._"
            )
        else:
            names = ", ".join(p.name for p in selected_paths[:5])
            if n_selected > 5:
                names += f", … (+{n_selected - 5} more)"
            st.caption(f"**Selected:** {names}")

    # --- Per-file actions (only when exactly one file is selected) ---
    if n_selected == 1:
        only = selected_paths[0]
        st.divider()
        st.subheader(f"Preview: {only.name}")

        try:
            csv_bytes = only.read_bytes()
            preview_df = pd.read_csv(only, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError) as exc:
            st.error(f"Couldn't read `{only.name}`: {exc}")
            return

        st.download_button(
            label=f"📥 Download {only.name}",
            data=csv_bytes,
            file_name=only.name,
            mime="text/csv",
            use_container_width=True,
            key=f"history_dl_{only.name}_{len(csv_bytes)}",
        )

        preview_cols = [
            "No",
            "User Seat",
            "User Cards",
            "Hand Stage",
            "Context",
            "Question",
            "option 1",
            "option 2",
            "option 3",
            "option 4",
            "Correct Answer",
            "Answer Explanation",
            "Difficulty Rating",
            "action_frequencies",
        ]
        present_cols = [c for c in preview_cols if c in preview_df.columns]
        st.dataframe(
            preview_df[present_cols].head(20),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Showing first 20 of {len(preview_df):,} rows · "
            f"{len(preview_df.columns)} columns total · "
            "download for the full file."
        )
    elif n_selected > 1:
        st.caption(
            "_Select exactly one row to preview/download. With multiple "
            "rows selected, only bulk delete is available._"
        )


# --- page: Browse ----------------------------------------------------------
def render_browse_page() -> None:
    st.title("Browse the legacy demo dataset")
    st.warning(
        "⚠️ This page shows a **static hand-authored demo file** "
        "(`test_output/tier1_consolidated.csv`) — 70 **postflop** example "
        "questions, NOT pipeline output. To read + grade the questions the "
        "pipeline actually generates (preflop), use the **Review** tab."
    )

    if not TIER1_CSV.exists():
        st.error(f"No CSV at {TIER1_CSV}")
        return

    df = _read_csv_cached(str(TIER1_CSV), TIER1_CSV.stat().st_mtime)
    st.metric("Questions in dataset", len(df))

    # --- Filters ---
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        scenarios_filter = st.multiselect(
            "Scenarios",
            options=sorted(df["scenario"].unique()),
            default=[],
        )
    with col2:
        stages_filter = st.multiselect(
            "Hand stage",
            options=sorted(df["Hand Stage"].unique()),
            default=[],
        )
    with col3:
        difficulty_range = st.slider(
            "Difficulty Rating",
            int(df["Difficulty Rating"].min()),
            int(df["Difficulty Rating"].max()),
            (int(df["Difficulty Rating"].min()), int(df["Difficulty Rating"].max())),
        )

    filtered = df
    if scenarios_filter:
        filtered = filtered[filtered["scenario"].isin(scenarios_filter)]
    if stages_filter:
        filtered = filtered[filtered["Hand Stage"].isin(stages_filter)]
    filtered = filtered[
        (filtered["Difficulty Rating"] >= difficulty_range[0])
        & (filtered["Difficulty Rating"] <= difficulty_range[1])
    ]

    st.caption(f"Showing {len(filtered)} of {len(df)} questions.")

    # --- Table ---
    display_cols = [
        "No",
        "scenario",
        "Hand Stage",
        "User Cards",
        "Cards on Table",
        "Correct Answer",
        "Difficulty Rating",
        "ev_gap_bb",
        "concept_tags",
        "validation_status",
    ]
    st.dataframe(
        filtered[display_cols],
        hide_index=True,
        use_container_width=True,
        height=400,
    )

    # --- Row detail ---
    if len(filtered) > 0:
        st.subheader("Row detail")
        row_no = st.selectbox(
            "Question No",
            options=filtered["No"].tolist(),
        )
        row = filtered[filtered["No"] == row_no].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Question**")
            st.text(row["Question"])
            st.markdown("**Options**")
            for i in (1, 2, 3, 4):
                opt = row[f"option {i}"]
                if pd.notna(opt) and str(opt).strip():
                    marker = "✅ " if opt == row["Correct Answer"] else "   "
                    st.text(f"{marker}{opt}")
        with col2:
            st.markdown("**Answer Explanation**")
            st.write(row["Answer Explanation"])
            st.markdown("**Meta**")
            st.text(f"Difficulty:    {row['Difficulty Rating']}")
            st.text(f"EV gap (bb):   {row['ev_gap_bb']}")
            st.text(f"Board texture: {row['board_texture']}")
            st.text(f"Action freq:   {row['action_frequencies']}")


# --- page: Review -----------------------------------------------------------
_REVIEW_STATUS_LABEL = {
    "approved": "✅ Approved",
    "needs_review": "⚠️ Needs review",
    "rejected": "❌ Rejected",
}


def _cell(row: pd.Series, col: str) -> str:
    """A row cell as a clean string ('' for missing / NaN)."""
    if col not in row:
        return ""
    val = row[col]
    if pd.isna(val):
        return ""
    return str(val)


def _md_lines(text: str) -> str:
    """Render multi-line CSV text in Markdown with line breaks preserved
    (Markdown collapses single newlines otherwise).

    Dollar signs are escaped: st.markdown treats ``$...$`` as inline LaTeX, so
    a dollar-stakes question like "You open to $6. The Small Blind 3-bets to
    $24." rendered everything between the two amounts as math (teal
    code-style text). ``\\$`` renders as a literal dollar sign."""
    return text.replace("$", "\\$").replace("\n", "  \n")


# NOTE: the old `_autosave_review_cell` on_change callback was DELETED
# (2026-07-04). on-blur autosave + plain nav buttons is the racy pattern that
# kept losing edits; the Review pages now save via per-question st.forms whose
# submits carry the edit atomically. Do not reintroduce on_change autosave.
def _flush_review_edit(
    csv_path: Path, no: str, *, key_prefix: str = "review"
) -> None:
    """Persist the CURRENTLY-shown question's pending editor + difficulty to the
    CSV, reading straight from ``session_state``.

    WHY THIS EXISTS (and must stay): the Prev/Next/Jump buttons are rendered
    ABOVE the answer-explanation editor and short-circuit with ``st.rerun()`` the
    moment they're clicked -- so on a navigation rerun the editor is never
    re-rendered, and persistence cannot rely on the editor's ``on_change`` firing
    at exactly the right instant. This flush closes that gap: Streamlit copies
    the incoming widget values into ``session_state`` BEFORE the script body runs,
    so by the time a nav button's handler calls this, the user's just-typed edit
    is already in ``session_state`` and gets written before we leave the question.
    Race-free and independent of callback timing. ``_update_cell`` no-ops when the
    value is unchanged, so calling this on every navigation is free.

    Invariant for future refactors: ANY control that navigates away from the
    current question (changes ``review_idx`` / grades / removes a row) MUST be
    a ``form_submit_button`` of the question's form, and the submit handler
    MUST call this first. A plain ``st.button`` RACES the text_area's on-blur
    commit (type -> click could rerun without the edit ever reaching
    ``session_state``) -- that race was the recurring "my edit vanished" bug;
    the form's atomic submit removes the timing dependence entirely. Covered
    by ``tests/test_review_autosave.py``.

    ``key_prefix``: "review" (preflop page) or "postflop_review" (postflop
    page) -- the two pages namespace their widget keys differently.
    """
    expl_key = f"{key_prefix}_expl::{csv_path.name}::{no}"
    diff_key = f"{key_prefix}_diff::{csv_path.name}::{no}"
    if expl_key in st.session_state:
        review.update_explanation(csv_path, no, str(st.session_state[expl_key]))
    if diff_key in st.session_state:
        try:
            review.update_difficulty(
                csv_path, no, str(int(st.session_state[diff_key]))
            )
        except (ValueError, TypeError):
            pass


def _render_inline_spot_ranges(node, pack, *, key_prefix: str) -> None:
    """Render one decision node's range charts inline: the hero's strategy grid
    (coloured by action) plus each still-in villain's strategy at the node where
    THEY acted -- the same charts the Range viewer shows. Lets the preflop Review
    page show ranges in a dropdown instead of navigating to a separate tab
    (parity with the postflop Review). Uses the shared range_view renderer.
    """
    from pipeline.preflop.action_history import resolve_preflop_history  # noqa: PLC0415
    from pipeline.preflop.format_writer import _villain_decision_node  # noqa: PLC0415
    from pipeline.preflop.grammars.types import (  # noqa: PLC0415
        ParsedAction,
        PreflopActionType,
    )
    from pipeline.preflop_ranges import parse_range_file  # noqa: PLC0415

    _color = {
        PreflopActionType.FOLD: range_view.COLOR_FOLD,
        PreflopActionType.CALL: range_view.COLOR_CALL,
        PreflopActionType.RAISE: range_view.COLOR_RAISE,
        PreflopActionType.ALL_IN: range_view.COLOR_ALLIN,
    }
    _order = {
        PreflopActionType.FOLD: 0, PreflopActionType.CALL: 1,
        PreflopActionType.RAISE: 2, PreflopActionType.ALL_IN: 3,
    }
    _verbs = {
        PreflopActionType.FOLD: "folds", PreflopActionType.CALL: "calls",
        PreflopActionType.ALL_IN: "shoves all-in",
    }
    all_hands = [range_view.hand_at(i, j) for i in range(13) for j in range(13)]
    state = resolve_preflop_history(node.history_before, pack)

    def _verb(a: ParsedAction, size: float | None) -> str:
        if a.action_type is PreflopActionType.RAISE and size is not None:
            return f"raises to {size:g}bb"
        return _verbs.get(a.action_type, a.action_type.value.lower())

    if node.history_before:
        st.caption(
            "Action so far → "
            + " · ".join(
                f"{a.position} {_verb(a, size)}"
                for a, size in zip(node.history_before, state.sizes_bb, strict=True)
            )
        )
    legend = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;background:{c};'
        f'border-radius:2px;margin:0 5px 0 14px;vertical-align:middle;"></span>{n}'
        for n, c in (("fold", range_view.COLOR_FOLD), ("call", range_view.COLOR_CALL),
                     ("raise", range_view.COLOR_RAISE), ("all-in", range_view.COLOR_ALLIN))
    )
    st.html(f'<div style="font-size:13px;margin-bottom:4px;">{legend}</div>')

    def _mix(grid_node) -> dict[str, list[tuple[float, str, str]]]:
        segs: dict[str, list[tuple[float, str, str]]] = {h: [] for h in all_hands}
        for opt in sorted(grid_node.actions, key=lambda o: _order.get(o.action_type, 9)):
            try:
                weights = parse_range_file(opt.range_file.path)
            except (OSError, ValueError):
                weights = {}
            col = _color.get(opt.action_type, "#888888")
            for hand in all_hands:
                fr = weights.get(hand, 0.0)
                if fr > 0.0:
                    # Label -> grid_html renders a tap/hover frequency tooltip.
                    segs[hand].append((fr, col, opt.label))
        return segs

    st.markdown(f"**{node.actor} — strategy (this decision)**")
    st.html(range_view.grid_html(_mix(node)))

    last: dict[str, tuple[ParsedAction, float | None]] = {}
    for a, size in zip(node.history_before, state.sizes_bb, strict=True):
        if a.position == node.actor:
            continue
        if a.action_type is PreflopActionType.FOLD:
            last.pop(a.position, None)
        else:
            last[a.position] = (a, size)
    for pos, (action, size) in last.items():
        vnode = _villain_decision_node(node, pos, pack)
        if vnode is None:
            continue
        st.markdown(f"**{pos}** {_verb(action, size)} — their range/strategy")
        st.html(range_view.grid_html(_mix(vnode)))


def render_review_page() -> None:
    """Read each generated question in full and grade it.

    Grades save to a sidecar ``<batch>.review.json`` (see
    :mod:`admin_panel.review`) -- the generated CSV is never modified, so
    review is fully non-destructive and reversible.
    """
    st.title("Review questions")
    st.caption(
        "Read each generated question in full and grade it. Grades save to a "
        "sidecar `.review.json` next to the batch -- the CSV is never touched."
    )

    outputs = _scan_preflop_outputs()
    if outputs.empty:
        st.info("No batches yet. Generate one on the **Generate** page first.")
        return

    # --- batch picker ---
    # Select by FILENAME (a stable identity), NOT list position. The list is
    # sorted newest-modified-first, and ANY edit -- removing a question, or
    # an auto-saved explanation/difficulty -- rewrites the CSV and bumps its
    # mtime, which reorders the list. A positional selectbox would then
    # silently switch you to a different batch on the next rerun (and make a
    # just-removed question look like it "came back"). Keying on the filename
    # keeps you on the same batch across reorders.
    paths_by_name = dict(
        zip(outputs["filename"], outputs["_path"], strict=True)
    )
    label_by_name = {
        fn: f"{fn}  ({q} questions · {m})"
        for fn, q, m in zip(
            outputs["filename"], outputs["questions"], outputs["modified"],
            strict=True,
        )
    }
    picked_name = st.selectbox(
        "Batch",
        options=list(outputs["filename"]),
        format_func=lambda n: label_by_name.get(n, n),
        key="review_batch_pick",
    )
    csv_path = Path(paths_by_name[picked_name])

    try:
        df = _read_csv_cached(str(csv_path), csv_path.stat().st_mtime)
    except (OSError, ValueError) as exc:
        st.error(f"Couldn't read {csv_path.name}: {exc}")
        return
    if df.empty:
        st.warning("This batch has no rows.")
        return

    reviews = review.load_reviews(csv_path)
    summary = review.summarize(df["No"].tolist(), reviews)

    # --- progress summary ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviewed", f"{summary.reviewed}/{summary.total}")
    c2.metric("✅ Approved", summary.approved)
    c3.metric("⚠️ Needs review", summary.needs_review)
    c4.metric("❌ Rejected", summary.rejected)
    if summary.quality_pct is not None:
        st.caption(
            f"Quality so far: **{summary.quality_pct:.0f}%** approved "
            f"(of {summary.approved + summary.rejected} approve/reject grades)."
        )

    # Prompt provenance for this batch (from the .meta.json sidecar).
    _meta = review.load_batch_meta(csv_path)
    if _meta:
        _pname = str(_meta.get("prompt_name") or "(unnamed prompt)")
        _pmodel = str(_meta.get("model") or "dry-run")
        st.caption(
            f"🧪 Prompt: **{_pname}**  ·  model `{_pmodel}`  ·  "
            f"temp `{_meta.get('temperature')}`  ·  seed `{_meta.get('seed')}`"
        )
    else:
        st.caption("🧪 Prompt: _no metadata for this batch_")

    # Prominent banner for the experimental 4-LLM-call audit & auto-fix batch,
    # so a reviewer instantly knows this batch ran the special pipeline (and how
    # the auto-fix resolved the questions it flagged).
    _revise_line = review.revise_summary_line(_meta)
    if _revise_line is not None:
        st.info(
            "🔬 **Experimental batch — Audit & Auto-fix pass (up to 4 LLM calls "
            "per question).** Pipeline: 1) generate → 2) claim-check gate → "
            "3) rewrite if flagged → 4) final audit. You see only the final "
            "rewritten version below; any 4th-call flags are marked in blue.\n\n"
            + _revise_line
        )

    # Download the whole batch -- including any Answer Explanation edits,
    # which are written straight into this CSV on disk when saved below.
    st.download_button(
        "⬇  Download this batch (CSV)",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        help="The full batch CSV, with any explanation edits you've saved baked in.",
    )

    # Routed-to-human-review queue: spots a validator rejected during
    # generation (persisted in the batch meta). Surfaced here so they're no
    # longer invisible once you leave the Generate page -- with one-click
    # promote that keeps the rejected explanation. present_keys lets an
    # already-added spot read as added instead of offering the button again.
    _present_keys = {
        (_cell(r, "solver_reference"), _cell(r, "User Cards"))
        for _, r in df.iterrows()
    }
    _render_review_failures(csv_path, _present_keys)

    st.divider()

    # --- per-file navigation state ---
    nav_key = f"review_idx::{csv_path.name}"
    idx = st.session_state.get(nav_key, 0)
    idx = max(0, min(int(idx), len(df) - 1))

    nos = df["No"].tolist()
    row = df.iloc[idx]
    no = str(row["No"])
    existing = reviews.get(no, {})

    # EDIT-LOSS INVARIANT (the recurring "my explanation edit vanished" bug):
    # every control that can leave this question -- Prev/Next/Go, the grade
    # buttons, Remove -- MUST be an st.form_submit_button of THIS form, with the
    # editor + difficulty inside the same form. A form submit ships the current
    # widget values ATOMICALLY with the click (framework contract). A plain
    # st.button races the text_area's on-blur commit: type -> click Next could
    # rerun WITHOUT the edit ever reaching session_state, and the server can
    # never recover it. NEVER add a plain st.button that navigates here.
    _form = st.form(key=f"review_form::{csv_path.name}::{no}", border=False)
    with _form:
        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            _prev_clicked = st.form_submit_button(
                "◀ Prev", use_container_width=True, disabled=idx == 0
            )
        with nav2:
            _jcol, _gcol = st.columns([3, 1])
            jump = _jcol.selectbox(
                "Jump to question",
                options=range(len(df)),
                index=idx,
                format_func=lambda i: f"#{nos[i]}  ({i + 1}/{len(df)})",
                label_visibility="collapsed",
            )
            _go_clicked = _gcol.form_submit_button("Go", use_container_width=True)
        with nav3:
            _next_clicked = st.form_submit_button(
                "Next ▶", use_container_width=True, disabled=idx >= len(df) - 1
            )

        # Persistent confirmation of the last removal. After a remove the view
        # shifts to the NEXT question (it slides into this slot) and remaining
        # questions keep their original numbers (gaps are intentional), so spell
        # out exactly what happened -- otherwise it looks like the wrong question
        # was removed or things got reordered.
        _removed = st.session_state.pop("_review_last_removed", None)
        if _removed is not None and str(_removed) != no:
            st.success(f"🗑 Removed #{_removed}. Now showing #{no}.")

        # --- the question card ---
        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([1, 2, 2, 2])
            h1.markdown(f"**#{_cell(row, 'No')}**")
            h2.markdown(f"Seat&nbsp;**{_cell(row, 'User Seat')}**")
            h3.markdown(f"Hand&nbsp;**{_cell(row, 'User Cards')}**")
            h4.markdown(f"Difficulty&nbsp;**{_cell(row, 'Difficulty Rating')}**")
            if existing.get("status"):
                st.markdown(
                    "Current grade: "
                    + _REVIEW_STATUS_LABEL.get(existing["status"], existing["status"])
                )

            # Per-question meta record (join on user_cards + solver_reference);
            # used for the soft-flag text and the revise-pass lifecycle panel.
            _qmeta = (
                review.meta_question_for(
                    _meta,
                    user_cards=_cell(row, "User Cards"),
                    solver_reference=_cell(row, "solver_reference"),
                )
                if _meta
                else None
            )
            # Flag visibility: a row that SHIPPED but was flagged. The two sources
            # are shown SEPARATELY and labeled by what they are -- deterministic
            # soft validators (rule checks) vs the AI claim checker (a second LLM
            # pass). Previously both were lumped under one "soft validator" badge,
            # which made the LLM checker's eloquent notes look like a rule check.
            _vstatus = _cell(row, "validation_status")
            if _vstatus in ("flagged", "needs_review"):
                _soft = (
                    [str(w) for w in (_qmeta.get("validator_warnings") or [])]
                    if _qmeta else []
                )
                _claims = (
                    [str(w) for w in (_qmeta.get("claim_check_issues") or [])]
                    if _qmeta else []
                )
                _sanity = (
                    [str(w) for w in (_qmeta.get("sanity_check_issues") or [])]
                    if _qmeta else []
                )
                if _vstatus == "needs_review":
                    st.warning("⚠️ **Marked needs-review.**")
                if _soft:
                    st.warning(
                        "🟠 **Soft validator (deterministic rule check)** — shipped "
                        "to the CSV but flagged:\n\n"
                        + "\n".join(f"- {w}" for w in _soft)
                    )
                if _claims:
                    st.warning(
                        "🤖 **AI claim checker (a second LLM pass, not a rule "
                        "check)** — review these:\n\n"
                        + "\n".join(f"- {w}" for w in _claims)
                    )
                if _sanity:
                    st.warning(
                        "🩺 **AI sanity audit challenged the SOLVER FACTS "
                        "themselves** (it uses its own poker knowledge and "
                        "CAN be wrong — treat as hypotheses, verify against "
                        "the data panels below):\n\n"
                        + "\n".join(f"- {w}" for w in _sanity)
                    )
                # Bare fallback only when nothing else explains the flag. A revise
                # batch's discarded/unchanged rows are explained by the auto-fix
                # panel below, so don't double up with a generic badge.
                _has_revise = bool(_qmeta and _qmeta.get("revise"))
                if (_vstatus == "flagged" and not _soft and not _claims
                        and not _sanity and not _has_revise):
                    st.warning("🟠 **Flagged.**")
            # Deterministic post-batch cross-check findings (July 2026):
            # first-principles fact checks (position from seats, domination
            # direction, bands, sums). Rendered OUTSIDE the flagged-status
            # gate because these are machine-verified problems, not
            # hypotheses -- they must always be visible.
            _xchecks = (
                [str(w) for w in (_qmeta.get("cross_check_issues") or [])]
                if _qmeta else []
            )
            if _xchecks:
                st.error(
                    "🔬 **Deterministic cross-check found factual problems "
                    "in this row** (machine-verified from first principles, "
                    "not an AI opinion):\n\n"
                    + "\n".join(f"- {w}" for w in _xchecks)
                )
            # Audit & auto-fix lifecycle (revise_pass batches): how this question's
            # final shipped version was produced, plus any distinct 4th-call flags.
            _render_revise_panel(_qmeta)

            ctx = _cell(row, "Context")
            if ctx:
                st.caption(ctx)
            st.markdown("**Question**")
            st.markdown(_md_lines(_cell(row, "Question")))

            st.markdown("**Options**")
            correct = _cell(row, "Correct Answer")
            for i in (1, 2, 3, 4):
                opt = _cell(row, f"option {i}")
                if opt:
                    st.markdown(("✅ " if opt == correct else "▫️ ") + opt)

            st.markdown(
                "**Answer Explanation** _(saves on any button click: "
                "Save, Prev/Next, grade)_"
            )
            _expl_key = f"review_expl::{csv_path.name}::{no}"
            st.text_area(
                "Answer Explanation",
                value=_cell(row, "Answer Explanation"),
                key=_expl_key,
                # Tall enough to show a full in-depth explanation (250-400 words)
                # without scrolling -- the in-depth prompts run long.
                height=500,
                label_visibility="collapsed",
            )
            _save_clicked = st.form_submit_button("💾 Save edits")
            # The deterministic "Show the math" strip, right under the
            # explanation (the decision-math stats: pot odds, equity, range
            # advantage, blockers, what you're up against).
            # The panels expect a clean {str: str} dict; the raw Series carries
            # NaN floats for empty cells (the review page's read_csv doesn't
            # fillna), which would crash a `.get(col).strip()`. _cell coerces
            # NaN -> "" and everything to str.
            row_strs = {str(c): _cell(row, c) for c in row.index}
            _render_stat_panel(row_strs)
            _render_why_factors_panel(row_strs)
            _render_exploit_panel(row_strs)
            _render_claim_check_panel(row_strs)
            # Editable difficulty -- auto-saves into the CSV just like the
            # explanation (no Save button; the on_change callback writes it).
            _diff_key = f"review_diff::{csv_path.name}::{no}"
            try:
                _cur_diff = int(float(_cell(row, "Difficulty Rating") or 0))
            except ValueError:
                _cur_diff = 0
            st.number_input(
                "Difficulty Rating (saves with any button click)",
                min_value=0,
                max_value=3500,
                step=10,
                value=_cur_diff,
                key=_diff_key,
            )
            # Rendered preview (suit emojis etc.) of the saved explanation.
            with st.expander("Preview (rendered)", expanded=False):
                st.info(_md_lines(_cell(row, "Answer Explanation")))

            st.markdown(
                "**Solver frequencies:**&nbsp;"
                + _cell(row, "action_frequencies")
            )

            # Compact strategic facts.
            bits = []
            for col, label in (
                ("archetype", "archetype"),
                ("ev_gap_bb", "EV gap"),
                ("Position Matchup", "matchup"),
                ("Pot Participant", "pot"),
            ):
                val = _cell(row, col)
                if val:
                    bits.append(f"{label}: `{val}`")
            if bits:
                st.caption(" · ".join(bits))
            if _cell(row, "concept_tags"):
                st.caption(f"concept tags: {_cell(row, 'concept_tags')}")
            if _cell(row, "skills"):
                st.caption(f"skills: {_cell(row, 'skills')}")

            # Ranges: shown INLINE in a dropdown (like the postflop Review) instead
            # of navigating to a separate Range-viewer tab. Resolve the pack from
            # the batch meta's pack_id and the node from the solver_reference.
            ranges_val = _cell(row, "ranges")
            n_players = review.range_player_count(ranges_val)
            with st.expander("📊  Ranges for this spot"):
                _rng_pack = None
                _pid = (_meta or {}).get("pack_id") if isinstance(_meta, dict) else None
                if _pid:
                    _rng_pack = {p.pack_id: p for p in _cached_preflop_packs()}.get(_pid)
                _rng_node = None
                if _rng_pack is not None:
                    _nid = range_view.node_id_from_solver_reference(
                        _cell(row, "solver_reference")
                    )
                    _rng_node = _cached_ranges_index(_rng_pack.pack_id)[0].get(_nid)
                if _rng_node is not None:
                    _render_inline_spot_ranges(
                        _rng_node, _rng_pack, key_prefix=f"{csv_path.name}::{no}"
                    )
                else:
                    st.caption(
                        "Couldn't resolve this spot's node inline "
                        "(older batch, or the pack isn't on disk). Raw JSON below."
                    )
            if n_players:
                with st.expander(f"raw ranges JSON · {n_players} players"):
                    try:
                        st.code(
                            json.dumps(json.loads(ranges_val), indent=2),
                            language="json",
                        )
                    except (json.JSONDecodeError, TypeError):
                        st.code(ranges_val)

            # --- prompt + inputs inspector (read-only) ---
            with st.expander("🔍 Prompt & inputs — exactly what the LLM saw"):
                _q = None
                if _meta:
                    _q = review.meta_question_for(
                        _meta,
                        user_cards=_cell(row, "User Cards"),
                        solver_reference=_cell(row, "solver_reference"),
                    )
                if _meta is None or _q is None:
                    st.caption(
                        "No matching inputs in this batch's metadata "
                        "(older batch, or generated outside the admin panel)."
                    )
                else:
                    _raw_opts = _q.get("options", [])
                    _opts = (
                        [o for o in _raw_opts if isinstance(o, str)]
                        if isinstance(_raw_opts, list)
                        else []
                    )
                    st.markdown("**Per-question inputs** (computed, not editable)")
                    st.markdown("Options: " + ", ".join(f"`{o}`" for o in _opts))
                    st.markdown(f"Correct answer: `{_q.get('correct_answer', '')}`")
                    st.markdown("SOLVER DATA fed to the LLM:")
                    st.code(
                        json.dumps(_q.get("solver_data"), indent=2, default=str),
                        language="json",
                    )
                    st.markdown("**Full assembled prompt** (system + gold + this spot)")
                    st.code(review.assembled_prompt(_meta, _q))

        # --- grading (inside the form: grade clicks also carry the edit) ---
        st.markdown("**Grade**")
        note = st.text_area(
            "Note (optional)",
            value=existing.get("note", ""),
            key=f"review_note::{csv_path.name}::{no}",
            height=70,
        )
        g1, g2, g3 = st.columns(3)
        _approve_clicked = g1.form_submit_button(
            "✅ Approve", use_container_width=True, type="primary"
        )
        _needs_clicked = g2.form_submit_button(
            "⚠️ Needs review", use_container_width=True
        )
        _reject_clicked = g3.form_submit_button(
            "❌ Reject", use_container_width=True
        )

        # --- remove from batch: ONE click (destructive -- edits the CSV -- but
        #     recoverable by regenerating). No confirm gate by request; the
        #     button names the # it'll remove and the persistent note up top
        #     confirms it afterward, so a misclick is obvious and cheap. ---
        st.divider()
        _remove_clicked = st.form_submit_button(
            f"🗑  Remove #{no} from this batch",
            help=(
                "Deletes this question from the CSV in one click. Remaining "
                "questions keep their original numbers (gaps are fine). Can't "
                "be undone here, but you can regenerate the batch."
            ),
        )
    # --- the single post-form handler: the ONLY place navigation happens.
    # The form submit delivered the editor/difficulty/note values atomically
    # with whichever button was clicked, so saving first can never lose an
    # in-flight edit (see the EDIT-LOSS INVARIANT above).
    if any((_prev_clicked, _next_clicked, _go_clicked, _save_clicked,
            _approve_clicked, _needs_clicked, _reject_clicked, _remove_clicked)):
        _flush_review_edit(csv_path, no)
        if _remove_clicked:
            if review.remove_question(csv_path, no):
                # Stay in this slot so the NEXT question slides into view;
                # clamp against the now-shorter batch.
                st.session_state[nav_key] = max(0, min(idx, len(df) - 2))
                st.session_state["_review_last_removed"] = no
                st.rerun()
            st.warning(f"#{no} was not found in the batch.")
        else:
            new_idx = idx
            if _approve_clicked or _needs_clicked or _reject_clicked:
                status = (
                    "approved" if _approve_clicked
                    else "needs_review" if _needs_clicked
                    else "rejected"
                )
                review.save_review(csv_path, no, status, note)
                # Auto-advance to the next question after grading.
                new_idx = min(idx + 1, len(df) - 1)
            elif _prev_clicked:
                new_idx = idx - 1
            elif _next_clicked:
                new_idx = idx + 1
            elif _go_clicked:
                new_idx = int(jump)
            st.session_state[nav_key] = new_idx
            st.rerun()

    # --- cross-batch approved pool (mirrors the PLO Review page) ------------
    # Every question graded "approved" -- here or finalized on the Compare
    # page -- gathered from EVERY batch's sidecar into one downloadable set.
    # Derived live from the grades, so approving adds and un-approving drops.
    st.divider()
    st.subheader("✅ Approved questions (all batches)")
    approved_sources = review.collect_approved_sources(PREFLOP_OUTPUT_DIR)
    if not approved_sources:
        st.caption(
            "No approved questions yet. Grade questions **approved** above "
            "(or finalize on the Compare page) and they collect here across "
            "batches."
        )
    else:
        appr_rows = [r for _csv, _no, r in approved_sources]
        appr_fields = list(appr_rows[0].keys())
        st.caption(
            f"**{len(appr_rows)}** approved across all batches (deduped by "
            "spot). Updates live as you grade."
        )
        dcol, ccol = st.columns([3, 2])
        dcol.download_button(
            "⬇️  Download approved (CSV)",
            review.approved_rows_to_csv(appr_fields, appr_rows),
            file_name="nlhe_approved_all_batches.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            key="nlhe_download_approved",
        )
        # Clear-all is destructive (un-approves everything): confirm in 2 steps.
        # (NLHE grades via buttons, not state-holding radios, so the PLO
        # widget-resurrect issue can't happen here.)
        if st.session_state.get("nlhe_confirm_clear_approved"):
            if ccol.button(
                f"⚠️ Confirm: clear all {len(appr_rows)}",
                key="nlhe_clear_approved_confirm",
                use_container_width=True,
            ):
                n_cleared = review.clear_all_approved(PREFLOP_OUTPUT_DIR)
                st.session_state["nlhe_confirm_clear_approved"] = False
                st.toast(f"Cleared {n_cleared} approved question(s)")
                st.rerun()
        elif ccol.button(
            "🧹 Clear all approved",
            key="nlhe_clear_approved",
            use_container_width=True,
        ):
            st.session_state["nlhe_confirm_clear_approved"] = True
            st.rerun()

        with st.expander("🗑  Remove individual questions"):
            for src_csv, src_no, src_row in approved_sources:
                rcol, xcol = st.columns([10, 1])
                rcol.markdown(
                    f"**{src_row.get('User Cards', '')}**  ·  "
                    f"{src_row.get('Correct Answer', '')}  ·  "
                    f"`{src_row.get('archetype', '')}`  ·  diff "
                    f"{src_row.get('Difficulty Rating', '')}"
                )
                if xcol.button(
                    "🗑",
                    key=f"nlhe_appr_del_{src_csv.name}_{src_no}",
                    help="Un-approve",
                ):
                    review.remove_review(src_csv, src_no)
                    st.rerun()


def _render_action_tree_nav(
    pack: PreflopPack,
    node_by_id: dict[str, PreflopDecisionNode],
    node_by_hist: dict[tuple, PreflopDecisionNode],
) -> str | None:
    """Action-tree navigator for the Range viewer (browse mode).

    Starts at the opening node (first seat, empty history) and lets you click
    the acting player's action -- each labelled with its big-blind size -- to
    walk DOWN the tree the way the hand actually unfolds (UTG opens → UTG+1
    faces it → ...), instead of hunting a node-id in a dropdown of tens of
    thousands. Returns the current node's id; the page renders its charts.

    The child of (node, action) is found in O(1): append the chosen action to
    the node's history and look it up in ``node_by_hist``. No child = the
    action ends preflop (folded out to a winner, or a call to the flop).
    """
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        resolve_preflop_history,
    )
    from pipeline.preflop.grammars.types import (  # noqa: PLC0415
        ParsedAction,
        PreflopActionType,
    )

    root = node_by_hist.get(())
    cur_id = st.session_state.get("ranges_nav_id")
    if cur_id not in node_by_id:
        cur_id = root.node_id if root is not None else None
    if cur_id is None:
        st.warning("No opening node found in this pack.")
        return None
    node = node_by_id[cur_id]

    def _verb(at: PreflopActionType, size_bb: float | None) -> str:
        if at is PreflopActionType.RAISE and size_bb is not None:
            return f"raises to {size_bb:g}bb"
        if at is PreflopActionType.ALL_IN:
            return f"all-in {size_bb:g}bb" if size_bb else "all-in"
        return {
            PreflopActionType.FOLD: "folds",
            PreflopActionType.CALL: "calls",
        }.get(at, at.value.lower())

    state = resolve_preflop_history(node.history_before, pack)

    # --- controls: restart / back ---
    root_actor = root.actor if root is not None else "UTG"
    c_restart, c_back, _sp = st.columns([1.3, 1, 4])
    with c_restart:
        if st.button(
            f"↩︎ Restart at {root_actor}", disabled=not node.history_before,
            key="ranges_tree_restart", use_container_width=True,
        ):
            st.session_state["ranges_nav_id"] = root.node_id
            st.rerun()
    with c_back:
        if st.button(
            "↑ Back", disabled=not node.history_before,
            key="ranges_tree_back", use_container_width=True,
        ):
            parent = node_by_hist.get(node.history_before[:-1])
            if parent is not None:
                st.session_state["ranges_nav_id"] = parent.node_id
                st.rerun()

    # --- breadcrumb: the action that led here ---
    if node.history_before:
        path = "  →  ".join(
            f"{a.position} {_verb(a.action_type, s)}"
            for a, s in zip(node.history_before, state.sizes_bb, strict=True)
        )
        st.markdown(f"**Action so far:**  {path}")
    else:
        st.markdown("**Action so far:**  _start of the hand — first to act_")
    st.markdown(
        f"### ▶︎ {node.actor} to act &nbsp;·&nbsp; {node_action_context(node)}"
    )
    st.caption(f"Click what **{node.actor}** does to walk to the next spot:")

    # --- action buttons: one per option, each descends (or ends preflop) ---
    order = {
        PreflopActionType.FOLD: 0,
        PreflopActionType.CALL: 1,
        PreflopActionType.RAISE: 2,
        PreflopActionType.ALL_IN: 3,
    }
    opts = sorted(node.actions, key=lambda o: order.get(o.action_type, 9))
    cols = st.columns(len(opts) or 1)
    for col, opt in zip(cols, opts, strict=False):
        pa = ParsedAction(node.actor, opt.action_type, opt.raise_size_pct)
        child_hist = node.history_before + (pa,)
        child = node_by_hist.get(child_hist)
        try:
            size_bb = resolve_preflop_history(child_hist, pack).sizes_bb[-1]
        except Exception:  # noqa: BLE001 - size is cosmetic; never break the walk
            size_bb = None
        label = _verb(opt.action_type, size_bb)
        with col:
            if child is not None:
                if st.button(
                    label, key=f"ranges_tree_{cur_id}_{opt.label}",
                    use_container_width=True,
                ):
                    st.session_state["ranges_nav_id"] = child.node_id
                    st.rerun()
                st.caption(f"→ {child.actor} acts")
            else:
                st.button(
                    label, key=f"ranges_tree_{cur_id}_{opt.label}",
                    disabled=True, use_container_width=True,
                )
                st.caption("_ends preflop_")

    # --- advanced: jump straight to any node (the old dropdowns) ---
    with st.expander("🔎 Jump to a specific node (advanced)"):
        _seats = preflop_order(pack.table_size)
        _by_id, ids_by_actor, node_labels = _cached_ranges_index(pack.pack_id)
        actors = [s for s in _seats if s in ids_by_actor] + sorted(
            a for a in ids_by_actor if a not in _seats
        )
        ja, jb = st.columns([1, 4])
        with ja:
            jactor = st.selectbox("Position", options=actors, key="ranges_jump_actor")
        jids = ids_by_actor.get(jactor, ())
        with jb:
            jid = st.selectbox(
                f"Node — {len(jids)} where {jactor} acts",
                options=jids, format_func=node_labels.__getitem__,
                key="ranges_jump_node",
            )
        if st.button("Go to this node", key="ranges_jump_go") and jid:
            st.session_state["ranges_nav_id"] = jid
            st.rerun()

    st.divider()
    return cur_id


# --- page: Ranges -----------------------------------------------------------
def render_ranges_page() -> None:
    """Visual 13x13 range grids for any decision node.

    Reached by browsing (pick a position + node) or by the Review page's
    "View ranges for this spot" button, which stashes the node_id and
    navigates here. Renders the hero's per-action ranges (the strategy) plus
    each active villain's range.
    """
    from pipeline.preflop.format_writer import (  # noqa: PLC0415
        _villain_decision_node,
    )
    from pipeline.preflop.grammars.types import (  # noqa: PLC0415
        ParsedAction,
        PreflopActionType,
    )
    from pipeline.preflop_ranges import parse_range_file  # noqa: PLC0415

    st.title("Range viewer")
    st.caption(
        "Standard 13×13 charts: upper-right = suited, diagonal = pairs, "
        "lower-left = offsuit. Greener = higher frequency. Hero's actions "
        "show the strategy (what to do with each hand); villains show the "
        "range they're in the pot with."
    )
    # Reuse the cached, registry-safe accessors (a raw discover_packs() here
    # would re-register the pack and raise on the second page visit).
    pack = _select_preflop_pack("ranges_pack")
    if pack is None:
        st.warning("No range pack found in `ranges/`.")
        return

    if pack.grammar_name == "monker_nlhe":
        with st.popover("📋  Match this pack in GTO Wizard"):
            st.markdown(
                "To sanity-check a spot in GTO Wizard, load a 9-max cash "
                "solution. **This pack's actual rake is 10% with a 3bb cap** "
                "(micro-stakes structure) -- far heavier than any GTOW "
                "preset, so expect this pack to be NOTICEABLY tighter than "
                "whatever you load:\n\n"
                "- **Solutions:** Cash\n"
                "- **Players:** 9-max (full ring)\n"
                "- **Stack:** 100bb\n"
                "- **Opening size:** 4x (this pack opens 120% pot = 4bb "
                "UTG-HJ, 3.5bb CO, ~3bb BTN/SB)\n"
                "- **Rake:** the heaviest preset available (NL10-ish)\n\n"
                "**Calibration example from this pack:** UTG+1 folds QQ 99% "
                "against the UTG open -- and the solve's own EVs show a true "
                "indifference (call -0.001bb). The extreme tightness is the "
                "rake, not a parsing bug. See docs/nlhe9_pack_notes.md."
            )
    else:
        with st.popover("📋  Match this pack in GTO Wizard"):
            st.markdown(
                "To sanity-check a spot in GTO Wizard, load a solution with these "
                "settings. **This pack's actual rake is 4% with a 0.3bb cap** "
                "(low-stakes online style), which is why cold-call ranges run "
                "tight:\n\n"
                "- **Solutions:** Cash\n"
                "- **Type:** Classic\n"
                "- **Players:** 6-max\n"
                "- **Stack:** 100bb\n"
                "- **Preflop bet sizes:** With cold calls\n"
                "- **Opening size:** 2.5x\n"
                "- **Rake:** closest GTOW preset is **NL50** (its presets don't "
                "let you type 4% / 0.3bb directly)\n"
                "- **Postflop bet sizes:** any (doesn't affect preflop)\n\n"
                "**Confirming the match:** pick the rake where **T9s in the CO "
                "facing a HJ open folds ~90% with just a sliver of call**. Less "
                "rake (NL100+) flats more; more rake (NL10/NL25) goes pure "
                "3-bet-or-fold. The tightness is correct for a raked game, not a bug."
            )

    node_by_id, ids_by_actor, node_labels = _cached_ranges_index(pack.pack_id)
    if not node_by_id:
        st.warning("The range pack has no parseable nodes.")
        return

    # Target node: from a Review question, or browse.
    from_q = st.session_state.get("ranges_from_q")
    target_id = st.session_state.get("ranges_node_id")
    if target_id in node_by_id and from_q is not None:
        browsing = False
        st.success(f"Showing ranges for review question #{from_q}.")
        if st.button("← Browse the action tree instead", key="ranges_clear"):
            st.session_state["ranges_nav_id"] = target_id  # start the walk here
            st.session_state.pop("ranges_node_id", None)
            st.session_state.pop("ranges_from_q", None)
            st.rerun()
    else:
        # Action-tree walk: start at the opening node, click each player's
        # action (with its bb size) to descend. Replaces the old position +
        # node-id dropdowns (kept under "Jump to a specific node").
        browsing = True
        node_by_hist = _cached_ranges_by_hist(pack.pack_id)
        target_id = _render_action_tree_nav(pack, node_by_id, node_by_hist)

    node = node_by_id.get(target_id) if target_id else None
    if node is None:
        st.info("Pick a node to view its ranges.")
        return

    # The pack encodes raise sizes as a percent-of-pot token (e.g. "60%").
    # Convert to big blinds using the SAME shared pot walk the Question
    # prose uses, so the viewer's amounts match the question text (e.g.
    # "opens to 2.5bb") instead of showing the raw percent token. bb (not
    # dollars) because the ranges are stake-independent.
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        resolve_preflop_history,
    )

    _verbs = {
        PreflopActionType.FOLD: "folds",
        PreflopActionType.CALL: "calls",
        PreflopActionType.ALL_IN: "shoves all-in",
    }

    def _verb(a: ParsedAction, size_bb: float | None) -> str:
        if a.action_type is PreflopActionType.RAISE and size_bb is not None:
            return f"raises to {size_bb:g}bb"
        return _verbs.get(a.action_type, a.action_type.value.lower())

    _state = resolve_preflop_history(node.history_before, pack)

    # In browse mode the action-tree navigator already prints the breadcrumb +
    # "<actor> to act" header, so only show this for the from-a-review path.
    if not browsing:
        st.subheader(f"{node.actor} · {node_action_context(node)}")
        if node.history_before:
            st.caption(
                "Action so far → "
                + " · ".join(
                    f"{a.position} {_verb(a, size)}"
                    for a, size in zip(
                        node.history_before, _state.sizes_bb, strict=True
                    )
                )
            )

    _action_color = {
        PreflopActionType.FOLD: range_view.COLOR_FOLD,
        PreflopActionType.CALL: range_view.COLOR_CALL,
        PreflopActionType.RAISE: range_view.COLOR_RAISE,
        PreflopActionType.ALL_IN: range_view.COLOR_ALLIN,
    }
    _action_order = {
        PreflopActionType.FOLD: 0,
        PreflopActionType.CALL: 1,
        PreflopActionType.RAISE: 2,
        PreflopActionType.ALL_IN: 3,
    }
    all_hands = [range_view.hand_at(i, j) for i in range(13) for j in range(13)]

    # --- hero strategy: ONE grid, cells coloured by the action mix ---
    st.markdown(f"### {node.actor} strategy — one grid, coloured by action")
    legend = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;'
        f"background:{color};border-radius:2px;margin:0 5px 0 14px;"
        f'vertical-align:middle;"></span>{name}'
        for name, color in (
            ("fold", range_view.COLOR_FOLD),
            ("call", range_view.COLOR_CALL),
            ("raise", range_view.COLOR_RAISE),
            ("all-in", range_view.COLOR_ALLIN),
        )
    )
    st.html(f'<div style="font-size:13px;margin-bottom:4px;">{legend}</div>')

    # Build (segments, freqs) for ANY node, coloured by ACTION (fold=blue,
    # call=green, raise=red, all-in=dark). Used for the hero AND every
    # villain so the grids read the same way -- a villain who raised shows
    # red, not a flat "in range" green that looks like a call.
    def _mix_segments(
        grid_node: PreflopDecisionNode,
    ) -> tuple[dict[str, list[tuple[float, str]]], dict[str, dict[str, float]]]:
        acts: list[tuple[str, dict[str, float], str]] = []
        for opt in sorted(
            grid_node.actions, key=lambda o: _action_order.get(o.action_type, 9)
        ):
            try:
                weights = parse_range_file(opt.range_file.path)
            except (OSError, ValueError):
                weights = {}
            acts.append(
                (opt.label, weights, _action_color.get(opt.action_type, "#888888"))
            )
        segments: dict[str, list[tuple[float, str, str]]] = {}
        freqs: dict[str, dict[str, float]] = {}
        for hand in all_hands:
            segs: list[tuple[float, str, str]] = []
            fr: dict[str, float] = {}
            for label, hand_weights, color in acts:
                freq = hand_weights.get(hand, 0.0)
                fr[label] = freq
                if freq > 0.0:
                    # 3rd element = label -> grid_html renders a tap/hover
                    # tooltip on the cell with each action's frequency.
                    segs.append((freq, color, label))
            segments[hand] = segs
            freqs[hand] = fr
        return segments, freqs

    hero_segments, hero_freqs = _mix_segments(node)
    st.html(range_view.grid_html(hero_segments))

    # --- villains already in: each grid is THAT player's full strategy at
    #     the node where they acted, coloured by action (same legend as the
    #     hero). Fixes the old flat-green grid that made a villain who RAISED
    #     look like they were only calling. ---
    last: dict[str, tuple[ParsedAction, float | None]] = {}
    for a, size in zip(node.history_before, _state.sizes_bb, strict=True):
        if a.position == node.actor:
            continue
        if a.action_type is PreflopActionType.FOLD:
            last.pop(a.position, None)
            continue
        last[a.position] = (a, size)

    villain_freqs: dict[str, dict[str, dict[str, float]]] = {}
    if last:
        st.markdown("### Players already in — their ranges")
        st.caption(
            "Each grid is that player's FULL strategy when it was on them "
            "(same colour legend as above) — not just the one action that "
            "kept them in the pot."
        )
        for pos, (action, size) in last.items():
            villain_node = _villain_decision_node(node, pos, pack)
            if villain_node is None:
                st.caption(f"**{pos}**: decision node unavailable")
                continue
            v_segments, v_freqs = _mix_segments(villain_node)
            villain_freqs[pos] = v_freqs
            st.markdown(
                f"**{pos}** {_verb(action, size)} — their strategy with each hand"
            )
            st.html(range_view.grid_html(v_segments))

    # --- inspect one hand across everyone ---
    st.markdown("### Inspect a hand")
    st.caption(
        "Tip: you can also tap/click any cell in the grids above for a quick "
        "frequency readout. This picker shows one hand's breakdown across "
        "EVERY player at once."
    )
    pick = st.selectbox("Hand", options=all_hands, key="ranges_inspect_hand")
    if pick:
        fr = hero_freqs.get(pick, {})
        active = [f"{lbl} {f * 100:.1f}%" for lbl, f in fr.items() if f > 0.0]
        st.markdown(
            f"**{node.actor}** with **{pick}**: "
            + (" · ".join(active) if active else "not in range / pure fold")
        )
        for pos, v_freqs in villain_freqs.items():
            pos_fr = v_freqs.get(pick, {})
            pos_active = [
                f"{lbl} {f * 100:.1f}%" for lbl, f in pos_fr.items() if f > 0.0
            ]
            st.markdown(
                f"**{pos}** with **{pick}**: "
                + (" · ".join(pos_active) if pos_active else "not in range / pure fold")
            )


# --- compare-run persistence (shared by NLHE + PLO compare pages) -----------
def _compare_side_name(csv_path: Path) -> str:
    """A display name for one side of a past comparison, read from its
    ``.meta.json`` sidecar (prompt name + model), falling back to the stem."""
    meta = csv_path.with_suffix(".meta.json")
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return csv_path.stem
    name = str(data.get("prompt_name") or csv_path.stem)
    model = str(data.get("model") or "")
    return f"{name} · {model}" if model else name


def _compare_runs(out_dir: Path) -> list[dict[str, object]]:
    """Past A/B comparison runs in ``out_dir``, newest first.

    Each run wrote ``compare_<ts>_A.csv`` + ``compare_<ts>_B.csv`` (plus
    their meta/verdict sidecars), so prior comparisons survive tab switches
    AND panel restarts on disk -- this powers the "load a past comparison"
    picker. Returns the same shape the live run stashes in session_state
    (``a_csv``/``b_csv``/``a_name``/``b_name``) plus a ``ts`` for labeling.
    """
    if not out_dir.is_dir():
        return []
    runs: list[dict[str, object]] = []
    for a_csv in out_dir.glob("compare_*_A.csv"):
        b_csv = a_csv.with_name(a_csv.name.replace("_A.csv", "_B.csv"))
        if not b_csv.is_file():
            continue
        ts = a_csv.stem[len("compare_"):-len("_A")]
        runs.append({
            "a_csv": str(a_csv),
            "b_csv": str(b_csv),
            "a_name": _compare_side_name(a_csv),
            "b_name": _compare_side_name(b_csv),
            "ts": ts,
            "mtime": a_csv.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)  # type: ignore[arg-type,return-value]
    return runs


def _compare_run_label(run: dict[str, object]) -> str:
    """Human label for the past-comparison picker."""
    ts = str(run["ts"])
    try:
        when = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        when = ts
    return f"{when}  —  A: {run['a_name']}  vs  B: {run['b_name']}"


def _render_past_comparisons(out_dir: Path, state_key: str) -> dict[str, object] | None:
    """Render the past-comparison picker + auto-restore the latest run.

    Returns the comparison to display (the live/just-run one, the
    auto-restored latest, or a user-picked older one), or ``None`` when no
    comparison exists yet. ``state_key`` is the session_state slot the page
    uses for its current result (``cmp_result`` / ``plo_cmp_result``).
    """
    runs = _compare_runs(out_dir)
    result = st.session_state.get(state_key)
    # Auto-restore the newest run when session_state is empty (e.g. after a
    # tab switch that reset state, or a panel restart) so a comparison is
    # never lost just by leaving the page.
    if result is None and runs:
        result = {k: runs[0][k] for k in ("a_csv", "b_csv", "a_name", "b_name")}
        st.session_state[state_key] = result
    if runs:
        with st.expander(f"📂 Past comparisons ({len(runs)})"):
            st.caption(
                "Every comparison is saved to disk, so it stays here after you "
                "switch tabs or restart. Pick one to reopen its side-by-side "
                "view, verdicts, and finalize buttons."
            )
            cur = st.session_state.get(state_key) or {}
            ts_options = [str(r["ts"]) for r in runs]
            cur_ts = next(
                (str(r["ts"]) for r in runs if r["a_csv"] == cur.get("a_csv")),
                ts_options[0],
            )
            pick = st.selectbox(
                "Load a past comparison",
                options=ts_options,
                index=ts_options.index(cur_ts),
                format_func=lambda t: next(
                    _compare_run_label(r) for r in runs if str(r["ts"]) == t
                ),
                key=f"{state_key}_load_pick",
            )
            if pick != cur_ts and st.button(
                "Load this comparison", key=f"{state_key}_load_btn"
            ):
                chosen = next(r for r in runs if str(r["ts"]) == pick)
                st.session_state[state_key] = {
                    k: chosen[k] for k in ("a_csv", "b_csv", "a_name", "b_name")
                }
                st.rerun()
    return st.session_state.get(state_key)


def _finish_comparison(
    res_a: object,
    res_b: object,
    a_name: str,
    b_name: str,
    state_key: str,
) -> str | None:
    """Persist a finished comparison, or return an error message.

    A real-API side can finish with ZERO rows -- every spot failed
    generation or validation -- in which case the batch writes no CSV.
    Persisting the pair anyway left a half-written run that pointed at a
    missing file (the comparison "didn't save"), and the past-comparisons
    picker silently skipped it. So: only stash the result when BOTH sides
    actually wrote rows; otherwise return a message naming the failed side
    and why. A side that DID write is left in place -- those are real
    (paid) questions, reviewable as a normal batch -- and the picker hides
    the incomplete pair on its own (it requires both _A and _B). Returns
    ``None`` on success. Reads the two result objects via getattr so it
    works for both the NLHE and PLO batch-result types.
    """
    def _wrote(res: object) -> bool:
        return (
            getattr(res, "output_path", None) is not None
            and getattr(res, "questions_written", 0) > 0
        )

    def _attempted(res: object) -> int:
        return int(
            getattr(res, "questions_attempted", None)
            or getattr(res, "questions_requested", None)
            or getattr(res, "questions_written", 0)
        )

    def _reasons(res: object) -> str:
        raw = getattr(res, "failures", None)
        if raw is None:
            raw = getattr(res, "explanation_failure_reasons", ())
        return "; ".join(str(f) for f in list(raw)[:4])

    a_ok, b_ok = _wrote(res_a), _wrote(res_b)
    if a_ok and b_ok:
        st.session_state[state_key] = {
            "a_csv": str(res_a.output_path),
            "b_csv": str(res_b.output_path),
            "a_name": a_name,
            "b_name": b_name,
        }
        return None
    bits: list[str] = []
    for ok, name, res in ((a_ok, a_name, res_a), (b_ok, b_name, res_b)):
        if not ok:
            why = _reasons(res) or "no spots matched the filters"
            bits.append(f"**{name}** wrote 0 of {_attempted(res)} ({why})")
    return "Comparison not saved — " + "  •  ".join(bits)


def _render_equity_bar(row: dict[str, str]) -> None:
    """A small visual of hero's hand equity and range equity vs villain's
    range, with the break-even-to-call threshold marked. Reads the
    deterministic decision-math columns (hero_equity / range_equity /
    pot_odds); no-ops when they're blank (open/first-in spots, PLO, postflop,
    or batches generated before those columns existed). Shows the numbers
    only -- never a "you have the price" verdict, since implied odds can make
    a sub-threshold call correct (the answer explanation owns the decision).
    """
    # stat_notes is the source of truth for the panel; the NLHE preflop CSV
    # dropped the flat hero_equity / pot_odds columns (June 2026) but their
    # values still live inside stat_notes, so read the column first (PLO still
    # ships the flat columns) and fall back to stat_notes when it's blank.
    from pipeline.preflop.stat_notes import parse_stat_notes  # noqa: PLC0415

    parsed_notes = parse_stat_notes(row.get("stat_notes", ""))
    sn_values = {sn.get("key"): (sn.get("value", "") or "") for sn in parsed_notes}

    def _pct(key: str) -> float | None:
        raw = (row.get(key) or "").strip().rstrip("%")
        if not raw:
            raw = sn_values.get(key, "").strip().rstrip("%")
        try:
            v = float(raw)
        except ValueError:
            return None
        return v if 0.0 <= v <= 100.0 else None

    # range_equity was dropped entirely from the NLHE CSV and never had a
    # stat_notes row, so it resolves to None there (the range bar is omitted);
    # PLO still carries the column.
    hand_eq, range_eq, be = _pct("hero_equity"), _pct("range_equity"), _pct("pot_odds")

    # Multi-way all-in: the hero_equity number is the HEADS-UP value (vs one
    # opponent), which overstates a multi-way pot. stat_notes carry the FIELD
    # equity (beat everyone) under a "(multi-way)" label -- prefer it here so
    # the bar isn't misleading, and relabel.
    multiway_eq: float | None = None
    for sn in parsed_notes:
        if sn.get("key") == "hero_equity" and "multi-way" in sn.get("label", "").lower():
            try:
                multiway_eq = float((sn.get("value", "") or "").rstrip("%"))
            except ValueError:
                multiway_eq = None
            break

    if hand_eq is None and range_eq is None and multiway_eq is None:
        return

    # Name the seat we're measured against (the most-recent raiser) so a
    # multiway pot makes clear WHICH opponent. Position Matchup is
    # "HERO_vs_VILLAIN" (e.g. "BTN_vs_UTG+1"); fall back to "villain's".
    matchup = (row.get("Position Matchup") or "").strip()
    villain = f"{matchup.split('_vs_')[1]}'s" if "_vs_" in matchup else "villain's"

    def _bar(label: str, val: float | None) -> str:
        if val is None:
            return ""
        marker = (
            f'<div style="position:absolute;top:-3px;bottom:-3px;left:{be:.1f}%;'
            'width:2px;background:#D62728;"></div>'
            if be is not None
            else ""
        )
        return (
            f'<div style="font-size:0.8em;color:#444;margin:6px 0 1px;">{label}</div>'
            '<div style="position:relative;background:#E9E9E9;border-radius:3px;'
            'height:20px;width:100%;">'
            f'<div style="background:#4C78A8;height:100%;width:{val:.1f}%;'
            'border-radius:3px;"></div>'
            f'<div style="position:absolute;top:0;left:6px;line-height:20px;'
            f'font-size:0.8em;color:#fff;font-weight:600;">{val:.0f}%</div>'
            f"{marker}</div>"
        )

    if multiway_eq is not None:
        # Multi-way all-in: show the field equity (beat everyone) and skip the
        # heads-up range bar -- it's apples-to-oranges next to a field number.
        html = _bar("Your equity vs the WHOLE field (you must beat everyone)", multiway_eq)
        if be is not None:
            html += (
                f'<div style="font-size:0.75em;color:#D62728;margin-top:3px;">'
                f"Red line = {be:.0f}% break-even to call</div>"
            )
        html += (
            '<div style="font-size:0.75em;color:#444;margin-top:4px;">'
            "Multi-way all-in: this is your equity against everyone still in. "
            "Heads-up against any one of them you'd have more, but here you "
            "have to beat them all on the same board.</div>"
        )
        st.markdown(html, unsafe_allow_html=True)
        return

    html = _bar(f"Your hand vs {villain} range", hand_eq)
    html += _bar(f"Your whole range vs {villain} range", range_eq)
    if be is not None:
        html += (
            f'<div style="font-size:0.75em;color:#D62728;margin-top:3px;">'
            f"Red line = {be:.0f}% break-even to call</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def _user_cards_to_class(cards: str) -> str:
    """'A-spades, 9-spades' -> 'A9s'; 'A-spades, A-hearts' -> 'AA'; '' on failure."""
    parts = [p.strip() for p in (cards or "").split(",") if p.strip()]
    if len(parts) != 2:
        return ""
    suit_i = {"spades": "s", "hearts": "h", "diamonds": "d", "clubs": "c"}
    order = "23456789TJQKA"
    pc: list[tuple[str, str]] = []
    for p in parts:
        bits = p.split("-")
        if len(bits) != 2:
            return ""
        rank, suit = bits[0].strip().upper(), suit_i.get(bits[1].strip().lower())
        if not suit or rank not in order:
            return ""
        pc.append((rank, suit))
    (r1, s1), (r2, s2) = pc
    if r1 == r2:
        return r1 + r2
    if order.index(r1) < order.index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def _parse_top_villain_classes(cell: str) -> list[str]:
    """'AA, KK, AKs (~70% of 4.2%)' -> ['AA','KK','AKs']."""
    head = (cell or "").split("(")[0]
    return [c.strip() for c in head.split(",") if c.strip()]


def _parse_blocker_classes(cell: str) -> list[str]:
    """'AKo:6, AA:3' -> ['AKo','AA']."""
    out: list[str] = []
    for part in (cell or "").split(","):
        part = part.strip()
        if ":" in part:
            out.append(part.split(":")[0].strip())
    return out


def _facing_raise_level_from_row(row: dict[str, str]) -> int | None:
    """The number of raises hero faces (1 = single open, 2 = a 3-bet, 3 = a
    4-bet, ...), reconstructed from a CSV row so the exploit engine names hero's
    re-raise correctly. The archetype is authoritative for value/bluff raise
    spots; otherwise (folds) the deterministic Question narrative names the
    villain's raise. 0 for a first-in open; None when nothing indicates a raise.
    """
    by_arch = {
        "open_for_value": 0,
        "3bet_for_value": 1, "3bet_as_bluff": 1,
        "squeeze_for_value": 1, "squeeze_as_bluff": 1,
        "4bet_for_value": 2, "4bet_as_bluff": 2,
        "5bet_for_value": 3, "5bet_as_bluff": 3,
    }
    arch = (row.get("archetype") or "").strip()
    if arch in by_arch:
        return by_arch[arch]
    q = (row.get("Question") or "").lower()
    for token, level in (("5-bet", 4), ("4-bet", 3), ("3-bet", 2)):
        if token in q:
            return level
    if "raise" in q or "opens" in q or "open " in q:
        return 1  # a single open in the narrative
    return None


def recompute_exploit_notes_for_row(row: dict[str, str]) -> str:
    """Recompute the ``exploit_notes`` JSON for one CSV row from the CURRENT
    engine. The single source of truth used by BOTH the Review panel (which
    always recomputes, so an exploit-engine fix shows immediately for every
    batch) and ``scripts/rerender_exploit_notes.py`` (which re-bakes the column
    in existing CSVs). Returns "" for a blank / unclassified archetype.

    INVARIANT: display and re-bake must go through THIS function so a fix to
    `pipeline.preflop.exploit` can never be silently overridden by a stale baked
    column again. Do not read `row["exploit_notes"]` for display.
    """
    from pipeline.preflop.domination import dominating_map  # noqa: PLC0415
    from pipeline.preflop.exploit import (  # noqa: PLC0415
        exploit_adjustments,
        exploit_notes_to_json,
    )

    archetype = (row.get("archetype") or "").strip()
    if not archetype or archetype == "unclassified":
        return ""
    raw_eq = (row.get("hero_equity") or "").strip().rstrip("%")
    try:
        hero_equity: float | None = float(raw_eq) / 100.0
    except ValueError:
        hero_equity = None
    dominated_by = you_dominate = None
    hand_class = _user_cards_to_class(row.get("User Cards", ""))
    villain_classes = _parse_top_villain_classes(row.get("top_villain_combos", ""))
    if hand_class and villain_classes:
        dm = dominating_map(hand_class, villain_classes)
        dominated_by = dm["dominated_by"] or None
        you_dominate = dm["you_dominate"] or None
    blockers = _parse_blocker_classes(row.get("blocker_combos", "")) or None
    is_pp = bool(hand_class and len(hand_class) == 2 and hand_class[0] == hand_class[1])
    notes = exploit_adjustments(
        archetype,
        hero_equity=hero_equity,
        dominated_by=dominated_by,
        you_dominate=you_dominate,
        blockers=blockers,
        is_pocket_pair=is_pp,
        facing_raise_level=_facing_raise_level_from_row(row),
    )
    return exploit_notes_to_json(notes)


def _render_exploit_panel(row: dict[str, str]) -> None:
    """A "🎯 Exploit adjustments" strip: how to deviate from GTO vs a nit /
    station / maniac, computed deterministically from this hand's archetype
    (its strategic role), refined with its equity, domination, and blockers.
    Directional guidance, not solver-exact frequencies. No-ops on rows with no
    archetype (an open spot still has one; only blank / unclassified skip).
    """
    from pipeline.preflop.exploit import parse_exploit_notes  # noqa: PLC0415

    # ALWAYS recompute from the current engine -- NEVER read the baked
    # exploit_notes column here. Old batches baked notes from an earlier engine;
    # trusting that column silently overrode later fixes (the "my exploit fix
    # doesn't show up" bug). recompute_exploit_notes_for_row is the shared source
    # of truth (same function the re-bake script uses).
    notes: list[dict[str, str]] = parse_exploit_notes(
        recompute_exploit_notes_for_row(row)
    )
    if not notes:
        return
    with st.expander("🎯 Exploit adjustments (vs player types)"):
        st.caption(
            "Directional deviations from GTO for this hand, computed from its "
            "role. Not solver-exact frequencies."
        )
        for n in notes:
            st.markdown(f"**{n.get('label', '')}** · {n.get('headline', '')}")
            st.caption(n.get("detail", ""))


def _render_why_factors_panel(row: dict[str, str]) -> None:
    """A "🧭 Why this action" strip: the deterministic factors pushing toward
    each action, computed from this row's tags / archetype / equity /
    position (pipeline.preflop.why_factors). Grouped for/against the answer,
    like the app's "Why this action?" view. Lean is deterministic; the
    strengths are coarse, not solver-exact. No-ops when the row has neither
    an archetype nor concept tags to reason from.
    """
    from pipeline.preflop.why_factors import why_factors_from_row  # noqa: PLC0415

    has_signal = (row.get("archetype") or "").strip() or (
        row.get("concept_tags") or ""
    ).strip()
    if not has_signal:
        return
    breakdown = why_factors_from_row(row)
    if not breakdown.factors:
        return
    strength_word = {1: "weak", 2: "moderate", 3: "strong"}
    for_answer = [f for f in breakdown.factors if f.favors == breakdown.answer]
    against = [f for f in breakdown.factors if f.favors != breakdown.answer]
    with st.expander("🧭 Why this action — factor breakdown"):
        st.caption(
            "Deterministic factors and which action each leans toward, from "
            "the solver's facts. The lean is deterministic; the strengths are "
            "coarse buckets, not solver-exact magnitudes."
        )
        if for_answer:
            st.markdown(f"**Arguing for {breakdown.answer}**")
            for f in for_answer:
                st.markdown(
                    f"- **{f.label}** ({strength_word[f.strength]}) — {f.detail}"
                )
        if against:
            st.markdown(f"**Arguing for {breakdown.alternative}**")
            for f in against:
                st.markdown(
                    f"- **{f.label}** ({strength_word[f.strength]}) — {f.detail}"
                )
        st.info(breakdown.summary)


def _claim_checker_prompt_path() -> Path:
    """File backing the editable claim-checker system prompt (gitignored, like
    the other prompt files under admin_panel/prompts/)."""
    return Path(__file__).resolve().parent / "prompts" / "claim_checker_system.txt"


def _load_claim_checker_prompt() -> str:
    """The saved editable claim-checker prompt, or the built-in default."""
    from pipeline.preflop.claim_checker import CHECKER_SYSTEM_PROMPT  # noqa: PLC0415

    path = _claim_checker_prompt_path()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return CHECKER_SYSTEM_PROMPT


def _save_claim_checker_prompt(text: str) -> None:
    path = _claim_checker_prompt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _postflop_claim_checker_prompt_path() -> Path:
    """File backing the editable POSTFLOP claim-checker system prompt (its own
    file -- the postflop checker targets postflop errors, distinct from the
    preflop one). Gitignored like the other prompts."""
    return (
        Path(__file__).resolve().parent / "prompts" / "postflop_claim_checker_system.txt"
    )


def _load_postflop_claim_checker_prompt() -> str:
    """The saved editable postflop claim-checker prompt, or the built-in default."""
    from pipeline.postflop.claim_checker import (  # noqa: PLC0415
        POSTFLOP_CHECKER_SYSTEM_PROMPT,
    )

    path = _postflop_claim_checker_prompt_path()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return POSTFLOP_CHECKER_SYSTEM_PROMPT


def _save_postflop_claim_checker_prompt(text: str) -> None:
    path = _postflop_claim_checker_prompt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _preflop_entry_prompt_path() -> Path:
    """Override file for the preflop-ENTRY prompt (the play-through preflop leg /
    the standalone preflop-from-postflop questions). Distinct from the postflop
    system prompt."""
    return Path(__file__).resolve().parent / "prompts" / "preflop_entry_system.txt"


def _load_preflop_entry_prompt() -> str:
    """The active preflop-entry prompt (admin override else the built-in)."""
    from pipeline.postflop.preflop_entry import (  # noqa: PLC0415
        load_preflop_entry_system_prompt,
    )

    return load_preflop_entry_system_prompt()


def _save_preflop_entry_prompt(text: str) -> None:
    path = _preflop_entry_prompt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- postflop prompt library (mirror of the preflop library) ----------------
def _postflop_prompt_library():
    """The postflop PromptLibrary instance (its own dir; same generic class)."""
    from admin_panel.prompt_library import PromptLibrary  # noqa: PLC0415

    return PromptLibrary(base_dir=POSTFLOP_LIBRARY_DIR)


def _sync_postflop_active_to_override(lib) -> None:
    """Mirror the active library entry into POSTFLOP_PROMPT_OVERRIDE_PATH so
    load_postflop_system_prompt() (the Generate child, the CLI, the Compare
    prefill) reads it -- the same legacy-sync trick the preflop library uses."""
    text = lib.active_text()
    if text is not None:
        POSTFLOP_PROMPT_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        POSTFLOP_PROMPT_OVERRIDE_PATH.write_text(text, encoding="utf-8")


def _ensure_postflop_library_seeded(lib) -> None:
    """Seed an empty postflop library with two starting entries: the built-in
    code default, and (if the legacy single-file override exists) the prompt the
    user has been generating with -- imported and made active so generation is
    unchanged. Idempotent."""
    from pipeline.postflop.explanation_generator import (  # noqa: PLC0415
        POSTFLOP_SYSTEM_PROMPT,
    )

    if lib.list():
        return
    lib.create("Built-in default", POSTFLOP_SYSTEM_PROMPT)
    if POSTFLOP_PROMPT_OVERRIDE_PATH.is_file():
        try:
            text = POSTFLOP_PROMPT_OVERRIDE_PATH.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if text.strip() and text.strip() != POSTFLOP_SYSTEM_PROMPT.strip():
            imported = lib.create("Factor-list (imported)", text)
            lib.set_active(imported.slug)
    _sync_postflop_active_to_override(lib)


def _render_claim_check_panel(row: dict[str, str]) -> None:
    """Layer-7 claim-checker output under the explanation. The ``claim_check``
    cell is "" when the checker did not run (no panel), "[]" when it ran and
    found nothing, or a JSON list of {claim, problem} it flagged.

    Both verdicts render as a dropdown so the checker's output is visible under
    every question it ran on -- a flagged expander opens by default (it needs
    attention); a clean one stays collapsed (reassurance, one click away).
    """
    from pipeline.preflop.claim_checker import parse_claim_check  # noqa: PLC0415

    cell = (row.get("claim_check") or "").strip()
    if not cell:
        return
    issues = parse_claim_check(cell)
    if not issues:
        with st.expander("✅ Claim check: no issues found"):
            st.caption(
                "The Layer-7 claim checker audited this explanation against the "
                "SOLVER DATA block and flagged nothing."
            )
        return
    with st.expander(f"⚠️ Claim check flagged {len(issues)} claim(s)", expanded=True):
        for it in issues:
            st.markdown(f"**{it.get('claim', '')}**")
            st.caption(it.get("problem", ""))


def _render_revise_panel(qmeta: dict[str, object] | None) -> None:
    """Per-question auto-fix lifecycle on the Review page.

    Drives off the ``revise`` record the batch writes when revise_pass is on
    (status clean/fixed/discarded/unchanged + the 4th-call final-audit flags);
    falls back to the legacy ``revision`` key for older batches. The shipped
    explanation shown in the card is ALWAYS the final version -- this panel only
    explains how it got there, and surfaces any remaining 4th-call flags in a
    DISTINCT blue box so they never read as a regular (orange) claim check.
    """
    rev = (qmeta or {}).get("revise") if qmeta else None
    if rev is None and qmeta and isinstance(qmeta.get("revision"), dict):
        old = qmeta["revision"]  # legacy: a kept fix only
        rev = {
            "status": "fixed",
            "gate_issues": old.get("issues_fixed") or [],
            "original_explanation": old.get("original_explanation", ""),
            "revised_explanation": old.get("revised_explanation", ""),
        }
    if not isinstance(rev, dict):
        return
    # Every state leads with REWRITTEN vs ORIGINAL -- the one thing a reviewer
    # needs to know about the text below. Only "fixed" means the LLM changed it.
    status = rev.get("status")
    if status == "fixed":
        st.success(
            "✍️ **REWRITTEN by the auto-fix.** The text below is the LLM's "
            "**rewritten final version** (re-validated before shipping); the "
            "original first draft is in the expander."
        )
        with st.expander("See the original first draft (not shipped)"):
            st.caption("The claim-check gate (2nd LLM call) flagged the draft for:")
            for iss in rev.get("gate_issues") or []:
                st.caption(f"- {iss}")
            st.markdown("**Original draft**")
            st.info(_md_lines(str(rev.get("original_explanation", ""))))
    elif status == "clean":
        st.success(
            "🔎 **Layer-7 audit ran — came back CLEAN.** The auto-fix gate "
            "checked this explanation (best-of-2 claim-check passes) and flagged "
            "nothing, so the original text shipped unchanged. (If you still see a "
            "confusing sentence here, the checker missed it — worth a manual grade.)"
        )
    elif status == "discarded":
        st.warning(
            "🛠️⚠️ **ORIGINAL text (NOT changed).** The auto-fix tried to rewrite "
            "this, but the rewrite broke a hard rule and was discarded, so the "
            "original shipped (flagged for review). Why the rewrite was "
            f"rejected:\n\n> {rev.get('rejected_reason') or 'unknown'}"
        )
        _revise_unresolved(rev)
    elif status == "unchanged":
        st.warning(
            "🛠️ **ORIGINAL text (NOT changed).** The auto-fix reviewed the flags "
            "but made no edit, so the original shipped (flagged for review)."
        )
        _revise_unresolved(rev)

    # The 4th LLM call: a claim check on the REWRITTEN version. Shown in a
    # DISTINCT blue box (the regular claim checker is orange) so a reviewer can
    # never confuse "the 4th-call audit still flagged this" with a normal run.
    final_issues = rev.get("final_audit_issues") or []
    if final_issues:
        st.info(
            "4️⃣ **Final audit — the 4th LLM call (claim check on the rewritten "
            "version)** still flagged these. This is the experimental pass, not "
            "the regular checker:\n\n"
            + "\n".join(f"- {i}" for i in final_issues)
        )


def _revise_unresolved(rev: dict[str, object]) -> None:
    """List the gate issues an auto-fix did NOT resolve (discarded/unchanged)."""
    issues = rev.get("gate_issues") or []
    if issues:
        st.caption("Gate (2nd LLM call) issues that remain unresolved:")
        for iss in issues:
            st.caption(f"- {iss}")


def _render_action_ev_bars(row: dict[str, str]) -> None:
    """Per-action solver EV as a small zero-centered bar chart.

    Reads the ``action_ev_bb`` column (e.g. ``"Fold: +0.00, Call: -0.02"``),
    one bar per action measured from the break-even line (0 bb). Makes the
    EV gap concrete: when the bars are nearly equal the spot is a
    near-indifference mix (which is WHY the solver mixes its action); when
    they're far apart the gap is what the wrong action costs. The ✅ marks
    the highest-EV action. Hidden when the column is blank -- the PioViewer
    Ryan pack stores no solver EVs, and older batches predate the column.
    """
    cell = (row.get("action_ev_bb") or "").strip()
    if not cell:
        return
    actions: list[tuple[str, float]] = []
    for part in cell.split(","):
        label, sep, num = part.strip().partition(":")
        if not sep:
            continue
        try:
            actions.append((label.strip(), float(num.strip())))
        except ValueError:
            continue
    if not actions:
        return
    best = max(bb for _, bb in actions)
    worst = min(bb for _, bb in actions)
    # Symmetric scale floored at +-1bb so a genuinely tiny gap reads as tiny
    # rather than getting stretched to fill the bar.
    span = max(1.0, max(abs(bb) for _, bb in actions)) * 1.15
    html = [
        '<div style="font-size:0.85em;color:#222;font-weight:600;'
        'margin-top:10px;">EV of each action (solver)</div>'
    ]
    for label, bb in actions:
        half = min(50.0, abs(bb) / span * 50.0)
        if bb >= 0:
            seg, color = f"left:50%;width:{half:.1f}%;", "#2CA02C"
        else:
            seg, color = f"left:{50.0 - half:.1f}%;width:{half:.1f}%;", "#D62728"
        mark = " ✅" if abs(bb - best) < 1e-9 else ""
        html.append(
            f'<div style="font-size:0.8em;color:#444;margin:6px 0 1px;">'
            f"{label}{mark} · {bb:+.2f} bb</div>"
            '<div style="position:relative;background:#E9E9E9;border-radius:3px;'
            'height:16px;width:100%;">'
            '<div style="position:absolute;top:-2px;bottom:-2px;left:50%;'
            'width:1px;background:#888;"></div>'
            f'<div style="position:absolute;top:0;{seg}height:100%;'
            f'background:{color};border-radius:2px;"></div></div>'
        )
    gap = best - worst
    # "why the solver mixes" only fits a MIXED spot; a pure 100% action
    # with a tiny gap (a bottom-of-range open) needs different wording.
    is_pure = "100%" in (row.get("action_frequencies") or "")
    if gap < 0.10 and is_pure:  # noqa: PLR2004
        tail = (
            "The bars are nearly equal — the wrong action costs little "
            "here, but the solver still always takes the marked one."
        )
    elif gap < 0.10:  # noqa: PLR2004
        tail = (
            "The bars are nearly equal, so the actions are about the same "
            "EV — that's why the solver mixes."
        )
    else:
        tail = (
            f"The gap between best and worst is about {gap:.2f} bb — what "
            "picking the wrong action costs."
        )
    html.append(
        '<div style="font-size:0.75em;color:#888;margin-top:3px;">'
        f"Center line = break-even (0 bb). {tail}</div>"
    )
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_stat_panel(row: dict[str, str]) -> None:
    """A collapsible "Show the math" strip under an answer explanation.

    One compact row per deterministic decision-math stat (pot odds, your
    equity, range advantage, blockers, what you're up against), read from
    the row's ``stat_notes`` cell. The phrases are written in Python by
    :mod:`pipeline.preflop.stat_notes` -- never the LLM -- so they can't
    misframe the numbers. Open/first-in spots have no villain to price
    against so their ``stat_notes`` is empty BY DESIGN -- but they still
    carry per-action solver EVs, and the EV chart is real math, so the
    panel renders whenever EITHER exists (July 2026; the old
    stat_notes-only gate silently hid the panel on every open spot).
    No-ops only when there is nothing at all to show: PLO, postflop, or
    batches generated before the columns existed.
    """
    from pipeline.preflop.stat_notes import parse_stat_notes  # noqa: PLC0415

    notes = parse_stat_notes(row.get("stat_notes", ""))
    has_action_evs = bool((row.get("action_ev_bb") or "").strip())
    if not notes and not has_action_evs:
        return
    with st.expander("📊 Show the math"):
        _render_equity_bar(row)
        _render_action_ev_bars(row)
        for note in notes:
            st.markdown(f"**{note.get('label', '')}** · {note.get('value', '')}")
            st.caption(note.get("note", ""))
        # Plain-English methodology, shown only on multi-way all-ins (where a
        # multi-way equity note is present) so it explains the number in context
        # without cluttering ordinary heads-up spots.
        if any("multi-way" in n.get("label", "").lower() for n in notes):
            st.caption(
                "ℹ️ **Heads-up vs multi-way equity:** heads-up equity is your "
                "chance to win against ONE opponent's range. Multi-way equity "
                "is your chance against EVERYONE all-in at once — always lower, "
                "because each extra player is another hand that can beat you. "
                "On a multi-way all-in we use the multi-way number, since you "
                "have to beat them all to win the pot."
            )


# --- page: Compare (head-to-head prompt A/B) --------------------------------
def render_compare_page() -> None:
    """Run two prompts on the SAME spots and judge them side by side.

    Reuses :func:`generate_preflop_batch` twice with the same seed (so both
    prompts see identical spots) and temperature 0, then joins the two CSVs
    spot-by-spot and lets you pick a winner per spot with a running tally.
    """
    import os  # noqa: PLC0415

    from admin_panel.prompt_library import PromptLibrary  # noqa: PLC0415
    from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
        build_preflop_system_prompt,
    )

    st.title("Compare prompts (A/B)")
    st.caption(
        "Run two prompts on the SAME spots (same hands, temperature 0) and "
        "judge them side by side — so any difference is the prompt, not luck."
    )

    lib = PromptLibrary()
    lib.ensure_seeded(
        build_preflop_system_prompt, legacy_override=PREFLOP_PROMPT_OVERRIDE_PATH
    )
    entries = lib.list()
    if len(entries) < 2:
        st.warning("Create at least two prompts on the Prompt library page first.")
        return
    slugs = [e.slug for e in entries]
    names = {e.slug: e.name for e in entries}

    c1, c2 = st.columns(2)
    with c1:
        a_slug = st.selectbox(
            "Prompt A", slugs, index=0, format_func=lambda s: names[s], key="cmp_a"
        )
    with c2:
        b_slug = st.selectbox(
            "Prompt B", slugs, index=1, format_func=lambda s: names[s], key="cmp_b"
        )

    # Spot filters — applied IDENTICALLY to both prompts so the A/B stays fair.
    # Both sides run on the SAME pack (a 6-max-vs-9-max comparison would
    # confound the prompt A/B with a tree change).
    # Default the pack selector to the 9-max pack (the one in active use),
    # seeded before the widget exists so it takes on first visit.
    if "cmp_pack" not in st.session_state and any(
        p.pack_id == NLHE9_PACK_ID for p in _cached_preflop_packs()
    ):
        st.session_state["cmp_pack"] = NLHE9_PACK_ID
    cmp_pack = _select_preflop_pack("cmp_pack")
    if cmp_pack is None:
        st.error("No range pack found in `ranges/`.")
        return
    # Metadata only -- Compare needs just which seats exist, not full nodes.
    _cmp_actors = _cached_node_filter_meta(cmp_pack.pack_id)
    _cmp_seat_order = preflop_order(cmp_pack.table_size)
    _cmp_positions = [p for p in _cmp_seat_order if p in _cmp_actors]
    f1, f2 = st.columns(2)
    with f1:
        positions = st.multiselect(
            "Hero positions",
            options=_cmp_positions,
            default=_cmp_positions,
            key=f"cmp_pos_{cmp_pack.pack_id}",
            help="Empty = all positions.",
        )
    with f2:
        contexts = st.multiselect(
            "Action faced",
            options=[
                "Opening",
                "Facing single raise",
                "Facing 3-bet",
                "Facing 4-bet+",
                "After one call",
                "After multiple calls",
            ],
            default=["Opening", "Facing single raise", "Facing 3-bet"],
            key="cmp_ctx",
            help=ACTION_FACED_HELP,
        )

    difficulty_bands = {
        "Easy": (400, 1300),
        "Medium": (1300, 2100),
        "Hard": (2100, 3200),
        "Mixed": (DIFFICULTY_MIN, DIFFICULTY_MAX),
    }
    s1, s2, s3 = st.columns(3)
    with s1:
        n_spots = int(
            st.number_input("Spots", min_value=1, max_value=25, value=5, key="cmp_n")
        )
    with s2:
        preset = st.radio(
            "Difficulty",
            options=list(difficulty_bands),
            index=3,  # Mixed = full band
            horizontal=True,
            key="cmp_diff",
        )
    with s3:
        # Per-side models (mirrors PLO Compare): same cost as a single-model
        # compare and it unlocks model-vs-model A/B on identical spots.
        _short_lbl = lambda lbl: lbl.split(" (")[0]  # noqa: E731
        model_a_label = st.selectbox(
            "Model A",
            options=list(_MODEL_LABEL_TO_API),
            index=0,
            format_func=_short_lbl,
            key="cmp_model_a",
        )
        model_b_label = st.selectbox(
            "Model B",
            options=list(_MODEL_LABEL_TO_API),
            index=0,
            format_func=_short_lbl,
            key="cmp_model_b",
        )
    band_low, band_high = difficulty_bands[preset]

    o1, o2 = st.columns(2)
    with o1:
        cmp_display_in_bb = (
            st.radio(
                "Amounts",
                options=["Big blinds", "Dollars"],
                index=0,
                horizontal=True,
                key="cmp_amounts",
                help="How the question renders. Big blinds is the default.",
            )
            == "Big blinds"
        )
    with o2:
        cmp_style = ANSWER_STYLE_FROM_RADIO_LABEL[
            st.radio(
                "Answer option style",
                options=list(ANSWER_STYLE_FROM_RADIO_LABEL),
                index=list(ANSWER_STYLE_FROM_RADIO_LABEL.values()).index("auto"),
                key="cmp_answer_style",
                help="Same styles as the Generate page. Auto-pick = Basic for "
                "dominant-action spots, GTO for mixed.",
            )
        ]

    s4, s5 = st.columns(2)
    with s4:
        seed = int(
            st.number_input(
                "Seed", min_value=0, max_value=1_000_000, value=42, key="cmp_seed"
            )
        )
    with s5:
        dry = st.toggle("Dry run", key="cmp_dry", help="No API calls — flow check.")

    # Advanced filters — the same worthiness-window + EV-gap gates the
    # Generate page uses, so the spots a comparison samples match a real
    # batch (both sides still see IDENTICAL spots). Defaults mirror Generate
    # (65-99 window, 90-95% trap band excluded).
    with st.expander("Advanced filters (worthiness window · EV-gap gate)"):
        cmp_freq_low, cmp_freq_high = st.slider(
            "Solver frequency worthiness window (%)",
            min_value=50,
            max_value=100,
            value=(65, 99),
            key="cmp_worthiness_slider",
            help="Below 65% = no clear best answer; 100% = trivial.",
        )
        cmp_exclude_band = st.checkbox(
            "Exclude ambiguous 90–95% band",
            value=False,
            key="cmp_exclude_ambiguous_band",
            help="Optional extra hole at 90–95% (off by default). With the "
            "near-pure exclusion below, checking this tightens the window to 65–90%.",
        )
        cmp_exclude_near_pure_band = st.checkbox(
            "Exclude ambiguous 95–99% band (recommended)",
            value=True,
            key="cmp_exclude_near_pure_band",
            help="Punches a hole at 95–99% (nearly pure: the 'Mostly X' answer "
            "reads as 'Always X'). On by default. A literal 100% spot still qualifies.",
        )
        cmp_min_ev_gap = st.slider(
            "Minimum EV gap (bb) — 0 = off",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            key="cmp_min_ev_gap",
            help="Drops call/fold spots whose EV gap to the 2nd-best action "
            "is below this. Raise spots (no computed EV) always pass.",
        )

    # --- Layer-7 claim checker (opt-in, runs on BOTH sides) -----------------
    cmp_run_claim_checker = st.checkbox(
        "Run claim checker (Layer 7) on both sides",
        value=False,
        key="cmp_run_claim_checker",
        help="After each explanation is written, a second LLM pass audits it "
        "against the data block and flags suspect poker claims. Adds ONE API "
        "call per question per side. Its verdict (clean or flagged) shows in a "
        "dropdown under each spot below. Uses the prompt saved on the Generate "
        "page.",
    )
    cmp_claim_checker_prompt = (
        _load_claim_checker_prompt() if cmp_run_claim_checker else None
    )

    # Side-identity checks: identical sides = pointless; two variables at
    # once = a confounded verdict.
    same_content = a_slug == b_slug
    same_models = model_a_label == model_b_label
    if same_content and same_models:
        st.info(
            "Pick two different prompts, or two different models — identical "
            "sides would generate the same thing twice."
        )
    elif not same_content and not same_models:
        st.warning(
            "Both the prompt and the model differ between sides, so a verdict "
            "won't tell you which one caused the difference. For a clean "
            "test, vary one at a time."
        )
    if st.button(
        "Run comparison",
        type="primary",
        disabled=(same_content and same_models) or jobs.has_active_job(),
        key="cmp_run",
    ):
        if not dry and not os.environ.get("ANTHROPIC_API_KEY"):
            st.error(
                "ANTHROPIC_API_KEY is not set. Add it to `.env`, or enable Dry "
                "run to test the flow."
            )
            return
        pack = cmp_pack  # both sides sample the same selected pack
        # Venue/stakes are display-only framing. Compare has no venue widget
        # (it's a prompt A/B test, not a framing tool), so take the pack's
        # default straight from the shared helper -- without this both sides
        # silently rendered "Online · 9-Handed" even on the Live Monker pack.
        cmp_venue, cmp_stakes_bb = _pack_display_framing(pack)
        model_api_a = _MODEL_LABEL_TO_API.get(model_a_label, model_a_label)
        model_api_b = _MODEL_LABEL_TO_API.get(model_b_label, model_b_label)
        # When the models differ, bake the model into each side's label so
        # the tally + verdict buttons say exactly what they're crediting.
        a_name = names[a_slug] + (
            f" · {_short_lbl(model_a_label)}" if not same_models else ""
        )
        b_name = names[b_slug] + (
            f" · {_short_lbl(model_b_label)}" if not same_models else ""
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        PREFLOP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_a = PREFLOP_OUTPUT_DIR / f"compare_{ts}_A.csv"
        out_b = PREFLOP_OUTPUT_DIR / f"compare_{ts}_B.csv"
        with st.status(
            "Running both prompts on the same spots…", expanded=True
        ) as status:
            st.write(f"A — {a_name}")
            res_a = generate_preflop_batch(
                pack=pack,
                output_path=out_a,
                total_questions=n_spots,
                hero_positions=positions or None,
                action_contexts=contexts or None,
                min_difficulty=band_low,
                max_difficulty=band_high,
                min_frequency=cmp_freq_low / 100.0,
                max_frequency=cmp_freq_high / 100.0,
                exclude_ambiguous_band=cmp_exclude_band,
                exclude_near_pure_band=cmp_exclude_near_pure_band,
                min_ev_gap_bb=(
                    None if cmp_min_ev_gap == 0.0 else float(cmp_min_ev_gap)
                ),
                answer_style=cmp_style,
                display_in_bb=cmp_display_in_bb,
                live_or_online=cmp_venue,
                stakes_bb_dollars=cmp_stakes_bb,
                system_prompt=lib.get_text(a_slug),
                prompt_name=names[a_slug],
                random_seed=seed,
                temperature=0.0,
                model=model_api_a,
                run_claim_checker=cmp_run_claim_checker,
                claim_checker_prompt=cmp_claim_checker_prompt,
                dry_run=dry,
            )
            st.write(f"B — {b_name}")
            res_b = generate_preflop_batch(
                pack=pack,
                output_path=out_b,
                total_questions=n_spots,
                hero_positions=positions or None,
                action_contexts=contexts or None,
                min_difficulty=band_low,
                max_difficulty=band_high,
                min_frequency=cmp_freq_low / 100.0,
                max_frequency=cmp_freq_high / 100.0,
                exclude_ambiguous_band=cmp_exclude_band,
                exclude_near_pure_band=cmp_exclude_near_pure_band,
                min_ev_gap_bb=(
                    None if cmp_min_ev_gap == 0.0 else float(cmp_min_ev_gap)
                ),
                answer_style=cmp_style,
                display_in_bb=cmp_display_in_bb,
                live_or_online=cmp_venue,
                stakes_bb_dollars=cmp_stakes_bb,
                system_prompt=lib.get_text(b_slug),
                prompt_name=names[b_slug],
                random_seed=seed,
                temperature=0.0,
                model=model_api_b,
                run_claim_checker=cmp_run_claim_checker,
                claim_checker_prompt=cmp_claim_checker_prompt,
                dry_run=dry,
            )
            status.update(label="Comparison ready", state="complete")
        # Compare runs are real API spend: log both sides to the lifetime
        # total (no-op for dry runs, where model_used is empty).
        _log_batch_result_usage(res_a)
        _log_batch_result_usage(res_b)
        err = _finish_comparison(res_a, res_b, a_name, b_name, "cmp_result")
        if err:
            st.error(err)
            st.caption(
                "Nothing was saved. Adjust the prompts / models / filters and "
                "run again."
            )
            return
        st.rerun()

    result = _render_past_comparisons(PREFLOP_OUTPUT_DIR, "cmp_result")
    if not result:
        return
    a_csv = Path(str(result["a_csv"]))
    b_csv = Path(str(result["b_csv"]))
    if not (a_csv.is_file() and b_csv.is_file()):
        st.info("Run a comparison above to see results.")
        return

    df_a = _read_csv_cached(str(a_csv), a_csv.stat().st_mtime, as_str=True)
    df_b = _read_csv_cached(str(b_csv), b_csv.stat().st_mtime, as_str=True)
    rows_a = [{str(k): str(v) for k, v in r.items()} for r in df_a.to_dict("records")]
    rows_b = [{str(k): str(v) for k, v in r.items()} for r in df_b.to_dict("records")]
    pairs = compare.join_by_spot(rows_a, rows_b)
    verdicts = compare.load_verdicts(a_csv)
    counts = compare.tally(verdicts)

    st.divider()
    st.markdown(
        f"### Tally — **{result['a_name']}** {counts['A']}  ·  "
        f"**{result['b_name']}** {counts['B']}  ·  tie {counts['tie']}   "
        f"({len(verdicts)}/{len(pairs)} judged)"
    )
    if not pairs:
        st.warning("No shared spots to compare (did both runs produce rows?).")
        return

    opts = [f"{result['a_name']} better", "Tie", f"{result['b_name']} better"]
    to_verdict = {opts[0]: "A", opts[1]: "tie", opts[2]: "B"}
    from_verdict = {"A": opts[0], "tie": opts[1], "B": opts[2]}

    # Finalize grades live in each compare CSV's own .review.json sidecar (the
    # same store the Review page uses), so a question finalized here flows
    # into the cross-batch "Approved questions" pool. Loaded once per rerun.
    reviews_a = review.load_reviews(a_csv)
    reviews_b = review.load_reviews(b_csv)

    # --- batch finalize: send an ENTIRE side to the approved pool at once ---
    def _approve_all(win_csv: Path, lose_csv: Path, side: str) -> None:
        # Exclusive per spot (like the per-spot buttons): approve the winning
        # side and drop the other's approval, so the pool keeps one side per
        # spot. Uses each row's as-generated prose (per-spot edits are not
        # captured -- use the per-spot buttons for those).
        for _k, r_a, r_b in pairs:
            win_row = r_a if win_csv == a_csv else r_b
            lose_row = r_b if win_csv == a_csv else r_a
            review.save_review(
                win_csv, str(win_row.get("No", "")), "approved",
                f"batch-approved ({side}) from compare",
            )
            review.remove_review(lose_csv, str(lose_row.get("No", "")))

    ba1, ba2 = st.columns(2)
    if ba1.button(
        f"✅ Approve all of A — {result['a_name']} ({len(pairs)})",
        key="cmp_approve_all_a",
        use_container_width=True,
    ):
        _approve_all(a_csv, b_csv, "A")
        st.success(f"Sent all {len(pairs)} A-side questions to the approved pool.")
        st.rerun()
    if ba2.button(
        f"✅ Approve all of B — {result['b_name']} ({len(pairs)})",
        key="cmp_approve_all_b",
        use_container_width=True,
    ):
        _approve_all(b_csv, a_csv, "B")
        st.success(f"Sent all {len(pairs)} B-side questions to the approved pool.")
        st.rerun()
    st.caption(
        "Approves every spot's chosen side into the Approved pool (download at "
        "the bottom of this page or on the Review page). The per-spot buttons "
        "below override individual picks and keep any edits you made."
    )

    for key, row_a, row_b in pairs:
        with st.container(border=True):
            if row_a.get("Context"):
                st.caption(row_a["Context"])
            st.markdown(_md_lines(row_a.get("Question", "")))
            picks = ", ".join(
                row_a.get(f"option {i}", "")
                for i in (1, 2, 3, 4)
                if row_a.get(f"option {i}", "")
            )
            st.caption(
                f"Options: {picks}  ·  Correct: **{row_a.get('Correct Answer', '')}**"
            )
            # Shared strategic facts (identical for both prompts -- same spot),
            # shown once, mirroring the Review card.
            if row_a.get("action_frequencies"):
                st.markdown(
                    "**Solver frequencies:** "
                    + row_a["action_frequencies"]
                )
            fact_bits: list[str] = []
            for col, lbl in (
                ("archetype", "archetype"),
                ("ev_gap_bb", "EV gap"),
                ("Difficulty Rating", "difficulty"),
                ("Position Matchup", "matchup"),
                ("Pot Participant", "pot"),
            ):
                val = row_a.get(col, "")
                if val:
                    fact_bits.append(f"{lbl}: `{val}`")
            if fact_bits:
                st.caption(" · ".join(fact_bits))
            if row_a.get("concept_tags"):
                st.caption(f"concept tags: {row_a['concept_tags']}")
            if row_a.get("skills"):
                st.caption(f"skills: {row_a['skills']}")
            # Editable explanations: tweak the prose before finalizing -- the
            # finalize buttons save whatever is in the box. Keys carry the
            # CSV stem so a fresh comparison never inherits stale edits.
            orig_a = row_a.get("Answer Explanation", "")
            orig_b = row_b.get("Answer Explanation", "")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{result['a_name']}**")
                edited_a = st.text_area(
                    "Explanation A",
                    value=orig_a,
                    height=400,
                    key=f"cmp_exp_a_{a_csv.stem}_{key}",
                    label_visibility="collapsed",
                )
                if edited_a.strip() != orig_a.strip():
                    st.caption("✏️ Edited — finalizing A saves this text.")
            with col_b:
                st.markdown(f"**{result['b_name']}**")
                edited_b = st.text_area(
                    "Explanation B",
                    value=orig_b,
                    height=400,
                    key=f"cmp_exp_b_{b_csv.stem}_{key}",
                    label_visibility="collapsed",
                )
                if edited_b.strip() != orig_b.strip():
                    st.caption("✏️ Edited — finalizing B saves this text.")
            # Both sides are the SAME spot, so the deterministic math is
            # identical -- one shared panel under the pair. (No-ops on PLO,
            # whose rows carry no stat_notes yet.)
            _render_stat_panel(row_a)
            _render_exploit_panel(row_a)
            _render_claim_check_panel(row_a)
            cur = verdicts.get(key)
            idx = opts.index(from_verdict[cur]) if cur in from_verdict else None
            choice = st.radio(
                "Which is better?",
                opts,
                index=idx,
                horizontal=True,
                key=f"cmp_v_{key}",
            )
            if choice is not None and to_verdict[choice] != cur:
                compare.save_verdict(a_csv, key, to_verdict[choice])
                st.rerun()

            # --- finalize: save the chosen explanation to the approved pool ---
            no_a, no_b = str(row_a.get("No", "")), str(row_b.get("No", ""))
            fin_a = reviews_a.get(no_a, {}).get("status") == "approved"
            fin_b = reviews_b.get(no_b, {}).get("status") == "approved"
            fcol_a, fcol_b = st.columns(2)
            if fcol_a.button(
                "Save A to finalized",
                key=f"cmp_fin_a_{key}",
                disabled=fin_a,
                use_container_width=True,
            ):
                # Exclusive: finalizing one variant un-finalizes the other.
                # An edited explanation rides along as a sidecar override --
                # the compare CSV itself keeps the original prose.
                review.save_review(
                    a_csv,
                    no_a,
                    "approved",
                    "finalized from compare",
                    explanation=(
                        edited_a if edited_a.strip() != orig_a.strip() else None
                    ),
                )
                review.remove_review(b_csv, no_b)
                st.rerun()
            if fcol_b.button(
                "Save B to finalized",
                key=f"cmp_fin_b_{key}",
                disabled=fin_b,
                use_container_width=True,
            ):
                review.save_review(
                    b_csv,
                    no_b,
                    "approved",
                    "finalized from compare",
                    explanation=(
                        edited_b if edited_b.strip() != orig_b.strip() else None
                    ),
                )
                review.remove_review(a_csv, no_a)
                st.rerun()
            if fin_a or fin_b:
                which = result["a_name"] if fin_a else result["b_name"]
                _fin_grade = (reviews_a.get(no_a) if fin_a else reviews_b.get(no_b)) or {}
                _edited_note = " (with your edits)" if _fin_grade.get("explanation") else ""
                st.caption(f"✅ Saved to finalized using **{which}**{_edited_note}.")
                if st.button("Remove from finalized", key=f"cmp_unfin_{key}"):
                    review.remove_review(a_csv, no_a)
                    review.remove_review(b_csv, no_b)
                    st.rerun()

    # --- download the shared finalized pool (same set as the Review page) -----
    st.divider()
    fin_fields, fin_rows = review.collect_approved_rows(PREFLOP_OUTPUT_DIR)
    if fin_rows:
        st.download_button(
            f"⬇️  Download finalized questions (CSV) — {len(fin_rows)} total",
            review.approved_rows_to_csv(fin_fields, fin_rows),
            file_name="nlhe_approved_all_batches.csv",
            mime="text/csv",
            type="primary",
            key="cmp_download_finalized",
        )
        st.caption(
            "Every question you save to finalized here or approve on the "
            "Review page, across all batches, deduped by spot."
        )
    else:
        st.caption(
            "Save questions to finalized above (or approve them on the Review "
            "page) to build your downloadable set."
        )


def _scan_postflop_outputs() -> pd.DataFrame:
    """One row per postflop batch CSV (newest first) -- like
    :func:`_scan_preflop_outputs` but over ``POSTFLOP_OUTPUT_DIR``."""
    cols = ["filename", "modified", "size_kb", "questions", "_path"]
    if not POSTFLOP_OUTPUT_DIR.is_dir():
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, object]] = []
    for path in POSTFLOP_OUTPUT_DIR.glob("*.csv"):
        stat = path.stat()
        try:
            with path.open("rb") as fh:
                n = max(0, sum(1 for _ in fh) - 1)
        except OSError:
            n = 0
        rows.append({
            "filename": path.name,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size_kb": round(stat.st_size / 1024),
            "questions": n,
            "_path": str(path),
        })
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values("modified", ascending=False).reset_index(drop=True)
    return df


def _street_decision_ordinal(df, row) -> tuple[int, int]:
    """``(k, n)`` for a leg that is the k-th of n decisions on its street within
    its hand (e.g. a river check then a river facing-a-bet = 2 decisions). Returns
    ``(1, 1)`` for a standalone row or a street with a single decision -- so the
    caller only shows the label when ``n > 1``. Makes an escalating same-street
    line (check -> face a bet -> face a raise) legible instead of two bare
    'River' cards."""
    hid = str(row.get("hand_id", "") or "").strip()
    stage = str(row.get("Hand Stage", "") or "")
    if not hid:
        return (1, 1)

    def _seq(r) -> int:
        try:
            return int(float(str(r.get("sequence_index", "") or 0)))
        except ValueError:
            return 0

    same = [
        r for _i, r in df.iterrows()
        if str(r.get("hand_id", "") or "").strip() == hid
        and str(r.get("Hand Stage", "") or "") == stage
    ]
    if len(same) <= 1:
        return (1, len(same) or 1)
    same.sort(key=_seq)
    my_seq = _seq(row)
    k = next((i + 1 for i, r in enumerate(same) if _seq(r) == my_seq), 1)
    return (k, len(same))


def _render_postflop_question_card(
    row, *, df, csv_path: Path, reviews: dict, meta, grouped: bool = False,
    nav_key: str | None = None, idx: int | None = None,
) -> None:
    """Render ONE postflop question with the FULL review UI -- the single source
    of truth for both the single-question navigator and the grouped play-through
    view, so a leg in a hand is reviewed exactly like a standalone question.

    Includes: header + grade, question + options (correct/neutral marked),
    auto-saving explanation editor, Layer-7 revise/claim-check panels + prompt
    inspector, the facts line, the preflop+this-street range grids, the
    auto-saving difficulty input, the approve/needs/reject grading, and remove.

    ``grouped`` controls two things: the ranges expander starts collapsed (it's
    heavy, and a hand has several legs), and grading/remove do NOT advance a nav
    index (there is none). All widget keys are namespaced by ``No`` so the same
    card renders many times on one page without colliding.
    """
    no = _cell(row, "No")
    existing = reviews.get(no, {})

    # k-th of n decisions on this street within the hand (only shown when n > 1).
    _sd_k, _sd_n = _street_decision_ordinal(df, row)
    _stage_label = _cell(row, "Hand Stage")
    if _sd_n > 1:
        _stage_label += f" · decision {_sd_k} of {_sd_n}"

    # EDIT-LOSS INVARIANT: the editor, difficulty, and EVERY control that can
    # leave this question (nav / grade / remove) live inside ONE st.form, so a
    # click ships the current edit atomically with the click (framework
    # contract -- see _flush_review_edit). A plain st.button here races the
    # text_area's blur commit and loses edits. NEVER add a plain st.button
    # that navigates.
    _has_nav = not grouped and nav_key is not None and idx is not None
    _prev_clicked = _next_clicked = _go_clicked = False
    _jump_val = idx if idx is not None else 0
    with st.form(key=f"pf_review_form::{csv_path.name}::{no}", border=False):
        if _has_nav:
            _n1, _n2, _n3 = st.columns([1, 2, 1])
            _prev_clicked = _n1.form_submit_button(
                "◀ Prev", use_container_width=True, disabled=idx == 0
            )
            with _n2:
                _jc, _gc = st.columns([3, 1])
                _nos = df["No"].tolist()
                _jump_val = _jc.selectbox(
                    "Jump to",
                    options=list(range(len(df))),
                    index=idx,
                    format_func=lambda i: f"#{_nos[i]}  ({i + 1}/{len(df)})",
                    label_visibility="collapsed",
                )
                _go_clicked = _gc.form_submit_button("Go", use_container_width=True)
            _next_clicked = _n3.form_submit_button(
                "Next ▶", use_container_width=True, disabled=idx >= len(df) - 1
            )
        with st.container(border=True):
            # In the grouped view each leg leads with its place in the hand.
            if grouped:
                seq = _cell(row, "sequence_index")
                total = _cell(row, "sequence_total")
                st.markdown(f"**{seq}/{total} · {_stage_label}**  (#{no})")
            elif _sd_n > 1:
                # Single-question view: note the same-street decision context too.
                st.caption(f"Part of a hand — **{_stage_label}**")
            h = st.columns(4)
            h[0].markdown(f"**#{no}**")
            h[1].markdown(f"Seat&nbsp;**{_cell(row, 'User Seat')}**")
            h[2].markdown(f"Hand&nbsp;**{_cell(row, 'User Cards')}**")
            h[3].markdown(f"Difficulty&nbsp;**{_cell(row, 'Difficulty Rating')}**")
            if existing.get("status"):
                st.caption(
                    "Current grade: **"
                    + _REVIEW_STATUS_LABEL.get(existing["status"], existing["status"])
                    + "**"
                )
            elif _cell(row, "validation_status"):
                st.caption(f"Validation: `{_cell(row, 'validation_status')}`")

            if _cell(row, "Context"):
                st.caption(_cell(row, "Context"))
            st.markdown("**Question**")
            st.markdown(_md_lines(_cell(row, "Question")))

            st.markdown("**Options**")
            correct = _cell(row, "Correct Answer")
            neutral = {
                x.strip() for x in (_cell(row, "neutral_credit") or "").split(",") if x.strip()
            }
            for i in (1, 2, 3, 4):
                opt = _cell(row, f"option {i}")
                if not opt:
                    continue
                mark = "✅ " if opt == correct else ("😐 " if opt in neutral else "▫️ ")
                st.markdown(mark + opt)
            if neutral:
                st.caption("✅ correct · 😐 neutral credit · ▫️ mistake")

            st.markdown(
                "**Answer Explanation** _(saves on any button click: "
                "Save, Prev/Next, grade)_"
            )
            ekey = f"postflop_review_expl::{csv_path.name}::{no}"
            st.text_area(
                "Answer Explanation",
                value=_cell(row, "Answer Explanation"),
                key=ekey,
                height=240,
                label_visibility="collapsed",
            )
            _save_clicked = st.form_submit_button("💾 Save edits")
            with st.expander("Preview (rendered)", expanded=False):
                st.info(_md_lines(_cell(row, "Answer Explanation")))

            # The deterministic "Show the math" strip (pot odds / equity / currently
            # ahead / blockers / SPR), from the row's stat_notes column -- now
            # populated on postflop rows, so it renders here like the preflop Review.
            _render_stat_panel(row)

            # --- Layer-7: auto-fix lifecycle + claim-checker flags + prompt inspector ---
            _qrecs = meta.get("questions", []) if isinstance(meta, dict) else []
            _meta_node_ids = {q.get("node_id") for q in _qrecs}
            _ref_parts = [p for p in _cell(row, "solver_reference").split("/") if p]
            _ref_node = next((p for p in reversed(_ref_parts) if p in _meta_node_ids), "")
            _ref_combo = _ref_parts[-1] if _ref_parts else ""
            _qrec = next(
                (
                    q for q in _qrecs
                    if q.get("node_id") == _ref_node and q.get("hero_combo") == _ref_combo
                ),
                None,
            )
            row_strs = {c: _cell(row, c) for c in df.columns}
            _render_revise_panel(_qrec)         # REWRITTEN vs ORIGINAL (if revise ran)
            _render_claim_check_panel(row_strs)  # claim-checker flags (if it ran)
            # The preflop leg of a play-through is written by a different (preflop)
            # prompt than the postflop legs -- say so where the reviewer is looking.
            _is_preflop_leg = _cell(row, "Hand Stage").lower() == "preflop"
            with st.expander("🔍 Prompt & inputs — exactly what the LLM saw"):
                if _qrec and _qrec.get("solver_data"):
                    st.markdown("**SOLVER DATA block** (the facts the model wrote from)")
                    st.code(str(_qrec["solver_data"]))
                else:
                    st.caption(
                        "No solver-data snapshot for this row (preflop-entry leg, "
                        "older batch, or a reordered row)."
                    )
                st.markdown("**Question shown**")
                st.code(_cell(row, "Question"))
                _opts = [_cell(row, f"option {i}") for i in (1, 2, 3, 4)]
                st.markdown(f"**Options:** {[o for o in _opts if o]}")
                st.markdown(f"**Correct answer:** {_cell(row, 'Correct Answer')}")
                if _is_preflop_leg:
                    st.caption(
                        "This is the PREFLOP-ENTRY leg: it uses the built-in PREFLOP "
                        "prompt (not the postflop system prompt, and not yet "
                        "admin-editable)."
                    )
                else:
                    st.caption(
                        "The model writes only the prose, from the SOLVER DATA above, "
                        "using the active postflop system prompt (Prompt page → Postflop)."
                    )

            st.markdown(f"**Solver frequencies:**&nbsp;{_cell(row, 'action_frequencies')}")
            if _cell(row, "action_ev_bb"):
                st.markdown(f"**Per-action EV (bb):**&nbsp;{_cell(row, 'action_ev_bb')}")
            fbits = []
            for col, lbl in (
                ("hero_equity", "equity"),
                ("pot_odds", "pot odds"),
                ("spr", "SPR"),
                ("board_texture", "board"),
                ("archetype", "archetype"),
                ("Position Matchup", "matchup"),
            ):
                val = _cell(row, col)
                if val:
                    fbits.append(f"{lbl}: `{val}`")
            if fbits:
                st.caption(" · ".join(fbits))
            if _cell(row, "skills"):
                st.caption(f"🎯 skills: {_cell(row, 'skills')}")
            if _cell(row, "concept_tags"):
                st.caption(f"concept tags: {_cell(row, 'concept_tags')}")

            # --- visual ranges: every player, preflop + current street ---
            _meta_nodes = {q.get("node_id") for q in _qrecs}
            _parts = [p for p in _cell(row, "solver_reference").split("/") if p]
            _ref_node2 = next((p for p in reversed(_parts) if p in _meta_nodes), "")
            _street_ranges = next(
                (q.get("street_ranges") for q in _qrecs if q.get("node_id") == _ref_node2),
                None,
            )
            _preflop_ranges = meta.get("preflop_ranges") if isinstance(meta, dict) else None
            _prior_ranges = next(
                (q.get("prior_street_ranges") for q in _qrecs if q.get("node_id") == _ref_node2),
                None,
            )
            _prior_label = next(
                (q.get("prior_street_label") for q in _qrecs if q.get("node_id") == _ref_node2),
                None,
            )
            _street_strategy = next(
                (q.get("street_strategy") for q in _qrecs if q.get("node_id") == _ref_node2),
                None,
            )
            if not isinstance(_street_strategy, dict):
                _street_strategy = {}
            _preflop_entry = meta.get("preflop_entry_actions", {}) if isinstance(meta, dict) else {}
            if _street_ranges or _preflop_ranges:
                with st.expander(
                    "📊 Player ranges — the street before + this-street strategy",
                    expanded=not grouped,
                ):
                    _hero_seat = _cell(row, "User Seat").split("-", 1)[0]
                    _seats = sorted(set(_street_ranges or {}) | set(_preflop_ranges or {}))
                    _seats.sort(key=lambda p: p != _hero_seat)  # hero first

                    _act_rank = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4, "all-in": 5}
                    _act_color = {
                        "fold": "#6b7280", "check": range_view.COLOR_FOLD,
                        "call": range_view.COLOR_CALL, "bet": range_view.COLOR_RAISE,
                        "raise": range_view.COLOR_ALLIN, "all-in": "#3d0c0c",
                    }

                    def _segs(snap: dict | None, color: str) -> dict:  # single-colour
                        return {h: [(w, color)] for h, w in (snap or {}).items() if w > 0.004}

                    def _strat_segs(strat: dict) -> dict:  # action-coloured strategy
                        segs: dict = {}
                        for action in sorted(
                            strat, key=lambda a: _act_rank.get(a.split()[0].lower(), 9)
                        ):
                            col = _act_color.get(action.split()[0].lower(), range_view.COLOR_INRANGE)
                            for h, w in strat[action].items():
                                if w > 0.004:
                                    segs.setdefault(h, []).append((w, col))
                        return segs

                    st.caption(
                        "**How to read a cell:** fill height = how much of this player's "
                        "range that hand is here; colour = the action (🟦 check · 🟩 call "
                        "· 🟥 bet/raise · ⬜ in range, no action yet). Left grid = the "
                        "street before this decision, right grid = this street."
                    )
                    with st.popover("ℹ️ Where does each range come from?"):
                        st.markdown(
                            "**Hero (the player to act)**\n"
                            "- **Left grid** — hero's range as it stood on the *street "
                            "before* this decision (a turn question shows the flop range, "
                            "a river question the turn range). It's the set of hands hero "
                            "could still have coming into this street.\n"
                            "- **Right grid** — hero's *strategy on this street*: every "
                            "hand coloured by what the solver does with it here "
                            "(🟦 check / 🟩 call / 🟥 bet or raise). This is the decision "
                            "the question is asking about.\n\n"
                            "**Villain (the other player)**\n"
                            "- **Left grid** — villain's range on the street before.\n"
                            "- **Right grid** — if villain has NOT acted yet this street "
                            "(e.g. they checked back the previous street), this is the "
                            "hands they can still have *right now* (⬜ neutral = holdings "
                            "only, no action to colour) — \"what you're up against\", "
                            "carried forward from earlier streets. If villain DID already "
                            "act this street (e.g. they bet into you), it instead shows "
                            "*their* strategy at that action (🟦/🟩/🟥), so you see the "
                            "range behind the bet you face.\n\n"
                            "Toggle **Conditional view** to make every in-range cell full "
                            "height, coloured by what that specific hand does when held "
                            "(useful for reading a mixed strategy hand-by-hand)."
                        )
                    _cond_key = f"pf_range_conditional::{csv_path.name}::{no}"
                    _conditional = st.toggle(
                        "Conditional view — full-height cells (strategy *when the hand is held*)",
                        value=False,
                        key=_cond_key,
                        help="Off (range-weighted): a cell's fill = how much of the range "
                        "that hand is. On (conditional): every in-range cell is full height, "
                        "coloured by what the hand does WHEN HELD.",
                    )

                    def _maybe_conditional(segs: dict) -> dict:
                        if not _conditional:
                            return segs
                        out: dict = {}
                        for hnd, bands in segs.items():
                            tot = sum(w for w, _c in bands)
                            out[hnd] = [(w / tot, c) for w, c in bands] if tot > 0 else bands
                        return out

                    for _pos in _seats:
                        _role = "🎯 hero (to act)" if _pos == _hero_seat else "villain"
                        st.markdown(f"**{_pos}** &nbsp;·&nbsp; _{_role}_")
                        _gp, _gc = st.columns(2)
                        _left = (_prior_ranges or {}).get(_pos) if _prior_ranges else None
                        _cur = (_street_ranges or {}).get(_pos)
                        _strat = _street_strategy.get(_pos)
                        _entry = _preflop_entry.get(_pos, "call")
                        _entry_word = "raised" if _entry == "raise" else "called"
                        with _gp:
                            if _left:
                                st.caption(
                                    f"{str(_prior_label).capitalize()} range — "
                                    f"~{range_view.range_pct(_left):.0f}% of hands"
                                )
                                st.html(range_view.grid_html(_maybe_conditional(
                                    _segs(_left, range_view.COLOR_INRANGE)
                                )))
                            elif (_pre := (_preflop_ranges or {}).get(_pos)):
                                st.caption(
                                    f"Preflop — {_entry_word} ~{range_view.range_pct(_pre):.0f}% of hands"
                                )
                                st.html(range_view.grid_html(_maybe_conditional(
                                    _segs(_pre, _act_color.get(_entry, range_view.COLOR_CALL))
                                )))
                            else:
                                st.caption("Preflop — n/a")
                        with _gc:
                            if _strat:
                                st.caption(
                                    f"This street — strategy "
                                    f"(~{range_view.range_pct(_cur):.0f}% of hands in range)"
                                )
                                st.html(range_view.grid_html(_maybe_conditional(_strat_segs(_strat))))
                            elif _cur:
                                st.caption(
                                    f"**{_pos}'s current holdings** — every hand "
                                    f"{_pos} can still have right now "
                                    f"(~{range_view.range_pct(_cur):.0f}% of all hands). "
                                    "⬜ Grey = holdings only: it is NOT this player's turn "
                                    "to act on this street, so there is no strategy to "
                                    "colour. This is just \"what you're up against\", not "
                                    "an action."
                                )
                                st.html(range_view.grid_html(_maybe_conditional(
                                    _segs(_cur, range_view.COLOR_INRANGE)
                                )))
                            else:
                                st.caption("This street — n/a")

            # Deterministic exploit adjustments (vs nit / station / maniac). Reads
            # the baked exploit_notes column, now populated on postflop rows too.
            _render_exploit_panel(row)

            dkey = f"postflop_review_diff::{csv_path.name}::{no}"
            try:
                cur_diff = int(float(_cell(row, "Difficulty Rating") or 0))
            except ValueError:
                cur_diff = 0
            st.number_input(
                "Difficulty Rating (saves with any button click)",
                min_value=0,
                max_value=3500,
                step=10,
                value=cur_diff,
                key=dkey,
            )

        # --- grading (inside the form: grade clicks also carry the edit) ---
        st.markdown("**Grade**")
        note = st.text_area(
            "Note (optional)",
            value=existing.get("note", ""),
            key=f"postflop_review_note::{csv_path.name}::{no}",
            height=70,
        )
        g1, g2, g3 = st.columns(3)
        _approve_clicked = g1.form_submit_button(
            "✅ Approve", use_container_width=True, type="primary"
        )
        _needs_clicked = g2.form_submit_button(
            "⚠️ Needs review", use_container_width=True
        )
        _reject_clicked = g3.form_submit_button(
            "❌ Reject", use_container_width=True
        )
        _remove_clicked = st.form_submit_button(
            f"🗑  Remove #{no} from this batch",
            help="Deletes this question from the CSV (regenerate to recover).",
        )
    # --- the single post-form handler: saving first can never lose an edit
    # (the submit shipped editor/difficulty/note atomically with the click).
    if any((_prev_clicked, _next_clicked, _go_clicked, _save_clicked,
            _approve_clicked, _needs_clicked, _reject_clicked, _remove_clicked)):
        _flush_review_edit(csv_path, no, key_prefix="postflop_review")
        if _remove_clicked:
            if review.remove_question(csv_path, no):
                if _has_nav:
                    st.session_state[nav_key] = max(0, min(idx, len(df) - 2))
                st.rerun()
            st.warning(f"#{no} was not found in the batch.")
        else:
            if _approve_clicked or _needs_clicked or _reject_clicked:
                status = (
                    "approved" if _approve_clicked
                    else "needs_review" if _needs_clicked
                    else "rejected"
                )
                review.save_review(csv_path, no, status, note)
                if _has_nav:
                    st.session_state[nav_key] = min(idx + 1, len(df) - 1)
            elif _has_nav and _prev_clicked:
                st.session_state[nav_key] = idx - 1
            elif _has_nav and _next_clicked:
                st.session_state[nav_key] = idx + 1
            elif _has_nav and _go_clicked:
                st.session_state[nav_key] = int(_jump_val)
            st.rerun()


def _render_postflop_grouped_review(df, csv_path: Path, reviews: dict, meta) -> None:
    """Per-hand grouped review for full-hand (play-through) batches.

    Groups rows by ``hand_id`` (ordered by ``sequence_index``) into one
    expandable card per hand. Each leg is rendered with the SAME full review UI
    as the single-question navigator (:func:`_render_postflop_question_card`):
    edit + auto-save the explanation, Layer-7 panels, ranges, difficulty, grade,
    and remove. Standalone rows (blank ``hand_id``) get their own bucket.
    """
    groups: dict[str, list] = {}
    standalone: list = []
    for _i, row in df.iterrows():
        hid = str(row.get("hand_id", "") or "").strip()
        (standalone if not hid else groups.setdefault(hid, [])).append(row)

    def _seq(r) -> int:
        try:
            return int(float(str(r.get("sequence_index", "") or 0)))
        except ValueError:
            return 0

    st.caption(
        f"**{len(groups)}** full hand(s)"
        + (f" · {len(standalone)} standalone row(s)" if standalone else "")
        + ". Each card is one hand, played preflop → river — open a hand to "
        "review every leg exactly as on the single-question view."
    )

    ordered = sorted(
        groups.items(),
        key=lambda kv: min(int(_cell(r, "No") or 0) for r in kv[1]),
    )
    for _hid, rows in ordered:
        legs = sorted(rows, key=_seq)
        first = legs[0]
        hero_cards = _cell(first, "User Cards")
        board = _cell(legs[-1], "Cards on Table")
        graded = sum(1 for r in legs if reviews.get(_cell(r, "No"), {}).get("status"))
        title = (
            f"🃏 {hero_cards}  ·  {len(legs)} questions"
            + (f"  ·  board {board}" if board else "")
            + (f"  ·  {graded}/{len(legs)} graded" if graded else "")
        )
        with st.expander(title, expanded=False):
            st.caption(f"`{_hid}`  ·  {_cell(first, 'Context')}")
            for r in legs:
                _render_postflop_question_card(
                    r, df=df, csv_path=csv_path, reviews=reviews, meta=meta,
                    grouped=True,
                )
                st.divider()

    if standalone:
        with st.expander(f"Standalone rows ({len(standalone)})", expanded=False):
            for r in standalone:
                _render_postflop_question_card(
                    r, df=df, csv_path=csv_path, reviews=reviews, meta=meta,
                    grouped=True,
                )
                st.divider()


def render_postflop_review_page() -> None:
    """Review a postflop batch -- mirrors the preflop Review page.

    Browse the batch question by question; edit the explanation + difficulty
    inline (auto-saved into the CSV); grade approve / needs-review / reject;
    remove a bad question; and collect every approved question across batches.
    Reuses the generic ``admin_panel.review`` sidecar helpers (they key off any
    batch CSV + ``No``), and shows the postflop-specific facts (per-action EV,
    SPR, board texture, neutral-credit marking)."""
    st.title("Review postflop questions")
    st.caption(
        "Browse a postflop batch, read each question, edit + grade it. Edits "
        "auto-save into the CSV (no Save button)."
    )

    scan = _scan_postflop_outputs()
    if scan.empty:
        st.info(
            "No postflop batches yet. Generate one on the **Generate** page "
            "(Postflop mode) — it writes to `test_output/postflop_batches/`."
        )
        return

    options = scan["_path"].tolist()
    labels = {
        r["_path"]: f"{r['filename']}  ({r['questions']} questions · {r['modified']})"
        for _, r in scan.iterrows()
    }
    csv_path = Path(
        st.selectbox(
            "Batch",
            options=options,
            format_func=lambda p: labels[p],
            key="postflop_review_batch",
        )
    )
    df = _read_csv_cached(str(csv_path), csv_path.stat().st_mtime, as_str=True)
    if df.empty:
        st.warning("This batch CSV is empty.")
        return

    reviews = review.load_reviews(csv_path)
    meta = review.load_batch_meta(csv_path)
    nos = df["No"].tolist()
    summary = review.summarize(nos, reviews)

    m = st.columns(4)
    m[0].metric("Reviewed", f"{summary.reviewed}/{summary.total}")
    m[1].metric("Approved", summary.approved)
    m[2].metric("Needs review", summary.needs_review)
    m[3].metric("Rejected", summary.rejected)
    if meta:
        rs = meta.get("run_settings", {}) if isinstance(meta, dict) else {}
        bits = []
        if meta.get("model"):
            bits.append(f"model `{meta['model']}`")
        if rs.get("answer_style"):
            bits.append(f"style `{rs['answer_style']}`")
        if "display_in_bb" in rs:
            bits.append("amounts: bb" if rs["display_in_bb"] else "amounts: dollars")
        if bits:
            st.caption(" · ".join(bits))
        # Layer-7 lifecycle banner: only when the opt-in claim/auto-fix ran.
        ctr = meta.get("counters", {}) if isinstance(meta, dict) else {}
        if rs.get("revise_pass"):
            st.info(
                f"🛠️ **Audit & auto-fix pass ran.** "
                f"{ctr.get('revise_flagged', 0)} flagged · "
                f"{ctr.get('revise_fixed', 0)} auto-fixed · "
                f"{ctr.get('revise_discarded', 0)} discarded (original kept) · "
                f"{ctr.get('revise_unchanged', 0)} unchanged. Each card shows "
                "REWRITTEN vs ORIGINAL below its explanation."
            )
        elif rs.get("run_claim_checker"):
            st.info(
                f"🔎 **Claim checker ran (flag only).** "
                f"{ctr.get('claim_flagged_rows', 0)} of {summary.total} rows "
                "flagged. Flags show under each explanation."
            )

    st.download_button(
        "⬇️ Download this batch (CSV)",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        key="pf_review_dl",
    )
    _render_postflop_skills_explainer()

    # --- grouped play-through view (Option B: hand_id + sequence_index) ---
    # When the batch has linked full-hand sequences, offer a per-hand grouped
    # view so a whole hand is reviewable in order before it ships. The single-
    # question navigator below stays available (toggle off).
    has_hands = "hand_id" in df.columns and df["hand_id"].astype(str).str.strip().any()
    if has_hands:
        grouped = st.toggle(
            "🔗 Group by hand (play-through view)",
            value=True,
            key=f"postflop_review_grouped::{csv_path.name}",
            help=(
                "This batch has linked full-hand sequences. Show one expandable "
                "card per hand (ordered preflop → river); toggle off for the "
                "single-question navigator."
            ),
        )
        if grouped:
            _render_postflop_grouped_review(df, csv_path, reviews, meta)
            _render_postflop_approved_pool()
            return

    # --- navigation ---
    nav_key = f"postflop_review_idx::{csv_path.name}"
    idx = max(0, min(int(st.session_state.get(nav_key, 0)), len(df) - 1))
    # Nav (Prev / Jump+Go / Next) renders INSIDE the question card's form so a
    # nav click atomically carries any in-flight explanation edit -- see the
    # EDIT-LOSS INVARIANT in _render_postflop_question_card.

    row = df.iloc[idx]
    _render_postflop_question_card(
        row, df=df, csv_path=csv_path, reviews=reviews, meta=meta,
        grouped=False, nav_key=nav_key, idx=idx,
    )
    _render_postflop_approved_pool()


def _render_postflop_approved_pool() -> None:
    """Cross-batch approved-question pool + download. Shared by the single and
    grouped review views (so both end with the same approved set)."""
    st.divider()
    st.subheader("✅ Approved postflop questions (all batches)")
    approved_sources = review.collect_approved_sources(POSTFLOP_OUTPUT_DIR)
    if not approved_sources:
        st.caption(
            "No approved questions yet. Grade questions **approved** above and "
            "they collect here across postflop batches."
        )
        return
    appr_rows = [r for _c, _n, r in approved_sources]
    appr_fields = list(appr_rows[0].keys())
    st.caption(f"**{len(appr_rows)}** approved across all postflop batches.")
    st.download_button(
        "⬇️  Download approved (CSV)",
        data=review.approved_rows_to_csv(appr_fields, appr_rows),
        file_name="postflop_approved.csv",
        mime="text/csv",
        key="pf_approved_dl",
    )


def render_postflop_compare_page() -> None:
    """Run two postflop prompts (or models) on the SAME spots, judged side by side.

    The postflop analog of :func:`render_compare_page`. Postflop uses a single
    admin-editable system prompt (no PromptLibrary), so the A/B is two free-text
    prompt boxes (and/or two models). Both sides run in ONE subprocess job on
    identical, fully-deterministic spots from one ``.db`` solve (postflop spot
    selection needs no shared RNG seed), so any difference is the prompt/model,
    not luck. Reuses the generic compare/review/claim-check helpers.
    """
    import os  # noqa: PLC0415

    from pipeline.postflop.adapters.sqlite_db import discover_db_solves  # noqa: PLC0415
    from pipeline.postflop.explanation_generator import (  # noqa: PLC0415
        load_postflop_system_prompt,
    )
    from pipeline.postflop.options import (  # noqa: PLC0415
        ANSWER_STYLE_FROM_RADIO_LABEL,
    )
    from pipeline.postflop.run import (  # noqa: PLC0415
        POSTFLOP_OUTPUT_DIR as _PF_OUT,
    )
    from pipeline.postflop.run import (  # noqa: PLC0415
        compare_postflop_batches_from_db,
    )

    st.title("Compare postflop prompts (A/B)")
    st.caption(
        "Run two prompts (or two models) on the SAME spots from one solve and "
        "judge them side by side — so any difference is the prompt, not luck. "
        "Postflop spots are fully deterministic, so both sides see identical hands."
    )

    # --- job lifecycle: while running show progress; on finish stash + clear ---
    _STATE = "pf_cmp_result"
    job = jobs.get_current_job()
    if job is not None and str(job.label).startswith("PostflopCompare:"):
        with st.container(border=True):
            if job.is_active:
                _render_active_job_progress()
                st.caption("Both sides run in one job; this page updates when it finishes.")
                return
            if job.status is jobs.JobStatus.COMPLETED:
                res = job.result if isinstance(job.result, dict) else None
                if res and res.get("a_written") and res.get("b_written"):
                    st.session_state[_STATE] = {
                        k: res[k] for k in ("a_csv", "b_csv", "a_name", "b_name")
                    }
                    msg = (
                        f"✅ Comparison ready — {res['a_name']} "
                        f"({res['a_written']}) vs {res['b_name']} ({res['b_written']})."
                    )
                    if not res.get("dry_run") and (res.get("in_tokens") or res.get("out_tokens")):
                        msg += (
                            f" Tokens: {res['in_tokens']:,} in / {res['out_tokens']:,} out."
                        )
                    st.success(msg)
                else:
                    a_w = (res or {}).get("a_written", 0)
                    b_w = (res or {}).get("b_written", 0)
                    st.warning(
                        f"A side wrote {a_w} rows, B wrote {b_w} — both need rows to "
                        "compare. Adjust the prompts/filters and run again."
                    )
                jobs.clear_current_job()
            elif job.status is jobs.JobStatus.CANCELLED:
                st.warning("⛔ Comparison cancelled.")
                jobs.clear_current_job()
            else:  # FAILED
                st.error("❌ Comparison job failed.")
                with st.expander("Traceback"):
                    st.code(job.error or "(no traceback captured)")
                jobs.clear_current_job()

    # --- 1. pick a solve (reuse the Generate-page discovery) ------------------
    folder = st.text_input(
        "Solves folder", value=str(_POSTFLOP_SOLVES_DIR), key="pfcmp_solves_dir",
        help="Folder holding your `.db` postflop solves (scanned recursively).",
    )
    usable = [s for s in discover_db_solves(folder) if s.ok]
    if not usable:
        st.info(f"No readable `.db` solves found in `{folder}`.")
        return
    by_path = {s.path: s for s in usable}
    picked = st.selectbox(
        f"{len(usable)} solve(s) found",
        options=[s.path for s in usable],
        format_func=lambda p: f"{Path(p).name}   —   {by_path[p].label}",
        key="pfcmp_pick_solve",
    )
    solve_sum = by_path[picked]
    st.caption(f"`{solve_sum.label}`")

    # --- 2. the two prompts: load a saved library entry and/or edit freely ----
    _pflib = _postflop_prompt_library()
    _ensure_postflop_library_seeded(_pflib)
    _pf_entries = _pflib.list()
    active_prompt = load_postflop_system_prompt()
    for k in ("pfcmp_prompt_a", "pfcmp_prompt_b"):
        if k not in st.session_state:
            st.session_state[k] = active_prompt
    st.markdown(
        "**Prompts** — load a saved library prompt into either side (Prompt page "
        "→ Postflop to manage them), then edit freely. Both start from the ★ "
        "active postflop prompt."
    )
    if _pf_entries:
        _slugs = [e.slug for e in _pf_entries]
        _name_by = {e.slug: e.name for e in _pf_entries}
        _act = _pflib.active_slug()

        def _pf_lbl(slug: str) -> str:
            return f"{_name_by[slug]}{'  ★' if slug == _act else ''}"

        lc1, lc2 = st.columns(2)
        with lc1:
            pick_a = st.selectbox(
                "Load A from library", _slugs, format_func=_pf_lbl, key="pfcmp_lib_a"
            )
            if st.button("⬇ Load into A", key="pfcmp_loada"):
                st.session_state["pfcmp_prompt_a"] = _pflib.get_text(pick_a)
                st.rerun()
        with lc2:
            pick_b = st.selectbox(
                "Load B from library", _slugs, format_func=_pf_lbl,
                index=min(1, len(_slugs) - 1), key="pfcmp_lib_b",
            )
            if st.button("⬇ Load into B", key="pfcmp_loadb"):
                st.session_state["pfcmp_prompt_b"] = _pflib.get_text(pick_b)
                st.rerun()
    pc1, pc2 = st.columns(2)
    with pc1:
        prompt_a = st.text_area("Prompt A", key="pfcmp_prompt_a", height=260)
        if st.button("↺ Reset A to active", key="pfcmp_reset_a"):
            st.session_state["pfcmp_prompt_a"] = active_prompt
            st.rerun()
    with pc2:
        prompt_b = st.text_area("Prompt B", key="pfcmp_prompt_b", height=260)
        if st.button("↺ Reset B to active", key="pfcmp_reset_b"):
            st.session_state["pfcmp_prompt_b"] = active_prompt
            st.rerun()

    # --- 3. per-side models + spot count + filters ----------------------------
    _short = lambda lbl: lbl.split(" (")[0]  # noqa: E731
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        model_a_label = st.selectbox(
            "Model A", list(_MODEL_LABEL_TO_API), index=0,
            format_func=_short, key="pfcmp_model_a",
        )
    with mc2:
        model_b_label = st.selectbox(
            "Model B", list(_MODEL_LABEL_TO_API), index=0,
            format_func=_short, key="pfcmp_model_b",
        )
    with mc3:
        n_spots = int(st.number_input("Spots", min_value=1, max_value=50, value=6, key="pfcmp_n"))

    fc1, fc2 = st.columns(2)
    with fc1:
        heroes = st.multiselect(
            "Whose decisions", options=[solve_sum.ip_position, solve_sum.oop_position],
            default=[solve_sum.ip_position, solve_sum.oop_position], key="pfcmp_heroes",
        )
    with fc2:
        streets = st.multiselect(
            "Streets", options=["flop", "turn", "river"], default=["flop"],
            key="pfcmp_streets", help="Flop only is fastest; both sides see the same spots either way.",
        )
    diversify = st.toggle("Vary the decision types", value=True, key="pfcmp_diversify")

    oc1, oc2 = st.columns(2)
    with oc1:
        display_in_bb = st.radio(
            "Amounts", ["Big blinds", "Dollars"], index=0, horizontal=True,
            key="pfcmp_amounts",
        ) == "Big blinds"
    with oc2:
        answer_style = ANSWER_STYLE_FROM_RADIO_LABEL[
            st.radio(
                "Answer option style", list(ANSWER_STYLE_FROM_RADIO_LABEL), index=1,
                horizontal=True, key="pfcmp_style",
            )
        ]

    with st.expander("Worthiness window (advanced)"):
        freq_lo, freq_hi = st.slider(
            "Solver frequency window (%)", 50, 100, (65, 99), key="pfcmp_freq",
        )

    cmp_run_claim_checker = st.checkbox(
        "Run claim checker (Layer 7) on both sides", value=False, key="pfcmp_claim",
        help="A second LLM pass audits each explanation; its verdict shows under "
        "each spot. One extra API call per question per side.",
    )
    dry = st.toggle("Dry run", key="pfcmp_dry", help="No API calls — placeholder prose, flow check.")

    # --- 4. guards + run ------------------------------------------------------
    same_prompt = prompt_a.strip() == prompt_b.strip()
    same_model = model_a_label == model_b_label
    if same_prompt and same_model:
        st.info("Edit one prompt (or pick a different model) — identical sides generate the same thing twice.")
    elif not same_prompt and not same_model:
        st.warning("Both the prompt AND the model differ, so a verdict won't isolate the cause. Vary one at a time.")

    blocked = (same_prompt and same_model) or jobs.has_active_job() or not heroes or not streets
    if st.button("Run comparison", type="primary", disabled=blocked, key="pfcmp_run"):
        if not dry and not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY is not set. Add it to `.env`, or enable Dry run.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _PF_OUT.mkdir(parents=True, exist_ok=True)
        out_a = _PF_OUT / f"compare_{ts}_A.csv"
        out_b = _PF_OUT / f"compare_{ts}_B.csv"
        name_a = f"A · {_short(model_a_label)}" if not same_model else "Prompt A"
        name_b = f"B · {_short(model_b_label)}" if not same_model else "Prompt B"
        live = "Live" if (solve_sum.table_size or 9) >= 9 else "Online"  # noqa: PLR2004
        try:
            jobs.start_subprocess_job(
                compare_postflop_batches_from_db,
                label=f"PostflopCompare: {Path(picked).name} ({n_spots} q ×2)",
                db_path=picked,
                output_path_a=str(out_a),
                output_path_b=str(out_b),
                total_questions=n_spots,
                system_prompt_a=prompt_a,
                system_prompt_b=prompt_b,
                model_a=_MODEL_LABEL_TO_API[model_a_label],
                model_b=_MODEL_LABEL_TO_API[model_b_label],
                name_a=name_a,
                name_b=name_b,
                heroes=tuple(heroes),
                streets=tuple(streets),
                diversify=diversify,
                answer_style=answer_style,
                display_in_bb=display_in_bb,
                live_or_online=live,
                min_frequency=freq_lo / 100.0,
                max_frequency=freq_hi / 100.0,
                run_claim_checker=cmp_run_claim_checker,
                claim_checker_prompt=(
                    _load_postflop_claim_checker_prompt() if cmp_run_claim_checker else None
                ),
                dry_run=dry,
            )
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    # --- 5. results: join the two CSVs + judge side by side -------------------
    result = _render_past_comparisons(_PF_OUT, _STATE)
    if not result:
        st.caption("Run a comparison above to see the side-by-side view.")
        return
    a_csv = Path(str(result["a_csv"]))
    b_csv = Path(str(result["b_csv"]))
    if not (a_csv.is_file() and b_csv.is_file()):
        st.info("Run a comparison above to see results.")
        return

    df_a = _read_csv_cached(str(a_csv), a_csv.stat().st_mtime, as_str=True)
    df_b = _read_csv_cached(str(b_csv), b_csv.stat().st_mtime, as_str=True)
    rows_a = [{str(k): str(v) for k, v in r.items()} for r in df_a.to_dict("records")]
    rows_b = [{str(k): str(v) for k, v in r.items()} for r in df_b.to_dict("records")]
    # Postflop solver_reference is ".../<node_id>/<combo>" — its LAST segment is
    # the combo, so the default (node_id, cards) key would collide across nodes.
    # Key on the FULL ref instead (node+combo unique).
    pf_key = lambda r: r.get("solver_reference", "")  # noqa: E731
    pairs = compare.join_by_spot(rows_a, rows_b, key_fn=pf_key)
    verdicts = compare.load_verdicts(a_csv)
    counts = compare.tally(verdicts)

    st.divider()
    st.markdown(
        f"### Tally — **{result['a_name']}** {counts['A']}  ·  "
        f"**{result['b_name']}** {counts['B']}  ·  tie {counts['tie']}   "
        f"({len(verdicts)}/{len(pairs)} judged)"
    )
    if not pairs:
        st.warning("No shared spots to compare (did both runs produce rows?).")
        return

    opts = [f"{result['a_name']} better", "Tie", f"{result['b_name']} better"]
    to_verdict = {opts[0]: "A", opts[1]: "tie", opts[2]: "B"}
    from_verdict = {"A": opts[0], "tie": opts[1], "B": opts[2]}
    reviews_a = review.load_reviews(a_csv)
    reviews_b = review.load_reviews(b_csv)

    for key, row_a, row_b in pairs:
        with st.container(border=True):
            if row_a.get("Context"):
                st.caption(row_a["Context"])
            st.markdown(_md_lines(row_a.get("Question", "")))
            picks = ", ".join(
                row_a.get(f"option {i}", "") for i in (1, 2, 3, 4) if row_a.get(f"option {i}", "")
            )
            st.caption(f"Options: {picks}  ·  Correct: **{row_a.get('Correct Answer', '')}**")
            if row_a.get("action_frequencies"):
                st.markdown("**Solver frequencies:** " + row_a["action_frequencies"])
            fact_bits: list[str] = []
            for col, lbl in (
                ("archetype", "archetype"), ("Difficulty Rating", "difficulty"),
                ("spr", "SPR"), ("hero_equity", "equity"), ("range_equity", "range eq"),
                ("Position Matchup", "matchup"),
            ):
                if row_a.get(col):
                    fact_bits.append(f"{lbl}: `{row_a[col]}`")
            if fact_bits:
                st.caption(" · ".join(fact_bits))
            if row_a.get("concept_tags"):
                st.caption(f"concept tags: {row_a['concept_tags']}")
            if row_a.get("skills"):
                st.caption(f"skills: {row_a['skills']}")

            orig_a = row_a.get("Answer Explanation", "")
            orig_b = row_b.get("Answer Explanation", "")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{result['a_name']}**")
                edited_a = st.text_area(
                    "Explanation A", value=orig_a, height=320,
                    key=f"pfcmp_exp_a_{a_csv.stem}_{key}", label_visibility="collapsed",
                )
                if edited_a.strip() != orig_a.strip():
                    st.caption("✏️ Edited — finalizing A saves this text.")
                _render_claim_check_panel(row_a)
            with col_b:
                st.markdown(f"**{result['b_name']}**")
                edited_b = st.text_area(
                    "Explanation B", value=orig_b, height=320,
                    key=f"pfcmp_exp_b_{b_csv.stem}_{key}", label_visibility="collapsed",
                )
                if edited_b.strip() != orig_b.strip():
                    st.caption("✏️ Edited — finalizing B saves this text.")
                _render_claim_check_panel(row_b)

            cur = verdicts.get(key)
            idx = opts.index(from_verdict[cur]) if cur in from_verdict else None
            choice = st.radio(
                "Which is better?", opts, index=idx, horizontal=True, key=f"pfcmp_v_{key}",
            )
            if choice is not None and to_verdict[choice] != cur:
                compare.save_verdict(a_csv, key, to_verdict[choice])
                st.rerun()

            no_a, no_b = str(row_a.get("No", "")), str(row_b.get("No", ""))
            fin_a = reviews_a.get(no_a, {}).get("status") == "approved"
            fin_b = reviews_b.get(no_b, {}).get("status") == "approved"
            fcol_a, fcol_b = st.columns(2)
            if fcol_a.button("Save A to finalized", key=f"pfcmp_fin_a_{key}", disabled=fin_a, use_container_width=True):
                review.save_review(
                    a_csv, no_a, "approved", "finalized from postflop compare",
                    explanation=(edited_a if edited_a.strip() != orig_a.strip() else None),
                )
                review.remove_review(b_csv, no_b)
                st.rerun()
            if fcol_b.button("Save B to finalized", key=f"pfcmp_fin_b_{key}", disabled=fin_b, use_container_width=True):
                review.save_review(
                    b_csv, no_b, "approved", "finalized from postflop compare",
                    explanation=(edited_b if edited_b.strip() != orig_b.strip() else None),
                )
                review.remove_review(a_csv, no_a)
                st.rerun()
            if fin_a or fin_b:
                which = result["a_name"] if fin_a else result["b_name"]
                st.caption(f"✅ Saved to finalized using **{which}**.")
                if st.button("Remove from finalized", key=f"pfcmp_unfin_{key}"):
                    review.remove_review(a_csv, no_a)
                    review.remove_review(b_csv, no_b)
                    st.rerun()

    # --- download the shared finalized pool (same set as the Postflop Review) -
    st.divider()
    fin_fields, fin_rows = review.collect_approved_rows(_PF_OUT)
    if fin_rows:
        st.download_button(
            f"⬇️  Download finalized postflop questions (CSV) — {len(fin_rows)} total",
            review.approved_rows_to_csv(fin_fields, fin_rows),
            file_name="postflop_approved_all_batches.csv",
            mime="text/csv", type="primary", key="pfcmp_download_finalized",
        )


def _render_postflop_prompt_library() -> None:
    """The postflop prompt LIBRARY: create, name, edit, and switch between
    postflop Layer-6 system prompts -- the postflop analog of the preflop
    library, using the same generic PromptLibrary class. Entries live under
    ``admin_panel/prompts/postflop_library/`` (gitignored); the ★ active entry is
    mirrored into ``postflop_system.txt`` so generation (Generate, the CLI, the
    Compare prefill) reads it on the next batch -- no restart.
    """
    from pipeline.postflop.explanation_generator import (  # noqa: PLC0415
        POSTFLOP_SYSTEM_PROMPT,
    )

    lib = _postflop_prompt_library()
    _ensure_postflop_library_seeded(lib)

    def _sync() -> None:
        _sync_postflop_active_to_override(lib)

    def _autosave_text(slug: str) -> None:
        text = st.session_state.get(f"pf_lib_edit_{slug}")
        if text is not None and text != lib.get_text(slug):
            lib.update_text(slug, text)
            if slug == lib.active_slug():
                _sync()

    def _autosave_name(slug: str) -> None:
        name = str(st.session_state.get(f"pf_lib_rename_{slug}", "")).strip()
        if name and name != lib.get(slug).name:
            lib.rename(slug, name)

    def _autosave_notes(slug: str) -> None:
        notes = str(st.session_state.get(f"pf_lib_notes_{slug}", ""))
        if notes != lib.get(slug).notes:
            lib.update_notes(slug, notes)

    entries = lib.list()
    with st.expander("➕  New postflop prompt", expanded=not entries):
        new_name = st.text_input("Name", key="pf_lib_new_name")
        seed_from = st.radio(
            "Start from",
            ["Built-in default", "Copy of active prompt", "Blank"],
            horizontal=True,
            key="pf_lib_new_seed",
        )
        if st.button("Create prompt", type="primary", key="pf_lib_create"):
            if not new_name.strip():
                st.error("Give the prompt a name first.")
            else:
                if seed_from == "Built-in default":
                    seed_text = POSTFLOP_SYSTEM_PROMPT
                elif seed_from == "Copy of active prompt":
                    act = lib.active_entry()
                    seed_text = act.text if act else POSTFLOP_SYSTEM_PROMPT
                else:
                    seed_text = ""
                created = lib.create(new_name, seed_text)
                lib.set_active(created.slug)
                _sync()
                st.session_state["_pf_lib_pending"] = created.slug
                st.success(f"Created '{created.name}' and made it active.")
                st.rerun()

    entries = lib.list()
    if not entries:
        st.info("No prompts yet — create one above.")
        return

    active_slug = lib.active_slug()
    slugs = [e.slug for e in entries]
    name_by_slug = {e.slug: e.name for e in entries}
    if "_pf_lib_pending" in st.session_state:
        st.session_state["pf_lib_select"] = st.session_state.pop("_pf_lib_pending")
    if st.session_state.get("pf_lib_select") not in slugs:
        st.session_state["pf_lib_select"] = active_slug or slugs[0]

    def _label(slug: str) -> str:
        return f"{name_by_slug[slug]}{'  ★ active' if slug == active_slug else ''}"

    sel = st.selectbox("Prompt", options=slugs, format_func=_label, key="pf_lib_select")
    entry = lib.get(sel)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if sel == active_slug:
            st.success("★ Active")
        elif st.button("Set active", key="pf_lib_setactive", use_container_width=True):
            lib.set_active(sel)
            _sync()
            st.rerun()
    with c2:
        if st.button("Duplicate", key="pf_lib_dup", use_container_width=True):
            dup = lib.duplicate(sel)
            st.session_state["_pf_lib_pending"] = dup.slug
            st.rerun()
    with c3:
        if st.button(
            "Delete", key="pf_lib_del", use_container_width=True,
            disabled=len(entries) == 1,
        ):
            lib.delete(sel)
            _sync()
            st.rerun()
    with c4:
        updated = f" · updated {entry.updated_at[:10]}" if entry.updated_at else ""
        st.caption(
            f"{len(entry.text):,} chars · ~{len(entry.text) // 4:,} tokens{updated}"
        )

    m1, m2 = st.columns(2)
    with m1:
        st.text_input(
            "Name", value=entry.name, key=f"pf_lib_rename_{sel}",
            on_change=_autosave_name, args=(sel,),
        )
    with m2:
        st.text_input(
            "Notes (what you're trying)", value=entry.notes,
            key=f"pf_lib_notes_{sel}", on_change=_autosave_notes, args=(sel,),
        )

    st.text_area(
        "Postflop system prompt", value=entry.text, height=520,
        key=f"pf_lib_edit_{sel}", on_change=_autosave_text, args=(sel,),
        help="Saves automatically on blur/Enter -- no Save button.",
    )
    st.caption(
        "✓ Auto-saved. The ★ active prompt drives Generate, the CLI, and "
        "Compare's prefill (mirrored to postflop_system.txt on every edit)."
    )
    with st.expander("Built-in default (read-only reference)"):
        st.text_area(
            "default", value=POSTFLOP_SYSTEM_PROMPT, height=300,
            disabled=True, label_visibility="collapsed",
        )

    # --- preflop-ENTRY prompt (the play-through preflop leg) -----------------
    # The preflop leg of a play-through (and the standalone preflop-from-postflop
    # questions) is written by a SEPARATE prompt -- that's why it reads
    # differently from the postflop legs. It's a single editable override (no
    # library), so it lives here under the postflop prompt page.
    st.divider()
    st.subheader("Preflop-entry prompt (play-through preflop leg)")
    st.caption(
        "Used for the PREFLOP leg of a full-hand play-through and for standalone "
        "preflop-from-postflop-solve questions. Separate from the postflop system "
        "prompt above (which writes the flop/turn/river legs)."
    )
    from pipeline.postflop.preflop_entry import (  # noqa: PLC0415
        PREFLOP_ENTRY_SYSTEM_PROMPT,
    )

    _pe_key = "preflop_entry_prompt_edit"
    if _pe_key not in st.session_state:
        st.session_state[_pe_key] = _load_preflop_entry_prompt()
    edited_pe = st.text_area(
        "Preflop-entry system prompt", height=300, key=_pe_key,
    )
    pcol1, pcol2 = st.columns([1, 4])
    if pcol1.button("💾 Save preflop-entry prompt", key="pe_save"):
        _save_preflop_entry_prompt(edited_pe)
        st.success("Saved. Takes effect on the next batch.")
    if pcol2.button("↺ Reset to built-in", key="pe_reset"):
        p = _preflop_entry_prompt_path()
        if p.is_file():
            p.unlink()
        st.session_state[_pe_key] = PREFLOP_ENTRY_SYSTEM_PROMPT
        st.rerun()
    with st.expander("Built-in preflop-entry default (read-only)"):
        st.text_area(
            "pe_default", value=PREFLOP_ENTRY_SYSTEM_PROMPT, height=240,
            disabled=True, label_visibility="collapsed",
        )


# --- page: Prompt -----------------------------------------------------------
def render_prompt_page() -> None:
    """The prompt library: create, name, edit, and switch between the
    Layer 6 system prompts you're workshopping.

    Both pipelines now have a library (the Postflop radio renders the postflop
    one via :func:`_render_postflop_prompt_library`). Prompts live under
    ``admin_panel/prompts/library/`` (preflop) and ``.../postflop_library/``
    (postflop) via :class:`admin_panel.prompt_library.PromptLibrary`. The ACTIVE
    prompt is the default for new batches and is mirrored into the legacy
    ``preflop_system.txt`` / ``postflop_system.txt`` so any code path that reads
    ``load_*_system_prompt()`` stays in sync. The Generate page can run any
    library prompt per batch and tags each output with it.
    """
    st.title("Prompt library")
    st.caption(
        "Create, name, and switch between the system prompts Layer 6 sends "
        "to Claude. The ★ active prompt is the default for new batches; edits "
        "take effect on the next batch — no restart needed."
    )

    mode = st.radio(
        "Pipeline path",
        options=["Preflop (editable)", "Postflop (editable)"],
        index=0,
        horizontal=True,
        key="prompt_mode",
    )

    if mode.startswith("Postflop"):
        _render_postflop_prompt_library()
        return

    # --- Preflop mode: the prompt LIBRARY ---
    from admin_panel.prompt_library import PromptLibrary  # noqa: PLC0415
    from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
        build_preflop_system_prompt,
    )

    lib = PromptLibrary()
    lib.ensure_seeded(
        build_preflop_system_prompt, legacy_override=PREFLOP_PROMPT_OVERRIDE_PATH
    )

    def _sync_legacy_override() -> None:
        """Mirror the active prompt into the legacy single-file override so
        ``load_preflop_system_prompt()`` (any path that doesn't pass
        system_prompt explicitly) stays in sync with the library."""
        active_text = lib.active_text()
        if active_text is not None:
            PREFLOP_PROMPT_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREFLOP_PROMPT_OVERRIDE_PATH.write_text(active_text, encoding="utf-8")

    # Auto-save callbacks: name / notes / system-prompt all persist to disk
    # on every edit (blur or Enter), so nothing is ever lost by saving one
    # field and not another -- the bug where saving the name dropped unsaved
    # system-prompt edits. on_change fires on COMMIT (blur/Enter), NOT per
    # keystroke, so it's one small write per edit, no typing lag. Slugs are
    # stable across rename, so the selection never jumps. (June 2026.)
    def _autosave_prompt_text(slug: str) -> None:
        text = st.session_state.get(f"prompt_edit_{slug}")
        if text is not None and text != lib.get_text(slug):
            lib.update_text(slug, text)
            if slug == lib.active_slug():
                _sync_legacy_override()

    def _autosave_prompt_name(slug: str) -> None:
        name = str(st.session_state.get(f"rename_{slug}", "")).strip()
        if name and name != lib.get(slug).name:
            lib.rename(slug, name)

    def _autosave_prompt_notes(slug: str) -> None:
        notes = str(st.session_state.get(f"notes_{slug}", ""))
        if notes != lib.get(slug).notes:
            lib.update_notes(slug, notes)

    # --- create a new prompt ---
    entries = lib.list()
    with st.expander("➕  New prompt", expanded=not entries):
        new_name = st.text_input("Name", key="new_prompt_name")
        seed_from = st.radio(
            "Start from",
            ["Built-in default", "Copy of active prompt", "Blank"],
            horizontal=True,
            key="new_prompt_seed",
            help=(
                "Built-in default gives you the FULL editable prompt — voice "
                "rules, archetype catalog, banned phrases, output rules — to "
                "tweak. (The solver-data block and gold examples are assembled "
                "per question and aren't part of the saved prompt.) Blank is a "
                "clean canvas."
            ),
        )
        if st.button("Create prompt", type="primary", key="create_prompt_btn"):
            if not new_name.strip():
                st.error("Give the prompt a name first.")
            else:
                if seed_from == "Built-in default":
                    seed_text = build_preflop_system_prompt()
                elif seed_from == "Copy of active prompt":
                    act = lib.active_entry()
                    seed_text = act.text if act else build_preflop_system_prompt()
                else:
                    seed_text = ""
                created = lib.create(new_name, seed_text)
                lib.set_active(created.slug)
                _sync_legacy_override()
                st.session_state["_prompt_pending"] = created.slug
                st.success(f"Created '{created.name}' and made it active.")
                st.rerun()

    entries = lib.list()
    if not entries:
        st.info("No prompts yet — create one above.")
        return

    active_slug = lib.active_slug()
    slugs = [e.slug for e in entries]
    name_by_slug = {e.slug: e.name for e in entries}

    # Keep the selection valid across create / delete reruns.
    # Apply a pending selection (Create / Duplicate) BEFORE the selectbox is
    # instantiated -- Streamlit forbids writing a widget key after the widget.
    if "_prompt_pending" in st.session_state:
        st.session_state["prompt_select"] = st.session_state.pop("_prompt_pending")
    if st.session_state.get("prompt_select") not in slugs:
        st.session_state["prompt_select"] = active_slug or slugs[0]

    def _label(slug: str) -> str:
        star = "  ★ active" if slug == active_slug else ""
        return f"{name_by_slug[slug]}{star}"

    sel = st.selectbox("Prompt", options=slugs, format_func=_label, key="prompt_select")
    entry = lib.get(sel)

    # --- row of actions ---
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if sel == active_slug:
            st.success("★ Active")
        elif st.button("Set active", key="set_active_btn", use_container_width=True):
            lib.set_active(sel)
            _sync_legacy_override()
            st.rerun()
    with c2:
        if st.button("Duplicate", key="dup_btn", use_container_width=True):
            dup = lib.duplicate(sel)
            st.session_state["_prompt_pending"] = dup.slug
            st.success(f"Duplicated as '{dup.name}'.")
            st.rerun()
    with c3:
        if st.button(
            "Delete",
            key="del_btn",
            use_container_width=True,
            disabled=len(entries) == 1,
        ):
            lib.delete(sel)
            # Don't touch the widget key (already instantiated); the guard
            # above re-selects a survivor on the rerun.
            _sync_legacy_override()
            st.rerun()
    with c4:
        updated = f" · updated {entry.updated_at[:10]}" if entry.updated_at else ""
        st.caption(
            f"{len(entry.text):,} chars · ~{len(entry.text) // 4:,} tokens{updated}"
        )

    # --- rename + notes (auto-save on edit -- no Save button) ---
    m1, m2 = st.columns(2)
    with m1:
        st.text_input(
            "Name", value=entry.name, key=f"rename_{sel}",
            on_change=_autosave_prompt_name, args=(sel,),
        )
    with m2:
        st.text_input(
            "Notes (what you're trying)", value=entry.notes, key=f"notes_{sel}",
            on_change=_autosave_prompt_notes, args=(sel,),
        )

    # --- the editable prompt text (auto-saves on edit) ---
    edited = st.text_area(
        "System prompt",
        value=entry.text,
        height=520,
        key=f"prompt_edit_{sel}",
        on_change=_autosave_prompt_text, args=(sel,),
        help="Saves automatically when you click out of the box or press "
        "Enter -- no Save button, and switching fields never loses an edit.",
    )
    st.caption("✓ Auto-saved — name, notes, and the prompt all save as you edit.")

    # --- preview the FULL prompt the model receives (sample spot) ---
    with st.expander("👁  Preview the FULL prompt sent to Claude (sample spot)"):
        st.info(
            "**How the two parts fit together.** Each question is ONE API call "
            "that contains BOTH parts:\n\n"
            "1. **SYSTEM prompt** — the standing rules (the editable text "
            "above): how to write, how to frame each archetype, what's banned, "
            "the output format. **Identical for every question**, so it's "
            "prompt-cached and cheap to repeat.\n"
            "2. **USER message** — the gold examples plus the facts for ONE "
            "specific hand (the SOLVER DATA block + the ask). **Changes every "
            "question.**\n\n"
            "Only the SYSTEM prompt is saved/edited here; the USER message is "
            "built automatically by the deterministic pipeline for each hand. "
            "The SOLVER DATA block feeds `concept_tags` and the villain's range "
            "(`villain_stats.top_combos`); it does NOT feed `skills`."
        )
        sample = _preview_sample_spot()
        if sample is None:
            st.caption("(Couldn't build a sample spot — is the range pack present?)")
        else:
            from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
                build_explanation_prompt_parts,
            )

            s_facts, s_options, s_correct = sample
            parts = build_explanation_prompt_parts(
                s_facts, s_options, s_correct, system_prompt=edited
            )
            st.markdown("**1. SYSTEM prompt** (the editable text above)")
            st.code(parts["system_prompt"], language="markdown")
            st.markdown(
                "**2. USER message, part A — gold examples** (cached; identical "
                "every question):"
            )
            st.code(parts["gold_block"])
            st.markdown(
                "**2. USER message, part B — this hand**: the scenario framing, "
                "the options, the SOLVER DATA block, and the ask. This is the "
                "per-question input the software builds:"
            )
            st.code(parts["live_block"])
            st.markdown(
                "**The SOLVER DATA block on its own** — exactly the structured "
                "facts the deterministic pipeline computes and feeds the LLM. "
                "Every number and claim in the explanation must trace back to a "
                "field here:"
            )
            st.json(parts["solver_data"])

    # --- compare against the built-in default ---
    with st.expander("👁  Compare with built-in default"):
        default_prompt = build_preflop_system_prompt()
        st.caption(
            f"Built-in default: {len(default_prompt):,} chars  ·  "
            f"this prompt: {len(entry.text):,} chars  ·  "
            f"diff {len(entry.text) - len(default_prompt):+,}"
        )
        st.text_area(
            "Built-in default (read-only)",
            value=default_prompt,
            height=320,
            disabled=True,
            key=f"default_ro_{sel}",
        )

    st.divider()
    st.subheader("⚠️  Working with prompts — what to know")
    st.markdown(
        """
- **The ★ active prompt is the default for new batches.** On the Generate
  page you can also pick any prompt per run, and every batch records which
  prompt produced it.
- **Test with a dry-run first.** A typo can break the JSON output format
  and waste a batch's spend. Dry-run is free — verify shape first.
- **The built-in default encodes hard-won lessons** -- 10 voice rules,
  banned phrases, archetype framing, the May 2026 Ryan-feedback fixes.
  Treat big rewrites as research, not casual editing.
- **Iterate cheap.** Experiment on Sonnet 4.6 (the cheap tier) and only
  validate the winners on Opus 4.7 (the production model).
- **The library is gitignored** -- copy prompts you want to keep across
  machines somewhere safe.
        """
    )


# --- page: Concept Tags ----------------------------------------------------
# Hand-grouped section mapping for the 38 preflop concept tag functions
# in pipeline.preflop.concept_tags. Mirrors the module's section headers
# (the "# --- Position context (5) ---" style comments). Manually
# maintained so the order is reviewer-friendly; adding a new tag
# requires adding it to one of these lists.
_PREFLOP_TAG_SECTIONS: dict[str, tuple[str, ...]] = {
    "Position context (5)": (
        "early_position", "middle_position", "late_position",
        "small_blind", "big_blind",
    ),
    "Decision context (7)": (
        "open_decision", "facing_single_raise", "facing_3bet",
        "facing_4bet_plus", "squeeze_opportunity", "bvb_spot",
        "multiway_pot",
    ),
    "Stack depth (3)": (
        "short_stack", "standard_stack", "deep_stack",
    ),
    "Hand strength (8)": (
        "premium_pair", "medium_pair", "small_pair",
        "premium_unpaired", "suited_broadway", "suited_connector",
        "suited_ace", "unconnected_offsuit",
    ),
    "Strategy shape (5)": (
        "mixed_strategy", "near_pure_strategy", "dominant_is_aggressive",
        "dominant_is_passive", "dominant_is_fold",
    ),
    "Equity context (4)": (
        "equity_dominant", "equity_favorite", "coinflip", "dominated",
    ),
    "Blockers (3)": (
        "ace_blocker", "king_blocker", "blocks_villain_top_value",
    ),
    "Range dynamics (3)": (
        "hero_range_advantage", "villain_range_advantage",
        "roughly_equal_ranges",
    ),
}


def render_concept_tags_page() -> None:
    """Reference catalog for the 38 preflop concept tags.

    For each tag, shows: section, plain-English trigger (the function's
    docstring), and the rule source code (via ``inspect.getsource``).
    Distinct from the Skills page -- concept tags are the COMPUTATIONAL
    atoms (read by the LLM in the SOLVER DATA block and by the skill
    tagger as building blocks); skills are the user-facing labels.

    Postflop tag library (42 tags in
    ``pipeline.fact_extractor.concept_tags``) is intentionally not
    rendered here: this page focuses on what fires on preflop output,
    which is what we generate today. When postflop comes online, a
    sibling page or a path-toggle here is the natural next step.
    """
    import inspect  # noqa: PLC0415

    from pipeline.preflop import concept_tags  # noqa: PLC0415

    st.title("Concept tags — preflop catalog")
    st.caption(
        "38 deterministic Python predicates over `PreflopFacts`. Each "
        "fires a tag the LLM sees in the SOLVER DATA block and the "
        "skill tagger reads as a building block. Adding a new tag "
        "= adding a function in `pipeline/preflop/concept_tags.py` "
        "and listing it in `_TAG_REGISTRY` (the aggregator picks it "
        "up automatically)."
    )

    # Validate that every advertised section name maps to real tags
    # exposed on the module. Catches drift if a tag is renamed without
    # updating this page.
    all_listed = {
        name for tags in _PREFLOP_TAG_SECTIONS.values() for name in tags
    }
    missing_on_module = [
        n for n in all_listed if not hasattr(concept_tags, n)
    ]
    if missing_on_module:
        st.warning(
            "Section mapping is stale -- these tag names don't exist on "
            f"`pipeline.preflop.concept_tags`: {missing_on_module}. "
            "Update `_PREFLOP_TAG_SECTIONS` in this file."
        )

    # Summary metric.
    st.metric(
        "Preflop tags",
        len(all_listed),
        help="Each tag is a pure Python function -- no LLM, no "
        "judgement, fully deterministic from facts.",
    )

    st.divider()

    # Section filter.
    sections = list(_PREFLOP_TAG_SECTIONS.keys())
    section_filter = st.multiselect(
        "Filter by section",
        options=sections,
        default=[],
        placeholder="All sections",
    )
    visible_sections = (
        [s for s in sections if s in section_filter]
        if section_filter
        else sections
    )

    for section in visible_sections:
        st.subheader(section)
        for tag_name in _PREFLOP_TAG_SECTIONS[section]:
            fn = getattr(concept_tags, tag_name, None)
            if fn is None:
                continue
            docstring = (fn.__doc__ or "").strip() or "(no description)"
            with st.expander(f"`{tag_name}`  ·  _{docstring[:80]}_"):
                st.markdown(f"**Trigger.** {docstring}")
                try:
                    src = inspect.getsource(fn).strip()
                    st.markdown("**Rule source.**")
                    st.code(src, language="python")
                except (OSError, TypeError):
                    st.caption("(source not available)")


# --- page: Skills ----------------------------------------------------------
@st.cache_resource(show_spinner="Loading PLO pack…")
def _plo_pack_and_nodes(
    pack_dir: str,
) -> tuple[PloPack, tuple[PloDecisionNode, ...]]:
    """Discover the PLO pack + enumerate its nodes once per pack dir (cached)."""
    from admin_panel import plo_preview  # noqa: PLC0415

    return plo_preview.load_pack_and_nodes(Path(pack_dir))


_PLO_BATCH_DIR = REPO_ROOT / "test_output" / "plo_batches"
_PLO_DIFFICULTY_BANDS = {
    "Easy": (400, 1300),
    "Medium": (1300, 2100),
    "Hard": (2100, 3200),
    "Mixed": (400, 3200),
}
# Opus first = the default in every PLO picker (Generate + both Compare sides).
_PLO_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6"]

# --- PLO Generate settings persistence ---------------------------------------
# The page's widget state is snapshotted to disk when a batch launches and
# re-seeded on every render, so after a batch completes (and across page
# switches or panel restarts) the Generate tab still shows exactly the setup
# that batch ran with -- regenerating is one click.
_PLO_GEN_SETTINGS_PATH = _PLO_BATCH_DIR / ".plo_generate_settings.json"
_PLO_SEATS = ["LJ", "HJ", "CO", "BU", "SB", "BB"]
_PLO_STYLE_LABELS = {
    "Basic (Fold / Call / 3-bet)": "basic",
    "GTO (Always / Mostly spectrum)": "gto",
    "Auto-pick (Basic when dominant, GTO when mixed)": "auto",
}
_PLO_AMOUNT_LABELS = ["Dollars", "Big blinds"]
#: Every persisted widget key on the PLO Generate page.
_PLO_GEN_SAVED_KEYS: tuple[str, ...] = (
    "plo_clean_only",
    "plo_gen_positions",
    "plo_gen_contexts",
    "plo_gen_player_counts",
    "plo_difficulty_preset",
    "plo_gen_custom_band",
    "plo_worthiness_slider",
    "plo_exclude_ambiguous",
    "plo_min_ev_gap",
    "plo_answer_style",
    "plo_gen_count",
    "plo_gen_pin_seed",
    "plo_gen_seed_val",
    "plo_gen_amounts",
    "plo_gen_out_prefix",
    "plo_gen_model",
    "plo_gen_temperature",
    "plo_gen_compute_eq",
    "plo_gen_prompt_select",
)


def _seed_plo_generate_settings() -> None:
    """Re-seed the Generate page's widget state from the last batch's snapshot.

    Only keys ABSENT from session state are filled (a fresh session, or a
    widget unmounted by visiting another page) -- live edits always win.
    Saved values are sanitized against each widget's vocabulary and bounds,
    so a stale file (renamed model, changed options) can never crash a
    widget; anything invalid falls back to the hardcoded default.
    """
    from pipeline.plo.node_enumerator import PLO_ACTION_CONTEXTS  # noqa: PLC0415

    saved = gen_settings.load_settings(_PLO_GEN_SETTINGS_PATH)

    def _choice(v: object, options: list[str], default: str) -> str:
        return v if isinstance(v, str) and v in options else default

    def _subset(v: object, options: list, default: list) -> list:
        if not isinstance(v, list):
            return default
        return [x for x in v if x in options]  # empty = a real choice ("any")

    def _num(v: object, lo: float, hi: float, default: float, cast: type) -> object:
        try:
            x = cast(v)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return default
        return min(max(x, lo), hi)

    def _rng(
        v: object, lo: int, hi: int, default: tuple[int, int]
    ) -> tuple[int, int]:
        if isinstance(v, (list, tuple)) and len(v) == 2:  # noqa: PLR2004
            a = _num(v[0], lo, hi, default[0], int)
            b = _num(v[1], lo, hi, default[1], int)
            if isinstance(a, int) and isinstance(b, int) and a <= b:
                return (a, b)
        return default

    def _flag(v: object, default: bool) -> bool:
        return v if isinstance(v, bool) else default

    restored: dict[str, object] = {
        "plo_clean_only": _flag(saved.get("plo_clean_only"), True),
        "plo_gen_positions": _subset(saved.get("plo_gen_positions"), _PLO_SEATS, []),
        "plo_gen_contexts": _subset(
            saved.get("plo_gen_contexts"),
            list(PLO_ACTION_CONTEXTS),
            ["Opening", "Facing single raise", "Facing 3-bet"],
        ),
        "plo_gen_player_counts": _subset(
            saved.get("plo_gen_player_counts"), [1, 2, 3, 4, 5, 6], [1, 2, 3]
        ),
        "plo_difficulty_preset": _choice(
            saved.get("plo_difficulty_preset"),
            [*_PLO_DIFFICULTY_BANDS, "Custom"],
            "Mixed",
        ),
        "plo_gen_custom_band": _rng(
            saved.get("plo_gen_custom_band"), 400, 3200, (400, 3200)
        ),
        "plo_worthiness_slider": _rng(
            saved.get("plo_worthiness_slider"), 50, 100, (65, 99)
        ),
        "plo_exclude_ambiguous": _flag(saved.get("plo_exclude_ambiguous"), True),
        "plo_min_ev_gap": _num(saved.get("plo_min_ev_gap"), 0.0, 3.0, 0.0, float),
        "plo_answer_style": _choice(
            saved.get("plo_answer_style"),
            list(_PLO_STYLE_LABELS),
            "GTO (Always / Mostly spectrum)",
        ),
        "plo_gen_count": _num(saved.get("plo_gen_count"), 1, 200, 12, int),
        "plo_gen_pin_seed": _flag(saved.get("plo_gen_pin_seed"), False),
        "plo_gen_seed_val": _num(saved.get("plo_gen_seed_val"), 0, 1_000_000, 42, int),
        "plo_gen_amounts": _choice(
            saved.get("plo_gen_amounts"), _PLO_AMOUNT_LABELS, "Big blinds"
        ),
        "plo_gen_out_prefix": (
            saved["plo_gen_out_prefix"]
            if isinstance(saved.get("plo_gen_out_prefix"), str)
            else "plo_batch"
        ),
        "plo_gen_model": _choice(
            saved.get("plo_gen_model"), _PLO_MODELS, "claude-opus-4-7"
        ),
        "plo_gen_temperature": _num(
            saved.get("plo_gen_temperature"), 0.0, 1.0, 0.6, float
        ),
        "plo_gen_compute_eq": _flag(saved.get("plo_gen_compute_eq"), False),
    }
    # The prompt picker validates itself at render (a stale slug falls back
    # to the active prompt), so the saved slug is seeded as-is.
    if isinstance(saved.get("plo_gen_prompt_select"), str):
        restored["plo_gen_prompt_select"] = saved["plo_gen_prompt_select"]
    for key, value in restored.items():
        if key not in st.session_state:
            st.session_state[key] = value
_PLO_MODEL_NAMES = {
    "claude-opus-4-7": "Opus 4.7 (high fidelity)",
    "claude-sonnet-4-6": "Sonnet 4.6 (cheapest, fastest)",
}


def _render_plo_pack_loader() -> tuple[PloPack, tuple[PloDecisionNode, ...]] | None:
    pack_dir = st.text_input(
        "PLO pack folder", value="plo_ranges", help="Folder holding the `.rng` files."
    )
    try:
        return _plo_pack_and_nodes(pack_dir)
    except FileNotFoundError:
        st.error(
            f"No PLO pack (`.rng` files) found under `{pack_dir}/`. The 3.8 GB "
            "pack is gitignored, so point this at the extracted `plo_ranges/` "
            "folder on this machine."
        )
        return None


def _render_plo_difficulty_explainer() -> None:
    """Popover body: how the PLO 3-axis difficulty rating is computed."""
    from pipeline.plo.difficulty import (  # noqa: PLC0415
        W_CONCEPT,
        W_FREQ,
        W_HAND,
    )

    st.markdown(
        "PLO difficulty is a **3-axis score** computed from solver facts "
        "(never the LLM). Each axis is an *ease* in [0, 1] (0 = hardest, "
        "1 = easiest):\n\n"
        f"- **Frequency** (weight {W_FREQ:.0%}) — how dominant the correct "
        "action is. 55% = hardest, 100% = trivial.\n"
        f"- **Concept** (weight {W_CONCEPT:.0%}) — the archetype + concept "
        "tags (a clear fold is easy, a thin squeeze is hard).\n"
        f"- **Hand** (weight {W_HAND:.0%}) — hand-class strength (premiums and "
        "clear trash are easy, marginal shapes are hard).\n\n"
        f"Then `easy = {W_FREQ:.2f}·freq + {W_CONCEPT:.2f}·concept + "
        f"{W_HAND:.2f}·hand` and `difficulty = round(3000 − easy·2500)`, "
        "clipped to **400–3200**."
    )
    st.info(
        "**Why no EV-gap axis?** Hold'em blends a 4th axis for the EV gap "
        "between the best and 2nd-best action, but PLO leaves it OUT of the "
        "rating. A *worthy* spot is mixed-frequency by definition, and a spot "
        "mixes precisely because its top actions are nearly equal in EV — so "
        "across worthy PLO spots the gap is ~0 (mean ~0.06 bb) and redundant "
        "with the Frequency axis. Including it added no signal and shoved every "
        "score up ~350–500 points, which made the Easy tier unreachable (no "
        "worthy spot rated below ~1420).\n\n"
        "The gap is still computed: it powers the **min EV-gap** quality gate "
        "(drop true coinflips) and the **`easy_ev`** CSV column for analysis. "
        "**Adding it back to the rating** could be worth it if a future pack "
        "spans more EV-separated decisions, or if it's rescaled to PLO's "
        "compressed magnitude (full credit near ~0.5 bb instead of 3 bb) — the "
        "`easy_ev` column is kept so that call can be made from real data."
    )


def render_plo_generate_page() -> None:
    """Generate Pot-Limit Omaha question batches, the same way as Hold'em.

    Mirrors the NLHE Generate page: difficulty presets, a model + temperature
    choice, a free no-API preview, and a real run that writes LLM explanations
    and logs its spend to the SAME lifetime metric in the sidebar. Range-chart
    options are intentionally absent (PLO has no range display).
    """
    from admin_panel import plo_preview  # noqa: PLC0415

    st.title("PLO Generate")
    st.caption(
        "Pot-Limit Omaha questions, generated the same way as Hold'em. Options, "
        "difficulty, tags and skills are deterministic from the solver; only the "
        "written explanation uses the LLM. Spend tallies in the sidebar lifetime "
        "metric, same as Hold'em."
    )

    # Restore the last batch's settings into any widget whose state is gone
    # (fresh session, or a visit to another page unmounted it) -- so the tab
    # always shows the setup the last batch ran with.
    _seed_plo_generate_settings()

    loaded = _render_plo_pack_loader()
    if loaded is None:
        return
    pack, nodes = loaded
    st.success(f"Loaded **{len(nodes):,}** decision nodes from `{pack.label}`.")

    from pipeline.plo.node_enumerator import (  # noqa: PLC0415
        PLO_ACTION_CONTEXTS,
        plo_active_player_count,
        plo_node_action_context,
    )

    # --- 1. Hero context: position + action faced + players in pot ---
    st.subheader("1. Hero context")
    clean_only = st.toggle(
        "🧹 Clean lines only (recommended)",
        key="plo_clean_only",
        help="ON: restrict to the solver's CONVERGED lines -- opens, "
        "single-raised pots, and heads-up / 3-way 3-bet pots (<=2 raises, <=3 "
        "players). OFF: also include Monker's deep-multiway 4-bet+/jam tail, "
        "which is largely UNCONVERGED (absurd EV gaps, inverted ranges like AA "
        "folding a jam). Leave ON unless you specifically want the wild lines.",
    )
    hc1, hc2 = st.columns(2)
    with hc1:
        positions = st.multiselect(
            "Hero positions (blank = any)",
            options=_PLO_SEATS,
            key="plo_gen_positions",
            # Display the NLHE/app seat names (the pack's internal codes are
            # LJ/BU; everything player-facing says UTG/BTN).
            format_func=lambda s: {"LJ": "UTG", "BU": "BTN"}.get(s, s),
            help="Which seats hero is in. Empty = all positions. "
            "UTG is the pack's 'Lojack' seat (same chair, two names).",
        )
    with hc2:
        action_contexts = st.multiselect(
            "Action faced",
            options=list(PLO_ACTION_CONTEXTS),
            key="plo_gen_contexts",
            help=ACTION_FACED_HELP + " (With 'Clean lines only' on, the 4-bet+ "
            "tail stays excluded even if selected.)",
        )
        player_counts = st.multiselect(
            "Players in the pot",
            options=[1, 2, 3, 4, 5, 6],
            key="plo_gen_player_counts",
            format_func=lambda n: (
                "1 (open)" if n == 1 else "2 (heads-up)" if n == 2 else f"{n}-way"
            ),
            help="How many players are still in at hero's decision. (With "
            "'Clean lines only' on, 4+ way stays excluded even if selected.)",
        )

    # Live count of matching nodes (filenames only -- cheap), like Hold'em.
    # The clean-lines toggle caps raises (<=2, i.e. not 'Facing 4-bet+') and
    # players (<=3), matching the max_prior_raises / max_active_players the
    # batch + preview apply below.
    _ctx = set(action_contexts) if action_contexts else None
    _pc = set(player_counts) if player_counts else None
    _pos = set(positions) if positions else None
    _matching = sum(
        1
        for n in nodes
        if (_pos is None or n.actor in _pos)
        and (_ctx is None or plo_node_action_context(n) in _ctx)
        and (_pc is None or plo_active_player_count(n) in _pc)
        and (
            not clean_only
            or (
                plo_node_action_context(n) != "Facing 4-bet+"
                and plo_active_player_count(n) <= 3  # noqa: PLR2004
            )
        )
    )
    st.caption(
        f"**{_matching:,}** decision nodes match these filters "
        f"(of {len(nodes):,} total)."
    )
    if not clean_only:
        st.warning(
            "Clean lines OFF: this includes Monker's largely UNCONVERGED "
            "deep-multiway 4-bet+/jam tail (absurd EV gaps, inverted ranges). "
            "Good for exploration, not for production questions."
        )

    st.divider()

    # --- 2. Difficulty (same 4-axis rating + gates as Hold'em) ---
    st.subheader("2. Difficulty")
    with st.popover("ℹ️  How is the Difficulty Rating calculated?"):
        _render_plo_difficulty_explainer()
    preset = st.radio(
        "Preset",
        options=[*_PLO_DIFFICULTY_BANDS, "Custom"],
        horizontal=True,
        key="plo_difficulty_preset",
    )
    if preset == "Custom":
        lo, hi = st.slider(
            "Difficulty rating band", 400, 3200, step=50, key="plo_gen_custom_band"
        )
    else:
        lo, hi = _PLO_DIFFICULTY_BANDS[preset]
        st.caption(f"Difficulty band: **{lo}–{hi}** (computed 4-axis rating).")

    with st.expander(
        "Advanced filters (worthiness window · EV-gap gate)", expanded=True
    ):
        st.caption(
            "The frequency window gates whether a decision is teachable at all "
            "(the 55-95% sweet spot). The EV-gap gate drops near-coinflip spots."
        )
        freq_low, freq_high = st.slider(
            "Solver frequency worthiness window (%)",
            min_value=50,
            max_value=100,
            key="plo_worthiness_slider",
            help="Below 65% = no clear best answer; 100% = trivial.",
        )
        exclude_ambiguous = st.checkbox(
            "Exclude ambiguous 90-95% band (recommended)",
            key="plo_exclude_ambiguous",
            help="Spots at 90-95% read as 'mostly' but sit just under the 95% "
            "'always' line, so the right read can still be marked wrong. "
            "On = caps the effective ceiling at 90%.",
        )
        min_ev_gap = st.slider(
            "Minimum EV gap (bb) — 0 = off",
            min_value=0.0,
            max_value=3.0,
            step=0.05,
            key="plo_min_ev_gap",
            help="Drops spots whose EV gap to the 2nd-best action is below "
            "this. PLO has a real EV gap on every spot, raises included.",
        )
    _max_freq = freq_high / 100.0
    _ev_txt = "off" if min_ev_gap == 0.0 else f"≥ {min_ev_gap:.2f} bb"
    _band_note = (
        "  ·  90-95% band excluded"
        if exclude_ambiguous and freq_high > 90  # noqa: PLR2004
        else ""
    )
    st.info(
        f"**Numbers in effect** — difficulty **{lo}–{hi}**  ·  worthiness "
        f"**{freq_low}–{freq_high}%**{_band_note}  ·  EV-gap gate "
        f"**{_ev_txt}**."
    )

    st.divider()

    # --- 3. Answer option style (same as Hold'em; Sizing N/A for pot-limit) ---
    st.subheader("3. Answer option style")
    style = _PLO_STYLE_LABELS[
        st.radio(
            "Style",
            options=list(_PLO_STYLE_LABELS),
            key="plo_answer_style",
            help="**Basic** = bare action labels. **GTO** = the Always/Mostly "
            "spectrum that surfaces mixed strategies. **Auto-pick** = Basic for "
            "dominant-action spots, GTO for mixed. (There is no Sizing style: "
            "every PLO raise is pot-sized.)",
        )
    ]

    st.divider()

    # --- 4. Batch size + output ---
    st.subheader("4. Batch size + output")
    bo1, bo2, bo3 = st.columns(3)
    count = bo1.number_input(
        "Questions",
        min_value=1,
        max_value=200,
        step=1,
        key="plo_gen_count",
        help="How many questions to generate (spread across matching nodes).",
    )
    # Fresh spots every run by default. The old bare "Random seed" input
    # defaulted to 0 -- a FIXED seed, so every batch reproduced the identical
    # spots (same nodes, same hands, same order). Pin only for a repeatable
    # test set (e.g. prompt comparisons).
    _pin_plo_seed = bo2.toggle(
        "Pin a fixed test set",
        key="plo_gen_pin_seed",
        help="Off (default): every batch draws fresh random spots. On: the "
        "seed below reproduces the identical spots each run — useful when "
        "comparing prompts on the same hands.",
    )
    _plo_seed_input = bo2.number_input(
        "Test-set seed",
        min_value=0,
        max_value=1_000_000,
        step=1,
        key="plo_gen_seed_val",
        disabled=not _pin_plo_seed,
    )
    seed: int | None = int(_plo_seed_input) if _pin_plo_seed else None
    display_in_bb = (
        bo3.radio(
            "Amounts",
            options=_PLO_AMOUNT_LABELS,
            horizontal=True,
            key="plo_gen_amounts",
        )
        == "Big blinds"
    )
    out_prefix = st.text_input(
        "Output filename (prefix)",
        key="plo_gen_out_prefix",
        help="A timestamp is appended; every batch lands in its own file.",
    )

    st.divider()

    # --- 5. Model + API settings ---
    st.subheader("5. Model + API settings")
    ms1, ms2 = st.columns(2)
    model = ms1.selectbox(
        "Model",
        options=_PLO_MODELS,
        key="plo_gen_model",
        format_func=lambda m: _PLO_MODEL_NAMES.get(m, m),
    )
    temperature = ms2.slider(
        "Temperature",
        0.0,
        1.0,
        step=0.05,
        key="plo_gen_temperature",
        help="Higher = more varied prose. 0.6 is a good start with no "
        "examples. (Opus ignores temperature; it affects Sonnet.)",
    )
    compute_eq = st.checkbox(
        "Compute hand equity for the explanation (~1s/spot; real generate only)",
        key="plo_gen_compute_eq",
        help="Off (default): no equity numbers reach the LLM, so explanations "
        "can't cite percentages — in PLO equities run close and the numbers "
        "read as noise. On = the LLM gets equity numbers, at ~1s/spot (PLO "
        "equity is ~60x heavier than Hold'em). The preview is always "
        "equity-off regardless.",
    )
    # Rough per-question estimates by model tier.
    _cost_per_q = 0.15 if "opus" in model else 0.08
    st.info(
        f"**Estimated**: {int(count)} questions · "
        f"~${int(count) * _cost_per_q:.2f} · {_matching:,} nodes available"
    )

    st.divider()

    # --- 6. Prompt (pick which library prompt this batch runs on) ---
    st.subheader("6. Prompt")
    from pipeline.plo.explanation_generator import (  # noqa: PLC0415
        build_plo_system_prompt,
    )

    _plib = _plo_prompt_library()
    _pentries = _plib.list()
    _pactive = _plib.active_slug()
    plo_prompt_text: str | None = None
    plo_prompt_name = ""
    if _pentries:
        _pslugs = [e.slug for e in _pentries]
        _pnames = {e.slug: e.name for e in _pentries}
        if st.session_state.get("plo_gen_prompt_select") not in _pslugs:
            st.session_state["plo_gen_prompt_select"] = _pactive or _pslugs[0]
        _chosen = st.selectbox(
            "Run this batch with prompt",
            options=_pslugs,
            format_func=lambda s: (
                f"{_pnames[s]}  ★ active" if s == _pactive else _pnames[s]
            ),
            key="plo_gen_prompt_select",
            help="Defaults to the ★ active prompt. Create / edit prompts on "
            "the **PLO Prompt** page.",
        )
        _pe = _plib.get(_chosen)
        plo_prompt_text = _pe.text
        plo_prompt_name = _pe.name
        _na = "" if _chosen == _pactive else "  ·  (not the active prompt)"
        st.caption(f"**{plo_prompt_name}** · {len(plo_prompt_text):,} chars{_na}")
    with st.expander("🔍 View this prompt (exactly what the LLM is told)"):
        st.code(plo_prompt_text or build_plo_system_prompt(), language="markdown")

    st.divider()
    g1, g2 = st.columns(2)
    preview_clicked = g1.button("🎲 Preview spots (no API, free)")
    generate_clicked = g2.button("✍️ Generate with explanations (uses API)", type="primary")

    # Last completed batch, re-rendered after the post-generate rerun (the rerun
    # lets the sidebar lifetime-spend pick up the new log entry).
    _done = st.session_state.get("plo_gen_done")
    if _done and not generate_clicked and not preview_clicked:
        _dp = Path(_done["path"])
        st.success(
            f"Generated **{_done['written']}/{_done['requested']}** questions to "
            f"`{_dp.name}` ({_done['explanations']} explanations)."
        )
        st.info(
            f"💰 This batch: **{usage.format_cost(_done['cost'])}** "
            f"({_done['out_tokens']:,} output tokens). Tallied in the sidebar "
            "lifetime spend."
        )
        if _done["failed"]:
            st.warning(
                f"{_done['failed']} explanation(s) failed — those questions "
                "were dropped from the CSV and backfilled with other spots "
                "where possible."
            )
            _reasons = _done.get("failure_reasons") or []
            if _reasons:
                with st.expander("Why did they fail?"):
                    for _r in _reasons:
                        st.markdown(f"- {_r}")
                    st.caption(
                        "Most are the Layer 6 validators (e.g. a card-fabrication "
                        "guard rejecting a card not in the hand, or a banned em "
                        "dash / semicolon) firing on both the attempt and its one "
                        "retry. Your settings are kept — regenerate any time, or "
                        "edit the prompt to address the cause."
                    )
        if _done["shortfall"]:
            st.warning(
                f"{_done['shortfall']} short of {_done['requested']} "
                f"({_done['difficulty_filtered']} difficulty-filtered, "
                f"{_done['ev_filtered']} EV-gap-filtered). Widen the band, "
                "positions, action contexts, or worthiness window."
            )
        if _dp.is_file():
            st.download_button(
                "Download CSV", _dp.read_bytes(), file_name=_dp.name, mime="text/csv"
            )
        st.caption("Grade and edit these on the **PLO Review** page.")

    if generate_clicked:
        import os  # noqa: PLC0415

        from pipeline.plo.batch import generate_plo_batch  # noqa: PLC0415

        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error(
                "No `ANTHROPIC_API_KEY` found. It's loaded from `.env` the same "
                "way as Hold'em, so set it there and restart the panel."
            )
            return
        # Snapshot THIS batch's settings so the page re-seeds them after the
        # run (and across page switches / panel restarts) -- regenerating
        # with the same setup is one click.
        gen_settings.save_settings(
            _PLO_GEN_SETTINGS_PATH,
            {k: st.session_state.get(k) for k in _PLO_GEN_SAVED_KEYS},
        )
        _PLO_BATCH_DIR.mkdir(parents=True, exist_ok=True)
        _stem = (out_prefix or "plo_batch").removesuffix(".csv").strip() or "plo_batch"
        out_path = _PLO_BATCH_DIR / f"{_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        acc = {"in": 0, "out": 0, "cc": 0, "cr": 0}
        model_seen = [model]

        def _usage_cb(mdl: str, in_t: int, out_t: int, cc: int, cr: int) -> None:
            acc["in"] += in_t
            acc["out"] += out_t
            acc["cc"] += cc
            acc["cr"] += cr
            model_seen[0] = mdl

        with st.spinner(f"Generating {int(count)} PLO questions with {model}…"):
            result = generate_plo_batch(
                pack,
                output_path=out_path,
                total_questions=int(count),
                seed=seed,
                hero_positions=positions or None,
                action_contexts=action_contexts or None,
                player_counts=player_counts or None,
                max_prior_raises=2 if clean_only else None,
                max_active_players=3 if clean_only else None,
                min_frequency=freq_low / 100.0,
                max_frequency=_max_freq,
                exclude_ambiguous_band=exclude_ambiguous,
                min_ev_gap_bb=(None if min_ev_gap == 0.0 else float(min_ev_gap)),
                min_difficulty=lo,
                max_difficulty=hi,
                compute_equity=compute_eq,
                answer_style=style,
                display_in_bb=display_in_bb,
                generate_explanations=True,
                explanation_model=model,
                explanation_temperature=temperature,
                explanation_system_prompt=plo_prompt_text,
                usage_callback=_usage_cb,
            )
        cost = usage.compute_cost_usd(
            model=model_seen[0],
            input_tokens=acc["in"],
            output_tokens=acc["out"],
            cache_creation_tokens=acc["cc"],
            cache_read_tokens=acc["cr"],
        )
        usage.append_log_entry(
            USAGE_LOG_PATH,
            model=model_seen[0],
            input_tokens=acc["in"],
            output_tokens=acc["out"],
            cache_creation_tokens=acc["cc"],
            cache_read_tokens=acc["cr"],
            cost_usd=cost,
            questions_written=result.questions_written,
            output_filename=out_path.name,
        )
        st.session_state["plo_gen_done"] = {
            "path": str(out_path),
            "cost": cost,
            "out_tokens": acc["out"],
            "written": result.questions_written,
            "requested": int(count),
            "explanations": result.explanations_written,
            "failed": result.explanations_failed,
            "failure_reasons": list(result.explanation_failure_reasons),
            "shortfall": result.shortfall,
            "difficulty_filtered": result.difficulty_filtered_out,
            "ev_filtered": result.ev_gap_filtered_out,
        }
        # So the PLO Review page auto-selects this batch when you switch to it.
        st.session_state["_plo_review_jump"] = out_path.name
        # Rerun so the sidebar's lifetime-spend metric -- rendered BEFORE this
        # page on every run -- re-reads the log entry we just appended. Without
        # it the new spend wouldn't show until the next interaction. The result
        # is re-rendered from session_state above the buttons.
        st.rerun()

    if not preview_clicked:
        return
    with st.spinner("Sampling worthy PLO spots… (reads range files; equity off for speed)"):
        rows = plo_preview.build_preview_rows(
            pack,
            nodes,
            count=int(count),
            seed=seed,
            hero_positions=positions or None,
            action_contexts=action_contexts or None,
            player_counts=player_counts or None,
            max_prior_raises=2 if clean_only else None,
            max_active_players=3 if clean_only else None,
            min_frequency=freq_low / 100.0,
            max_frequency=_max_freq,
            exclude_ambiguous_band=exclude_ambiguous,
            min_ev_gap_bb=(None if min_ev_gap == 0.0 else float(min_ev_gap)),
            compute_equity=False,
            answer_style=style,
            display_in_bb=display_in_bb,
        )
    if not rows:
        st.warning(
            "No worthy spots with those filters. Widen the positions, action "
            "contexts, worthiness window, or try another seed."
        )
        return
    st.caption(
        f"{len(rows)} worthy spots (no explanations, free preview). The "
        "action line below is the REAL Question text the CSV will carry."
    )
    for i, r in enumerate(rows, start=1):
        header = f"#{i}  {r.cards}  ·  {r.position} ({r.relative_position})"
        if r.archetype:
            header += f"  ·  {r.archetype}"
        with st.expander(header, expanded=True):
            st.markdown(f"**{r.action_line}**")
            st.markdown(
                "Options:  "
                + "  ·  ".join(
                    f"✅ **{o}**" if o == r.correct_answer else o for o in r.options
                )
            )
            if r.action_frequencies:
                st.caption(
                    "**Solver mix:** "
                    + ", ".join(
                        f"{lbl} {freq:.0%}" for lbl, freq in r.action_frequencies
                    )
                )
            mc = st.columns(3)
            mc[0].metric("Difficulty", r.difficulty)
            mc[1].metric("Top freq", f"{r.dominant_freq:.0%}", help=r.dominant_action)
            mc[2].metric(
                "EV gap",
                f"{r.ev_gap_bb:.2f} bb" if r.ev_gap_bb is not None else "n/a",
            )
            st.caption("**Skills:** " + (", ".join(r.skills) or "none"))
            st.caption("**Concept tags:** " + (", ".join(r.concept_tags) or "none"))


def _plo_batch_label(name: str) -> str:
    """A readable picker label for a PLO batch file: ``<prefix> · <date time>``.

    Batches are saved as ``<prefix>_YYYYMMDD_HHMMSS.csv`` (see the PLO Generate
    page), so the creation timestamp is in the filename -- surface it so batches
    are easy to tell apart. Falls back to the file's modified time if the name
    has no parseable stamp.
    """
    import re  # noqa: PLC0415

    stem = name[:-4] if name.endswith(".csv") else name
    m = re.search(r"_(\d{8})_(\d{6})$", stem)
    if m:
        try:
            when = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            prefix = stem[: m.start()] or stem
            return f"{prefix} · {when:%Y-%m-%d %H:%M:%S}"
        except ValueError:
            pass
    try:
        mtime = (_PLO_BATCH_DIR / name).stat().st_mtime
        return f"{name} · {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M}"
    except OSError:
        return name


def render_plo_review_page() -> None:
    """Grade, edit, and prune PLO question batches (mirrors the NLHE Review page).

    Reuses :mod:`admin_panel.review` (the grade sidecar + in-place CSV edits are
    game-agnostic). No range-chart button, since PLO has no range display.
    """
    import csv as _csv  # noqa: PLC0415

    st.title("PLO Review")
    # ALL batch CSVs (any filename prefix the user chose on Generate), newest
    # first -- but NOT the Compare A/B artifacts (compare_*.csv), which are
    # graded on the Compare page. Globbing "plo_*.csv" used to hide any batch
    # the user named with a custom prefix.
    csvs = (
        sorted(
            (
                p
                for p in _PLO_BATCH_DIR.glob("*.csv")
                if not p.name.startswith("compare_")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if _PLO_BATCH_DIR.exists()
        else []
    )
    if not csvs:
        st.info("No PLO batches yet. Generate one on the **PLO Generate** page.")
        return

    # Jump to a just-generated batch (set on the Generate page) BEFORE the
    # selectbox is instantiated. One-shot: pop it so normal selection persists.
    names = [p.name for p in csvs]
    jump = st.session_state.pop("_plo_review_jump", None)
    if jump in names:
        st.session_state["plo_review_batch"] = jump

    # Pick by FILENAME (not position): any edit bumps mtime and reorders the list.
    # The label shows each batch's creation date/time so they're easy to tell apart.
    pick = st.selectbox(
        "Batch", options=names, key="plo_review_batch", format_func=_plo_batch_label
    )
    csv_path = _PLO_BATCH_DIR / pick
    with csv_path.open(encoding="utf-8-sig") as handle:
        questions = list(_csv.DictReader(handle))
    st.caption(f"🕒 {_plo_batch_label(pick)}  ·  {len(questions)} questions  ·  {pick}")
    if not questions:
        st.warning("That batch is empty.")
        return

    reviews = review.load_reviews(csv_path)
    summary = review.summarize([q.get("No") for q in questions], reviews)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Questions", summary.total)
    s2.metric("Approved", summary.approved)
    s3.metric("Needs review", summary.needs_review)
    s4.metric("Rejected", summary.rejected)
    if summary.quality_pct is not None:
        st.caption(f"Approved share of decided grades: **{summary.quality_pct:.0f}%**.")
    st.caption(
        "Edit explanations and difficulty inline below (they auto-save to the "
        "CSV). The **Download this batch** button is at the bottom and always "
        "reflects your latest edits."
    )
    st.divider()

    grade_opts = ["ungraded", "approved", "needs_review", "rejected"]
    for q in questions:
        no = str(q.get("No"))
        status = reviews.get(no, {}).get("status", "ungraded")
        badge = {"approved": "✅", "needs_review": "🟠", "rejected": "❌"}.get(status, "·")
        label = (
            f"{badge} #{no}  {q.get('User Cards', '')}  ·  "
            f"{q.get('archetype', '')}  ·  diff {q.get('Difficulty Rating', '')}"
        )
        with st.expander(label, expanded=True):
            st.markdown(f"**Context:** {q.get('Context', '')}")
            st.markdown(f"**Question:** {q.get('Question', '')}")
            opts = [q.get(f"option {i}", "") for i in range(1, 5)]
            st.markdown(
                "Options:  "
                + "  ·  ".join(
                    f"✅ **{o}**" if o == q.get("Correct Answer") else o
                    for o in opts if o
                )
            )
            freqs = q.get("action_frequencies", "")
            if freqs:
                st.caption(f"**Solver frequencies:** {freqs}")
            new_expl = st.text_area(
                "Answer Explanation (auto-saves)",
                value=q.get("Answer Explanation", ""),
                key=f"plo_expl_{pick}_{no}",
                height=320,
            )
            if new_expl != q.get("Answer Explanation", ""):
                review.update_explanation(csv_path, no, new_expl)
                st.toast(f"Saved #{no} explanation")
            st.caption("**Skills:** " + (q.get("skills", "") or "none"))
            st.caption("**Concept tags:** " + (q.get("concept_tags", "") or "none"))

            gcol, dcol, rcol = st.columns([3, 1, 1])
            choice = gcol.radio(
                "Grade",
                options=grade_opts,
                index=grade_opts.index(status) if status in grade_opts else 0,
                key=f"plo_grade_{pick}_{no}",
                horizontal=True,
            )
            if choice != status:
                if choice == "ungraded":
                    review.remove_review(csv_path, no)
                else:
                    review.save_review(csv_path, no, choice, "")
                st.rerun()
            new_diff = dcol.text_input(
                "Difficulty", value=q.get("Difficulty Rating", ""), key=f"plo_diff_{pick}_{no}"
            )
            if new_diff != q.get("Difficulty Rating", "") and new_diff.strip().isdigit():
                review.update_difficulty(csv_path, no, new_diff.strip())
                st.toast(f"Saved #{no} difficulty")
            if rcol.button("🗑 Remove", key=f"plo_rm_{pick}_{no}"):
                review.remove_question(csv_path, no)
                st.rerun()

    # Download AFTER the per-question loop: every inline edit above writes
    # straight back to the CSV (review.update_explanation / update_difficulty),
    # so reading the bytes here -- past all those writes in this same rerun --
    # guarantees the downloaded file carries your latest edits.
    st.divider()
    st.download_button(
        "⬇️  Download this batch (with your edits)",
        csv_path.read_bytes(),
        file_name=pick,
        mime="text/csv",
        type="primary",
    )

    # --- cross-batch approved pool -----------------------------------------
    # Every question graded "approved", gathered from EVERY batch's sidecar
    # into one downloadable set. Derived live from the grades (not a separate
    # stored file), so it's always in sync: approving adds, un-approving drops.
    st.divider()
    st.subheader("✅ Approved questions (all batches)")
    approved_sources = review.collect_approved_sources(_PLO_BATCH_DIR)
    if not approved_sources:
        st.caption(
            "No approved questions yet. Grade questions **approved** above (or "
            "finalize on the Compare page) and they collect here across batches."
        )
    else:
        appr_rows = [row for _csv, _no, row in approved_sources]
        appr_fields = list(appr_rows[0].keys())  # DictReader preserves column order
        st.caption(
            f"**{len(appr_rows)}** approved across all batches (deduped by spot). "
            "Updates live as you grade."
        )
        dcol, ccol = st.columns([3, 2])
        dcol.download_button(
            "⬇️  Download approved (CSV)",
            review.approved_rows_to_csv(appr_fields, appr_rows),
            file_name="plo_approved_all_batches.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
        # Clear-all is destructive (un-approves everything), so confirm in 2 steps.
        if st.session_state.get("plo_confirm_clear_approved"):
            if ccol.button(
                f"⚠️ Confirm: clear all {len(appr_rows)}",
                key="plo_clear_approved_confirm",
                use_container_width=True,
            ):
                n = review.clear_all_approved(_PLO_BATCH_DIR)
                # The open batch's grade radios still hold the OLD grade in
                # widget session state; left alone, the very next rerun sees
                # radio != sidecar and re-saves "approved" -- the clear that
                # never sticks. Dropping the keys re-initializes every radio
                # from the sidecar (the source of truth).
                for k in [
                    k for k in st.session_state if str(k).startswith("plo_grade_")
                ]:
                    del st.session_state[k]
                st.session_state["plo_confirm_clear_approved"] = False
                st.toast(f"Cleared {n} approved question(s)")
                st.rerun()
        elif ccol.button(
            "🧹 Clear all approved", key="plo_clear_approved", use_container_width=True
        ):
            st.session_state["plo_confirm_clear_approved"] = True
            st.rerun()

        with st.expander("🗑  Remove individual questions"):
            for csv_path, no, row in approved_sources:
                rcol, xcol = st.columns([10, 1])
                rcol.markdown(
                    f"**{row.get('User Cards', '')}**  ·  "
                    f"{row.get('Correct Answer', '')}  ·  "
                    f"`{row.get('archetype', '')}`  ·  diff "
                    f"{row.get('Difficulty Rating', '')}"
                )
                if xcol.button(
                    "🗑", key=f"plo_appr_del_{csv_path.name}_{no}", help="Un-approve"
                ):
                    review.remove_review(csv_path, no)
                    # If this row belongs to the batch open above, its grade
                    # radio would re-save "approved" on rerun -- drop the
                    # widget state so it re-reads the sidecar instead.
                    st.session_state.pop(f"plo_grade_{csv_path.name}_{no}", None)
                    st.rerun()


def _plo_override_path() -> Path:
    """The persistent 'genuine default' file the active prompt syncs to."""
    from admin_panel.prompt_library import PROMPTS_DIR  # noqa: PLC0415

    return PROMPTS_DIR / "plo_system.txt"


def _plo_default_prompt_text() -> str:
    """The GENUINE PLO default: the saved override (your edits to the default)
    if present, otherwise the factory prompt assembled from code. This is what
    new library entries seed from."""
    from pipeline.plo.explanation_generator import (  # noqa: PLC0415
        build_plo_system_prompt,
    )

    path = _plo_override_path()
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if text.strip():
            return text
    return build_plo_system_prompt()


def _plo_sync_default(text: str | None) -> None:
    """Write the active prompt out as the genuine-default override, so editing
    the built-in default actually updates the default everything resolves to
    (and survives even if the library is wiped). No-op on empty text."""
    if not text:
        return
    path = _plo_override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plo_prompt_library() -> PromptLibrary:
    """The PLO prompt library, seeded from the genuine default (override-or-code)."""
    from admin_panel.prompt_library import PROMPTS_DIR, PromptLibrary  # noqa: PLC0415

    lib = PromptLibrary(base_dir=PROMPTS_DIR / "plo_library")
    lib.ensure_seeded(_plo_default_prompt_text)
    return lib


@st.cache_resource(show_spinner="Building a sample PLO spot…")
def _plo_preview_sample_spot(
    pack_dir: str = "plo_ranges",
) -> tuple[PloFacts, list[str], str] | None:
    """A representative ``(facts, options, correct)`` for the prompt preview.

    Samples one clean, worthy spot from the pack with equity computed, so the
    preview shows the same SOLVER DATA a real generate would. Cached (one ~1s
    equity sim). Returns None if the pack isn't present.
    """
    import random  # noqa: PLC0415

    from pipeline.plo.fact_extractor import extract_plo_facts  # noqa: PLC0415
    from pipeline.plo.hand_order import HAND_COUNT  # noqa: PLC0415
    from pipeline.plo.node_enumerator import (  # noqa: PLC0415
        plo_active_player_count,
        plo_node_action_context,
    )
    from pipeline.plo.options import build_options  # noqa: PLC0415
    from pipeline.plo.question_extractor import is_question_worthy  # noqa: PLC0415
    from pipeline.plo.spot_sampler import sample_plo_spot  # noqa: PLC0415

    try:
        pack, nodes = _plo_pack_and_nodes(pack_dir)
    except FileNotFoundError:
        return None
    rng = random.Random(7)
    candidates = [
        n
        for n in nodes
        if plo_node_action_context(n) in {"Facing single raise", "Facing 3-bet"}
        and plo_active_player_count(n) <= 3  # noqa: PLR2004
    ]
    rng.shuffle(candidates)
    for node in candidates[:40]:
        for idx in rng.sample(range(HAND_COUNT), k=min(300, HAND_COUNT)):
            spot = sample_plo_spot(node, idx)
            if spot.presence >= 0.5 and is_question_worthy(spot):  # noqa: PLR2004
                facts = extract_plo_facts(
                    spot, pack, compute_equity=True, rng=random.Random(7)
                )
                options, correct = build_options(facts, style="auto")
                return facts, options, correct
    return None


def render_plo_prompt_page() -> None:
    """The PLO prompt library: create, edit, and switch Layer 6 PLO prompts.

    Mirrors the NLHE Prompt page but for PLO -- the same game-agnostic
    :class:`~admin_panel.prompt_library.PromptLibrary` pointed at a separate
    ``plo_library/`` dir, seeded from ``build_plo_system_prompt()``. The ★
    active prompt is the default for new PLO batches; PLO Generate + PLO
    Compare can run any library prompt.
    """
    from pipeline.plo.explanation_generator import (  # noqa: PLC0415
        build_plo_system_prompt,
    )

    st.title("PLO Prompt library")
    st.caption(
        "Create, name, and switch between the system prompts the PLO Layer 6 "
        "sends to Claude. The ★ active prompt is the default for new PLO "
        "batches; edits take effect on the next batch, no restart needed."
    )

    lib = _plo_prompt_library()

    entries = lib.list()
    with st.expander("➕  New prompt", expanded=not entries):
        new_name = st.text_input("Name", key="plo_new_prompt_name")
        seed_from = st.radio(
            "Start from",
            ["Built-in default", "Copy of active prompt", "Blank"],
            horizontal=True,
            key="plo_new_prompt_seed",
            help="Built-in default gives you the FULL editable PLO prompt -- "
            "voice rules, archetype frames, banned phrases, output rules. (The "
            "SOLVER DATA block is assembled per question and isn't part of the "
            "saved prompt.) Blank is a clean canvas.",
        )
        if st.button("Create prompt", type="primary", key="plo_create_prompt_btn"):
            if not new_name.strip():
                st.error("Give the prompt a name first.")
            else:
                if seed_from == "Built-in default":
                    seed_text = build_plo_system_prompt()
                elif seed_from == "Copy of active prompt":
                    act = lib.active_entry()
                    seed_text = act.text if act else build_plo_system_prompt()
                else:
                    seed_text = ""
                created = lib.create(new_name, seed_text)
                lib.set_active(created.slug)
                _plo_sync_default(lib.active_text())  # new active = the default
                st.session_state["_plo_prompt_pending"] = created.slug
                st.success(f"Created '{created.name}' and made it active.")
                st.rerun()

    entries = lib.list()
    if not entries:
        st.info("No prompts yet -- create one above.")
        return

    active_slug = lib.active_slug()
    slugs = [e.slug for e in entries]
    name_by_slug = {e.slug: e.name for e in entries}
    # Apply a pending selection (set by Create / Duplicate) BEFORE the selectbox
    # is instantiated -- Streamlit forbids writing a widget's session_state key
    # after the widget exists, which is what crashed Duplicate.
    if "_plo_prompt_pending" in st.session_state:
        st.session_state["plo_prompt_select"] = st.session_state.pop("_plo_prompt_pending")
    if st.session_state.get("plo_prompt_select") not in slugs:
        st.session_state["plo_prompt_select"] = active_slug or slugs[0]

    def _label(slug: str) -> str:
        star = "  ★ active" if slug == active_slug else ""
        return f"{name_by_slug[slug]}{star}"

    sel = st.selectbox(
        "Prompt", options=slugs, format_func=_label, key="plo_prompt_select"
    )
    entry = lib.get(sel)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if sel == active_slug:
            st.success("★ Active")
        elif st.button("Set active", key="plo_set_active_btn", use_container_width=True):
            lib.set_active(sel)
            _plo_sync_default(lib.active_text())  # new active = the default
            st.rerun()
    with c2:
        if st.button("Duplicate", key="plo_dup_btn", use_container_width=True):
            dup = lib.duplicate(sel)
            st.session_state["_plo_prompt_pending"] = dup.slug
            st.rerun()
    with c3:
        if st.button(
            "Delete",
            key="plo_del_btn",
            use_container_width=True,
            disabled=len(entries) == 1,
        ):
            lib.delete(sel)
            # Don't touch the widget key here (it's already instantiated); the
            # guard above re-selects a survivor on the rerun.
            _plo_sync_default(lib.active_text())  # the active may have changed
            st.rerun()
    with c4:
        updated = f" · updated {entry.updated_at[:10]}" if entry.updated_at else ""
        st.caption(
            f"{len(entry.text):,} chars · ~{len(entry.text) // 4:,} tokens{updated}"
        )

    m1, m2 = st.columns(2)
    with m1:
        new_title = st.text_input("Rename", value=entry.name, key=f"plo_rename_{sel}")
        if st.button(
            "Save name",
            key=f"plo_renamebtn_{sel}",
            disabled=(not new_title.strip() or new_title == entry.name),
        ):
            lib.rename(sel, new_title)
            st.rerun()
    with m2:
        notes = st.text_input(
            "Notes (what you're trying)", value=entry.notes, key=f"plo_notes_{sel}"
        )
        if st.button(
            "Save notes", key=f"plo_notesbtn_{sel}", disabled=notes == entry.notes
        ):
            lib.update_notes(sel, notes)
            st.rerun()

    edited = st.text_area(
        "System prompt",
        value=entry.text,
        height=520,
        key=f"plo_prompt_edit_{sel}",
        help="Edits are session-local until you click Save prompt.",
    )
    if edited != entry.text:
        st.caption(
            f"🔵 Unsaved edits ({len(edited) - len(entry.text):+,} chars vs. saved)."
        )
    if st.button(
        "💾  Save prompt",
        type="primary",
        key=f"plo_save_{sel}",
        disabled=(edited == entry.text),
    ):
        lib.update_text(sel, edited)
        _plo_sync_default(lib.active_text())
        extra = (
            "  This is the active default, so the genuine default is now updated "
            "everywhere."
            if sel == active_slug
            else ""
        )
        st.success("✅ Saved." + extra)
        st.rerun()

    # The GENUINE default is your active entry (synced to plo_system.txt, so
    # editing + saving it updates the default everything resolves to). The
    # 'factory default' is the original prompt assembled in code -- this button
    # only reverts to it if you want to discard customisations / a stale entry.
    _factory = build_plo_system_prompt()
    _matches_factory = entry.text == _factory
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        if st.button(
            "↩️  Reset to factory default",
            key=f"plo_reset_{sel}",
            disabled=_matches_factory,
            use_container_width=True,
        ):
            lib.update_text(sel, _factory)
            _plo_sync_default(lib.active_text())
            st.success("✅ Reset to the factory default.")
            st.rerun()
    with rc2:
        if _matches_factory:
            st.caption("This entry matches the factory default (the code prompt).")
        else:
            st.caption(
                "Differs from the factory default -- your edits (kept as the "
                "genuine default), or a newer code prompt. Reset to discard."
            )

    with st.expander("👁  Preview the FULL prompt sent to Claude (sample spot)"):
        st.info(
            "**How the two parts fit together.** Each question is ONE API call "
            "that contains BOTH parts:\n\n"
            "1. **SYSTEM prompt** — the standing rules (the editable text above): "
            "how to write, how to frame each archetype, what's banned, the output "
            "format. **Identical for every question**, so it's prompt-cached and "
            "cheap to repeat.\n"
            "2. **USER message** — the facts for ONE specific hand (the SOLVER "
            "DATA block). **Changes every question.**\n\n"
            "Claude reads the SYSTEM prompt for the *style and rules*, then the "
            "USER message for *this hand's data*, and writes the explanation. "
            "Think of #1 as the coaching style-guide and #2 as the specific spot "
            "to write about. Only #1 is saved/edited here; #2 is built "
            "automatically from the solver for each hand."
        )
        sample = _plo_preview_sample_spot()
        if sample is None:
            st.caption(
                "(Couldn't build a sample spot -- is the PLO pack present under "
                "`plo_ranges/`?)"
            )
        else:
            from pipeline.plo.explanation_generator import (  # noqa: PLC0415
                build_plo_user_prompt,
            )

            s_facts, s_options, s_correct = sample
            st.markdown("**1. SYSTEM prompt** (the editable text above)")
            st.code(edited or build_plo_system_prompt(), language="markdown")
            st.markdown(
                "**2. USER message** -- the per-question SOLVER DATA block + the "
                "ask. This is where the hand's data is injected:"
            )
            st.code(build_plo_user_prompt(s_facts, s_options, s_correct))

    with st.expander("👁  Compare with factory default (the code prompt)"):
        default_prompt = build_plo_system_prompt()
        st.caption(
            f"Factory default: {len(default_prompt):,} chars  ·  this prompt: "
            f"{len(entry.text):,} chars  ·  "
            f"diff {len(entry.text) - len(default_prompt):+,}"
        )
        st.text_area(
            "Factory default (read-only)",
            value=default_prompt,
            height=320,
            disabled=True,
            key=f"plo_default_ro_{sel}",
        )

    st.divider()
    st.caption(
        "The ★ active prompt is the default on PLO Generate (you can pick any "
        "per run). Test with a free preview/dry-run first -- a typo can break "
        "the JSON output and waste a batch. The library is gitignored, so copy "
        "prompts you want to keep somewhere safe."
    )


def render_plo_compare_page() -> None:
    """Run two PLO prompts on the SAME spots and judge them side by side.

    Mirrors the NLHE Compare page: ``generate_plo_batch`` twice with the same
    seed (identical spots) at temperature 0, then join the two CSVs
    spot-by-spot (:func:`compare.join_by_spot`, keyed on ``solver_reference`` +
    ``User Cards``) and pick a winner per spot with a running tally. Reuses the
    game-agnostic :mod:`admin_panel.compare` verbatim.
    """
    import os  # noqa: PLC0415

    from pipeline.plo.batch import generate_plo_batch  # noqa: PLC0415
    from pipeline.plo.node_enumerator import PLO_ACTION_CONTEXTS  # noqa: PLC0415

    st.title("PLO Compare (A/B)")
    st.caption(
        "A/B two variants on the SAME spots (same hands, temperature 0) and "
        "judge them side by side, so any difference is the variant, not luck. "
        "Compare two prompts, or one prompt with vs without the tagged skills "
        "in the data block."
    )

    lib = _plo_prompt_library()
    entries = lib.list()
    if not entries:
        st.warning("No prompts yet -- create one on the **PLO Prompt** page.")
        return
    slugs = [e.slug for e in entries]
    names = {e.slug: e.name for e in entries}
    active = lib.active_slug()

    cmp_mode = st.radio(
        "What to compare",
        options=["Two prompts", "Skills in the data (same prompt, off vs on)"],
        key="plo_cmp_mode",
        help="'Two prompts' A/Bs two system prompts on the same spots. 'Skills "
        "in the data' runs ONE prompt twice -- the only difference is whether "
        "the tagged skills are injected into the SOLVER DATA -- so you can test "
        "head-to-head whether naming the skills helps the prose.",
    )
    run_disabled = False
    # (slug, include_skills, display_name) for each side.
    if cmp_mode.startswith("Two"):
        if len(slugs) < 2:  # noqa: PLR2004
            st.info("Create a second prompt on the **PLO Prompt** page to A/B two prompts.")
            return
        c1, c2 = st.columns(2)
        with c1:
            a_slug = st.selectbox(
                "Prompt A", slugs, index=0, format_func=lambda s: names[s], key="plo_cmp_a"
            )
        with c2:
            b_slug = st.selectbox(
                "Prompt B", slugs, index=1, format_func=lambda s: names[s], key="plo_cmp_b"
            )
        a_cfg = (a_slug, False, names[a_slug])
        b_cfg = (b_slug, False, names[b_slug])
        # NOTE: same prompt on both sides is allowed -- paired with two
        # different models below it's the model-vs-model A/B. The
        # identical-sides guard lives after the model pickers.
    else:
        one_slug = st.selectbox(
            "Prompt (used for both sides)",
            slugs,
            index=slugs.index(active) if active in slugs else 0,
            format_func=lambda s: names[s],
            key="plo_cmp_one",
        )
        a_cfg = (one_slug, False, "No skills")
        b_cfg = (one_slug, True, "With skills")
        st.caption(
            "Both sides use the same prompt. **A** omits the tagged skills from "
            "the SOLVER DATA; **B** adds them as `skills_this_spot_tests`."
        )

    # Spot filters -- applied IDENTICALLY to both prompts so the A/B stays fair.
    f1, f2 = st.columns(2)
    with f1:
        contexts = st.multiselect(
            "Action faced",
            options=list(PLO_ACTION_CONTEXTS),
            default=["Opening", "Facing single raise", "Facing 3-bet"],
            key="plo_cmp_ctx",
            help=ACTION_FACED_HELP,
        )
    with f2:
        player_counts = st.multiselect(
            "Players in the pot",
            options=[1, 2, 3, 4, 5, 6],
            default=[1, 2, 3],
            format_func=lambda n: (
                "1 (open)" if n == 1 else "2 (heads-up)" if n == 2 else f"{n}-way"
            ),
            key="plo_cmp_pc",
        )

    s1, s2, s3 = st.columns(3)
    with s1:
        n_spots = int(
            st.number_input("Spots", min_value=1, max_value=25, value=5, key="plo_cmp_n")
        )
    with s2:
        preset = st.radio(
            "Difficulty",
            options=[*_PLO_DIFFICULTY_BANDS],
            index=3,
            horizontal=True,
            key="plo_cmp_diff",
        )
    with s3:
        # Per-side models: same cost as a single-model compare (still two
        # batches) and it unlocks model-vs-model A/B -- same spots, same
        # prompt, e.g. Opus 4.7 vs Sonnet 4.6.
        _short_model = lambda m: _PLO_MODEL_NAMES.get(m, str(m)).split(" (")[0]  # noqa: E731
        model_a = st.selectbox(
            "Model A",
            options=_PLO_MODELS,
            index=0,
            format_func=_short_model,
            key="plo_cmp_model_a",
        )
        model_b = st.selectbox(
            "Model B",
            options=_PLO_MODELS,
            index=0,
            format_func=_short_model,
            key="plo_cmp_model_b",
        )
    band_low, band_high = _PLO_DIFFICULTY_BANDS[preset]

    o1, o2 = st.columns(2)
    with o1:
        display_in_bb = (
            st.radio(
                "Amounts",
                options=["Big blinds", "Dollars"],
                index=0,  # bb default: matches the data block the LLM writes from
                horizontal=True,
                key="plo_cmp_amounts",
                help="How the question + pot render. Big blinds (default) "
                "matches the units the LLM's data block uses, so the prose "
                "and the explanation agree.",
            )
            == "Big blinds"
        )
    with o2:
        _cmp_style_labels = {
            "Basic (Fold / Call / 3-bet)": "basic",
            "GTO (Always / Mostly spectrum)": "gto",
            "Auto-pick (Basic when dominant, GTO when mixed)": "auto",
        }
        answer_style = _cmp_style_labels[
            st.radio(
                "Answer option style",
                options=list(_cmp_style_labels),
                index=2,  # auto: the page's historical behavior
                key="plo_cmp_answer_style",
                help="Same styles as PLO Generate. **Basic** = bare action "
                "labels. **GTO** = the Always/Mostly spectrum. **Auto-pick** "
                "= Basic for dominant-action spots, GTO for mixed.",
            )
        ]

    seed = int(
        st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, key="plo_cmp_seed")
    )

    # Side-identity checks, now that both the prompts/skills AND the models
    # are chosen. Identical sides = pointless (same output twice); two
    # variables at once = a confounded verdict.
    _sides_differ_in_content = a_cfg[0] != b_cfg[0] or a_cfg[1] != b_cfg[1]
    if not _sides_differ_in_content and model_a == model_b:
        st.info(
            "Pick two different prompts, or two different models — identical "
            "sides would generate the same thing twice."
        )
        run_disabled = True
    elif _sides_differ_in_content and model_a != model_b:
        st.warning(
            "Both the prompt and the model differ between sides, so a verdict "
            "won't tell you which one caused the difference. For a clean test, "
            "vary one at a time."
        )

    if st.button("Run comparison", type="primary", disabled=run_disabled, key="plo_cmp_run"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY is not set. Add it to `.env`, then retry.")
            return
        try:
            pack, _nodes = _plo_pack_and_nodes("plo_ranges")
        except FileNotFoundError:
            st.error("No PLO pack under `plo_ranges/`. Load it on PLO Generate first.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _PLO_BATCH_DIR.mkdir(parents=True, exist_ok=True)
        out_a = _PLO_BATCH_DIR / f"compare_{ts}_A.csv"
        out_b = _PLO_BATCH_DIR / f"compare_{ts}_B.csv"

        def _run(
            out_path: Path, slug: str, include_skills: bool, run_model: str
        ) -> None:
            # Lifetime-spend accounting: the same accumulate-and-log pattern
            # as the PLO Generate page. Compare runs are real API spend too --
            # without this they were invisible to the sidebar's lifetime total.
            acc = {"in": 0, "out": 0, "cc": 0, "cr": 0}
            model_seen = [run_model]

            def _usage_cb(mdl: str, in_t: int, out_t: int, cc: int, cr: int) -> None:
                acc["in"] += in_t
                acc["out"] += out_t
                acc["cc"] += cc
                acc["cr"] += cr
                model_seen[0] = mdl

            result = generate_plo_batch(
                pack,
                output_path=out_path,
                total_questions=n_spots,
                seed=seed,
                action_contexts=contexts or None,
                player_counts=player_counts or None,
                max_prior_raises=None,
                max_active_players=None,
                min_difficulty=band_low,
                max_difficulty=band_high,
                compute_equity=False,
                answer_style=answer_style,
                display_in_bb=display_in_bb,
                generate_explanations=True,
                explanation_model=run_model,
                explanation_temperature=0.0,
                explanation_system_prompt=lib.get_text(slug),
                explanation_include_skills=include_skills,
                usage_callback=_usage_cb,
            )
            usage.append_log_entry(
                USAGE_LOG_PATH,
                model=model_seen[0],
                input_tokens=acc["in"],
                output_tokens=acc["out"],
                cache_creation_tokens=acc["cc"],
                cache_read_tokens=acc["cr"],
                cost_usd=usage.compute_cost_usd(
                    model=model_seen[0],
                    input_tokens=acc["in"],
                    output_tokens=acc["out"],
                    cache_creation_tokens=acc["cc"],
                    cache_read_tokens=acc["cr"],
                ),
                questions_written=result.questions_written,
                output_filename=out_path.name,
            )
            return result

        # When the models differ, bake the model into each side's label so
        # the tally + verdict buttons say exactly what they're crediting.
        _models_differ = model_a != model_b
        a_label = a_cfg[2] + (f" · {_short_model(model_a)}" if _models_differ else "")
        b_label = b_cfg[2] + (f" · {_short_model(model_b)}" if _models_differ else "")

        with st.status("Running both sides on the same spots…", expanded=True) as status:
            st.write(f"A — {a_label}")
            res_a = _run(out_a, a_cfg[0], a_cfg[1], model_a)
            st.write(f"B — {b_label}")
            res_b = _run(out_b, b_cfg[0], b_cfg[1], model_b)
            status.update(label="Comparison ready", state="complete")
        err = _finish_comparison(res_a, res_b, a_label, b_label, "plo_cmp_result")
        if err:
            st.error(err)
            st.caption(
                "Nothing was saved. Adjust the prompts / models / filters and "
                "run again."
            )
            return
        st.rerun()

    result = _render_past_comparisons(_PLO_BATCH_DIR, "plo_cmp_result")
    if not result:
        return
    a_csv = Path(str(result["a_csv"]))
    b_csv = Path(str(result["b_csv"]))
    if not (a_csv.is_file() and b_csv.is_file()):
        st.info("Run a comparison above to see results.")
        return

    df_a = _read_csv_cached(str(a_csv), a_csv.stat().st_mtime, as_str=True)
    df_b = _read_csv_cached(str(b_csv), b_csv.stat().st_mtime, as_str=True)
    rows_a = [{str(k): str(v) for k, v in r.items()} for r in df_a.to_dict("records")]
    rows_b = [{str(k): str(v) for k, v in r.items()} for r in df_b.to_dict("records")]
    pairs = compare.join_by_spot(rows_a, rows_b)
    verdicts = compare.load_verdicts(a_csv)
    counts = compare.tally(verdicts)

    st.divider()
    st.markdown(
        f"### Tally — **{result['a_name']}** {counts['A']}  ·  "
        f"**{result['b_name']}** {counts['B']}  ·  tie {counts['tie']}   "
        f"({len(verdicts)}/{len(pairs)} judged)"
    )
    if not pairs:
        st.warning("No shared spots to compare (did both runs produce rows?).")
        return

    opts = [f"{result['a_name']} better", "Tie", f"{result['b_name']} better"]
    to_verdict = {opts[0]: "A", opts[1]: "tie", opts[2]: "B"}
    from_verdict = {"A": opts[0], "tie": opts[1], "B": opts[2]}

    # Finalize grades live in each compare CSV's own .review.json sidecar (the
    # same store the Review page uses), so a question finalized here flows into
    # the shared cross-batch "Approved questions" pool. Loaded once per rerun.
    reviews_a = review.load_reviews(a_csv)
    reviews_b = review.load_reviews(b_csv)

    # --- batch finalize: send an ENTIRE side to the approved pool at once ---
    def _approve_all_plo(win_csv: Path, lose_csv: Path, side: str) -> None:
        for _k, r_a, r_b in pairs:
            win_row = r_a if win_csv == a_csv else r_b
            lose_row = r_b if win_csv == a_csv else r_a
            review.save_review(
                win_csv, str(win_row.get("No", "")), "approved",
                f"batch-approved ({side}) from compare",
            )
            review.remove_review(lose_csv, str(lose_row.get("No", "")))

    bap1, bap2 = st.columns(2)
    if bap1.button(
        f"✅ Approve all of A — {result['a_name']} ({len(pairs)})",
        key="plo_cmp_approve_all_a",
        use_container_width=True,
    ):
        _approve_all_plo(a_csv, b_csv, "A")
        st.success(f"Sent all {len(pairs)} A-side questions to the approved pool.")
        st.rerun()
    if bap2.button(
        f"✅ Approve all of B — {result['b_name']} ({len(pairs)})",
        key="plo_cmp_approve_all_b",
        use_container_width=True,
    ):
        _approve_all_plo(b_csv, a_csv, "B")
        st.success(f"Sent all {len(pairs)} B-side questions to the approved pool.")
        st.rerun()
    st.caption(
        "Approves every spot's chosen side into the Approved pool (download at "
        "the bottom of this page or on the PLO Review page). The per-spot "
        "buttons below override individual picks and keep any edits you made."
    )

    for key, row_a, row_b in pairs:
        with st.container(border=True):
            if row_a.get("Context"):
                st.caption(row_a["Context"])
            st.markdown(_md_lines(row_a.get("Question", "")))
            picks = ", ".join(
                row_a.get(f"option {i}", "")
                for i in (1, 2, 3, 4)
                if row_a.get(f"option {i}", "")
            )
            st.caption(
                f"Options: {picks}  ·  Correct: **{row_a.get('Correct Answer', '')}**"
            )
            if row_a.get("action_frequencies"):
                st.markdown(f"**Solver frequencies:** {row_a['action_frequencies']}")
            fact_bits: list[str] = []
            for col, lbl in (
                ("archetype", "archetype"),
                ("ev_gap_bb", "EV gap"),
                ("Difficulty Rating", "difficulty"),
            ):
                val = row_a.get(col, "")
                if val:
                    fact_bits.append(f"{lbl}: `{val}`")
            if fact_bits:
                st.caption(" · ".join(fact_bits))
            if row_a.get("concept_tags"):
                st.caption(f"concept tags: {row_a['concept_tags']}")
            if row_a.get("skills"):
                st.caption(f"skills: {row_a['skills']}")
            # Editable explanations: tweak the prose before finalizing -- the
            # finalize buttons save whatever is in the box. Keys carry the
            # CSV stem so a fresh comparison never inherits stale edits.
            orig_a = row_a.get("Answer Explanation", "")
            orig_b = row_b.get("Answer Explanation", "")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{result['a_name']}**")
                edited_a = st.text_area(
                    "Explanation A",
                    value=orig_a,
                    height=400,
                    key=f"plo_cmp_exp_a_{a_csv.stem}_{key}",
                    label_visibility="collapsed",
                )
                if edited_a.strip() != orig_a.strip():
                    st.caption("✏️ Edited — finalizing A saves this text.")
            with col_b:
                st.markdown(f"**{result['b_name']}**")
                edited_b = st.text_area(
                    "Explanation B",
                    value=orig_b,
                    height=400,
                    key=f"plo_cmp_exp_b_{b_csv.stem}_{key}",
                    label_visibility="collapsed",
                )
                if edited_b.strip() != orig_b.strip():
                    st.caption("✏️ Edited — finalizing B saves this text.")
            # Both sides are the SAME spot, so the deterministic math is
            # identical -- one shared panel under the pair. (No-ops on PLO,
            # whose rows carry no stat_notes yet.)
            _render_stat_panel(row_a)
            _render_exploit_panel(row_a)
            _render_claim_check_panel(row_a)
            cur = verdicts.get(key)
            idx = opts.index(from_verdict[cur]) if cur in from_verdict else None
            choice = st.radio(
                "Which is better?",
                opts,
                index=idx,
                horizontal=True,
                key=f"plo_cmp_v_{key}",
            )
            if choice is not None and to_verdict[choice] != cur:
                compare.save_verdict(a_csv, key, to_verdict[choice])
                st.rerun()

            # --- finalize: save the chosen explanation to the approved pool ---
            no_a, no_b = str(row_a.get("No", "")), str(row_b.get("No", ""))
            fin_a = reviews_a.get(no_a, {}).get("status") == "approved"
            fin_b = reviews_b.get(no_b, {}).get("status") == "approved"
            fcol_a, fcol_b = st.columns(2)
            if fcol_a.button(
                "Save A to finalized",
                key=f"plo_cmp_fin_a_{key}",
                disabled=fin_a,
                use_container_width=True,
            ):
                # Exclusive: finalizing one variant un-finalizes the other.
                # An edited explanation rides along as a sidecar override --
                # the compare CSV itself keeps the original prose.
                review.save_review(
                    a_csv,
                    no_a,
                    "approved",
                    "finalized from compare",
                    explanation=(
                        edited_a if edited_a.strip() != orig_a.strip() else None
                    ),
                )
                review.remove_review(b_csv, no_b)
                st.rerun()
            if fcol_b.button(
                "Save B to finalized",
                key=f"plo_cmp_fin_b_{key}",
                disabled=fin_b,
                use_container_width=True,
            ):
                review.save_review(
                    b_csv,
                    no_b,
                    "approved",
                    "finalized from compare",
                    explanation=(
                        edited_b if edited_b.strip() != orig_b.strip() else None
                    ),
                )
                review.remove_review(a_csv, no_a)
                st.rerun()
            if fin_a or fin_b:
                which = result["a_name"] if fin_a else result["b_name"]
                _fin_grade = (reviews_a.get(no_a) if fin_a else reviews_b.get(no_b)) or {}
                _edited_note = " (with your edits)" if _fin_grade.get("explanation") else ""
                st.caption(f"✅ Saved to finalized using **{which}**{_edited_note}.")
                if st.button("Remove from finalized", key=f"plo_cmp_unfin_{key}"):
                    review.remove_review(a_csv, no_a)
                    review.remove_review(b_csv, no_b)
                    st.rerun()

    # --- download the shared finalized pool (same set as the Review page) -----
    st.divider()
    fin_fields, fin_rows = review.collect_approved_rows(_PLO_BATCH_DIR)
    if fin_rows:
        st.download_button(
            f"⬇️  Download finalized questions (CSV) — {len(fin_rows)} total",
            review.approved_rows_to_csv(fin_fields, fin_rows),
            file_name="plo_approved_all_batches.csv",
            mime="text/csv",
            type="primary",
            key="plo_cmp_download_finalized",
        )
        st.caption(
            "Every question you save to finalized here or approve on the Review "
            "page, across all batches, deduped by spot."
        )
    else:
        st.caption(
            "Save questions to finalized above (or approve them on the Review "
            "page) to build your downloadable set."
        )


def render_skills_page() -> None:
    """Reference catalog for the 42 user-facing skills.

    For each skill, shows: section, current firing status, plain-English
    trigger description, and the actual rule source code. Lets a
    reviewer reading a tagged question understand WHY each skill fired
    and what the rule looks like in code.
    """
    import inspect  # noqa: PLC0415

    from pipeline import skill_tagger  # noqa: PLC0415

    st.title("Skills — user-facing tag catalog")
    st.caption(
        "The labels the app surfaces to users, mapped from the pipeline's "
        "computational outputs (archetype + concept tags + scenario metadata). "
        "Two catalogs: **No-Limit Hold'em** below, then **PLO / Omaha** at the "
        "bottom."
    )
    st.header("♠️  No-Limit Hold'em — 42 skills")
    st.caption(
        "`pipeline.skill_tagger.SKILL_CATALOG`. Use this to understand exactly "
        "why a skill fires on a given Hold'em spot."
    )

    catalog = skill_tagger.SKILL_CATALOG
    meta = skill_tagger.SKILL_META

    # --- summary metrics ---
    status_counts: dict[str, int] = {"preflop_fires": 0, "postflop_fires": 0, "todo": 0}
    for m in meta.values():
        status_counts[m.status] = status_counts.get(m.status, 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total skills", len(catalog))
    col2.metric(
        "Fires today",
        status_counts["preflop_fires"],
        help="Rules that tag real spots on preflop output (which is "
        "what's generated today).",
    )
    col3.metric(
        "Awaits Pio solves",
        status_counts["postflop_fires"],
        help="Rules wired and tested but won't fire until postflop "
        "generation lands.",
    )
    col4.metric(
        "TODO Phase 4",
        status_counts["todo"],
        help="Predicates hard-coded to False -- need new computational "
        "signals or scenario metadata to fire.",
    )

    st.divider()

    # --- filter + grouping ---
    sections = sorted({m.section for m in meta.values()})
    col_filter1, col_filter2 = st.columns([2, 3])
    with col_filter1:
        section_filter = st.multiselect(
            "Filter by section",
            options=sections,
            default=[],
            placeholder="All sections",
        )
    with col_filter2:
        status_filter = st.multiselect(
            "Filter by status",
            options=["Fires today (preflop)", "Awaits Pio solves (postflop)",
                     "TODO Phase 4"],
            default=[],
            placeholder="All statuses",
        )

    _STATUS_LABEL = {
        "preflop_fires": "Fires today (preflop)",
        "postflop_fires": "Awaits Pio solves (postflop)",
        "todo": "TODO Phase 4",
    }
    _STATUS_EMOJI = {
        "preflop_fires": "✅",
        "postflop_fires": "⏳",
        "todo": "🚧",
    }

    def _matches(name: str) -> bool:
        m = meta[name]
        if section_filter and m.section not in section_filter:
            return False
        if status_filter and _STATUS_LABEL[m.status] not in status_filter:
            return False
        return True

    visible = [name for name in catalog if _matches(name)]
    st.caption(f"Showing **{len(visible)}** of {len(catalog)} skills.")

    # --- render: group by section, one expander per skill ---
    current_section = ""
    for name in visible:
        m = meta[name]
        if m.section != current_section:
            st.subheader(m.section)
            current_section = m.section

        emoji = _STATUS_EMOJI[m.status]
        label = f"{emoji}  **{name}**  ·  _{_STATUS_LABEL[m.status]}_"
        with st.expander(label):
            st.markdown(f"**Trigger.** {m.description}")
            try:
                src = inspect.getsource(catalog[name]).strip()
                # Strip trailing comma if it's an inline-dict lambda entry.
                src = src.rstrip(",")
                st.markdown("**Rule source.**")
                st.code(src, language="python")
            except (OSError, TypeError):
                # Fallback for stripped or built-in callables.
                st.caption("(source not available)")

    # --- PLO / Omaha skills (separate catalog) ---
    from pipeline.plo import skill_tagger as plo_skills  # noqa: PLC0415

    st.divider()
    st.header("🃏  PLO / Omaha — 23 skills")
    st.caption(
        "`pipeline.plo.skill_tagger.SKILL_CATALOG`. Preflop only (no PLO "
        "postflop solves yet). Strict tagging: ~2-5 fire per question. Grouped "
        "by category; ⏸ marks a skill that's wired but dormant on this "
        "single-raise-size pack. Postflop skills (Wrap/Draw, Nut Blockers, Set "
        "Mining, C-betting, Bluff-catching, Pot Control) are out of scope until "
        "PLO postflop solves exist."
    )
    plo_catalog = plo_skills.SKILL_CATALOG
    plo_meta = plo_skills.SKILL_META
    for category in plo_skills.SKILL_CATEGORIES:
        st.subheader(category)
        for name in [n for n in plo_catalog if plo_meta[n].category == category]:
            pm = plo_meta[name]
            badge = "✅" if pm.fires else "⏸"
            with st.expander(f"{badge}  **{name}**"):
                st.markdown(f"**Trigger.** {pm.description}")
                if not pm.fires:
                    st.caption(
                        "Wired but dormant on this pack -- needs a multi-raise-"
                        "size tree to ever fire."
                    )
                try:
                    src = inspect.getsource(plo_catalog[name]).strip().rstrip(",")
                    st.markdown("**Rule source.**")
                    st.code(src, language="python")
                except (OSError, TypeError):
                    st.caption("(source not available)")


# --- main router ------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Poker Pipeline Admin",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🎰 Poker Pipeline")
    st.sidebar.caption("Preflop pipeline · Phase 3 (skill tagging)")
    # Apply any pending programmatic navigation (e.g. the Review page's
    # "View ranges" button) BEFORE the nav widget is created -- a widget's
    # session value can't be set after it's instantiated in the same run.
    if "_pending_nav" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("_pending_nav")
    page = st.sidebar.radio(
        "Page",
        options=["Files", "Generate", "Review", "Postflop Review",
                 "Postflop Compare", "Ranges",
                 "History", "Browse", "Prompt", "Compare", "Skills",
                 "Concept Tags",
                 "PLO Generate", "PLO Review", "PLO Prompt", "PLO Compare"],
        index=0,
        key="nav_page",
    )

    # Live job indicator -- visible from any page, auto-refreshes once
    # a second via the fragment so progress ticks up while the user is
    # on (say) the History tab. Streamlit requires fragments that write
    # to the sidebar to be CALLED inside a `with st.sidebar:` block --
    # the fragment itself just uses bare `st.info(...)` etc.
    with st.sidebar:
        _render_sidebar_job_indicator()

    # Sidebar status indicators
    st.sidebar.divider()
    st.sidebar.subheader("Status")
    ranges_ok, _ = ranges_pack_status()
    st.sidebar.text(f"Ranges 6-max: {'✅ ready' if ranges_ok else '❌ missing'}")
    _n9 = _monker_pack_file_count(NLHE9_PACK_ID)
    st.sidebar.text(
        "Ranges 9-max: "
        + (
            "✅ ready"
            if _n9 == EXPECTED_NLHE9_RANGE_COUNT
            else ("❌ missing" if _n9 == 0 else f"⚠️ {_n9:,} files")
        )
    )
    total_cfrs = sum(count_cfrs(s.name) for s in SCENARIOS)
    expected_cfrs = len(SCENARIOS) * 25
    if total_cfrs == 0:
        st.sidebar.text("Solves: ❌ awaiting William")
    elif total_cfrs == expected_cfrs:
        st.sidebar.text(f"Solves: ✅ {total_cfrs}/{expected_cfrs}")
    else:
        st.sidebar.text(f"Solves: ⚠️  {total_cfrs}/{expected_cfrs}")

    # Lifetime API spend, summed from usage_log.jsonl. Survives admin-
    # panel restarts since the log is on disk. Renders nothing when
    # the log is empty (fresh install / dry-runs only).
    st.sidebar.divider()
    st.sidebar.subheader("API spend")
    stats = usage.compute_lifetime_stats(USAGE_LOG_PATH)
    if stats.total_batches == 0:
        st.sidebar.caption(
            "No real-API batches logged yet. Run a batch with "
            "dry-run off to start tracking."
        )
    else:
        st.sidebar.metric(
            "Lifetime",
            usage.format_cost(stats.total_cost_usd),
            delta=(
                f"{stats.total_batches} batches · "
                f"{stats.total_questions:,} questions"
            ),
            delta_color="off",
            help=(
                f"Summed across every real-API batch in "
                f"`{USAGE_LOG_PATH.relative_to(REPO_ROOT)}`. Models "
                f"so far: {', '.join(stats.models_used) or 'none'}."
            ),
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Preflop and postflop both generate end-to-end. Postflop reads "
        "third-party `.db` solves from `solves/postflop/`."
    )

    if page == "Files":
        render_files_page()
    elif page == "Generate":
        render_generate_page()
    elif page == "Review":
        render_review_page()
    elif page == "Postflop Review":
        render_postflop_review_page()
    elif page == "Postflop Compare":
        render_postflop_compare_page()
    elif page == "Ranges":
        render_ranges_page()
    elif page == "History":
        render_history_page()
    elif page == "Browse":
        render_browse_page()
    elif page == "Prompt":
        render_prompt_page()
    elif page == "Compare":
        render_compare_page()
    elif page == "Skills":
        render_skills_page()
    elif page == "Concept Tags":
        render_concept_tags_page()
    elif page == "PLO Generate":
        render_plo_generate_page()
    elif page == "PLO Review":
        render_plo_review_page()
    elif page == "PLO Prompt":
        render_plo_prompt_page()
    elif page == "PLO Compare":
        render_plo_compare_page()


@st.fragment(run_every=1.0)
def _render_sidebar_active_job() -> None:
    """Ticking sidebar progress line -- mounted only while a job runs.

    Writes via bare ``st.info`` (NOT ``st.sidebar.X``): Streamlit
    forbids fragments from calling ``st.sidebar`` directly; the caller
    wraps the invocation in ``with st.sidebar:`` instead. When the job
    flips to done, triggers one full-app rerun (the user may be on any
    page -- this fragment is the only ticker there) so the static
    done/failed line below takes over and the ticking stops.
    """
    job = jobs.get_current_job()
    if job is None:
        return
    if not job.is_active:
        _job_done_rerun_once(job)
        return
    p = job.progress
    if p.total > 0:
        pct = ((p.current + 1) / p.total) * 100
        st.info(
            f"🔄 Job: {p.current + 1}/{p.total}  ({pct:.0f}%) · "
            f"{job.elapsed_seconds:.0f}s"
        )
    else:
        st.info(f"🔄 Job: starting · {job.elapsed_seconds:.0f}s")


def _render_sidebar_job_indicator() -> None:
    """Tiny sidebar widget showing current job status (any page).

    Live progress comes from the ticking fragment above, mounted only
    while the job is active; done/failed states are plain static lines
    on normal reruns (June 2026: the always-ticking version kept the
    whole app busy once a second forever after a batch finished).
    """
    job = jobs.get_current_job()
    if job is None:
        return
    if job.is_active:
        _render_sidebar_active_job()
    elif job.status is jobs.JobStatus.COMPLETED:
        st.success(f"✅ Job done · {job.elapsed_seconds:.0f}s")
    elif job.status is jobs.JobStatus.FAILED:
        st.error("❌ Job failed (see Generate tab)")


if __name__ == "__main__":
    main()
