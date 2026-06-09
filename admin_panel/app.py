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
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
from admin_panel import compare, jobs, range_view, review, usage  # noqa: E402

# Imports from the pipeline (safe at module load -- these touch no I/O and
# don't require a PioSolver binary or API key to import).
from pipeline.explanation_generator import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    build_system_prompt,
)
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
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopDecisionNode,
    enumerate_nodes_by_actor,
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
# Fable 5 first = the default everywhere (highest-quality explanations:
# adaptive thinking + max effort are applied automatically by the pipeline).
_MODEL_LABEL_TO_API: dict[str, str] = {
    "Fable 5 (best quality, 2× Opus price)": "claude-fable-5",
    "Opus 4.7 (high fidelity)": "claude-opus-4-7",
    "Sonnet 4.6 (cheapest, fastest)": "claude-sonnet-4-6",
}

# One-line note shown wherever Fable 5 is selectable: it has no temperature
# control (the pipeline drops the param) and self-manages its reasoning.
_FABLE_NOTE = (
    "Fable 5 ignores the temperature setting (the API removed it) and runs "
    "with adaptive thinking at max effort automatically — the "
    "highest-quality configuration."
)

# Where preflop generation writes its CSV output. Sibling of test_output/
# tier1_consolidated.csv but kept in its own subdir so the Browse page's
# existing data isn't accidentally shadowed.
PREFLOP_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "test_output" / "preflop_batches"
)

# JSONL log of every completed real-API batch. Sibling of the CSVs
# the History tab manages. Gitignored. The sidebar's "Lifetime spend"
# widget + the History page's totals read from this file.
USAGE_LOG_PATH = (
    Path(__file__).resolve().parent.parent / "test_output" / "usage_log.jsonl"
)

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

# --- repo paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SOLVES_DIR = REPO_ROOT / "solves"
RANGES_DIR = REPO_ROOT / "ranges"
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
def _cached_preflop_nodes_by_actor() -> dict[str, tuple[PreflopDecisionNode, ...]]:
    """Walk the preflop packs once and group nodes by hero position."""
    packs = _cached_preflop_packs()
    if not packs:
        return {}
    return enumerate_nodes_by_actor(packs)


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
    for nodes in _cached_preflop_nodes_by_actor().values():
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


def ranges_pack_status() -> tuple[bool, dict[str, int]]:
    """Return (is_complete, per_position_counts) for the ranges pack."""
    counts = {}
    for pos in POSITION_FOLDERS:
        folder = RANGES_SUBDIR / pos
        counts[pos] = len(list(folder.glob("*.txt"))) if folder.is_dir() else 0
    is_complete = all(
        counts[pos] == EXPECTED_RANGE_COUNTS[pos] for pos in POSITION_FOLDERS
    )
    return is_complete, counts


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
    # ranges/ryan_preflop_tree/ instead). Postflop uses the original
    # pipeline (needs solves/ which are currently unavailable post-William).
    mode = st.radio(
        "Mode",
        options=["Postflop", "Preflop"],
        index=1,  # default to Preflop since the backend is ready
        horizontal=True,
        help=(
            "Postflop uses PioSolver .cfr files (currently blocked -- no "
            "solves available since William left the project). Preflop "
            "uses Ryan's preflop pack (loaded and tested)."
        ),
        key="generate_mode",
    )
    st.divider()

    if mode == "Preflop":
        _render_generate_page_preflop()
        return

    # --- POSTFLOP PATH (original; unchanged below) ---
    st.caption(
        "Configure a batch run. Filters cascade — pick a format, then "
        "stack, then scenarios. Only options with available solves are "
        "selectable."
    )

    # Discover what's actually available on disk
    available_scenarios = [s for s in SCENARIOS if count_cfrs(s.name) > 0]
    available_formats = sorted({s.format for s in available_scenarios})
    if not available_formats:
        available_formats = ["Cash"]  # show the UI even with no solves

    # --- Cascading filters ---
    st.subheader("1. Game settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        fmt = st.selectbox(
            "Format",
            options=["Cash", "MTT"],
            index=0,
            help="MTT solves aren't loaded yet. Only formats with "
            "available solves are usable.",
        )
    with col2:
        stack = st.selectbox(
            "Stack depth",
            options=["100 bb", "60 bb", "40 bb"],
            index=0,
            help="Only 100bb solves exist in Ryan's pack. Others would "
            "need different range packs.",
        )
    with col3:
        table = st.selectbox(
            "Table size",
            options=["6-max", "9-max", "Heads-Up"],
            index=0,
        )

    # Filter scenarios by the above selections
    def matches(s: ScenarioMeta) -> bool:
        if s.format != fmt:
            return False
        if s.stack_bb != int(stack.split()[0]):
            return False
        if table == "6-max" and s.table_size != 6:
            return False
        if table == "9-max" and s.table_size != 9:
            return False
        return True

    scenarios_after_filter = [s for s in SCENARIOS if matches(s)]
    if not scenarios_after_filter:
        st.error(
            f"⚠️ No scenarios match `{fmt}` / `{stack}` / `{table}`. "
            f"This combination has no registered scenarios yet."
        )
        return

    # --- Scenario picker ---
    st.subheader("2. Scenarios")
    st.caption(
        f"{len(scenarios_after_filter)} registered scenarios match. "
        "Greyed-out ones are missing solves."
    )
    selected_scenarios = []
    for s in scenarios_after_filter:
        n = count_cfrs(s.name)
        if n == 25:
            label = f"✅  **{s.name}**  ·  {s.preflop_action}"
            picked = st.checkbox(label, value=False, key=f"sc_{s.name}")
            if picked:
                selected_scenarios.append(s)
        elif n == 0:
            st.checkbox(
                f"❌  ~~{s.name}~~  ·  {s.preflop_action}  ·  *no solves*",
                value=False,
                disabled=True,
                key=f"sc_{s.name}",
            )
        else:
            label = f"⚠️  **{s.name}**  ·  {s.preflop_action}  ·  *only {n}/25 flops*"
            picked = st.checkbox(label, value=False, key=f"sc_{s.name}")
            if picked:
                selected_scenarios.append(s)

    st.divider()

    # --- Content filters (hand class + board texture) ---
    st.subheader("3. Content filters")
    st.caption(
        "Optional. Narrow what kinds of spots show up. Leave empty to "
        "include everything that survives Layer 4."
    )
    col1, col2 = st.columns(2)
    with col1:
        _hand_classes = st.multiselect(
            "Hero hand strength buckets",
            options=list(STRENGTH_BUCKETS),
            default=[],
            help=(
                "Sourced from pipeline/fact_extractor/hand_class.py. "
                "Each spot's hero hand maps to one bucket. Empty = all."
            ),
        )
    with col2:
        _board_composites = st.multiselect(
            "Board texture (composite)",
            options=list(BOARD_COMPOSITES),
            default=[],
            help=("From board_texture.py's composite descriptor. Empty = all."),
        )
    col1, col2 = st.columns(2)
    with col1:
        _board_suits = st.multiselect(
            "Board suit distribution",
            options=list(BOARD_SUIT_DISTRIBUTIONS),
            default=[],
        )
    with col2:
        _board_pairs = st.multiselect(
            "Board pair status",
            options=list(BOARD_PAIR_STATUSES),
            default=[],
        )

    st.divider()

    # --- Difficulty ---
    st.subheader("4. Difficulty")
    st.caption("How dominant the correct answer is in the solver.")
    preset = st.radio(
        "Preset",
        options=["Easy", "Medium", "Hard", "Mixed", "Custom"],
        index=4,
        horizontal=True,
        key="difficulty_preset",
    )
    presets_map = {
        "Easy": (85, 95),
        "Medium": (70, 85),
        "Hard": (55, 70),
        "Mixed": (55, 95),
        "Custom": (65, 75),  # default custom value
    }
    default_low, default_high = presets_map[preset]
    freq_low, freq_high = st.slider(
        "Solver frequency window (%) — correct answer is this dominant",
        min_value=50,
        max_value=100,
        value=(default_low, default_high),
        disabled=(preset != "Custom"),
        help="Lower = harder (close decisions). Higher = easier (clear-cut).",
    )

    st.divider()

    # --- Answer style ---
    st.subheader("5. Answer option style")
    col1, col2 = st.columns([2, 3])
    with col1:
        answer_style = st.radio(
            "Style",
            options=[
                "Basic (fold/call/raise)",
                "GTO (always/mostly)",
                "Sizing (33%/75%/150%) — coming soon",
                "Auto-pick",
            ],
            index=1,
            help="Basic and GTO work today. Sizing variant needs ~2-3 "
            "days of pipeline work.",
        )
    with col2:
        st.caption("Style preview:")
        if answer_style.startswith("Basic"):
            st.code("option 1: Fold\noption 2: Call\noption 3: Raise")
        elif answer_style.startswith("GTO"):
            st.code(
                "option 1: Always check\n"
                "option 2: Mostly check, sometimes bet\n"
                "option 3: Mostly bet, sometimes check\n"
                "option 4: Always bet"
            )
        elif answer_style.startswith("Sizing"):
            st.code(
                "option 1: Check\noption 2: Bet $5 (33%)\n"
                "option 3: Bet $11 (75%)\noption 4: Bet $22 (150%)"
            )
        else:
            st.code("Layer 6 chooses Basic / GTO / Sizing per spot")

    st.divider()

    # --- Sampling targets ---
    st.subheader("6. How many questions, where")
    mode = st.radio(
        "Targeting mode",
        options=[
            "Total batch size (auto-distribute)",
            "Per-street targets (power user)",
        ],
        index=0,
        horizontal=True,
    )
    if mode.startswith("Total"):
        total = st.number_input(
            "Total questions in this batch",
            min_value=1,
            max_value=10_000,
            value=20,
            step=5,
        )
        st.caption(
            "Will spread evenly across selected scenarios + streets, "
            "with diversity-stratified sampling within each bucket."
        )
        # Compute per-bucket distribution preview
        flop_t = turn_t = river_t = 0
        if selected_scenarios:
            per_scenario = total // len(selected_scenarios)
            flop_t = max(1, per_scenario // 5)
            turn_t = max(1, (per_scenario * 2) // 5)
            river_t = per_scenario - flop_t - turn_t
        st.caption(
            f"Distribution preview: {len(selected_scenarios)} scenarios × "
            f"({flop_t} flop + {turn_t} turn + {river_t} river) = "
            f"{len(selected_scenarios) * (flop_t + turn_t + river_t)} "
            "questions"
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            flop_t = st.number_input("Flop / scenario", 0, 25, 1)
        with col2:
            turn_t = st.number_input("Turn / scenario", 0, 25, 2)
        with col3:
            river_t = st.number_input("River / scenario", 0, 25, 2)
        with col4:
            _per_cfr_cap = st.number_input("Max per .cfr", 1, 25, 5)
        total = len(selected_scenarios) * (flop_t + turn_t + river_t)
        st.caption(f"Total questions: **{total}**")

    st.divider()

    # --- Output options ---
    st.subheader("7. Output")
    col1, col2 = st.columns(2)
    with col1:
        _currency = st.radio(
            "Display amounts as",
            options=["Dollars ($1.25)", "Big blinds (2.5 bb)"],
            index=0,
            horizontal=True,
        )
    with col2:
        _out_filename = st.text_input(
            "Output filename",
            value="batch_2026-05-24.csv",
        )

    # Stake scaling: pick from common levels or specify a custom BB-in-dollars
    # value. The pipeline's scale_scenario() helper rebuilds each scenario at
    # the chosen stake. Stack depth (in bb) is preserved -- a 100bb game stays
    # 100bb at any stake.
    def _stake_label(bb_dollars: float) -> str:
        sb = bb_dollars / 2
        sb_str = f"${sb:.2f}".rstrip("0").rstrip(".") if sb < 1 else f"${int(sb)}"
        bb_str = (
            f"${bb_dollars:.2f}".rstrip("0").rstrip(".")
            if bb_dollars < 1
            else f"${int(bb_dollars)}"
        )
        return f"{sb_str}/{bb_str}"

    stake_options = [_stake_label(bb) for bb in COMMON_STAKE_LEVELS_BB_DOLLARS]
    default_stake_index = list(COMMON_STAKE_LEVELS_BB_DOLLARS).index(0.50)
    col_stake1, col_stake2 = st.columns([2, 1])
    with col_stake1:
        _stake_choice = st.selectbox(
            "Stake level (rendered in output CSV)",
            options=stake_options + ["Custom..."],
            index=default_stake_index,
            help=(
                "All dollar amounts in the generated questions render at "
                "this stake. The underlying chip math is identical -- only "
                "the displayed dollar values change. Stack depth in bb is "
                "preserved."
            ),
        )
    with col_stake2:
        if _stake_choice == "Custom...":
            _custom_bb = st.number_input(
                "Custom BB ($)",
                min_value=0.01,
                max_value=10_000.0,
                value=0.50,
                step=0.25,
                format="%.2f",
            )
            st.caption(f"= {_stake_label(_custom_bb)}")
        else:
            picked_bb = COMMON_STAKE_LEVELS_BB_DOLLARS[
                stake_options.index(_stake_choice)
            ]
            st.caption(f"BB = ${picked_bb:g}")

    st.divider()

    # --- Model + API settings ---
    st.subheader("8. Model + API settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.radio(
            "Model",
            options=list(_MODEL_LABEL_TO_API),
            index=0,
            help="Fable 5 for the highest-quality explanations (the default), "
            "Opus 4.7 at half the price, Sonnet 4.6 for cheap experimentation.",
        )
        if "Fable" in model:
            st.caption(_FABLE_NOTE)
    with col2:
        batch_size = st.selectbox(
            "Questions per API call",
            options=[1, 5, 10, 25],
            index=1,
            help="5 is the recommended starting sweet spot. Higher = "
            "faster but more risk if a call fails.",
        )
    with col3:
        dry_run = st.toggle(
            "Dry run (no API calls)",
            help="Show what would be generated without spending API tokens.",
        )

    # Estimated cost (rough per-question figures by model tier; Fable 5 is
    # 2x Opus list price plus thinking tokens).
    if "Fable" in model:
        est_cost_per_q = 0.45
    elif model.startswith("Opus"):
        est_cost_per_q = 0.15
    else:
        est_cost_per_q = 0.08
    est_total_cost = total * est_cost_per_q if total else 0
    est_minutes = total * 0.5 / max(batch_size, 1) if total else 0

    st.info(
        f"**Estimated**: {total} questions · "
        f"~${est_total_cost:.2f} · "
        f"~{est_minutes:.1f} min · "
        f"model={model.split()[0]} · "
        f"batch={batch_size}"
    )

    st.divider()

    # --- Generate button ---
    can_generate = (
        len(selected_scenarios) > 0
        and total > 0
        and (dry_run or True)  # API key check would go here
    )
    waiting_on_solves = not any(count_cfrs(s.name) > 0 for s in scenarios_after_filter)

    if waiting_on_solves:
        st.button(
            "⏳ GENERATE BATCH  (awaiting solves from William)",
            disabled=True,
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "The postflop Generate button activates once Pio solves land "
            "in `solves/<scenario_name>/<flop_stem>.cfr`. The preflop "
            "path is fully wired and does not need solves -- switch the "
            "Mode toggle above."
        )
    elif not can_generate:
        st.button(
            "GENERATE BATCH",
            disabled=True,
            type="primary",
            use_container_width=True,
        )
        st.caption("Select at least one scenario and set a target above.")
    else:
        if st.button("GENERATE BATCH", type="primary", use_container_width=True):
            st.warning(
                "Postflop batch orchestrator is not implemented yet. The "
                "preflop equivalent (`pipeline.preflop.batch`) is the "
                "template -- a sibling needs to be written for postflop "
                "once Pio `.cfr` solves are available. Switch the Mode "
                "toggle to **Preflop** above to generate questions today."
            )


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
        "Filters narrow which decision spots get sampled. The generate "
        "button activates once Layer 6 wiring lands (~1 day of work)."
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
            "`ranges/ryan_preflop_tree/`.)"
        )
        return
    for p in packs:
        st.success(
            f"✅ **{p.pack_id}** · {p.table_size}-max · "
            f"{p.stack_depth_bb}bb · {p.open_size_bb}x open · "
            f"{p.description}"
        )

    nodes_by_actor = _cached_preflop_nodes_by_actor()
    total_nodes = sum(len(ns) for ns in nodes_by_actor.values())
    st.caption(
        f"Walked the pack: **{total_nodes:,} preflop decision nodes** enumerated."
    )

    st.divider()

    # --- 2. Hero context ---
    st.subheader("2. Hero context")
    col1, col2 = st.columns(2)
    with col1:
        positions_available = sorted(nodes_by_actor.keys())
        hero_positions = st.multiselect(
            "Hero positions",
            options=positions_available,
            default=positions_available,
            help="Which seats hero is in. Empty = include all positions.",
        )
    with col2:
        context_options = [
            "Opening",
            "Facing single raise",
            "Facing 3-bet",
            "Facing 4-bet+",
            "After call(s)",
        ]
        action_contexts = st.multiselect(
            "Action faced",
            options=context_options,
            default=["Facing single raise", "Facing 3-bet"],
            help="What hero is responding to. Empty = include all.",
        )
        player_counts = st.multiselect(
            "Players in the pot",
            options=[1, 2, 3, 4, 5, 6],
            default=[1, 2, 3, 4, 5, 6],
            format_func=lambda n: (
                "1 (open)" if n == 1 else "2 (heads-up)" if n == 2 else f"{n}-way"
            ),
            help="How many players are still in at hero's decision. Narrow "
                 "this (e.g. just 3) for clean three-way spots instead of deep "
                 "multiway bloodbaths. Empty = include all.",
        )

    # Filter the node catalog by all three filters; show a live count.
    _count_set = set(player_counts) if player_counts else None
    filtered_nodes: list[PreflopDecisionNode] = []
    for actor in hero_positions:
        for node in nodes_by_actor.get(actor, ()):
            ctx = node_action_context(node)
            if action_contexts and ctx not in action_contexts:
                continue
            if _count_set is not None and active_player_count(node) not in _count_set:
                continue
            filtered_nodes.append(node)
    st.caption(
        f"**{len(filtered_nodes):,}** decision nodes match these filters "
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
            value=(55, 95),
            key="preflop_worthiness_slider",
            help="Below 55% = no clear best answer to teach; 100% = trivial.",
        )
        exclude_ambiguous_band = st.checkbox(
            "Exclude ambiguous 90–95% band (recommended)",
            value=True,
            key="preflop_exclude_ambiguous_band",
            help=(
                "Spots where the solver takes one action 90–95% of the time "
                "read as \"mostly\" but sit just under the 95% \"always\" "
                "line, so a player with the right read can still pick "
                "\"always\" and be marked wrong. On by default: caps the "
                "effective ceiling at 90%. Uncheck to allow 90–95% spots in."
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

    # Visible settings summary -- exactly what THIS batch will use. Promoted
    # to a prominent info box so the numbers are obvious the moment you pick
    # a preset (the preset moves the difficulty band; worthiness + EV-gap are
    # separate gates, shown here so all three are always visible).
    _ev_txt = "off" if min_ev_gap == 0.0 else f"≥ {min_ev_gap:.2f} bb"
    _band_note = (
        "  ·  90–95% band excluded"
        if exclude_ambiguous_band and freq_high > 90
        else ""
    )
    st.info(
        f"**Numbers in effect for this batch** — difficulty rating "
        f"**{band_low}–{band_high}**  ·  worthiness frequency "
        f"**{freq_low}–{freq_high}%**{_band_note}  ·  EV-gap gate "
        f"**{_ev_txt}**.  "
        "Presets move the difficulty band; the worthiness window + EV-gap "
        "are separate gates you set in Advanced filters above."
    )

    st.divider()

    # --- 5. Answer option style (reuse) ---
    st.subheader("5. Answer option style")
    answer_style = st.radio(
        "Style",
        options=list(ANSWER_STYLE_FROM_RADIO_LABEL.keys()),
        index=0,  # Basic is the natural default for preflop
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

    st.divider()

    # --- 8. Model + API ---
    st.subheader("8. Model + API settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        _model = st.radio(
            "Model",
            options=list(_MODEL_LABEL_TO_API),
            index=0,  # Fable 5 -- highest-quality explanations by default
            key="preflop_model",
        )
        if "Fable" in _model:
            st.caption(_FABLE_NOTE)
    with col2:
        _batch_size = st.selectbox(
            "Questions per API call",
            options=[1, 5, 10, 25],
            index=1,
            key="preflop_batch_size",
        )
    with col3:
        _dry_run = st.toggle(
            "Dry run (no API calls)",
            key="preflop_dry_run",
        )

    # Cost estimate (rough per-question by model tier; Fable 5 is 2x Opus
    # list price plus thinking tokens).
    cost_per_q = 0.45 if "Fable" in _model else (0.15 if "Opus" in _model else 0.08)
    est_cost = total * cost_per_q
    st.info(
        f"**Estimated**: {total} questions · ~${est_cost:.2f} · "
        f"difficulty {band_low}-{band_high} · {len(filtered_nodes):,} "
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
    # Inputs ready when: at least one position selected AND at least one
    # action context AND filtered_nodes non-empty AND total > 0.
    # Disabled while another job is in flight -- the active-job panel
    # rendered above shows that one.
    can_generate = (
        bool(hero_positions)
        and bool(action_contexts)
        and len(filtered_nodes) > 0
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
        _start_preflop_job(
            pack=packs[0],
            hero_positions=list(hero_positions),
            action_contexts=list(action_contexts),
            player_counts=list(player_counts),
            freq_min=freq_low / 100.0,
            freq_max=freq_high / 100.0,
            exclude_ambiguous_band=exclude_ambiguous_band,
            min_difficulty=int(band_low),
            max_difficulty=int(band_high),
            min_ev_gap_bb=(None if min_ev_gap == 0.0 else float(min_ev_gap)),
            display_in_bb=_currency.startswith("Big blinds"),
            total_questions=int(total),
            output_filename=_out_filename,
            model_label=_model,
            dry_run=bool(_dry_run),
            answer_style=answer_style_canonical,
            system_prompt=_prompt_text,
            prompt_name=_prompt_name,
            random_seed=_seed_val,
            temperature=_temp_val,
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
    min_difficulty: int,
    max_difficulty: int,
    min_ev_gap_bb: float | None,
    display_in_bb: bool,
    total_questions: int,
    output_filename: str,
    model_label: str,
    dry_run: bool,
    answer_style: str,
    system_prompt: str | None = None,
    prompt_name: str = "",
    random_seed: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
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
        jobs.start_job(
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
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
            min_ev_gap_bb=min_ev_gap_bb,
            display_in_bb=display_in_bb,
            answer_style=answer_style,
            model=model_api,
            temperature=temperature,
            system_prompt=system_prompt,
            prompt_name=prompt_name,
            dry_run=dry_run,
            random_seed=random_seed,
        )
    except RuntimeError as exc:
        # Another job is already running. The button-disable check
        # normally prevents this, but races are possible if two browser
        # tabs hit Generate at the same time.
        st.error(f"⚠️ Could not start batch: {exc}")
        return

    # Re-run immediately so the job panel takes over the next render.
    st.rerun()


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


@st.fragment(run_every=1.0)
def _render_preflop_job_panel() -> None:
    """Top-of-page panel showing the current (or last) background job.

    Wrapped in ``@st.fragment(run_every=1.0)`` so it auto-refreshes
    once a second without rerunning the whole filter form -- the user
    can keep configuring the NEXT batch while the current one runs.

    Three visual states:

    * **Active** (PENDING / RUNNING)  -- progress bar + status line.
    * **Completed** -- success + download + preview (same UI the
      synchronous version rendered, now reusable).
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
        elif job.status is jobs.JobStatus.COMPLETED:
            st.markdown(f"**✅ Last batch:** {job.label}")
            if isinstance(job.result, BatchResult):
                # Append to the usage log exactly once -- module-level
                # dedupe across browser sessions + fragment reruns.
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
        else:  # FAILED
            st.error(f"**❌ Job failed:** {job.label}")
            with st.expander("Traceback"):
                st.code(job.error or "(no traceback captured)")
            if st.button("Dismiss failure", key=f"dismiss_job_{job.id}"):
                jobs.clear_current_job()
                st.rerun()

    st.divider()


def _maybe_log_completed_job(job: jobs.Job[BatchResult], result: BatchResult) -> None:
    """Append this job's usage to the JSONL log -- exactly once per process.

    Idempotent: tracks logged job ids in a module-level set so the
    fragment's per-second re-render doesn't duplicate the entry.
    Dry-runs (``model_used == ""``) are dropped by
    :func:`usage.append_log_entry`.
    """
    logged = _logged_job_ids()
    if job.id in logged:
        return
    logged.add(job.id)
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
            f"failures: **{len(result.failures)}**. "
            "Try a wider difficulty band, the Mixed preset, or a wider "
            "worthiness window."
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
        st.success(
            f"Wrote **{result.questions_written}** questions to "
            f"`{result.output_path}` "
            f"(attempted {result.questions_attempted}, "
            f"{result.worthy_spots_available} worthy spots available"
            f"{_band_note}{_noise_note})."
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

    df = pd.read_csv(TIER1_CSV, encoding="utf-8-sig")
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
    (Markdown collapses single newlines otherwise)."""
    return text.replace("\n", "  \n")


def _autosave_review_cell(
    csv_path: Path, no: str, widget_key: str, kind: str
) -> None:
    """on_change callback: persist a Review edit (explanation or difficulty)
    straight into the batch CSV, so there's no Save button to forget.

    Runs before the rerun Streamlit triggers after a widget change, so the
    re-read of the CSV at the top of the page reflects the edit immediately.
    """
    value = st.session_state.get(widget_key, "")
    if kind == "difficulty":
        ok = review.update_difficulty(csv_path, no, str(int(value)))
    else:
        ok = review.update_explanation(csv_path, no, str(value))
    if ok:
        st.toast(f"Saved #{no}")


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
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
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

    # Download the whole batch -- including any Answer Explanation edits,
    # which are written straight into this CSV on disk when saved below.
    st.download_button(
        "⬇  Download this batch (CSV)",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        help="The full batch CSV, with any explanation edits you've saved baked in.",
    )

    st.divider()

    # --- per-file navigation state ---
    nav_key = f"review_idx::{csv_path.name}"
    idx = st.session_state.get(nav_key, 0)
    idx = max(0, min(int(idx), len(df) - 1))

    nos = df["No"].tolist()
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("◀ Prev", use_container_width=True, disabled=idx == 0):
            st.session_state[nav_key] = idx - 1
            st.rerun()
    with nav3:
        if st.button(
            "Next ▶", use_container_width=True, disabled=idx >= len(df) - 1
        ):
            st.session_state[nav_key] = idx + 1
            st.rerun()
    with nav2:
        jump = st.selectbox(
            "Jump to question",
            options=range(len(df)),
            index=idx,
            format_func=lambda i: f"#{nos[i]}  ({i + 1}/{len(df)})",
            label_visibility="collapsed",
        )
        if jump != idx:
            st.session_state[nav_key] = jump
            st.rerun()

    row = df.iloc[idx]
    no = str(row["No"])
    existing = reviews.get(no, {})

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

        st.markdown("**Answer Explanation** _(edits auto-save into the CSV)_")
        _expl_key = f"review_expl::{csv_path.name}::{no}"
        st.text_area(
            "Answer Explanation",
            value=_cell(row, "Answer Explanation"),
            key=_expl_key,
            height=200,
            label_visibility="collapsed",
            on_change=_autosave_review_cell,
            args=(csv_path, no, _expl_key, "explanation"),
        )
        # Editable difficulty -- auto-saves into the CSV just like the
        # explanation (no Save button; the on_change callback writes it).
        _diff_key = f"review_diff::{csv_path.name}::{no}"
        try:
            _cur_diff = int(float(_cell(row, "Difficulty Rating") or 0))
        except ValueError:
            _cur_diff = 0
        st.number_input(
            "Difficulty Rating (edits auto-save)",
            min_value=0,
            max_value=3500,
            step=10,
            value=_cur_diff,
            key=_diff_key,
            on_change=_autosave_review_cell,
            args=(csv_path, no, _diff_key, "difficulty"),
        )
        # Rendered preview (suit emojis etc.) of the saved explanation.
        with st.expander("Preview (rendered)", expanded=False):
            st.info(_md_lines(_cell(row, "Answer Explanation")))

        st.markdown(
            f"**Solver frequencies:**&nbsp;{_cell(row, 'action_frequencies')}"
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

        # Ranges: a button to the visual Range viewer, raw JSON tucked away.
        ranges_val = _cell(row, "ranges")
        n_players = review.range_player_count(ranges_val)
        if st.button(
            "📊  View ranges for this spot",
            key=f"view_ranges::{csv_path.name}::{no}",
            use_container_width=True,
            help="Open the Range viewer tab with this question's node loaded.",
        ):
            st.session_state["ranges_node_id"] = (
                range_view.node_id_from_solver_reference(
                    _cell(row, "solver_reference")
                )
            )
            st.session_state["ranges_from_q"] = no
            st.session_state["_pending_nav"] = "Ranges"
            st.rerun()
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

    # --- grading ---
    st.markdown("**Grade**")
    note = st.text_area(
        "Note (optional)",
        value=existing.get("note", ""),
        key=f"review_note::{csv_path.name}::{no}",
        height=70,
    )

    def _grade(status: str) -> None:
        review.save_review(csv_path, no, status, note)
        # Auto-advance to the next ungraded-friendly question (just next).
        st.session_state[nav_key] = min(idx + 1, len(df) - 1)
        st.rerun()

    g1, g2, g3 = st.columns(3)
    if g1.button("✅ Approve", use_container_width=True, type="primary"):
        _grade("approved")
    if g2.button("⚠️ Needs review", use_container_width=True):
        _grade("needs_review")
    if g3.button("❌ Reject", use_container_width=True):
        _grade("rejected")

    # --- remove from batch: ONE click (destructive -- edits the CSV -- but
    #     recoverable by regenerating). No confirm gate by request; the
    #     button names the # it'll remove and the persistent note up top
    #     confirms it afterward, so a misclick is obvious and cheap. ---
    st.divider()
    if st.button(
        f"🗑  Remove #{no} from this batch",
        key=f"review_rm_btn::{csv_path.name}::{no}",
        help=(
            "Deletes this question from the CSV in one click. Remaining "
            "questions keep their original numbers (gaps are fine). Can't be "
            "undone here, but you can regenerate the batch."
        ),
    ):
        if review.remove_question(csv_path, no):
            # Stay in this slot so the NEXT question slides into view; clamp
            # against the now-shorter batch. The note up top names what went.
            st.session_state[nav_key] = max(0, min(idx, len(df) - 2))
            st.session_state["_review_last_removed"] = no
            st.rerun()
        else:
            st.warning(f"#{no} was not found in the batch.")


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

    ranges_ok, _ = ranges_pack_status()
    # Reuse the cached, registry-safe accessors (a raw discover_packs() here
    # would re-register the pack and raise on the second page visit).
    packs = _cached_preflop_packs() if ranges_ok else ()
    if not packs:
        st.warning("No range pack found in `ranges/`.")
        return
    pack = packs[0]
    by_actor = _cached_preflop_nodes_by_actor()
    node_by_id = {n.node_id: n for ns in by_actor.values() for n in ns}
    if not node_by_id:
        st.warning("The range pack has no parseable nodes.")
        return

    # Target node: from a Review question, or browse.
    from_q = st.session_state.get("ranges_from_q")
    target_id = st.session_state.get("ranges_node_id")
    if target_id in node_by_id and from_q is not None:
        st.success(f"Showing ranges for review question #{from_q}.")
        if st.button("← Browse all ranges instead", key="ranges_clear"):
            st.session_state.pop("ranges_node_id", None)
            st.session_state.pop("ranges_from_q", None)
            st.rerun()
    else:
        actors = sorted(by_actor)
        col_a, col_b = st.columns([1, 4])
        with col_a:
            actor = st.selectbox("Position", options=actors, key="ranges_actor")
        actor_nodes = sorted(by_actor.get(actor, ()), key=lambda n: n.node_id)
        with col_b:
            target_id = st.selectbox(
                f"Node — {len(actor_nodes)} where {actor} acts",
                options=[n.node_id for n in actor_nodes],
                format_func=lambda nid: (
                    f"{node_action_context(node_by_id[nid])} · {nid}"
                ),
                key="ranges_node",
            )

    node = node_by_id.get(target_id) if target_id else None
    if node is None:
        st.info("Pick a node to view its ranges.")
        return

    # The pack encodes raise sizes as a percent-of-pot token (e.g. "60%").
    # Convert to big blinds using the SAME helper the Question prose uses, so
    # the viewer's amounts match the question text (e.g. "opens to 2.5bb")
    # instead of showing the raw percent token. bb (not dollars) because the
    # ranges are stake-independent.
    from pipeline.preflop.action_history import _raise_size_bb  # noqa: PLC0415

    _verbs = {
        PreflopActionType.FOLD: "folds",
        PreflopActionType.CALL: "calls",
        PreflopActionType.ALL_IN: "shoves all-in",
    }

    def _verb(a: ParsedAction, raise_level: int) -> str:
        if a.action_type is PreflopActionType.RAISE:
            return f"raises to {_raise_size_bb(a, raise_level, pack):g}bb"
        return _verbs.get(a.action_type, a.action_type.value.lower())

    # Tag each prior action with its bet level (1 = open, 2 = 3-bet, ...) so
    # the size converter knows which raise it is.
    leveled: list[tuple[ParsedAction, int]] = []
    _lvl = 0
    for a in node.history_before:
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN):
            _lvl += 1
        leveled.append((a, _lvl))

    st.subheader(f"{node.actor} · {node_action_context(node)}")
    if leveled:
        st.caption(
            "Action so far → "
            + " · ".join(f"{a.position} {_verb(a, lvl)}" for a, lvl in leveled)
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
        segments: dict[str, list[tuple[float, str]]] = {}
        freqs: dict[str, dict[str, float]] = {}
        for hand in all_hands:
            segs: list[tuple[float, str]] = []
            fr: dict[str, float] = {}
            for label, hand_weights, color in acts:
                freq = hand_weights.get(hand, 0.0)
                fr[label] = freq
                if freq > 0.0:
                    segs.append((freq, color))
            segments[hand] = segs
            freqs[hand] = fr
        return segments, freqs

    hero_segments, hero_freqs = _mix_segments(node)
    st.html(range_view.grid_html(hero_segments))

    # --- villains already in: each grid is THAT player's full strategy at
    #     the node where they acted, coloured by action (same legend as the
    #     hero). Fixes the old flat-green grid that made a villain who RAISED
    #     look like they were only calling. ---
    last: dict[str, tuple[ParsedAction, int]] = {}
    for a, lvl in leveled:
        if a.position == node.actor:
            continue
        if a.action_type is PreflopActionType.FOLD:
            last.pop(a.position, None)
            continue
        last[a.position] = (a, lvl)

    villain_freqs: dict[str, dict[str, dict[str, float]]] = {}
    if last:
        st.markdown("### Players already in — their ranges")
        st.caption(
            "Each grid is that player's FULL strategy when it was on them "
            "(same colour legend as above) — not just the one action that "
            "kept them in the pot."
        )
        for pos, (action, lvl) in last.items():
            villain_node = _villain_decision_node(node, pos, pack)
            if villain_node is None:
                st.caption(f"**{pos}**: decision node unavailable")
                continue
            v_segments, v_freqs = _mix_segments(villain_node)
            villain_freqs[pos] = v_freqs
            st.markdown(
                f"**{pos}** {_verb(action, lvl)} — their strategy with each hand"
            )
            st.html(range_view.grid_html(v_segments))

    # --- inspect one hand across everyone (the "click a cell" stand-in) ---
    st.markdown("### Inspect a hand")
    st.caption(
        "Streamlit can't capture a click on a coloured cell, so pick a hand "
        "here to see its exact breakdown across every player."
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
    f1, f2 = st.columns(2)
    with f1:
        positions = st.multiselect(
            "Hero positions",
            options=sorted(_cached_preflop_nodes_by_actor().keys()),
            default=sorted(_cached_preflop_nodes_by_actor().keys()),
            key="cmp_pos",
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
                "After call(s)",
            ],
            default=["Facing single raise", "Facing 3-bet"],
            key="cmp_ctx",
            help="Empty = all action types.",
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
        model_label = st.radio(
            "Model", options=list(_MODEL_LABEL_TO_API), index=0, key="cmp_model",
            help="Compare with the model you'll ship with so verdicts carry over.",
        )
        if "Fable" in model_label:
            st.caption(
                "Fable 5 has no temperature control, so 'deterministic' runs "
                "may still vary slightly in wording."
            )
    band_low, band_high = difficulty_bands[preset]
    s4, s5 = st.columns(2)
    with s4:
        seed = int(
            st.number_input(
                "Seed", min_value=0, max_value=1_000_000, value=42, key="cmp_seed"
            )
        )
    with s5:
        dry = st.toggle("Dry run", key="cmp_dry", help="No API calls — flow check.")

    same = a_slug == b_slug
    if same:
        st.info("Pick two different prompts to compare.")
    if st.button(
        "Run comparison",
        type="primary",
        disabled=same or jobs.has_active_job(),
        key="cmp_run",
    ):
        if not dry and not os.environ.get("ANTHROPIC_API_KEY"):
            st.error(
                "ANTHROPIC_API_KEY is not set. Add it to `.env`, or enable Dry "
                "run to test the flow."
            )
            return
        packs = _cached_preflop_packs()
        if not packs:
            st.error("No range pack found in `ranges/`.")
            return
        pack = packs[0]
        model_api = _MODEL_LABEL_TO_API.get(model_label, model_label)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        PREFLOP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_a = PREFLOP_OUTPUT_DIR / f"compare_{ts}_A.csv"
        out_b = PREFLOP_OUTPUT_DIR / f"compare_{ts}_B.csv"
        with st.status(
            "Running both prompts on the same spots…", expanded=True
        ) as status:
            st.write(f"Prompt A — {names[a_slug]}")
            res_a = generate_preflop_batch(
                pack=pack,
                output_path=out_a,
                total_questions=n_spots,
                hero_positions=positions or None,
                action_contexts=contexts or None,
                min_difficulty=band_low,
                max_difficulty=band_high,
                system_prompt=lib.get_text(a_slug),
                prompt_name=names[a_slug],
                random_seed=seed,
                temperature=0.0,
                model=model_api,
                dry_run=dry,
            )
            st.write(f"Prompt B — {names[b_slug]}")
            res_b = generate_preflop_batch(
                pack=pack,
                output_path=out_b,
                total_questions=n_spots,
                hero_positions=positions or None,
                action_contexts=contexts or None,
                min_difficulty=band_low,
                max_difficulty=band_high,
                system_prompt=lib.get_text(b_slug),
                prompt_name=names[b_slug],
                random_seed=seed,
                temperature=0.0,
                model=model_api,
                dry_run=dry,
            )
            status.update(label="Comparison ready", state="complete")
        st.session_state["cmp_result"] = {
            "a_csv": str(res_a.output_path or out_a),
            "b_csv": str(res_b.output_path or out_b),
            "a_name": names[a_slug],
            "b_name": names[b_slug],
        }
        st.rerun()

    result = st.session_state.get("cmp_result")
    if not result:
        return
    a_csv = Path(result["a_csv"])
    b_csv = Path(result["b_csv"])
    if not (a_csv.is_file() and b_csv.is_file()):
        st.info("Run a comparison above to see results.")
        return

    df_a = pd.read_csv(a_csv, encoding="utf-8-sig", dtype=str).fillna("")
    df_b = pd.read_csv(b_csv, encoding="utf-8-sig", dtype=str).fillna("")
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
                st.markdown(f"**Solver frequencies:** {row_a['action_frequencies']}")
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
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{result['a_name']}**")
                st.info(_md_lines(row_a.get("Answer Explanation", "")))
            with col_b:
                st.markdown(f"**{result['b_name']}**")
                st.info(_md_lines(row_b.get("Answer Explanation", "")))
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


# --- page: Prompt -----------------------------------------------------------
def render_prompt_page() -> None:
    """The prompt library: create, name, edit, and switch between the
    Layer 6 system prompts you're workshopping.

    Preflop only (postflop is blocked on Pio solves; its prompt is shown
    read-only for reference). Prompts live under
    ``admin_panel/prompts/library/`` via
    :class:`admin_panel.prompt_library.PromptLibrary`. The ACTIVE prompt is
    the default for new batches and is mirrored into the legacy
    ``preflop_system.txt`` so any code path that reads
    ``load_preflop_system_prompt()`` stays in sync. The Generate page can
    run any library prompt per batch and tags each output with it.
    """
    st.title("Prompt library")
    st.caption(
        "Create, name, and switch between the system prompts Layer 6 sends "
        "to Claude. The ★ active prompt is the default for new batches; edits "
        "take effect on the next batch — no restart needed."
    )

    mode = st.radio(
        "Pipeline path",
        options=["Preflop (editable)", "Postflop (read-only -- blocked on solves)"],
        index=0,
        horizontal=True,
        key="prompt_mode",
    )

    if mode.startswith("Postflop"):
        try:
            prompt_text = build_system_prompt()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load postflop default prompt: {exc}")
            return
        st.info(
            "Postflop generation is currently blocked (no Pio solves on "
            "Mac). Its system prompt is shown here for reference -- "
            "editing it would have no effect until postflop generation "
            "is unblocked. Use Preflop mode above for live edits."
        )
        st.caption(
            f"Built-in postflop default · {len(prompt_text):,} chars · "
            f"~{len(prompt_text) // 4:,} tokens"
        )
        st.text_area(
            "Postflop prompt (read-only reference)",
            value=prompt_text,
            height=600,
            disabled=True,
        )
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

    # --- rename + notes ---
    m1, m2 = st.columns(2)
    with m1:
        new_title = st.text_input("Rename", value=entry.name, key=f"rename_{sel}")
        if st.button(
            "Save name",
            key=f"renamebtn_{sel}",
            disabled=(not new_title.strip() or new_title == entry.name),
        ):
            lib.rename(sel, new_title)
            st.rerun()
    with m2:
        notes = st.text_input(
            "Notes (what you're trying)", value=entry.notes, key=f"notes_{sel}"
        )
        if st.button(
            "Save notes", key=f"notesbtn_{sel}", disabled=notes == entry.notes
        ):
            lib.update_notes(sel, notes)
            st.rerun()

    # --- the editable prompt text ---
    edited = st.text_area(
        "System prompt",
        value=entry.text,
        height=520,
        key=f"prompt_edit_{sel}",
        help="Edits are session-local until you click Save prompt.",
    )
    if edited != entry.text:
        st.caption(
            f"🔵 Unsaved edits ({len(edited) - len(entry.text):+,} chars vs. saved)."
        )
    if st.button(
        "💾  Save prompt",
        type="primary",
        key=f"save_{sel}",
        disabled=(edited == entry.text),
    ):
        lib.update_text(sel, edited)
        if sel == active_slug:
            _sync_legacy_override()
        st.success("✅ Saved.")
        st.rerun()

    # --- preview the FULL prompt the model receives (sample spot) ---
    with st.expander("👁  Preview the FULL prompt sent to Claude (sample spot)"):
        st.caption(
            "Everything the model receives for one example question, using the "
            "text above as the system prompt. The gold examples, the SOLVER "
            "DATA block, and the instructions around it are assembled per "
            "question — shown here, but not part of the saved prompt. Note the "
            "SOLVER DATA block already feeds `concept_tags` and the villain's "
            "range (`villain_stats.top_combos`); it does NOT feed `skills`."
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
            st.code(parts["assembled"])

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
  validate the winners on Fable 5 (the production model).
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
_PLO_MODELS = ["claude-fable-5", "claude-opus-4-7", "claude-sonnet-4-6"]
_PLO_MODEL_NAMES = {
    "claude-fable-5": "Fable 5 (best quality, 2x Opus price)",
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
        value=True,
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
            options=["LJ", "HJ", "CO", "BU", "SB", "BB"],
            default=[],
            help="Which seats hero is in. Empty = all positions.",
        )
    with hc2:
        action_contexts = st.multiselect(
            "Action faced",
            options=list(PLO_ACTION_CONTEXTS),
            default=["Opening", "Facing single raise", "Facing 3-bet"],
            help="What hero is responding to. Empty = all. (With 'Clean lines "
            "only' on, the 4-bet+ tail stays excluded even if selected.)",
        )
        player_counts = st.multiselect(
            "Players in the pot",
            options=[1, 2, 3, 4, 5, 6],
            default=[1, 2, 3],
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
        index=3,
        horizontal=True,
        key="plo_difficulty_preset",
    )
    if preset == "Custom":
        lo, hi = st.slider("Difficulty rating band", 400, 3200, (400, 3200), step=50)
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
            value=(60, 99),
            key="plo_worthiness_slider",
            help="Below 55% = no clear best answer; 100% = trivial.",
        )
        exclude_ambiguous = st.checkbox(
            "Exclude ambiguous 90-95% band (recommended)",
            value=True,
            key="plo_exclude_ambiguous",
            help="Spots at 90-95% read as 'mostly' but sit just under the 95% "
            "'always' line, so the right read can still be marked wrong. "
            "On = caps the effective ceiling at 90%.",
        )
        min_ev_gap = st.slider(
            "Minimum EV gap (bb) — 0 = off",
            min_value=0.0,
            max_value=3.0,
            value=0.0,
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
    _style_labels = {
        "Basic (Fold / Call / 3-bet)": "basic",
        "GTO (Always / Mostly spectrum)": "gto",
        "Auto-pick (Basic when dominant, GTO when mixed)": "auto",
    }
    style = _style_labels[
        st.radio(
            "Style",
            options=list(_style_labels),
            index=1,  # default to GTO (Always / Mostly)
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
        value=12,
        step=1,
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
        value=42,
        step=1,
        key="plo_gen_seed_val",
        disabled=not _pin_plo_seed,
    )
    seed: int | None = int(_plo_seed_input) if _pin_plo_seed else None
    display_in_bb = (
        bo3.radio(
            "Amounts",
            options=["Dollars", "Big blinds"],
            index=1,
            horizontal=True,
        )
        == "Big blinds"
    )
    out_prefix = st.text_input(
        "Output filename (prefix)",
        value="plo_batch",
        help="A timestamp is appended; every batch lands in its own file.",
    )

    st.divider()

    # --- 5. Model + API settings ---
    st.subheader("5. Model + API settings")
    ms1, ms2 = st.columns(2)
    model = ms1.selectbox(
        "Model",
        options=_PLO_MODELS,
        index=_PLO_MODELS.index("claude-fable-5"),  # default to the best model
        format_func=lambda m: _PLO_MODEL_NAMES.get(m, m),
    )
    _is_fable = "fable" in model
    temperature = ms2.slider(
        "Temperature",
        0.0,
        1.0,
        0.6,
        0.05,
        disabled=_is_fable,
        help="Higher = more varied prose. 0.6 is a good start with no "
        "examples. Fable 5 has no temperature control (ignored).",
    )
    if _is_fable:
        st.caption(_FABLE_NOTE)
    compute_eq = st.checkbox(
        "Compute hand equity for the explanation (~1s/spot; real generate only)",
        value=True,
        help="On = the LLM gets equity numbers to cite, at ~1s/spot (PLO "
        "equity is ~60x heavier than Hold'em). Off = fast. The preview is "
        "always equity-off for speed regardless of this.",
    )
    # Rough per-question estimates by model tier (Fable 5 = 2x Opus list
    # price plus thinking tokens).
    _cost_per_q = 0.45 if "fable" in model else (0.15 if "opus" in model else 0.08)
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
            st.warning(f"{_done['failed']} explanations failed and shipped blank.")
            _reasons = _done.get("failure_reasons") or []
            if _reasons:
                with st.expander("Why did they fail?"):
                    for _r in _reasons:
                        st.markdown(f"- {_r}")
                    st.caption(
                        "Most are the Layer 6 validators (e.g. a card-fabrication "
                        "guard rejecting a card not in the hand, or a banned em "
                        "dash / semicolon) firing on both the attempt and its one "
                        "retry. Regenerate, or edit the prompt to address the cause."
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
        if a_slug == b_slug:
            st.info("Pick two different prompts to compare.")
            run_disabled = True
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
            help="Empty = all action types.",
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
        model = st.radio(
            "Model",
            options=_PLO_MODELS,
            index=0,  # Fable 5 -- compare with the model you ship with
            format_func=lambda m: _PLO_MODEL_NAMES.get(m, str(m)).split(" (")[0],
            key="plo_cmp_model",
        )
        if "fable" in model:
            st.caption(
                "Fable 5 has no temperature control, so A/B runs may vary "
                "slightly in wording between reruns."
            )
    band_low, band_high = _PLO_DIFFICULTY_BANDS[preset]
    seed = int(
        st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, key="plo_cmp_seed")
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

        def _run(out_path: Path, slug: str, include_skills: bool) -> None:
            generate_plo_batch(
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
                answer_style="auto",
                generate_explanations=True,
                explanation_model=model,
                explanation_temperature=0.0,
                explanation_system_prompt=lib.get_text(slug),
                explanation_include_skills=include_skills,
            )

        with st.status("Running both sides on the same spots…", expanded=True) as status:
            st.write(f"A — {a_cfg[2]}")
            _run(out_a, a_cfg[0], a_cfg[1])
            st.write(f"B — {b_cfg[2]}")
            _run(out_b, b_cfg[0], b_cfg[1])
            status.update(label="Comparison ready", state="complete")
        st.session_state["plo_cmp_result"] = {
            "a_csv": str(out_a),
            "b_csv": str(out_b),
            "a_name": a_cfg[2],
            "b_name": b_cfg[2],
        }
        st.rerun()

    result = st.session_state.get("plo_cmp_result")
    if not result:
        return
    a_csv = Path(result["a_csv"])
    b_csv = Path(result["b_csv"])
    if not (a_csv.is_file() and b_csv.is_file()):
        st.info("Run a comparison above to see results.")
        return

    df_a = pd.read_csv(a_csv, encoding="utf-8-sig", dtype=str).fillna("")
    df_b = pd.read_csv(b_csv, encoding="utf-8-sig", dtype=str).fillna("")
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
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{result['a_name']}**")
                st.info(_md_lines(row_a.get("Answer Explanation", "")))
            with col_b:
                st.markdown(f"**{result['b_name']}**")
                st.info(_md_lines(row_b.get("Answer Explanation", "")))
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
                review.save_review(a_csv, no_a, "approved", "finalized from compare")
                review.remove_review(b_csv, no_b)
                st.rerun()
            if fcol_b.button(
                "Save B to finalized",
                key=f"plo_cmp_fin_b_{key}",
                disabled=fin_b,
                use_container_width=True,
            ):
                review.save_review(b_csv, no_b, "approved", "finalized from compare")
                review.remove_review(a_csv, no_a)
                st.rerun()
            if fin_a or fin_b:
                which = result["a_name"] if fin_a else result["b_name"]
                st.caption(f"✅ Saved to finalized using **{which}**.")
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
        options=["Files", "Generate", "Review", "Ranges", "History", "Browse",
                 "Prompt", "Compare", "Skills", "Concept Tags",
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
    st.sidebar.text(f"Ranges: {'✅ ready' if ranges_ok else '❌ missing'}")
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
        "Preflop generation runs end-to-end. Postflop path is wired "
        "but waits for PioSolver `.cfr` solves in `solves/`."
    )

    if page == "Files":
        render_files_page()
    elif page == "Generate":
        render_generate_page()
    elif page == "Review":
        render_review_page()
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
def _render_sidebar_job_indicator() -> None:
    """Tiny sidebar widget that shows current job status.

    Self-refreshing fragment so "running 12/30" ticks up even when the
    user is on a different page. Renders nothing when there's no job.

    IMPORTANT: This fragment writes via bare ``st.info`` / ``st.success``
    / ``st.error`` (NOT ``st.sidebar.X``). Streamlit forbids fragments
    from calling ``st.sidebar`` directly; the caller wraps the
    invocation in ``with st.sidebar:`` instead, and inside that
    context bare ``st.X`` lands in the sidebar.
    """
    job = jobs.get_current_job()
    if job is None:
        return
    if job.is_active:
        p = job.progress
        if p.total > 0:
            pct = ((p.current + 1) / p.total) * 100
            st.info(
                f"🔄 Job: {p.current + 1}/{p.total}  ({pct:.0f}%) · "
                f"{job.elapsed_seconds:.0f}s"
            )
        else:
            st.info(f"🔄 Job: starting · {job.elapsed_seconds:.0f}s")
    elif job.status is jobs.JobStatus.COMPLETED:
        st.success(f"✅ Job done · {job.elapsed_seconds:.0f}s")
    elif job.status is jobs.JobStatus.FAILED:
        st.error("❌ Job failed (see Generate tab)")


if __name__ == "__main__":
    main()
