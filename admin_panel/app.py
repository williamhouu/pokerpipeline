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
from admin_panel import jobs, review, usage  # noqa: E402

# Imports from the pipeline (safe at module load -- these touch no I/O and
# don't require a PioSolver binary or API key to import).
from pipeline.explanation_generator import build_system_prompt  # noqa: E402
from pipeline.fact_extractor.hand_class import STRENGTH_BUCKETS  # noqa: E402
from pipeline.preflop.batch import (  # noqa: E402
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    BatchResult,
    generate_preflop_batch,
    node_action_context,
)
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
_MODEL_LABEL_TO_API: dict[str, str] = {
    "Opus 4.7 (highest fidelity)": "claude-opus-4-7",
    "Sonnet 4.6 (5× cheaper, faster)": "claude-sonnet-4-6",
}

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

# Dedup set: job IDs we've already appended to the usage log. Module-
# level (not session_state) because the same job can be observed by
# multiple browser tabs and the fragment re-renders once a second;
# we want at-most-once-per-process logging without racing the worker.
# Memory: one short uuid per completed real-API batch, so even at 1000
# batches/day this is < 50 KB. No eviction policy needed for the
# admin-panel's typical multi-hour process lifetime.
_LOGGED_JOB_IDS: set[str] = set()

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
            options=["Opus 4.7 (highest fidelity)", "Sonnet 4.6 (5× cheaper, faster)"],
            index=0,
            help="Use Sonnet 4.6 for experimentation, Opus 4.7 for batches "
            "you'll ship.",
        )
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

    # Estimated cost
    est_cost_per_q = 0.40 if model.startswith("Opus") else 0.08
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

    # Filter the node catalog by both filters; show a live count.
    filtered_nodes: list[PreflopDecisionNode] = []
    for actor in hero_positions:
        for node in nodes_by_actor.get(actor, ()):
            ctx = node_action_context(node)
            if not action_contexts or ctx in action_contexts:
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
    with st.expander("Advanced filters (worthiness window · EV-gap gate)"):
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

    # Visible settings summary -- exactly what THIS batch will use.
    _ev_txt = "off" if min_ev_gap == 0.0 else f"≥ {min_ev_gap:.2f} bb"
    st.caption(
        f"**Active settings** — difficulty rating **{band_low}–{band_high}** · "
        f"worthiness freq **{freq_low}–{freq_high}%** · EV-gap gate **{_ev_txt}**"
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
            options=[
                "Opus 4.7 (highest fidelity)",
                "Sonnet 4.6 (5× cheaper, faster)",
            ],
            index=0,
            key="preflop_model",
        )
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

    # Cost estimate
    cost_per_q = 0.40 if "Opus" in _model else 0.08
    est_cost = total * cost_per_q
    st.info(
        f"**Estimated**: {total} questions · ~${est_cost:.2f} · "
        f"difficulty {band_low}-{band_high} · {len(filtered_nodes):,} "
        f"nodes available"
    )

    st.divider()

    # --- 9. Generate button (kicks off a BACKGROUND job) ---
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
            freq_min=freq_low / 100.0,
            freq_max=freq_high / 100.0,
            min_difficulty=int(band_low),
            max_difficulty=int(band_high),
            min_ev_gap_bb=(None if min_ev_gap == 0.0 else float(min_ev_gap)),
            display_in_bb=_currency.startswith("Big blinds"),
            total_questions=int(total),
            output_filename=_out_filename,
            model_label=_model,
            dry_run=bool(_dry_run),
            answer_style=answer_style_canonical,
        )


def _start_preflop_job(
    *,
    pack: PreflopPack,
    hero_positions: list[str],
    action_contexts: list[str],
    freq_min: float,
    freq_max: float,
    min_difficulty: int,
    max_difficulty: int,
    min_ev_gap_bb: float | None,
    display_in_bb: bool,
    total_questions: int,
    output_filename: str,
    model_label: str,
    dry_run: bool,
    answer_style: str,
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
            min_frequency=freq_min,
            max_frequency=freq_max,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
            min_ev_gap_bb=min_ev_gap_bb,
            display_in_bb=display_in_bb,
            answer_style=answer_style,
            model=model_api,
            dry_run=dry_run,
            random_seed=None,
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
in [400, 3200]). The four per-axis values + any bump names that fired
are surfaced as diagnostic columns in the output CSV
(`easy_freq`, `easy_ev`, `easy_concept`, `easy_hand`, `difficulty_bumps`)
so reviewers can see exactly why a spot got its score.
"""
    )

    # --- axis 1: freq ---
    st.markdown(
        f"""
### Axis 1 — Top action frequency &nbsp;·&nbsp; weight **{W_FREQ:.0%}**

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
### Axis 2 — EV gap &nbsp;·&nbsp; weight **{W_EV:.0%}**

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
### Axis 3 — Concept &nbsp;·&nbsp; weight **{W_CONCEPT:.0%}**

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
### Axis 4 — Hand class &nbsp;·&nbsp; weight **{W_HAND:.0%}**

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
    if job.id in _LOGGED_JOB_IDS:
        return
    _LOGGED_JOB_IDS.add(job.id)
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
            f"failures: **{len(result.failures)}**. "
            "Try a wider difficulty band, the Mixed preset, or a wider "
            "worthiness window."
        )
    else:
        _band_note = (
            f", {result.difficulty_filtered_out} rejected by difficulty/EV "
            f"filters" if result.difficulty_filtered_out else ""
        )
        st.success(
            f"Wrote **{result.questions_written}** questions to "
            f"`{result.output_path}` "
            f"(attempted {result.questions_attempted}, "
            f"{result.worthy_spots_available} worthy spots available"
            f"{_band_note})."
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
    st.title("Browse generated questions")
    st.caption(
        "Showing the existing 70-question Tier-1 dataset. Once batches "
        "start generating, this view shows live results."
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
            st.text(f"Hand class:    {row['hand_class']}")
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
    labels = [
        f"{r.filename}  ({r.questions} questions · {r.modified})"
        for r in outputs.itertuples()
    ]
    paths = list(outputs["_path"])
    pick = st.selectbox(
        "Batch", options=range(len(labels)), format_func=lambda i: labels[i]
    )
    csv_path = Path(paths[pick])

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

        st.markdown("**Answer Explanation**")
        st.info(_md_lines(_cell(row, "Answer Explanation")))

        st.markdown(
            f"**Solver frequencies:**&nbsp;{_cell(row, 'action_frequencies')}"
        )

        # Compact strategic facts.
        bits = []
        for col, label in (
            ("archetype", "archetype"),
            ("ev_gap_bb", "EV gap"),
            ("hand_class", "hand"),
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

        # Ranges: tucked away small at the end -- "we know it's there".
        ranges_val = _cell(row, "ranges")
        n_players = review.range_player_count(ranges_val)
        if n_players:
            with st.expander(f"ranges · {n_players} players (rarely needed)"):
                try:
                    st.code(
                        json.dumps(json.loads(ranges_val), indent=2),
                        language="json",
                    )
                except (json.JSONDecodeError, TypeError):
                    st.code(ranges_val)
        else:
            st.caption("ranges: (none)")

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


# --- page: Prompt -----------------------------------------------------------
def render_prompt_page() -> None:
    """The system prompt editor.

    Right now only the **preflop** prompt is editable -- preflop is the only
    path that actually generates questions (postflop is blocked on Pio
    solves; its prompt is shown read-only for reference).

    Edits to the preflop prompt save to ``admin_panel/prompts/preflop_system.txt``.
    :func:`pipeline.preflop.explanation_generator.load_preflop_system_prompt`
    checks for that file at every call -- so edits take effect on the
    NEXT batch you start (no admin-panel restart needed). Reset deletes
    the file, reverting to the built-in default.
    """
    st.title("System prompt editor")
    st.caption(
        "Edit the system prompt Layer 6 sends to Claude. Edits save to a "
        "file under `admin_panel/prompts/` and take effect on the next batch "
        "you start -- no restart needed."
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

    # --- Preflop mode: editable with override file ---
    from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
        build_preflop_system_prompt,
        load_preflop_system_prompt,
    )

    override_path = PREFLOP_PROMPT_OVERRIDE_PATH
    default_prompt = build_preflop_system_prompt()
    active_prompt = load_preflop_system_prompt()
    using_override = override_path.is_file()

    # Status banner.
    if using_override:
        st.warning(
            f"🟡 **Override active.** Edits are loaded from "
            f"`{override_path.relative_to(REPO_ROOT)}`. Click "
            "**Reset to default** below to revert."
        )
    else:
        st.success(
            "🟢 **Using built-in default prompt.** Save your first edit "
            "to switch to override mode."
        )

    st.caption(
        f"Active: **{'override' if using_override else 'built-in default'}** · "
        f"{len(active_prompt):,} chars · "
        f"~{len(active_prompt) // 4:,} tokens (rough estimate)"
    )

    edited = st.text_area(
        "Preflop system prompt (edit live)",
        value=active_prompt,
        height=600,
        key="preflop_prompt_textarea",
        help=(
            "Edits to this box are session-local until you click Save. "
            "Save writes to admin_panel/prompts/preflop_system.txt and "
            "takes effect on the next batch."
        ),
    )

    # Edit-diff indicator.
    if edited != active_prompt:
        diff_chars = len(edited) - len(active_prompt)
        st.caption(
            f"🔵 Unsaved edits ({diff_chars:+,} chars vs. currently active prompt). "
            "Click Save to persist."
        )

    st.divider()

    # --- Action buttons ---
    col1, col2, col3 = st.columns(3)
    with col1:
        save_clicked = st.button(
            "💾  Save",
            type="primary",
            use_container_width=True,
            disabled=(edited == active_prompt),
            help=(
                "Writes the textarea content to "
                "admin_panel/prompts/preflop_system.txt. Next batch picks "
                "it up automatically."
            ),
        )
    with col2:
        reset_clicked = st.button(
            "↺  Reset to default",
            use_container_width=True,
            disabled=not using_override,
            help=(
                "Deletes the override file. Next batch uses the built-in "
                "default from build_preflop_system_prompt()."
            ),
        )
    with col3:
        show_default_clicked = st.button(
            "👁  Show built-in default",
            use_container_width=True,
            disabled=not using_override,
            help="Diff the current override against the built-in default.",
        )

    if save_clicked:
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(edited, encoding="utf-8")
        st.success(
            f"✅ Saved to `{override_path.relative_to(REPO_ROOT)}`. "
            "Next batch will use these edits."
        )
        st.rerun()

    if reset_clicked:
        override_path.unlink()
        st.success("✅ Override deleted. Next batch will use the built-in default.")
        st.rerun()

    if show_default_clicked:
        with st.expander("Built-in default prompt (read-only)", expanded=True):
            st.text_area(
                "Default",
                value=default_prompt,
                height=400,
                disabled=True,
                key="default_prompt_readonly",
            )
            st.caption(
                f"Default: {len(default_prompt):,} chars  ·  "
                f"Override: {len(active_prompt):,} chars  ·  "
                f"Diff: {len(active_prompt) - len(default_prompt):+,} chars"
            )

    st.divider()

    # --- Safety notes ---
    st.subheader("⚠️  Editing the prompt — what to know")
    st.markdown(
        """
- **Test with a dry-run first.** A typo in the prompt can break the
  JSON output format and waste a batch's API spend. Dry-run is free,
  so verify shape before any real generation.
- **The default prompt encodes hard-won lessons** -- 10 voice rules,
  banned phrases, archetype framing, the May 2026 Ryan-feedback fixes.
  Treat rewrites as research, not casual editing.
- **Big prompt changes change Claude's behavior in non-obvious ways.**
  When experimenting, switch to Sonnet 4.6 on the Generate page first
  (~5× cheaper) and only validate the winners with Opus 4.7.
- **The override file is gitignored by default** -- copy your edits
  somewhere safe if you need them across machines.
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
        "The 42-skill catalog `pipeline.skill_tagger.SKILL_CATALOG` maps "
        "the pipeline's computational outputs (archetype + concept tags + "
        "scenario metadata) onto the labels the app surfaces to users. "
        "Use this page to understand exactly why a skill fires on a "
        "given spot."
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


# --- main router ------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Poker Pipeline Admin",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🎰 Poker Pipeline")
    st.sidebar.caption("Preflop pipeline · Phase 3 (skill tagging)")
    page = st.sidebar.radio(
        "Page",
        options=["Files", "Generate", "Review", "History", "Browse", "Prompt",
                 "Skills", "Concept Tags"],
        index=0,
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
    elif page == "History":
        render_history_page()
    elif page == "Browse":
        render_browse_page()
    elif page == "Prompt":
        render_prompt_page()
    elif page == "Skills":
        render_skills_page()
    elif page == "Concept Tags":
        render_concept_tags_page()


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
