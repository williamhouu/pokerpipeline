"""Streamlit admin panel for the poker question pipeline.

Run from repo root:
    venv/bin/streamlit run admin_panel/app.py

Four pages selected via the sidebar:

  * Files     -- live disk scan of solves/ and ranges/; per-scenario status
                 indicators. Upload widgets are visual-only in this preview.
  * Generate  -- full UI for configuring a batch: cascading filters (format
                 → stack → table → scenarios), content filters (hand class /
                 board texture), difficulty (presets + custom slider),
                 answer style, sampling targets, model + batch size, stake
                 scaling (real -- backed by pipeline.scenario_config.
                 scale_scenario), currency toggle, dry run. The Generate
                 button is disabled until solves are present on disk.
  * Browse    -- table view of test_output/tier1_consolidated.csv to
                 demonstrate what generated-question browsing will look
                 like when batches start producing output.
  * Prompt    -- system-prompt editor; shows the actual default from
                 build_system_prompt(). Save/Test/Reset/Set-default buttons
                 are disabled pending the version-store backend.

Known limitations of this v1 preview:
  * No backend wiring for the Generate button -- this is a UI scaffold to
    validate the design before we hook up the real pipeline.
  * No file-upload handling -- the widgets are visual placeholders.
  * Prompt versioning store not implemented; only the built-in default
    appears in the version dropdown.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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

# Imports from the pipeline (safe at module load -- these touch no I/O and
# don't require a PioSolver binary or API key to import).
from pipeline.explanation_generator import build_system_prompt  # noqa: E402
from pipeline.fact_extractor.hand_class import STRENGTH_BUCKETS  # noqa: E402
from pipeline.preflop.batch import (  # noqa: E402
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
            "The Generate button activates as soon as solves are uploaded "
            "to `solves/`. Tracking via task #4."
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
                "Backend not wired yet. This preview shows the UI design; "
                "real generation lands once Layer 6 batching is implemented "
                "(task #10) and solves are present."
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

    # --- 4. Difficulty (reuses the postflop preset+slider) ---
    st.subheader("4. Difficulty")
    st.caption(
        "How dominant the correct answer is in the solver. Same filter as "
        "postflop -- a 55-95% window is the question-worthy sweet spot."
    )
    preset = st.radio(
        "Preset",
        options=["Easy", "Medium", "Hard", "Mixed", "Custom"],
        index=3,  # default Mixed for preflop
        horizontal=True,
        key="preflop_difficulty_preset",
    )
    presets_map = {
        "Easy": (85, 95),
        "Medium": (70, 85),
        "Hard": (55, 70),
        "Mixed": (55, 95),
        "Custom": (65, 75),
    }
    default_low, default_high = presets_map[preset]
    freq_low, freq_high = st.slider(
        "Solver frequency window (%)",
        min_value=50,
        max_value=100,
        value=(default_low, default_high),
        disabled=(preset != "Custom"),
        key="preflop_difficulty_slider",
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
        value=20,
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
            "Output filename",
            value="preflop_batch.csv",
            key="preflop_out_filename",
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
        f"freq {freq_low}-{freq_high}% · {len(filtered_nodes):,} nodes available"
    )

    st.divider()

    # --- 9. Generate button (active -- runs the full preflop pipeline) ---
    # Inputs ready when: at least one position selected AND at least one
    # action context AND filtered_nodes non-empty AND total > 0.
    can_generate = (
        bool(hero_positions)
        and bool(action_contexts)
        and len(filtered_nodes) > 0
        and total > 0
    )
    if not can_generate:
        st.button(
            "GENERATE BATCH",
            disabled=True,
            type="primary",
            use_container_width=True,
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
        _run_preflop_generation(
            pack=packs[0],
            hero_positions=list(hero_positions),
            action_contexts=list(action_contexts),
            freq_min=freq_low / 100.0,
            freq_max=freq_high / 100.0,
            total_questions=int(total),
            output_filename=_out_filename,
            model_label=_model,
            dry_run=bool(_dry_run),
            answer_style=answer_style_canonical,
        )


def _run_preflop_generation(
    *,
    pack: PreflopPack,
    hero_positions: list[str],
    action_contexts: list[str],
    freq_min: float,
    freq_max: float,
    total_questions: int,
    output_filename: str,
    model_label: str,
    dry_run: bool,
    answer_style: str,
) -> None:
    """Execute one preflop batch and render the result UI.

    Split out of ``_render_generate_page_preflop`` so the button-click
    branch is short + easy to read. Lives in this file (not the
    orchestrator module) because everything here is Streamlit-specific
    -- progress bars, expanders, dataframes, download buttons.
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
    # If the user didn't end the filename in .csv, fix it silently.
    if not output_filename.endswith(".csv"):
        output_filename = output_filename + ".csv"
    output_path = PREFLOP_OUTPUT_DIR / output_filename

    model_api = _MODEL_LABEL_TO_API.get(model_label, model_label)

    # Streamlit progress + status UI. The orchestrator's progress
    # callback fires before each LLM call; we update both the bar and
    # the inline text.
    progress_bar = st.progress(0.0)
    status_line = st.empty()

    def _on_progress(msg: str, current: int, total: int) -> None:
        if total > 0:
            progress_bar.progress(min(1.0, (current + 1) / total))
        status_line.text(msg)

    status_line.text("Starting batch...")
    with st.spinner(
        f"Generating {total_questions} preflop questions"
        + (" (dry-run, no API)" if dry_run else f" with {model_label}")
    ):
        result = generate_preflop_batch(
            pack=pack,
            output_path=output_path,
            total_questions=total_questions,
            hero_positions=hero_positions,
            action_contexts=action_contexts,
            min_frequency=freq_min,
            max_frequency=freq_max,
            answer_style=answer_style,
            model=model_api,
            dry_run=dry_run,
            # Streamlit reruns the script on every interaction -- if
            # the user clicks Generate again, give them a fresh sample.
            random_seed=None,
            progress_callback=_on_progress,
        )

    progress_bar.progress(1.0)
    status_line.empty()

    # Summary.
    if result.questions_written == 0:
        st.warning(
            f"No questions produced. "
            f"Nodes after filter: **{result.nodes_after_filter}**, "
            f"worthy spots available: **{result.worthy_spots_available}**, "
            f"failures: **{len(result.failures)}**. "
            "Try a wider frequency window or different filters."
        )
    else:
        st.success(
            f"✅ Wrote **{result.questions_written}** questions to "
            f"`{result.output_path}` "
            f"(attempted {result.questions_attempted}, "
            f"{result.worthy_spots_available} worthy spots available)."
        )

    if result.failures:
        with st.expander(f"⚠️ {len(result.failures)} per-spot failures"):
            for failure in result.failures:
                st.text(failure)

    if result.output_path is not None and result.output_path.is_file():
        # Download button: read the bytes (utf-8-sig, BOM intact) so Excel
        # picks up the encoding right. mime=text/csv is the standard MIME.
        csv_bytes = result.output_path.read_bytes()
        st.download_button(
            label=f"📥 Download {result.output_path.name}",
            data=csv_bytes,
            file_name=result.output_path.name,
            mime="text/csv",
            use_container_width=True,
        )

        # In-place preview: first ~20 rows of the most useful columns.
        df = pd.read_csv(result.output_path, encoding="utf-8-sig")
        preview_cols = [
            "No",
            "User Seat",
            "User Cards",
            "Hand Stage",
            "option 1",
            "option 2",
            "option 3",
            "option 4",
            "Correct Answer",
            "Answer Explanation",
            "Difficulty Rating",
            "action_frequencies",
        ]
        present_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(
            df[present_cols].head(20), hide_index=True, use_container_width=True
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


# --- page: Prompt -----------------------------------------------------------
def render_prompt_page() -> None:
    st.title("System prompt editor")
    st.caption(
        "Edit the system prompt Layer 6 sends to Claude for explanation "
        "generation. Version everything; test on one spot before "
        "committing to a batch."
    )

    # In v1 preview the only "version" is the current default built in code.
    # When the prompt-versioning backend lands, this dropdown lists saved
    # versions from admin_panel/prompts/*.txt (or a DB table).
    versions = ["v1-default (built-in)"]
    col1, col2 = st.columns([3, 1])
    with col1:
        active_version = st.selectbox(
            "Active version",
            options=versions,
            index=0,
            help=(
                "Versioning UI is wired here, but saved versions need "
                "the backend store. Right now only v1-default exists."
            ),
        )
    with col2:
        st.write("")  # spacer for vertical alignment
        st.button(
            "+ New version",
            disabled=True,
            help="Backend prompt-version store not implemented yet.",
        )

    # Pull the actual current default prompt from the pipeline so what
    # you see here is exactly what generation would send.
    try:
        prompt_text = build_system_prompt()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load default prompt: {exc}")
        return

    st.caption(
        f"Active: **{active_version}** · "
        f"{len(prompt_text):,} chars · "
        f"~{len(prompt_text) // 4:,} tokens (rough estimate)"
    )
    edited = st.text_area(
        "Prompt content (edit live)",
        value=prompt_text,
        height=600,
        help=(
            "Currently displays the built-in default from "
            "pipeline.explanation_generator.build_system_prompt(). Edits "
            "to this box are local to your browser session until the "
            "version-store backend lands."
        ),
    )

    # Show edit indicator
    if edited != prompt_text:
        st.warning(
            f"Edits not yet saved. Diff: {len(edited) - len(prompt_text):+,} "
            "chars from default."
        )

    st.divider()

    # --- Action buttons ---
    st.subheader("Actions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button(
            "💾  Save as new version",
            disabled=True,
            use_container_width=True,
            help="Saves to admin_panel/prompts/<name>.txt (backend pending).",
        )
    with col2:
        st.button(
            "🧪  Test on 1 spot",
            disabled=True,
            use_container_width=True,
            help=(
                "Generates a single question with this prompt so you can "
                "verify output format before running a batch. Needs ANTHROPIC_API_KEY + a sample spot."
            ),
        )
    with col3:
        st.button(
            "↺  Reset to default",
            disabled=True,
            use_container_width=True,
        )
    with col4:
        st.button(
            "✅  Set as default",
            disabled=True,
            type="primary",
            use_container_width=True,
            help="Future batches use this version unless overridden.",
        )

    st.divider()

    # --- Safety notes ---
    st.subheader("⚠️  Editing the prompt — what to know")
    st.markdown(
        """
- **Test on 1 spot before any batch** — a typo in the prompt can break
  the JSON output format and waste a batch's API spend.
- **Versions are tracked.** Every generated question records which prompt
  version produced it. If a batch comes out badly, you can trace which
  prompt is responsible.
- **The default prompt encodes hard-won lessons** (9 voice rules, banned
  phrases, archetype framing, the May 2026 Ryan-feedback fixes). Treat
  rewrites as research, not editing — keep the original around as
  `v1-default` and ship from named alternatives.
- **Big prompt changes change Claude's behavior in non-obvious ways.**
  When experimenting, use Sonnet 4.6 first (cheap) and only validate the
  winners with Opus 4.7.
        """
    )


# --- main router ------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Poker Pipeline Admin",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🎰 Poker Pipeline")
    st.sidebar.caption("Admin panel · v1 preview")
    page = st.sidebar.radio(
        "Page",
        options=["Files", "Generate", "Browse", "Prompt"],
        index=0,
    )

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

    st.sidebar.divider()
    st.sidebar.caption(
        "This is a preview build. Generate is wired to the UI but not "
        "to the backend yet — see task #10/#11 for full integration."
    )

    if page == "Files":
        render_files_page()
    elif page == "Generate":
        render_generate_page()
    elif page == "Browse":
        render_browse_page()
    elif page == "Prompt":
        render_prompt_page()


if __name__ == "__main__":
    main()
