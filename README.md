# poker-pipeline

A system for generating 10,000+ poker training questions with high-quality,
solver-verified explanations. Each question is a single decision spot (preflop,
flop, turn, or river) in cash or tournament play, rendered as one row of a
38-column CSV/Google-Sheet matching the team's question template.

> **The one design principle that governs everything:** the LLM never thinks
> about poker. The LLM only writes the words. Every strategic claim — correct
> answer, equity, range shape, archetype, concept tags — comes from PioSolver
> output and from plain Python math on solver output. The LLM receives a
> fully-resolved structured data block and turns it into 2–5 sentences in the
> team's coaching voice.

## What this produces

- **Tier-1 dataset**: `test_output/tier1_consolidated.csv` — questions across
  all 14 target scenarios (5 SRP, 5 3-bet pot, 4 4-bet pot at 100bb 6-max),
  using the May-2026 38-column schema (including the per-row `ip_range` /
  `oop_range` snapshots for future UI range-grid rendering).
- **Per-batch outputs**: `test_output/batch_questions_v*.csv` — historical
  iteration batches (V5 → V7.3), kept local for diff-against-Tier-1 review.
- **Solves**: `solves/<scenario_name>/<flop>.cfr` — 350 PioSolver Edge solves
  (14 scenarios × 25 flops, ~50–115 MB each, ~25 GB total). Local-only.

## The 8-layer architecture

```
                                ┌────────────────────────────┐
                                │  PioSolver Edge .cfr files │
                                │  (Layer 2 batch_solve.py)  │
                                └─────────────┬──────────────┘
                                              │
                  ┌───────────────────────────┼───────────────────────────┐
                  │                           ▼                           │
            Layer 1                    Layer 3                       Layer 4
        Spot Generator              Path Sampler                Question Extractor
        (scenario_spec)        (enumerate decision nodes)        (filter <55-95%
              │              │                  │              │   freq, EV gap <0.5bb)
              └──────────────┴───────┬──────────┴──────────────┘
                                     ▼
                                Layer 5
                            Fact Extractor
                  hand_class · board_texture · equity · range_data
                  archetype · aggression_history · villain_top_value_combos
                  ip_range_snapshot · oop_range_snapshot · 42 concept tags
                                     │
                                     ▼
                                Layer 6  ── ANTHROPIC API ──┐
                          Explanation Generator             │
                  4 options + correct_answer + 2-5 sentence │
                       explanation in coaching voice        │
                                     │                      │
                                     ▼                      │
                                Layer 7                     │
                              Validator Stack          (1 corrective retry)
                       hard: option_set · archetype_consistency
                       soft: suit_emoji · villain_combo_citation
                                     │
                                     ▼
                                Layer 8
                              Format Writer
                          38-column CSV row out
```

Every layer except Layer 6 is deterministic Python. Layer 6 is the only
external-LLM call site. Layer 7 catches the failure modes a May-2026 audit
caught the LLM committing (notably trap-check spots framed as "villain has
the nut advantage" — see `docs/v7_3_results.md` Fix 5).

## Quick start (Windows 11 + PioSolver Edge installed)

```powershell
# 1. Clone + venv
git clone https://github.com/williamhouu/pokerpipeline.git
cd pokerpipeline
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key (Layer 6 + LLM validators).
#    Get one at console.anthropic.com. ANY other layer runs without it.
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# 4. Verify PioSolver Edge is auto-detected.
python -c "from pipeline.piosolver import find_piosolver; print(find_piosolver())"
# Expected: C:\PioSOLVER\PioSOLVER3-edge.exe

# 5. Run the full test suite.
python -m pytest tests/ -q
# Expected: 292 passed

# 6. End-to-end demo on one scenario (BTN-vs-BB SRP), 1 flop + 2 turn + 2 river
#    = 5 questions. ~6-8 min, ~$0.40.
python scripts/tier1_consolidated_batch.py `
    --scenarios Cash6max_100bb_BTN_open_BB_call `
    --flop-target 1 --turn-target 2 --river-target 2 `
    --out test_output\demo.csv
```

The full Tier-1 consolidated batch is `scripts/tier1_consolidated_batch.py`
with no `--scenarios` flag (defaults to all 14). Expect ~90 min and ~$5–10.

You do **not** need to run the solves yourself — the 350 `.cfr` files are
local to the box already (gitignored, ~25 GB). If you need to regenerate
solves see "How to add a new scenario" below.

## What's done

- **All 14 target scenarios solved** with Ryan's authoritative preflop ranges
  (350 `.cfr` files in `solves/`, full SRP + 3-bet pot + 4-bet pot coverage at
  100bb 6-max). Scenarios 1–14 per `docs/ryan_range_pack_index.md`.
- **Tier-1 consolidated batch** at `test_output/tier1_consolidated.csv` —
  70 questions across all 14 scenarios (5 per scenario: 1 flop + 2 turn + 2
  river per scenario; river-heavy to favour richer learning spots). Single
  CSV with a `scenario` column at index 0 plus the 38 standard columns.
- **5 Ryan-feedback fixes shipped** (verified end-to-end in V7.3 batch,
  `docs/v7_3_results.md`):
  1. Dollar amounts rounded to the nearest $0.25 (no fractional cents in
     prose); whole-dollar amounts render `$50` not `$50.00`.
  2. `Always` / `Mostly` / composite labels only — no ambiguous standalone
     `Sometimes X` / `Rarely X` options. Strict prefix mapping from Pio's
     range-frequency to label.
  3. New `action_frequencies` column shows the exact solver percentages
     (`call: 60%, fold: 30%, raise: 10%`) so QA reviewers can see Pio's
     strategy at a glance without opening the `.cfr`.
  4. Suit emojis (`♠️ ❤️ ♦️ ♣️`) throughout the answer explanation,
     matching the Question column. Plain-text card notation (`Kh`, `AdKd`)
     is banned and caught by a soft validator.
  5. Archetype-aware strategic reasoning: Layer 5 classifies each spot's
     recommended action into 1 of 13 strategic archetypes (`trap_check`,
     `bluff_catch`, `pot_control_check`, `value_bet`, etc.) and Layer 6's
     prompt frames the explanation around that archetype. Villain's range
     is anchored to 2–3 named combos from `villain_top_value_combos`. A
     hard `validate_archetype_consistency` validator catches the V7.1
     failure mode where the LLM matched `nut_advantage_villain` literally
     and wrote the wrong frame on trap-check spots.
- **38-column output schema** (`pipeline/format_writer.py:CSV_COLUMNS`)
  including the May-2026 `ip_range` / `oop_range` columns — per-row 169-entry
  snapshots of each player's range at the decision node, serialised in
  Ryan's preflop-pack format (`AA:1.0,A2s:0.621,...`). Enables future UI
  range-grid rendering per question.
- **LLM scope audited** (`docs/audit_llm_scope.md`): only 6 of 38 columns
  are LLM prose; the four `option N` labels and `Correct Answer` are
  structurally constrained to deterministic frequency thresholds. The
  remaining 32 columns are deterministic Python from solver output,
  scenario config, or pipeline metadata.
- **292 tests passing** (`python -m pytest tests/ -q`). Coverage spans
  every layer, every concept-tag rule, every validator, and the
  archetype classifier.

## What's NOT done

Sized in human days of focused work. None are blockers for the current
Tier-1 dataset.

- **Preflop question generator** — ~3-5 days. Infrastructure 20% built in
  `pipeline/preflop_ranges.py` (combo enumeration, hand-class aggregation).
  Needs the equivalent of Layer 3-5 for preflop nodes (preflop has its own
  range/strategy/EV shape, distinct from a postflop node).
- **Filter capabilities** (board texture, difficulty window, frequency
  window) — ~3-5 days. CSV consumers currently filter post-hoc; an in-
  pipeline filter would cut Layer 6 cost on selective batches.
- **Basic and Sizing option-style variants** — ~2-3 days each. Today Layer 6
  picks one of three option styles (binary action / frequency / sizing); the
  team's mockup also calls for a "Basic" simplified style and a richer
  sizing style with explicit chip/dollar amounts.
- **Stake variations beyond $0.25/$0.50** — ~15 min per stake + ~2 hours
  for a scaling toggle. Stakes are hardcoded in `ScenarioConfig`; adding a
  scaling helper that derives `(sb, bb, default_stack_dollars)` from a single
  buy-in figure is the right pattern.
- **Backend API exposing the pipeline** — ~1 week. FastAPI + job queue +
  state store; the pipeline today is script-driven, no HTTP surface.
- **Frontend UI matching the mockup** — ~1-2 weeks. Next.js or similar.
  The `ip_range` / `oop_range` columns are pre-positioned for the range-grid
  rendering the mockup shows.
- **Multiway preflop scenarios** — ~3-5 days, lower priority. PioSolver is
  heads-up at the flop; multiway needs MonkerSolver or equivalent at the
  preflop layer. None of the Tier-1 14 scenarios are multiway.
- **Production deployment** — ~1 week. CI / staging / monitoring / cost
  caps; today everything runs on the engineer's box.

## Repository structure

```
poker-pipeline/
├── CLAUDE.md                       project / agent context (read this first)
├── README.md                       this file
├── requirements.txt                runtime + test deps
├── docs/
│   ├── engineering_brief.docx      full architecture brief (source of truth)
│   ├── ryan_range_pack_index.md    14 scenarios → range-pack file paths
│   ├── audit_llm_scope.md          per-column LLM-vs-deterministic audit
│   ├── output_format_examples.xlsx authoritative CSV layout + 10 gold examples
│   ├── tier1_consolidated_results.md  per-scenario stats for the Tier-1 run
│   ├── v7_3_results.md             V7.3 verification of the 5 Ryan fixes
│   └── v7_1_results.md / v7_results.md  earlier batch reviews
├── pipeline/                       all 8 layers (every layer is one module)
│   ├── scenario_spec.py            Layer 1 (solver spec) + Layer 2 helpers
│   ├── batch_solver.py             Layer 2 (PioSolver Edge driver)
│   ├── piosolver.py                Pio Edge UPI client
│   ├── path_sampler.py             Layer 3 (decision-node enumeration)
│   ├── question_extractor.py       Layer 4 (worthiness + difficulty)
│   ├── fact_extractor/             Layer 5 (the structured data block)
│   │   ├── __init__.py             orchestrator: extract_facts(...)
│   │   ├── spot_data.py            the SpotData dataclass
│   │   ├── hand_class.py           24 made hands × 7 draws × 6 strength buckets
│   │   ├── board_texture.py        suit/connectedness/rank/composite classifier
│   │   ├── equity.py               hand evaluator + equity-vs-range solver
│   │   ├── equity_range_blockers.py blockers + villain_top_value_combos
│   │   ├── archetypes.py           13-archetype classifier (Fix 5)
│   │   └── concept_tags/           42 boolean rule functions + registry
│   ├── explanation_generator.py    Layer 6 (the only LLM call)
│   ├── validators.py               Layer 7 (4 hard + 2 soft validators)
│   ├── format_writer.py            Layer 8 (CSV row writer)
│   ├── action_history.py           deterministic action-history prose (no LLM)
│   ├── scenario_config.py          per-`.cfr` display metadata + spot_to_hand
│   ├── preflop_ranges.py           169-class aggregation + Ryan-pack format
│   ├── flop_sets.py                MINIMAL_DEBUG / STANDARD_25_FLOPS catalog
│   └── cards.py                    card parsing primitives
├── scripts/
│   ├── tier1_consolidated_batch.py THE tier-1 entry point (14 scenarios → 1 CSV)
│   ├── batch_solve.py              Layer 2 CLI (run STANDARD_25_FLOPS solves)
│   ├── build_ryan_ranges_template.py  Pio-template builder per scenario
│   ├── batch_demo_v6_stratified.py the per-scenario V6/V7 batch driver
│   └── demo_layer*.py              older single-layer demos
├── templates/                      Pio templates per scenario (committed)
│   ├── 2bpot-full.txt              SRP source template (Pio shipped)
│   ├── 3bpot-full.txt              3-bet pot source template (Pio shipped)
│   ├── 4bpot-full.txt              4-bet pot source template (Pio shipped)
│   └── Cash6max_100bb_*_ryan_ranges.txt  14 Ryan-ranges-substituted templates
├── tests/                          292 unit tests
├── solves/                         350 .cfr files (gitignored, ~25 GB local)
├── ranges/                         Ryan's preflop pack (gitignored, ~94 MB)
└── test_output/                    batch CSVs + logs (gitignored except tier1)
```

**Where to look first** for common questions:

- "What does the LLM actually do?" → `docs/audit_llm_scope.md`
- "Why is this column / value what it is?" →
  `pipeline/format_writer.py:CSV_COLUMNS` (the column list) and
  `docs/audit_llm_scope.md` (per-column source).
- "What scenarios exist?" → `docs/ryan_range_pack_index.md`,
  `pipeline/scenario_spec.py:SOLVER_SPECS`.
- "How was the trap-check failure mode fixed?" → `docs/v7_3_results.md`
  (Fix 5 section).
- "How much does a batch cost?" → see "Pipeline economics" below.

## How to add a new scenario

Walkthrough using Scenario 6 (BTN open, BB 3-bet, BTN call) as the worked
example. Substitute your own scenario keys throughout.

```powershell
# 1. Look up the range-pack file paths for the scenario in
#    docs/ryan_range_pack_index.md. For Scenario 6:
#      OOP (BB 3-bet): BB/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%.txt
#      IP  (BTN call): BTN/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_Call.txt

# 2. Register the scenario in scripts/build_ryan_ranges_template.py's
#    SCENARIO_REGISTRY dict (key = scenario name after Cash6max_100bb_).
#    For Scenario 6: "BTN_open_BB_3bet_BTN_call" with the two paths above.
#    Then generate the template:
python scripts\build_ryan_ranges_template.py `
    --pot-type 3bp `
    --scenario BTN_open_BB_3bet_BTN_call
# Output: templates/Cash6max_100bb_BTN_open_BB_3bet_BTN_call_ryan_ranges.txt
# Pot type is auto-inferred from the key ("3bet" -> 3bp, "4bet" -> 4bp,
# else srp); --pot-type only required to override.

# 3. Register a SolverSpec in pipeline/scenario_spec.py:SOLVER_SPECS
#    (use the _3bp_spec / _4bp_spec / _srp_spec helper for the right pot type)
#    AND a ScenarioConfig in pipeline/scenario_config.py:SCENARIOS (use the
#    _srp_scenario_template helper -- it works for any pot type, only the
#    preflop_actions tuple differs). Both files have copy-paste-friendly
#    examples for scenarios 1-14.

# 4. Solve the 25 standard flops (~3-25 min depending on pot type;
#    SRP is slowest, 4bp is fastest). Resume-safe: re-running skips
#    existing .cfr files.
python scripts\batch_solve.py `
    --scenario Cash6max_100bb_BTN_open_BB_3bet_BTN_call `
    --flop-set STANDARD_25_FLOPS
# Output: solves/Cash6max_100bb_BTN_open_BB_3bet_BTN_call/<flop>.cfr × 25

# 5. Add the new scenario to TIER1_SCENARIOS in
#    scripts/tier1_consolidated_batch.py so the consolidated run picks it up.

# 6. Generate questions for just this scenario to verify quality:
python scripts\tier1_consolidated_batch.py `
    --scenarios Cash6max_100bb_BTN_open_BB_3bet_BTN_call `
    --flop-target 1 --turn-target 2 --river-target 2 `
    --out test_output\new_scenario_verify.csv

# 7. Spot-check the answer explanations; if good, add to the next
#    consolidated batch run.
```

Phase-0 audit guidance for confirming a new scenario's ranges with Ryan
before tier-1 production lives in `pipeline/scenario_spec.py`'s docstring.

## How to regenerate a batch

The Tier-1 consolidated dataset is regenerated with one command:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."        # not persisted; session-only

python scripts\tier1_consolidated_batch.py
# Defaults: all 14 scenarios, 1 flop + 2 turn + 2 river per scenario
# = 70 questions; --flop-target / --turn-target / --river-target to override.
# Output:    test_output/tier1_consolidated.csv  (38 columns + 'scenario')
# Wall-clock: ~60-90 min
# Cost:       ~$5-10 on Sonnet 4.6 with prompt caching
```

Smaller verification batches (a single scenario, the old V6 driver):

```powershell
python scripts\batch_demo_v6_stratified.py `
    --scenario Cash6max_100bb_BTN_open_BB_call `
    --target-per-street 4 `
    --out test_output\verify.csv
# 12 questions, ~10 min, ~$0.90.
```

The system prompt and the gold-example block are **prompt-cached** across
calls within a batch (`pipeline/explanation_generator.py:_build_messages_
payload`). The first call in a batch incurs cache-write tokens; subsequent
calls read the cached blocks at 0.1× the input price.

For inline-export of the API key without persisting it to your shell
profile: `$env:ANTHROPIC_API_KEY = "sk-ant-..."` in PowerShell, or
`export ANTHROPIC_API_KEY=sk-ant-...` in bash — both stay in the current
session only.

## Pipeline economics

| Workload | Per-question cost | Notes |
|---|---:|---|
| V7.3 verification batch (12 questions, Scenario 1) | **$0.075** | One scenario, fresh prompt cache. |
| Tier-1 consolidated batch (~70 questions, 14 scenarios) | **$0.06–$0.08** | Cache warms across the batch; cache-read tokens ≈ 4× input tokens. |
| Production scale (10,000 questions) | **$600–$800 one-time** | Linear extrapolation. Cache savings compound when scenarios are batched. |
| PioSolver solve compute | **$0** | One-time $870 for Pio Edge license; no per-solve charge. |

Sonnet 4.6 pricing as of May 2026:
- Input tokens: $3.00 / 1M
- Output tokens: $15.00 / 1M
- Cache-write tokens: $3.75 / 1M (one-time per cache fill)
- Cache-read tokens: $0.30 / 1M (per call thereafter)

Per-batch wall-clock is dominated by the per-scenario Pio walk (~5–10 min
per scenario at default settings), not API calls (~30 s/call). Pio is
single-instance — the 14 scenarios serialise. Parallelising would require
either multiple Pio Edge licenses or a workaround that's beyond the current
license terms.

The V7.3 batch's per-call latency was ~52 s, dominated by the LLM
generating the 4 options + correct_answer + explanation as JSON. Sonnet
4.6's prompt caching kept input cost flat across a long batch.

---

**Engineering contact**: `docs/engineering_brief.docx` is the full original
brief; `CLAUDE.md` is the working summary the engineering team maintains
day-to-day. When the two disagree, the brief wins for design questions and
CLAUDE.md wins for current operational reality.
