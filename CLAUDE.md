# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

/ The full source of truth is `docs/engineering_brief.docx`. This file is a working
summary; when the two disagree, the brief wins. /

## What this project is

`poker-pipeline` is a system that generates **10,000+ poker training questions** with
high-quality, accurate explanations. Each question is a single decision spot (preflop,
flop, turn, or river) in cash or tournament play, output as rows in a CSV/Google Sheet
that matches the team's existing question template.

The product problem this solves: LLMs are confidently wrong about poker (bad equity
numbers, reversed blocker logic, ICM talk in cash games, range advantage assigned to
the wrong player). Generating *scenarios* is easy; generating *correct explanations*
is the hard part.

## The one design principle that governs everything

**The LLM never thinks about poker. The LLM only writes the words.**

Every piece of strategic content — correct answer, equity, frequencies, range shape,
concept tags, decision class — comes from a **solver** and from **plain Python math on
solver output**. The LLM receives a fully-resolved structured data block and turns it
into a 2-5 sentence explanation in the voice of the team's existing gold questions.

> The solver gives you strategic truth. The fact extractor formalizes it. The LLM
> writes prose about it. The validators catch leaks. The CSV writer formats it.

If the LLM is ever *deciding* something strategic ("this is a thin value spot") from
raw EVs, the design has failed — fix the upstream layer (add a rule to the concept
tagger), don't make the LLM smarter. Defend this boundary at every step.

## Current state

This is a **greenfield repo** — no pipeline code has been written yet.

| Path | Status |
|------|--------|
| `venv/` | Python 3.13.13 virtualenv, only `pip` installed |
| `test_solves/btn_vs_bb_srp_2cJs7s.cfr` | ~2.7 GB PioSolver `.cfr` solve: BTN vs BB single-raised pot, flop `2c Js 7s`. This is the Phase 0 end-to-end test solve. |
| `docs/engineering_brief.docx` | Full engineering brief (architecture + rationale) |

So the project sits at **Phase 0**: the proof-of-concept solve exists, but solver
scripting, the range library, and all pipeline modules are still to be built. The
brief asks for the code to be written cleanly from day one — clear module boundaries
between the 8 layers, minimal coupling, easy to refactor — because thresholds and
rules get tuned heavily as the project proceeds.

## The pipeline — 8 layers

Each layer is a Python module with clear inputs and outputs:

```
1 Spot Generator  → 2 Tree Resolver → 3 Path Sampler  → 4 Question Extractor
                 → 5 Fact Extractor → 6 Explanation Generator → 7 Validator → 8 Format Writer
```

1. **Spot Generator** — defines scenario *shells* (game type, stack depth, table size,
   opening action; cards come later). Scenarios are prioritized into tiers (see below).
2. **Tree Resolver** — for a shell, gets the solved game tree. Postflop = a local
   PioSolver Edge solve (30 min–2 hr each, solved to <0.5% pot exploitability).
   Preflop = Monker output or a pre-solved range pack. **Saves the full solver output
   file** to the local cache as the canonical record — never just extracted points.
   The batch driver is `pipeline/batch_solver.py` with CLI at
   `scripts/batch_solve.py`; specs in `pipeline/scenario_spec.py:SOLVER_SPECS`
   and flop sets in `pipeline/flop_sets.py:FLOP_SETS`. See "Adding a new
   scenario" below.
3. **Path Sampler** — sets up individual decision spots inside a solved tree: action
   sequence, hero's hand (from hero's range), board, pot/stack sizes (computed from
   action history). Tags each decision with `parent_node_id` + `action_to_reach`.
4. **Question Extractor** — decides whether a spot is worth a question. Keep only spots
   where the solver's top action sits at **55–95% frequency** AND the **EV gap to the
   best wrong answer is ≥ 0.5 bb**. The same signals (frequency dominance, EV gap,
   concept count) drive the difficulty rating.
5. **Fact Extractor** — *the most important layer.* Emits the structured data block
   (raw solver numbers **and** strategic conclusions: decision class, equity math,
   range shape, blocker effects, concept tags). Every claim in the final explanation
   must trace back to here. See "Concept tags" below.
6. **Explanation Generator** (`pipeline/explanation_generator.py`) — the **only**
   layer that calls an external LLM. One Anthropic API call per spot (Claude
   Opus 4.7 `claude-opus-4-7` by default since June 2026 — the production
   model for every batch; `call_messages_create` drops `temperature` for
   Opus 4.x models that reject it. Sonnet 4.6 is selectable in the admin
   panel for cheap prompt iteration); reads `ANTHROPIC_API_KEY` from the
   environment. Inputs: a populated SpotData, the 8 voice rules distilled from
   the gold examples in `docs/output_format_examples.xlsx` Sheet 2, the brief's
   banned-phrase list, and ~8 in-context gold examples. Output: a
   `GeneratedExplanation` dataclass (`option_1..4`, `correct_answer`,
   `answer_explanation`) — the six LLM-written CSV columns. Option style is
   detected from solver signals before the call (binary action / frequency /
   sizing). Validation: `correct_answer` must equal one of the four options
   exactly; one corrective retry, then `ExplanationValidationError` for Layer 7.
   The system prompt and gold-example block use prompt caching since they are
   identical across thousands of generations. Example retrieval (Phase-2 work)
   will match on **structured features** (street, position, hand class, board
   texture, action context), not raw text — for now the same 8 examples ship
   on every call.
7. **Validator Stack** — five checks in series; a question ships only if it passes all:
   format checker, number checker (equity/pot-odds verified, reject if off >3%),
   strategy checker (LLM rechecks claims vs the data block), failure-pattern checker
   (hunts known LLM-poker errors), voice/format checker. Expect **30–50% rejected on
   first pass**; after regeneration, **<15%** should still need a human. Format and
   number checkers are built up front; the other three are tuned to failures actually
   observed.
8. **Format Writer** — writes the CSV in the team's exact column format.

## Solver stack & conventions

- **Postflop: PioSolver Edge** (~$870, one-time). Edge specifically — it has the
  heads-up preflop solver, multi-machine licensing, 64-core support, and scripting
  (UPI protocol) for batch automation. Solved trees are `.cfr` files.
- **Preflop / multiway: MonkerSolver** (~$530, preferred) **or a pre-solved range
  pack** (~$200–500). Whatever the source, it **must include strategy at every
  preflop decision node**, not just final ranges entering the flop — otherwise
  preflop questions and future hand-replay break.
- **GTO Wizard** is a reference/spot-check tool only; its API closed in 2023.
- *Note:* `docs/engineering_brief.docx` contains an older "Solvers Options" appendix
  recommending PioSolver Pro. It is superseded — the authoritative choice is **Edge**.

Solve conventions that are **architectural decisions, not optional**:

- **Store full solve trees, not extracted decision points.** Storage is cheap
  (~50–200 MB/solve compressed); re-solving everything later is not.
- **Bet sizing trees: ≥2 sizes per node.** Recommended: flop 33%/75%, turn 50%/75%,
  river 50%/75%/overbet(125%). Plus one realistic raise size per street.
- **Preflop data includes intermediate node strategies**, not just final ranges.
- Every question carries `parent_node_id` + `action_to_reach` so full-hand replay
  can be added later with no schema change.

Cache layout — hierarchical by format / scenario / board. The Layer 2 batch
solver writes to `solves/<scenario_name>/<flop_stem>.cfr`; the brief's
deeper hierarchy is for a future migration once we span multiple formats.

```
/solves/Cash6max_100bb_BTN_open_BB_call/2cJs7s.cfr
/solves/Cash6max_100bb_BTN_open_BB_call/AsKd9h.cfr
/solves/Cash6max_100bb_3bet_BB_vs_BTN/...
```

## Layer 2: running batch solves

The driver is `scripts/batch_solve.py`. Required: PioSolver Edge installed
locally (auto-detected via `pipeline.piosolver.find_piosolver` or pass
`--pio-exe`). Each solve takes 30 min–2 hr to hit ~0.5%-pot exploitability
on a 100bb spot.

```bash
# 1. Dry-run first — no compute, just validate the spec + show the plan.
python scripts/batch_solve.py \
    --scenario Cash6max_100bb_BTN_open_BB_call \
    --flop-set MINIMAL_DEBUG --dry-run

# 2. Real run on the single MINIMAL_DEBUG flop (2c Js 7s) — verifies Layer 2
#    produces a solve structurally equivalent to test_solves/btn_vs_bb_srp_2cJs7s.cfr.
python scripts/batch_solve.py \
    --scenario Cash6max_100bb_BTN_open_BB_call \
    --flop-set MINIMAL_DEBUG

# 3. Overnight: 25 flops × scenario count. Resume-safe — re-running skips
#    existing .cfrs, so an interrupted run can be picked up cleanly.
python scripts/batch_solve.py \
    --scenario Cash6max_100bb_BTN_open_BB_call \
    --flop-set STANDARD_25_FLOPS
```

Per-flop failures don't stop the batch — a `.failed` or `.timeout` marker
file is written next to where the `.cfr` would have been, and the next flop
proceeds. The 4-hour-per-solve wall-clock cap raises a `.timeout` marker if
hit (very slow solves should be investigated, not silently absorbed).

### Adding a new scenario

1. **Register the solver spec** in `pipeline/scenario_spec.py:SOLVER_SPECS`.
   The dataclass requires: format, stack_bb, oop/ip positions, OOP/IP
   ranges (Pio range string or .txt path), pot/stack chip geometry, bet
   sizes per actor, raise sizes, accuracy target, bb_in_chips. Naming
   convention: `<Format><TableSize>_<StackBB>_<preflop_action>` (e.g.
   `Cash6max_100bb_3bet_BB_vs_BTN`).
2. **Pick a flop set** from `pipeline/flop_sets.py:FLOP_SETS`, or add a
   new one if STANDARD_25_FLOPS isn't the right coverage. Register the new
   set in the same module and update tests.
3. **Dry-run, then real run.** Phase-0 audit checklist in
   `pipeline/scenario_spec.py`'s docstring describes the placeholder
   ranges to confirm with Ryan before Tier 1 production.
4. **Register a ScenarioConfig** for downstream rendering. Today this is
   hand-authored in `pipeline/scenario_config.py:SCENARIOS`, keyed by
   `.cfr` filename stem. A future refactor will auto-derive ScenarioConfig
   from SolverSpec + flop so the registry doesn't grow linearly with
   solve count.

## Scenario tiers

- **Tier 1 (ship v1):** 10 scenarios, 6-max 100bb cash — 5 single-raised pots + 5
  3-bet pots. ~20 flops each ≈ 200 solves ≈ a 10k–40k question runway.
- **Tier 2:** wider 6-max coverage (more SRP/3-bet positions, 4-bet pots, limped SB).
- **Tier 3:** 9-max cash, MTT stacks (40/25/15 bb).
- **Tier 4:** exotic (squeeze, 5-bet, multiway — needs a separate multiway solver).

Lock Tier 1 and get it producing clean output before touching Tier 2+.

### NLHE preflop range packs (June 2026: two registered)

The NLHE preflop path generates from range packs (no Pio needed), registered
in `pipeline/preflop/pack.py:KNOWN_PACK_SIGNATURES` and selected per batch
via the admin Generate/Compare/Ranges pack selector (choice persists to
`test_output/preflop_batches/.preflop_generate_settings.json`; batch
`meta.json` records `pack_id` + `table_size`):

- **`ryan_preflop_tree_6max_100bb`** — PioViewer format (`ryan_pack`
  grammar, per-position `.txt` folders under `ranges/`), 2.5x opens,
  rake 4%/0.3bb.
- **`monker_nlhe_9max_100bb`** — MonkerViewer export (`monker_nlhe`
  grammar, flat `.rng` files under `nlhe9_ranges/`, gitignored), 93,235
  files = 44,058 nodes, 4x opens, **rake 10%/3bb cap → visibly tighter
  ranges** (UTG RFI 8%; UTG+1 folds QQ 99% vs the UTG open — verified
  real, not a bug). Format + EV-unit calibration (milli-bb from hand
  start; **not** PLO's milli-sb) in `docs/nlhe9_pack_notes.md`;
  re-runnable audit in `scripts/audit_nlhe9_pack.py`. Monker raise
  tokens are pot-relative; bb sizes come from the shared
  `pipeline/preflop/action_history.py:resolve_preflop_history` walk
  (the Ryan pack keeps its `(pct, level)` lookup table), and the
  registered pack quantizes rendered sizes to a 0.5bb grid
  (`size_round_bb`) with the pot math following the rounded game. The
  9 seats flow through as `UTG, UTG+1, UTG+2, LJ, HJ, CO, BTN, SB,
  BB` — the grammar normalizes, downstream never sees Monker's
  UTG1/BU dialect. No pipeline math consumes the pack's file EVs (the
  analytic ev_engine drives `ev_gap_bb`/`easy_ev` for both packs).
  9-max batches default to **Live $1/$2** framing (stakes/venue
  selectors in Generate §7; bb display drops stakes from Context).

### Admin-panel node cache (June 2026 perf fix)

Walking the 9-max pack (93,235 files → 44,058 nodes) takes ~6s, and it
is **I/O/parse-bound** (the panel sat near 0% CPU during the multi-second
"first Generate/Compare/Ranges visit" stall). `pipeline/preflop/node_cache.py`
persists the enumeration to disk (gitignored `.node_cache/`, keyed by a
cheap `scandir`-of-root signature so re-extracting a pack auto-rebuilds;
versioned + atomic-write + corrupt-fallback for safety). Pickling the full
node objects does NOT help — 57 MB / 6.5s load, no faster than re-parsing,
because *materialising* the 44k nested dataclasses is the real cost — so it
keeps **two independent caches**:

- `<pack>.meta.pkl` (~5.5 MB, ~380ms load) — per-node precomputed
  `(actor, action_context, player_count)` via a `derive` callback the admin
  passes (`_meta_derive`, reusing the real `node_action_context` /
  `active_player_count`, so values match the full-node path). Serves the
  sidebar file-count, Generate, Compare, and the filter recount **without
  materialising a single node** → those tabs are instant after a one-time
  ~380ms startup load. `META_CACHE_VERSION` (bump if the derive logic
  changes; blast radius is UI-only — generation never reads this cache).
- `<pack>.nodes.pkl` (~18 MB) — compact descriptors (plain strings/floats)
  from which `descriptor_to_node` rebuilds the FULL byte-identical node,
  loaded **only** where a complete node is needed (Range viewer, prompt
  sampler), so the 18 MB never loads just to render a list.

The two caches are independent (a brand-new pack walks twice, once-ever).
Net: Generate/Compare went ~5.8s→instant; the sidebar file-count is served
from the small cache (was a ~650ms 93k-file rglob every render);
`ranges_pack_status` is `cache_resource` (was a 20k-file glob every 60s).
NOTE: batch generation still walks the pack in-process per run — a future
option is to have `generate_preflop_batch` reuse this cache. Generation
also runs as an in-process thread (`admin_panel/jobs.py`), so a live batch
can still contend with the UI; subprocess jobs remain the documented
future step (judged not worth the risk while the baseline walk was the
real cost).

### Preflop quality loop (June 2026)

The batch-audit protocol + severity taxonomy live in
`docs/quality_audit_playbook.md`; the deterministic re-verifier is
`scripts/audit_preflop_batch.py` (rebuilds every CSV row from the pack
and diffs — equity is per-spot seeded via `fact_extractor._spot_rng`,
runouts 400, so recomputation is byte-identical and ANY drift is a
regression). Layer 6's SOLVER DATA carries multiway-awareness facts
(`other_players_still_in_hand`, `still_to_act_after_you`,
`your_call_or_fold_closes_the_action` from
`action_history.compute_action_pending`) and voice rule 11 forbids
characterizing any range except `villain_stats`. Generation applies two
premise-realism gates by default (`min_villain_line_pct` 0.25%,
`min_hero_premise_freq` 5% — Advanced filters in Generate), and batch
`meta.json` records full `run_settings` + per-question solver-data
blocks (the audit's join key). Audit flags are written into the batch's
`.review.json` sidecar (status `needs_review`, note prefix
`AUDIT (Claude):`) so they render inline on the Review page.

**Round-2 fixes (June 12 PM).** The round-2 audit (20q, all three
round-1 major modes dead; new #1 = false blocker claims) produced:
voice rules 12–16 (blocker claims only from the `blockers` fact, never
invent WHY alternative-action hands prefer their line, position wording
from `hero_position`, equity/flip talk vs the FULL range only,
cold-call/squeeze/open-fold/raise-ladder definitions) in the factory
AND the 3 gitignored prompt snapshots; three new HARD validators in
`pipeline/preflop/validators.py` (suit-emoji-vs-hero-cards,
blocker-claims-vs-blockers-fact, terminology-vs-action-history) wired
into the generation retry stack; a SOFT flag-not-reject path
(`run_preflop_soft_validators`, v1 = position-wording check) that sets
`validation_status="flagged"` + a `validator_warnings` list in the meta
question record; and meta.json `counters` (worthy/filtered/rare-gate
skip counts + `soft_flagged_rows`) next to `run_settings`. Replayed on
the round-2 batch: 7/20 rows hard-fail (all were audit-flagged; zero
hits on clean rows), soft warns on 3. Known non-catches by design:
subject-bound ladder slips inside the reachable bound ("CO folds to a
5-bet" where only CO could make it) and equity-vs-named-hand claims
(rule 15 is prompt-only).

## Postflop pipeline (`pipeline/postflop/`, June 2026)

A **separate, self-contained** flop/turn/river generator, kept apart from the
preflop NLHE and PLO pipelines so work on it can't disturb them. Full docs in
`pipeline/postflop/README.md`. Key facts:

- **Solver-agnostic IR.** Everything runs on `pipeline/postflop/solve.py`
  (`PostflopSolve` / `PostflopNode` / `NodeAction`), NOT a vendor format. A
  thin adapter per source populates the IR; `pipeline/postflop/fixtures.py`
  builds an in-memory BTN-vs-BB SRP `2c Js 7s` solve so the whole pipeline
  runs/tests with **no solver file and no API key** (this Mac can't run Pio;
  the trial `.cfr`/`.db` solves live outside the repo). `validate_solve` is
  the adapter contract.
- **Reuses only pure leaf utilities** (`cards`, `fact_extractor.hand_class` /
  `board_texture` / `equity`, `action_history.format_card`, and
  `explanation_generator.{call_messages_create, GeneratedExplanation,
  BANNED_LITERAL_PHRASES}`). Imports no other pipeline's batch/facts/
  validators/writer; modifies no shared module.
- **Layers**: `spot_sampler` → `question_extractor` (worthy: dominant freq
  65–99%; EV-gap gate optional, off by default) → `facts` (equity, hand class, board texture,
  SPR, pot odds, EV gap, archetype, `concept_tags`) → `action_history`
  (**multi-street**: a turn question renders preflop + flop + turn ahead of
  it) → `options`/`difficulty` → `explanation_generator` (Layer 6, dry-run
  placeholder OR real Anthropic call + 1 retry) → `validators` (deterministic
  hard + soft) → `format_writer` (36-col team CSV: +`neutral_credit`; and
  `ev_gap_bb` column → full per-action `action_ev_bb`, June 2026) →
  `batch.generate_postflop_batch`
  (+ `meta.json`). CLI: `scripts/generate_postflop.py --dry-run`
  (`--solve <path>.db` runs a real vendor solve; `--diversify` for variety).
- **Deterministic**: seeded per-spot equity + sorted spot order + no
  meta timestamps ⇒ byte-identical CSV (guarded by a test). 39 tests in
  `tests/test_postflop_pipeline.py`.
- **Real `.db` adapter DONE (June 2026)** — `pipeline/postflop/adapters/
  sqlite_db.py` reads the third-party SQLite postflop solve into the IR (flop
  nodes for v1). It derives check/call + bet/raise from the node-string
  **betting state** (vendor action *labels* are unreliable — a check-back is
  stored as `CALL`), reach-weights each side's range for equity, and converts
  chips→bb. First real questions came from `BTN_vs_BB_SRP_100bb_QsJd9s_v8.db`
  ([[project-postflop-v8-solve]]); audit it with
  `scripts/audit_postflop_db.py <file.db>`. **NOTE: the `.db` is a POSTFLOP
  solve, NOT a preflop range pack — it does NOT register in
  `pack.py:KNOWN_PACK_SIGNATURES` and does NOT use the preflop `node_cache`;
  those are preflop-only.**
- **Solves self-describe + a solve-picker admin page (June 2026).** Each `.db`'s
  `metadata` table fully describes its scenario, so the adapter no longer
  hardcodes it: `sqlite_db.py:derive_scenario` reads table size (`9max`→9),
  positions (from the `ip_range`/`oop_range` filenames), and cash-vs-tournament;
  `load_postflop_db` applies them (explicit kwargs still win). `summarize_db` /
  `discover_db_solves` read JUST the metadata (no node walk) to list a solves
  folder; `DbSolveSummary.label` is the one-liner the picker shows ("BTN vs BB ·
  9-max · 100bb · Qs Jd 9s · 10% rake"). The admin **Generate → Postflop** page
  (`admin_panel/app.py:_render_generate_page_postflop`) is a *picker*, not a
  filter cascade: choose a `.db` from `solves/postflop/` (gitignored; the user
  drops files there), set count / whose-decisions (BTN/BB) / variety / worthiness
  / stakes, and run via `jobs.start_subprocess_job(generate_postflop_batch_from_db)`
  (`pipeline/postflop/run.py` — the picklable wrapper that ships the `.db` PATH
  to the child, loads + generates there). Spot curation (hero filter + diversify)
  lives in `pipeline/postflop/spot_selection.py` (shared with the CLI). Board-
  texture filters were dropped — pointless with a handful of single-flop solves;
  they'd return only at library scale, derived from each solve's metadata.
- **Postflop Generate/Review parity (June 2026).** The Generate page has the
  preflop controls: **answer-option style** (`options.build_options(spot,
  style=)` — basic plain-labels / gto always-mostly-spectrum-on-binary / auto),
  **display amounts in bb or dollars** (`display_in_bb` threaded through
  `action_history.make_amount_fmt` + `format_writer`), **output filename**,
  model, and worthiness — all threaded `batch → run.generate_postflop_batch_from_db`.
  The **postflop system prompt is admin-editable**: `explanation_generator.
  load_postflop_system_prompt` reads `admin_panel/prompts/postflop_system.txt`
  (gitignored override, else the built-in `POSTFLOP_SYSTEM_PROMPT`); the Prompt
  page's **Postflop mode** edits it (simple single-prompt override, NOT the
  preflop PromptLibrary). **`render_postflop_review_page`** ("Postflop Review"
  nav) mirrors the preflop Review — browse a batch, grade approve/needs/reject,
  edit explanation + difficulty inline (auto-saved) — reusing the GENERIC
  `admin_panel.review` sidecar helpers (they key off any batch CSV + `No`).
- **Postflop worthiness = frequency window only (EV-gap filter is OPTIONAL,
  off by default — mirrors preflop, June 2026).** Real solves mix heavily ⇒
  the interesting decisions (c-bet/lead/size) have ~0 EV gap by construction
  (genuine indifference, not noise), so the brief's hard 0.5bb gate collapsed
  variety to facing-big-bet call/folds (QsJd9s: 144 worthy, all one type).
  Now `evaluate_spot`/`generate_postflop_batch` default `min_ev_gap_bb=None`
  → frequency-only → full variety (1014 worthy, all archetypes/both players),
  same as PLO dropping its EV axis. `min_ev_gap_bb` is an opt-in quality filter
  (suggested value `MIN_EV_GAP_BB=0.5`); the EV gap still feeds difficulty's
  `easy_ev`. `--diversify` round-robins the worthy pool across the 4 flop
  decision types so a fill-to-N batch isn't dominated by one archetype.
- **NOT done (seamed extension points)**: turn/river nodes in the `.db`
  adapter (flop-only today; chance tokens reset the street in the walk), a
  `.cfr` adapter (Pio UPI), the admin **postflop Review** page (Generate is
  built — a postflop Review mirroring the preflop one is the next admin step),
  LLM prompt tuning against gold examples, a **board-emoji hard validator** (the
  LLM occasionally garbles a board suit emoji), and exact Runout app table-state
  token formatting. See the README's "Done vs. next".

## Build phases & status

Each phase ships something usable on its own. Check-in milestones gate phases 2, 3, 4.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **0** | Solver + data infrastructure (Pio Edge scriptable, range source, compute server, batch solve pipeline, range library, one end-to-end test solve) | In progress — test solve exists |
| **1** | Core pipeline end-to-end on Tier 1 → CSV of N questions with difficulty, audit metadata, concept tags | Not started |
| **2** | Validator / QA layer (all 5 checkers, triage queue, auto-regeneration) | Not started |
| **3** | Skill taxonomy mapping (computational tags → app's user-facing skills) | Not started |
| **4** | UI-formatted output (convert pipeline CSV to existing app UI format) | Not started |
| **5** | Admin panel (batch-generation GUI, question lifecycle, cost monitoring) | Not started |
| **6** | Range chart display data (range JSON per question) | Not started |
| **7** | Full-hand replay — *app-layer only, no pipeline work* if Phase 0 was done right | Future |

**Do not skip the Phase 1 proof step.** Build the smallest end-to-end pipeline on one
scenario (recommended: BTN opens vs BB calls SRP — the test solve already covers it),
generate 30–50 questions, and have the poker reviewers grade them. Proceed only if
quality is ≥70% gold-equivalent; otherwise debug before building anything on top.

## Difficulty rating

MVP formula, from solver frequency of the correct answer, mapped onto a 500–3000 Elo
scale (500 = easiest, 3000 = hardest):

```
difficulty_score = 3000 - ((correct_freq - 0.55) / 0.40) * 2500
```

EV gap and concept count can be folded in later for finer calibration.

## Concept tags (Layer 5 detail)

Tags are **computed by Python rules from solver output** — never written by an LLM or
by hand. The library is **42 boolean tags across 7 sections** (range characterization,
decision class, postflop action-type, blocker effects, range advantage, equity/math,
pot/action context) plus 3 edge-case tags needing human review. The tagger is a
registry of pure functions (`def tag_name(spot_data) -> bool`) — trivially extensible
and unit-testable. Naming: lowercase, underscored; `_spot` suffix for decision classes.

Two derived fields live **outside** the tag list (they are descriptive, not strategic):

- `hand_class` — 24 made-hand categories + 7 draw types + 6 strength buckets
  (premium/strong/medium/vulnerable/marginal/air). Computed from hole cards + board.
- `board_texture` — 5 axes (suit distribution, pair status, connectedness, rank
  distribution) + a composite descriptor (dry/semi_wet/wet/very_wet/dynamic/static).

Tag thresholds in the brief are starting values — tune them against the ~800 gold
explanations before the tagger goes to production.

## Output format

Source of truth is **Google Sheets**. The **shared** `CSV_COLUMNS` schema (in
`pipeline/format_writer.py`, used by the PLO writer and the old postflop
writer) is **50 columns**. The **NLHE preflop** writer emits a **40-column
subset** — `PREFLOP_CSV_COLUMNS` in `pipeline/preflop/format_writer.py`, the
shared list minus the 10 columns the NLHE path dropped in June 2026 (see the
trim note after the table). The pipeline writes the team's template columns
via a formatter and adds the new pipeline columns below.

**`neutral_credit` (June 2026).** A deterministic partial-credit column
inserted **right after `Correct Answer`** in all four writers (shared
schema + the self-contained postflop tuple), so the answer-key columns read
together. It lists the "close enough" options (comma-separated; `""` for a
clear spot) that should be held harmless instead of scored wrong. Computed by
`pipeline/neutral_credit.py` (never the LLM) via the **20-point rule**: an
option earns credit when the hand's real solver frequency is within 20 points
of what it claims — `Always X` needs `freq(X) ≥ 80%`, `Mostly X` / bare `X`
needs `freq(X) ≥ 20%`. The full-credit `Correct Answer` is excluded; the rest
are mistakes. Frequency-based on purpose: a solver mixes only when EVs are
~equal, so the EV gap is ~0 across *every* mix (85/15 and 65/35 alike) and
can't tell a thin sliver from a real toss-up — the frequency split can. NB:
inserting it after `Correct Answer` shifts every shared-schema column position
in the table below by +1.

**May-2026 reorg:** `tag_1`/`tag_2`/`tag_3` (the old empty Phase-3
placeholder template columns) were **dropped** — `skills` superseded them.
The **`Relative Position`** column was **repurposed** from the
hero-vs-villain seat matchup to hero's IP/OOP standing (`In Position` /
`Out of Position`); the old matchup string (`BB_vs_BTN`) now lives in a new
**`Position Matchup`** column right after `concept_tags`. Columns were also
reordered: the difficulty/skill/EV diagnostic cluster sits right after
`Difficulty Rating`, and `hand_class` + `Notes` close out the row.

Column numbers below are 1-indexed positions in the **shared/PLO**
`CSV_COLUMNS` order (defined in `pipeline/format_writer.py`). The **NLHE
preflop** CSV drops 10 of these (June 2026 — see the trim note below), so its
positions differ; the column *meanings* below still apply to the ones it
keeps:

| Col | Name | Purpose |
|-----|------|---------|
| 21 | `Relative Position` | hero's IP/OOP standing: `In Position` / `Out of Position`. Postflop reads `meta.hero_in_position`; preflop derives it from postflop action order (`In Position` when hero acts last postflop). (Repurposed May 2026.) |
| 26 | `skills` | comma-separated user-facing skill labels from `pipeline/skill_tagger.py` — the 42-skill catalog the app uses for "study X" features. Distinct from `concept_tags` (computational atoms) — `skills` is the mapped user-readable layer. Strict tagging: 2–5 skills/question typical. (Phase 3, May 2026.) |
| 27 | `action_frequencies` | comma-separated `<verb>: <integer>%` (Fix 3, Apr 2026) — Pio's range strategy at a glance |
| 28 | `ev_gap_bb` | EV gap to second-best action; `<0.30` surfaces questionable questions |
| 29 | `concept_tags` | comma-separated tags (5–10/question); LLM input + retrieval key |
| 30 | `Notes` | provenance string, e.g. "Auto-generated by poker-pipeline (preflop path)." **Sits right after `concept_tags` — June 2026.** |
| 31 | `Position Matchup` | hero-vs-villain seat matchup, e.g. `BB_vs_BTN` (just the hero seat on open spots with no villain). Was the old `Relative Position` value. (New May 2026.) |
| 32 | `ranges` | **Multiway-capable** range column: compact JSON mapping each STILL-ACTIVE player's position to its **full per-hand action mix** — the exact preflop chart for that seat *at the node where it acted*. Shape `{"<POS>":{"<hand>":{"<action>":{"freq":f,"to_bb":b}}}}` where action ∈ `fold/call/raise/allin`, raise/allin carry a bb size, and pure-fold hands are omitted (renderer defaults missing → 100% fold). Hero comes from the current decision node; each villain from the node where *they* last acted (resolved via `_villain_decision_node`, which looks up the villain's own node by `(actor, history_before)`). Built in `pipeline/preflop/format_writer.py:_render_active_ranges` / `_action_mix_for_node`. This is what the app's range UI consumes. **June 2026 rewrite:** replaced the old single-weight-per-hand format (hero was a useless all-1s *presence* mask; villains showed only the one action they took) with the full mix for every position. Replaced the heads-up-only `ip_range`/`oop_range` pair, **dropped May 2026**. |
| 33 | `archetype` | preflop strategic frame: one of the 20 labels in `pipeline.preflop.explanation_generator.PREFLOP_ARCHETYPE_GUIDANCE` (`open_for_value`, `3bet_for_value`, `squeeze_as_bluff`, `fold_dominated`, `call_allin` — a pure pot-odds call of a jam, NO implied odds —, `sb_complete` — SB first-in limp, ported from PLO for the 9-max pack (June 2026) —, `fold_no_continue` — a fold at a node offering NO call, e.g. SB junk-ace vs a 3-bet (fold-or-4bet); never framed around a calling price, added June 2026 to kill a reversed equity-vs-price misfire — etc.) or `unclassified`. Empty for postflop rows (postflop has no archetype layer). The LLM already gets this in its SOLVER DATA block as the strategic frame; the CSV column is for analytics + reviewer QA. (May 2026.) |
| 34 | `board_texture` | single label, e.g. `monotone_connected_broadway` (empty preflop) |
| 35 | `solver_reference` | path back to the exact solve node — the key QA/debugging column |
| 36 | `validation_status` | `draft`/`auto_approved`/`flagged`/`needs_review`/`approved`/`rejected` |
| 37 | `easy_freq` | Difficulty-algorithm diagnostic. Per-spot ease score on the frequency axis in [0, 1] -- 0 = max-hard (55% dominant), 1 = max-easy (100% pure). See `pipeline/preflop/difficulty.py` for the formula. Preflop only; empty for postflop rows. |
| 38 | `easy_ev` | Per-spot ease on the EV-gap axis in [0, 1] -- 0 = max-hard (0bb gap), 1 = max-easy (3bb+ gap). Empty when the EV engine couldn't score the spot (raise-involved spots). Preflop only. |
| 39 | `easy_concept` | Per-spot ease on the archetype-and-concept-tag axis. Lookup table in `pipeline/preflop/difficulty.py:ARCHETYPE_BASE_EASE` plus `CONCEPT_TAG_MODIFIERS`. Preflop only. |
| 40 | `easy_hand` | Per-spot ease on the hand-class axis. U-shaped: premium hands AND clear trash (incl. suited junk like 73s) are easy; marginal hands are hard. Preflop only. |

> **June 2026 schema trim (earlier, 42 → 40 columns):** dropped
> `difficulty_bumps` (always empty — `BUMP_RULES` is unpopulated) and
> `hand_class` (it duplicated `User Cards` on preflop rows; Compare/Review now
> key their spot joins on `User Cards`).
>
> **June 2026 NLHE CSV declutter (NLHE preflop only; PLO keeps the full shared
> schema):** the NLHE preflop writer now has its own `PREFLOP_CSV_COLUMNS` =
> the shared `CSV_COLUMNS` minus 10 columns:
> `pot_odds` / `hero_equity` / `blocker_combos` / `top_villain_combos` (flat
> duplicates of values already inside the kept `stat_notes` JSON),
> `range_equity` (a QA-only number no longer in the panel), `ev_gap_bb`
> (superseded by the per-action `action_ev_bb` column the Review page charts —
> the gap is still computed INTERNALLY for difficulty + the worthiness gate),
> and the four `easy_freq`/`easy_ev`/`easy_concept`/`easy_hand` difficulty
> diagnostics (the `Difficulty Rating` itself stays). Kept the substantive
> decision-math (`stat_notes`) + per-action EV (`action_ev_bb`). The admin
> equity bar reads `hero_equity`/`pot_odds` column-first then falls back to
> `stat_notes`, so PLO (full columns) and NLHE (stat_notes only) both render.
> `_PREFLOP_DROPPED_COLUMNS` in `pipeline/preflop/format_writer.py` is the
> authoritative drop list.

**App-format table columns (May 2026).** The 7 "table-state" columns
`User Seat`, `User Cards`, `Cards on Table`, `Table Size`, `Default
Stack`, `Seats`, `POT` are emitted in the Runout app's exact poker-table
format (e.g. `User Seat` = `HJ-$49-$1.25-raise`, `Seats` =
`SB-$45-$5-3-bet, BB-$50-$0.5-FOLD`, cards as `K-spades, J-spades`) so the
CSV feeds the app's chip/seat renderer directly. Built natively from
structured facts in `pipeline/preflop/app_table_format.py` — a port of the
team's `gto-formatter` engine's output *spec* (that engine regex-parses the
LLM prose; we build from structured data instead). It reuses
`action_history.build_hand_dict`'s resolved dollar amounts so the tokens
always agree with the Question prose + pot. Cash: remaining stacks round to
whole dollars, wagers keep cents; tournament: BB units.

**Difficulty presets are score-band filters (May 2026).** The admin
Generate page's Easy/Medium/Hard/Mixed presets now filter on the COMPUTED
4-axis difficulty rating (Easy 400–1300, Medium 1300–2100, Hard 2100–3200,
Mixed full), not the frequency window. `generate_preflop_batch` takes
`min_difficulty`/`max_difficulty`/`min_ev_gap_bb` and rejects out-of-band
spots *before* the LLM call (no wasted spend); `BatchResult.
difficulty_filtered_out` reports the rejection count. The frequency
worthiness window (default 65–99%) + an optional min-EV-gap quality gate live in the
page's "Advanced filters" expander. (Postflop tab still uses legacy freq
presets — dormant, pending Pio solves.)

The NLHE difficulty algorithm (cols 37-41 are its diagnostics) is a
weighted sum: `easy = 0.40 * easy_freq + 0.30 * easy_ev + 0.20 *
easy_concept + 0.10 * easy_hand`, then `difficulty = round(clip(3000 -
easy * 2500, 400, 3200))`. EV-weight redistributes across the other
three when unavailable. Full details in `pipeline/preflop/difficulty.py`
and the Generate page's "How is Difficulty calculated?" popover (which
reads the constants live).

**PLO drops the EV axis (June 2026).** `pipeline/plo/difficulty.py` is
3-axis — `easy = 0.57 * easy_freq + 0.29 * easy_concept + 0.14 *
easy_hand` (the old freq/concept/hand 4:2:1 split renormalised). A
*worthy* PLO spot is mixed-frequency by definition and therefore
near-equal in EV, so the solver gap is ~0 across the worthiness window
AND redundant with the frequency axis; including it added no signal and
shoved every score up ~350-500 points, making the Easy tier unreachable
(nothing rated below ~1420). `ev_gap_bb` is still computed for the
`min_ev_gap_bb` worthiness gate and the `easy_ev` CSV diagnostic (so
`easy_ev` is populated on PLO rows but does NOT feed the score); it could
be re-added if rescaled to PLO's compressed magnitude (~0.5 bb full
credit) or if the spot pool widens. The PLO popover documents this.

Migrate off Sheets to Airtable/Firestore only around 7k–10k active questions.

### Authoritative format sample — `docs/output_format_examples.xlsx`

A real sample of the team's output, and the **authoritative format spec**. Two sheets:

- **Sheet 1 "Example formatting"** — 6 fully-populated example rows. The baseline
  layout is **35 columns** (0-indexed `No` … `validation_status`): a leading `No`
  (question id), 28 template columns, the 5 brief-spec'd new columns
  (`concept_tags`, `hand_class`, `board_texture`, `solver_reference`,
  `ev_gap_bb`), then `validation_status`. So the brief's "29 existing" = `No` +
  the 28 named template columns. Layer 8 (`format_writer.py`) emits this layout
  plus columns added post-baseline: `action_frequencies` (Fix
  3, Apr 2026), `skills` (Phase 3 user-facing skill tagger, May 2026),
  `archetype` (preflop strategic frame surfaced for QA, May 2026),
  `easy_freq` + `easy_ev` + `easy_concept` + `easy_hand`
  (difficulty-algorithm diagnostics, May 2026),
  `Position Matchup` (May 2026 reorg), and `ranges` (multiway-capable
  position-labeled range JSON, May 2026). Dropped along the way: the three
  empty `tag_1`/`tag_2`/`tag_3` placeholders, the heads-up-only
  `ip_range`/`oop_range` pair (superseded by `ranges`), and — June 2026 —
  `difficulty_bumps` (always empty) + `hand_class` (duplicated `User
  Cards`); `Notes` also moved to right after `concept_tags` — current
  total 40 columns.
- **Sheet 2 "Golden explanation examples - I"** — ~10 sample `Answer Explanation`s
  showing the coaching voice Layer 6's LLM prompt must reproduce.

Format conventions from the sample: headers are lowercase `option 1`…`option 4` and
`Live or Online`; `User Cards` / `Cards on Table` use `rank-suitword` form
(`T-hearts, 9-hearts`); the `Question` narrative and explanations use suit emojis
(♠️❤️♦️♣️); `board_texture` is a 3-axis string (`monotone_connected_broadway`), not
the single `composite` word; `solver_reference` is a descriptive cache-style path;
`Stack Depth` / `Preflop Pot Type` / `Pot Participant` are prose buckets. The current
`format_writer.py` predates this sample and does not yet fully match it.

## Action history format (deterministic, no LLM)

The context block and action-history block before every question are produced by a
plain Python script (~200–400 lines) from structured hand data — zero LLM, zero
variance. Build this first; it is the deterministic foundation everything else sits on.
Key pieces: position-phrase lookup (UTG/UTG+1/UTG+2 take no "the"), verb conjugation
(hero base verb, villain third-person), a raise-level counter
(open→3-bet→4-bet→5-bet), running pot tracking, fold filtering (dropped preflop, kept
postflop), card-emoji formatting, and validation hooks for legal action order.

## Failure modes to watch for

1. **Subtle strategic errors that pass Python validators.** The LLM writes fluent prose
   that quietly misframes a concept. Spend a full day reading raw output yourself to
   calibrate the quality bar before building automated checkers.
2. **Bad example matches.** If retrieval pulls gold examples that aren't truly similar,
   the example block hurts more than it helps. Use structured-feature matching, not raw
   text embeddings.
3. **Strategic thinking creeping back into the LLM** — the biggest architectural risk.
   The data block must always carry the decision class and concept tags. If you ever
   want to "just let the LLM decide what this spot teaches," stop and add a tagger rule.
4. **Solver coverage gaps in the preflop source.** A pack with only final ranges (no
   intermediate nodes) silently breaks preflop questions and hand replay.
5. **Under-spec'd bet sizing trees.** One bet size per node makes questions feel
   artificial. Two per node is the floor.

## Environment

- **Python 3.11+** (`venv/` is 3.13.13). Activate before running anything:
  PowerShell `venv\Scripts\Activate.ps1`, cmd `venv\Scripts\activate.bat`.
- **`ANTHROPIC_API_KEY` is required** for Layer 6 (`pipeline/explanation_generator.py`)
  and the LLM-based Layer 7 checkers. Every other layer is deterministic and runs
  without it. The Layer 6 demo (`scripts/demo_layer6_real_api.py`) skips cleanly
  when the key is unset; tests use a mock client and never touch the real API.
- Installed pipeline dependencies so far: `anthropic` (Layer 6 SDK), `openpyxl`
  (loads `docs/output_format_examples.xlsx` for the gold-example pool), and
  `python-docx` (reads the engineering brief). An embedding API for example search
  and an equity calculator for the number checker are still expected.
- `.cfr` files are large (GBs). Add a `.gitignore` for `venv/` and `test_solves/`
  before `git init` — this directory is not yet a git repository.
