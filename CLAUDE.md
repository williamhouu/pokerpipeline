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
  ranges** (UTG RFI 8%). **CORRECTION (June 2026):** a subset (~7) of
  facing-a-single-open nodes show a BROKEN heavy QQ/JJ/TT fold (e.g.
  UTG+1 folds QQ ~99% vs the UTG open). This is a Monker convergence/
  export artifact that collapses the sub-AK continue-EVs to ~0 — AA/KK/
  AKs still 3-bet the SAME node, and QQ continues *less* than JJ/TT,
  which is impossible in a real solve — NOT "verified real" as an earlier
  version of this note claimed. (Folding QQ a lot early-vs-early IS
  directionally real under heavy rake; the broken part is the ~99%
  magnitude + the strength inversion.) Flagged, not dropped, by
  `soft_validate_fold_as_equity_favorite`, and incidentally excluded by
  the default 95–99% near-pure worthiness gate; see
  `scripts/audit_nlhe9_pack.py` and the monker-9max-qq-fold-bug memory.
  Format + EV-unit calibration (milli-bb from hand
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

**July 2026 fact-correctness fixes (from user Review QC):**
- **BvB position bug.** `pipeline/{preflop,plo}/position.py` + both skill
  taggers claimed the blind-vs-blind SB "is the dealer and acts LAST
  postflop" (true only at a 2-player table). At ring tables the BvB SB is
  OOP and the BB is IP; the wrong value flowed into `Relative Position`,
  the SOLVER DATA position line (and thus prose + the position soft
  validator, which reads the same source), and the In/Out of Position
  skills. Exception deleted in all 4 sites; every consumer routes through
  the two position modules; tests pin the ring-table rule.
- **Domination map read a 5-class sample.** `dominating_map` was fed the
  capped `most_common_combos` digest, so `dominated_by: []` meant "none of
  the 5 sampled classes" while reading as a fact about the whole range
  (A8o vs a BTN open showed ZERO dominators; the Layer-7 checker then
  false-flagged correct "often outkicked" prose). New
  `VillainRangeStats.in_range_classes` = EVERY class at weight >= 0.10,
  combo-share ordered; both call sites (data block + exploit notes) use it;
  buckets still cap at 6 for prose. Also fixed the tie-break: the
  "premium-first" canonical order was actually ALPHABETICAL (AA, A2s, A2o,
  ... KK last), surfacing "likely hands: A2o-A6o" on wide ranges -- now
  `_class_strength_key` (AKo, AQo, ... lead equal-weight ties).
- **Reverse Implied Odds vs re-raises.** The RIO concept tag's 20%-width
  gate was calibrated for facing an OPEN; a 3-bet range is dominator-heavy
  at any realistic width (KTo SB-vs-BB-3bet at 25.4% missed the tag). Now:
  facing a re-raise the cutoff is 30% AND the dominant action must be a
  fold (so the tag can't push RIO framing against a correct call); facing
  an ALL-IN never fires (no postflop play = no implied odds either way).

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
  SPR, pot odds, EV gap, archetype, `concept_tags`, **range-vs-range advantage**) → `action_history`
  (**multi-street**: a turn question renders preflop + flop + turn ahead of
  it) → `options`/`difficulty` → `explanation_generator` (Layer 6, dry-run
  placeholder OR real Anthropic call + 1 retry) → `validators` (deterministic
  hard + soft) → **Layer-7 LLM audit (opt-in)**: `claim_checker` (2-call flag)
  / `reviser` (4-call flag→revise→re-audit) → `format_writer` (team CSV:
  +`neutral_credit`, +`range_equity`, +`claim_check`; `ev_gap_bb` column →
  full per-action `action_ev_bb`, June 2026) → `batch.generate_postflop_batch`
  (+ `meta.json`). CLI: `scripts/generate_postflop.py --dry-run`
  (`--solve <path>.db` runs a real vendor solve; `--diversify` for variety;
  `--claim-checker` / `--revise` / `--final-audit` for the Layer-7 pass).
- **Range-vs-range advantage (June 2026, the brief's #1 failure mode).**
  `facts.compute_range_advantage` computes the node-level "who is ahead on this
  board" verdict in PYTHON (never the LLM): `hero_range_equity`
  (`equity.range_vs_range_equity`, fixed seed → deterministic),
  `range_advantage`/`nut_advantage` ∈ `hero`/`villain`/`even` (resolved via
  `_advantage_label` + a margin), and the two strong-made (two-pair+) shares.
  Surfaced as the `RANGE ADVANTAGE`/`NUT ADVANTAGE` lines in the SOLVER DATA
  block + the `range_advantage`/`range_disadvantage`/`nut_advantage`/
  `nut_disadvantage` concept tags + the `range_equity` CSV column. Voice rule 8
  (built-in) / rule 18 (`postflop_system.txt`) BIND any range-advantage claim
  to the fact (mirror it, never compute/reverse). Verified on v8: BTN (the
  aggressor) correctly gets the range+nut advantage, BB the disadvantage.
- **Blocker value/bluff decomposition (June 2026 — the brief's #2 LLM failure
  mode, reversed blocker logic).** `facts.compute_blocker_decomposition` resolves
  in PYTHON whether hero's cards remove villain's VALUE or their BLUFFS:
  classify every board-unblocked villain combo (`classify_hand`) into value
  (premium/strong) vs bluff (air), measure the weighted fraction of each hero
  removes by card-sharing. On a facing-bet node `villain_range` IS the
  reach-weighted betting range, so this is the real composition of the bet hero
  faces. `blocker_effect` ∈ `value`/`bluffs`/`neutral` (resolved verdict;
  thresholds: ≥5% removal + ≥3pt margin). Emits a `BLOCKERS` line in the data
  block ONLY when non-neutral + `blocks_value`/`blocks_bluffs` concept tags.
  **This LIFTS the old "no blocker data" ban**: the prompts now ALLOW blocker
  talk that matches the fact (rule 9 built-in / rule 12 `postflop_system.txt`),
  the claim checker flags blocker claims that CONTRADICT it (was: flagged all),
  and a new soft validator `soft_validate_blocker_direction` flags a reversed
  claim. Real-API proof: the LLM said "you block ~7% of bluffs, a mark AGAINST
  calling" (correct direction for a bluff-catch) and the checker flagged a
  mis-framed one.
- **Curation filters (June 2026 — the postflop analog of preflop's hand-strength
  + action-faced filters).** `spot_selection.py`: `spot_strength_bucket` (hero's
  made-hand bucket via `classify_hand`) + `spot_decision_type` (SITUATION-based —
  C-bet / Lead / Facing-a-bet / Check-back — so filtering never leaks the
  answer). `make_spot_selector(strength_buckets=, decision_types=, aggressor=,
  ip_position=)` filters BEFORE the equity sim (no wasted spend). Admin Generate
  multiselects + CLI `--strength` / `--decision`. `STRENGTH_BUCKETS` /
  `DECISION_TYPES` are the option lists.
- **Solve-quality / node-reach gate (June 2026 — postflop's convergence guard,
  ON by default).** `quality.py:node_quality_issue` skips a node when too few
  hero OR villain combos reach it (<6 — rarely-reached / down-sample-stranded)
  or nearly all combos play one IDENTICAL non-pure mix (an untrained default; a
  uniform PURE action is legit and kept). Matters most for third-party `.db`
  solves + down-sampled turn/river. Wired in `_collect_worthy(quality_gate=)`
  (default True) → `meta.counters.low_quality_nodes_skipped`; admin checkbox +
  CLI `--no-quality-gate`. Verified on v8: skips only the degenerate big-donk
  response lines (3-4 combos), keeps every real node; skipped 104/429 turn+river
  nodes.
- **Layer-7 LLM audit DONE (June 2026, ported from preflop).** `claim_checker.py`
  (postflop-specific prompt: range-advantage-to-wrong-player, mislabeled draw,
  invented blockers, equity-vs-price reversal) + `reviser.py` (prose-only
  rewrite, re-validated by the hard validators, discarded if it breaks one).
  Same toggles as preflop (`run_claim_checker`/`revise_pass`/`final_audit`),
  threaded batch→`run.py`→admin Generate (+ editable
  `postflop_claim_checker_system.txt`). Lifecycle in `meta.counters`
  (`claim_flagged_rows`, `revise_*`) + per-question `revise` record; the
  Postflop Review page reuses the generic claim/revise render helpers + a prompt
  inspector. Real-API proof on v8: 4/4 flagged→fixed, catches were all genuine
  (false blocker, position reversal, overcounted outs, contradictory draw logic).
- **Batch re-verifier DONE** — `scripts/audit_postflop_batch.py <batch.csv>`
  rebuilds every CSV row from the source `.db` (via `meta.provenance`: db_path +
  streets + sampling) and diffs EXACT (deterministic) vs TOLERANCED (MC equity);
  the postflop analogue of `audit_preflop_batch.py`. Proven 0/0 on real Opus output.
- **Deterministic**: seeded per-spot equity + fixed-seed range equity + sorted
  spot order + no meta timestamps ⇒ byte-identical CSV (guarded by a test).
  72 tests in `tests/test_postflop_pipeline.py`.
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
- **Postflop Compare page DONE (June 2026).** `render_postflop_compare_page`
  ("Postflop Compare" nav) is the postflop analog of `render_compare_page`:
  pick a `.db` solve, edit **two free-text prompt boxes** (Prompt A / Prompt B,
  prefilled with the active postflop prompt) and/or pick **two models**, set
  count / heroes / streets / worthiness / claim-checker, and run BOTH sides in
  ONE `jobs.start_subprocess_job(run.compare_postflop_batches_from_db)` — which
  loads the solve once and drives both batches with the SAME deterministic spot
  selector (postflop spots need no shared RNG seed, so both sides see identical
  hands). Side-by-side with A/B/tie verdicts + per-spot finalize, reusing the
  generic `admin_panel.compare` + `review` + `_render_claim_check_panel` helpers.
  Postflop uses two free-text boxes, NOT a PromptLibrary (a full postflop prompt
  library is the remaining optional piece). **Join key fix:** postflop
  `solver_reference` is `…/<node_id>/<combo>`, so the generic join's default key
  (last segment = combo) collides across nodes; `compare.join_by_spot` gained an
  optional `key_fn` and the postflop page passes the full ref (node+combo-unique).
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
- **Difficulty = 3-axis + trap-aware (June 2026, upgraded from 2-axis).**
  `pipeline/postflop/difficulty.py` is now `freq`/`concept`/`hand` (weights
  0.55/0.30/0.15), mirroring PLO — it **drops EV from the score** (a worthy
  postflop spot mixes at ~0 EV gap by construction, so EV is redundant with
  freq; `easy_ev` is kept as a CSV diagnostic only). `easy_concept` =
  `ARCHETYPE_BASE_EASE` (per the 13 postflop archetypes — bluff-catch/trap-check
  hard, value-bet easy) + `CONCEPT_TAG_MODIFIERS` (multiway/wet/range-disadvantage);
  `easy_hand` = U-shaped on strength bucket (premium & clear air easy, the
  medium/marginal middle hard; a real draw nudges a weak hand off "clear air").
  **Trap-aware is an opt-in toggle (off by default, like preflop)**:
  `compute_difficulty(facts, apply_trap_bump=)` floors a counterintuitive
  heads-up facing-a-bet spot (solver folds despite equity ≥ price, or continues
  below it) to a **GRADED floor 1800–2900** (July 2026, was a flat 2400) scaled
  by the |equity − price| contradiction — shared leaf `pipeline/trap_grading.py`,
  same map as preflop. Admin Generate checkbox "🪤
  Trap-aware difficulty" + a "How is postflop difficulty calculated?" popover;
  threaded batch→`run.py`→admin + CLI `--trap-difficulty`; `meta.counters.
  trap_floored` reports the re-rated count. New CSV diagnostics `easy_concept` /
  `easy_hand` (alongside `easy_freq` / `easy_ev`).
- **`skills` column DONE (June 2026).** `pipeline/postflop/skills.py` maps
  `PostflopFacts` → the app's skill catalog (deterministic, never the LLM),
  using the EXACT canonical names from `pipeline/skill_tagger.py:SKILL_CATALOG`
  (a parity test guards against drift). Postflop-native (the shared tagger's
  postflop rules read the OLD `fact_extractor` tag vocab; this reads the new
  `pipeline.postflop` tags), so the package stays self-contained. ~22 skills
  tagged (C-Betting, Facing a C-Bet, Check-Raising, Donk/Probe/Overbet, Bet
  Sizing, Value Betting, Bluffing, Bluff Catching, Floating, Pot Control, Pot
  Odds, Implied Odds, SPR, Range Polarization, In/Out of Position, Multiway,
  Drawing Hand Strategy, + tournament when applicable); **Blockers never fires
  postflop** (no blocker data). Each rule has a plain-English explainer
  (`POSTFLOP_SKILL_EXPLAINERS`) surfaced in the admin **📋 How each postflop
  skill is tagged** dropdown (Generate + Review pages); deliberately-untagged
  skills + why in `POSTFLOP_SKILLS_NOT_TAGGED`. Strict tagging (~2-5/question).
- **App table-state tokens DONE (June 2026).** `pipeline/postflop/
  app_table_format.py:build_postflop_app_table_columns` emits the Runout app's
  exact chip/seat/board render tokens — `User Seat` (`BTN-$97.5`), `Seats`
  (`BB-$95.7-$1.8-bet`, a NEW column between `Default Stack` and `POT`),
  `Cards on Table` (`2-clubs, J-spades, 7-spades` — the board, now non-empty),
  `User Cards`, `Default Stack`, `POT` — all from the same bb-denominated
  amounts the question prose uses (so they never disagree). Built natively (NOT
  by importing the preflop engine — postflop stays self-contained). Differs from
  preflop per the team's `docs/output_format_examples.xlsx` postflop rows:
  **remaining stacks keep cents** (preflop rounds), **both players always render**
  (a postflop decision has both still in), and **per-player remaining is
  reconstructed by a betting walk** over the preflop line + every postflop street.
  `POSTFLOP_CSV_COLUMNS` is +1 (`Seats`); the audit's `EXACT_COLS` includes it.
- **`.cfr` adapter DONE (June 2026, NOT verified on this Mac).** `pipeline/
  postflop/adapters/cfr_pio.py:CfrPioAdapter` maps a PioSolver Edge `.cfr` into
  the IR by driving the solver over UPI (`show_node`/`show_children`/
  `show_strategy`/`show_range`/`calc_ev`/`show_hand_order`). Pio's node-string
  grammar == the `.db` adapter's, so the betting-state walk mirrors it
  (check/call + bet/raise derived from STATE, never a label); `show_range` gives
  each node's range directly (no manual reach-walk); `calc_ev(actor, child)`
  gives per-action EVs. Written against a `UpiClient` `Protocol` + unit-tested
  with a mocked client (`tests/test_postflop_cfr_adapter.py`). **This Mac can't
  run Pio (Windows-only), so it is NOT verified end-to-end** — full integration
  needs a Windows host with Pio Edge + a `.cfr` (e.g. `test_solves/
  btn_vs_bb_srp_2cJs7s.cfr`), driven via `load_postflop_cfr`. v1 = flop nodes
  (turn/river is a BFS-through-chance-nodes extension; the walk already handles
  chance tokens). `validate_solve` is the contract both adapters target.
- **`currently_ahead` showdown-equity fact DONE (June 2026 — the brief's #1 LLM
  failure mode, mis-attributing a made hand's equity to draws).**
  `facts.compute_currently_ahead(hero_cards, villain_range, board)` resolves in
  PYTHON the reach-weighted share of villain's range hero's hand BEATS at
  showdown RIGHT NOW (exact 5-/6-/7-card comparison via the shared evaluator —
  no runouts, so it's perfectly deterministic, unlike sampled `hero_equity`).
  Ties count as neither; board/hero-blocked combos excluded; empty range → 0.
  `PostflopFacts.currently_ahead_pct`/`currently_behind_pct`; a `CURRENTLY AHEAD`
  data-block line (worded "the hands villain is betting" on a facing-bet node,
  else "villain's range") + a `chat_context` field. The point: a small pair
  AHEAD of 50% of villain's betting range has SHOWDOWN equity (it beats their
  air now), NOT draw equity — so the LLM stops writing "your equity is backdoor
  outs / spiking a deuce." Generation rule 10 (built-in) / rule 19
  (`postflop_system.txt`) bind the equity-source narrative to this line + the
  DRAWS line; a new claim-checker bullet flags equity pinned to draws when the
  hand is a made hand ahead of the range. PURELY ADDITIVE — does not touch
  equity/pot-odds/worthiness/difficulty/tags (one `PostflopFacts` construction
  site). Verified: 22 facing a K73 c-bet beats 49% (its air); a set beats 99%.
- **0.5bb display rounding DONE (June 2026, ALL pipelines).** Solver sizes are
  pot fractions, so bb amounts land on ugly values (2.14bb). NEW shared leaf
  `pipeline/bb_display.py:round_to_half_bb` snaps DISPLAYED bb amounts to the
  nearest 0.5bb (2.14→2, 4.36→4.5, 7.8→8). **Display-ONLY** — every strategic
  fact (equity, pot odds, EV, SPR, worthiness, difficulty, concept tags) uses
  the EXACT amount; rounding the IR geometry would shift the pot-odds price and
  could flip a borderline tag, which we explicitly avoid. Applied in: postflop
  (`action_history.make_amount_fmt`, `app_table_format`, the data block's
  POT/stack/to_call, the `Raise to X bb`/`Bet X bb` adapter labels — `pot_fraction`
  "Bet 33%" labels stay exact, computed from chips); PLO (`_money`,
  `app_table_format._fmt_bb`, the villain-action line); preflop already
  quantizes via each pack's `size_round_bb=0.5` (now on the Ryan pack too, a
  verified no-op). **Dollar display untouched.** Accepted cosmetic caveat: a
  multi-street pot from several independently-rounded wagers can read ≤0.5bb off
  the literal sum (single-bet spots stay exact); `_seat_states` keeps EXACT
  floats so the stack-invariant test still holds (proving display-only).
  Deterministic ⇒ byte-identical CSV + audit 0/0 still hold.
- **Claim-checker reliability — best-of-N gate (June 2026).** The claim checker
  is non-deterministic even at temperature 0 (a single pass can miss a real
  issue, so a batch sometimes got a "lucky clean" and no rewrite fired). The
  **revise gate now runs `_REVISE_GATE_PASSES=2` passes and UNIONs the issues**
  (`batch._gate_check_best_of`) so a flaky miss can't slip through; the flag-only
  path stays one call. Also strengthened the checker prompt to catch an
  incoherent CLAUSE (subject that no longer fits its predicate, e.g. "the hands
  that continue ... fold out the bluffs" — calling hands don't fold anything
  out). The Review page's "clean" case is now a prominent st.success ("🔎 Layer-7
  audit ran — came back CLEAN") so a clean question shows EVIDENCE the audit ran,
  not nothing. Verified live: the muddled sentence went from 0/3 caught to 2/2.
- **Stack depth in the Context when ≠ 100bb (June 2026, all pipelines).** Readers
  assume 100bb, so a non-100bb game must say so. Postflop `build_context_line`
  appends `{stack}bb` for cash when `round(effective_stack_bb) != 100` (e.g.
  "Live · $1/$2 · 200bb · 8% cap 2bb rake"); preflop `_context_column` appends it
  when `pack.stack_depth_bb != 100` (the 20bb/30bb packs). 100bb stays clean.
- **Range visuals = "the street before" + current (June 2026).** The Review
  range panel's LEFT grid was always the flop-entry (preflop) range; now a TURN
  question shows the FLOP range and a RIVER question the TURN range (per-question
  `prior_street_ranges` + `prior_street_label`, via `batch._prior_street_node` =
  the deepest prior-street decision ancestor by node-id prefix — flop nodes are
  never down-sampled so a turn question always resolves; flop questions stay on
  the shared preflop ranges). RIGHT grid unchanged (current-street strategy).
- **Pack-backed preflop legs for full-hand play-throughs (July 2026).**
  `pipeline/postflop/preflop_leg_pack.py` — the ONE sanctioned
  cross-pipeline seam (lazy preflop imports; nothing else in
  `pipeline/postflop` may import `pipeline.preflop`). When a preflop range
  pack provably matches the solve's preflop line (3 gates: geometry — same
  table size/stack/open size within 0.26bb, derived from the flop pot;
  line — the opener + defender nodes exist with all-folds-before; and
  per-hand coherence — the pack's dominant action == the as-played
  action), the preflop leg is built with the FULL preflop pipeline
  (per-action EVs, GTO options under the EV-secondary rule, stat_notes,
  `ranges` JSON, domination, skills, 4-axis difficulty + trap/razor,
  soft validators) and adapted onto the postflop schema; otherwise the
  entry-derived leg is the fallback (per-hand). Auto-selected
  (IMPROVED-preferred, deterministic) from `ranges/` by
  `run.generate_full_hand_batch_from_db(use_pack_preflop_legs=True)`;
  verified live: v7 (8-max 200bb) matches `preflop_8max_200bb_IMPROVED`
  at 3.0bb, v8 (9-max) correctly falls back. Counters
  `preflop_leg_pack_used` / `preflop_leg_entry_fallback`; run_settings
  `preflop_leg_pack`; `audit_full_hand_batch.py` rebuilds pack legs
  (0/0 on the real v7 batch). Tests: `tests/test_full_hand_pack_legs.py`.
- **Full-hand difficulty (July 2026).** Every leg keeps its own per-
  question `Difficulty Rating` (the app's per-question scoring); a new
  END-of-schema column **`hand_difficulty` = MAX over the hand's legs**
  (same value on every leg; blank on standalone rows) is the hand-level
  selector — a hand demands what its hardest decision demands, so a mean
  would wash out a 2400 river bluff-catch behind three easy calls.
  Computed in a pre-LLM pre-pass (facts cached and reused by generation,
  no double equity sims) so the optional hand band filter
  (`min/max_hand_difficulty`; admin full-hand mode radio Easy/Medium/
  Hard/Mixed/Custom) costs no tokens; `counters.hands_difficulty_
  filtered`. The full-hand admin mode also exposes 🔪 razor's-edge for
  the pack-backed preflop leg. Postflop GTO options: a PURE 3+-verb spot
  now spectrums the dominant verb vs the SECOND-BEST verb BY EV
  (`options._best_ev_alternative_verb`, the same standing rule as
  preflop; plain-labels fallback when the solve ships no EVs).
- **NOT done (seamed extension points)**: a postflop **prompt library** (the
  Compare page uses two free-text boxes today); LLM prompt tuning against gold
  postflop examples; the harder-to-detect skills (`Facing a Check-Raise`, MDF,
  Reverse Implied Odds — see `POSTFLOP_SKILLS_NOT_TAGGED`); real-host `.cfr`
  verification + turn/river in the `.cfr` adapter. (DONE since earlier notes:
  turn/river `.db` nodes, board-emoji validator, the postflop Review **and
  Compare** pages, the Layer-7 claim-checker/reviser, range-vs-range advantage,
  the batch re-verifier, 3-axis+trap difficulty, the `skills` column, blocker
  value/bluff decomposition, the curation filters, the solve-quality/node-reach
  gate, the **app table-state tokens**, and the **`.cfr` adapter**.)

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
writer) is **51 columns**. The **NLHE preflop** writer emits a **41-column
subset** — `PREFLOP_CSV_COLUMNS` in `pipeline/preflop/format_writer.py`, the
shared list minus the 10 columns the NLHE path dropped in June 2026 (see the
trim note after the table). The pipeline writes the team's template columns
via a formatter and adds the new pipeline columns below.

**`chat_context` (June 2026) — the AI-chatbot column.** The app's per-question
chatbot (a user chats about a spot after answering) is fed this ONE
self-contained JSON blob of all deterministic truth, so it never invents poker
facts. Built by the shared `pipeline/chat_context.py:build_chat_context`
(a pure formatter, like `neutral_credit`) — ONE schema across all four writers
(preflop, postflop, PLO, + the self-contained postflop tuple). It carries MORE
than the generation SOLVER DATA block, because a chatbot user asks comparative
+ hypothetical questions: the **full action strategy with per-action EVs** (not
just the recommended action), the **villain's likely hands**, the
**neutral-credit alternatives**, the **explanation already shown**, all the
facts (equity incl. multiway field + range-vs-range, pot odds, SPR, range/nut
advantage, blockers value/bluff, board texture, archetype, concept tags,
skills, difficulty), and a **`guardrails`** line baked in ("answer only from
this data; never invent numbers; for a hypothetical hand, say it's outside this
spot's data"). The app's chatbot system prompt should ALSO enforce that rule.
Last column in every writer.

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

**Difficulty ceiling + fail-fast (July 2026).** The score is bounded
above by a pure function of the dominant frequency alone:
`difficulty.max_achievable_difficulty(min_frequency, trap_difficulty=)`
(mirrors the batch's near-pure EV credit; INVARIANT guarded by
`tests/test_preflop_difficulty.py`). A pure 100% spot can never rate
above **1125** (measured on the 8-max packs: ~875), so "Medium/Hard band
+ 100% frequency" is structurally EMPTY unless trap-aware is on (traps
grade 1800–2900 there). Two consumers: `generate_preflop_batch` skips
out-of-reach spots BEFORE the ~1.5s/spot equity sim in `extract_facts`
(a doomed 6k-spot scan used to grind ~2.8h; now instant, still counted
in `difficulty_filtered_out`), and the admin Generate page shows an
upfront error/warning when the band + frequency combo is unreachable
(`difficulty.classify_band_reachability` — pure, browserlessly tested —
including the trap-on case where the band misses the 1800–2900 range).

**Trap-aware difficulty (opt-in June 2026; GRADED floor July 2026).** The
weighted sum is ~70%
frequency+EV, which BOTH say "easy" exactly when the solver is
near-indifferent — so by default "hard" === close-mix spots, and a PURE
(100%) spot can never exceed difficulty ~2000 (caps below the Hard floor
2100). To make genuinely counterintuitive PURE spots rate Medium-to-Hard,
the `trap_difficulty` batch flag (admin Generate checkbox "🪤 Trap-aware
difficulty" + its own info popover; `compute_difficulty(...,
apply_trap_bump=)`) floors a **trap** spot to a **graded value 1800–2900**
(`pipeline/trap_grading.py:graded_trap_floor`, shared with postflop; was a
flat 2400) scaled by the RAKE-BLIND |equity − price| contradiction: the
detection-threshold margin (4pts) grades 1800, saturation (25pts) grades
2900, and the median 8-max trap (~16pts) grades ~2430 ≈ the old flat
floor. Detection is unchanged: a trap = the solver's dominant action
CONTRADICTS the
equity-vs-price pot-odds baseline by a clear margin (`_TRAP_EQUITY_MARGIN`
0.04, + a rake cushion for folds): folds despite equity ≥ price, or
calls/3-bets despite equity <
price (`difficulty._is_counterintuitive_spot`). HEADS-UP facing-a-bet
spots only (opening spots have no price → never traps; multiway is
skipped — field-equity-vs-price is mis-specified and ~all such hits were
degenerate deep-all-in nodes, measured 88% of raw trap hits). It changes the SCORE only,
never the answer/options/prose/worthiness; OFF by default = unchanged
behaviour; `meta.counters.trap_floored` reports how many were re-rated.
Deliberately the same signal as `soft_validate_fold_as_equity_favorite`,
so trap-FOLDs are also soft-flagged for review (guards against rating a
broken solve like the Monker QQ-fold as "hard"). Recommended ON for
Medium and Hard batches (mild traps now land in Medium), off for Easy.
The three audit re-verifiers mirror the flag from `meta.run_settings`
(July 2026) so trap-aware batches re-verify without false difficulty
drift.

**Razor's-edge difficulty (opt-in, July 2026).** The OTHER route (besides
trap-aware) to Medium/Hard 100%-frequency questions: a pure hand sitting
ON a range boundary — its grid NEIGHBOR does the opposite at the same
node (ATo folds where AJo calls / the ATs twin raises / TT's adjacent
pairs differ) — is floored to a GRADED value by how many neighbors oppose
it (`pipeline/preflop/razor_edge.py`: 1 → 2000, 2 → 2300, 3+ "island" →
2600). Detection is pure range-file math via `sample_spot` on the
neighbor classes (presence-gated; raise SIZES don't count as opposite);
`compute_difficulty(..., apply_razor_bump=)`, batch flag
`razor_difficulty`, `meta.counters.razor_floored`, admin checkbox "🔪
Razor's-edge difficulty" + popover next to trap-aware. Composes with the
trap floor (higher wins). `max_achievable_difficulty` /
`classify_band_reachability` know both modes (the "trap_only" verdict is
now `"special_only"`); the audit re-verifier mirrors the flag from meta.

**Sanity audit (opt-in Layer-7 pass, July 2026).** The ONE LLM pass
allowed to use its own poker knowledge, pointed at the SOLVER DATA facts
(never prose): `pipeline/preflop/sanity_checker.py`, batch flag
`run_sanity_audit` (admin checkbox "🩺 Sanity audit" under the Layer-7
section). Exists because every other check verifies internal consistency
and therefore AGREES with a wrong deterministic fact (the BvB position
bug + empty-domination bug shipped that way). Strictly flag-only
(hypotheses for human review — LLMs are confidently wrong about poker,
the project's founding premise), fails open, restricted by prompt to
basic high-confidence facts (action order, domination direction, equity
plausibility ±10pts, arithmetic, action-history consistency). Flags land
in the meta question record (`sanity_check_issues`) + `counters.
sanity_flagged_rows`, rendered on Review with a distinct 🩺 badge that
warns the checker itself can be wrong. **v2 (July 5 2026): the first
live calibration produced 7 flags, ALL false positives — one stated the
BvB rule backwards.** v2 embeds the deterministic REFERENCE RULES (seat
order, break-even arithmetic per seat, gapper counting, suited-twin
equity) + each v1 misfire as a do-not-flag counter-example, and a flag
ships only when **two independent passes challenge the same fact**
(`consensus_issues`; the confirm call runs only on first-pass flags).
Awaiting live re-calibration (API credits). Keep OFF for routine
batches; the deterministic cross-check below covers its fact categories
with zero false positives.

**Deterministic batch cross-check (auto-run, July 2026).**
`pipeline/preflop/batch_cross_check.py` re-reads every batch CSV AS
WRITTEN at the end of `generate_preflop_batch` and verifies it against
FIRST-PRINCIPLES facts — its own seat-order list (deliberately NOT
imported from `pipeline.preflop.position`; independence is the point),
position skills, BvB skill hygiene, domination-list direction + the
empty-`dominated_by`-vs-likely-hands challenge, difficulty band
membership, frequency sums, no-RIO-on-all-in, and hero-subject prose
position claims. Zero LLM, zero false positives in calibration. Findings
→ meta `cross_check_issues` + `counters.cross_check_problems`, rendered
on Review as an always-visible 🔬 error badge (machine-verified, not an
AI opinion). Re-check any OLD batch with
`scripts/cross_check_preflop_batch.py <batch.csv>`.

**Ground-truth test tier (July 2026).** `tests/test_ground_truth_poker.py`
asserts poker facts from OUTSIDE the pipeline (ring-table position
mechanics, domination direction, preflop equity landmarks like AA-vs-KK
~81%, implied-odds logic) so an internally-consistent bug still fails CI.
Rule stated in the file header: a failure there means the CODE is wrong —
never adjust an assertion to match the code without independently
confirming the poker fact.

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

**`Question Type` (June 2026):** a fixed label `Hand Scenario Question` with **NO
trailing period**, emitted identically by all four writers (preflop, postflop,
PLO, shared). Was `Hand Scenario Question.` (with period) on three writers and
`Postflop Decision` on the postflop writer; unified + de-periodised per team
feedback. (The `docs/output_format_examples.xlsx` sample shows sentence-case
`Hand scenario question`; the code uses title-case per the team's direct ask —
flag if they want the sheet's casing instead.)

**bb amounts render on a 0.5bb display grid** (`2bb` / `4.5bb`, never `2.14bb`)
via `pipeline/bb_display.py` — display-only; the dollar path and all strategic
math use the exact amounts. See the postflop "0.5bb display rounding" bullet.

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

## Fix durability — make regressions impossible, not just gone

The admin panel (Streamlit) has a history of the SAME bug returning every few
days — most infamously "edit an Answer Explanation in Review, navigate away,
come back, edit gone." A bug recurs when all three hold: the logic lives in an
**untestable framework seam** (Streamlit callback timing, `session_state` GC,
`value=`-vs-`key` precedence), it has **no automated test**, and the page gets
**refactored often**. Fix-by-hand + no test + churn = guaranteed regression.

So a fix is not "done" until it's the version that **can't silently come back**.
Before calling any non-trivial fix done, ensure all three:

1. **Root cause, not symptom.** Fix the upstream layer (a tagger rule, a data
   contract), not the surface. Re-patching the fragile layer IS the loop.
2. **A regression test that runs in CI without a browser.** Move the logic OUT
   of the untestable seam into a pure function the test can call directly. (The
   Review save fix: `_flush_review_edit` writes the pending edit from
   `session_state` on every navigation; `tests/test_review_autosave.py` fakes
   `session_state` as a dict — no Streamlit runtime needed.)
3. **An invariant comment at the code point** stating the contract, so the next
   refactor knows what must hold ("ANY control that navigates away MUST call
   `_flush_review_edit` first").

**Mandatory** for: anything that has ALREADY recurred, anything in the
admin/Streamlit UI seam, and anything touching persistence or the CSV/data.
**Skip** only for genuinely trivial mechanical edits. If a fix can't be tested
without a browser, that difficulty IS the signal to restructure it so it can.

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
