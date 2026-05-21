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
6. **Explanation Generator** — one LLM call. Prompt = format/voice rules + banned
   phrases, 8–12 similar gold examples retrieved from the ~800-question gold pool,
   the spot description, and the Layer-5 data block. The LLM translates, never reasons.
   Example retrieval matches on **structured features** (street, position, hand class,
   board texture, action context), not raw text.
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

Cache layout — hierarchical by format / scenario / board:

```
/solves/6max_100bb/srp_BTN_vs_BB/flop_AhKsQd.solve
/solves/6max_100bb/3b_BB_vs_BTN/...
/solves/preflop/6max_100bb_full_tree.monker
```

## Scenario tiers

- **Tier 1 (ship v1):** 10 scenarios, 6-max 100bb cash — 5 single-raised pots + 5
  3-bet pots. ~20 flops each ≈ 200 solves ≈ a 10k–40k question runway.
- **Tier 2:** wider 6-max coverage (more SRP/3-bet positions, 4-bet pots, limped SB).
- **Tier 3:** 9-max cash, MTT stacks (40/25/15 bb).
- **Tier 4:** exotic (squeeze, 5-bet, multiway — needs a separate multiway solver).

Lock Tier 1 and get it producing clean output before touching Tier 2+.

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

Source of truth is **Google Sheets**. The existing **29 columns stay unchanged**; the
pipeline writes to them via a formatter and adds **5 new columns**:

| Col | Name | Purpose |
|-----|------|---------|
| 30 | `concept_tags` | comma-separated tags (5–10/question); LLM input + retrieval key |
| 31 | `hand_class` | single label, e.g. `top_pair_with_flush_draw` |
| 32 | `board_texture` | single label, e.g. `monotone_connected_broadway` |
| 33 | `solver_reference` | path back to the exact solve node — the key QA/debugging column |
| 34 | `ev_gap_bb` | EV gap to second-best action; `<0.30` surfaces questionable questions |

Optional column 35 `validation_status` (`draft`/`auto_approved`/`flagged`/
`needs_review`/`approved`/`rejected`). Migrate off Sheets to Airtable/Firestore only
around 7k–10k active questions.

### Authoritative format sample — `docs/output_format_examples.xlsx`

A real sample of the team's output, and the **authoritative format spec**. Two sheets:

- **Sheet 1 "Example formatting"** — 6 fully-populated example rows. The real layout
  is **35 columns** (0-indexed `No` … `validation_status`): a leading `No` (question
  id), 28 template columns, the 5 new columns (`concept_tags`, `hand_class`,
  `board_texture`, `solver_reference`, `ev_gap_bb`), then `validation_status`. So the
  brief's "29 existing" = `No` + the 28 named template columns. This is the spec
  Layer 8 (`format_writer.py`) must match.
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
- No pipeline dependencies are installed yet. Expected external services as code lands:
  a Claude LLM API (Opus-class) for explanations and LLM checkers, an embedding API for
  example search, and an equity calculator for the number checker.
- `.cfr` files are large (GBs). Add a `.gitignore` for `venv/` and `test_solves/`
  before `git init` — this directory is not yet a git repository.
