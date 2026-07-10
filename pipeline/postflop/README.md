# Postflop question pipeline

A **separate, self-contained** pipeline that generates flop / turn / river
training questions, mirroring `pipeline/preflop/` but kept apart from the
preflop NLHE and PLO generators so work here cannot change their behaviour.

The governing rule is the repo's: **the LLM never thinks about poker, it only
writes the words.** Every strategic fact (correct action, equity, frequencies,
hand class, board texture, concept tags, archetype) is computed
deterministically; Layer 6 turns that resolved block into prose.

## Why it's built on a solver-agnostic IR

The pipeline never touches a vendor solve format directly. It runs on a clean
intermediate representation (`solve.py`); a thin **adapter** per solve source
populates it:

```
PioSolver .cfr  ─┐
third-party .db ─┼─► adapter ─► PostflopSolve (IR) ─► pipeline ─► CSV + meta.json
(future formats)─┘                     ▲
                          synthetic fixture (fixtures.py) — tests / demo
```

This is deliberate. This Mac cannot run PioSolver (Windows-only), and the
trial solves are multi-GB and live outside the repo, so the whole pipeline is
built and tested against an **in-memory synthetic solve**
(`fixtures.btn_vs_bb_srp_2cJs7s`). When a real solve is adapted into the IR,
no pipeline code changes — only the input.

## Layers

| File | Layer | Role |
|------|-------|------|
| `solve.py` | 1-2 | The IR: `PostflopSolve` / `PostflopNode` / `NodeAction` (+ `validate_solve`). |
| `fixtures.py` | 1-2 | Synthetic BTN-vs-BB SRP `2c Js 7s` solve (4 nodes). |
| `spot_sampler.py` | 3 | `PostflopNode` + a hero combo -> `PostflopSpot` (+ per-spot EV gap). |
| `question_extractor.py` | 4 | Worthiness gate: dominant freq in 65–99% (EV-gap gate optional, off by default). |
| `facts.py` | 5 | `PostflopFacts`: equity, hand class, board texture, blockers, SPR, pot odds, EV gap, archetype, concept tags. |
| `concept_tags.py` | 5 | Postflop concept-tag registry + archetype classifier (pure functions). |
| `action_history.py` | — | Deterministic **multi-street** Context/Question prose (a turn question shows preflop + flop). |
| `options.py` | — | The four multiple-choice options + the correct answer. |
| `difficulty.py` | — | 500–3000 difficulty from the frequency + EV-gap axes. |
| `explanation_generator.py` | 6 | The only LLM step (or a deterministic placeholder for dry runs). |
| `validators.py` | 7 | Deterministic hard (reject + retry) and soft (flag) checks. |
| `format_writer.py` | 8 | The team-format CSV row + writer. |
| `batch.py` | — | `generate_postflop_batch` — the end-to-end driver + `meta.json`. |

## What it reuses (and never modifies)

Only **pure, game-agnostic leaf utilities** from the rest of the repo:

- `pipeline.cards` — card parsing.
- `pipeline.fact_extractor.hand_class.classify_hand` — made-hand classification.
- `pipeline.fact_extractor.board_texture.classify_board` — board texture.
- `pipeline.fact_extractor.equity.equity_vs_range` — the 7-card equity evaluator.
- `pipeline.action_history.format_card` — the suit-emoji card renderer.
- `pipeline.explanation_generator` — `call_messages_create`,
  `GeneratedExplanation`, `BANNED_LITERAL_PHRASES`, `ExplanationValidationError`.

It imports **no** other pipeline's batch driver, fact extractor, validators,
or format writer.

## Run it

```bash
# Deterministic dry run — no solver file, no API key:
venv/bin/python scripts/generate_postflop.py --dry-run --out test_output/postflop_demo.csv

# Real run (needs ANTHROPIC_API_KEY):
venv/bin/python scripts/generate_postflop.py --out test_output/postflop.csv -n 30

# Tests:
venv/bin/python -m pytest tests/test_postflop_pipeline.py -q
```

## Determinism

Equity is seeded per spot (`facts._spot_rng`), the worthy-spot order is sorted
(not shuffled), and `meta.json` carries no timestamps — so the same solve +
settings produce a **byte-identical** CSV. `test_batch_is_deterministic`
guards this; any drift is a regression.

## Done vs. next (extension points, all clearly seamed)

**Done (this pass):** the full deterministic spine + dry-run end-to-end + a
real-LLM path (mock-tested) + 39 tests + this doc. The pipeline produces a
CSV of worthy questions with full multi-street action history, difficulty,
concept tags, archetype, and provenance.

**Done (June 2026):** the **third-party `.db` adapter**
(`adapters/sqlite_db.py`) — the first real-solve integration. It reads a
vendor SQLite postflop solve into the IR (flop nodes for v1), deriving
check/call + bet/raise from the node-string betting state (the vendor's
action *labels* are unreliable), reach-weighting each side's range for
equity, and converting chips→bb. First real questions generated from
`BTN_vs_BB_SRP_100bb_QsJd9s_v8.db` (audit: `scripts/audit_postflop_db.py`).
The CLI loads it via `--solve <path>.db`; `--diversify` rounds-robin across
decision types (see the calibration note below).

**Structure report (July 2026).** `structure_report.py` walks a `.db`'s
node strings (node+action columns only, sub-second) and reports the REAL
action menus per street x actor x situation, plus auto-derived
plain-English limitations ("after a river check, BTN can never bet",
"the only raise facing a turn bet is all-in", "turn bets come in one
size only"). Cached in a `<name>.structure.json` sidecar next to the
solve (keyed by size+mtime+version); the admin picker's solve preview
renders it in the "Betting options at each street" expander, computing
inline on a first-ever selection. These are properties of the tree the
vendor solved, surfaced so you know what a batch can and cannot ask.

**3-bet-pot solves (July 2026).** The adapter parses the `preflop_line`
metadata ("BTN open 3bb, BB 3bet 17bb, BTN call") into the preflop summary,
so a 3-bet-pot `.db` renders the true line everywhere downstream (prose,
`Preflop Pot Type` = "Three bet pot", the aggressor — and with it C-bet vs
Facing-a-C-Bet classification, skills, and the seat-token stack walk).
Files without the key (the older exports) are all SRP and keep the legacy
open+call summary. Chips-per-bb prefers the file's own `pot`/`pot_bb` +
`eff_stack`/`eff_stack_bb` pairs (exact for any pot type) over the
SRP-only `btn_open` identity.

**Full-hand mode supports 3-bet pots via pack LINE legs (July 2026).**
`preflop_leg_pack.find_pack_leg_source` matches the solve's WHOLE preflop
line to a range pack, one node per decision (`PackLineStep`): SRP = open +
defend; 3-bet pot = the open, the 3-bet, and the call — so the opener gets
TWO preflop questions and the 3-bettor one, each built by the full preflop
pipeline. Geometry gates every raise size on the line (the v7 3-bet pots
match `preflop_8max_200bb_IMPROVED` exactly at 3bb/17bb). On a multi-raise
solve there is NO entry-derived fallback (the solve's entry weights cannot
express a raise-or-call-or-fold decision): a hand whose pack strategy
contradicts the as-played line DROPS that preflop leg
(`counters.preflop_line_legs_dropped`) and its postflop legs still narrate
the full line. **Standalone preflop-entry mode stays SRP-only** (its
continue-or-fold framing has the same limitation); the driver and admin UI
fail fast with a plain message.

> **Worthiness = the frequency window; the EV-gap filter is OPTIONAL (off by
> default — mirrors preflop).** Real solves mix heavily, so the
> genuinely-interesting decisions (c-bet? lead? which size?) have **near-zero
> EV gaps by construction** — that's genuine indifference, not solver noise.
> The brief's hard 0.5bb gate therefore collapsed the worthy pool to
> high-variance "facing-a-big-bet" call/fold spots (on QsJd9s: 144 worthy, ALL
> one type). So `evaluate_spot` / `generate_postflop_batch` now default
> `min_ev_gap_bb=None` (frequency-only) → full variety (1014 worthy across
> c-bets/leads/raises/folds for both players), exactly PLO's reason for
> dropping its EV axis. Enabling `min_ev_gap_bb` is an opt-in quality filter
> (the admin "advanced filter"); the EV gap still feeds difficulty's `easy_ev`.
> `--diversify` round-robins the worthy pool across the four flop decision
> types so a fill-to-N batch isn't dominated by one archetype.

**Done (June 2026, range-advantage + Layer-7 pass):**

- **Range-vs-range advantage** (`facts.compute_range_advantage`): the node-level
  "who is ahead on this board" verdict, resolved in PYTHON (the brief's #1 LLM
  failure mode). `hero_range_equity` (fixed-seed `equity.range_vs_range_equity`),
  `range_advantage`/`nut_advantage` ∈ hero/villain/even, + two strong-made
  (two-pair+) shares. Surfaced in the SOLVER DATA block (`RANGE ADVANTAGE` /
  `NUT ADVANTAGE`), the `range_advantage`/`nut_advantage` concept tags, and the
  `range_equity` CSV column. Voice rule binds claims to the fact (mirror, never
  compute/reverse). Verified on v8: BTN (aggressor) gets the edge, BB the deficit.
- **Layer-7 LLM audit (opt-in), ported from preflop:** `claim_checker.py` (a 2nd
  pass that FLAGS confusing/wrong postflop claims — range-advantage-to-wrong-
  player, mislabeled draws, invented blockers, equity-vs-price reversals) and
  `reviser.py` (a 3rd pass that rewrites flagged prose, re-validated by the hard
  validators, discarded if it breaks one; optional 4th-call final audit). Same
  toggles as preflop (`run_claim_checker`/`revise_pass`/`final_audit`), threaded
  batch→`run.py`→admin Generate + CLI; lifecycle in `meta`. Real-API proof on v8:
  4/4 flagged→fixed, all catches genuine.
- **Batch re-verifier:** `scripts/audit_postflop_batch.py` rebuilds every CSV row
  from the source `.db` (via `meta.provenance`) and diffs (0/0 on real output).

**Done (June 2026, the `.cfr` adapter):** `adapters/cfr_pio.py` maps a PioSolver
Edge `.cfr` solve into the IR by driving the solver over its **UPI protocol**
(`show_node` / `show_children` / `show_strategy` / `show_range` / `calc_ev` /
`show_hand_order` — the same commands the Phase-0 client uses). Pio's node-string
grammar is identical to the `.db` adapter's, so the betting-state walk mirrors it
(check/call + bet/raise derived from the betting STATE, never a label); `show_range`
gives each node's range directly (no manual reach-walk), and `calc_ev(actor, child)`
gives per-action EVs. Written against a small `UpiClient` `Protocol` and
**unit-tested with a mocked client** (`tests/test_postflop_cfr_adapter.py`) — this
Mac can't run Pio, so it is **NOT verified end-to-end here**. Full integration
needs a Windows host with Pio Edge + a `.cfr` (e.g. `test_solves/
btn_vs_bb_srp_2cJs7s.cfr`), driven via `load_postflop_cfr`. v1 builds flop nodes
(turn/river is a BFS-through-chance-nodes extension; the walk already handles
chance tokens). `validate_solve` is the contract both adapters target.

**Next:**

1. **Real-host `.cfr` verification.** Run `load_postflop_cfr` on a Windows host
   with Pio Edge against `test_solves/btn_vs_bb_srp_2cJs7s.cfr` and confirm the
   built IR matches the synthetic fixture's shape; then extend the BFS through
   chance nodes for turn/river.
2. **Admin Generate + Review pages — DONE (June 2026).** `admin_panel/app.py:
   _render_generate_page_postflop` is a *solve picker* (scans `solves/postflop/`
   via `discover_db_solves`, launches `run.generate_postflop_batch_from_db`
   through `jobs.start_subprocess_job` with the `.db` PATH). It has preflop
   parity controls: answer-option style (`build_options(spot, style=)` —
   basic/gto/auto), display amounts in bb or dollars (`display_in_bb` threaded
   through `action_history`/`format_writer`), output filename, model, and
   worthiness. `render_postflop_review_page` mirrors the preflop Review (browse
   a batch, grade, edit explanation + difficulty inline) reusing the generic
   `admin_panel.review` sidecar helpers.
3. **LLM prompt tuning.** `POSTFLOP_SYSTEM_PROMPT` is the default; it's now
   admin-editable (`load_postflop_system_prompt` reads
   `admin_panel/prompts/postflop_system.txt`; the Prompt page → Postflop mode
   edits it). Still wanted: tune against gold postflop examples and grow the
   soft validators from observed failures (same loop as preflop).
4. **App table-state format — DONE.** `pipeline/postflop/app_table_format.py`
   emits the exact Runout chip/seat/board tokens (`User Seat = "BTN-$97.5"`,
   `Seats = "BB-$95.7-$1.8-bet"`, `Cards on Table = "2-clubs, J-spades,
   7-spades"`), built from the same bb-denominated amounts as the question
   prose. Postflop is where `Cards on Table` is finally non-empty. Remaining
   stacks keep cents and both players always render (unlike the preflop engine);
   per-player remaining is reconstructed by a betting walk over the preflop line
   + every postflop street. A new `Seats` column sits between `Default Stack`
   and `POT`. Verified against `docs/output_format_examples.xlsx` postflop rows.
5. **Parity + tuning.** A postflop Compare page + prompt library; the
   harder-to-detect skills (`Facing a Check-Raise`, MDF, Reverse Implied Odds —
   see `skills.py:POSTFLOP_SKILLS_NOT_TAGGED`); LLM prompt tuning against gold
   examples. (DONE: range-vs-range advantage; **blocker value/bluff
   decomposition** via `facts.compute_blocker_decomposition` — the LLM now knows
   deterministically whether it blocks villain's value or bluffs, with prompt +
   claim-checker + soft-validator guards; the `skills` column; 3-axis + trap
   difficulty; **curation filters** (hand-strength + decision-type) in
   `spot_selection.py`; the **solve-quality / node-reach gate** in
   `quality.py`.)
