# V7 results — first batch on Ryan-ranges re-solves

V7 is the first 45-question batch produced from solves that consumed Ryan's
preflop pack instead of the Pio-default placeholder ranges that drove V1-V6.
This doc captures the V7-vs-V6 comparison for future-us and locks down the
strategic shift signal that should appear whenever the Ryan-ranges pattern
is extended to scenarios 2-14.

## Headline

| Metric | V6 (placeholder ranges) | V7 (Ryan ranges) | Δ |
|---|---:|---:|---:|
| Questions written | 41 | **45** | +4 (target hit) |
| Distinct source `.cfr` files | 14 | **17** | +3 |
| Distinct `board_texture` values | 17 | **21** | +4 |
| Per-street (flop / turn / river) | 11 / 15 / 15 | **15 / 15 / 15** | flop now full |
| Layer-6 validation failures | 0 | 0 | — |
| Layer-6 API failures | 0 | 0 | — |
| Anthropic calls (incl. retries) | 45 | 54 | +9 |
| Approx. cost (Sonnet 4.6) | $2.41 | $2.65 | +$0.24 |
| Wall-clock | 31.4 min | 30.9 min | -0.5 min |

Sources: `test_output/batch_questions_v7.csv` and `test_output/batch_v7_run.log`
(both gitignored under `test_output/*`); compared against the V6 baseline
locked at commit `2af21d7`.

## Strategic shift in `concept_tags`

The clearest sign that the new ranges materially changed the postflop game tree.

| Tag                                  | V6 | V7 | Δ |
|--------------------------------------|---:|---:|---:|
| `single_raised_pot`                  | 41 | 45 | +4  *(volume)* |
| `nut_advantage_hero`                 | 24 | 27 | +3 |
| `nut_advantage_villain`              | 13 | 15 | +2 |
| `no_blocker_effects`                 | 21 | 22 | +1 |
| **`range_advantage_villain`**        | **8** | **14** | **+6** |
| **`range_advantage_hero`**           | **1** | **0** | **-1** |
| `mdf_defense_threshold`              | 10 | 14 | +4 |
| `villain_polarized`                  | 10 | 12 | +2 |
| `blocks_value_unblocks_bluffs`       | 10 | 11 | +1 |
| `blocks_value`                       | 10 | 11 | +1 |
| `facing_overbet_spot`                |  3 |  6 | +3 |
| `facing_probe_spot`                  |  9 |  6 | -3 |
| `villain_linear`                     |  1 |  3 | +2 |
| `overbet_spot`                       |  8 |  7 | -1 |
| `facing_check_raise_spot`            |  9 |  8 | -1 |
| `implied_odds_call`                  |  1 |  0 | -1 |

### Why these shifts make sense

- **BTN range advantage rises (`range_advantage_villain` 8→14)**. Spot-check
  at flop `4d4sKh` showed BB went from 43.5% → 37.3% and BTN from 45.0%
  → 41.6%; the bigger drop on BB means BTN's range is now *relatively*
  stronger postflop on the boards where his open hits hardest.
- **BB's edge collapses (`range_advantage_hero` 1→0, `implied_odds_call`
  1→0)**. The Pio-default placeholder over-fattened BB's defense with
  hands like A2o/A3o/J4s; those produce sporadic "BB has implied odds"
  spots that disappear once BB stops calling with them.
- **More aggression-facing spots (`facing_overbet_spot` 3→6,
  `villain_polarized` 10→12)**. Tighter ranges → more polarised solver
  responses → more spots where the LLM-rendered question features a
  large IP bet vs OOP. Consistent with real GTO theory.
- **`mdf_defense_threshold` rises (10→14)**. With BB's defending range
  tighter, MDF mathematics binds more often — calling thresholds become
  more important to the explanation.

## Cost per question

V6: $2.41 / 41 = **$0.0588 per question**.
V7: $2.65 / 45 = **$0.0588 per question**.

Identical to three decimals. The prompt caching kicks in after call 1 and
keeps the per-question cost flat across batches; the extra $0.24 in V7
versus V6 is entirely the 4 extra questions (45 vs 41) and 9 additional
corrective retries (54 vs 45 total calls).

## What this means for scenarios 2-14

The S1 wiring pattern (`scripts/build_ryan_ranges_template.py` +
`SolverSpec(using_ryan_ranges=True)` + `--force-resolve` re-solve +
batch-demo) **produced a clean V7 with no validator failures and visible
strategic shifts matching GTO expectations**. The 13 remaining scenarios
should be mechanical replication of:

1. Register the new `SolverSpec` in `pipeline/scenario_spec.py:SOLVER_SPECS`
   with the pack file paths and sizing tokens from
   `docs/ryan_range_pack_index.md`.
2. Run `scripts/build_ryan_ranges_template.py` (extended to accept a
   scenario name, picking the right OOP/IP files per the index doc) to
   emit `templates/<scenario_name>_ryan_ranges.txt`.
3. Run `scripts/batch_solve.py --scenario <name> --flop-set STANDARD_25_FLOPS`.
   No `--force-resolve` needed for new scenarios (cache is empty).
4. Once all 25 solves land, re-run the batch demo with `--scenario <name>
   --out test_output/batch_questions_<name>.csv` to produce the
   scenario's first 45-question batch.

The two open questions for Ryan in `docs/ryan_range_pack_index.md` —
sizing-token portability and the `60% = 2.5x` mapping — still need
locking in before Tier-1 production. They didn't block V7 because S1's
sizing tokens (`60%`, `Call`, `Fold`) are the simplest case in the pack.

## Open items surfaced by V7

- The per-spot `calls=0 (ok)` indicator in the batch-demo log is a
  pre-existing display bug carried over from V5/V6 (the `usage` tally
  lives in a local variable; `result.usage` is only assigned at the end
  of the loop, so per-row deltas compute against zero). The summary's
  `Anthropic calls : 54` is the authoritative count. Fix is one-line but
  scope-creep for this commit.
- The `_read_exploitability` helper in `pipeline/batch_solver.py` reports
  `?` for every solve. The `calc_results_ev` / `show_progress` UPI verbs
  the helper probes don't emit a parseable line on Edge 3; the .cfr is
  still valid but we have no quantified accuracy readout. Worth
  re-investigating before we scale.
- The `extract_facts` "no playable hero combos" skips (3 of them on V7)
  recur on the same flops V6 hit (8s8h2c, 8h5d2c). Layer 5 hand-filter
  edge case, non-fatal but worth a closer look once we're past S1.
