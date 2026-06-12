# Quality-audit playbook (NLHE preflop batches)

The repeatable two-layer audit for generated question batches, plus the
state of the quality loop. First run June 12, 2026 (round 1); the fixes
it produced are in commits `3264fa0` + `7dbc40b`. This document is the
durable copy of the method — per-batch artifacts (reports, flags) live
gitignored next to the CSVs in `test_output/preflop_batches/`.

## The loop

1. Generate a real-API batch from the admin panel.
2. **Layer 1 — deterministic re-verification** (scripted, binary):

       venv/bin/python scripts/audit_preflop_batch.py "test_output/preflop_batches/<batch>.csv"

   Rebuilds every row from the source pack (node by id from the meta
   sidecar, hand class, exact combo) and diffs every column. Prose,
   table tokens, pot math, frequencies, options/correct, non-equity
   tags, matchups, ranges JSON must match EXACTLY. Since equity became
   per-spot seeded (June 12), the equity-coupled fields (ev_gap_bb,
   difficulty, equity tags, archetype) must ALSO reproduce exactly —
   **any drift there is now a regression signal, not noise.**
3. **Layer 2 — claim-by-claim prose audit** (a human or Claude reading):
   for each row, decompose the Answer Explanation into factual claims
   and check each against the row's SOLVER DATA block in
   `<batch>.meta.json` (`questions[i].solver_data` — the exact input the
   LLM saw). Severity taxonomy:
   - **MAJOR**: contradicts the recorded action, or invents load-bearing
     facts (other players' ranges, false acts-behind claims).
   - **MODERATE**: unlicensed inventions (statistics, range claims
     beyond the block, blocker claims with no blockers fact), wrong
     poker terminology, frame violations.
   - **CLEAN**: every claim traces to the block (embellished *reasons*
     for licensed facts are acceptable).
4. Write flags into the batch's review sidecar so they appear inline on
   the Review page (`admin_panel.review.save_review`, status
   `needs_review`, note prefixed `AUDIT (Claude):`; clean rows left
   ungraded for the human).
5. Summarize systemic patterns → fixes land in pipeline/prompts →
   regenerate → repeat. Stop when remaining failures are one-offs;
   surviving checks graduate into the Phase-2 validator stack.

## Round-1 baseline (June 12, AUDIT BATCH 15q + NEWEST 4q, Opus 4.7)

- Layer 1: **0 hard failures / 19 rows** (pipeline integrity clean);
  every number cited in prose matched its block.
- Layer 2: TEST PROMPT 1 deep-multiway: 3 MAJOR + 6 MODERATE + 6 clean.
  Factor-list prompt, shallow spots: 0 MAJOR + 3 MODERATE + 1 clean.
- All three MAJORs were one disease: the block carried ONE villain's
  range while spots had 3–4 live players → invented cold-caller ranges,
  false "still live behind you" claims.

## Fixes landed in response (active for any batch after `7dbc40b`)

1. **Multiway facts** in SOLVER DATA: `other_players_still_in_hand`
   (all-in marked), `still_to_act_after_you`,
   `your_call_or_fold_closes_the_action` — computed by
   `pipeline.preflop.action_history.compute_action_pending`. **Voice
   rule 11**: only `villain_stats`' range may be characterized; other
   players are actions-only; acts-behind claims must come from the list.
   (Rule also patched into the three gitignored prompt snapshots.)
2. **Seeded equity**: `_spot_rng` seeds Monte-Carlo per
   (node_id, hand_class, combo); `DEFAULT_EQUITY_RUNOUTS` 200→400.
   Same spot ⇒ byte-identical equity/frames/difficulty, forever.
3. **Premise-realism gates** (pre-equity, pre-LLM, ON by default,
   tunable in Generate → Advanced filters): `min_villain_line_pct`
   (default 0.25% — kills ghost-villain lines) and
   `min_hero_premise_freq` (default 5% — kills "you opened K6s from the
   LJ" stories; checks each of hero's own prior actions). Skip counts
   surface in BatchResult / shortfall UI / meta.
4. **meta.json now records `run_settings`** (answer style, filters,
   gates, runouts, stakes/venue) — no more reverse-engineering batches.

## Round-2 expectations (next audit)

Target: `LATEST AUDIT READY_20260612_122054.csv` (generated post-fix).
The MAJOR class should be structurally dead; Layer 1 should show zero
drift everywhere including equity-coupled fields. Residual items to
watch (small, not yet fixed): terminology slips (limp vs cold-call,
"capped" misuse), blocker claims at open spots (no blockers fact exists
there), invented statistics/strategy asides, TEST PROMPT 1's formulaic
"against a nit, deviate" closer.
