# NLHE 6-max short-stack Monker packs — format notes & calibration (June 2026)

Working notes for the **NLHE 6-max 20bb and 30bb packs** under
`nlhe6_ranges/ranges/Hold'em/6-way/20bb(5p-0.5bb)/` and `30bb(5p-0.5bb)/`
(gitignored; local-only). Same MonkerViewer export family as the 9-max
pack (`docs/nlhe9_pack_notes.md`) and the PLO packs, but with two tokens
neither of those uses and a **different EV unit**. Everything below is
verified by `scripts/audit_nlhe6_pack.py` against the live packs.

The full 20–200bb family (20/30/40/50/70/100/150/200bb) lives in
`~/Downloads/NLH 6-max ranges/`; only 20bb and 30bb are extracted and
registered so far. A parallel PLO 6-max family sits alongside it.

## 1. Shape

| pack | `.rng` files | decision nodes |
|------|-------------:|---------------:|
| 20bb | 1,934 | 913 |
| 30bb | 4,868 | 2,247 |

- Flat in one directory per depth; each node = sibling files sharing a
  history prefix, one file per available action.
- Path contains an apostrophe (`Hold'em`) and parens (`20bb(5p-0.5bb)`) —
  always quote, prefer `Path` arithmetic over shell globs.
- Read by `pipeline/preflop/grammars/monker_nlhe.py` (filenames, shared
  with the 9-max pack) and `pipeline.preflop_ranges.parse_monker_rng_file`
  (contents). Registered in `pipeline/preflop/pack.py` as
  `monker_nlhe_6max_20bb` / `monker_nlhe_6max_30bb`,
  `grammar_name="monker_nlhe"`, `file_glob="*.rng"`, `table_size=6`,
  `size_round_bb=0.5`.

## 2. File contents — self-labeling 169-class pairs

Identical to the 9-max pack: ASCII, two lines per class, 169 classes per
file, line 1 = label, line 2 = `<weight>;<ev>` (joint probability; bare
weight with no `;` on some deep low-traffic lines parses as `ev = 0.0`).
See `docs/nlhe9_pack_notes.md` §2.

## 3. Filenames — dot-token action stems

`<tok>.<tok>...<tok>.rng`, seats implied by rotation (blinds last), at
6-max:

> UTG, HJ, CO, BTN, SB, BB

A caller/raiser rotates to the back of the queue; a folder or all-in seat
leaves it. Tokens:

| token | meaning | bb size |
|-------|---------|---------|
| `0` | fold | — |
| `1` | call / limp / check | matches the current bet |
| `3` | all-in jam | effective stack (20 or 30bb) |
| `5` | **min-raise** | **2bb** (twice the current bet) |
| `14` | **BB iso over a single SB limp** | **2.5bb** (75% pot) |
| `40<pct>` | raise `<pct>`% of pot | pot-relative |

- **`5` is the open.** At these depths there are no pot-% opens — every
  first-in raise is the `5` min-raise to 2bb. Verified: `5` *only* ever
  appears as the opening raise, never as a re-raise (so the "twice the
  current bet" min-raise rule and the 2bb open coincide; a true min-raise
  on a *re-raise* never occurs). It carries the `MIN_RAISE_PCT` sentinel
  through the grammar (not a pot fraction) and is sized relative to the
  running bet in `resolve_preflop_history`. Renders as `Raise min` in
  option labels / node ids.
- **`14` is the BvB iso.** It appears *only* in the `0.0.0.0.1.14...`
  node (folds to SB, SB limps, BB raises) — 12 files, identical in both
  packs. Emitted as a 75%-pot raise, which reproduces 2.5bb exactly
  through the normal size walk (`1 + 0.75 × (2 + 0) = 2.5`). If a future
  short-stack pack reuses `14` in a different pot context, the audit's
  token-size lock catches the drift.
- Any other bare-integer token (`2`, `15`, …) is still **rejected loudly**
  by the grammar — decode + add it before such a pack registers.
- Raise-to-bb for the pot-% tokens is Monker's pot-relative rule
  (`raise_to = high_bet + pct × (pot + to_call)`), same as the 9-max pack;
  rendered sizes snap to the 0.5bb grid (`size_round_bb=0.5`).

## 4. EV unit — **milli-SMALL-blinds from the start of the hand**

**This differs from the 9-max NLHE pack** (which is milli-*big*-blinds):
these 6-max packs report EV in milli-**small**-blinds, like the PLO packs.
A fold's EV equals minus the chips already committed, and:

- The SB folding first-in (only the 0.5bb blind posted) reads **1.0** —
  i.e. one small blind, proving the unit is sb (= 2× the bb amount).
- Given that, an opener who later folds reads **4.0 sb = 2bb** (the `5`
  open), and the BB who iso-raised `14` then folds reads **5.0 sb =
  2.5bb**. These are exactly the §3 token sizes, re-derived from the
  solve's own bookkeeping independent of the size walk.

Re-runnable as `scripts/audit_nlhe6_pack.py --section sizes`.

**Consumption (June 2026): the `action_ev_bb` CSV column.** The per-action
file EVs feed the `action_ev_bb` column via `pack.ev_units_per_bb = 2000`
(milli-sb → bb), giving the per-action EV breakdown. DISPLAY only — the
pipeline's *math* (`ev_gap_bb`, the worthiness gate, the difficulty
`easy_ev` axis) still comes from the analytic `ev_engine`, so no file EV
feeds a filter or a score. The `2000` divisor (1sb = 0.5bb) is verified by
`scripts/audit_nlhe6_pack.py --section sizes` (the BB-iso fold reads
−2.5bb).

## 5. Rake and ranges

Solved at **5% rake, 0.5bb cap** (the `(5p-0.5bb)` folder tag) — lighter
than the 9-max pack's 10%/3bb. First-in raise % (min-raise opens), from
the audit:

| seat | 20bb | 30bb |
|------|-----:|-----:|
| UTG  | 18.4% | 19.6% |
| HJ   | 21.3% | 23.1% |
| CO   | 25.2% | 29.4% |
| BTN  | 32.0% | 39.2% |
| SB   | 43.8% (+12.3% limp) | 41.1% (+14.1% limp) |

Tight-to-loose UTG→BTN as expected; 20bb opens slightly tighter than 30bb.
The SB mixes a meaningful limp at both depths (unlike the rake-punished
~0.1% SB limp on the 9-max pack), so `sb_complete` / `bb_check` /
BvB-iso (`14`) archetypes are live question sources here. AA continues
100% vs a UTG open at both depths (canary).

## 6. Framing

Short-stack cash/MTT geometry. The audit renders cash dollars at the Tier-1
$0.25/$0.50 default (a 2bb open = "opens to $1", a 30bb stack = "$15"); the
admin Generate page's stake/venue/bb-display selectors apply as for any
pack. Consider a bb-display or tournament default when wiring these into
production batches — at 20/30bb the bb framing usually reads cleaner than
small-stakes dollars.

## 7. Status

- Decode + sizing landed June 14 2026 (grammar `5`/`14` tokens; the shared
  min-raise size walk; the two `PreflopPackSignature`s). Tests:
  `tests/test_monker_nlhe_grammar.py` (token parsing + `Raise min`
  rendering) and `tests/test_monker_sizes.py` (the 2bb / 2.5bb size walk).
- Not yet calibrated for production beyond the audit: convergence-quality
  sampling, worthy-spot runway, and a full inventory table (cf.
  `nlhe9_pack_notes.md` §§6–8) are still TODO if these packs go to a real
  generation tier. The convergence guard's deep-jam-tail flag rate is a
  100bb-calibrated band and is *not* asserted on these short-stack packs
  (`tests/test_preflop_batch.py`).
