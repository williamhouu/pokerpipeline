# NLHE 9-max Monker pack — format notes & calibration (June 2026)

Working notes for the **NLHE 9-max 100bb pack** under
`nlhe9_ranges/ranges/Hold'em/9-way/100bb[10p-3bb]/` (gitignored; the
canonical 3.4 GB `.mkr` solve stays in the team Dropbox). This is the
companion to `docs/plo_rng_format.md` — same MonkerViewer export family,
**different hand universe, file shape, and EV unit**. Everything below is
verified by `scripts/audit_nlhe9_pack.py` against the live pack.

## 1. Shape

- **93,235 `.rng` files**, flat in one directory, = **44,058 decision
  nodes** (each node = sibling files sharing a history prefix, one file
  per available action).
- Path contains an apostrophe (`Hold'em`) and brackets (`100bb[10p-3bb]`)
  — always quote, and prefer `Path` arithmetic over shell globs.
- Read by `pipeline/preflop/grammars/monker_nlhe.py` (filenames) and
  `pipeline.preflop_ranges.parse_monker_rng_file` (contents); the pack
  registers with `grammar_name="monker_nlhe"`, `file_glob="*.rng"`.

## 2. File contents — self-labeling 169-class pairs

ASCII, two lines per hand class, 169 classes per file:

```
AA
1.0;8313.56063199635
A2s
0.546;0.017402295110777916
```

- Line 1 = hand-class label (the same 169 labels as the Ryan pack).
  **Unlike the PLO pack** (16,432 pattern lines, order-implicit), these
  files are self-labeling — the reader keys on the label and validates
  the full canonical set per file.
- Line 2 = `<weight>;<ev>`. `weight` is the JOINT probability (hand
  reaches the node AND takes this action) — same semantics as the
  PioViewer pack, so presence-normalisation downstream is unchanged.
- **Export quirk:** some deep low-traffic files drop the EV field and
  write a bare weight (`0.5`, no `;`) — e.g.
  `40120.40084.40046.3.1.1.1.1.0.rng` line 16. The reader parses these
  as `ev = 0.0` (PLO-reader tolerance).

## 3. Filenames — dot-token action stems

`<tok>.<tok>...<tok>.rng`, seats implied by rotation (blinds last):

> UTG, UTG+1, UTG+2, LJ, HJ, CO, BTN, SB, BB

A caller/raiser rotates to the back of the queue; a folder or all-in seat
leaves it. Tokens: `0` fold, `1` call, `3` all-in, `40<pct>` = raise
`<pct>`% of pot (3-digit zero-padded: `40084` = 84%). The `5` min-raise
token does **not** occur (the grammar rejects it loudly). 22 distinct
tokens appear; raise sizes vary by seat and context (NOT single-size like
the PLO pack):

- Root: `0.rng` / `40120.rng` only — UTG folds or opens **120% pot =
  4bb** (the resolver: `1 + 1.2 × (1.5 + 1)`).
- First-in sizes step down by seat: UTG–HJ 120% (4bb), CO 100% (3.5bb),
  BTN 81% (3.03bb), SB 100% (3bb).
- **The SB first-in has a limp** (`0.0.0.0.0.0.0.1.rng`) at ~0.1% of
  combos — nearly unused (rake punishes limping) but the node and its
  limp-war subtree exist, so `sb_complete` / `bb_check` archetypes are
  reachable. The Pio 6-max pack had no limps at all.
- Raise-to-bb conversion is Monker's pot-relative rule, implemented in
  `pipeline.preflop.action_history.resolve_preflop_history`:
  `raise_to = high_bet + pct × (pot + to_call)`.

## 4. EV unit — **milli-big-blinds from the start of the hand** (PROVEN)

The PLO pack's EVs are milli-SMALL-blinds; **this pack's are not**. Two
arithmetic-exact anchors pin it (re-runnable via
`scripts/audit_nlhe9_pack.py --section ev`):

1. **Fold files equal −commitment.** At
   `0.0.0.0.0.0.0.40100.40100.3.*` (SB opens 3bb, BB 3-bets to 9bb, SB
   jams), every folding hand in BB's fold file reads **−9000.0** =
   −9bb × 1000. At `0.0.0.0.0.0.40081.0.40130.3.*` the BB 3-bet resolves
   to 11.54bb and the fold file reads **−11540.0** — the pack's own
   bookkeeping reproduces `resolve_preflop_history`'s sizes to the
   milli-bb, independently validating the raise formula.
2. **Mixed hands are indifferent across branches.** Hand 66 mixes
   call/fold at that node and reads −9001.9 / −9000.0 (gap 0.002bb).
   At the root, mixed openers (QQ, A2s) read ~0.0 — UTG has nothing
   committed, and fold-from-start = 0.

So: **baseline = hero's stack at the start of the hand (pre-blinds);
unit = bb/1000**. Because the baseline is per-hand constant, EV
*differences* between actions at one node are baseline-free:
`gap_bb = (ev_a − ev_b) / 1000`.

**Current consumption: NONE.** The pipeline's `ev_gap_bb`, the EV-gap
worthiness gate, and the difficulty `easy_ev` axis all come from the
ANALYTIC engine (`pipeline/preflop/ev_engine.py`, call/fold spots only)
for both packs — no file EV reaches any pipeline math. If pack EVs are
ever wired in (they would unlock raise-spot EV gaps), use the gap formula
above and mind the rake skew (file EVs are post-rake; the analytic engine
is rake-blind).

## 5. Rake and what it does to ranges

Solved at **10% rake, 3bb cap** (micro-stakes structure) vs the 6-max
pack's 4%/0.3bb. Expect visibly tighter ranges everywhere:

| seat | first-in raise % | | seat | first-in raise % |
|------|-----------------|---|------|------------------|
| UTG  | 8.0%  | | CO  | 22.5% |
| UTG+1| 10.1% | | BTN | 38.4% |
| UTG+2| 11.1% | | SB  | 44.0% (+0.1% limp) |
| LJ   | 12.5% | | HJ  | 16.3% |

Headline example: **UTG+1 folds QQ 99% against the UTG 4bb open** — and
the file EVs prove it's genuine indifference, not noise (QQ call
−0.001bb, raise −0.08bb, vs KK +2.30bb, AKo +0.82bb). The GTO Wizard
settings popover text written for the 6-max pack does NOT describe this
pack.

## 6. Full inventory (June 11 audit)

**Nodes by action context × hero seat** (the admin filter buckets):

| context | UTG | UTG+1 | UTG+2 | LJ | HJ | CO | BTN | SB | BB | total |
|---|---|---|---|---|---|---|---|---|---|---|
| Opening | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 9 |
| Facing single raise | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 36 |
| Facing 3-bet | 8 | 7 | 7 | 8 | 10 | 13 | 17 | 22 | 28 | 120 |
| Facing 4-bet+ | 120 | 128 | 134 | 139 | 145 | 155 | 173 | 204 | 254 | 1,452 |
| After call(s) | 3,576 | 4,231 | 4,964 | 5,356 | 5,217 | 4,704 | 4,101 | 4,195 | 6,097 | 42,441 |

The FSR triangle (8+7+...+1 = 36) is complete: every seat-vs-seat
single-raise matchup exists. **96.3% of nodes are "After call(s)"** —
Monker allows cold-calls everywhere, so the tree explodes through call
branches; that bucket spans everything from clean squeezes to 5-way
limp-fests (the admin context filter is coarse here; skill tags slice
finer within a batch).

**Players still in at the decision:** 1: 8 · 2: 5,625 · 3: 13,594 ·
4: 13,120 · 5: 7,337 · 6: 2,993 · 7: 1,109 · 8: 247 · 9: 25.

**Worthy-spot runway** (55–95% window, presence ≥ 0.01; before
difficulty/noise filters): Opening 177 · FSR 740 · F3B 743 · F4B+ 4,071 ·
After call(s) ~129,000 (sampled) — **≈135k worthy spots total.**

**Option shapes:** ZERO nodes offer two raise sizes (one raise ± jam ±
call per node — same single-size-per-node property as the 6-max pack, so
no preflop "pick the size" questions from either). Jams (token 3) first
become available at depth 2 — facing an open + 3-bet (e.g. the
`40120.40084` node, where UTG+2 may cold-4-bet-jam) — and are offered at
4,827 nodes (F3B 120 / F4B+ 330 / After-calls 4,377); there is NO
jam-over-a-single-open (correct at 100bb). 469 nodes (~1%) facing a bet
offer **no flat call** — jam-or-fold (± one raise size) at 4-bet+ depth,
standard Monker abstraction; none of them face an existing all-in (every
facing-a-jam node does have its call). Raise-size menus by level: opens
120/100/81%; 3-bets 84–141% pot (14 context-dependent sizes, IP smaller /
blinds bigger); 4-bets 43–56% pot; 5-bets are jam-only.

**SB-limp subtree:** 6 nodes (SB limps 0.1% of combos at this rake);
`sb_complete`/`bb_check` exist but will essentially never produce
questions from this pack.

## 7. Convergence quality (and the canary fix it forced)

Sampled per players-still-in bucket with the **conditional** AA canary
(continue mass ÷ presence):

| players in | flagged | | players in | flagged |
|---|---|---|---|---|
| 2 | ~3% | | 6 | ~7% |
| 3 | ~2% | | 7 | ~19% |
| 4 | ~2% | | 8 | ~35% |
| 5 | ~2% | | 9 | ~44% |

So the practical generation surface (HU–5-way) is **97–98% clean**, and
the garbage concentrates exactly where expected: the 7–9-way limp/call
tails (~1,400 nodes), which the noise filter + presence/worthiness gates
drop. Deep UTG facing-jam lines flag at 8–9% on BOTH packs.

**Canary bug found by this audit (fixed June 11):** the original
`node_is_unconverged` compared AA's RAW joint mass to ~1.0. Joint
weights include reach, so any node hero reached via an earlier
call/limp/raise carried AA mass ≪ 1 and got flagged — ~80% of
hero-acted-before nodes on this pack, i.e. "After call(s)" generation
would have been silently gutted on either pack. It now normalises by
presence (skipping hands below 0.5% presence) and flags only true
AA-misbehavior; fixture + real-pack tests pin both directions.

## 8. Known data quirks

- **EV field omitted on ~1.1% of payload lines** (172,045 lines across
  8,016 files, concentrated in deep low-traffic nodes): a bare `0.5`
  instead of `0.5;<ev>`. The reader parses these as `ev = 0.0`. Harmless
  while nothing consumes file EVs; if pack-EV gaps ever get wired in,
  bare-line entries must be treated as "EV unknown", not 0.
- ~760 heads-up jam-call closing nodes exist (the §4 calibration anchors).
- Depth runs to 17 tokens; intermediate strategy exists at every node
  (the brief's full-coverage requirement, verified June 11).
