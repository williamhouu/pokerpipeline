# CEV tournament preflop packs: audit findings (July 2026)

*Verdict: DO NOT INTEGRATE the July-2026 export. The underlying solves look
genuinely strong, but the export is missing every all-in action file, which
makes the strategy frequencies wrong at thousands of decision nodes (the
short depths are unusable outright). A corrected export should pass
`scripts/audit_cev_pack.py` cleanly; re-run it on every new drop.*

## What was delivered

`CEV.zip` (45.7 MB): 13 tournament preflop packs at stack depths
10 / 12.5 / 15 / 17.5 / 20 / 25 / 30 / 40 / 50 / 60 / 75 / 100 / 200 bb,
29,060 range files total. 8-max seats (UTG, UTG1, LJ, HJ, CO, BTN, SB, BB).
"CEV" is presumed ChipEV (no ICM); vendor to confirm. No EV data in the
files (weights only, like the Ryan pack; our analytic ev_engine covers EV).

## Format (fully derived and verified; grammar 100% clean)

Nested folders encode the action line; folds are implicit (skipped seats
folded). Each path segment is `<SEAT>_<VERB>[_<amount>]`:

| Token | Meaning | Amount |
|---|---|---|
| `R`, `3B`, `4B`, `5B` | open / re-raises | raise-TO total in bb |
| `C` | call / limp / complete | chips ADDED by the caller |
| `X` | BB check (vs a completed limp) | none |

Each node folder holds `<its-own-name>.txt` = the actor's per-hand
frequency in (0,1] for taking that action, conditional on reaching the node
(`class:freq` over the 169 grid, bare class = 1.0), plus one child
folder/file per continuation. Verified: per-hand sums across sibling
actions never exceed 1 (0 violations in 29,060 files), rotation legality
and call/raise arithmetic are exact at all depths, and mass is conserved
EXACTLY at fold-illegal nodes deep (see below) — the format semantics are
solid and the exporter's arithmetic is trustworthy.

Full spec + all checks live in `scripts/audit_cev_pack.py`'s docstring.

## Finding 1 (blocking): every all-in action is missing

No wager in any of the 13 depths ever reaches the stack (10bb pack: max
wager 4.75bb; 200bb pack: max 79.95bb). The token vocabulary contains no
all-in code. Everywhere the real solve used a jam, the exported files
simply lack that action — and since folds are implicit (fold = 1 - sum of
exported actions), **the missing jam mass reads as fold mass**, silently
corrupting the strategy at any node where jamming has frequency.

Proof it's an export gap, not a solved-without-jams tree:

1. **Premium "folds."** At 10bb, UTG1 facing UTG's 2bb open (flat-call is
   the only exported response — there is NO re-raise branch at 10-15bb):
   KK, QQ, AKs, AKo all show ~0% across every exported action (implied
   fold ~100%) while AA flats 100% and JJ 67%. KK folding a min-open at
   10bb is impossible under any tree; its mass is in a dropped jam file.
2. **Fold-illegal conservation.** BB facing a completed SB limp can only
   check or raise. At 50/100/200bb the exported X + R files sum to exactly
   1.0 for all 169 hands (proving files are complete conditional
   strategies where no jam exists). At 10-30bb the same node leaks mass on
   exactly the classic limp-jam hands (10bb: K6o missing ~100%, A7o 94%;
   20bb: 66 missing 100%). The leak IS the missing jam file.
3. **Explicit zeros.** Files sometimes carry `hand:0.0` entries (e.g.
   `AKo:0.0` in a 50bb sized-4-bet file) — the node knows the hand, its
   mass is in a sibling action that isn't in the export.

Damage by depth (decision nodes where a premium that genuinely reaches the
node — reach ≥ 5% through its own prior actions — has > 5% unaccounted
mass):

| Depth | Decisions | AA | KK | QQ | AKs | Structural symptom |
|---|---|---|---|---|---|---|
| 10bb | 1,564 | 722 | 764 | 770 | 562 | no re-raise vs open AT ALL; RFI shows CO/BTN opening 0.0% (they jam) |
| 12.5bb | 2,264 | 845 | 837 | 1,083 | 668 | same |
| 15bb | 2,279 | 1,199 | 1,084 | 1,122 | 625 | same |
| 17.5bb | 1,164 | 513 | 572 | 498 | 444 | positional small-3B exists; SB/BB 3-bet (= jam) absent |
| 20bb | 1,147 | 524 | 591 | 594 | 397 | no 4-bets (= jams) |
| 25bb | 1,145 | 434 | 513 | 599 | 444 | same |
| 30bb | 1,126 | 263 | 402 | 500 | 500 | same |
| 40bb | 1,476 | 396 | 538 | 700 | 639 | sized 4B exists; 5-bet (= jam) absent |
| 50bb | 2,679 | 1,051 | 1,044 | 1,482 | 1,169 | same |
| 60bb | 2,727 | 981 | 1,025 | 1,475 | 955 | same |
| 75bb | 2,537 | 908 | 1,172 | 1,691 | 1,060 | same |
| 100bb | 1,229 | 292 | 600 | 872 | 522 | same (no 5B token at 100bb) |
| 200bb | 1,884 | 494 | 1,010 | 1,291 | 778 | sized 5B exists; 6-bet jam absent |

Why we can't salvage a subset: at fold-LEGAL nodes the unaccounted mass is
fold + jam mixed, and separating them requires poker judgment per node —
exactly what this pipeline forbids (solver data must be truth). A question
generated off these frequencies (e.g. "QQ mostly folds here") would pass
every downstream gate while being flatly wrong.

## Finding 2: 139 sized-4-bet range files missing

At 50/60/75/100bb, 139 node folders exist WITH live children but WITHOUT
their own range file — all of them sized `4B` actions on multiway
3-bet/squeeze lines (11 / 33 / 91 / 4 by depth; full list regenerable via
the audit script). Example: `100bb/LJ_R_2.5/BTN_C_2.5/SB_3B_12/BTN_4B_29.1/`
contains SB's response to the 4-bet but not BTN's 4-bet range itself.
Likely a different export bug than Finding 1 (these are not all-ins).

## What checks out (tell the vendor the solves themselves look good)

* Grammar, rotation legality, wager arithmetic: 0 errors in 29,060 files.
* RFI widths are textbook ante-MTT ChipEV at ≥ 15bb and monotone by
  position at every depth (100bb: UTG 16.1% → BTN 54.8%; 20bb: UTG 17.2%
  → BTN 36.3%). BB defends 62.6% call + 14.5% 3-bet vs a BTN 2.5x open at
  100bb. SB plays limp-first deep (86-88% complete at 100/200bb) — a real
  solver strategy in ante games.
* Depth-scaled open sizes (2bb short → 2.5bb deep; SB opens larger,
  2.5-4.5bb; two positional deviations: 75bb BTN 2.5 vs others 2.25,
  200bb BTN 3.0 vs others 2.5; 30bb uses a 2.15bb open — odd but uniform).
* Intermediate-node strategy present at every decision (the brief's hard
  requirement) with limp/iso/squeeze/cold-4-bet lines included.

## Open questions for the vendor (data framing only)

1. Re-export WITH the all-in action files at every node, at all depths
   (this is the blocker). The all-in branch clearly existed in the solves.
2. The 139 missing sized-4B files (list available on request).
3. Confirm ante structure baked into the solves (BB ante? size?) — the
   range widths imply antes; we need the exact pot to price decisions.
4. Confirm ChipEV (no ICM) and the rake-free assumption.
5. The 30bb 2.15bb open size: intended tree config or a conversion slip?

### Vendor replies (2026-07-13, first round)

* **Ante/rake CONFIRMED: "1bb ante in the middle", ChipEV, no rake.**
  Preflop pot before any action = 0.5 SB + 1 BB + 1 ante = **2.5bb**.
  Still to clarify: big-blind ante (who posts) and whether quoted stack
  depths are pre-posting.
* **All-ins: vendor pushed back** ("ranges are in pio format, so all-in
  nodes are not relevant; at 10bb most of the ranges play limp or jam so
  there are no response vs the open"). This does not hold: his own
  description confirms the solves USE jams, and the export can't express
  them — at the 10bb UTG root the exported files show open 2.4% + limp
  2.4% (AA/KK/QQ fully inside those two — the trap pattern of a
  limp-or-jam strategy — while A5s/22 are 100% unaccounted and AKs 47%),
  so the entire jam range reads as fold. Pio-format handles all-ins as a
  raise TO the effective stack, so the format is not the obstacle.
  Counter-ask sent: export the jam action as a normal raise-to-stack
  folder (e.g. `UTG_R_10` at 10bb), same file format.
* **139 4B file list sent** (regenerated to
  `~/Downloads/cev_missing_4bet_files.txt`, paths relative to CEV root).

## Integration prep (for when a fixed export lands)

* Add grammar `cev_mtt` under `pipeline/preflop/grammars/` parsing the
  nested folder path (relative to pack root) instead of a flat filename:
  synthesize implicit folds via the rotation walk (the audit script's
  `RotationWalk` is the reference), map R/3B/4B/5B → RAISE with bb-native
  sizes (like `gto_preflop_8max`), C → CALL, X → CHECK-equivalent, plus
  ALL-IN once the fixed export shows how jams are encoded. Alternative:
  a converter to flat `gto_preflop_8max` .rng files
  (`scripts/convert_preflop_db_to_pack.py` is the precedent) — decide when
  we see the fixed export's all-in encoding.
* **Ante support is a real work item**: pack metadata (`ante_bb`,
  structure), pot math in `resolve_preflop_history`, the animation
  chip-walk's blind posts, pot-odds facts, and the Context line. Without
  it every equity-vs-price fact would use a wrong pot.
* 13 new tournament packs will need the admin pack selector's tournament
  framing (no rake, MTT context line) and `pack_allins_realistic` is True
  only ≤ 40bb (the standing artifact-jam rule).
* Re-run `scripts/audit_cev_pack.py <root>` first — integration starts
  only from a clean pass.
