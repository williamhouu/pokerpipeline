# Ryan's Preflop Range Pack — Scenario Index

This document is the canonical mapping from the 14 target UI-mockup scenarios
to the specific `.txt` range files in Ryan's PioViewer-format preflop pack.
Every scenario integration over the next 5-7 weeks of expansion work pulls
its preflop ranges from the file paths listed here. If a file path here
doesn't match what `pipeline.scenario_spec.SolverSpec` references for a
scenario, this doc is the source of truth — open an issue.

## Pack overview

- **Source**: NLH 6-max 100bb 2.5x Open (one PioViewer tree)
- **Location in repo**: `ranges/ryan_preflop_tree/PioViewer - NLH 6max 100bb 2.5x Open/`
- **Format**: single-line CSV of `Hand:weight` pairs, 169 entries per file
  (the 169 preflop hand classes: 13 pairs + 78 suited + 78 offsuit). Weights
  are floats in `[0.0, 1.0]` representing the frequency that hand class takes
  the action that closes the file's path. Example first chars of one row:
  `AA:1.0,A2s:0.621,A2o:0.0,A3s:1.0,...`
- **File count by position folder**:

  | Folder | Files | Notes |
  |--------|------:|-------|
  | `BB/`  | 3,815 | most files (BB is in the most preflop trees) |
  | `BTN/` | 3,649 |       |
  | `CO/`  | 3,473 |       |
  | `HJ/`  | 3,293 |       |
  | `SB/`  | 2,999 |       |
  | `UTG/` | 2,977 | fewest (UTG faces fewest preflop branches) |
  |        | **20,206** total | |

## Filename grammar

A file lives in the folder of the position whose action closes its path
(`BB/.../BB_Call.txt` = BB's range when BB calls something). The filename
itself is an ordered chain of `<Position>_<Action>` tokens separated by
underscores, with the chain ending in `<That-Folder's-Position>_<That-Position's-Action>.txt`.

**Round order**. The first round walks positions in postflop seat order
`UTG → HJ → CO → BTN → SB → BB`. If a player re-opens (3-bet/4-bet/5-bet/all-in),
action wraps back to the next-to-act players in the same UTG→BB cycle,
**skipping anyone already folded or all-in**. Multiple rounds are concatenated
in the filename. Round boundaries are not marked — the parser must track who is
still live.

Example (the longest BTN file in the pack, 134 chars):
```
UTG_60%_HJ_Call_CO_Call_BTN_Call_SB_Call_BB_198%_UTG_Call_HJ_Call_CO_Call_BTN_Call_SB_AI_BB_Fold_UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold
```
Round 1: UTG opens 60%, HJ-CO-BTN-SB all call, BB squeezes 198%.
Round 2: UTG-HJ-CO-BTN all call the squeeze, SB jams all-in, BB folds.
Round 3 (SB is all-in so skipped, BB folded so skipped):
UTG-HJ-CO-BTN each get a final fold-or-call decision; this file is the BTN
range for "BTN folds in round 3."

**Action tokens — semantic ones**:

| Token  | Meaning |
|--------|---------|
| `Fold` | folds |
| `Call` | calls (the current bet) |
| `AI`   | all-in (shove) |

**Action tokens — sizing tokens** are literal percent-of-pot at the decision
point, NOT semantic names. They vary by who is acting and how much pot has
already been built. The pack's tokens for *standard preflop actions* are:

| Action class               | Token(s) observed                  |
|----------------------------|------------------------------------|
| RFI from UTG/HJ/CO/BTN     | `60%`                              |
| RFI from SB (BvB)          | `76%`                              |
| 3-bet by HJ/CO/BTN vs prior open | `77%` (and `79%` if intermediate caller) |
| 3-bet by SB vs BTN open    | `150%`                             |
| 3-bet by BB vs UTG open    | `155%`                             |
| 3-bet by BB vs HJ/CO/BTN open | `182%`                          |
| 4-bet by HJ/CO/BTN over 3-bet | `50%` (and `95%` if CO 4-bets) |
| 4-bet by UTG over BB's 3-bet | `49%`                            |
| 4-bet by BB over SB 3-bet (in 3-way after BTN open) | `54%` |
| Squeeze by SB after open + call    | `85%`                      |
| Squeeze by BB after open + multi calls | `162%`, `198%`         |

**Sizing tokens are not portable across scenarios.** When integrating a new
scenario, do not assume the 3-bet size is `77%` — look up which token the
specific actor uses in the specific spot. The "Sizing convention" section at
the bottom of this doc lists the open question we still owe Ryan on the
chip-translation side.

**Resolving the "current actor"**. The current actor for any file is
the position name in the LAST `<Position>_<Action>` token of the filename.
The folder name is always equal to that position. So `BB/.../BB_Call.txt`'s
current actor is BB.

## Scenario → range files mapping

Format for each scenario:
- **OOP** is the player who acts first postflop (this is the order
  `SB < BB < UTG < HJ < CO < BTN` — earlier = OOP).
- **IP** is the last to act postflop.
- File paths are relative to `ranges/ryan_preflop_tree/PioViewer - NLH 6max 100bb 2.5x Open/`.
- Verification line summarises `<weighted % of all hands>` and the shape
  observation (top-weighted hands or notable composition feature). Numbers
  come from `scripts/range_pack_verify.py` on each file.

### Scenario 1 — BTN opens vs BB call (SRP)
*Reference baseline; already wired in `scenario_spec.py`.*
- **IP (BTN)** RFI range: `BTN/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%.txt`
  - 41.6% of hands, 86 full / 5 partial. Top: AA, A2s-A6s and broadways at 1.0.
- **OOP (BB)** call-vs-BTN-open: `BB/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Call.txt`
  - 37.3% of hands, 52 full / 45 partial. Broad call range with strong suited
    Ax (A2s-ATs at 1.0), most pairs partial, no AA/KK/AJs+ (those 3-bet).

### Scenario 2 — CO opens vs BB call (SRP)
- **IP (CO)** RFI: `CO/UTG_Fold_HJ_Fold_CO_60%.txt`
  - 28.2% of hands, 60 full / 6 partial. AA + all Ax suited + broadways.
- **OOP (BB)** call-vs-CO-open: `BB/UTG_Fold_HJ_Fold_CO_60%_BTN_Fold_SB_Fold_BB_Call.txt`
  - 27.0% of hands, 34 full / 48 partial. Tighter than vs BTN (CO opens tighter).

### Scenario 3 — BTN opens vs SB call (SRP)
- **IP (BTN)** RFI: `BTN/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%.txt`
  - Same file as Scenario 1.
- **OOP (SB)** call-vs-BTN-open: `SB/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Call.txt`
  - 6.0% of hands, 1 full / 44 partial. Top weights: 22 (1.0), A9s (0.94), 33,
    44, 55, A3s-A6s. **This is a very thin call range** — see Open Questions
    section.

### Scenario 4 — SB opens vs BB call (SRP, BvB)
**Position note**: BvB postflop order is SB → BB, so SB is OOP.
- **OOP (SB)** RFI: `SB/UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_76%.txt`
  - 44.0% of hands, 91 full / 6 partial. The widest open range in the pack —
    SB opens BvB very wide because only BB defends.
- **IP (BB)** call-vs-SB-open: `BB/UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_76%_BB_Call.txt`
  - 58.3% of hands, 59 full / 71 partial. Very broad call range, classic BvB
    defensive width (BB realises equity well IP postflop).

### Scenario 5 — HJ opens vs BB call (SRP)
- **IP (HJ)** RFI: `HJ/UTG_Fold_HJ_60%.txt`
  - 22.1% of hands, 42 full / 15 partial. Premium pairs + Ax suited + broadways.
- **OOP (BB)** call-vs-HJ-open: `BB/UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_Call.txt`
  - 25.3% of hands, 26 full / 56 partial. Defends pot odds, mostly suited
    middlings + low pairs.

### Scenario 6 — BTN opens, BB 3-bets, BTN calls (3BP)
- **OOP (BB)** 3-bet vs BTN: `BB/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%.txt`
  - 12.9%, 16 full / 36 partial. Polarised: AA, AJs+, AQo, AKo at full + 76s,
    T8s bluffs. Classic 3-bet construction.
- **IP (BTN)** call vs BB's 3-bet: `BTN/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_Call.txt`
  - 11.9%, 11 full / 34 partial. JTs, J9s, T9s, 66-99, ATs-AQs. Standard 3BP
    IP defense — calls the linear part of BTN's range.

### Scenario 7 — CO opens, BTN 3-bets, CO calls (3BP)
- **IP (BTN)** 3-bet vs CO: `BTN/UTG_Fold_HJ_Fold_CO_60%_BTN_77%.txt`
  - 11.3%, 5 full / 34 partial. AA, AQs, AKs, K9s, KK at full; QQ/AKo/A5s
    high partial.
- **OOP (CO)** call vs BTN's 3-bet: `CO/UTG_Fold_HJ_Fold_CO_60%_BTN_77%_SB_Fold_BB_Fold_CO_Call.txt`
  - 12.1%, 13 full / 30 partial. Pairs + suited Aces.

### Scenario 8 — HJ opens, BB 3-bets, HJ calls (3BP)
- **OOP (BB)** 3-bet vs HJ: `BB/UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%.txt`
  - 7.6%, 6 full / 39 partial. Tighter than vs BTN (since HJ opens tighter).
- **IP (HJ)** call vs BB's 3-bet: `HJ/UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%_HJ_Call.txt`
  - 6.5%, 3 full / 30 partial. ATs/AQs/KQs at full + TT/AJs/QQ/KJs/JJ partial.

### Scenario 9 — BTN opens, SB 3-bets, BTN calls (3BP)
- **OOP (SB)** 3-bet vs BTN: `SB/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_150%.txt`
  - 13.1%, 7 full / 36 partial. Linear value: AA, AQs, AKs, AKo, JJ, QQ, KK + KJs partial.
- **IP (BTN)** call vs SB's 3-bet: `BTN/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_150%_BB_Fold_BTN_Call.txt`
  - 13.0%, 14 full / 30 partial. ATs-AQs, 55-66 + 98s/T9s/J9s suited connectors.
  - Note the `BB_Fold` token in the path — BB has to fold the 3-bet before
    BTN's response, because BB was still live in round 1 when SB 3-bet.

### Scenario 10 — UTG opens, BB 3-bets, UTG calls (3BP)
- **OOP (BB)** 3-bet vs UTG: `BB/UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%.txt`
  - 6.5%, 6 full / 33 partial. Tightest BB 3-bet range (UTG opens tightest).
- **IP (UTG)** call vs BB's 3-bet: `UTG/UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%_UTG_Call.txt`
  - 6.6%, 4 full / 30 partial. AJs, AQs, QQ, KQs at full; TT/JJ partial.

### Scenario 11 — BTN opens, BB 3-bets, BTN 4-bets, BB calls (4BP)
- **IP (BTN)** 4-bet: `BTN/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_50%.txt`
  - 3.3%, 2 full (AKs, KK) / 22 partial. AA/QQ ~0.7, A3s/A8s as bluffs.
- **OOP (BB)** call vs BTN's 4-bet: `BB/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_50%_BB_Call.txt`
  - 6.4%, 9 full / 18 partial. AJs, AQs, AQo + 76s, T8s, T9s, KTs, KJs at full.
  - **Note**: this calling range is wider than the 4-bet range itself,
    because BB's 3-bet contains many semi-bluffs that defend by calling vs a 4-bet
    rather than folding. Worth eyeballing during integration — partials dominate.

### Scenario 12 — CO opens, BTN 3-bets, CO 4-bets, BTN calls (4BP)
- **OOP (CO)** 4-bet: `CO/UTG_Fold_HJ_Fold_CO_60%_BTN_77%_SB_Fold_BB_Fold_CO_95%.txt`
  - 4.5%, 3 full (AA, AKs, KK) / 27 partial. QQ 0.79, KJs 0.78, JJ 0.70.
  - Note CO uses `95%` for the 4-bet, not `50%`.
- **IP (BTN)** call vs CO's 4-bet: `BTN/UTG_Fold_HJ_Fold_CO_60%_BTN_77%_SB_Fold_BB_Fold_CO_95%_BTN_Call.txt`
  - 5.0%, 1 full (AQs) / 22 partial. AA 0.95, JJ/TT/99 mid-weight.

### Scenario 13 — HJ opens, BB 3-bets, HJ 4-bets, BB calls (4BP)
- **IP (HJ)** 4-bet: `HJ/UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%_HJ_50%.txt`
  - 1.9%, 0 full / 18 partial. AA 1.00, AKs/KK ~0.4, KTs/A4s as bluffs.
  - Extremely tight (no full-weight hand class other than AA).
- **OOP (BB)** call vs HJ's 4-bet: `BB/UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%_HJ_50%_BB_Call.txt`
  - 2.7%, 1 full (AQs) / 19 partial. KJs 0.94, QQ 0.67, JJ 0.63 + 87s/65s bluff continues.

### Scenario 14 — UTG opens, BB 3-bets, UTG 4-bets, BB calls (4BP)
- **IP (UTG)** 4-bet: `UTG/UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%_UTG_49%.txt`
  - 1.7%, 0 full / 19 partial. AA 0.99, AKs 0.51, KK 0.30 + A4s/A3s/A5s bluffs.
  - UTG uses `49%` for the 4-bet (vs `50%` for HJ/BTN).
- **OOP (BB)** call vs UTG's 4-bet: `BB/UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%_UTG_49%_BB_Call.txt`
  - 3.2%, 1 full (AQs, QQ) / 22 partial. ATs/JJ/AJs/AKo mid-weight.

## Open questions / ambiguities

### A. Scenario 3 (BTN open vs SB call) — does this scenario merit a slot?

The SB-flat-vs-BTN range exists in the pack (file `BTN_60%_SB_Call`) but is
**only 6.0% of hands** — exactly 1 full-weight class (22) and 44 partials.
That is, the GTO solution flats with only ~6% of the SB's starting range vs a
BTN open (the rest 3-bet or fold; the bulk of SB's defense is 3-bet, see S9).

This raises a real question for the UI mockup: **at the rates the GTO range
implies, this scenario will appear in maybe 1 in 25 hands a player actually
plays**. Compared to S1 (BB calls BTN ~37% of the time, much more frequent),
S3 is a much rarer real-world spot. Two options for Ryan to weigh in on:

1. Keep S3 as-is — questions will be skewed toward the small-pair / suited-Ax
   shape that the call range allows.
2. Drop S3 from the Tier-1 set and replace with another higher-frequency
   spot (e.g., CO opens vs BB 3-bet 3BP, which the pack also has).

### B. Sizing tokens are not portable

There's no universal "3-bet token" — `77%`, `150%`, `155%`, `182%` are all 3-bet
sizings, used by different actors in different contexts. Any future automation
(scenario template generator, downstream solver wiring) MUST look up the actor's
specific token; assuming a constant `77%` will silently produce wrong files in
13 of 14 scenarios. This doc's grammar section is the lookup table.

### C. Multi-way preflop branches not used

The pack contains many multi-way preflop lines (e.g., UTG open + HJ call + CO
call + ..., or open + cold-call + squeeze chains). None of the 14 target
scenarios use them — all 14 are heads-up at the flop. The 20k file count is
mostly multi-way branches we don't touch yet.

### D. `BB_Fold` vs no-`BB_Fold` in Scenario 6 vs 9 paths

S6 (BTN opens, BB 3-bets, BTN calls) IP file path: `..._BTN_60%_SB_Fold_BB_182%_BTN_Call.txt`.
S9 (BTN opens, SB 3-bets, BTN calls) IP file path: `..._BTN_60%_SB_150%_BB_Fold_BTN_Call.txt`.

The `BB_Fold` token only appears in S9's BTN-call file. Reason: in S6, SB
folds in round 1 before BB raises — so SB is out before BB's 3-bet token, and
BTN's round-2 response immediately follows the BB_182% token. In S9, SB raises
out of turn (3-bets after BTN opens), BB is still live and must respond
(folds), then action returns to BTN. The implementation should not hard-code
"BB always folds before the opener's call" — it must follow the round/live-player
rules in the grammar section.

## Sizing convention — open question for Ryan

- The pack is titled "**NLH 6max 100bb 2.5x Open**".
- All non-SB opens use the filename token `60%`. At 100bb-deep, a pot-bet-sizing
  of "60% pot" on the OPEN node corresponds to **2.5bb open over a 1.5bb pot
  (SB + BB)**, which is indeed 2.5x BB. So 60%-of-pot at the open node = the
  2.5x stated in the title. Consistent.
- SB open uses `76%`. At 100bb with SB+BB posted (pot = 1.5bb), an SB open of
  ~2.8x would be 76% of pot. That matches the "SB opens a bit larger than EP"
  convention.
- BB 3-bet over BTN open: `182%`. With pot ≈ 4bb (SB 0.5 + BB 1.0 + BTN 2.5),
  3-betting to 12bb means **putting in another 11bb on top of the 1bb BB
  posted**, i.e. raising 11/4 ≈ 275% — no, this is "% of pot the new bet
  represents above the previous". Need Ryan's confirmation of the exact
  percent-of-pot convention PioViewer uses for these tokens so we can derive
  chip amounts deterministically.

**Action item**: ask Ryan to confirm: (a) the open is 2.5x BB at 100bb stacks,
(b) the chip-amount formula PioViewer uses to convert the `N%` token to a raise
amount on each node. Once locked, encode it in
`pipeline.scenario_spec.SolverSpec.bet_sizes_by_actor` for each scenario.

## How to use this doc when wiring a new scenario

1. Look up the scenario number above; copy the OOP and IP filenames.
2. In `pipeline/scenario_spec.py`, add a `SolverSpec` entry whose
   `oop_range_path` / `ip_range_path` point at those two files.
3. Also update `pipeline/scenario_config.py` with the new `ScenarioConfig`
   (table size, stakes, preflop_action prose). This dual registration is the
   "future refactor" target in CLAUDE.md — when SolverSpec → ScenarioConfig
   auto-derivation lands, only step 2 will remain.
4. Run `python scripts/batch_solve.py --scenario <NewSpec> --flop-set MINIMAL_DEBUG --dry-run`
   first; a real solve takes 30 min – 2 hr.

The integration pattern was set by Scenario 1; subsequent scenarios should be
near-mechanical given the file paths above.
