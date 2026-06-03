# Monker `.rng` range-file format (PLO pack) — reverse-engineering notes

Working notes on the MonkerSolver/MonkerViewer `.rng` format, as used by the
purchased **PLO 6max 100bb (Rake 5%, 1bb cap)** pack. This is the input the
PLO pack parser (`pipeline/plo/pack.py`, not yet written) must read.

Status: **the format is ~90% cracked and validated; one piece — the exact
hand *enumeration order* — is still open** (see §5). Everything else below is
confirmed against the data, analytic combinatorics, and two independent
open-source reference parsers.

## 1. Where the data lives

```
plo_ranges/ranges/Omaha/6-way/100bb(5p-1bb)/<action-sequence>.rng
```
- **12,164 `.rng` files** (3.8 GB), one per preflop decision node. Gitignored.
- Source: rangeconverter.com / MonkerGuy. Seller notes: `~/Downloads/MonkerGuy
  - Instructions v10.txt`.
- The 2.24 GB `.mkr` (raw MonkerSolver solve) is **not needed** — only for
  re-solving in the full solver.

## 2. File structure (one node's range)

ASCII text, **two lines per entry**, **16,432 entries per file**:

```
<hand-pattern>
<p>;<ev*1000>
```

- **16,432 = the exact number of suit-isomorphic PLO starting hands** (verified
  by brute-force canonicalization of all C(52,4)=270,725 combos).
- `p` = probability the hand takes this branch in the GTO mixed strategy, in
  [0, 1].
- `ev*1000` = expected value in **small blinds × 1000** (so `-1000.0` → −1.0 sb).
  Confirmed against the `monkerware` reference parser.
- **The hand-pattern enumeration is FIXED across every file** (identical
  pattern-line hash on multiple nodes). So the position→hand mapping is decoded
  **once** and every file is then read purely by index.

## 3. Hand-pattern notation

Each pattern describes a 4-card hand up to suit isomorphism:

- `?` — a wildcard rank (see §5 — these are *not* fully self-describing).
- `(...)` — a **suited group**: the cards inside share one suit. `(2?)??` = two
  cards suited (one is a 2) + two off-suit singletons; `(2?)(3?)` =
  double-suited; `(26T)3` = three cards of one suit + a singleton; `(????)` =
  monotone.
- Ranks use `23456789TJQKA`.

**The parens (suit-pattern) are fully specified in every entry** — verified:
grouping each pattern by its suit-class reproduces the canonical-hand
distribution exactly:

| suit-class | (group sizes) | count |
|---|---|---|
| rainbow        | (1,1,1,1) | 1,820 |
| single-suited  | (2,1,1)   | 7,098 |
| double-suited  | (2,2)     | 3,081 |
| three-suited   | (3,1)     | 3,718 |
| monotone       | (4)       |   715 |
| **total** | | **16,432** |

## 4. Filename = action sequence

`<tok>.<tok>...<tok>.rng`, dot-separated action tokens, decoded per
`OwenQian/MonkerConverter`:

- Seat order (6-max): **`LJ, HJ, CO, BU, SB, BB`**.
- Token `0` = fold, `1` = call, `3` = all-in (jam), `5` = min-raise; any other
  token = a raise whose size is embedded in the token.
- After a call/raise the actor rotates; after a fold the seat drops out.
- Raise sizes follow Monker's pot-limit math (SB = 1 chip; BB = 2):
  `raise_to = Raise% × (all_previous_bets + previous_bet) + previous_bet`.
  e.g. a 50% open at 1/2 = `0.5×(1+2+2)+2 = 4.5`.

Example: `0.0.0.0.0.rng` = everyone folds to the BB / the initial node;
`40100.40100.3.rng` = raise, raise, jam.

## 5. OPEN: the hand enumeration order

The remaining unknown. The pattern text is **not** a self-contained hand label:
there are only **394 distinct patterns for 16,432 hands** (bare `????` appears
550×), so the true hand identity is the **position in a fixed enumeration**, and
the pattern is a lossy delta/checkpoint annotation.

What the data shows (so reconstruction can be validated):
- It is **not** per-slot carry-over (gives 26 distinct hands, not 16,432).
- It is **not** separable by suit-class (rainbow-only carry also fails).
- It is a **recursive tree-delta**: consecutive entries share a rank *prefix*
  and vary the suffix card + its suit together. Example block (positions shown):

  ```
  4573  2345     {2,3,4,5} rainbow
  4574  (26)34   {2,3,4,6} 2-6 suited
  4575  2349     {2,3,4,9} rainbow
  4576  (2T)34   {2,3,4,T} 2-T suited
  ...
  ```
  Fully-specified "anchor" hands (62 of them, 0 wildcards) act as resync
  checkpoints; ranks are shown only periodically (≈ every 4th value).

**Validation gate (no guessing):** any candidate ordering must satisfy *all*
62 fully-specified anchors + every partially-shown rank/suit across all 16,432
entries *and* be a bijection onto the 16,432 canonical hands. That over-
determines the answer — a wrong order cannot pass.

**Plan to close it:** extract the authoritative order from the reference
decoder (the `monkerviewer-1.4.jar`, Java — runs on this Mac), bake it in as a
verified 16,432-entry lookup table, and build the parser on top. Reversing the
proprietary tree serialization by inspection is deliberately avoided — too easy
to be subtly wrong.

## 6. References

- [OpenHUD/monkerware](https://github.com/OpenHUD/monkerware) — JS `.rng`
  importer (confirms the `p;ev*1000` payload; NLHE hand labels only).
- [OwenQian/MonkerConverter](https://github.com/OwenQian/MonkerConverter) —
  filename action-token decoding + seat order.
