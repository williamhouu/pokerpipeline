# Monker `.rng` range-file format (PLO pack) — reverse-engineering notes

Working notes on the MonkerSolver/MonkerViewer `.rng` format, as used by the
purchased **PLO 6max 100bb (Rake 5%, 1bb cap)** pack. This is the input the
PLO pack parser (`pipeline/plo/pack.py`, not yet written) must read.

Status: **fully cracked and validated.** The hand order — the one hard piece —
was resolved authoritatively from the MonkerViewer jar (§5). Every claim below
is confirmed against the data, analytic combinatorics, the decompiled viewer,
and a strategic sanity check on a real node.

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

## 5. Hand enumeration order — SOLVED (index → `q.k`)

**The `.rng` reader never needs to decode the pattern lines.** MonkerViewer's
own parser (`TreeFile.initFreqs`, decompiled) reads only the *payload* lines,
keyed by position, and its display code (`q.d()`) maps a stored position `i` to
a hand via `Client.a.b(i) == q.k.get(i)`. So:

> **The i-th payload in a `.rng` file is the hand `hand_order()[i]`.**

`q.k` is a fixed 16,432-entry list hardcoded in `monkerviewer-1.4.jar`
(concatenation of `Client.j.a + Client.j.b + Client.i.a + R.k`, ending `AAAA …
KKKK`). It is extracted to **`pipeline/plo/data/monker_hand_order.txt`** and
loaded by **`pipeline/plo/hand_order.py`**.

The pattern lines (`????`, `(2?)??`, …) are a **separate decorative annotation,
in a different order, that the viewer ignores** — which is exactly why decoding
them *as* the hand order kept failing (only 394 distinct patterns; a lossy
tree-delta). They are not used.

Notation in `q.k`: ranks `23456789TJQKA`; parens group same-suit cards (`AAKK`
rainbow, `(AK)(AK)` double-suited, `AA(2A)` = the 2 suited to one ace).

**Validated three independent ways** (`scripts/plo_hand_order_audit.py`):
1. MonkerViewer's own display code maps index → `q.k`.
2. **Bijection:** the 16,432 entries canonicalize to *exactly* the full set of
   suit-isomorphic PLO hands (enumerated from all 270,725 combos).
3. **Strategy:** on `40100.rng` (LJ open-raise), AAKK-ds / JT98-ds open 100% and
   rainbow trash (2233, K723) folds 100% — i.e. the map reads correct poker.

## 6. References

- [OpenHUD/monkerware](https://github.com/OpenHUD/monkerware) — JS `.rng`
  importer (confirms the `p;ev*1000` payload; NLHE hand labels only).
- [OwenQian/MonkerConverter](https://github.com/OwenQian/MonkerConverter) —
  filename action-token decoding + seat order.
- `monkerviewer-1.4.jar` (decompiled with CFR) — the authoritative source for
  the hand order (`Client.q.k` / `Client.a.b`) and the payload parser
  (`TreeFile.initFreqs`).
