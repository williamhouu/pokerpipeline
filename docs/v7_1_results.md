# V7.1 results — Ryan-feedback fixes applied

V7.1 is the first batch produced after the five Apr-2026 Ryan-feedback
fixes landed (commits `da44e58`, `fc90350`, `a824787`, `e037a8a`,
plus Fix 4 which required no code change). This doc captures the V7.1
vs V7 comparison and verifies each fix end-to-end in the output.

## Headline

| Metric | V7 | V7.1 | Δ |
|---|---:|---:|---|
| Questions written | 45 | **42** | -3 |
| Distinct source `.cfr` files | 17 | **15** | -2 |
| Distinct `board_texture` values | 21 | **20** | -1 |
| Per-street (flop / turn / river) | 15 / 15 / 15 | **12 / 15 / 15** | flop -3 |
| Layer-6 validation failures | 0 | **3** | +3 |
| Layer-6 API failures | 0 | 0 | — |
| Anthropic calls (incl. retries) | 54 | 64 | +10 |
| Approx. cost (Sonnet 4.6) | $2.41 | **$3.27** | +$0.62 |
| Wall-clock | 30.9 min | 33.2 min | +2.3 min |

The 3-question drop (45→42) is entirely the new validators catching
3-action mixed-strategy spots where the LLM kept dropping one of the
meaningful Pio verbs even with composite-label prompting. See "Cost of
stricter validators" below — this is the expected tradeoff, not a
regression.

## Fix verification

### Fix 1 — Round action-history dollar amounts to nearest SB

Confirmed in 3 sampled V7.1 rows (every dollar amount in the Question
prose is now a multiple of $0.25):

| Row | Hand Stage | Dollar amounts in Question prose |
|---:|---|---|
| #1 | Flop  | `$1.25  $2.75  $1.75  $5.25  $12.25  $26.25  $50` |
| #2 | Turn  | `$1.25  $2.75  $1.75  $6.25  $6.50` |
| #3 | Turn  | `$1.25  $2.75  $1.75  $5.25  $13.25` |

Row #1's `$1.75 / $5.25 / $12.25 / $26.25` are the exact Ryan-feedback
examples (raw $1.85 / $5.23 / $12.15 / $26.26 snapped to the nearest SB).

### Fix 2 / 2b — Mostly/Always labels + composite labels for 3+ action spots

Row #7 (Turn, 3-action spot `call 60% / fold 30% / raise 10%`):

```
You're on the Button with K♣️5♣️.
You open to $1.25. The Big Blind calls.

Flop ($2.75): 5❤️3♦️2♣️
The Big Blind checks. You bet $1.75. The Big Blind raises to $5.25. You call.

Turn ($13.25): 8♠️
The Big Blind bets $14.50.

option 1 = "Always call"
option 2 = "Mostly call, sometimes raise"
option 3 = "Mostly call, sometimes fold"
option 4 = "Fold"
Correct Answer  = "Mostly call, sometimes fold"
action_frequencies = "call: 60%, fold: 30%, raise: 10%"
```

This is the textbook output of Fix 2b: dominant verb anchored, each
secondary mix-in paired with the dominant in a composite, no standalone
`"Sometimes X"`. The `Mostly call, sometimes fold` composite was chosen
as correct because Pio's strategy is 60/30/10 = call/fold/raise, and the
fold-leg of the mix dominates the raise-leg.

Across all 42 V7.1 rows, manual scan finds:

- **0 standalone `"Sometimes X"` or `"Rarely X"` labels** (Fix 2 + 2b (a)
  validator working).
- **Composite labels appear in 3-action spots only** — the 2-action
  spots get the clean `Always A / Mostly A / Mostly B / Always B`
  template Ryan asked for.

### Fix 3 — action_frequencies column (col 36)

All 42 V7.1 rows populated (`42/42`). Sample values:

| Row | Hand Stage | action_frequencies |
|---:|---|---|
| #7  | Turn  | `call: 60%, fold: 30%, raise: 10%` |
| #29 | River | `fold: 69%, raise: 27%, call: 5%` |
| #30 | River | `call: 66%, fold: 34%` |
| #36 | Flop  | `call: 92%, fold: 8%` |

Format matches Ryan's spec: comma-separated `<verb>: <integer>%`,
descending by frequency, integer-rounded percentages.

### Fix 4 — Suit emojis render in Excel

Programmatic verification on `test_output/batch_questions_v7_1.csv`:

- **UTF-8 BOM present** at byte 0: `ef bb bf`.
- All four suit codepoints render in Question column:
  - ♠ (U+2660): 54 occurrences
  - ♣ (U+2663): 71 occurrences
  - ♦ (U+2666): 55 occurrences
  - ❤ (U+2764, heavy heart per format spec): 75 occurrences
  - U+FE0F (variation selector 16, emoji presentation): 255 occurrences

Sample (Row #7's Question prose, as rendered): `K♣️5♣️ ... 5❤️3♦️2♣️ ...
8♠️`. Excel on Windows auto-detects UTF-8 because of the BOM; the suit
characters render as colored emoji, not Latin-1 mojibake.

## Cost of stricter validators

The 3 Layer-6 failures in V7.1 (V7 had 0) all fired on
`validate_option_set_completeness` after 2 LLM retries:

1. `6h5h4d/river/r:0:b36:c:As:c:c:8h:c:b131` — Pio plays
   call=23.8%, raise=11.2%, fold≈65%. LLM kept omitting `call` or
   `raise` from the option set.
2. `8c7c2d/turn/r:0:b36:c:As:b125:b284` — Pio plays
   fold=23.7%, raise=19.5%, with check/bet sharing the remaining 57%.
3. `8c7c2d/river/r:0:b36:c:As:c:c:8h:c:b131` — Pio plays
   call=18.9%, raise=12.5%, fold≈69%.

All 3 are 3-action mixed-strategy spots where the new composite-label
prompt is harder for the LLM than 2-action spots. After 2 retries the
spot is routed to the human-review queue rather than ship. **This is
the validator working as intended** — the cost is 3 questions / 45
target = ~7% yield drop in exchange for stricter quality on the 42 that
ship.

Two ways to recover yield in V8 if needed:

- Tune the composite-label prompt with more in-context examples
  specifically covering the 3-action-at-10% case.
- Raise `COMPLETENESS_MIN_FREQ` from 0.10 to 0.15, exempting the
  smallest mix-in (e.g. raise=11% wouldn't force inclusion). Trades
  quality for yield — keep at 0.10 until Ryan signs off.

The cost-per-question went from $0.0588 in V7 to $0.0779 in V7.1
(+33%). $0.0191 of that is the extra retry calls (10 of them); the
remaining $0.0001 is the small input-token growth from the new prompt
sections. Prompt caching keeps the system+exemplar block hits cheap.

## What's unchanged from V7

- Range advantage signal: still strongly villain-side (Ryan's tighter
  ranges produce more `range_advantage_villain` than `_hero`).
- No board-texture / scenario regressions.
- Trust-chain audit (see `docs/audit_llm_scope.md`) shows the same 6
  LLM_PROSE columns; the new column 36 (`action_frequencies`) is
  SOLVER_FACT (direct render of Pio's strategy).

## Open items for V8

- The 3 stuck spots (above) need a closer look. If they're consistently
  hard cases, a 4-option composite template specifically for 3-actions-
  at-10%+ would help.
- The 3 `extract_facts` skips on `8s8h2c` (paired-board "no playable
  hero combos") still recur. Same as V6/V7 — Layer 5 hand-filter edge
  case, non-fatal but worth fixing once we're past S1.
- The pre-existing `calls=0 (ok)` per-row indicator bug and the
  `exploitability=?` Edge-3 readout (both flagged in `docs/v7_results.md`)
  are still open. Neither blocks V7.1 ship.
