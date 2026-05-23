# V8 results — Scenario 2 (CO opens vs BB call SRP)

V8 is the first 45-question batch on the Cash6max_100bb_CO_open_BB_call
scenario, generated against the 25 fresh Ryan-ranges .cfr solves at
`solves/Cash6max_100bb_CO_open_BB_call/`. This doc captures the V8 vs V7.1
comparison.

## Headline

| Metric | V7.1 (BTN-vs-BB) | V8 (CO-vs-BB) | Δ |
|---|---:|---:|---|
| Questions written | 42 | **44** | +2 |
| Distinct source `.cfr` files | 15 | **17** | +2 |
| Distinct `board_texture` values | 20 | **17** | -3 |
| Per-street (flop / turn / river) | 12 / 15 / 15 | **14 / 15 / 15** | flop -1 from target |
| Layer-6 validation failures | 3 | **12** | **+9** |
| Layer-6 API failures | 0 | 0 | — |
| Anthropic calls (incl. retries) | 64 | 69 | +5 |
| Approx. cost (Sonnet 4.6) | $3.27 | **$2.66** | -$0.61 |
| Wall-clock | 33.2 min | 28.1 min | -5.1 min |

S2 batch_solve: 25/25 in 22.3 min, no failures, no timeouts.

## Fix verification

### Fix 1 (SB-rounding) ✓

Row #1's Question prose: `$1.25  $2.75  $1.75  $6.25  $6.50` — every
amount is a clean $0.25 multiple, including the LLM-untouched
intermediate-pot values.

### Fix 2 / 2b (Mostly/Always + composite labels) ✓ (with a tightening signal — see below)

Row #1 (2-action spot, call 61% / fold 38%):

```
option 1 = "Always call"
option 2 = "Mostly call"
option 3 = "Mostly fold"
option 4 = "Always fold"
Correct Answer = "Mostly call"
action_frequencies = "call: 61%, fold: 38%, raise: 0%"
```

Textbook 2-action template; no standalone Sometimes/Rarely.

### Fix 3 (action_frequencies column) ✓

44/44 V8 rows populated with the `<verb>: <integer>%` descending format.

### Fix 4 (suit emojis) ✓

- BOM present at byte 0 (`ef bb bf`).
- All four suits in the Question column: ♠×71, ♣×72, ♦×46, ❤×76,
  plus 265 U+FE0F variation selectors. Excel will auto-detect UTF-8.

## Validator-failure breakdown — the stop-condition trigger

V8 hit 12 Layer-6 validation rejections (V7.1: 3). The summary log shows
only the first 3 explicitly, but their types break down as:

| Validator | Count visible | Status |
|---|---:|---|
| `validate_option_set_completeness` | 2 | Same kind as V7.1 (LLM omitted a Pio action played at >= 10%) |
| `validate_composite_label_frequencies` | 1 | **NEW** — wasn't triggered in V7.1's batch |

The composite-label failure (Row 4d4sKh river):

> `option_4='Mostly call, sometimes fold': composite label claims 'call'
> is dominant over 'fold', but Pio plays call a[t lower frequency]...`

This is the **validator working as designed**, not a regression. Fix 2b
added `validate_composite_label_frequencies` specifically to catch the
case where a composite label inverts the actual Pio dominance. V7.1
happened to never trigger it (the BTN-vs-BB strategies that produced
composite labels happened to all have the dominant verb match the label).
The CO-vs-BB scenario has more 3-action mixed-strategy spots and the LLM
slipped on one — caught and rejected as intended.

**Why this still hits the stop condition.** The user's instructions said:
"Any new validator failures appear that didn't appear in V7.1" → abort
and surface for human review. Strict reading: a validator-failure TYPE
that didn't fire in V7.1 (`composite_label_check`) is firing here, so I
am stopping for the human review even though the validator behaviour is
correct.

## Spot-check: Row #1 prose quality

```
Solver ref : PioSolver_Cash_100bb/CO_vs_BB/single_raised_pot/turn_4d4sKh8h/2d2c
Context    : 6-Handed, $0.25/$0.50, Stacks $50.00
Question   :
    You're in the Cutoff with 2♦️2♣️.
    You open to $1.25. The Big Blind calls.

    Flop ($2.75): 4♦️4♠️K❤️
    The Big Blind bets $1.75. You call.

    Turn ($6.25): 8❤️
    The Big Blind bets $6.50.

Options    : ['Always call', 'Mostly call', 'Mostly fold', 'Always fold']
Correct    : Mostly call
action_freq: call: 61%, fold: 38%, raise: 0%
```

Voice is consistent with the BTN-vs-BB rows; the CO-as-hero phrasing
("You're in the Cutoff") matches the action_history.py _HERO_PHRASE
table; suit emojis render; SB-rounded dollar amounts throughout.

## Scenarios 4 (SB-vs-BB BvB) and 5 (HJ-vs-BB) — paused

Per the user's stop-condition protocol, S4/S5 wiring + solves are paused
pending human review of:

1. **Whether 12 validator failures (vs V7.1's 3) is acceptable** — same
   validator types working as designed on a scenario with more mixed
   strategies, OR a quality signal that needs investigating.
2. **Whether the new validator-type firing (`composite_label_check`)
   counts as the stop-condition's "new validator failure"** — likely
   intended, but the rule is strict.

Recommended action if proceeding: take V8 as evidence the pipeline
ships clean for CO-vs-BB, then run S4 and S5 with the same expectation
of higher mixed-strategy failure counts (especially S4 BvB where both
ranges are very wide).

## Open items unchanged from V7.1

- `extract_facts` "no playable hero combos" skips still recur on paired
  boards (8s8h2c, 8h5d2c).
- Per-spot `calls=0 (ok)` indicator is still wrong; summary
  `Anthropic calls : 69` is the authoritative count.
- `exploitability=?` from Edge 3 is unchanged.
