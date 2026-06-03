# PLO Port Assessment — can we reuse this pipeline for Pot-Limit Omaha?

**Status:** **build underway (June 2026).** The viability question is settled —
the PLO port is happening, and its whole *foundation* is built, tested, and
committed (see "Build status" below). The original assessment is kept further
down for context. The NLHE preflop pipeline remains live + hardened.

## Build status — June 2026 (read this first)

**Decisions locked:** same repo + same admin panel with a game-type selector
(NLHE/PLO), **not** a separate site or a separate git branch; **range display
dropped** for PLO (generation only). Code lives in `pipeline/plo/`; the pack
lives in `plo_ranges/` (gitignored, 3.8 GB).

**The proprietary Monker `.rng` format is fully reverse-engineered + validated**
— full spec in [`plo_rng_format.md`](plo_rng_format.md). The purchased pack is
`PLO 6max 100bb (rake 5% / 1bb cap)`; its authoritative 16,432-hand order was
extracted from the decompiled MonkerViewer jar and validated three ways (the
jar's own code · a bijection onto the canonical PLO hand set · strategy sanity
on a real node). **Do not re-crack this — it's solved.**

**Built + committed (all tested, ruff + mypy clean):**
- `pipeline/plo/equity.py` — 4-card "best 2-of-4 + 3-of-5" equity evaluator
  (reuses the NLHE 5-card ranker; no new dependency).
- `pipeline/plo/hand_model.py` — `classify_plo_hand` (suit pattern, pairing,
  connectedness/wrap, nut-flush, danglers, tunable strength) — exhaustively
  audited over all 270,725 combos.
- `pipeline/plo/concept_tags.py` — 27 hand-structure tags (exact; same audit).
- `pipeline/plo/hand_order.py` (+ `data/monker_hand_order.txt`) — the `.rng`
  index→hand map; `scripts/plo_hand_order_audit.py` reproduces the validation.
- `pipeline/plo/pack.py` — `read_rng` (a node's range: hands with p>0 + ev in
  small blinds) and `parse_node_path` (filename → action sequence; seats
  `LJ,HJ,CO,BU,SB,BB`; tokens `0`/`1`/`3`/`40100`). Verified on the real pack
  (LJ opens 19.8%; top-EV opens are AAQQ/AAJJ double-suited).

**Next (assembly — mirrors the NLHE preflop pipeline, keep the "LLM never
thinks about poker" boundary):**
1. **Node enumeration** — group an actor's sibling action-files at a decision
   point; derive each hand's conditional strategy (raise/call/fold freqs) from
   its `p` across those files. Analog of `pipeline/preflop/node_enumerator.py` +
   `spot_sampler.py`.
2. **`PloFacts`** — equity + hand model + archetype + blockers per spot.
3. **Tags / archetypes / difficulty / skills** — wire the built hand tags; add
   the facts-relative tags, the PLO difficulty axes, and the skill mapper.
4. **Layer 6** — PLO gold examples + voice → prose.

**PLO skills catalog (designed with Zach):**
- *A — carry-over preflop decisions:* Preflop Hand Selection, 3-Betting, Facing
  a 3-Bet, 4-Betting, Facing a 4-Bet, Squeezing, Facing a Squeeze, Blind
  Defense, Blind vs Blind, Pot Odds, In/Out of Position, Multiway Pot Strategy.
- *B — PLO hand-reading (the new edge):* **Suitedness** (one master skill),
  Rundowns & Connectivity, Dangler Awareness, Nut-Flush Awareness, Big-Pair
  Construction (AAxx/KKxx with support), **Nuttedness & Non-Nut Traps** (one
  master skill — *not* split into flush/straight/set).
- *C — math / dynamics:* Reverse Implied Odds, Pot-Limit Bet Sizing. ("Equity
  Runs Close" was cut.)
- *D — postflop (later):* Wrap/Draw Strategy, Nut Blockers & Card Removal, Set
  Mining & Redraws, C-Betting, Bluff Catching, Pot Control.

The skill mapper must be spot-level (strict, ~2–4 skills/question) so it needs
the facts layer; the hand-reading skills map onto the already-built concept tags.

---

## TL;DR

**Same architecture, new guts.** The 8-layer pipeline shape, the core principle
("the LLM never thinks about poker — solver = truth, Python formalizes it, the
LLM only writes prose"), the admin panel, the CSV/Sheets format, the difficulty
framework, and the validator/guard structure all transfer almost wholesale —
roughly **the "spine," ~half the value**. The poker-specific guts — hand
abstraction, range representation + range UI, equity engine, concept tags,
archetypes, bet-sizing, pack parser — are **PLO-specific rewrites** (~the other
half). Buying a PLO preflop pack gives you a solver source but is **not a
drop-in**. Budget it as **~half a new build**.

## What transfers (the spine — reuse the design wholesale)

- **The core principle.** Variant-independent — and *more* valuable for PLO,
  since LLMs are even more confidently wrong about Omaha (less training data,
  worse equity intuition). The reason this project exists is amplified in PLO.
- **The 8-layer pipeline shape** (spot generator → tree resolver → path sampler
  → question extractor → fact extractor → explanation generator → validator
  stack → format writer).
- **The admin panel** — generate / review / compare / range-viewer /
  prompt-workshop, the 4-axis difficulty framework, the CSV/Sheets format, cost
  tracking, auto-save, the guards + validators *structure* (the specific rules
  change; the framework doesn't).
- **The deterministic action-history renderer** and the LLM-prose layer
  *structure* (swap the gold examples + voice content, not the mechanism).

## What's a PLO rewrite (the card/strategy guts)

1. **Hand abstraction — THE big one.** NLHE collapses to **169 classes** (the
   13×13 AKs/AKo/22 grid), and that abstraction is baked *everywhere*: range
   files, the Range Viewer grid, `hand_class`, the hand-axis of difficulty,
   option templates. PLO is **270,725 combos** with no clean 169-equivalent —
   you reason in rundowns, double-/single-suited vs rainbow, wraps,
   pairs+connectors, danglers, nuttedness. **The 13×13 grid UI does not apply**
   (displaying a PLO range is its own design problem). Hand model + range
   representation = new.
2. **Equity engine.** Today's Monte Carlo is 2-card-vs-range. PLO needs a
   **4-card "best 2-of-4 hole + 3-of-5 board" evaluator**, far more samples, and
   PLO equities run much closer (small edges, big multiway pots). New evaluator.
3. **Concept tags + archetypes + blockers.** The registry-of-pure-functions
   *pattern* reuses; the content is PLO-specific. **Blockers are dramatically
   more important and complex** in PLO (nut blockers, redraws, freerolls). The
   16 NLHE archetypes don't map 1:1.
4. **Bet sizing.** It's *pot-limit* — max bet is pot-constrained, so the
   raise-size math differs from NLHE opens.
5. **Multiway is the norm, not the exception.** PLO goes multiway constantly;
   the current facing-logic is mostly heads-up-shaped. Bigger lift.
6. **Pack parser + 4-card rendering** — pack grammar, range-file format, and
   `User Cards` / `Question` prose all assume 2 cards.

## On "just buy a PLO preflop pack"

Helps — you *do* need a solver source — but not plug-and-play:
- PLO packs are **much larger** (270k combos/node) and the format differs, so
  the pack parser/range loader is new work.
- Same hard requirement as NLHE: the pack must include **strategy at every
  intermediate preflop node**, not just final ranges — harder to satisfy in
  PLO's multiway trees.
- *Unverified:* specific vendors/formats and current PLO-solver capabilities
  (MonkerSolver does PLO; pre-solved PLO packs are sold). **Check before
  committing:** which solver, what stakes/rake, does it cover multiway, does it
  expose intermediate nodes.

## Recommendation

- **Viable, and the architecture is *more* valuable for PLO.**
- **Sequence:** prove NLHE first — get the Phase-1 quality gate done (a graded
  real-API batch showing ≥70% gold-equivalent) *before* splitting focus. PLO is
  a clean "v2," not a good parallel effort right now.
- **Fork vs shared core:** if serious about multiple variants long-term, extract
  a game-agnostic **core** (pipeline shape, admin panel, CSV format, difficulty,
  LLM-prose layer) with **per-variant plugins** (hand model, equity, tags, pack
  parser, range UI). The current NLHE code wasn't built with that plugin
  boundary, so that's a refactor — but cheaper than maintaining two diverging
  copies (every fix from the June-2026 session would otherwise need doing
  twice). If PLO is just an experiment, fork to move fast.

## Open questions to resolve next session

1. **Solver/pack:** which PLO solver or pre-solved pack, at what stakes/rake/
   format? Does it cover multiway and expose intermediate-node strategies?
2. **Range display:** how do you represent *and render* a PLO range with no
   13×13 grid? (Category buckets? A grid of hand-types? Top-N combos?)
3. **Fork vs shared-core** decision.
4. **Spine vs plugin, file-by-file:** map which current NLHE modules are
   game-agnostic spine vs would-be PLO plugins (the concrete refactor plan).
