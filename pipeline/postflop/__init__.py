"""Postflop question-generation pipeline (flop / turn / river).

This package is a SEPARATE, self-contained pipeline that mirrors the
architecture of :mod:`pipeline.preflop` but for postflop decision spots.
It is deliberately kept apart from the preflop NLHE and PLO generators so
that work here cannot accidentally change their behaviour: the only things
postflop borrows from the rest of the repo are a handful of *pure,
game-agnostic leaf utilities* (card parsing, the 7-card equity evaluator,
the hand-class and board-texture classifiers, and the shared Layer-6 LLM
call wrapper). It never imports the preflop/PLO batch drivers, fact
extractors, validators, or format writers, and it never mutates a shared
module.

The governing rule is the same one that governs the whole repo:

    The LLM never thinks about poker. The LLM only writes the words.

Every strategic fact (correct action, equity, frequencies, board texture,
hand class, concept tags, archetype) is produced deterministically from
solver output; Layer 6 turns that resolved data block into prose.

Architecture
------------
The pipeline is built on a *solver-agnostic intermediate representation*
(:mod:`pipeline.postflop.solve`) so the rest of the code never touches a
solver file format directly:

    PioSolver .cfr  ─┐
    third-party .db ─┼─► adapter ─► PostflopSolve (IR) ─► pipeline ─► CSV
    (future formats)─┘                     ▲
                                  synthetic fixture (tests / demo)

Because the pipeline runs entirely off the IR, the whole thing is testable
and demoable with NO external solver file (important: this Mac cannot run
PioSolver, and the trial solves are multi-GB and live outside the repo).
:mod:`pipeline.postflop.fixtures` provides an in-memory ``PostflopSolve``
that drives the end-to-end tests.

The 8 layers (see each module's docstring):

    1-2  solve / fixtures      the IR + a synthetic BTN-vs-BB SRP solve
    3    spot_sampler          PostflopNode + hero combo -> PostflopSpot
    4    question_extractor     worthiness gate (freq window + EV gap)
    5    facts                  PostflopSpot -> PostflopFacts (equity, hand
                                class, board texture, blockers, SPR, pot
                                odds, archetype, concept tags)
    6    explanation_generator  the ONLY LLM step (dry-run placeholder or
                                a real Anthropic call + validate/retry)
    7    validators             deterministic post-LLM checks (hard + soft)
    8    format_writer          the team's CSV row + multi-street action
                                history

    action_history             deterministic Context/Question prose that
                                renders the ENTIRE line ahead of the
                                decision (a turn question shows the preflop
                                AND flop action)
    options / difficulty       answer-option builder + difficulty score
    batch                      generate_postflop_batch end-to-end driver

What is NOT done yet (documented extension points -- see README.md):
    * real-solve adapters (.cfr via the Pio UPI client; the third-party .db)
    * the admin-panel Generate/Review pages
    * LLM prompt tuning against gold postflop examples
"""

from __future__ import annotations
