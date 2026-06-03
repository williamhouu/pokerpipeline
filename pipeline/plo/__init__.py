"""Pot-Limit Omaha pipeline (fork of ``pipeline.preflop``).

This package holds the PLO-specific "card guts" of the question-generation
pipeline. Game-agnostic spine modules (position, the worthiness gate, EV
arithmetic) are imported from :mod:`pipeline.preflop` rather than copied, so
they don't diverge; only card-touching layers live here.

First module: :mod:`pipeline.plo.equity` -- a 4-card "best 2-of-4 hole +
3-of-5 board" evaluator built on the existing 5-card ranker. Buildable and
validatable with no preflop pack in hand.
"""
