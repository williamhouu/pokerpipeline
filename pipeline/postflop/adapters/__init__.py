"""Real-solve adapters: vendor formats -> the postflop IR.

Each module here is the ONLY code that knows a specific vendor solve format;
it populates a :class:`pipeline.postflop.solve.PostflopSolve` and hands it to
the (solver-agnostic) pipeline. ``validate_solve`` is the contract to target.
"""
