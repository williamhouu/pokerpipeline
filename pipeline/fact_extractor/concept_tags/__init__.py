"""Concept tag rules (Fact Extractor / Layer 5).

The 42 concept tags from docs/engineering_brief.docx, "Concept Tag Library
Specification", split into one module per section for maintainability:

    section_a_range.py            Section A -- range characterization (4)
    section_b_decision_class.py   Section B -- decision class (9)

Each tag is a pure function `SpotData -> bool`. The `compute_tags` registry
that runs every tag over a spot is built once all sections are implemented.
"""
