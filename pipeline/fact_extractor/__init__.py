"""Layer 5 -- Fact Extractor.

Turns solver output into the structured data block every explanation is built
from: hand class, board texture, equity math, and concept tags. All strategic
reasoning lives here, in deterministic Python -- never in the LLM.

See docs/engineering_brief.docx, "Layer 5: Fact Extractor".
"""
