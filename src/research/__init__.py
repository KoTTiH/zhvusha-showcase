"""Research leaf module — short-time-budgeted multi-source lookup.

Used by :mod:`src.skills.ideation_to_spec` (Phase 12) before generating a
draft ``tasks/<slug>.yaml``. The module is a leaf: it has zero
``src.llm``/``src.skills``/``src.memory``/``src.knowledge`` imports and
receives KB and code-search callables through the service constructor.
This keeps the import-linter ``leaf_modules_isolation`` contract green.

Implementation status (Phase 10): MVP — KB + code search only. Web search
and KB-cache of ``research_finding`` records are deferred to Phase 11+.
"""

from src.research.presets import PRESETS, ResearchPreset
from src.research.protocols import (
    Citation,
    CitationSource,
    CodeSearchCallable,
    KBSearchCallable,
    ResearchResult,
    ResearchSourceCallable,
)
from src.research.service import ResearchService

__all__ = [
    "PRESETS",
    "Citation",
    "CitationSource",
    "CodeSearchCallable",
    "KBSearchCallable",
    "ResearchPreset",
    "ResearchResult",
    "ResearchService",
    "ResearchSourceCallable",
]
