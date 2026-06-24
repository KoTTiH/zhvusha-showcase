"""Research strategy presets — hard-coded per KB #73.

Strategy is part of the spec, not a soft heuristic the LLM can break under
task pressure. Four presets cover the realistic call sites:

* ``foundational`` — long-stable concepts (e.g. trust hierarchy, TDD).
  KB-only. Code search disabled — the goal is conceptual grounding,
  not implementation detail.
* ``current_practices`` — recent best-practice from vendor blogs / lab
  posts. KB + code search.
* ``api_integration`` — wiring against an external library / service.
  KB + code search; downstream callers are expected to add web/docs
  in Phase 11+ when web search is wired up.
* ``hot_topic`` — only-just-published research, narrow trust whitelist.
  KB + code; web reserved for Phase 11+.

The whitelist of acceptable web domains (KB #73 specifies "whitelist lives
in KB, not in code") is intentionally absent from this file — Phase 11
will fetch it through the KB at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchPreset:
    """Per-strategy configuration consumed by :class:`ResearchService`."""

    name: str
    use_kb: bool
    use_code_search: bool
    use_web: bool
    max_age_days: int | None  # None = any age (foundational)
    trust_minimum: float  # 0.0-1.0; downstream callers may inspect
    budget_seconds: int | None = None  # None = no time limit
    max_sources: int | None = None  # None = unlimited


PRESETS: dict[str, ResearchPreset] = {
    "foundational": ResearchPreset(
        name="foundational",
        use_kb=True,
        use_code_search=False,
        use_web=False,
        max_age_days=None,
        trust_minimum=0.5,
    ),
    "current_practices": ResearchPreset(
        name="current_practices",
        use_kb=True,
        use_code_search=True,
        use_web=False,  # Phase 11+: enable web with vendor-blog whitelist
        max_age_days=540,  # ≈18 months
        trust_minimum=0.7,
    ),
    "api_integration": ResearchPreset(
        name="api_integration",
        use_kb=True,
        use_code_search=True,
        use_web=False,  # Phase 11+: enable web with official-docs whitelist
        max_age_days=180,  # ≈6 months
        trust_minimum=0.7,
    ),
    "hot_topic": ResearchPreset(
        name="hot_topic",
        use_kb=True,
        use_code_search=True,
        use_web=False,  # Phase 11+: enable web with narrow vendor-blog list
        max_age_days=30,
        trust_minimum=0.7,
    ),
    "bug_investigation": ResearchPreset(
        name="bug_investigation",
        use_kb=True,
        use_code_search=True,
        use_web=False,
        max_age_days=None,
        trust_minimum=0.5,
        budget_seconds=60,
        max_sources=5,
    ),
}
