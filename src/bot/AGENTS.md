# Bot Orchestration Rules

- `src/bot` owns the user-facing orchestration boundary.
- Before changing dispatcher paths, identify which layer forms the final answer:
  the main orchestrator or a body-layer skill/tool.
- If `SkillResult.metadata.requires_zhvusha_response is True`, production code
  must synthesize through `ChatResponseSkill` instead of sending raw
  `SkillResult.response`.
- Avoid ad hoc keyword branches for ordinary long text. Narrow control/status
  fast paths are acceptable only when ambiguous text still has a classifier or
  LLM fallback.
- Changes to text dispatching, `SkillInvocationService` wiring, body observation
  synthesis or chat-facing routing require focused tests in `tests/bot` or an
  existing test that covers the same boundary.
