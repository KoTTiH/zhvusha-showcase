# Skills Boundary Rules

- A skill is a body-layer capability, not an independent mind.
- If a skill gathers data, starts an Agent Runtime job or calls an external
  tool, its normal output for further reasoning is a structured observation in
  `SkillResult.metadata.body_observation`.
- Results that require orchestrator interpretation must set
  `SkillResult.metadata.requires_zhvusha_response = True`.
- Skills should not expose context capsule `next_actions`, handoff text, debug
  traces or raw worker output as final user-facing text.
- Routing and response bugs should be fixed at the contract boundary, not with
  one-off phrase, regex or prompt branches.
- Every new or changed skill should have a focused test in `tests/skills`; if it
  changes a user-facing dispatcher path, add or update a `tests/bot` case too.
