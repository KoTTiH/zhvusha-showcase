# Agent Runtime Rules

- Agent Runtime workers return durable structured results: context capsules,
  findings, sources, artifacts, errors, memory candidates and constraints.
- Workers do not write final user-facing answers. The orchestrator owns final
  synthesis.
- Invocation profiles grant the minimum required capabilities for each job.
- Denied capabilities must be enforced by the Tool Gateway, not only described
  in prompts.
- Read-only profiles do not receive write, submit, login, publish or send
  capabilities.
- External side effects require explicit policy and approval paths.
- Worker `next_actions`, audit output and debug traces are internal material
  for orchestration and consolidation, not direct user-facing text.
- Changes to runtime contracts, profiles, Tool Gateway behavior or worker output
  schemas require focused tests in `tests/agent_runtime`.
