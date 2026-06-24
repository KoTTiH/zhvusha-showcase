# Architecture Overview

ZHVUSHA is a personal-agent codebase built around one central orchestration
loop and several capability-scoped body layers.

The important design choice is that tools do not become independent assistants.
They return structured observations, proposals, artifacts and errors. The main
orchestrator owns user-facing synthesis, follow-up decisions and memory updates.

## Runtime Flow

```text
Telegram / operator message
  -> bot dispatcher
  -> skill invocation service
  -> Agent Runtime job when work is delegated or long-running
  -> invocation profile and capability gateway
  -> tool gateway
  -> bounded worker or skill
  -> events, artifacts, structured result
  -> orchestrator response and memory staging
```

## Core Boundaries

- `src/llm` is the LLM gateway. Other modules route through tiers and do not
  hardcode provider adapters.
- `src/memory` owns episodic memory, consolidation and staging. Background
  processes propose memory candidates instead of mutating private memory
  directly.
- `src/knowledge` owns external knowledge storage and MCP-facing access.
- `src/agent_runtime` owns durable jobs, profiles, capability declarations,
  worker routing and tool-gateway enforcement.
- `src/skills` owns bounded user-facing capabilities. Skills go through the
  shared invocation lifecycle: match, prepare, dry-run, approval and execute.
- `src/bot` wires Telegram-specific delivery, context handling and dispatcher
  decisions.

## Safety Model

The public snapshot keeps the same engineering principle as the private system:
read-only work is the default, and side effects require explicit capability and
approval paths.

Examples of side effects that must stay gated:

- writing files;
- editing environment values;
- sending Telegram messages;
- publishing content;
- logging in or submitting browser forms;
- restarting services;
- committing or pushing code.

The codebase uses both tests and `import-linter` contracts to keep these
boundaries visible.

## Testing Strategy

The repository contains a broad test suite. The most relevant tests for review
are contract and boundary tests around:

- agent runtime models and job lifecycle;
- capability graph and invocation profiles;
- skill invocation behavior;
- LLM provider routing;
- memory/consolidation staging;
- Telegram/bot mode separation;
- safety and approval gates.

Full production-like verification requires real local services and credentials,
which are intentionally not included in this public snapshot.
