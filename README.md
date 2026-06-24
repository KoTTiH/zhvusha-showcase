# ZHVUSHA Agent Runtime Showcase

Portfolio snapshot of a Telegram-based personal AI agent with persistent memory,
capability-scoped skills, durable agent jobs and test-enforced architecture
boundaries.

This repository is a sanitized public case study. It keeps the engineering
surface that is useful to review: source code, tests, migrations, configuration
templates and a short architecture overview. Private runtime state, chat logs,
local workspaces, production secrets, operational reports and full development
history are intentionally excluded.

## What It Demonstrates

- Async Python 3.12 application design with aiogram, Pydantic v2 and SQLAlchemy.
- Provider-agnostic LLM routing through model tiers instead of hardcoded models.
- PostgreSQL + pgvector memory and knowledge storage.
- Redis-backed event/runtime patterns for background work.
- Capability-scoped Agent Runtime: durable jobs, invocation profiles, tool
  gateway checks, events, artifacts and structured results.
- Skill isolation through a common invocation lifecycle.
- Contract tests and import-linter rules that protect module boundaries.
- Safety defaults for side-effectful work: read-only first, explicit approval
  for write/publish/send/login-like actions.

## Architecture

The code is organized around explicit capability modules:

- `src/llm/` - LLM gateway, provider registry and tier routing.
- `src/memory/` - episodic memory, enrichment, consolidation and staging.
- `src/knowledge/` - knowledge store and MCP-facing access layer.
- `src/agent_runtime/` - job models, runtime, profiles, routing and workers.
- `src/skills/` - bounded skills invoked through the shared skill contract.
- `src/bot/` - Telegram interface and orchestration wiring.
- `src/daemon/` - background signals, safety and event-driven loops.

See [docs/architecture.md](docs/architecture.md) for the design notes used in
this public snapshot.

## Run Locally

Prerequisites:

- Python 3.12
- Docker with Compose
- `uv` or a normal Python virtual environment

Create local config:

```bash
cp .env.example .env
```

Start dependencies:

```bash
docker compose up -d
```

Install and run checks:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ --strict
uv run pytest -q --no-cov tests/agent_runtime tests/skills/test_invocation.py
```

To run the bot you must provide real Telegram and LLM credentials in `.env`.
The public snapshot does not include any live credentials, sessions, private
workspace data or production database dumps.

## Public Scope

This is not a turnkey hosted service. It is a codebase review artifact meant to
show engineering decisions, architecture boundaries and test coverage for a
non-trivial personal-agent system.

Excluded from this snapshot:

- `.env`, Telegram sessions, database dumps and local workspaces.
- Chat logs, runtime artifacts, reports and benchmark evidence.
- Private development history and operator-specific automation settings.
- Production deployment credentials.

## License

Source-available for portfolio review. See [LICENSE](LICENSE).
