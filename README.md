# ZHVUSHA Agent Runtime Showcase

Публичный портфолио-срез Telegram-агента с долговременной памятью, skills с
ограниченными capabilities, долговечными agent jobs и тестами, которые защищают
архитектурные границы.

Этот репозиторий - очищенный публичный кейс. В нем оставлена инженерная часть,
которую полезно ревьюить: исходный код, тесты, миграции, шаблоны конфигурации и
короткое описание архитектуры. Приватное runtime-состояние, логи переписок,
локальные рабочие данные, production-секреты, отчеты и полная история разработки
намеренно исключены.

English version is available below.

## Что Показывает Проект

- Асинхронный Python 3.12-сервис на aiogram, Pydantic v2 и SQLAlchemy.
- Маршрутизацию LLM-провайдеров через tiers, без жесткой привязки к одной
  модели.
- PostgreSQL + pgvector для памяти и knowledge storage.
- Redis-based event/runtime patterns для фоновой работы.
- Agent Runtime с ограничениями по capabilities: durable jobs, invocation
  profiles, tool gateway checks, events, artifacts и structured results.
- Изоляцию skills через общий invocation lifecycle.
- Контрактные тесты и import-linter rules, которые защищают module boundaries.
- Safety defaults для действий с побочными эффектами: read-only first и explicit
  approval для write/publish/send/login-like actions.

## Архитектура

Код организован вокруг явных capability modules:

- `src/llm/` - LLM gateway, provider registry и tier routing.
- `src/memory/` - episodic memory, enrichment, consolidation и staging.
- `src/knowledge/` - knowledge store и MCP-facing access layer.
- `src/agent_runtime/` - job models, runtime, profiles, routing и workers.
- `src/skills/` - bounded skills, вызываемые через общий skill contract.
- `src/bot/` - Telegram interface и orchestration wiring.
- `src/daemon/` - background signals, safety и event-driven loops.

Подробнее: [docs/architecture.md](docs/architecture.md).

## Локальный Запуск

Нужно:

- Python 3.12
- Docker с Compose
- `uv` или обычный Python virtual environment

Создать локальный конфиг:

```bash
cp .env.example .env
```

Запустить зависимости:

```bash
docker compose up -d
```

Поставить dev-зависимости и запустить проверки:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ --strict
uv run pytest -q --no-cov tests/agent_runtime tests/skills/test_invocation.py
```

Для запуска бота нужны реальные Telegram и LLM credentials в `.env`. Публичный
срез не содержит live credentials, sessions, private workspace data или
production database dumps.

## Публичные Границы

Это не готовый hosted service. Это артефакт для ревью кодовой базы, который
показывает инженерные решения, архитектурные границы и test coverage для
non-trivial personal-agent system.

Исключено из публичного среза:

- `.env`, Telegram sessions, database dumps и local workspaces;
- chat logs, runtime artifacts, reports и benchmark evidence;
- private development history и operator-specific automation settings;
- production deployment credentials.

## Лицензия

Source-available for portfolio review. См. [LICENSE](LICENSE).

## English

Public portfolio snapshot of a Telegram-based personal AI agent with persistent
memory, capability-scoped skills, durable agent jobs and test-enforced
architecture boundaries.

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
