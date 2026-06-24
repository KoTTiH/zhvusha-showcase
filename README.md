# ZHVUSHA - Agent Runtime Showcase

[![Public Snapshot Quality](https://github.com/KoTTiH/zhvusha-showcase/actions/workflows/quality.yml/badge.svg)](https://github.com/KoTTiH/zhvusha-showcase/actions/workflows/quality.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20%2B%20pgvector-memory-4169E1)
![Source Available](https://img.shields.io/badge/license-source--available-lightgrey)

Публичный портфолио-срез ZHVUSHA: AI-агента для Telegram с долговременной
памятью, Agent Runtime, capability gates, изолированными skills, durable jobs и
CI-проверками архитектурных границ.

Этот репозиторий не является выгрузкой production-системы. Он специально очищен
для публичного ревью: оставлены код, тесты, миграции, конфиги и архитектурные
заметки; исключены секреты, логи переписок, runtime-артефакты, личные данные и
полная приватная история разработки.

English version is available below.

## Коротко

ZHVUSHA демонстрирует, как устроить personal-agent систему так, чтобы
инструменты не превращались в хаотичный набор prompt-веток:

- намерение пользователя проходит через orchestration layer, связанный с
  Telegram;
- долгие или рискованные операции оформляются как Agent Runtime jobs;
- каждый job получает минимальный capability profile;
- Tool Gateway проверяет, что worker не выходит за разрешенные действия;
- skills возвращают structured results, а главный orchestrator собирает ответ
  пользователю и memory updates.

## Что Здесь Важно Для Ревью

- Асинхронный Python 3.12-сервис на aiogram, Pydantic v2 и SQLAlchemy.
- LLM routing через tiers и provider registry, без жесткой привязки к одной
  модели.
- PostgreSQL + pgvector для episodic memory, knowledge storage и retrieval.
- Redis-backed event/runtime patterns для фоновых процессов.
- Agent Runtime: durable jobs, invocation profiles, capability graph, workers,
  events, artifacts и structured results.
- Общий lifecycle для skills: match, prepare, dry-run, approval, execute.
- Safety defaults: read-only по умолчанию; write/publish/send/login-like actions
  только через явные capability и approval paths.
- Контрактные тесты, `ruff`, `mypy --strict`, `pytest` и `gitleaks` в CI.

## Технический Профиль

| Зона | Стек / подход |
| --- | --- |
| Bot interface | aiogram, Telegram handlers, delivery layer |
| LLM layer | provider registry, model tiers, OpenAI-compatible adapters |
| Memory | PostgreSQL, pgvector, episodic memory, staging, consolidation |
| Runtime | durable jobs, capability profiles, tool gateway, workers |
| Background work | Redis, daemon signals, event-driven loops |
| Quality gates | ruff, mypy strict, pytest, import-linter, gitleaks |

## Архитектура

Основная идея: tools и workers не становятся самостоятельными ассистентами. Они
возвращают факты, артефакты, ошибки и предложения. Главный orchestrator решает,
что сказать пользователю, что сохранить в memory staging и какой следующий шаг
допустим.

```text
Telegram / operator message
  -> bot dispatcher
  -> skill invocation service
  -> Agent Runtime job
  -> invocation profile + capability gateway
  -> tool gateway
  -> bounded worker / skill
  -> events + artifacts + structured result
  -> orchestrator response + memory staging
```

Подробнее: [docs/architecture.md](docs/architecture.md).

## Структура

- `src/llm/` - LLM gateway, provider registry, tier routing.
- `src/memory/` - episodic memory, enrichment, consolidation, staging.
- `src/knowledge/` - knowledge store и MCP-facing access layer.
- `src/agent_runtime/` - job models, runtime, profiles, routing, workers.
- `src/skills/` - bounded skills через общий skill contract.
- `src/bot/` - Telegram interface и orchestration wiring.
- `src/daemon/` - background signals, safety, event-driven loops.
- `tests/` - contract, boundary и integration-style tests.

## Локальный Запуск

Нужно:

- Python 3.12;
- Docker с Compose;
- `uv` или обычный Python virtual environment.

Создать локальный конфиг:

```bash
cp .env.example .env
```

Запустить PostgreSQL/pgvector и Redis:

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

Публичная версия предназначена для оценки кода и архитектуры, а не для
воспроизведения приватной live-среды.

Исключено:

- `.env`, Telegram sessions, database dumps и local workspaces;
- chat logs, runtime artifacts, reports и benchmark evidence;
- private development history и operator-specific automation settings;
- production deployment credentials.

Подробнее: [docs/public-scope.md](docs/public-scope.md).

## Лицензия

Source-available for portfolio review. См. [LICENSE](LICENSE).

---

## English

Public portfolio snapshot of ZHVUSHA: a Telegram-based AI agent with persistent
memory, Agent Runtime, capability gates, isolated skills, durable jobs and CI
checks for architecture boundaries.

This repository is not a production export. It is a sanitized public review
artifact: source code, tests, migrations, configuration templates and
architecture notes are included; secrets, chat logs, runtime artifacts, personal
data and full private development history are intentionally excluded.

## Summary

ZHVUSHA demonstrates how to structure a personal-agent system without turning
tools into scattered prompt branches:

- user intent enters through the Telegram-facing orchestration layer;
- long-running or risky operations are represented as Agent Runtime jobs;
- each job receives a minimal capability profile;
- Tool Gateway verifies that workers stay within allowed actions;
- skills return structured results, while the main orchestrator owns the
  user-facing response and memory updates.

## Review Highlights

- Async Python 3.12 service with aiogram, Pydantic v2 and SQLAlchemy.
- LLM routing through tiers and a provider registry instead of one hardcoded
  model.
- PostgreSQL + pgvector for episodic memory, knowledge storage and retrieval.
- Redis-backed event/runtime patterns for background processes.
- Agent Runtime: durable jobs, invocation profiles, capability graph, workers,
  events, artifacts and structured results.
- Shared skill lifecycle: match, prepare, dry-run, approval, execute.
- Safety defaults: read-only by default; write/publish/send/login-like actions
  require explicit capability and approval paths.
- Contract tests, `ruff`, `mypy --strict`, `pytest` and `gitleaks` in CI.

## Technical Profile

| Area | Stack / approach |
| --- | --- |
| Bot interface | aiogram, Telegram handlers, delivery layer |
| LLM layer | provider registry, model tiers, OpenAI-compatible adapters |
| Memory | PostgreSQL, pgvector, episodic memory, staging, consolidation |
| Runtime | durable jobs, capability profiles, tool gateway, workers |
| Background work | Redis, daemon signals, event-driven loops |
| Quality gates | ruff, mypy strict, pytest, import-linter, gitleaks |

## Architecture

The core idea: tools and workers do not become independent assistants. They
return facts, artifacts, errors and proposals. The main orchestrator decides what
to tell the user, what to stage for memory and which next step is allowed.

See [docs/architecture.md](docs/architecture.md) for more detail.

## Run Locally

Prerequisites:

- Python 3.12;
- Docker with Compose;
- `uv` or a normal Python virtual environment.

Create local config:

```bash
cp .env.example .env
```

Start PostgreSQL/pgvector and Redis:

```bash
docker compose up -d
```

Install dependencies and run checks:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ --strict
uv run pytest -q --no-cov tests/agent_runtime tests/skills/test_invocation.py
```

To run the bot, real Telegram and LLM credentials must be provided in `.env`.
The public snapshot does not include live credentials, sessions, private
workspace data or production database dumps.
