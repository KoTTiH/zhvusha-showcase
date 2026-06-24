# Обзор Архитектуры

ZHVUSHA - personal-agent codebase вокруг одного центрального orchestration loop
и нескольких исполняющих слоев с ограниченными capabilities.

Ключевое решение: tools, skills и workers не становятся независимыми
ассистентами. Они возвращают structured observations, proposals, artifacts и
errors. Главный orchestrator отвечает за ответ пользователю, follow-up decisions
и memory updates.

English version is available below.

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

- `src/llm` - LLM gateway. Остальные модули идут через tiers и provider
  registry, а не hardcode-ят конкретные adapters.
- `src/memory` - episodic memory, consolidation и staging. Background processes
  предлагают memory candidates вместо прямого изменения private memory.
- `src/knowledge` - external knowledge storage и MCP-facing access layer.
- `src/agent_runtime` - durable jobs, profiles, capability declarations, worker
  routing и tool-gateway enforcement.
- `src/skills` - bounded user-facing capabilities. Skills проходят общий
  invocation lifecycle: match, prepare, dry-run, approval, execute.
- `src/bot` - Telegram-specific delivery, context handling и dispatcher
  decisions.

## Safety Model

Публичный срез сохраняет основной инженерный принцип private system: read-only
work по умолчанию, side effects только через explicit capability и approval
paths.

Side effects, которые должны оставаться gated:

- запись файлов;
- изменение environment values;
- отправка Telegram messages;
- публикация контента;
- login или submit browser forms;
- restart services;
- commit или push code.

Границы удерживаются тестами, `import-linter` contracts и CI.

## Testing Strategy

Самые полезные для ревью зоны тестов:

- agent runtime models и job lifecycle;
- capability graph и invocation profiles;
- skill invocation behavior;
- LLM provider routing;
- memory/consolidation staging;
- Telegram/bot mode separation;
- safety и approval gates.

Полная production-like verification требует реальные local services и
credentials, которые намеренно не включены в публичный срез.

---

## English

ZHVUSHA is a personal-agent codebase built around one central orchestration loop
and several execution layers with scoped capabilities.

The important design choice is that tools, skills and workers do not become
independent assistants. They return structured observations, proposals,
artifacts and errors. The main orchestrator owns user-facing synthesis,
follow-up decisions and memory updates.

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

- `src/llm` is the LLM gateway. Other modules route through tiers and the
  provider registry instead of hardcoding provider adapters.
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

The codebase uses tests, `import-linter` contracts and CI to keep these
boundaries visible.

## Testing Strategy

The most relevant tests for review are contract and boundary tests around:

- agent runtime models and job lifecycle;
- capability graph and invocation profiles;
- skill invocation behavior;
- LLM provider routing;
- memory/consolidation staging;
- Telegram/bot mode separation;
- safety and approval gates.

Full production-like verification requires real local services and credentials,
which are intentionally not included in this public snapshot.
