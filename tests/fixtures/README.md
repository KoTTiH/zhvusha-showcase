# Shared test fixtures

Общие тестовые данные, используемые несколькими тестами.

## Правила (из KB #70)

1. Fixtures импортируются из этого места всеми тестами, которые их используют.
2. Изменение существующего fixture = отдельный коммит с явным объяснением «потому что контракт X изменился».
3. Fixture, используемый contract test модуля A и chain test модуля B, создаёт зависимость между тестами — это нормально и правильно.
4. При изменении контракта модуля chain tests могут начать падать — это не баг, это обнаружение несовместимости. Чини код/контракт, а не fixture.

## Структура

Каждый capability модуль получает свой файл fixtures при создании:

- `llm_fixtures.py` — после фазы 2 (LLM Gateway)
- `memory_fixtures.py` — после фазы 3 (Memory)
- `knowledge_fixtures.py` — после фазы 4 (Knowledge Base)
- `personality_fixtures.py` — после фазы 5 (Personality)
- `skills_fixtures.py` — после фазы 6 (Skills)
- `daemon_fixtures.py` — после фазы 7 (Daemon)
- `bot_fixtures.py` — после фазы 8 (Bot / Interfaces)

На фазе 1 здесь только этот README.
