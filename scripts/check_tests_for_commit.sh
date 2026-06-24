#!/bin/bash
# TDD gate: блокирует коммит если для staged src/ файлов нет тестов
# Используется как Codex/git pre-commit safety hook на git_commit

staged_src=$(git diff --cached --name-only --diff-filter=AM 2>/dev/null \
    | grep '^src/.*\.py$' \
    | grep -v '__init__\.py$' \
    | grep -v '__main__\.py$' \
    | grep -v '^src/scripts/')

if [ -z "$staged_src" ]; then
    exit 0
fi

missing=()
for f in $staged_src; do
    # src/skills/delegate/skill.py → src.skills.delegate
    module=$(echo "$f" | sed 's|/|.|g' | sed 's|\.py$||' | sed 's|\.[^.]*$||')
    basename=$(basename "$f" .py)

    # Ищем любой тест который импортирует этот модуль или тестирует этот файл
    if ! grep -rl --include='*.py' "$module\|test_${basename}" tests/ >/dev/null 2>&1; then
        missing+=("$f")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "BLOCKED: TDD violation — нет тестов для:"
    for f in "${missing[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "Напиши падающие тесты ДО реализации (AGENTS.md → TDD Workflow)"
    exit 2
fi
