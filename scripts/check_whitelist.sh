#!/bin/bash
# scripts/check_whitelist.sh
#
# Pre-commit gate: when on a ``zhvusha/<slug>`` branch with a
# zhvusha-flavoured git author, refuse to commit any staged file that
# isn't on the spec's ``whitelist_paths`` (read from
# ``tasks/*-<slug>.yaml``).
#
# For every other configuration (different author, different branch
# pattern, no staged files, ``WHITELIST_OVERRIDE`` env set) the script is
# a no-op and exits 0 — Никита's manual commits never trip on this.
#
# Pairs with ``check_tier3_protection.sh`` (which gates Tier-3 paths) and
# the in-skill PreToolUse hook from
# ``src/skills/implement_spec/hooks.py`` (same logic, different
# enforcement point — defence in depth).
#
# Related KB: #69 (Trust hierarchy), #82 (enforcement).

set -e

AUTHOR_NAME=$(git config user.name 2>/dev/null || echo "")
AUTHOR_EMAIL=$(git config user.email 2>/dev/null || echo "")

is_zhvusha_commit() {
    if [[ "$AUTHOR_EMAIL" == *"zhvusha"* ]] || [[ "$AUTHOR_NAME" == *"zhvusha"* ]]; then
        return 0
    fi
    return 1
}

if ! is_zhvusha_commit; then
    exit 0
fi

if [[ -n "${WHITELIST_OVERRIDE:-}" ]]; then
    echo "ℹ️  WHITELIST CHECK: WHITELIST_OVERRIDE='$WHITELIST_OVERRIDE' — skipped"
    exit 0
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [[ "$BRANCH" != zhvusha/* ]]; then
    # zhvusha author but on some other branch (e.g. main) — leave the
    # tier3 hook to handle it; whitelist is per-spec, no spec, no check.
    exit 0
fi

SLUG="${BRANCH#zhvusha/}"

# Find the spec file. Match either ``tasks/<date>-<slug>.yaml`` or
# ``tasks/<slug>.yaml`` (no date prefix), preferring the dated form.
SPEC=""
for candidate in tasks/*-"${SLUG}".yaml "tasks/${SLUG}.yaml"; do
    if [[ -f "$candidate" ]]; then
        SPEC="$candidate"
        break
    fi
done

if [[ -z "$SPEC" ]]; then
    echo "❌ WHITELIST CHECK: no tasks/*-${SLUG}.yaml for branch ${BRANCH}" >&2
    exit 1
fi

# Pull the whitelist out via Python (PyYAML is in the project venv).
# Resolution order:
#   1. $VIRTUAL_ENV/bin/python3 — works inside `uv run pytest` and any
#      shell where the venv is sourced.
#   2. uv run python3 — works in repo with pyproject.toml on disk.
#   3. bare python3 — fallback; only works if PyYAML is system-wide.
if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "$VIRTUAL_ENV/bin/python3" ]]; then
    PY_BIN="$VIRTUAL_ENV/bin/python3"
elif command -v uv >/dev/null 2>&1; then
    PY_BIN="uv run python3"
else
    PY_BIN="python3"
fi
WHITELIST=$($PY_BIN -c "
import yaml
with open('$SPEC', encoding='utf-8') as f:
    data = yaml.safe_load(f)
for p in (data.get('whitelist_paths') or []):
    print(p)
# Phase 16: legitimate-test-mutation channel — existing_tests_to_update
# entries are paths the spec has explicitly declared as touchable. Join
# them with whitelist_paths for the duration of the gate. The runtime
# PreToolUse hook and CommitRunner already do the same union.
for entry in (data.get('existing_tests_to_update') or []):
    if isinstance(entry, dict) and entry.get('path'):
        print(entry['path'])
" 2>/dev/null || echo "")

if [[ -z "$WHITELIST" ]]; then
    echo "❌ WHITELIST CHECK: empty/unreadable whitelist_paths in $SPEC" >&2
    exit 1
fi

STAGED=$(git diff --cached --name-only 2>/dev/null || echo "")
if [[ -z "$STAGED" ]]; then
    exit 0
fi

EXTRAS=()
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    if ! echo "$WHITELIST" | grep -Fxq "$file"; then
        EXTRAS+=("$file")
    fi
done <<< "$STAGED"

if [[ ${#EXTRAS[@]} -gt 0 ]]; then
    {
        echo "❌ WHITELIST CHECK: staged files outside whitelist for ${BRANCH}:"
        for f in "${EXTRAS[@]}"; do
            echo "    $f"
        done
        echo "   Spec: $SPEC"
        echo "   Override: WHITELIST_OVERRIDE='<reason>' git commit ..."
    } >&2
    exit 1
fi

exit 0
