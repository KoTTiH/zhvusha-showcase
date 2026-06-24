#!/bin/bash
# scripts/check_tier3_protection.sh
#
# Prevents commits to Tier 3 protected paths from non-human authors.
# A commit is considered "human" if:
#   - committed via git commit -m by Никита (manual)
#   - OR has explicit override marker [tier3-override] in commit message
#
# Жвуша's самокодинг workflow uses git commit with author "zhvusha-*",
# which this script blocks for Tier 3 paths.
#
# Related KB: #69 (Trust hierarchy), #82 (enforcement).

set -e

# Files that are Tier 3 — only Никита can modify
TIER3_PATHS=(
    "src/skills/base.py"
    "src/skills/__init__.py"
    "src/skills/registry.py"
    "src/personality/decision.py"
    "src/safety/"
    "src/*/protocols.py"
    "src/llm/protocols.py"
    "pillars.md"
    "personality/pillars.md"
    "zhvusha-workspace/personality/pillars.md"
    ".importlinter"
    "scripts/check_tier3_protection.sh"
    "AGENTS.md"
    "CLAUDE.md"
)

# Check if commit author is Жвуша
AUTHOR_EMAIL=$(git config user.email 2>/dev/null || echo "")
AUTHOR_NAME=$(git config user.name 2>/dev/null || echo "")

is_zhvusha_commit() {
    if [[ "$AUTHOR_EMAIL" == *"zhvusha"* ]] || [[ "$AUTHOR_NAME" == *"zhvusha"* ]]; then
        return 0
    fi
    return 1
}

# Check for override marker in latest commit message
has_override() {
    local msg
    msg=$(git log -1 --pretty=%B 2>/dev/null || echo "")
    if [[ "$msg" == *"[tier3-override]"* ]]; then
        return 0
    fi
    return 1
}

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only 2>/dev/null || echo "")

if [[ -z "$STAGED_FILES" ]]; then
    exit 0
fi

# If commit is from Жвуша, check for Tier 3 paths
if is_zhvusha_commit && ! has_override; then
    for path_pattern in "${TIER3_PATHS[@]}"; do
        for file in $STAGED_FILES; do
            # shellcheck disable=SC2053
            if [[ "$file" == $path_pattern ]] || [[ "$file" == ${path_pattern}* ]]; then
                echo "❌ TIER 3 PROTECTION: Жвуша cannot commit to $file"
                echo "   This is a Tier 3 architectural file (KB #69)."
                echo "   Only Никита can modify it."
                echo "   To override, add [tier3-override] to commit message and re-commit."
                exit 1
            fi
        done
    done
fi

exit 0
