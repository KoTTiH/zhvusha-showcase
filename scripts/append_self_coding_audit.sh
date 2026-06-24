#!/bin/bash
# scripts/append_self_coding_audit.sh
#
# Called by Жвуша after each successful self-coding action.
# Appends an entry to the audit log for weekly review by Никита.
#
# Usage:
#   ./scripts/append_self_coding_audit.sh ACTION SPEC_FILE TIER TESTS_PASSED TOKENS COST
#
# Related KB: #69 (Trust hierarchy), #82 (enforcement).

set -e

ACTION="$1"
SPEC_FILE="$2"
TIER="$3"
TESTS_PASSED="$4"
TOKENS_USED="$5"
COST_USD="$6"

AUDIT_LOG="$HOME/zhvusha-workspace/audit/self_coding_$(date +%Y-%m).log"
mkdir -p "$(dirname "$AUDIT_LOG")"

cat >> "$AUDIT_LOG" <<EOF

## $(date -Iseconds)
- Action: $ACTION
- Spec: $SPEC_FILE
- Tier: $TIER
- Tests passed: $TESTS_PASSED
- Tokens used: $TOKENS_USED
- Cost: \$$COST_USD
- Commit: $(git rev-parse HEAD)
- Branch: $(git rev-parse --abbrev-ref HEAD)
EOF

echo "Audit appended to $AUDIT_LOG"
