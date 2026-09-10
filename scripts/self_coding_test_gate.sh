#!/bin/bash
# scripts/self_coding_test_gate.sh
#
# Run before any merge from Жвуша's самокодинг branch.
# Implements monotonic progress requirement from KB #69:
# all previously passing tests must still pass.
#
# In phase 1 this gate is informational (not a hard blocker). It becomes
# blocking starting phase 2.

set -e

echo "=== Self-coding test gate ==="

echo "1/4: Lint check..."
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

echo "2/4: Type check..."
uv run mypy src/ --strict

echo "3/4: Import isolation..."
uv run lint-imports --config .importlinter

echo "4/4: Contract + chain tests..."
# Subset run — do not enforce --cov-fail-under (it applies to the whole
# codebase, not the selected subset). Full coverage is enforced by the
# default `pytest` invocation run separately.
uv run pytest -m "contract or chain" -x --tb=short --no-cov

echo ""
echo "✅ All gates passed. Safe to merge."
