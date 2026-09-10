"""Composition guard for budget-policy production wiring."""

from __future__ import annotations

import ast
from pathlib import Path


def test_bot_composition_does_not_wire_usd_budget_gate_by_default() -> None:
    source = Path("src/bot/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    agent_runtime_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AgentRuntime"
    ]

    assert agent_runtime_calls
    for call in agent_runtime_calls:
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert "budget_preflight_gate" not in keyword_names
        assert "budget_usage_recorder" not in keyword_names
