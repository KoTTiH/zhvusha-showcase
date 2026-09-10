from __future__ import annotations

import json
from pathlib import Path

from scripts.codex_architecture_gate import (
    commands_for_turn,
    evaluate_gate,
    touched_paths_for_turn,
)


def test_gate_allows_documentation_only_changes() -> None:
    decision = evaluate_gate(
        touched_paths={"docs/architecture-invariants.md", "src/skills/AGENTS.md"},
        commands=[],
    )

    assert decision.should_continue is True


def test_gate_blocks_architecture_source_change_without_behavior_check() -> None:
    decision = evaluate_gate(
        touched_paths={"src/skills/web_research/skill.py"},
        commands=["uv run ruff check src/skills/web_research/skill.py"],
    )

    assert decision.should_continue is False
    assert "no pytest" in decision.reason
    assert "scenario-specific" not in decision.reason


def test_gate_allows_architecture_source_change_with_pytest() -> None:
    decision = evaluate_gate(
        touched_paths={"src/bot/main.py"},
        commands=["uv run pytest -q --no-cov tests/bot/test_vscode_chat_pipeline.py"],
    )

    assert decision.should_continue is True


def test_gate_reads_current_turn_patch_and_commands(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "payload": {
                            "type": "patch_apply_end",
                            "turn_id": "turn-1",
                            "changes": {
                                str(tmp_path / "src" / "bot" / "main.py"): {
                                    "type": "update"
                                }
                            },
                        }
                    }
                ),
                json.dumps(
                    {
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "turn_id": "turn-1",
                            "arguments": json.dumps(
                                {
                                    "cmd": (
                                        "uv run pytest -q --no-cov "
                                        "tests/bot/test_vscode_chat_pipeline.py"
                                    )
                                }
                            ),
                        }
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert touched_paths_for_turn(transcript, "turn-1", tmp_path) == {"src/bot/main.py"}
    assert commands_for_turn(transcript, "turn-1") == [
        "uv run pytest -q --no-cov tests/bot/test_vscode_chat_pipeline.py"
    ]
