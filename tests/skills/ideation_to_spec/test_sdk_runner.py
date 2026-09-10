"""Contract tests for ``ideation_to_spec.sdk_runner`` Codex wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract
_PROJECT_ROOT = Path("project-root")


class _FakeCodexBackend:
    def __init__(self, *, codex_path: str = "codex", model: str = "") -> None:
        self.codex_path = codex_path
        self.model = model

    async def run_architect(self, request: Any) -> Any:
        from src.skills.code_agent.protocols import CodeAgentResult

        self.request = request
        return CodeAgentResult(text="```yaml\nslug: x\n```", backend="codex_cli")


class TestArchitectRunner:
    async def test_passes_codex_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.skills.ideation_to_spec import sdk_runner

        created: list[_FakeCodexBackend] = []

        def factory(*, codex_path: str = "codex", model: str = "") -> _FakeCodexBackend:
            backend = _FakeCodexBackend(codex_path=codex_path, model=model)
            created.append(backend)
            return backend

        monkeypatch.setattr(sdk_runner, "CodexCliBackend", factory)

        out = await sdk_runner.run_architect_sdk(
            user_prompt="draft spec",
            system_prompt="read-only rules",
            cwd=_PROJECT_ROOT,
            model="gpt-test",
            codex_path="codex-test",
        )

        assert "slug: x" in out
        backend = created[0]
        assert backend.codex_path == "codex-test"
        assert backend.model == "gpt-test"
        assert backend.request.user_prompt == "draft spec"
        assert backend.request.system_prompt == "read-only rules"
        assert backend.request.cwd == _PROJECT_ROOT
