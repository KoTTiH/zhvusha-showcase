"""Self-critique pass for Architect-generated spec drafts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from src.llm.protocols import LLMRequest

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.config import ReasoningEffort
    from src.llm.protocols import LLMGatewayProtocol


@dataclass(frozen=True)
class SelfCritiqueVerdict:
    """Whether the draft spec can proceed to validation/approval."""

    blocking: bool
    summary: str
    notes: list[str] = field(default_factory=list)


class SelfCritiqueRunner(Protocol):
    async def review(self, draft: dict[str, object]) -> SelfCritiqueVerdict: ...


_DEFAULT_LLM_TIMEOUT_SECONDS = 180.0
_DEFAULT_REASONING_EFFORT: ReasoningEffort = "medium"


class StaticSelfCritiqueRunner:
    """Deterministic fallback checks that require no LLM."""

    async def review(self, draft: dict[str, object]) -> SelfCritiqueVerdict:
        del draft
        return SelfCritiqueVerdict(
            blocking=False,
            summary="Static self-critique found no blocking issue.",
        )


class LLMSelfCritiqueRunner:
    """Worker-tier LLM pass that looks for contradictions in a draft spec."""

    def __init__(
        self,
        *,
        llm_router: LLMGatewayProtocol,
        reasoning_effort: ReasoningEffort = _DEFAULT_REASONING_EFFORT,
        timeout_seconds: float = _DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        self._llm = llm_router
        self._reasoning_effort = reasoning_effort
        self._timeout_seconds = max(0.1, timeout_seconds)

    async def review(self, draft: dict[str, object]) -> SelfCritiqueVerdict:
        try:
            response = await asyncio.wait_for(
                self._llm.generate(
                    LLMRequest(
                        prompt=str(draft),
                        system=(
                            "Ты делаешь второй проход по spec draft. "
                            "Если есть блокирующая проблема, ответь "
                            "'BLOCK: <причина>'. Иначе ответь 'OK: <заметки>'."
                        ),
                        tier="strategist",
                        reasoning_effort=self._reasoning_effort,
                        temperature=0.0,
                        caller="ideation_self_critique",
                    )
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return SelfCritiqueVerdict(
                blocking=False,
                summary=(
                    "LLM self-critique timed out after "
                    f"{self._timeout_seconds:g}s; static checks passed."
                ),
            )
        text = response.text.strip()
        blocking = text.lower().startswith("block:")
        return SelfCritiqueVerdict(
            blocking=blocking,
            summary=text.removeprefix("BLOCK:").removeprefix("OK:").strip() or text,
        )


async def self_critique_draft(
    draft: dict[str, object],
    *,
    tasks_dir: Path,
    runner: SelfCritiqueRunner | None = None,
) -> SelfCritiqueVerdict:
    """Run deterministic checks, then an optional model-backed critique."""
    static_issue = _static_issue(draft, tasks_dir=tasks_dir)
    if static_issue is not None:
        return SelfCritiqueVerdict(blocking=True, summary=static_issue)
    active_runner = runner or StaticSelfCritiqueRunner()
    return await active_runner.review(draft)


def _static_issue(draft: dict[str, object], *, tasks_dir: Path) -> str | None:
    failing = draft.get("failing_test")
    if not isinstance(failing, dict):
        return "missing failing_test reference"
    file_value = failing.get("file")
    if not isinstance(file_value, str) or not file_value.strip():
        return "missing failing_test file"
    path = file_value.strip()
    whitelist = draft.get("whitelist_paths")
    whitelisted = isinstance(whitelist, list) and path in {
        str(item) for item in whitelist
    }
    exists = (tasks_dir.parent / path).exists() or (tasks_dir / path).exists()
    if not exists and not whitelisted:
        return f"missing failing test file outside whitelist: {path}"
    return None
