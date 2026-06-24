"""Self-critique guard for Architect specs."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock


def _draft() -> dict[str, object]:
    return {
        "slug": "missing-test",
        "title": "Missing test",
        "created_at": datetime(2026, 5, 7, tzinfo=UTC).isoformat(),
        "created_by": "zhvusha",
        "tier": 1,
        "goal": "Добавить изменение с проверяемым планом и явной причиной.",
        "failing_test": {
            "file": "tests/missing/test_contract.py",
            "name": "test_new_contract",
            "spec": "Must pass.",
        },
        "whitelist_paths": ["src/example.py"],
        "blast_radius": ["small"],
        "rollback_path": ["git revert"],
        "rationale": "Local evidence.",
        "source_provenance": [
            {
                "url": "local:test",
                "source_type": "local_repo",
                "trust_tier": "direct",
                "claim": "Test fixture.",
            }
        ],
    }


async def test_self_critique_blocks_missing_failing_test_reference(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.skills.ideation_to_spec.self_critique import (
        LLMSelfCritiqueRunner,
        self_critique_draft,
    )

    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=type("Resp", (), {"text": "BLOCK: missing failing test file"})()
    )
    runner = LLMSelfCritiqueRunner(llm_router=llm)

    verdict = await self_critique_draft(_draft(), tasks_dir=tmp_path, runner=runner)

    assert verdict.blocking
    assert "missing failing test" in verdict.summary


async def test_self_critique_accepts_existing_or_new_whitelisted_test(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.skills.ideation_to_spec.self_critique import (
        StaticSelfCritiqueRunner,
        self_critique_draft,
    )

    draft = _draft()
    draft["whitelist_paths"] = ["tests/missing/test_contract.py", "src/example.py"]

    verdict = await self_critique_draft(
        draft, tasks_dir=tmp_path, runner=StaticSelfCritiqueRunner()
    )

    assert not verdict.blocking
    assert verdict.summary


async def test_llm_self_critique_uses_strategist_medium(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.skills.ideation_to_spec.self_critique import (
        LLMSelfCritiqueRunner,
        self_critique_draft,
    )

    draft = _draft()
    draft["whitelist_paths"] = ["tests/missing/test_contract.py", "src/example.py"]
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=type("Resp", (), {"text": "OK: clean"})())

    await self_critique_draft(
        draft,
        tasks_dir=tmp_path,
        runner=LLMSelfCritiqueRunner(llm_router=llm),
    )

    request = llm.generate.await_args.args[0]
    assert request.tier == "strategist"
    assert request.reasoning_effort == "medium"


async def test_llm_self_critique_timeout_falls_back_to_static_pass(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.skills.ideation_to_spec.self_critique import (
        LLMSelfCritiqueRunner,
        self_critique_draft,
    )

    draft = _draft()
    draft["whitelist_paths"] = ["tests/missing/test_contract.py", "src/example.py"]

    async def slow_generate(_request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(1)
        return type("Resp", (), {"text": "BLOCK: too late"})()

    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=slow_generate)

    verdict = await self_critique_draft(
        draft,
        tasks_dir=tmp_path,
        runner=LLMSelfCritiqueRunner(llm_router=llm, timeout_seconds=0.01),
    )

    assert not verdict.blocking
    assert "timed out" in verdict.summary
