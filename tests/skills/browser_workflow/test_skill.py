"""Browser workflow draft skill contracts."""

from __future__ import annotations

from typing import Any

from src.skills.base import AgentContext


def _context() -> AgentContext:
    return AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={"source": "vscode", "interface": "vscode"},
    )


async def test_browser_workflow_skill_routes_explicit_command() -> None:
    from src.skills.browser_workflow.skill import BrowserWorkflowDraftSkill

    class Runtime:
        pass

    skill = BrowserWorkflowDraftSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    assert (
        await skill.can_handle(
            '/browser_workflow_draft {"read_url":"https://example.com/form"}',
            _context(),
        )
        == 0.94
    )
    assert await skill.can_handle("подготовь браузерный черновик", _context()) == 0.0


async def test_browser_workflow_skill_runs_agent_runtime_job() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.browser_workflow.skill import BrowserWorkflowDraftSkill

    class Runtime:
        def __init__(self) -> None:
            self.created: list[AgentJob] = []

        async def create_job(self, **kwargs: Any) -> AgentJob:
            job = AgentJob.new(**kwargs)
            self.created.append(job)
            return job

        async def start(self, job_id: str) -> AgentJob:
            job = self.created[0]
            assert job.id == job_id
            return job.with_status(
                AgentJobStatus.DONE,
                result=ContextCapsule(
                    summary="Подготовлен browser form draft без отправки формы.",
                    processed_context=(
                        "Read URL: https://example.com/forms/post\n"
                        "Submit boundary: browser_submit was not used.\n\n"
                        "## Source excerpt\n"
                        '<form><input name="email"></form>'
                    ),
                    findings=(),
                    sources=("https://example.com/forms/post",),
                    artifacts=("agent_runtime/browser_artifacts/form-draft-test.json",),
                    next_actions=("internal handoff",),
                ),
            )

    runtime = Runtime()
    skill = BrowserWorkflowDraftSkill(admin_user_id=1291112109, runtime=runtime)
    message = (
        '/browser_workflow_draft {"read_url":"https://example.com/forms/post",'
        '"fields":{"email":"stage-l@example.invalid"}}'
    )

    result = await skill.execute(message, _context())

    assert result.success is True
    assert result.response == ""
    assert result.metadata["requires_zhvusha_response"] is True
    observation = result.metadata["body_observation"]
    assert observation["event"] == "browser_workflow_draft_completed"
    assert observation["source"] == "browser_workflow_draft"
    assert observation["sources"] == ["https://example.com/forms/post"]
    assert observation["artifacts"] == [
        "agent_runtime/browser_artifacts/form-draft-test.json"
    ]
    assert (
        "Submit boundary: browser_submit was not used."
        in observation["processed_context"]
    )
    assert "Source excerpt" not in observation["processed_context"]
    assert "<form>" not in observation["processed_context"]
    assert "internal handoff" not in json_dump(observation)
    assert runtime.created[0].kind == "browser_workflow_draft"
    assert runtime.created[0].profile.id == "browser_workflow.draft"
    assert "do_not_submit_forms" in runtime.created[0].context_pack.constraints


async def test_browser_workflow_skill_reports_missing_payload() -> None:
    from src.skills.browser_workflow.skill import BrowserWorkflowDraftSkill

    class Runtime:
        pass

    skill = BrowserWorkflowDraftSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    result = await skill.execute("/browser_workflow_draft", _context())

    assert result.success is False
    assert result.response == ""
    assert result.metadata["requires_zhvusha_response"] is True
    assert result.metadata["body_observation"]["event"] == "missing_required_input"


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)
