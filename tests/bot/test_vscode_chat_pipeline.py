from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from src.skills.base import AgentContext, InlineSkill, SkillResult

if TYPE_CHECKING:
    from pathlib import Path


def _write_chat_log(root: Path, chat_id: str, entries: list[dict[str, object]]) -> None:
    chat_dir = root / "logs" / chat_id
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    (chat_dir / f"chat_{today}.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries),
        encoding="utf-8",
    )


def test_vscode_chat_context_uses_separate_log_id_without_codex_transport_history(
    tmp_path: Path,
) -> None:
    from src.bot.main import _with_dialogue_context_metadata

    _write_chat_log(
        tmp_path,
        "vscode",
        [
            {"role": "user", "source": "vscode", "text": "сообщение из VS Code"},
            {
                "role": "user",
                "source": "vscode",
                "source_actor": "codex",
                "codex": True,
                "text": "сообщение от Codex",
            },
        ],
    )
    context = AgentContext(
        user_id=12345,
        chat_id=-7331,
        mode="personal",
        metadata={"chat_log_id": "vscode", "interface": "vscode"},
    )

    updated = _with_dialogue_context_metadata(
        "ответь",
        context,
        workspace_root=tmp_path,
    )

    assert "сообщение из VS Code" in updated.metadata["recent_messages"]
    assert "Codex: сообщение от Codex" not in updated.metadata["recent_messages"]
    assert (tmp_path / "logs" / "-7331").exists() is False


def test_vscode_chat_context_keeps_codex_transport_out_of_human_history(
    tmp_path: Path,
) -> None:
    from src.bot.main import _with_dialogue_context_metadata

    _write_chat_log(
        tmp_path,
        "vscode",
        [
            {
                "role": "user",
                "source": "vscode",
                "source_actor": "codex",
                "text": "проверяет bridge",
            },
            {
                "role": "assistant",
                "source": "vscode",
                "text": "Я в отдельном VS Code-чате.",
            },
            {
                "role": "user",
                "source": "vscode",
                "source_actor": "user",
                "text": "обычное человеческое сообщение",
            },
        ],
    )
    context = AgentContext(
        user_id=12345,
        chat_id=-7331,
        mode="personal",
        metadata={
            "chat_log_id": "vscode",
            "interface": "vscode",
            "source_actor": "user",
        },
    )

    updated = _with_dialogue_context_metadata(
        "че как ты",
        context,
        workspace_root=tmp_path,
    )

    recent_messages = updated.metadata["recent_messages"]
    assert "обычное человеческое сообщение" in recent_messages
    assert "Codex: проверяет bridge" not in recent_messages
    assert "Я в отдельном VS Code-чате." not in recent_messages


async def test_process_text_message_returns_skill_response_for_non_telegram(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main

    class FakeOutcome:
        handled = True
        result = SkillResult(
            success=True,
            response="ответ для VS Code",
            metadata={"skill_name": "fake"},
        )

    class FakeInvocationService:
        async def dispatch(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
        ) -> FakeOutcome:
            return FakeOutcome()

    async def no_control(text: str, context: AgentContext) -> None:
        return None

    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", FakeInvocationService())
    monkeypatch.setattr(bot_main, "_skills", [])

    response = await bot_main._process_text_message(
        "пинг",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            metadata={"return_response_text": True},
        ),
    )

    assert response == "ответ для VS Code"


def test_chat_context_budget_preselection_does_not_force_chat_only_for_slash_commands(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main

    def focused_budget(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(route="focused", reason="large_history")

    monkeypatch.setattr(bot_main, "classify_context_budget", focused_budget)

    context = AgentContext(
        user_id=12345,
        chat_id=-7331,
        mode="personal",
        metadata={"interface": "vscode"},
    )

    updated = bot_main._with_chat_context_budget_preselection_metadata(
        "/external_skill_search browser_read | docs | local_folder",
        context,
    )

    assert updated is context
    assert "prefer_chat_response_only" not in updated.metadata
    assert "chat_context_budget" not in updated.metadata


async def test_process_text_message_synthesizes_body_observation_instead_of_raw_skill_response(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main

    class FakeOutcome:
        def __init__(self, result: SkillResult) -> None:
            self.handled = True
            self.result = result

    class FakeInvocationService:
        def __init__(self) -> None:
            self.synthesis_metadata: list[dict[str, Any]] = []

        async def dispatch(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
        ) -> FakeOutcome:
            del text, context, skills
            return FakeOutcome(
                SkillResult(
                    success=True,
                    response=(
                        "RAW CAPSULE\n\nДальше:\n- Передать прочитанный контекст Жвуше"
                    ),
                    metadata={
                        "skill_name": "web_research",
                        "requires_zhvusha_response": True,
                        "body_observation": {
                            "event": "web_research_completed",
                            "query": "Python 3.14 release notes",
                            "sources": ["https://docs.python.org/3/whatsnew/3.14.html"],
                        },
                    },
                )
            )

        async def invoke_named_skill(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
            skill_name: str,
        ) -> FakeOutcome:
            del text, skills
            assert skill_name == "chat_response"
            self.synthesis_metadata.append(context.metadata)
            return FakeOutcome(
                SkillResult(
                    success=True,
                    response=(
                        "Python 3.14: краткий ответ Жвуши со ссылкой "
                        "https://docs.python.org/3/whatsnew/3.14.html"
                    ),
                    metadata={"skill_name": "chat_response"},
                )
            )

    async def no_control(text: str, context: AgentContext) -> None:
        del text, context
        return None

    invocation_service = FakeInvocationService()
    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(
        bot_main,
        "_with_chat_context_budget_preselection_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", invocation_service)
    monkeypatch.setattr(bot_main, "_skills", [])

    response = await bot_main._process_text_message(
        "Жвуша, найди источники про Python 3.14 release notes и дай ответ",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            metadata={"return_response_text": True},
        ),
    )

    assert response is not None
    assert response.startswith("Python 3.14")
    assert "RAW CAPSULE" not in response
    assert "Передать прочитанный контекст" not in response
    assert invocation_service.synthesis_metadata
    body_observation = invocation_service.synthesis_metadata[0]["body_observation"]
    assert "web_research_completed" in body_observation
    assert "https://docs.python.org/3/whatsnew/3.14.html" in body_observation


async def test_process_text_message_routes_digital_scenario_without_slash(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    import src.bot.main as bot_main
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.core.mode_config import is_skill_allowed
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill
    from src.skills.digital_scenario.skill import (
        DigitalScenarioIntent,
        DigitalScenarioSkill,
    )
    from src.skills.invocation import InMemorySkillApprovalStore, SkillInvocationService

    class FakeClassifier:
        def __init__(self) -> None:
            self.calls = 0

        async def classify(
            self,
            message: str,
            context: AgentContext,
        ) -> DigitalScenarioIntent:
            del context
            self.calls += 1
            assert not message.strip().startswith("/")
            return DigitalScenarioIntent(
                scenario_id="ai_cto_projects",
                confidence=0.91,
                rationale="broad CTO/project audit request",
            )

    class FakeChatResponseSkill(InlineSkill):
        name = "chat_response"
        description = "Fake chat response"
        llm_tier = "worker"
        approval_policy = "auto"

        async def can_handle(self, message: str, context: AgentContext) -> float:
            del message, context
            return 0.3

        async def execute(self, message: str, context: AgentContext) -> SkillResult:
            assert message == (
                "Жвуша, проверь репу ZHVUSHA как CTO и найди архитектурный долг"
            )
            body_observation = str(context.metadata.get("body_observation", ""))
            assert "digital_scenario_intent_detected" in body_observation
            assert "ai_cto_projects" in body_observation
            assert "natural_language_user_flow" in body_observation
            assert '"ready_for_live_matrix": true' in body_observation
            assert "dispatcher/code health checked" in body_observation
            assert "next_actions" not in body_observation
            return SkillResult(
                success=True,
                response="Приняла как AI-CTO сценарий без slash-команды.",
                metadata={"skill_name": "chat_response"},
            )

    async def explorer_runner(**_: Any) -> str:
        raise AssertionError("digital_scenario must outrank codebase fallback")

    async def action_runner(action: Any, context: AgentContext) -> SkillResult:
        del context
        assert action.skill_name == "codebase_explorer"
        assert not str(action.message).startswith("/")
        return SkillResult(
            success=True,
            response="dispatcher/code health checked",
            metadata={"skill_name": "codebase_explorer"},
        )

    async def classify(_: str) -> str:
        return "ambiguous"

    async def no_control(text: str, context: AgentContext) -> None:
        del text, context
        return None

    scenario = next(
        item for item in BUILTIN_DIGITAL_SCENARIOS if item.id == "ai_cto_projects"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.ai_cto_projects",
                label="AI-CTO для проектов",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.SKILL,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )
    classifier = FakeClassifier()
    invocation_service = SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=classify,  # type: ignore[arg-type]
        is_skill_allowed=lambda skill_name, mode: is_skill_allowed(skill_name, mode),
    )
    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(
        bot_main,
        "_with_chat_context_budget_preselection_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", invocation_service)
    monkeypatch.setattr(
        bot_main,
        "_skills",
        [
            CodebaseExplorerSkill(
                admin_user_id=12345,
                workspace_root=tmp_path,
                explorer_runner=explorer_runner,
            ),
            DigitalScenarioSkill(
                admin_user_id=12345,
                intent_classifier=classifier,
                capability_graph_provider=lambda: graph,
                action_runner=action_runner,
            ),
            FakeChatResponseSkill(),
        ],
    )

    response = await bot_main._process_text_message(
        "Жвуша, проверь репу ZHVUSHA как CTO и найди архитектурный долг",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            message_id=777,
            metadata={
                "return_response_text": True,
                "source": "vscode",
                "interface": "vscode",
                "source_actor": "codex",
                "digital_scenario_eval_variant": "happy_path",
                "digital_scenario_eval_run_id": "run-42",
            },
        ),
    )

    assert response == "Приняла как AI-CTO сценарий без slash-команды."
    assert classifier.calls == 1
    evidence_path = tmp_path / "runtime" / "digital_scenarios" / "live_evidence.jsonl"
    assert evidence_path.is_file()
    evidence = evidence_path.read_text(encoding="utf-8")
    assert "ai_cto_projects" in evidence
    assert '"variant": "happy_path"' in evidence
    assert "skill=codebase_explorer" in evidence
    assert "eval_run_id=run-42" in evidence


async def test_process_text_message_preserves_body_artifacts_for_final_delivery(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main

    class FakeOutcome:
        def __init__(self, result: SkillResult) -> None:
            self.handled = True
            self.result = result

    class FakeInvocationService:
        async def dispatch(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
        ) -> FakeOutcome:
            del text, context, skills
            return FakeOutcome(
                SkillResult(
                    success=True,
                    response="",
                    metadata={
                        "skill_name": "web_research",
                        "requires_zhvusha_response": True,
                        "artifacts": (
                            "agent_runtime/browser_artifacts/screenshot-page.png",
                        ),
                        "deliver_artifacts_to_chat": True,
                        "sources": ("https://example.com/page",),
                        "agent_job_id": "job-web",
                        "agent_profile": "web_research.readonly",
                        "body_observation": {
                            "event": "web_research_completed",
                            "sources": ["https://example.com/page"],
                            "artifacts": [
                                "agent_runtime/browser_artifacts/screenshot-page.png"
                            ],
                        },
                    },
                )
            )

        async def invoke_named_skill(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
            skill_name: str,
        ) -> FakeOutcome:
            del text, context, skills
            assert skill_name == "chat_response"
            return FakeOutcome(
                SkillResult(
                    success=True,
                    response="Готово, скриншот приложила.",
                    metadata={"skill_name": "chat_response"},
                )
            )

    emitted: list[SkillResult] = []

    async def capture_emit(result: SkillResult, context: AgentContext) -> str | None:
        del context
        emitted.append(result)
        return result.response

    async def no_control(text: str, context: AgentContext) -> None:
        del text, context
        return None

    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(
        bot_main,
        "_with_chat_context_budget_preselection_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", FakeInvocationService())
    monkeypatch.setattr(bot_main, "_emit_skill_response", capture_emit)
    monkeypatch.setattr(bot_main, "_skills", [])

    response = await bot_main._process_text_message(
        "Открой в интернете статью, сделай скриншот и пришли мне",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            metadata={"return_response_text": True},
        ),
    )

    assert response == "Готово, скриншот приложила."
    assert emitted
    assert emitted[0].metadata["artifacts"] == (
        "agent_runtime/browser_artifacts/screenshot-page.png",
    )
    assert emitted[0].metadata["deliver_artifacts_to_chat"] is True
    assert emitted[0].metadata["sources"] == ("https://example.com/page",)
    assert emitted[0].metadata["agent_job_id"] == "job-web"


async def test_process_text_message_routes_explicit_url_browser_request_to_web_research(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        Finding,
        FindingStatus,
    )
    from src.core.mode_config import is_skill_allowed
    from src.skills.invocation import InMemorySkillApprovalStore, SkillInvocationService
    from src.skills.web_research.skill import WebResearchSkill

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
                    summary="Прочитала 1 источник read-only.",
                    processed_context="Example Domain",
                    findings=(
                        Finding(
                            claim=(
                                "Источник прочитан через browser_read_url: "
                                "https://example.com/"
                            ),
                            status=FindingStatus.CONFIRMED,
                            confidence=0.9,
                            evidence=("https://example.com/",),
                        ),
                    ),
                    sources=("https://example.com/",),
                    artifacts=(
                        "agent_runtime/browser_artifacts/screenshot-example.png",
                    ),
                ),
            )

    class FakeChatResponseSkill(InlineSkill):
        name = "chat_response"
        description = "Fake chat response"
        llm_tier = "worker"
        approval_policy = "auto"

        async def can_handle(self, message: str, context: AgentContext) -> float:
            del message, context
            return 0.3

        async def execute(self, message: str, context: AgentContext) -> SkillResult:
            del message
            body_observation = str(context.metadata.get("body_observation", ""))
            assert "web_research_completed" in body_observation
            assert "https://example.com/" in body_observation
            return SkillResult(
                success=True,
                response=(
                    "Вижу Example Domain. "
                    "Артефакт: `agent_runtime/browser_artifacts/screenshot-example.png`"
                ),
                metadata={"skill_name": "chat_response"},
            )

    async def classify(_: str) -> str:
        return "ambiguous"

    async def no_control(text: str, context: AgentContext) -> None:
        del text, context
        return None

    runtime = Runtime()
    invocation_service = SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=classify,  # type: ignore[arg-type]
        is_skill_allowed=lambda skill_name, mode: is_skill_allowed(skill_name, mode),
    )
    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(
        bot_main,
        "_with_chat_context_budget_preselection_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", invocation_service)
    monkeypatch.setattr(
        bot_main,
        "_skills",
        [
            WebResearchSkill(admin_user_id=12345, runtime=runtime),
            FakeChatResponseSkill(),
        ],
    )

    response = await bot_main._process_text_message(
        "Browser-use smoke от Codex/operator: открой https://example.com/ "
        "через браузерные read-only tools, сделай скриншот и ответь только "
        "в этот VS Code chat.",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            metadata={
                "return_response_text": True,
                "source": "vscode",
                "interface": "vscode",
            },
        ),
    )

    assert response is not None
    assert "Example Domain" in response
    assert runtime.created
    assert runtime.created[0].kind == "web_research"
    assert runtime.created[0].context_pack.user_request == "https://example.com/"


async def test_process_text_message_routes_public_profile_screenshot_to_web_research(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.core.mode_config import is_skill_allowed
    from src.skills.invocation import InMemorySkillApprovalStore, SkillInvocationService
    from src.skills.web_research.skill import WebResearchSkill

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
                    summary="Открыла Dotabuff search page.",
                    processed_context="Dotabuff search for kereexa",
                    sources=(
                        "https://www.dotabuff.com/search?utf8=%E2%9C%93&q=kereexa",
                    ),
                    artifacts=(
                        "agent_runtime/browser_artifacts/screenshot-dotabuff.png",
                    ),
                ),
            )

    class FakeExternalSkillAcquisition(InlineSkill):
        name = "external_skill_acquisition"
        description = "Fake external skill acquisition"
        llm_tier = "worker"
        approval_policy = "required"

        async def can_handle(self, message: str, context: AgentContext) -> float:
            del message, context
            return 0.92

        async def execute(self, message: str, context: AgentContext) -> SkillResult:
            del message, context
            raise AssertionError("external skill acquisition must not run")

    class FakeChatResponseSkill(InlineSkill):
        name = "chat_response"
        description = "Fake chat response"
        llm_tier = "worker"
        approval_policy = "auto"

        async def can_handle(self, message: str, context: AgentContext) -> float:
            del message, context
            return 0.3

        async def execute(self, message: str, context: AgentContext) -> SkillResult:
            del message
            body_observation = str(context.metadata.get("body_observation", ""))
            assert "web_research_completed" in body_observation
            assert "dotabuff.com/search" in body_observation
            return SkillResult(
                success=True,
                response=(
                    "Открыла Dotabuff. "
                    "Артефакт: `agent_runtime/browser_artifacts/screenshot-dotabuff.png`"
                ),
                metadata={"skill_name": "chat_response"},
            )

    async def classify(_: str) -> str:
        return "ambiguous"

    async def no_control(text: str, context: AgentContext) -> None:
        del text, context
        return None

    runtime = Runtime()
    invocation_service = SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=classify,  # type: ignore[arg-type]
        is_skill_allowed=lambda skill_name, mode: is_skill_allowed(skill_name, mode),
    )
    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(
        bot_main,
        "_with_chat_context_budget_preselection_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", invocation_service)
    monkeypatch.setattr(
        bot_main,
        "_skills",
        [
            WebResearchSkill(admin_user_id=12345, runtime=runtime),
            FakeExternalSkillAcquisition(),
            FakeChatResponseSkill(),
        ],
    )

    response = await bot_main._process_text_message(
        "Открой дотабаф и найди игрока под ником kereexa, "
        "сделай скриншот его статистики",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            metadata={
                "return_response_text": True,
                "source": "vscode",
                "interface": "vscode",
            },
        ),
    )

    assert response is not None
    assert "dotabuff.com/search" in response
    assert runtime.created
    assert runtime.created[0].kind == "web_research"
    assert runtime.created[0].context_pack.user_request == (
        "https://www.dotabuff.com/search?utf8=%E2%9C%93&q=kereexa"
    )


async def test_process_text_message_never_falls_back_to_raw_body_response_when_synthesis_fails(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main

    raw_response = "RAW CAPSULE\n\nДальше:\n- Передать прочитанный контекст Жвуше"

    class FakeOutcome:
        def __init__(self, result: SkillResult | None, *, handled: bool = True) -> None:
            self.handled = handled
            self.result = result

    class FakeInvocationService:
        async def dispatch(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
        ) -> FakeOutcome:
            del text, context, skills
            return FakeOutcome(
                SkillResult(
                    success=True,
                    response=raw_response,
                    metadata={
                        "skill_name": "web_research",
                        "requires_zhvusha_response": True,
                        "body_observation": {
                            "event": "web_research_completed",
                            "sources": ["https://docs.python.org/3/whatsnew/3.14.html"],
                        },
                    },
                )
            )

        async def invoke_named_skill(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
            skill_name: str,
        ) -> FakeOutcome:
            del text, context, skills
            assert skill_name == "chat_response"
            return FakeOutcome(None, handled=False)

    async def no_control(text: str, context: AgentContext) -> None:
        del text, context
        return None

    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(
        bot_main,
        "_with_chat_context_budget_preselection_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", FakeInvocationService())
    monkeypatch.setattr(bot_main, "_skills", [])

    response = await bot_main._process_text_message(
        "Жвуша, найди источники про Python 3.14 release notes и дай ответ",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            metadata={"return_response_text": True},
        ),
    )

    assert response is not None
    assert "RAW CAPSULE" not in response
    assert "Передать прочитанный контекст" not in response
    assert "не смогла безопасно собрать" in response


async def test_process_text_message_returns_pending_approval_without_synthesis(
    monkeypatch: Any,
) -> None:
    import src.bot.main as bot_main

    class FakeOutcome:
        handled = True
        result = SkillResult(
            success=True,
            response="Нужно решение перед выполнением.",
            metadata={
                "skill_name": "external_skill_acquisition",
                "requires_zhvusha_response": True,
                "skip_dialogue_assistant_response": True,
            },
        )

    class FakeInvocationService:
        async def dispatch(
            self,
            text: str,
            context: AgentContext,
            skills: list[Any],
        ) -> FakeOutcome:
            return FakeOutcome()

    async def no_control(text: str, context: AgentContext) -> None:
        return None

    synthesize = AsyncMock()
    monkeypatch.setattr(bot_main, "_control_command_reply", no_control)
    monkeypatch.setattr(
        bot_main,
        "_with_dialogue_context_metadata",
        lambda text, context: context,
    )
    monkeypatch.setattr(bot_main, "_record_dialogue_user_message", lambda *_: None)
    monkeypatch.setattr(bot_main, "_record_dialogue_skill_result", lambda *_: None)
    monkeypatch.setattr(bot_main, "_skill_invocation_service", FakeInvocationService())
    monkeypatch.setattr(bot_main, "_synthesize_body_observation_response", synthesize)
    monkeypatch.setattr(bot_main, "_skills", [])

    response = await bot_main._process_text_message(
        "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            metadata={"return_response_text": True},
        ),
    )

    assert response == "Нужно решение перед выполнением."
    synthesize.assert_not_awaited()
