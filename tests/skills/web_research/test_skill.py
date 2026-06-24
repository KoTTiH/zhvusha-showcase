"""User-facing web research skill contracts."""

from __future__ import annotations

from typing import Any

from src.skills.base import AgentContext


def _context(
    *,
    source_actor: str = "user",
    operator_message_kind: str = "",
) -> AgentContext:
    metadata = {
        "source": "vscode",
        "interface": "vscode",
        "source_actor": source_actor,
    }
    if operator_message_kind:
        metadata["operator_message_kind"] = operator_message_kind
    return AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata=metadata,
    )


async def test_web_research_skill_routes_explicit_source_requests() -> None:
    from src.skills.web_research.skill import WebResearchSkill

    class Runtime:
        pass

    skill = WebResearchSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    assert await skill.can_handle("/web_research Python 3.14", _context()) == 0.94
    assert (
        await skill.can_handle(
            "Жвуша, найди в интернете источники про Python 3.14 release notes",
            _context(),
        )
        == 0.93
    )
    assert await skill.can_handle("найди в проекте обработчик", _context()) == 0.0


async def test_web_research_skill_does_not_claim_codex_goal_handoff() -> None:
    from src.skills.web_research.skill import WebResearchSkill

    class Runtime:
        pass

    skill = WebResearchSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    handoff = """
    Codex/operator handoff, sender=codex, не Никита.

    Task packet:
    - title: Continuous source scouting self-complexification loop
    - source_refs: local goal file docs/zhvusha-continuous-scouting-goal.txt

    Source Packet:
    1. [hacker_news_algolia/weak_signal] Agent runtime
       url: https://github.com/example/agent-runtime

    Readiness Capsule:
    - verdict=runtime_ready_for_gated_iteration

    Physical Evidence Packet:
    - source_read_matrix.json exists=True sha256=cccccccccccccccc

    Handoff prompt:
    Выбери один outcome и не запускай web research.
    """

    assert await skill.can_handle(handoff, _context(source_actor="codex")) == 0.0
    assert (
        await skill.can_handle(
            "/web_research Python 3.14 release notes",
            _context(source_actor="codex"),
        )
        == 0.94
    )


async def test_web_research_skill_does_not_claim_codex_proof_replay() -> None:
    from src.skills.web_research.skill import WebResearchSkill

    class Runtime:
        pass

    skill = WebResearchSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    proof_replay = """
    Codex/operator proof replay, sender=codex, не Никита.

    Это продолжение active goal loop, не новое сообщение пользователя.
    Source scouting сейчас не запускать.

    Runner state:
    - current_turn: turn-000420

    No-Write Proof Bundle:
    - status=complete
    - bridge-replay-receipt.json exists=True sha256=aaaaaaaaaaaaaaaa

    Decision request:
    Выбери approve_exact_implementation или blocked_by_dirty_target_or_missing_capability.
    """

    assert (
        await skill.can_handle(
            proof_replay,
            _context(
                source_actor="codex",
                operator_message_kind="goal_loop_proof_replay",
            ),
        )
        == 0.0
    )


async def test_web_research_skill_routes_browser_article_screenshot_requests() -> None:
    from src.skills.web_research.skill import WebResearchSkill

    class Runtime:
        pass

    skill = WebResearchSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    assert (
        await skill.can_handle(
            "Открой в интернете какую-то статью, сделай скриншот "
            "и пришли мне в этот чат",
            _context(),
        )
        == 0.93
    )
    assert (
        await skill.can_handle(
            "Сделай скриншот кода в проекте и покажи",
            _context(),
        )
        == 0.0
    )


async def test_web_research_skill_routes_explicit_url_browser_smoke() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
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
                    sources=("https://example.com/",),
                    artifacts=(
                        "agent_runtime/browser_artifacts/screenshot-example.png",
                    ),
                ),
            )

    runtime = Runtime()
    skill = WebResearchSkill(admin_user_id=1291112109, runtime=runtime)
    message = (
        "Browser-use smoke от Codex/operator: открой https://example.com/ "
        "через браузерные read-only tools, сделай скриншот и ответь только "
        "в этот VS Code chat."
    )

    assert await skill.can_handle(message, _context()) == 0.93
    result = await skill.execute(message, _context())

    assert result.success is True
    assert runtime.created[0].context_pack.user_request == "https://example.com/"
    assert result.metadata["artifacts"] == (
        "agent_runtime/browser_artifacts/screenshot-example.png",
    )
    assert result.metadata["deliver_artifacts_to_chat"] is True


async def test_web_research_skill_does_not_route_interactive_browser_tasks() -> None:
    from src.skills.web_research.skill import WebResearchSkill

    class Runtime:
        pass

    skill = WebResearchSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    assert (
        await skill.can_handle(
            "Пройди этот тест и скинь скриншот результата: "
            "https://psytests.org/work/kosA_1.html",
            _context(),
        )
        == 0.0
    )
    assert (
        await skill.can_handle(
            "Заполни форму https://example.com/form и отправь результат",
            _context(),
        )
        == 0.0
    )
    assert (
        await skill.can_handle(
            "Открой https://example.com/ и сделай скриншот",
            _context(),
        )
        == 0.93
    )


async def test_web_research_skill_routes_public_profile_screenshot_request() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
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
                    summary="Открыла public profile search page.",
                    processed_context="Dotabuff search for kereexa",
                    sources=(
                        "https://www.dotabuff.com/search?utf8=%E2%9C%93&q=kereexa",
                    ),
                    artifacts=(
                        "agent_runtime/browser_artifacts/screenshot-dotabuff.png",
                    ),
                ),
            )

    runtime = Runtime()
    skill = WebResearchSkill(admin_user_id=1291112109, runtime=runtime)
    message = (
        "Открой дотабаф и найди игрока под ником kereexa, "
        "сделай скриншот его статистики"
    )

    assert await skill.can_handle(message, _context()) == 0.93
    result = await skill.execute(message, _context())

    assert result.success is True
    assert runtime.created[0].context_pack.user_request == (
        "https://www.dotabuff.com/search?utf8=%E2%9C%93&q=kereexa"
    )
    assert result.metadata["deliver_artifacts_to_chat"] is True


async def test_web_research_skill_routes_dota_player_analysis_request() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
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
                    summary="Searched public Dota player sources.",
                    processed_context="No confirmed profile in public sources.",
                    sources=("https://www.opendota.com/",),
                ),
            )

    runtime = Runtime()
    skill = WebResearchSkill(admin_user_id=1291112109, runtime=runtime)
    message = (
        "Можешь проанализировать kereexa игрока в доте? "
        "Жёстко прям его проанализируй, всю необходимую информацию ищи сама"
    )

    assert await skill.can_handle(message, _context()) == 0.93
    result = await skill.execute(message, _context())

    assert result.success is True
    assert runtime.created[0].context_pack.user_request == (
        "kereexa Dota 2 Dotabuff OpenDota STRATZ SteamID"
    )


async def test_web_research_skill_does_not_claim_local_computer_research_task() -> None:
    from src.skills.web_research.skill import WebResearchSkill

    class Runtime:
        pass

    skill = WebResearchSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    message = (
        "Можешь проанализировать kereexa игрока в доте? "
        "Жёстко прям его проанализируй, всю необходимую информацию ищи и "
        "добывай сама, весь компьютер со всеми функциями в твоём распоряжении"
    )

    assert await skill.can_handle(message, _context()) == 0.0


async def test_web_research_skill_runs_agent_runtime_job() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
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
                    processed_context="Python 3.14 release notes",
                    sources=("https://docs.python.org/3/whatsnew/3.14.html",),
                    next_actions=(
                        "Передать прочитанный контекст Жвуше для синтеза ответа.",
                    ),
                ),
            )

    runtime = Runtime()
    skill = WebResearchSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute(
        "Жвуша, найди в интернете источники про Python 3.14 release notes",
        _context(),
    )

    assert result.success is True
    assert result.response == ""
    assert result.metadata["requires_zhvusha_response"] is True
    body_observation = result.metadata["body_observation"]
    assert body_observation["event"] == "web_research_completed"
    assert body_observation["source"] == "web_research"
    assert body_observation["query"] == "Python 3.14 release notes"
    assert body_observation["sources"] == [
        "https://docs.python.org/3/whatsnew/3.14.html"
    ]
    assert "Python 3.14 release notes" in body_observation["processed_context"]
    assert "Передать прочитанный контекст" not in json_dump(body_observation)
    assert runtime.created[0].kind == "web_research"
    assert runtime.created[0].profile.id == "web_research.readonly"
    assert runtime.created[0].context_pack.user_request == "Python 3.14 release notes"
    assert "cite_sources_in_result" in runtime.created[0].context_pack.constraints


async def test_web_research_skill_preserves_screenshot_artifacts_for_delivery() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
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
                    processed_context="article text",
                    sources=("https://example.com/article",),
                    artifacts=(
                        "agent_runtime/browser_artifacts/screenshot-article.png",
                    ),
                ),
            )

    runtime = Runtime()
    skill = WebResearchSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute(
        "Открой в интернете какую-то статью, сделай скриншот и пришли мне",
        _context(),
    )

    assert result.success is True
    assert result.metadata["artifacts"] == (
        "agent_runtime/browser_artifacts/screenshot-article.png",
    )
    assert result.metadata["deliver_artifacts_to_chat"] is True
    body_observation = result.metadata["body_observation"]
    assert body_observation["artifacts"] == [
        "agent_runtime/browser_artifacts/screenshot-article.png"
    ]
    assert runtime.created[0].kind == "web_research"
    assert (
        runtime.created[0].context_pack.user_request
        == "https://ru.wikipedia.org/wiki/Special:Random"
    )


async def test_web_research_skill_constrains_synthesis_when_no_sources() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
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
                    summary="не нашла URL для read-only web research.",
                    processed_context="",
                    sources=(),
                ),
            )

    runtime = Runtime()
    skill = WebResearchSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute(
        "/web_research latest Python 3.14 release date",
        _context(),
    )

    assert result.success is False
    assert result.metadata["requires_zhvusha_response"] is True
    assert result.metadata["sources"] == ()
    message = result.metadata["body_observation_synthesis_message"]
    assert "не дал проверенных источников" in message
    assert "Не отвечай" in message
    assert "BODY_OBSERVATION" in message


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)
