"""Chat-facing computer-use skill contracts."""

from __future__ import annotations

import json
from typing import Any

from src.skills.base import AgentContext


def _context() -> AgentContext:
    return AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={"source": "telegram", "interface": "telegram"},
    )


async def test_computer_use_skill_routes_interactive_browser_tasks() -> None:
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        pass

    skill = ComputerUseSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    assert (
        await skill.can_handle(
            "Пройди этот тест и скинь скриншот результата: "
            "https://psytests.org/work/kosA_1.html",
            _context(),
        )
        == 0.93
    )
    assert (
        await skill.can_handle(
            "Заполни форму https://example.com/form и отправь результат",
            _context(),
        )
        == 0.93
    )
    assert await skill.can_handle("обсудим тесты для проекта", _context()) == 0.0


async def test_computer_use_skill_runs_agent_runtime_action() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Computer-use action returned non-completed status.",
                    processed_context=(
                        "# Computer-use action\n"
                        "- action: browser_interactive_task\n"
                        "- capability: browser_interactive_task\n"
                        "- tool: browser_live_interactive_task\n"
                        "- status: completed\n\n"
                        "Completed bounded interactive browser task.\n"
                        "1. clicked_result: получить результат\n"
                        "2. result_detected: Result page"
                    ),
                    artifacts=("agent_runtime/computer_use/screenshots/result.png",),
                ),
            )

    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)
    result = await skill.execute(
        "Пройди этот тест и скинь скриншот результата: "
        "https://psytests.org/work/kosA_1.html",
        _context(),
    )

    assert result.success is True
    assert result.response == ""
    assert runtime.created[0].kind == "computer_use.action.browser_interactive_task"
    assert runtime.created[0].profile.id == "computer_use.active_gui"
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload == {
        "action": "browser_interactive_task",
        "metadata": {
            "answer_policy": (
                "use_zhvusha_personality_reference_for_opinion_preference_and_"
                "self_assessment_choices; use_user_task_for_goal_directed_choices; "
                "ask_if_credentials_payment_private_data_or_real_identity_are_needed"
            ),
            "capture_screenshot": "true",
            "persona_context_mode": "reference_only",
            "persona_context_ref": "workspace://personality/current-summary",
            "task_intent": (
                "Пройди этот тест и скинь скриншот результата: "
                "https://psytests.org/work/kosA_1.html"
            ),
        },
        "text": (
            "Пройди этот тест и скинь скриншот результата: "
            "https://psytests.org/work/kosA_1.html"
        ),
        "url": "https://psytests.org/work/kosA_1.html",
    }
    assert (
        "isolated_interactive_browser_task_allowed"
        in runtime.created[0].context_pack.constraints
    )
    assert "browser_submit_hard_stop" not in runtime.created[0].context_pack.constraints
    payload_json = runtime.created[0].context_pack.metadata["computer_use_payload"]
    assert "Ты — Жвуша" not in payload_json
    assert "personality/current-summary" in payload_json
    observation = result.metadata["body_observation"]
    assert observation["event"] == "computer_use_action_completed"
    assert observation["selected_action"] == "browser_interactive_task"
    assert observation["requested_task_requires_multi_step_interaction"] is False
    assert "chat tools" not in observation["processed_context"]
    assert result.metadata["deliver_artifacts_to_chat"] is True


async def test_computer_use_skill_uses_router_normalized_action() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Scrolled lower result block and captured screenshot.",
                    processed_context=(
                        "# Computer-use action\n"
                        "- action: browser_scroll\n"
                        "- status: completed\n\n"
                        "Screenshot captured after scrolling down."
                    ),
                    artifacts=("agent_runtime/computer_use/screenshots/lower.png",),
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "telegram",
            "interface": "telegram",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_scroll",
                "target": "down",
                "metadata": {"capture_screenshot": "true"},
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)
    result = await skill.execute(
        "Это тот же скрин, мне нужен скрин нижних результатов",
        context,
    )

    assert result.success is True
    assert runtime.created[0].kind == "computer_use.action.browser_scroll"
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload == {
        "action": "browser_scroll",
        "target": "down",
        "metadata": {"capture_screenshot": "true"},
    }
    assert result.metadata["deliver_artifacts_to_chat"] is True
    observation = result.metadata["body_observation"]
    assert observation["selected_action"] == "browser_scroll"
    assert observation["artifacts"] == [
        "agent_runtime/computer_use/screenshots/lower.png"
    ]


async def test_computer_use_skill_accepts_worker_normalized_desktop_actions() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        def __init__(self) -> None:
            self.created: list[AgentJob] = []

        async def create_job(self, **kwargs: Any) -> AgentJob:
            job = AgentJob.new(**kwargs)
            self.created.append(job)
            return job

        async def start(self, job_id: str) -> AgentJob:
            job = self.created[-1]
            assert job.id == job_id
            return job.with_status(
                AgentJobStatus.DONE,
                result=ContextCapsule(
                    summary="Desktop action completed.",
                    processed_context="# Computer-use action\n- status: completed",
                ),
            )

    actions = (
        (
            "desktop_app_launcher",
            {
                "target": "org.telegram.desktop.desktop",
                "goal": "открыть Telegram",
                "constraints": ["do_not_send_messages_without_approval"],
                "artifact_requirements": {"screenshots": "after_action"},
            },
        ),
        (
            "desktop_window_control",
            {
                "operation": "focus",
                "target": "class:^(Code)$",
                "goal": "сфокусировать VS Code",
                "constraints": ["only_focus_existing_window"],
            },
        ),
        (
            "desktop_hotkeys",
            {
                "target": "escape",
                "goal": "нажать Escape",
                "constraints": ["do_not_confirm_modal"],
            },
        ),
        (
            "desktop_input",
            {
                "text": "привет",
                "goal": "напечатать привет в активное поле",
                "constraints": ["active_field_only"],
            },
        ),
        (
            "desktop_screenshot",
            {
                "goal": "сделать скрин текущего рабочего стола",
                "artifact_requirements": {"screenshots": "current_desktop"},
            },
        ),
        (
            "desktop_media_control",
            {
                "operation": "pause",
                "goal": "поставить музыку на паузу",
                "constraints": ["media_control_only"],
            },
        ),
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)

    for action, payload in actions:
        context = AgentContext(
            user_id=1291112109,
            chat_id=-7331,
            mode="personal",
            metadata={
                "source": "telegram",
                "interface": "telegram",
                "skill_router_selected_skill": "computer_use",
                "skill_router_normalized_action": {
                    "action": action,
                    **payload,
                },
            },
        )
        result = await skill.execute(str(payload["goal"]), context)

        assert result.success is True
        assert runtime.created[-1].kind == f"computer_use.action.{action}"
        normalized = json.loads(
            runtime.created[-1].context_pack.metadata["computer_use_payload"]
        )
        assert normalized["action"] == action
        assert normalized["goal"] == payload["goal"]
        for key in ("target", "operation", "text", "constraints"):
            if key in payload:
                assert normalized[key] == payload[key]
        if "artifact_requirements" in payload:
            assert (
                normalized["artifact_requirements"] == payload["artifact_requirements"]
            )


async def test_computer_use_skill_preserves_runtime_contract_fields() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Read-only existing session lookup completed.",
                    processed_context="# Computer-use action\n- status: completed",
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "telegram",
            "interface": "telegram",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_interactive_task",
                "goal": "найти SteamID через уже открытый Steam",
                "text": "найди kereexa SteamID без ввода пароля",
                "constraints": [
                    "use_existing_session_only",
                    "do_not_enter_credentials",
                ],
                "artifact_requirements": {
                    "screenshots": "public_profile_evidence",
                    "sources": "steam_dotabuff_opendota_stratz",
                },
                "success_criteria": [
                    "steamid_extracted",
                    "sources_collected_or_precise_blocker",
                ],
                "risk_intent": "readonly_existing_session",
                "approval_scope": {
                    "allowed": "read already-open Steam profile data",
                    "forbidden": "password entry, send, purchase, delete, shell",
                },
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute("найди kereexa SteamID без ввода пароля", context)

    assert result.success is True
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload["goal"] == "найти SteamID через уже открытый Steam"
    assert payload["constraints"] == [
        "use_existing_session_only",
        "do_not_enter_credentials",
    ]
    assert payload["artifact_requirements"] == {
        "screenshots": "public_profile_evidence",
        "sources": "steam_dotabuff_opendota_stratz",
    }
    assert payload["success_criteria"] == [
        "steamid_extracted",
        "sources_collected_or_precise_blocker",
    ]
    assert payload["risk_intent"] == "readonly_existing_session"
    assert payload["approval_scope"]["forbidden"] == (
        "password entry, send, purchase, delete, shell"
    )


def test_computer_use_skill_requests_scoped_approval_for_dangerous_browser_action() -> (
    None
):
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        pass

    skill = ComputerUseSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_type",
                "target": "password",
                "selector": "#password",
                "text": "secret",
                "goal": "ввести пароль",
            },
        },
    )

    assert skill.requires_approval_for_message("введи пароль", context)


def test_computer_use_skill_explicit_command_ignores_stale_router_metadata() -> None:
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        pass

    skill = ComputerUseSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_type",
                "target": "password",
                "selector": "#password",
                "text": "secret",
                "goal": "ввести пароль",
            },
        },
    )

    assert (
        skill.requires_approval_for_message(
            '/computer_use {"action":"browser_status"}',
            context,
        )
        is False
    )


async def test_computer_use_skill_execute_explicit_command_ignores_stale_router_metadata() -> (
    None
):
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Browser status collected.",
                    processed_context="# Browser status\n- status: completed",
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "vscode",
            "interface": "vscode",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_interactive_task",
                "goal": "старый payload из предыдущей задачи",
                "metadata": {"source": "stale_router_state"},
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute(
        '/computer_use {"action":"browser_status"}',
        context,
    )

    assert result.success is True
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload == {"action": "browser_status"}


async def test_computer_use_skill_normalizes_explicit_interactive_fact_extract() -> (
    None
):
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Profile facts collected.",
                    processed_context="# Computer-use action\n- status: completed",
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={"source": "vscode", "interface": "vscode"},
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)
    command = (
        '/computer_use {"action":"browser_interactive_task",'
        '"url":"https://www.dotabuff.com/players/997362076",'
        '"goal":"собрать видимые stats/profile facts для анализа игрока kereexa"}'
    )

    result = await skill.execute(command, context)

    assert result.success is True
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload["artifact_requirements"]["include_sources"] == "true"
    assert (
        payload["artifact_requirements"]["text_extract"]
        == "relevant_profile_stats_and_visible_page_facts"
    )
    assert payload["metadata"]["capture_page_html"] == "true"


def test_computer_use_skill_accepts_structured_router_metadata_sources() -> None:
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        pass

    skill = ComputerUseSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_interactive_task",
                "text": (
                    "найти публичный идентификатор игрока kereexa в Dota/Steam "
                    "через уже открытую сессию"
                ),
                "goal": "получить SteamID/профиль и публичную статистику",
                "constraints": [
                    "use_existing_session_only",
                    "do_not_enter_credentials",
                    "do_not_click_sign_in",
                    "prefer_public_sources",
                ],
                "artifact_requirements": {
                    "screenshots": "relevant_profile_and_stats_pages_if_available",
                    "links": "Steam/Dotabuff/OpenDota/Stratz/profile_or_match_urls",
                    "deliver_to_chat": "true",
                },
                "success_criteria": [
                    "steamid_or_match_id_found_or_precise_blocker_reported",
                    "public_stats_collected",
                ],
                "risk_intent": "readonly_existing_session",
                "approval_scope": {
                    "allowed": (
                        "read already-open Steam/browser session data and public "
                        "web pages"
                    ),
                    "forbidden": (
                        "password entry, 2FA, sign-in click, sending messages, "
                        "friend requests, account mutation, shell"
                    ),
                },
                "metadata": {
                    "player_query": "kereexa",
                    "game": "Dota 2",
                    "sources": [
                        "Steam existing session if already open",
                        "Dotabuff",
                        "OpenDota",
                        "Stratz",
                    ],
                    "capture_screenshot": "true",
                    "capture_page_html": "true",
                },
            },
        },
    )

    assert (
        skill.requires_approval_for_message(
            "Можешь проанализировать kereexa игрока в доте?",
            context,
        )
        is False
    )


async def test_computer_use_skill_adds_search_start_url_for_discovery_task() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Started public search instead of about:blank.",
                    processed_context="# Computer-use action\n- status: completed",
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "telegram",
            "interface": "telegram",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_interactive_task",
                "text": (
                    "найти публичный идентификатор игрока kereexa в Dota/Steam "
                    "через уже открытую сессию"
                ),
                "goal": "получить SteamID/профиль и публичную статистику",
                "constraints": [
                    "use_existing_session_only",
                    "do_not_enter_credentials",
                    "prefer_public_sources",
                ],
                "artifact_requirements": {
                    "screenshots": "relevant_profile_and_stats_pages_if_available",
                    "deliver_to_chat": "true",
                },
                "risk_intent": "readonly_existing_session",
                "metadata": {
                    "player_query": "kereexa",
                    "game": "Dota 2",
                    "sources": ["Dotabuff", "OpenDota", "Stratz"],
                },
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute(
        "Можешь проанализировать kereexa игрока в доте?",
        context,
    )

    assert result.success is True
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload["url"].startswith("https://www.dotabuff.com/search?q=")
    assert "kereexa" in payload["url"]
    assert "about:blank" not in payload["url"]
    assert "duckduckgo.com" not in payload["url"]
    assert payload["metadata"]["auto_start_url"] == "dotabuff_player_search"


async def test_computer_use_skill_replaces_duckduckgo_router_search_for_dota_task() -> (
    None
):
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Started source-specific public search.",
                    processed_context="# Computer-use action\n- status: completed",
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "telegram",
            "interface": "telegram",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_interactive_task",
                "url": (
                    "https://duckduckgo.com/?q=kereexa+Dota+2+Dotabuff+"
                    "OpenDota+STRATZ+Steam+profile"
                ),
                "text": "найти публичную статистику kereexa в Dota 2",
                "goal": "получить SteamID/профиль и публичную статистику",
                "risk_intent": "readonly_existing_session",
                "metadata": {
                    "player_query": "kereexa",
                    "game": "Dota 2",
                    "sources": "Dotabuff OpenDota STRATZ Steam",
                },
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute(
        "Можешь проанализировать kereexa игрока в доте?",
        context,
    )

    assert result.success is True
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload["url"].startswith("https://www.dotabuff.com/search?q=")
    assert "kereexa" in payload["url"]
    assert "duckduckgo.com" not in payload["url"]
    assert payload["metadata"]["auto_start_url"] == "dotabuff_player_search"
    assert payload["metadata"]["replaced_start_url"] == "duckduckgo_search"


async def test_computer_use_skill_replaces_generic_google_search_for_dota_task() -> (
    None
):
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Replaced generic search with source-specific search.",
                    processed_context="# Computer-use action\n- status: completed",
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "vscode",
            "interface": "vscode",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_interactive_task",
                "url": "https://www.google.com/search?q=kereexa+Dota+2+player",
                "text": ("найти публично подтверждённый Dota-профиль игрока kereexa"),
                "goal": "собрать публичную статистику и профиль игрока",
                "risk_intent": "readonly_existing_session",
                "metadata": {
                    "target_nick": "kereexa",
                    "game": "Dota 2",
                    "domain_hints": ["dotabuff.com", "opendota.com", "stratz.com"],
                },
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute(
        "Можешь проанализировать kereexa игрока в доте?",
        context,
    )

    assert result.success is True
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload["url"] == "https://www.dotabuff.com/search?q=kereexa"
    assert payload["metadata"]["auto_start_url"] == "dotabuff_player_search"
    assert payload["metadata"]["replaced_start_url"] == "google_search"


async def test_computer_use_skill_routes_local_computer_research_task_without_router() -> (
    None
):
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Computer-use read-only discovery completed.",
                    processed_context="No confirmed Dota profile yet.",
                    sources=("https://steamcommunity.com/search/users/#text=kereexa",),
                ),
            )

    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)
    message = (
        "Можешь проанализировать kereexa игрока в доте? "
        "Жёстко прям его проанализируй, всю необходимую информацию ищи и "
        "добывай сама, весь компьютер со всеми функциями в твоём распоряжении"
    )

    assert await skill.can_handle(message, _context()) == 0.93
    result = await skill.execute(message, _context())

    assert result.success is True
    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload["action"] == "browser_interactive_task"
    assert payload["url"].startswith("https://www.dotabuff.com/search?q=")
    assert "kereexa" in payload["url"]
    assert "duckduckgo.com" not in payload["url"]
    assert payload["risk_intent"] == "readonly_existing_session"
    assert "use_existing_session_only" in payload["constraints"]
    assert "do_not_enter_credentials" in payload["constraints"]
    assert "do_not_send_messages" in payload["constraints"]
    assert payload["metadata"]["player_query"] == "kereexa"
    assert payload["metadata"]["game"] == "Dota 2"
    assert payload["metadata"]["auto_start_url"] == "dotabuff_player_search"
    assert payload["artifact_requirements"]["include_sources"] == "true"
    assert (
        payload["artifact_requirements"]["text_extract"]
        == "relevant_profile_stats_and_visible_page_facts"
    )


async def test_computer_use_skill_does_not_deterministically_route_plain_desktop_nl() -> (
    None
):
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        pass

    skill = ComputerUseSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    for phrase in (
        "открой Telegram",
        "сфокусируй VS Code",
        "нажми Escape",
        "напечатай привет в активное поле",
        "поставь музыку на паузу",
        "обсудим Telegram, окна и горячие клавиши",
    ):
        assert await skill.can_handle(phrase, _context()) == 0.0


def test_computer_use_skill_requires_approval_for_normalized_desktop_side_effects() -> (
    None
):
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        pass

    skill = ComputerUseSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    side_effect_context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "desktop_app_launcher",
                "target": "org.telegram.desktop.desktop",
                "goal": "открыть Telegram",
            },
        },
    )
    screenshot_context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "desktop_screenshot",
                "goal": "сделать скрин",
            },
        },
    )

    assert skill.requires_approval_for_message("открой Telegram", side_effect_context)
    assert not skill.requires_approval_for_message("сделай скрин", screenshot_context)


async def test_computer_use_skill_bridges_skill_approval_to_tool_gateway_grant() -> (
    None
):
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Desktop action completed.",
                    processed_context="# Computer-use action\n- status: completed",
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "telegram",
            "interface": "telegram",
            "skill_approval_granted": True,
            "skill_approval_id": "approval-computer-desktop",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "desktop_app_launcher",
                "target": "org.telegram.desktop.desktop",
                "goal": "открыть Telegram",
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)

    result = await skill.execute("открой Telegram", context)

    assert result.success is True
    metadata = runtime.created[0].context_pack.metadata
    assert metadata["agent_tool_approval_id"] == "approval-computer-desktop"
    assert metadata["agent_tool_approval_capabilities"] == "desktop_app_launcher"


async def test_computer_use_skill_uses_structured_artifact_requirements() -> None:
    from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
    from src.skills.computer_use.skill import ComputerUseSkill

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
                    summary="Captured all relevant result screenshots.",
                    processed_context="# Computer-use action\n- status: completed",
                    artifacts=(
                        "agent_runtime/computer_use/screenshots/result-1.png",
                        "agent_runtime/computer_use/screenshots/result-2.png",
                    ),
                ),
            )

    context = AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        metadata={
            "source": "telegram",
            "interface": "telegram",
            "skill_router_selected_skill": "computer_use",
            "skill_router_normalized_action": {
                "action": "browser_interactive_task",
                "url": "https://example.com/test",
                "goal": "пройти тест и вернуть все визуальные результаты",
                "artifact_requirements": {
                    "screenshots": "all_relevant_result_sections",
                    "deliver_to_chat": "true",
                },
            },
        },
    )
    runtime = Runtime()
    skill = ComputerUseSkill(admin_user_id=1291112109, runtime=runtime)
    result = await skill.execute(
        "пройди тест и верни все визуальные результаты https://example.com/test",
        context,
    )

    payload = json.loads(
        runtime.created[0].context_pack.metadata["computer_use_payload"]
    )
    assert payload["text"] == (
        "пройди тест и верни все визуальные результаты https://example.com/test"
    )
    assert payload["goal"] == "пройти тест и вернуть все визуальные результаты"
    assert payload["artifact_requirements"] == {
        "screenshots": "all_relevant_result_sections",
        "deliver_to_chat": "true",
    }
    assert payload["metadata"]["capture_screenshot"] == "true"
    assert payload["metadata"]["capture_result_screenshots"] == (
        "all_relevant_result_sections"
    )
    assert payload["metadata"]["capture_page_html"] == "true"
    assert payload["metadata"]["persona_context_ref"] == (
        "workspace://personality/current-summary"
    )
    assert result.metadata["deliver_artifacts_to_chat"] is True


async def test_computer_use_skill_routes_hidden_json_command() -> None:
    from src.skills.computer_use.skill import ComputerUseSkill

    class Runtime:
        pass

    skill = ComputerUseSkill(
        admin_user_id=1291112109,
        runtime=Runtime(),  # type: ignore[arg-type]
    )

    assert (
        await skill.can_handle('/computer_use {"action":"browser_status"}', _context())
        == 0.95
    )
