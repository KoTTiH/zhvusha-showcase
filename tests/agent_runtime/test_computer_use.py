"""Computer-use Agent Runtime contracts."""

from __future__ import annotations

import json
from typing import Any


class InteractiveTaskFakeCDPSession:
    def __init__(
        self,
        *,
        calls: list[tuple[str, dict[str, Any] | None]],
        steps: list[dict[str, Any]],
        screenshot_prefix: bytes = b"result-png",
        result_sections: list[dict[str, Any]] | None = None,
        page_content_sections: list[dict[str, Any]] | None = None,
        human_verification_states: list[dict[str, Any]] | None = None,
        consent_steps: list[dict[str, Any]] | None = None,
        profile_steps: list[dict[str, Any]] | None = None,
    ) -> None:
        self._calls = calls
        self._steps = steps
        self._screenshot_prefix = screenshot_prefix
        self._screenshot_count = 0
        self._result_sections = result_sections or []
        self._page_content_sections = page_content_sections or []
        self._human_verification_states = human_verification_states or []
        self._consent_steps = consent_steps or []
        self._profile_steps = profile_steps or []

    async def __aenter__(self) -> InteractiveTaskFakeCDPSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def send_raw(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        del session_id
        self._calls.append((method, params))
        if method == "Runtime.evaluate":
            return self._evaluate(str((params or {}).get("expression", "")))
        if method == "Page.captureScreenshot":
            import base64

            self._screenshot_count += 1
            payload = self._screenshot_prefix
            if self._screenshot_count > 1:
                payload = payload + f"-{self._screenshot_count}".encode()
            return {"data": base64.b64encode(payload).decode("ascii")}
        return {}

    def _evaluate(self, expression: str) -> dict[str, Any]:
        if "markers.find" in expression and "blocked: Boolean(marker)" in expression:
            if self._human_verification_states:
                return {"result": {"value": self._human_verification_states.pop(0)}}
            return {"result": {"value": {"blocked": False}}}
        if "cookieConsentCandidates" in expression:
            if self._consent_steps:
                return {"result": {"value": self._consent_steps.pop(0)}}
            return {"result": {"value": {"ok": True, "status": "no_consent"}}}
        if "publicProfileCandidates" in expression:
            if self._profile_steps:
                return {"result": {"value": self._profile_steps.pop(0)}}
            return {"result": {"value": {"ok": True, "status": "no_profile_result"}}}
        if "radiosByName" in expression:
            assert "workspace://personality/current-summary" in expression
            assert "Ты — Жвуша" not in expression
            return {"result": {"value": self._steps.pop(0)}}
        section_value = self._evaluate_section_extract_expression(expression)
        if section_value is not None:
            return {"result": {"value": section_value}}
        handled, value = self._evaluate_static_page_expression(expression)
        if handled:
            return {"result": {"value": value}}
        return {"result": {"value": []}}

    def _evaluate_section_extract_expression(self, expression: str) -> Any | None:
        if "resultSectionCandidates" in expression:
            return self._result_sections
        if "contentTerms" in expression and "focusTerms" in expression:
            return self._page_content_sections
        return None

    def _evaluate_static_page_expression(self, expression: str) -> tuple[bool, Any]:
        if "document.readyState" in expression:
            return True, "complete"
        if "document.documentElement.outerHTML" in expression:
            return (
                True,
                "<html><title>Result page</title><body>saved result</body></html>",
            )
        if "scrollHeight" in expression and "viewportHeight" in expression:
            return True, {
                "scrollHeight": 1600,
                "viewportHeight": 700,
                "scrollY": 0,
            }
        if "window.scrollTo" in expression:
            return True, True
        if "document.title" in expression:
            return True, "Result page"
        if "location.href" in expression:
            return True, "https://example.com/result"
        return False, None


async def test_local_chrome_devtools_client_navigates_and_captures_screenshot(
    tmp_path,
) -> None:
    import base64

    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []

    class FakeCDPSession:
        async def __aenter__(self) -> FakeCDPSession:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def send_raw(
            self,
            method: str,
            params: dict[str, Any] | None = None,
            session_id: str | None = None,
        ) -> dict[str, Any]:
            del session_id
            calls.append((method, params))
            if method == "Runtime.evaluate":
                expression = str((params or {}).get("expression", ""))
                if "document.title" in expression:
                    return {"result": {"value": "Result page"}}
                if "location.href" in expression:
                    return {"result": {"value": "https://example.com/result"}}
                if "document.readyState" in expression:
                    return {"result": {"value": "complete"}}
                return {"result": {"value": "ok"}}
            if method == "Page.captureScreenshot":
                return {"data": base64.b64encode(b"png-bytes").decode("ascii")}
            return {}

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: FakeCDPSession(),
    )

    result = await client.execute_action(
        "browser_navigate",
        {
            "action": "browser_navigate",
            "url": "https://example.com/result",
            "metadata": {"capture_screenshot": "true"},
        },
    )

    assert result.status == "completed"
    assert result.artifact.startswith("agent_runtime/computer_use/screenshots/")
    assert (tmp_path / result.artifact).read_bytes() == b"png-bytes"
    assert ("Page.navigate", {"url": "https://example.com/result"}) in calls
    assert any(method == "Page.captureScreenshot" for method, _params in calls)


async def test_local_chrome_devtools_client_names_navigation_timeout(tmp_path) -> None:
    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    class TimeoutCDPSession:
        async def __aenter__(self) -> TimeoutCDPSession:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def send_raw(
            self,
            method: str,
            params: dict[str, Any] | None = None,
            session_id: str | None = None,
        ) -> dict[str, Any]:
            del method, params, session_id
            raise TimeoutError

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: TimeoutCDPSession(),
    )

    result = await client.execute_action(
        "browser_navigate",
        {"url": "https://psytests.org/work/kosA_1.html"},
    )

    assert result.status == "degraded"
    assert "timed out" in result.message
    assert "psytests.org" in result.message
    assert result.metadata["error"] == "timeout"
    assert result.metadata["target_host"] == "psytests.org"


async def test_local_chrome_devtools_client_scroll_with_url_restores_page(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []

    class FakeCDPSession:
        async def __aenter__(self) -> FakeCDPSession:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def send_raw(
            self,
            method: str,
            params: dict[str, Any] | None = None,
            session_id: str | None = None,
        ) -> dict[str, Any]:
            del session_id
            calls.append((method, params))
            if method != "Runtime.evaluate":
                return {}
            expression = str((params or {}).get("expression", ""))
            if "document.readyState" in expression:
                return {"result": {"value": "complete"}}
            if "document.title" in expression:
                return {"result": {"value": "Result page"}}
            if "location.href" in expression:
                return {"result": {"value": "https://example.com/result"}}
            return {"result": {"value": []}}

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: FakeCDPSession(),
    )

    result = await client.execute_action(
        "browser_scroll",
        {
            "action": "browser_scroll",
            "url": "https://example.com/result",
            "target": "ниже",
        },
    )

    assert result.status == "completed"
    assert "Navigated to https://example.com/result." in result.message
    assert "Scrolled by" in result.message
    assert ("Page.navigate", {"url": "https://example.com/result"}) in calls


async def test_local_chrome_devtools_client_completes_interactive_task_with_persona_ref(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []
    interactive_steps = [
        {"ok": True, "status": "clicked_result", "detail": "получить результат"},
        {"ok": True, "status": "result_detected", "detail": "Result page"},
    ]

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: InteractiveTaskFakeCDPSession(
            calls=calls,
            steps=interactive_steps,
        ),
    )

    result = await client.execute_action(
        "browser_interactive_task",
        {
            "action": "browser_interactive_task",
            "url": "https://example.com/test",
            "text": "пройди тест как Жвуша",
            "metadata": {
                "capture_screenshot": "true",
                "persona_context_mode": "reference_only",
                "persona_context_ref": "workspace://personality/current-summary",
                "answer_policy": "use referenced personality for self-assessment",
            },
        },
    )

    assert result.status == "completed"
    assert "clicked_result" in result.message
    assert "result_detected" in result.message
    assert result.artifact.startswith("agent_runtime/computer_use/screenshots/")
    assert (tmp_path / result.artifact).read_bytes() == b"result-png"
    html_artifact = result.metadata["page_html_artifact"]
    assert html_artifact.startswith("agent_runtime/computer_use/page-snapshots/")
    assert "saved result" in (tmp_path / html_artifact).read_text(encoding="utf-8")
    assert ("Page.navigate", {"url": "https://example.com/test"}) in calls
    assert any(method == "Page.captureScreenshot" for method, _params in calls)


async def test_local_chrome_devtools_client_waits_for_human_captcha_then_continues(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []
    interactive_steps = [
        {"ok": True, "status": "clicked_result", "detail": "открыть профиль"},
        {"ok": True, "status": "result_detected", "detail": "Steam profile"},
    ]
    verification_states = [
        {"blocked": True, "detail": "captcha"},
        {"blocked": True, "detail": "captcha"},
        {"blocked": False},
    ]

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Steam",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: InteractiveTaskFakeCDPSession(
            calls=calls,
            steps=interactive_steps,
            human_verification_states=verification_states,
        ),
        human_verification_timeout_seconds=1.0,
        human_verification_poll_seconds=0.01,
    )

    result = await client.execute_action(
        "browser_interactive_task",
        {
            "action": "browser_interactive_task",
            "url": "https://steamcommunity.com/search/users/#text=kereexa",
            "text": "найди профиль kereexa",
            "metadata": {
                "capture_screenshot": "true",
                "persona_context_mode": "reference_only",
                "persona_context_ref": "workspace://personality/current-summary",
            },
        },
    )

    assert result.status == "completed"
    assert "human_verification_resolved" in result.message
    assert "continued automatically" in result.message
    assert "clicked_result" in result.message
    assert "result_detected" in result.message


async def test_local_chrome_devtools_client_clicks_dotabuff_player_result(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []
    profile_steps = [
        {
            "ok": True,
            "status": "clicked_profile",
            "detail": "Kereexa -> https://www.dotabuff.com/players/123456789",
        },
        {
            "ok": True,
            "status": "result_detected",
            "detail": "Dotabuff player profile",
        },
    ]

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "DOTABUFF Search",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: InteractiveTaskFakeCDPSession(
            calls=calls,
            steps=[],
            profile_steps=profile_steps,
        ),
    )

    result = await client.execute_action(
        "browser_interactive_task",
        {
            "action": "browser_interactive_task",
            "url": "https://www.dotabuff.com/search?q=kereexa",
            "text": "найди профиль игрока kereexa на Dotabuff",
            "goal": "получить профиль игрока kereexa",
            "metadata": {
                "player_query": "kereexa",
                "game": "Dota 2",
            },
        },
    )

    assert result.status == "completed"
    assert "clicked_profile" in result.message
    assert "Kereexa -> https://www.dotabuff.com/players/123456789" in result.message
    assert "result_detected" in result.message
    assert any(
        method == "Runtime.evaluate"
        and "publicProfileCandidates" in str((params or {}).get("expression", ""))
        for method, params in calls
    )


async def test_local_chrome_devtools_client_extracts_profile_page_facts_for_sources(
    tmp_path,
) -> None:
    import json

    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []
    profile_steps = [
        {
            "ok": True,
            "status": "clicked_profile",
            "detail": "Kereexa -> https://www.dotabuff.com/players/997362076",
        },
        {
            "ok": True,
            "status": "result_detected",
            "detail": "https://www.dotabuff.com/players/997362076",
        },
    ]
    page_content_sections = [
        {
            "index": 1,
            "text": "Kereexa Overview Last Match 2 days ago Record 852-875-18 Win Rate 48.83%",
            "x": 16,
            "y": 320,
            "width": 900,
            "height": 160,
        },
        {
            "index": 2,
            "text": "Most Played Heroes Shadow Fiend 207 52.17% KDA 2.14 Phantom Assassin 201 52.74% KDA 2.51",
            "x": 16,
            "y": 650,
            "width": 900,
            "height": 320,
        },
    ]

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "DOTABUFF Search",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: InteractiveTaskFakeCDPSession(
            calls=calls,
            steps=[],
            profile_steps=profile_steps,
            page_content_sections=page_content_sections,
        ),
    )

    result = await client.execute_action(
        "browser_interactive_task",
        {
            "action": "browser_interactive_task",
            "url": "https://www.dotabuff.com/search?q=kereexa",
            "text": "найди и проанализируй профиль игрока kereexa",
            "goal": "собрать факты профиля и статистику для анализа",
            "artifact_requirements": {
                "include_sources": "true",
                "screenshots": "relevant_profile_and_stats_pages_if_available",
                "deliver_to_chat": "true",
            },
            "metadata": {
                "player_query": "kereexa",
                "game": "Dota 2",
            },
        },
    )

    assert result.status == "completed"
    assert json.loads(result.metadata["result_sections"]) == page_content_sections
    assert any(
        method == "Runtime.evaluate"
        and "contentTerms" in str((params or {}).get("expression", ""))
        for method, params in calls
    )


async def test_local_chrome_devtools_client_dismisses_consent_before_profile_click(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []
    consent_steps = [
        {"ok": True, "status": "dismissed_consent", "detail": "Alles afwijzen"},
    ]
    profile_steps = [
        {
            "ok": True,
            "status": "clicked_profile",
            "detail": "Kereexa -> https://www.dotabuff.com/players/123456789",
        },
        {
            "ok": True,
            "status": "result_detected",
            "detail": "https://www.dotabuff.com/players/123456789",
        },
    ]

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Google consent",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: InteractiveTaskFakeCDPSession(
            calls=calls,
            steps=[],
            consent_steps=consent_steps,
            profile_steps=profile_steps,
        ),
    )

    result = await client.execute_action(
        "browser_interactive_task",
        {
            "action": "browser_interactive_task",
            "url": "https://www.google.com/search?q=kereexa+dotabuff",
            "text": "найди профиль игрока kereexa на Dotabuff",
            "goal": "получить публичный профиль игрока kereexa",
            "metadata": {
                "player_query": "kereexa",
                "game": "Dota 2",
            },
        },
    )

    assert result.status == "completed"
    assert "dismissed_consent: Alles afwijzen" in result.message
    assert "clicked_profile" in result.message
    assert "result_detected" in result.message
    assert any(
        method == "Runtime.evaluate"
        and "cookieConsentCandidates" in str((params or {}).get("expression", ""))
        for method, params in calls
    )


async def test_local_chrome_devtools_client_captures_required_result_sections(
    tmp_path,
) -> None:
    import json

    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []
    interactive_steps = [
        {"ok": True, "status": "clicked_result", "detail": "получить результат"},
        {"ok": True, "status": "result_detected", "detail": "Result page"},
    ]

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: InteractiveTaskFakeCDPSession(
            calls=calls,
            steps=interactive_steps,
            screenshot_prefix=b"section-png",
        ),
    )

    result = await client.execute_action(
        "browser_interactive_task",
        {
            "action": "browser_interactive_task",
            "url": "https://example.com/test",
            "text": "пройти тест и вернуть все визуальные результаты",
            "artifact_requirements": {
                "screenshots": "all_relevant_result_sections",
                "deliver_to_chat": "true",
            },
            "metadata": {
                "capture_screenshot": "true",
                "capture_result_screenshots": "all_relevant_result_sections",
                "persona_context_mode": "reference_only",
                "persona_context_ref": "workspace://personality/current-summary",
                "answer_policy": "use referenced personality for self-assessment",
            },
        },
    )

    assert result.status == "completed"
    screenshot_artifacts = json.loads(result.metadata["screenshot_artifacts"])
    assert len(screenshot_artifacts) >= 2
    assert result.artifact == screenshot_artifacts[0]
    for artifact in screenshot_artifacts:
        assert artifact.startswith("agent_runtime/computer_use/screenshots/")
        assert (tmp_path / artifact).is_file()
    assert (
        sum(1 for method, _params in calls if method == "Page.captureScreenshot") >= 2
    )
    assert any(
        method == "Runtime.evaluate"
        and params is not None
        and "window.scrollTo" in str(params.get("expression", ""))
        for method, params in calls
    )


async def test_local_chrome_devtools_client_understands_semantic_artifact_contract(
    tmp_path,
) -> None:
    import json

    from src.agent_runtime.computer_use import LocalChromeDevToolsClient

    calls: list[tuple[str, dict[str, Any] | None]] = []
    result_sections = [
        {
            "index": 1,
            "text": "Результат: Младенец. Вы набрали 9 баллов.",
            "x": 195,
            "y": 175,
            "width": 740,
            "height": 210,
        },
        {
            "index": 2,
            "text": "Результат: Малыш. Вы набрали 10 баллов.",
            "x": 195,
            "y": 405,
            "width": 740,
            "height": 280,
        },
    ]

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    client = LocalChromeDevToolsClient(
        debug_url="http://127.0.0.1:9222",
        workspace_root=tmp_path,
        tab_discovery=tab_discovery,
        session_factory=lambda _url: InteractiveTaskFakeCDPSession(
            calls=calls,
            steps=[
                {"ok": True, "status": "clicked_result", "detail": "результат"},
                {"ok": True, "status": "result_detected", "detail": "Result page"},
            ],
            screenshot_prefix=b"semantic-section-png",
            result_sections=result_sections,
        ),
    )

    result = await client.execute_action(
        "browser_interactive_task",
        {
            "action": "browser_interactive_task",
            "url": "https://example.com/test",
            "text": "доделай выдачу всех результатов и интерпретаций",
            "artifact_requirements": {
                "screenshots": (
                    "all_result_sections_separately_plus_full_page_if_possible"
                ),
                "text_extract": "all_scores_titles_descriptions",
                "interpretation": "brief_human_readable_per_section",
                "deliver_to_chat": "true",
            },
            "metadata": {
                "capture_screenshot": "true",
                "capture_result_screenshots": (
                    "all_result_sections_separately_plus_full_page_if_possible"
                ),
                "persona_context_mode": "reference_only",
                "persona_context_ref": "workspace://personality/current-summary",
            },
        },
    )

    assert result.status == "completed"
    screenshot_artifacts = json.loads(result.metadata["screenshot_artifacts"])
    assert len(screenshot_artifacts) == 2
    assert json.loads(result.metadata["result_sections"]) == result_sections
    assert all((tmp_path / artifact).is_file() for artifact in screenshot_artifacts)
    capture_params = [
        params
        for method, params in calls
        if method == "Page.captureScreenshot" and params is not None
    ]
    assert len(capture_params) == 2
    assert all("clip" in params for params in capture_params)


async def test_computer_use_worker_persists_browser_result_record(tmp_path) -> None:
    from types import SimpleNamespace

    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
    )
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI

    screenshot = "agent_runtime/computer_use/screenshots/result.png"
    html = "agent_runtime/computer_use/page-snapshots/result.html"
    (tmp_path / screenshot).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / screenshot).write_bytes(b"png")
    (tmp_path / html).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / html).write_text("<html>saved result</html>", encoding="utf-8")

    class FakeToolGateway:
        def registered_tools(self) -> tuple[SimpleNamespace, ...]:
            return (
                SimpleNamespace(
                    name="browser_live_interactive_task",
                    capability="browser_interactive_task",
                ),
            )

        async def execute(
            self,
            _profile: object,
            _tool_name: str,
            _payload: dict[str, Any],
        ) -> dict[str, Any]:
            return ComputerUseActionResult(
                status="completed",
                message="Completed bounded interactive browser task.",
                artifact=screenshot,
                metadata={
                    "current_url": "https://example.com/result/abc",
                    "title": "Result page",
                    "page_html_artifact": html,
                },
            ).model_dump()

    worker = ComputerUseWorkerBackend(
        tool_gateway=FakeToolGateway(),  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    capsule = await worker.run(
        job=SimpleNamespace(profile=COMPUTER_USE_ACTIVE_GUI),
        context_pack=ContextPack(
            user_request="пройди тест",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "browser_interactive_task",
                        "url": "https://example.com/test",
                        "text": "пройди тест",
                        "metadata": {"capture_screenshot": "true"},
                    }
                )
            },
        ),
    )

    result_record = next(
        artifact
        for artifact in capsule.artifacts
        if artifact.startswith("agent_runtime/computer_use/browser-results/")
    )
    payload = json.loads((tmp_path / result_record).read_text(encoding="utf-8"))

    assert capsule.sources == ("https://example.com/result/abc",)
    assert screenshot in capsule.artifacts
    assert html in capsule.artifacts
    assert payload["kind"] == "computer_use_browser_result"
    assert payload["source_url"] == "https://example.com/test"
    assert payload["result_url"] == "https://example.com/result/abc"
    assert payload["artifacts"] == [screenshot, html]
    assert "browser_result_artifact" in capsule.processed_context


async def test_computer_use_worker_persists_multiple_screenshot_artifacts(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
    )
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI

    first = "agent_runtime/computer_use/screenshots/result-1.png"
    second = "agent_runtime/computer_use/screenshots/result-2.png"
    for artifact in (first, second):
        (tmp_path / artifact).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / artifact).write_bytes(b"png")

    class FakeToolGateway:
        def registered_tools(self) -> tuple[SimpleNamespace, ...]:
            return (
                SimpleNamespace(
                    name="browser_live_interactive_task",
                    capability="browser_interactive_task",
                ),
            )

        async def execute(
            self,
            _profile: object,
            _tool_name: str,
            _payload: dict[str, Any],
        ) -> dict[str, Any]:
            return ComputerUseActionResult(
                status="completed",
                message="Completed bounded interactive browser task.",
                artifact=first,
                metadata={
                    "current_url": "https://example.com/result/abc",
                    "title": "Result page",
                    "screenshot_artifacts": json.dumps([first, second]),
                },
            ).model_dump()

    worker = ComputerUseWorkerBackend(
        tool_gateway=FakeToolGateway(),  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    capsule = await worker.run(
        job=SimpleNamespace(profile=COMPUTER_USE_ACTIVE_GUI),
        context_pack=ContextPack(
            user_request="пройти тест и вернуть все визуальные результаты",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "browser_interactive_task",
                        "url": "https://example.com/test",
                        "text": "пройти тест и вернуть все визуальные результаты",
                        "artifact_requirements": {
                            "screenshots": "all_relevant_result_sections",
                            "deliver_to_chat": "true",
                        },
                    }
                )
            },
        ),
    )

    result_record = next(
        artifact
        for artifact in capsule.artifacts
        if artifact.startswith("agent_runtime/computer_use/browser-results/")
    )
    payload = json.loads((tmp_path / result_record).read_text(encoding="utf-8"))

    assert first in capsule.artifacts
    assert second in capsule.artifacts
    assert payload["artifacts"] == [first, second]
    assert payload["screenshot_artifacts"] == [first, second]


async def test_computer_use_worker_returns_extracted_result_sections_to_zhvusha(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
    )
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI

    screenshot = "agent_runtime/computer_use/screenshots/result-1.png"
    (tmp_path / screenshot).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / screenshot).write_bytes(b"png")
    result_sections = [
        {
            "index": 1,
            "text": "Результат: Младенец. Вы набрали 9 баллов.",
            "x": 195,
            "y": 175,
            "width": 740,
            "height": 210,
        }
    ]

    class FakeToolGateway:
        def registered_tools(self) -> tuple[SimpleNamespace, ...]:
            return (
                SimpleNamespace(
                    name="browser_live_interactive_task",
                    capability="browser_interactive_task",
                ),
            )

        async def execute(
            self,
            _profile: object,
            _tool_name: str,
            _payload: dict[str, Any],
        ) -> dict[str, Any]:
            return ComputerUseActionResult(
                status="completed",
                message="Completed bounded interactive browser task.",
                artifact=screenshot,
                metadata={
                    "current_url": "https://example.com/result/abc",
                    "title": "Result page",
                    "screenshot_artifacts": json.dumps([screenshot]),
                    "result_sections": json.dumps(
                        result_sections,
                        ensure_ascii=False,
                    ),
                },
            ).model_dump()

    worker = ComputerUseWorkerBackend(
        tool_gateway=FakeToolGateway(),  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    capsule = await worker.run(
        job=SimpleNamespace(profile=COMPUTER_USE_ACTIVE_GUI),
        context_pack=ContextPack(
            user_request="доделай все результаты и интерпретации",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "browser_interactive_task",
                        "url": "https://example.com/result/abc",
                        "text": "доделай все результаты и интерпретации",
                        "artifact_requirements": {
                            "screenshots": (
                                "all_result_sections_separately_plus_full_page_if_possible"
                            ),
                            "text_extract": "all_scores_titles_descriptions",
                            "interpretation": "brief_human_readable_per_section",
                            "deliver_to_chat": "true",
                        },
                    }
                )
            },
        ),
    )

    result_record = next(
        artifact
        for artifact in capsule.artifacts
        if artifact.startswith("agent_runtime/computer_use/browser-results/")
    )
    payload = json.loads((tmp_path / result_record).read_text(encoding="utf-8"))

    assert "section 1: Результат: Младенец" in capsule.processed_context
    assert payload["result_sections"] == result_sections


async def test_computer_use_worker_restores_result_url_for_browser_followup(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
    )
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI

    result_dir = tmp_path / "agent_runtime" / "computer_use" / "browser-results"
    result_dir.mkdir(parents=True)
    result_record = result_dir / "browser-result-existing.json"
    result_record.write_text(
        json.dumps(
            {
                "kind": "computer_use_browser_result",
                "chat_id": "-7331",
                "owner_user_id": "1291112109",
                "result_url": "https://example.com/result/abc",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    captured_payload: dict[str, Any] = {}

    class FakeToolGateway:
        def registered_tools(self) -> tuple[SimpleNamespace, ...]:
            return (
                SimpleNamespace(
                    name="browser_live_scroll",
                    capability="browser_scroll",
                ),
            )

        async def execute(
            self,
            _profile: object,
            _tool_name: str,
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            captured_payload.update(payload)
            return ComputerUseActionResult(
                status="completed",
                message="Scrolled by 630px.",
                metadata={
                    "current_url": "https://example.com/result/abc",
                    "title": "Result page",
                },
            ).model_dump()

    worker = ComputerUseWorkerBackend(
        tool_gateway=FakeToolGateway(),  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    capsule = await worker.run(
        job=SimpleNamespace(
            profile=COMPUTER_USE_ACTIVE_GUI,
            owner_user_id=1291112109,
            chat_id=-7331,
        ),
        context_pack=ContextPack(
            user_request="пролистай ниже и заскринь",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "browser_scroll",
                        "target": "ниже",
                        "metadata": {"capture_screenshot": "true"},
                    }
                )
            },
        ),
    )

    assert captured_payload["url"] == "https://example.com/result/abc"
    assert (
        captured_payload["metadata"]["restored_browser_result_artifact"]
        == "agent_runtime/computer_use/browser-results/browser-result-existing.json"
    )
    assert (
        captured_payload["metadata"]["restored_result_url"]
        == "https://example.com/result/abc"
    )
    assert capsule.sources == ("https://example.com/result/abc",)


async def test_managed_chrome_devtools_client_launches_visible_with_proxy(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import ManagedChromeDevToolsClient

    launched: list[tuple[str, ...]] = []
    discovery_calls = 0

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        nonlocal discovery_calls
        discovery_calls += 1
        if discovery_calls == 1:
            raise OSError("connection refused")
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    async def process_launcher(argv: tuple[str, ...]) -> object:
        launched.append(argv)
        return object()

    client = ManagedChromeDevToolsClient(
        debug_url="http://127.0.0.1:9333",
        workspace_root=tmp_path,
        browser_executable="/usr/bin/chromium",
        user_data_dir=tmp_path / "profile",
        proxy="http://127.0.0.1:7897",
        tab_discovery=tab_discovery,
        process_launcher=process_launcher,
        launch_timeout_seconds=0.5,
    )

    tabs = await client.list_tabs()

    assert tabs[0]["id"] == "tab-1"
    assert launched == [
        (
            "/usr/bin/chromium",
            "--remote-debugging-port=9333",
            f"--user-data-dir={tmp_path / 'profile'}",
            "--no-first-run",
            "--no-default-browser-check",
            "--noerrdialogs",
            "--disable-dev-shm-usage",
            "--window-size=1365,900",
            "--proxy-server=http://127.0.0.1:7897",
            "about:blank",
        )
    ]
    assert "--headless=new" not in launched[0]


async def test_managed_chrome_devtools_client_can_launch_headless(tmp_path) -> None:
    from src.agent_runtime.computer_use import ManagedChromeDevToolsClient

    launched: list[tuple[str, ...]] = []
    discovery_calls = 0

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        nonlocal discovery_calls
        discovery_calls += 1
        if discovery_calls == 1:
            raise OSError("connection refused")
        return (
            {
                "id": "tab-1",
                "title": "Example",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    async def process_launcher(argv: tuple[str, ...]) -> object:
        launched.append(argv)
        return object()

    client = ManagedChromeDevToolsClient(
        debug_url="http://127.0.0.1:9333",
        workspace_root=tmp_path,
        browser_executable="/usr/bin/chromium",
        user_data_dir=tmp_path / "profile",
        headless=True,
        tab_discovery=tab_discovery,
        process_launcher=process_launcher,
        launch_timeout_seconds=0.5,
    )

    tabs = await client.list_tabs()

    assert tabs[0]["id"] == "tab-1"
    assert "--headless=new" in launched[0]


async def test_managed_chrome_devtools_client_refuses_old_headless_endpoint(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import ManagedChromeDevToolsClient

    launched: list[tuple[str, ...]] = []

    async def tab_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "id": "tab-1",
                "title": "Headless",
                "url": "about:blank",
                "type": "page",
                "webSocketDebuggerUrl": "ws://chrome/tab-1",
            },
        )

    async def version_discovery(
        _debug_url: str,
        _timeout_seconds: float,
    ) -> dict[str, str]:
        return {
            "Browser": "Chrome/147.0.0.0",
            "User-Agent": "Mozilla/5.0 HeadlessChrome/147.0.0.0",
        }

    async def process_launcher(argv: tuple[str, ...]) -> object:
        launched.append(argv)
        return object()

    client = ManagedChromeDevToolsClient(
        debug_url="http://127.0.0.1:9333",
        workspace_root=tmp_path,
        browser_executable="/usr/bin/chromium",
        user_data_dir=tmp_path / "profile",
        headless=False,
        tab_discovery=tab_discovery,
        version_discovery=version_discovery,
        process_launcher=process_launcher,
        launch_timeout_seconds=0.5,
    )

    result = await client.execute_action(
        "browser_status",
        {"action": "browser_status"},
    )

    assert result.status == "degraded"
    assert "endpoint is headless" in result.message
    assert "visible live browser mode is required" in result.message
    assert launched == []


async def test_chrome_devtools_live_browser_adapter_uses_injected_cdp_client() -> None:
    from src.agent_runtime.computer_use import (
        ChromeDevToolsLiveBrowserAdapter,
        ComputerUseActionKind,
        ComputerUseActionRequest,
    )

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def list_tabs(self) -> tuple[dict[str, str], ...]:
            return (
                {
                    "id": "tab-1",
                    "title": "Example",
                    "url": "https://example.com",
                    "type": "page",
                    "webSocketDebuggerUrl": "ws://chrome/tab-1",
                },
            )

        async def execute_action(
            self,
            action: str,
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            self.calls.append((action, payload))
            return {"status": "completed", "action": action, "artifact": "shot.png"}

    client = FakeClient()
    adapter = ChromeDevToolsLiveBrowserAdapter(cdp_client=client)

    status = await adapter.status()
    result = await adapter.execute(
        ComputerUseActionRequest(
            action=ComputerUseActionKind.BROWSER_NAVIGATE,
            url="https://example.com/page",
        )
    )

    assert status.status == "completed"
    assert status.metadata["tab_count"] == "1"
    assert result.status == "completed"
    assert result.artifact == "shot.png"
    assert client.calls == [
        (
            "browser_navigate",
            {
                "action": "browser_navigate",
                "goal": "",
                "operation": "",
                "target": "",
                "text": "",
                "url": "https://example.com/page",
                "selector": "",
                "tab_id": "",
                "artifact_requirements": {},
                "metadata": {},
            },
        )
    ]


async def test_chrome_devtools_live_browser_adapter_reports_degraded_attach() -> None:
    from src.agent_runtime.computer_use import ChromeDevToolsLiveBrowserAdapter

    async def unavailable(_url: str, _timeout: float) -> tuple[dict[str, Any], ...]:
        raise OSError("connection refused")

    adapter = ChromeDevToolsLiveBrowserAdapter(
        debug_url="http://127.0.0.1:9222",
        tab_discovery=unavailable,
    )

    result = await adapter.status()

    assert result.status == "degraded"
    assert "remote debugging" in result.message


async def test_playwright_isolated_adapter_is_configured_only_without_backend() -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseActionKind,
        ComputerUseActionRequest,
        PlaywrightIsolatedBrowserAdapter,
    )

    adapter = PlaywrightIsolatedBrowserAdapter()

    result = await adapter.execute(
        ComputerUseActionRequest(action=ComputerUseActionKind.BROWSER_NAVIGATE)
    )

    assert result.status == "configured_only"
    assert "isolated" in result.message


async def test_hyprland_desktop_adapter_captures_screenshot_artifact(tmp_path) -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseActionKind,
        ComputerUseActionRequest,
        HyprlandDesktopComputerUseAdapter,
    )

    calls: list[tuple[str, ...]] = []

    async def runner(argv: tuple[str, ...]) -> str:
        calls.append(argv)
        return "screenshot saved"

    adapter = HyprlandDesktopComputerUseAdapter(
        workspace_root=tmp_path,
        runner=runner,
    )

    result = await adapter.execute(
        ComputerUseActionRequest(action=ComputerUseActionKind.DESKTOP_SCREENSHOT)
    )

    assert result.status == "completed"
    assert result.artifact.startswith("agent_runtime/computer_use/screenshots/")
    assert calls == [("grim", str(tmp_path / result.artifact))]


async def test_computer_use_gateway_exposes_reversible_tools_without_submit(
    tmp_path,
) -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        build_computer_use_tool_gateway,
    )
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI

    class FakeBrowser:
        async def status(self) -> ComputerUseActionResult:
            return ComputerUseActionResult(status="completed", message="ready")

        async def execute(self, _request: Any) -> ComputerUseActionResult:
            return ComputerUseActionResult(status="completed", message="ok")

    gateway = build_computer_use_tool_gateway(
        live_browser_adapter=FakeBrowser(),
        workspace_root=tmp_path,
    )

    toolset = gateway.build_toolset(COMPUTER_USE_ACTIVE_GUI)
    assert "browser_live_navigate" in toolset
    assert "browser_live_click" in toolset
    assert "browser_live_interactive_task" in toolset
    assert "computer_browser_submit" not in toolset
    assert "desktop_shell" not in toolset

    approval = AgentToolApproval.approved(
        approval_id="approval-submit",
        capabilities=("browser_submit",),
        approved_by=1291112109,
    )
    assert "computer_browser_submit" in gateway.build_toolset(
        COMPUTER_USE_ACTIVE_GUI,
        approval=approval,
    )


async def test_computer_use_worker_executes_approved_desktop_tool_gateway_action(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
        build_computer_use_tool_gateway,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeDesktop:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def execute(self, request: Any) -> ComputerUseActionResult:
            self.requests.append(request)
            return ComputerUseActionResult(status="completed", message="paused")

    desktop = FakeDesktop()
    gateway = build_computer_use_tool_gateway(
        desktop_adapter=desktop,
        workspace_root=tmp_path,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "computer_use": ComputerUseWorkerBackend(
                tool_gateway=gateway,
                workspace_root=tmp_path,
            )
        },
    )
    job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="computer:desktop:1",
        fingerprint="computer-desktop-media",
        kind="computer_use.action.desktop_media_control",
        profile=COMPUTER_USE_ACTIVE_GUI,
        context_pack=ContextPack(
            user_request="поставь музыку на паузу",
            metadata={
                "computer_use_payload": json.dumps(
                    {"action": "desktop_media_control", "operation": "pause"}
                ),
                "agent_tool_approval_id": "approval-computer-desktop",
                "agent_tool_approval_capabilities": "desktop_media_control",
            },
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.result is not None
    assert completed.result.summary == "Computer-use action completed."
    assert desktop.requests[0].action.value == "desktop_media_control"
    assert "desktop_media_control" in completed.result.processed_context


async def test_irreversible_detector_ignores_safety_policy_metadata() -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseActionKind,
        ComputerUseActionRequest,
        ComputerUseRiskClass,
        IrreversibleActionDetector,
    )

    detector = IrreversibleActionDetector()

    policy_decision = detector.inspect(
        ComputerUseActionRequest(
            action=ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
            text="пройди тест",
            metadata={
                "answer_policy": (
                    "ask_if_credentials_payment_private_data_or_real_identity_are_needed"
                ),
                "persona_context_ref": "workspace://personality/current-summary",
                "persona_context_mode": "reference_only",
            },
        )
    )
    payment_decision = detector.inspect(
        ComputerUseActionRequest(
            action=ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
            text="оплати форму",
        )
    )

    assert policy_decision.allowed is True
    assert policy_decision.hard_stop is False
    assert payment_decision.allowed is False
    assert payment_decision.requires_approval is True
    assert payment_decision.risk_class is ComputerUseRiskClass.ACCOUNT_MUTATION
    assert payment_decision.required_capability == "purchase"


async def test_irreversible_detector_treats_existing_session_password_negation_as_readonly() -> (
    None
):
    from src.agent_runtime.computer_use import (
        ComputerUseActionKind,
        ComputerUseActionRequest,
        ComputerUseRiskClass,
        IrreversibleActionDetector,
    )

    detector = IrreversibleActionDetector()

    decision = detector.inspect(
        ComputerUseActionRequest(
            action=ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
            goal=(
                "зайди в уже открытый Steam и найди kereexa SteamID без ввода пароля"
            ),
            text=(
                "через существующую сессию найди SteamID, без логина и без ввода пароля"
            ),
            risk_intent="readonly_existing_session",
            constraints=[
                "use_existing_session_only",
                "do_not_enter_credentials",
                "do_not_click_login",
            ],
        )
    )

    assert decision.allowed is True
    assert decision.requires_approval is False
    assert decision.risk_class is ComputerUseRiskClass.READONLY_EXISTING_SESSION


async def test_computer_use_worker_requires_scoped_approval_for_login_and_approved_passes(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
        FileComputerUseControlStateStore,
        build_computer_use_tool_gateway,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeBrowser:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def status(self) -> ComputerUseActionResult:
            return ComputerUseActionResult(status="completed", message="ready")

        async def execute(self, request: Any) -> ComputerUseActionResult:
            self.calls.append(request)
            return ComputerUseActionResult(status="completed", message="clicked")

    browser = FakeBrowser()
    control_state = FileComputerUseControlStateStore(tmp_path / "computer-state.json")
    gateway = build_computer_use_tool_gateway(
        live_browser_adapter=browser,
        workspace_root=tmp_path,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "computer_use": ComputerUseWorkerBackend(
                tool_gateway=gateway,
                workspace_root=tmp_path,
                control_state=control_state,
            )
        },
    )
    job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="computer:1",
        fingerprint="computer-login-approval-required",
        kind="computer_use.action",
        profile=COMPUTER_USE_ACTIVE_GUI,
        context_pack=ContextPack(
            user_request="введи пароль в поле password",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "browser_type",
                        "target": "password",
                        "selector": "#password",
                        "text": "secret",
                    }
                )
            },
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.result is not None
    assert completed.result.summary == "Computer-use action requires approval."
    assert "dangerous_action_requires_approval" in completed.result.processed_context
    assert "login" in completed.result.processed_context
    assert browser.calls == []

    approved_job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="computer:2",
        fingerprint="computer-login-approved",
        kind="computer_use.action",
        profile=COMPUTER_USE_ACTIVE_GUI,
        context_pack=ContextPack(
            user_request="введи пароль в поле password",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "browser_type",
                        "target": "password",
                        "selector": "#password",
                        "text": "secret",
                    }
                ),
                "agent_tool_approval_id": "approval-login",
                "agent_tool_approval_capabilities": "login",
            },
        ),
    )

    approved_completed = await runtime.start(approved_job.id)

    assert approved_completed.result is not None
    assert approved_completed.result.summary == "Computer-use action completed."
    assert browser.calls[0].action.value == "browser_type"


async def test_computer_use_worker_refuses_wrong_scoped_approval(tmp_path) -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
        build_computer_use_tool_gateway,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeBrowser:
        async def status(self) -> ComputerUseActionResult:
            return ComputerUseActionResult(status="completed", message="ready")

        async def execute(self, _request: Any) -> ComputerUseActionResult:
            raise AssertionError("wrong approval must not execute")

    gateway = build_computer_use_tool_gateway(
        live_browser_adapter=FakeBrowser(),
        workspace_root=tmp_path,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "computer_use": ComputerUseWorkerBackend(
                tool_gateway=gateway,
                workspace_root=tmp_path,
            )
        },
    )
    job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="computer:wrong-approval",
        fingerprint="computer-wrong-approval",
        kind="computer_use.action",
        profile=COMPUTER_USE_ACTIVE_GUI,
        context_pack=ContextPack(
            user_request="введи пароль в поле password",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "browser_type",
                        "target": "password",
                        "selector": "#password",
                        "text": "secret",
                    }
                ),
                "agent_tool_approval_id": "approval-submit-only",
                "agent_tool_approval_capabilities": "browser_submit",
            },
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.result is not None
    assert completed.result.summary == "Computer-use action refused."
    assert "requires login approval" in completed.result.findings[0].claim


async def test_computer_use_worker_executes_approved_structured_shell_command(
    tmp_path,
) -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseWorkerBackend,
        build_computer_use_tool_gateway,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_APPROVED_SHELL
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    calls: list[tuple[str, ...]] = []

    async def runner(
        argv: tuple[str, ...],
        *,
        cwd: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        calls.append(argv)
        return {
            "argv": list(argv),
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "exit_code": 0,
            "stdout": "ok TOKEN=redacted",
            "stderr": "",
        }

    gateway = build_computer_use_tool_gateway(
        workspace_root=tmp_path,
        shell_runner=runner,
        shell_allowed_executables=("echo",),
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "computer_use": ComputerUseWorkerBackend(
                tool_gateway=gateway,
                workspace_root=tmp_path,
            )
        },
    )
    job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="computer:shell",
        fingerprint="computer-shell",
        kind="computer_use.action.desktop_shell_command",
        profile=COMPUTER_USE_APPROVED_SHELL,
        context_pack=ContextPack(
            user_request="выполни shell команду",
            metadata={
                "computer_use_payload": json.dumps(
                    {
                        "action": "desktop_shell_command",
                        "argv": ["echo", "ok"],
                        "cwd": ".",
                        "timeout_seconds": 3,
                    }
                ),
                "agent_tool_approval_id": "approval-shell",
                "agent_tool_approval_capabilities": "desktop.shell",
            },
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.result is not None
    assert completed.result.summary == "Computer-use action completed."
    assert calls == [("echo", "ok")]
    assert "TOKEN=redacted" not in completed.result.processed_context


async def test_computer_use_worker_refuses_when_paused(tmp_path) -> None:
    from src.agent_runtime.computer_use import (
        ComputerUseActionResult,
        ComputerUseWorkerBackend,
        FileComputerUseControlStateStore,
        build_computer_use_tool_gateway,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import COMPUTER_USE_ACTIVE_GUI
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeBrowser:
        async def status(self) -> ComputerUseActionResult:
            return ComputerUseActionResult(status="completed", message="ready")

        async def execute(self, _request: Any) -> ComputerUseActionResult:
            raise AssertionError("paused computer-use must not execute")

    control_state = FileComputerUseControlStateStore(tmp_path / "computer-state.json")
    control_state.pause(reason="manual pause")
    gateway = build_computer_use_tool_gateway(
        live_browser_adapter=FakeBrowser(),
        workspace_root=tmp_path,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "computer_use": ComputerUseWorkerBackend(
                tool_gateway=gateway,
                workspace_root=tmp_path,
                control_state=control_state,
            )
        },
    )
    job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="computer:2",
        fingerprint="computer-paused",
        kind="computer_use.action",
        profile=COMPUTER_USE_ACTIVE_GUI,
        context_pack=ContextPack(
            user_request="открой сайт",
            metadata={
                "computer_use_payload": json.dumps(
                    {"action": "browser_navigate", "url": "https://example.com"}
                )
            },
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.result is not None
    assert completed.result.summary == "Computer-use action refused."
    assert "paused" in completed.result.findings[0].claim
