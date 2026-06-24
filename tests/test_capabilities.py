"""Tests for CapabilityRegistry."""

from __future__ import annotations

from src.core.capabilities import (
    Capability,
    CapabilityRegistry,
    get_default_registry,
)


class TestCapability:
    def test_create_available(self) -> None:
        cap = Capability(name="test", tool="tool", description="desc")
        assert cap.status == "available"
        assert cap.reason == ""

    def test_create_unavailable(self) -> None:
        cap = Capability(
            name="test",
            tool="tool",
            status="unavailable",
            reason="not ready",
            workaround="use X instead",
        )
        assert cap.status == "unavailable"
        assert cap.workaround == "use X instead"


class TestCapabilityRegistry:
    def test_register_and_get(self) -> None:
        reg = CapabilityRegistry()
        cap = Capability(name="search", tool="episodic_memory")
        reg.register(cap)

        available = reg.get_available()
        assert len(available) == 1
        assert available[0].name == "search"

    def test_unregister(self) -> None:
        reg = CapabilityRegistry()
        reg.register(Capability(name="search", tool="mem"))
        reg.unregister("search")

        assert len(reg.get_available()) == 0

    def test_unregister_nonexistent(self) -> None:
        reg = CapabilityRegistry()
        reg.unregister("ghost")  # should not raise
        assert "ghost" not in reg.get_available()

    def test_available_and_unavailable(self) -> None:
        reg = CapabilityRegistry()
        reg.register(Capability(name="chat", tool="chat_skill"))
        reg.register(
            Capability(
                name="browse",
                tool="browser",
                status="unavailable",
                reason="not implemented",
            )
        )

        assert len(reg.get_available()) == 1
        assert len(reg.get_unavailable()) == 1

    def test_can_do_available(self) -> None:
        reg = CapabilityRegistry()
        reg.register(
            Capability(name="ответить на вопрос", tool="chat", description="отвечает")
        )

        can, explanation = reg.can_do("ответить")
        assert can is True
        assert "Могу" in explanation

    def test_can_do_unavailable(self) -> None:
        reg = CapabilityRegistry()
        reg.register(
            Capability(
                name="зайти на сайт",
                tool="browser",
                status="unavailable",
                reason="не подключён",
            )
        )

        can, explanation = reg.can_do("сайт")
        assert can is False
        assert "Не могу" in explanation

    def test_can_do_unknown(self) -> None:
        reg = CapabilityRegistry()
        can, explanation = reg.can_do("fly to moon")
        assert can is False
        assert "Не знаю" in explanation

    def test_format_for_prompt_available(self) -> None:
        reg = CapabilityRegistry()
        reg.register(Capability(name="chat", tool="chat_skill"))

        prompt = reg.format_for_prompt()
        assert "Что я умею" in prompt
        assert "chat" in prompt

    def test_format_for_prompt_unavailable(self) -> None:
        reg = CapabilityRegistry()
        reg.register(
            Capability(
                name="browse",
                tool="browser",
                status="unavailable",
                reason="не подключён",
                workaround="попроси Никиту",
            )
        )

        prompt = reg.format_for_prompt()
        assert "Чего пока не умею" in prompt
        assert "попроси Никиту" in prompt

    def test_format_for_prompt_empty(self) -> None:
        reg = CapabilityRegistry()
        prompt = reg.format_for_prompt()
        assert prompt == ""

    def test_overwrite_capability(self) -> None:
        reg = CapabilityRegistry()
        reg.register(Capability(name="x", tool="a"))
        reg.register(Capability(name="x", tool="b"))

        available = reg.get_available()
        assert len(available) == 1
        assert available[0].tool == "b"


class TestDefaultRegistry:
    def test_has_default_unavailable(self) -> None:
        reg = get_default_registry()
        unavailable = reg.get_unavailable()
        names = {c.name for c in unavailable}
        assert "зайти на сайт" in names

    def test_daemon_is_available(self) -> None:
        reg = get_default_registry()
        available = reg.get_available()
        names = {c.name for c in available}
        assert "автономно работать в фоне" in names

    def test_default_capabilities_do_not_suggest_claude_code(self) -> None:
        reg = get_default_registry()
        formatted = reg.format_for_prompt()
        assert "Claude Code" not in formatted
        assert "Codex" in formatted

    def test_agent_runtime_browser_capability_registered_when_enabled(self) -> None:
        from src.core.capabilities import register_agent_runtime_capabilities

        reg = get_default_registry()
        register_agent_runtime_capabilities(reg, enable_browser_use=True)

        available = {cap.name: cap for cap in reg.get_available()}
        assert "читать web-источники read-only" in available
        assert available["читать web-источники read-only"].tool == (
            "agent_runtime.web_research"
        )
        can, explanation = reg.can_do("web-источники")
        assert can is True
        assert "agent_runtime.web_research" in explanation

    def test_agent_runtime_live_browser_capability_reports_disabled_flags(self) -> None:
        from src.core.capabilities import register_agent_runtime_capabilities

        reg = get_default_registry()
        register_agent_runtime_capabilities(
            reg,
            enable_browser_use=True,
            computer_use_enabled=False,
            live_browser_control_enabled=False,
        )

        unavailable = {cap.name: cap for cap in reg.get_unavailable()}
        assert "интерактивно работать с живым браузером" in unavailable
        assert unavailable["интерактивно работать с живым браузером"].reason == (
            "COMPUTER_USE_ENABLED выключен"
        )

    def test_agent_runtime_live_browser_capability_registered_when_enabled(
        self,
    ) -> None:
        from src.core.capabilities import register_agent_runtime_capabilities

        reg = get_default_registry()
        register_agent_runtime_capabilities(
            reg,
            enable_browser_use=True,
            computer_use_enabled=True,
            live_browser_control_enabled=True,
        )

        available = {cap.name: cap for cap in reg.get_available()}
        assert "интерактивно работать с живым браузером" in available
        assert available["интерактивно работать с живым браузером"].tool == (
            "agent_runtime.computer_use"
        )
        can, explanation = reg.can_do("живым браузером")
        assert can is True
        assert "agent_runtime.computer_use" in explanation
