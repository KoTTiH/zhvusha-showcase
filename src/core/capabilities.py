"""Deprecated legacy capability registry.

This registry is kept only for old tests and non-chat compatibility code.
User-facing personal capability truth must come from
``src.agent_runtime.capability_graph`` manager summaries, not this prompt-era
registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Capability:
    """Single capability entry."""

    name: str
    tool: str
    description: str = ""
    status: Literal["available", "unavailable"] = "available"
    reason: str = ""
    workaround: str = ""


class CapabilityRegistry:
    """Registry of what Zhvusha can and cannot do."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        """Register a capability."""
        self._capabilities[capability.name] = capability

    def unregister(self, name: str) -> None:
        """Remove a capability."""
        self._capabilities.pop(name, None)

    def get_available(self) -> list[Capability]:
        """Return all available capabilities."""
        return [c for c in self._capabilities.values() if c.status == "available"]

    def get_unavailable(self) -> list[Capability]:
        """Return all unavailable capabilities."""
        return [c for c in self._capabilities.values() if c.status == "unavailable"]

    def can_do(self, action: str) -> tuple[bool, str]:
        """Check if an action is possible. Returns (can_do, explanation)."""
        action_lower = action.lower()
        for cap in self._capabilities.values():
            if (
                action_lower in cap.name.lower()
                or action_lower in cap.description.lower()
            ):
                if cap.status == "available":
                    return True, f"Могу: {cap.name} (инструмент: {cap.tool})"
                return False, f"Не могу: {cap.name}. Причина: {cap.reason}"
        return False, "Не знаю такой возможности."

    def format_for_prompt(self) -> str:
        """Format capabilities for inclusion in system prompt (Russian)."""
        available = self.get_available()
        unavailable = self.get_unavailable()

        lines: list[str] = []

        if available:
            lines.append("## Что я умею прямо сейчас")
            for cap in available:
                lines.append(f"- {cap.name} ({cap.tool})")

        if unavailable:
            lines.append("")
            lines.append("## Чего пока не умею")
            for cap in unavailable:
                line = f"- {cap.name}: {cap.reason}"
                if cap.workaround:
                    line += f" → {cap.workaround}"
                lines.append(line)

        return "\n".join(lines)


# Default capabilities that don't come from skills
_DEFAULT_CAPABILITIES: list[Capability] = [
    Capability(
        name="автономно работать в фоне",
        tool="daemon",
        description="Фоновый агент: knowledge_store, workspace_read, send_telegram",
        status="available",
    ),
    Capability(
        name="зайти на сайт",
        tool="browser_use",
        status="unavailable",
        reason="Browser Use ещё не подключён",
        workaround="могу попросить тебя проверить",
    ),
    Capability(
        name="написать и запустить код",
        tool="code_executor",
        status="unavailable",
        reason="Code Executor ещё не подключён",
        workaround="могу подготовить задачу для Codex",
    ),
]


def get_default_registry() -> CapabilityRegistry:
    """Create a registry with default capabilities."""
    registry = CapabilityRegistry()
    for cap in _DEFAULT_CAPABILITIES:
        registry.register(cap)
    return registry


def register_agent_runtime_capabilities(
    registry: CapabilityRegistry,
    *,
    enable_browser_use: bool,
    image_generation_enabled: bool = False,
    computer_use_enabled: bool = False,
    live_browser_control_enabled: bool = False,
) -> None:
    """Register dynamic Agent Runtime capabilities for the current process."""
    live_browser_available = computer_use_enabled and live_browser_control_enabled
    if enable_browser_use or live_browser_available:
        website_tool = (
            "agent_runtime.computer_use"
            if live_browser_available
            else "agent_runtime.web_research"
        )
        website_description = (
            "Открыть сайт в живом браузере через Agent Runtime; "
            "submit форм остаётся под approval/hard-stop policy."
            if live_browser_available
            else (
                "Открыть и прочитать публичную страницу read-only, "
                "сделать screenshot artifact; без login/form submit."
            )
        )
        registry.register(
            Capability(
                name="зайти на сайт",
                tool=website_tool,
                description=website_description,
                status="available",
            )
        )
    if enable_browser_use:
        registry.register(
            Capability(
                name="читать web-источники read-only",
                tool="agent_runtime.web_research",
                description=(
                    "Чтение URL, bounded download artifacts и headless screenshots "
                    "без login/form submit."
                ),
                status="available",
            )
        )
    if not computer_use_enabled:
        live_browser_reason = "COMPUTER_USE_ENABLED выключен"
    elif not live_browser_control_enabled:
        live_browser_reason = "LIVE_BROWSER_CONTROL_ENABLED выключен"
    else:
        live_browser_reason = ""
    registry.register(
        Capability(
            name="интерактивно работать с живым браузером",
            tool="agent_runtime.computer_use",
            description=(
                "Открыть сайт в живом браузере, проверять статус, кликать, "
                "печатать и снимать desktop screenshots через Agent Runtime; "
                "submit форм остаётся под approval/hard-stop policy."
            ),
            status="available" if live_browser_available else "unavailable",
            reason=live_browser_reason,
            workaround=""
            if live_browser_available
            else "используй read-only web research либо включи computer-use/live browser",
        )
    )
    visual_available = enable_browser_use or image_generation_enabled
    registry.register(
        Capability(
            name="готовить визуалы для постов канала",
            tool="agent_runtime.channel_visual",
            description=(
                "Планирование visual metadata, source screenshots/images и "
                "generated artifacts для approved channel drafts."
            ),
            status="available" if visual_available else "unavailable",
            reason=""
            if visual_available
            else "image generation и browser artifacts выключены",
            workaround="сохраню text-only черновик и явно покажу причину деградации",
        )
    )
