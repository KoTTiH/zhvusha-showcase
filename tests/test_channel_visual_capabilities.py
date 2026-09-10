"""Capability reporting for channel visual artifacts."""

from __future__ import annotations

from src.core.capabilities import (
    CapabilityRegistry,
    register_agent_runtime_capabilities,
)


def test_channel_visual_capability_reports_disabled_generation() -> None:
    registry = CapabilityRegistry()

    register_agent_runtime_capabilities(
        registry,
        enable_browser_use=False,
        image_generation_enabled=False,
    )
    unavailable = {cap.name: cap for cap in registry.get_unavailable()}

    assert "готовить визуалы для постов канала" in unavailable
    assert unavailable["готовить визуалы для постов канала"].tool == (
        "agent_runtime.channel_visual"
    )


def test_channel_visual_capability_reports_enabled_generation() -> None:
    registry = CapabilityRegistry()

    register_agent_runtime_capabilities(
        registry,
        enable_browser_use=False,
        image_generation_enabled=True,
    )
    available = {cap.name: cap for cap in registry.get_available()}

    assert "готовить визуалы для постов канала" in available
