"""Bridge from KworkMonitorSkill to daemon signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.daemon.signals import Signal

if TYPE_CHECKING:
    from src.daemon.stream import SignalStream


async def push_kwork_project(
    stream: SignalStream,
    *,
    project_id: int,
    title: str,
    budget: int,
    details: dict[str, Any] | None = None,
) -> None:
    """Push a new Kwork project as a normal-priority signal."""
    signal = Signal(
        source="kwork_monitor",
        priority="normal",
        signal_type="new_project",
        payload={
            "project_id": project_id,
            "title": title,
            "budget": budget,
            **(details or {}),
        },
        ttl_minutes=60,
    )
    await stream.push(signal)
