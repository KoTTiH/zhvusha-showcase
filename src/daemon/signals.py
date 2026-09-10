"""Signal model for the daemon's event-driven architecture."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

SignalPriority = Literal["critical", "normal", "background"]


@dataclass
class Signal:
    """Single event from any source, routed through Redis Streams."""

    id: str = field(default_factory=lambda: str(uuid4()))
    source: str = ""
    priority: SignalPriority = "normal"
    signal_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    requires_response: bool = False
    ttl_minutes: int | None = None
    # Set by SignalStream after reading from Redis; used for XACK
    stream_entry_id: bytes | str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, str]:
        """Serialize to flat string dict for Redis XADD."""
        return {
            "id": self.id,
            "source": self.source,
            "priority": self.priority,
            "signal_type": self.signal_type,
            "payload": json.dumps(self.payload, ensure_ascii=False),
            "timestamp": self.timestamp.isoformat(),
            "requires_response": str(self.requires_response),
            "ttl_minutes": str(self.ttl_minutes)
            if self.ttl_minutes is not None
            else "",
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Signal:
        """Deserialize from Redis XREADGROUP result."""
        return cls(
            id=data.get("id", ""),
            source=data.get("source", ""),
            priority=data.get("priority", "normal"),  # type: ignore[arg-type]
            signal_type=data.get("signal_type", ""),
            payload=json.loads(data.get("payload", "{}")),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(tz=UTC),
            requires_response=data.get("requires_response", "False") == "True",
            ttl_minutes=int(data["ttl_minutes"]) if data.get("ttl_minutes") else None,
        )
