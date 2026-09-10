"""Topic backlog signals for Agent Runtime / daemon planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TopicSignalRoute = Literal["spec", "proposal", "post", "report"]


class TopicClusterReadySignal(BaseModel):
    """Structured topic backlog signal, not an execution command."""

    model_config = ConfigDict(extra="ignore")

    signal_type: Literal["topic_cluster_ready"] = "topic_cluster_ready"
    cluster_key: str
    title: str
    summary: str
    final_priority: float
    recommended_route: TopicSignalRoute
    tier: int = Field(ge=1, le=3)
    requires_approval: bool = True
    requires_nikita: bool = False
    auto_publish_allowed: bool = False
    auto_execute_allowed: bool = False
    safety_notes: str = ""
    payload: dict[str, str] = Field(default_factory=dict)

    @field_validator("cluster_key", "title", "summary", "safety_notes", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()
