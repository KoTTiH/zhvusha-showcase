from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.dialogue.decisions import (
    DecisionResolution,
    FilePendingDecisionStore,
    PendingDecision,
    resolution_from_approval_signal,
    should_defer_to_cognitive_loop,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_pending_decision_serializes_structured_proposal() -> None:
    decision = PendingDecision(
        decision_id="decision-1",
        kind="external_action",
        owner="telegram_mcp_personal",
        action="telegram_send",
        summary="Отправить сообщение Тоше",
        proposal={
            "recipient_hint": "Тоше",
            "draft_message": "расскумаримся в дотке сегодня",
        },
        missing_fields=("chat_id",),
        constraints=("executable_chat_id_required",),
    )

    dumped = decision.model_dump(mode="json")

    assert dumped["kind"] == "external_action"
    assert dumped["proposal"]["recipient_hint"] == "Тоше"
    assert dumped["missing_fields"] == ["chat_id"]
    assert "approve" in dumped["allowed_outcomes"]
    assert "ask_more" in dumped["allowed_outcomes"]


def test_conditional_reply_defers_to_cognitive_loop() -> None:
    assert should_defer_to_cognitive_loop("да, но мягче") is True
    assert should_defer_to_cognitive_loop("можно, но сначала покажи текст") is True
    assert should_defer_to_cognitive_loop("не Тоше, а Сане") is True
    assert should_defer_to_cognitive_loop("да") is False


def test_approval_signal_maps_to_decision_resolution() -> None:
    pending = PendingDecision(
        decision_id="decision-1",
        kind="external_action",
        owner="required_skill",
        action="post",
        summary="Опасное действие",
    )

    approved = resolution_from_approval_signal("yes", pending)
    deferred = resolution_from_approval_signal("later", pending)
    unclear = resolution_from_approval_signal("ambiguous", pending)

    assert isinstance(approved, DecisionResolution)
    assert approved.outcome == "approve"
    assert deferred.outcome == "defer"
    assert unclear.outcome == "ask_more"


def test_file_pending_decision_store_survives_restart_and_resolves(
    tmp_path: Path,
) -> None:
    store = FilePendingDecisionStore(tmp_path)
    decision = PendingDecision(
        decision_id="decision-telegram-send",
        kind="external_action",
        owner="telegram_mcp_personal",
        action="telegram_send",
        summary="Отправить сообщение Тоше",
        proposal={"recipient_hint": "Тоше", "draft_message": "мягче"},
        missing_fields=("chat_id",),
        constraints=("executable_chat_id_required",),
    )

    store.save("123", decision)
    reloaded = FilePendingDecisionStore(tmp_path)

    assert reloaded.get("123", decision.decision_id) == decision
    assert [item.decision_id for item in reloaded.list_pending("123")] == [
        decision.decision_id
    ]

    resolution = DecisionResolution(
        decision_id=decision.decision_id,
        outcome="ask_more",
        reason="Нужен явный @username/id.",
        missing_fields=("chat_id",),
        user_message="пиши Тоше",
        confidence=0.92,
        source="cognitive_loop",
    )

    resolved = reloaded.resolve("123", resolution)

    assert resolved == decision
    assert reloaded.get("123", decision.decision_id) is None
    assert reloaded.list_pending("123") == ()
    resolved_path = (
        tmp_path
        / "logs"
        / "123"
        / "pending_decisions"
        / "resolved"
        / "decision-telegram-send.json"
    )
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert payload["decision"]["decision_id"] == decision.decision_id
    assert payload["resolution"]["outcome"] == "ask_more"
