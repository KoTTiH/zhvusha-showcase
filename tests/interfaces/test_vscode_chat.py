from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.skills.base import AgentContext


async def test_vscode_chat_bridge_logs_codex_and_passes_shared_context(
    tmp_path: Path,
) -> None:
    from src.interfaces.vscode_chat import VSCODE_CHAT_LOG_ID, VscodeChatBridge

    project_root = tmp_path / "project"
    project_root.mkdir()
    seen_contexts: list[AgentContext] = []

    async def processor(text: str, context: AgentContext) -> str:
        seen_contexts.append(context)
        return f"ответ на: {text}"

    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )

    response = await bridge.send_message(
        text="проверь проект",
        sender="codex",
        workspace_context={
            "workspacePath": str(project_root),
            "digitalScenarioEvalVariant": "paraphrase",
            "digitalScenarioId": "ai_cto_projects",
            "digitalScenarioEvalRunId": "run-42",
        },
    )

    assert response["text"] == "ответ на: проверь проект"
    assert seen_contexts
    context = seen_contexts[0]
    assert context.user_id == 12345
    assert context.chat_id == -7331
    assert context.message_id == 1
    assert context.mode == "personal"
    assert context.metadata["source"] == "vscode"
    assert context.metadata["source_actor"] == "codex"
    assert context.metadata["chat_log_id"] == VSCODE_CHAT_LOG_ID
    assert "технического тела Жвуши" in context.metadata["interface_context"]
    assert context.metadata["return_response_text"] is True
    assert context.metadata["skip_response_log"] is True
    assert context.metadata["project_root"] == str(project_root.resolve())
    assert str(project_root.resolve()) in context.metadata["interface_context"]
    assert context.metadata["digital_scenario_eval_variant"] == "paraphrase"
    assert context.metadata["digital_scenario_id"] == "ai_cto_projects"
    assert context.metadata["digital_scenario_eval_run_id"] == "run-42"
    assert "workspace_context" not in context.metadata

    log_file = next((tmp_path / "logs" / VSCODE_CHAT_LOG_ID).glob("chat_*.jsonl"))
    entries = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["role"] for entry in entries] == ["user", "assistant"]
    assert entries[0]["source_actor"] == "codex"
    assert entries[0]["message_id"] == 1
    assert entries[0]["codex"] is True
    assert entries[0]["author_label"] == "Codex"
    assert entries[0]["digital_scenario_eval_variant"] == "paraphrase"
    assert entries[0]["digital_scenario_id"] == "ai_cto_projects"
    assert entries[0]["digital_scenario_eval_run_id"] == "run-42"
    assert entries[1]["source"] == "vscode"
    assert entries[1]["author_label"] == "Жвуша"
    assert entries[1]["reply_to_message_id"] == 1

    history = bridge.list_messages(limit=10)
    assert len(history) == 2
    assert history[0]["source_actor"] == "codex"


async def test_vscode_chat_bridge_ignores_unknown_eval_metadata(
    tmp_path: Path,
) -> None:
    from src.interfaces.vscode_chat import VscodeChatBridge

    seen_contexts: list[AgentContext] = []

    async def processor(text: str, context: AgentContext) -> str:
        del text
        seen_contexts.append(context)
        return "ok"

    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )

    await bridge.send_message(
        text="проверь матрицу",
        sender="codex",
        workspace_context={
            "digitalScenarioEvalVariant": "unknown_variant",
            "digitalScenarioId": "unknown_scenario",
        },
    )

    assert seen_contexts
    metadata = seen_contexts[0].metadata
    assert "digital_scenario_eval_variant" not in metadata
    assert "digital_scenario_id" not in metadata


async def test_vscode_chat_bridge_rejects_empty_messages(tmp_path: Path) -> None:
    from src.interfaces.vscode_chat import VscodeChatBridge

    async def processor(text: str, context: AgentContext) -> str | None:
        raise AssertionError("processor must not run for empty input")

    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )

    response = await bridge.send_message(text="   ", sender="codex")

    assert response["text"] == ""
    assert response["warnings"] == ["empty_message"]


async def test_vscode_chat_bridge_enqueue_returns_before_reply(
    tmp_path: Path,
) -> None:
    from src.interfaces.vscode_chat import VSCODE_CHAT_LOG_ID, VscodeChatBridge

    release_reply = asyncio.Event()

    async def processor(text: str, context: AgentContext) -> str:
        await release_reply.wait()
        return f"готово: {text}"

    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )

    response = await bridge.enqueue_message(text="долгая задача", sender="codex")

    assert response["text"] == ""
    assert response["warnings"] == ["response_pending"]
    log_file = next((tmp_path / "logs" / VSCODE_CHAT_LOG_ID).glob("chat_*.jsonl"))
    entries = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["role"] for entry in entries] == ["user"]
    assert entries[0]["source_actor"] == "codex"

    release_reply.set()
    await bridge.drain_pending()

    entries = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["role"] for entry in entries] == ["user", "assistant"]
    assert entries[1]["text"] == "готово: долгая задача"


async def test_vscode_chat_http_requires_token(tmp_path: Path) -> None:
    from src.interfaces.vscode_chat import VscodeChatBridge, VscodeChatHttpServer

    seen: list[str] = []

    async def processor(text: str, context: AgentContext) -> str:
        del context
        seen.append(text)
        return "ok"

    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )
    server = VscodeChatHttpServer(bridge=bridge, auth_token="secret-token")
    body = json.dumps({"text": "пинг", "sender": "codex"}).encode("utf-8")

    denied_status, denied_payload = await server._handle_request(
        "POST",
        "/message",
        body,
        {},
    )
    allowed_status, allowed_payload = await server._handle_request(
        "POST",
        "/message",
        body,
        {"authorization": "Bearer secret-token"},
    )
    await bridge.drain_pending()

    assert denied_status == 401
    assert denied_payload == {"error": "unauthorized"}
    assert allowed_status == 202
    assert allowed_payload["warnings"] == ["response_pending"]
    assert seen == ["пинг"]


async def test_vscode_chat_http_allows_readonly_history_without_token(
    tmp_path: Path,
) -> None:
    from src.interfaces.vscode_chat import VscodeChatBridge, VscodeChatHttpServer

    async def processor(text: str, context: AgentContext) -> str:
        del context
        return f"ответ: {text}"

    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )
    await bridge.send_message(text="покажи историю", sender="codex")
    server = VscodeChatHttpServer(bridge=bridge, auth_token="secret-token")

    status, payload = await server._handle_request(
        "GET",
        "/messages?limit=80",
        b"",
        {},
    )

    assert status == 200
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "покажи историю"
    assert messages[1]["text"] == "ответ: покажи историю"
    assert messages[0]["ts"].endswith("+03:00")
    assert messages[0]["ts_utc"].endswith("+00:00")
    assert messages[1]["ts"].endswith("+03:00")
    assert messages[1]["ts_utc"].endswith("+00:00")


async def test_vscode_chat_history_converts_utc_ts_for_display(
    tmp_path: Path,
) -> None:
    from src.interfaces.vscode_chat import VSCODE_CHAT_LOG_ID, VscodeChatBridge

    async def processor(text: str, context: AgentContext) -> str:
        raise AssertionError("processor must not run for history read")

    chat_dir = tmp_path / "logs" / VSCODE_CHAT_LOG_ID
    chat_dir.mkdir(parents=True)
    (chat_dir / "chat_2026-05-21.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-05-21T14:33:33.925592+00:00",
                "role": "user",
                "source": "vscode",
                "text": "проверка времени",
                "chat_id": VSCODE_CHAT_LOG_ID,
                "mode": "personal",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )

    messages = bridge.list_messages(limit=1)

    assert messages[0]["ts"] == "2026-05-21T17:33:33.925592+03:00"
    assert messages[0]["ts_utc"] == "2026-05-21T14:33:33.925592+00:00"
    assert messages[0]["id"] == "2026-05-21T14:33:33.925592+00:00:user:0"


async def test_vscode_chat_history_tails_large_logs_without_read_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.interfaces.vscode_chat import VSCODE_CHAT_LOG_ID, VscodeChatBridge

    async def processor(text: str, context: AgentContext) -> str:
        raise AssertionError("processor must not run for history read")

    chat_dir = tmp_path / "logs" / VSCODE_CHAT_LOG_ID
    chat_dir.mkdir(parents=True)
    log_file = chat_dir / "chat_2026-06-02.jsonl"
    old_payload = "x" * (70 * 1024)
    entries = [
        {
            "ts": "2026-06-02T07:00:00+00:00",
            "role": "user",
            "source": "vscode",
            "text": old_payload,
            "chat_id": VSCODE_CHAT_LOG_ID,
            "mode": "personal",
        },
        {
            "ts": "2026-06-02T07:01:00+00:00",
            "role": "user",
            "source": "vscode",
            "text": "recent-1",
            "chat_id": VSCODE_CHAT_LOG_ID,
            "mode": "personal",
        },
        {
            "ts": "2026-06-02T07:02:00+00:00",
            "role": "assistant",
            "source": "vscode",
            "text": "recent-2",
            "chat_id": VSCODE_CHAT_LOG_ID,
            "mode": "personal",
        },
    ]
    log_file.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )

    def fail_read_text(*args: object, **kwargs: object) -> str:
        raise AssertionError("list_messages must tail chat logs, not read them fully")

    monkeypatch.setattr(type(log_file), "read_text", fail_read_text)
    bridge = VscodeChatBridge(
        workspace_root=tmp_path,
        admin_user_id=12345,
        processor=processor,
    )

    messages = bridge.list_messages(limit=2)

    assert [message["text"] for message in messages] == ["recent-1", "recent-2"]


async def test_vscode_chat_http_rejects_nonlocal_browser_origin(
    tmp_path: Path,
) -> None:
    from src.interfaces.vscode_chat import VscodeChatBridge, VscodeChatHttpServer

    async def processor(text: str, context: AgentContext) -> str:
        raise AssertionError("processor must not run for rejected origin")

    server = VscodeChatHttpServer(
        bridge=VscodeChatBridge(
            workspace_root=tmp_path,
            admin_user_id=12345,
            processor=processor,
        ),
        auth_token="secret-token",
    )
    body = json.dumps({"text": "пинг", "sender": "codex"}).encode("utf-8")

    status, payload = await server._handle_request(
        "POST",
        "/message",
        body,
        {
            "authorization": "Bearer secret-token",
            "origin": "https://example.com",
        },
    )

    assert status == 403
    assert payload == {"error": "forbidden_origin"}
