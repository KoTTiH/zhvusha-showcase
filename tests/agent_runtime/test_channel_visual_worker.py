"""Channel visual worker stays behind the Tool Gateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from src.agent_runtime.image_artifacts import (
    ChannelVisualImageTool,
    ChannelVisualLocalCardTool,
)
from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextPack
from src.agent_runtime.profiles import CHANNEL_VISUAL_READONLY
from src.agent_runtime.tools import FunctionAgentTool, ToolGateway
from src.agent_runtime.workers.channel_visual import ChannelVisualWorkerBackend
from src.llm.protocols import LLMGatewayProtocol, LLMImageRequest, LLMImageResponse


async def test_channel_visual_worker_uses_gateway_for_generated_artifacts() -> None:
    calls: list[dict[str, Any]] = []

    async def generate(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {
            "status": "ready",
            "asset_path": "agent_runtime/channel_visual_artifacts/generated.png",
            "caption": payload["caption"],
        }

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "channel_visual_generate_image",
                "channel_visual_image_generation",
                generate,
            ),
        )
    )
    worker = ChannelVisualWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=1,
        source_message_id="1",
        fingerprint="visual",
        kind="channel_visual",
        profile=CHANNEL_VISUAL_READONLY,
        context_pack=ContextPack(
            user_request="Жвуша self-coding architecture",
            chat_context=("Как Agent Runtime объясняет работу Жвуши.",),
        ),
        status=AgentJobStatus.QUEUED,
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert calls
    assert capsule.artifacts == (
        "agent_runtime/channel_visual_artifacts/generated.png",
    )
    assert "generated" in capsule.markdown_report


async def test_channel_visual_worker_enriches_architecture_prompt_from_workspace() -> (
    None
):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def generate(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(("generate", payload))
        return {
            "status": "ready",
            "asset_path": "agent_runtime/channel_visual_artifacts/generated.png",
            "caption": payload["caption"],
            "prompt": payload["prompt"],
        }

    async def read_workspace(payload: dict[str, Any]) -> str:
        calls.append(("read", payload))
        if payload["path"] == "docs/agent-runtime-principles.md":
            return "Agent Runtime: durable jobs, capability profiles, ToolGateway."
        raise FileNotFoundError(payload["path"])

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "channel_visual_generate_image",
                "channel_visual_image_generation",
                generate,
            ),
            FunctionAgentTool(
                "read_project_file",
                "read_workspace",
                read_workspace,
            ),
        )
    )
    worker = ChannelVisualWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=1,
        source_message_id="1",
        fingerprint="visual",
        kind="channel_visual",
        profile=CHANNEL_VISUAL_READONLY,
        context_pack=ContextPack(
            user_request="Жвуша Agent Runtime architecture",
            chat_context=(
                "Объяснить через картинку, как устроены capability profiles.",
            ),
        ),
        status=AgentJobStatus.QUEUED,
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    generate_call = next(payload for kind, payload in calls if kind == "generate")
    assert "Agent Runtime: durable jobs" in generate_call["prompt"]
    assert "ToolGateway" in generate_call["prompt"]
    assert "не рисуй код" in generate_call["prompt"]
    assert "agent_runtime/channel_visual_artifacts/generated.png" in capsule.artifacts


async def test_channel_visual_worker_falls_back_to_local_card_when_screenshot_fails() -> (
    None
):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def screenshot(payload: dict[str, Any]) -> str:
        calls.append(("screenshot", payload))
        raise RuntimeError("browser screenshot looks like a challenge screen")

    async def generate_card(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(("card", payload))
        return {
            "status": "ready",
            "asset_path": "agent_runtime/channel_visual_artifacts/card.png",
            "caption": payload["caption"],
            "source_url": payload["source_url"],
        }

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "browser_screenshot_url", "browser_screenshot", screenshot
            ),
            FunctionAgentTool(
                "channel_visual_generate_card",
                "channel_visual_image_generation",
                generate_card,
            ),
        )
    )
    worker = ChannelVisualWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=1,
        source_message_id="1",
        fingerprint="visual",
        kind="channel_visual",
        profile=CHANNEL_VISUAL_READONLY,
        context_pack=ContextPack(
            user_request="OpenAI Codex mobile",
            chat_context=("source_url: https://openai.com/index/post",),
        ),
        status=AgentJobStatus.QUEUED,
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert [kind for kind, _payload in calls] == ["screenshot", "card"]
    card_call = calls[1][1]
    assert card_call["source_url"] == "https://openai.com/index/post"
    assert capsule.artifacts == ("agent_runtime/channel_visual_artifacts/card.png",)
    assert "source_card" in capsule.markdown_report


async def test_channel_visual_image_tool_returns_workspace_relative_asset_path(
    tmp_path: Path,
) -> None:
    class LLM:
        def __init__(self) -> None:
            self.requests: list[LLMImageRequest] = []

        async def generate_image(
            self,
            request: LLMImageRequest,
        ) -> LLMImageResponse:
            self.requests.append(request)
            return LLMImageResponse(
                image=b"png",
                model="gpt-image-2",
                mime_type="image/png",
            )

    llm = LLM()
    tool = ChannelVisualImageTool(
        workspace_root=tmp_path,
        llm=cast("LLMGatewayProtocol", llm),
    )

    artifact = await tool.execute(
        {"prompt": "Карта архитектуры Жвуши", "caption": "Схема"}
    )

    assert artifact["asset_path"].startswith("agent_runtime/channel_visual_artifacts/")
    assert (tmp_path / artifact["asset_path"]).read_bytes() == b"png"
    assert llm.requests[0].caller == "agent_runtime.channel_visual"


async def test_channel_visual_local_card_tool_returns_workspace_relative_png(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def renderer(html: str, target: Path, timeout_seconds: float) -> None:
        captured["html"] = html
        captured["timeout"] = timeout_seconds
        target.write_bytes(b"png-card")

    tool = ChannelVisualLocalCardTool(
        workspace_root=tmp_path,
        renderer=renderer,
    )

    artifact = await tool.execute(
        {
            "title": "агент теперь в кармане",
            "body": "Codex в мобильном ChatGPT: смотреть сессии.",
            "source_url": "https://openai.com/index/work-with-codex-from-anywhere/",
            "caption": "Визуал",
        }
    )

    assert artifact["asset_path"].startswith(
        "agent_runtime/channel_visual_artifacts/fallback-card-"
    )
    assert artifact["mime_type"] == "image/png"
    assert artifact["model"] == "local-chromium-card"
    assert (tmp_path / artifact["asset_path"]).read_bytes() == b"png-card"
    assert "агент теперь" in captured["html"]
    assert "openai.com" in captured["html"]
