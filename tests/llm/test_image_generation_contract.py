"""LLM Gateway image-generation contract."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.llm.protocols import (
    LLMGatewayProtocol,
    LLMImageRequest,
    LLMImageResponse,
    ProviderUnavailableError,
)
from src.llm.router import LLMRouter


def _mock_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.name = "mock"
    adapter.default_model = "haiku"
    adapter.generate = AsyncMock()
    adapter.generate_with_tools = AsyncMock()
    return adapter


async def test_router_implements_optional_image_generation_contract() -> None:
    adapter = _mock_adapter()
    image_generator = AsyncMock(
        return_value=LLMImageResponse(
            image=b"png",
            model="configured-image-model",
            mime_type="image/png",
        )
    )
    router = LLMRouter(
        adapters={"worker": adapter, "analyst": adapter, "strategist": adapter},
        models={"worker": "haiku", "analyst": "haiku", "strategist": "haiku"},
        image_generator=image_generator,
    )

    result = await router.generate_image(
        LLMImageRequest(prompt="Нарисуй карту мысли", caller="test")
    )

    assert isinstance(router, LLMGatewayProtocol)
    assert result.image == b"png"
    image_generator.assert_awaited_once()


async def test_router_reports_image_generation_unavailable_without_provider() -> None:
    adapter = _mock_adapter()
    router = LLMRouter(
        adapters={"worker": adapter, "analyst": adapter, "strategist": adapter},
        models={"worker": "haiku", "analyst": "haiku", "strategist": "haiku"},
    )

    with pytest.raises(ProviderUnavailableError):
        await router.generate_image(LLMImageRequest(prompt="x"))


async def test_cli_image_generator_reads_prompt_and_output_artifact() -> None:
    from src.llm.cli_images import CLIImageGenerator

    command = (
        f"{sys.executable} -c "
        '"import os,sys; '
        "prompt=sys.stdin.read(); "
        "assert prompt == os.environ['ZHVUSHA_IMAGE_PROMPT']; "
        "assert os.environ['ZHVUSHA_IMAGE_MODEL'] == 'local-image-cli'; "
        "open(os.environ['ZHVUSHA_IMAGE_OUTPUT'], 'wb').write(b'png-bytes')\""
    )
    generator = CLIImageGenerator(command=command, model="local-image-cli")

    result = await generator.generate_image(
        LLMImageRequest(prompt="Карта мысли", caller="test")
    )

    assert result.image == b"png-bytes"
    assert result.model == "local-image-cli"
    assert result.mime_type == "image/png"


def test_router_builds_cli_image_generator_from_settings() -> None:
    from src.core.config import Settings
    from src.llm.cli_images import CLIImageGenerator
    from src.llm.router import _build_image_generator

    generator = _build_image_generator(
        Settings(
            bot_token="fake",
            channel_id="@test",
            admin_user_id=1,
            image_generation_enabled=True,
            image_generation_provider="cli",
            image_generation_cli_command=f"{sys.executable} -c pass",
        )
    )

    assert isinstance(generator, CLIImageGenerator)
