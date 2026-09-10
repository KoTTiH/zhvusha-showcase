import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.core.config import get_settings
from src.llm.gemini import GeminiAdapter
from src.llm.protocols import LLMRequest, LLMResponse, LLMVisionRequest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def adapter():
    return GeminiAdapter()


@pytest.fixture
def mock_genai():
    mock = MagicMock()
    client = MagicMock()
    mock.Client.return_value = client
    with patch("src.llm.gemini._get_genai", return_value=mock):
        yield client, mock


def _req(
    prompt: str = "test",
    *,
    system: str = "",
    model: str | None = None,
) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        system=system,
        model=model,
        tier="worker",
        caller="test",
    )


def _vision_req(
    images: list[bytes],
    *,
    prompt: str = "Опиши что ты видишь на изображении(ях). Детально.",
) -> LLMVisionRequest:
    return LLMVisionRequest(images=images, prompt=prompt, caller="test")


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
async def test_generate_returns_response(adapter, mock_genai):
    client, _genai = mock_genai
    response = MagicMock()
    response.text = "Generated draft response"
    client.aio.models.generate_content = AsyncMock(return_value=response)

    result = await adapter.generate(_req("Write a response"))

    assert isinstance(result, LLMResponse)
    assert result.text == "Generated draft response"
    assert result.model == "gemini-2.5-flash-lite"
    client.aio.models.generate_content.assert_awaited_once()


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
async def test_generate_with_model(adapter, mock_genai):
    client, _genai = mock_genai
    response = MagicMock()
    response.text = "Draft"
    client.aio.models.generate_content = AsyncMock(return_value=response)

    await adapter.generate(
        _req(
            "Write a response",
            system="You are a freelancer",
            model="gemini-2.5-flash",
        )
    )

    call_kwargs = client.aio.models.generate_content.call_args
    assert call_kwargs.kwargs.get("model") == "gemini-2.5-flash"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
async def test_generate_passes_system_instruction(adapter, mock_genai):
    client, genai_mock = mock_genai
    response = MagicMock()
    response.text = "Draft"
    client.aio.models.generate_content = AsyncMock(return_value=response)

    await adapter.generate(_req("Write a response", system="You are a freelancer"))

    genai_mock.types.GenerateContentConfig.assert_called_once_with(
        system_instruction="You are a freelancer"
    )


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
async def test_generate_no_system_instruction(adapter, mock_genai):
    client, genai_mock = mock_genai
    response = MagicMock()
    response.text = "Draft"
    client.aio.models.generate_content = AsyncMock(return_value=response)

    await adapter.generate(_req("Write a response"))

    genai_mock.types.GenerateContentConfig.assert_called_once_with()


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
async def test_generate_empty_response(adapter, mock_genai):
    client, _genai = mock_genai
    response = MagicMock()
    response.text = None
    client.aio.models.generate_content = AsyncMock(return_value=response)

    result = await adapter.generate(_req("Write"))

    assert result.text == ""


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
async def test_generate_propagates_exception(adapter, mock_genai):
    client, _genai = mock_genai
    client.aio.models.generate_content = AsyncMock(
        side_effect=RuntimeError("API error")
    )

    with pytest.raises(RuntimeError, match="API error"):
        await adapter.generate(_req("Write"))


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
async def test_generate_uses_default_model(adapter, mock_genai):
    client, _genai = mock_genai
    response = MagicMock()
    response.text = "ok"
    client.aio.models.generate_content = AsyncMock(return_value=response)

    await adapter.generate(_req("test"))

    call_kwargs = client.aio.models.generate_content.call_args
    assert call_kwargs.kwargs.get("model") == "gemini-2.5-flash-lite"


# --- Vision (describe_images) ---


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
        "VISION_MODEL": "gemini-2.5-flash",
    },
    clear=True,
)
async def test_describe_images_multimodal(adapter, mock_genai):
    client, genai_mock = mock_genai
    response = MagicMock()
    response.text = "A photo of a cat"
    client.aio.models.generate_content = AsyncMock(return_value=response)
    fake_part = MagicMock()
    genai_mock.types.Part.from_bytes.return_value = fake_part

    img = b"\xff\xd8\xff\xe0fake-jpeg"
    result = await adapter.describe_images(_vision_req([img]))

    assert isinstance(result, LLMResponse)
    assert result.text == "A photo of a cat"
    assert result.model == "gemini-2.5-flash"
    genai_mock.types.Part.from_bytes.assert_called_once_with(
        data=img, mime_type="image/jpeg"
    )
    call_kwargs = client.aio.models.generate_content.call_args
    contents = call_kwargs.kwargs.get("contents")
    assert isinstance(contents, list)
    assert len(contents) == 2  # prompt text + 1 image part
    assert contents[1] is fake_part


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
        "VISION_MODEL": "gemini-2.5-flash",
    },
    clear=True,
)
async def test_describe_images_multiple_images(adapter, mock_genai):
    client, genai_mock = mock_genai
    response = MagicMock()
    response.text = "Two photos"
    client.aio.models.generate_content = AsyncMock(return_value=response)
    genai_mock.types.Part.from_bytes.return_value = MagicMock()

    imgs = [b"img1", b"img2", b"img3"]
    result = await adapter.describe_images(_vision_req(imgs))

    assert result.text == "Two photos"
    assert genai_mock.types.Part.from_bytes.call_count == 3
    call_kwargs = client.aio.models.generate_content.call_args
    contents = call_kwargs.kwargs.get("contents")
    assert len(contents) == 4  # prompt + 3 image parts


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "GOOGLE_API_KEY": "test-key",
        "VISION_MODEL": "gemini-2.5-flash",
    },
    clear=True,
)
async def test_describe_images_empty_response(adapter, mock_genai):
    client, genai_mock = mock_genai
    response = MagicMock()
    response.text = None
    client.aio.models.generate_content = AsyncMock(return_value=response)
    genai_mock.types.Part.from_bytes.return_value = MagicMock()

    result = await adapter.describe_images(_vision_req([b"img"]))

    assert result.text == ""
