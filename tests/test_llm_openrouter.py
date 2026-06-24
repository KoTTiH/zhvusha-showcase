"""Tests for ``src.llm.openrouter.OpenRouterAdapter``.

Mocks ``httpx.AsyncClient.post`` so no real HTTP traffic happens.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.llm.openrouter import OpenRouterAdapter
from src.llm.protocols import LLMError, LLMRequest


def _ok_response(text: str = "hello", *, prompt: int = 11, completion: int = 7) -> Any:
    """Build an httpx-like Response stub matching OpenRouter's payload."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value={
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )
    return resp


@pytest.fixture
def patched_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub get_settings() to expose ``openrouter_api_key`` without env."""
    from src.llm import openrouter

    fake = MagicMock()
    fake.openrouter_api_key = "sk-or-test"
    monkeypatch.setattr(openrouter, "get_settings", lambda: fake)


def _patch_client(monkeypatch: pytest.MonkeyPatch, post_mock: AsyncMock) -> MagicMock:
    """Replace ``httpx.AsyncClient`` with a class returning a mock instance."""
    instance = MagicMock()
    instance.post = post_mock
    instance.aclose = AsyncMock()
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr("src.llm.openrouter.httpx.AsyncClient", factory)
    return factory


class TestAdapterContract:
    def test_name_and_default_model(self) -> None:
        adapter = OpenRouterAdapter()
        assert adapter.name == "openrouter"
        assert adapter.default_model == ""

    async def test_empty_model_raises(self, patched_settings: None) -> None:
        adapter = OpenRouterAdapter()
        with pytest.raises(LLMError, match="empty model"):
            await adapter.generate(LLMRequest(prompt="hi", model=""))


class TestGenerate:
    async def test_returns_text_and_usage(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        post = AsyncMock(return_value=_ok_response("привет", prompt=12, completion=4))
        _patch_client(monkeypatch, post)

        adapter = OpenRouterAdapter()
        resp = await adapter.generate(
            LLMRequest(prompt="hi", model="deepseek/deepseek-chat", caller="t")
        )

        assert resp.text == "привет"
        assert resp.model == "deepseek/deepseek-chat"
        assert resp.usage.input_tokens == 12
        assert resp.usage.output_tokens == 4
        assert resp.usage.cache_read_tokens == 0
        assert resp.usage.cache_write_tokens == 0

    async def test_posts_to_chat_completions_with_bearer_auth(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        post = AsyncMock(return_value=_ok_response())
        factory = _patch_client(monkeypatch, post)

        adapter = OpenRouterAdapter()
        await adapter.generate(LLMRequest(prompt="hi", model="x/y"))

        # AsyncClient(...) constructed with base_url + bearer header
        kwargs = factory.call_args.kwargs
        assert kwargs["base_url"].rstrip("/") == "https://openrouter.ai/api/v1"
        assert kwargs["headers"]["Authorization"] == "Bearer sk-or-test"

        # POST landed on /chat/completions with correct payload shape
        post.assert_awaited_once()
        path = post.call_args.args[0]
        assert path.endswith("/chat/completions")
        body = post.call_args.kwargs["json"]
        assert body["model"] == "x/y"
        assert body["messages"][-1] == {"role": "user", "content": "hi"}

    async def test_system_prompt_prepended_as_system_message(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        post = AsyncMock(return_value=_ok_response())
        _patch_client(monkeypatch, post)

        adapter = OpenRouterAdapter()
        await adapter.generate(
            LLMRequest(prompt="hi", system="you are zhvusha", model="x/y")
        )

        body = post.call_args.kwargs["json"]
        assert body["messages"][0] == {"role": "system", "content": "you are zhvusha"}
        assert body["messages"][1] == {"role": "user", "content": "hi"}

    async def test_temperature_passed_through(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        post = AsyncMock(return_value=_ok_response())
        _patch_client(monkeypatch, post)

        adapter = OpenRouterAdapter()
        await adapter.generate(LLMRequest(prompt="hi", model="x/y", temperature=0.3))

        body = post.call_args.kwargs["json"]
        assert body["temperature"] == 0.3

    async def test_http_error_wrapped_as_llm_error(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        post = AsyncMock(side_effect=RuntimeError("boom: 401 unauthorized"))
        _patch_client(monkeypatch, post)

        adapter = OpenRouterAdapter()
        with pytest.raises(LLMError, match="OpenRouter"):
            await adapter.generate(
                LLMRequest(prompt="hi", model="deepseek/deepseek-chat")
            )

    async def test_429_surfaces_openrouter_reason_in_message(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OpenRouter wraps the *why* in body; surface it instead of a bare
        URL+status. On 429 the user needs to know if it's free-tier limit,
        insufficient credits, or upstream model overload."""
        import httpx

        err_response = httpx.Response(
            status_code=429,
            content=(
                b'{"error":{"message":"Rate limit exceeded for free tier","code":429}}'
            ),
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/x"),
        )
        post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "429", request=err_response.request, response=err_response
            )
        )
        _patch_client(monkeypatch, post)

        adapter = OpenRouterAdapter()
        with pytest.raises(LLMError) as excinfo:
            await adapter.generate(LLMRequest(prompt="hi", model="x/y"))

        msg = str(excinfo.value)
        assert "429" in msg
        assert "Rate limit exceeded for free tier" in msg

    async def test_http_error_with_non_json_body_falls_back_to_snippet(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the error body isn't JSON, surface a text snippet so we still
        get *some* signal (not just the URL)."""
        import httpx

        err_response = httpx.Response(
            status_code=502,
            content=b"<html>bad gateway</html>",
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/x"),
        )
        post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "502", request=err_response.request, response=err_response
            )
        )
        _patch_client(monkeypatch, post)

        adapter = OpenRouterAdapter()
        with pytest.raises(LLMError) as excinfo:
            await adapter.generate(LLMRequest(prompt="hi", model="x/y"))

        assert "502" in str(excinfo.value)
        assert "bad gateway" in str(excinfo.value)

    async def test_empty_choices_raises(
        self,
        patched_settings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bad = MagicMock()
        bad.raise_for_status = MagicMock()
        bad.json = MagicMock(return_value={"choices": []})
        _patch_client(monkeypatch, AsyncMock(return_value=bad))

        adapter = OpenRouterAdapter()
        with pytest.raises(LLMError, match=r"(?i)empty"):
            await adapter.generate(LLMRequest(prompt="hi", model="x/y"))
