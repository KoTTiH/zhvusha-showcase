from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramNetworkError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from aiogram.methods import TelegramMethod
    from aiogram.types import InputFile

TelegramResult = TypeVar("TelegramResult")


class TelegramBotHttpProxySession(BaseSession):
    """Telegram Bot API session using httpx for local HTTP proxy support."""

    def __init__(
        self,
        *,
        request_proxy: str,
        max_attempts: int = 5,
        retry_delay_seconds: float = 0.35,
    ) -> None:
        super().__init__()
        self.request_proxy = request_proxy
        self.max_attempts = max(1, max_attempts)
        self.retry_delay_seconds = retry_delay_seconds
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _reset_client(self) -> None:
        await self.close()

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        client = self._ensure_client(timeout=float(timeout))
        async with client.stream("GET", url, headers=headers) as response:
            if raise_for_status:
                response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                yield chunk

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramResult],
        timeout: int | None = None,
    ) -> TelegramResult:
        url = self.api.api_url(token=bot.token, method=method.__api_method__)
        data, upload_files = await self._build_request_data(bot=bot, method=method)
        request_timeout = float(self.timeout if timeout is None else timeout)

        for attempt in range(1, self.max_attempts + 1):
            client = self._ensure_client(timeout=request_timeout)
            try:
                response = await client.post(
                    url,
                    data=data,
                    files=upload_files or None,
                )
                break
            except httpx.TimeoutException as exc:
                raise TelegramNetworkError(
                    method=method,
                    message="Request timeout error",
                ) from exc
            except httpx.ConnectError as exc:
                await self._reset_client()
                if attempt >= self.max_attempts:
                    raise TelegramNetworkError(
                        method=method,
                        message=f"ConnectError after {attempt} attempts: {exc}",
                    ) from exc
                await asyncio.sleep(self.retry_delay_seconds * attempt)
            except httpx.HTTPError as exc:
                raise TelegramNetworkError(
                    method=method,
                    message=f"{type(exc).__name__}: {exc}",
                ) from exc
        telegram_response = self.check_response(
            bot=bot,
            method=method,
            status_code=response.status_code,
            content=response.text,
        )
        return cast("TelegramResult", telegram_response.result)

    def _ensure_client(self, *, timeout: float) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                proxy=self.request_proxy,
                timeout=timeout,
            )
        return self._client

    async def _build_request_data(
        self,
        *,
        bot: Bot,
        method: TelegramMethod[TelegramResult],
    ) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes]]]]:
        data: dict[str, Any] = {}
        files: dict[str, InputFile] = {}
        for key, value in method.model_dump(warnings=False).items():
            prepared = self.prepare_value(value, bot=bot, files=files)
            if prepared is not None:
                data[key] = prepared

        upload_files: list[tuple[str, tuple[str, bytes]]] = []
        for key, value in files.items():
            chunks: list[bytes] = []
            async for chunk in value.read(bot):
                chunks.append(chunk)
            upload_files.append((key, (value.filename or key, b"".join(chunks))))
        return data, upload_files


def build_telegram_bot(*, token: str, proxy: str = "") -> Bot:
    proxy = proxy.strip()
    if not proxy:
        return Bot(token=token)
    return Bot(
        token=token,
        session=TelegramBotHttpProxySession(request_proxy=proxy),
    )
