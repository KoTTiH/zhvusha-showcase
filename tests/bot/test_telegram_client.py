import asyncio

import httpx
from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import GetMe
from src.bot.telegram_client import TelegramBotHttpProxySession, build_telegram_bot


class _FakeResponse:
    status_code = 200
    text = (
        '{"ok":true,"result":{"id":8654831259,"is_bot":true,'
        '"first_name":"Zhvusha","username":"Zhvusha_bot"}}'
    )


class _FlakyClient:
    def __init__(self) -> None:
        self.attempts = 0
        self.is_closed = False
        self.close_count = 0

    async def post(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        self.attempts += 1
        if self.attempts < 3:
            raise httpx.ConnectError("proxy reset")
        return _FakeResponse()

    async def aclose(self) -> None:
        self.close_count += 1
        self.is_closed = False


class _RetrySession(TelegramBotHttpProxySession):
    def __init__(self, client: _FlakyClient) -> None:
        super().__init__(
            request_proxy="http://127.0.0.1:7897",
            max_attempts=3,
            retry_delay_seconds=0,
        )
        self.client = client

    def _ensure_client(self, *, timeout: float) -> _FlakyClient:
        return self.client

    async def _reset_client(self) -> None:
        await self.client.aclose()


class _FakeBotInfo:
    username = "Zhvusha_bot"


class _FailingGetMeBot:
    async def get_me(self) -> _FakeBotInfo:
        raise TelegramNetworkError(method=GetMe(), message="network down")


class _WorkingGetMeBot:
    async def get_me(self) -> _FakeBotInfo:
        return _FakeBotInfo()


class _HangingGetMeBot:
    async def get_me(self) -> _FakeBotInfo:
        await asyncio.sleep(60)
        return _FakeBotInfo()


def test_build_telegram_bot_uses_default_session_without_proxy() -> None:
    bot = build_telegram_bot(token="123:test", proxy="")

    try:
        assert isinstance(bot, Bot)
        assert not isinstance(bot.session, TelegramBotHttpProxySession)
    finally:
        # No network session is opened by construction, but keep the test robust
        # if aiogram starts allocating one eagerly in a future release.
        asyncio.run(bot.session.close())


def test_build_telegram_bot_uses_http_proxy_session() -> None:
    bot = build_telegram_bot(token="123:test", proxy="http://127.0.0.1:7897")

    try:
        assert isinstance(bot.session, TelegramBotHttpProxySession)
        assert bot.session.request_proxy == "http://127.0.0.1:7897"
    finally:
        asyncio.run(bot.session.close())


def test_http_proxy_session_retries_connect_errors() -> None:
    client = _FlakyClient()
    session = _RetrySession(client)
    bot = Bot(token="123:test", session=session)

    try:
        user = asyncio.run(session.make_request(bot, GetMe()))
    finally:
        asyncio.run(bot.session.close())

    assert user.username == "Zhvusha_bot"
    assert client.attempts == 3
    assert client.close_count == 2


async def test_social_trigger_username_resolution_is_not_startup_fatal() -> None:
    from src.bot.main import _resolve_bot_username_for_social_trigger

    assert await _resolve_bot_username_for_social_trigger(_FailingGetMeBot()) == ""
    assert (
        await _resolve_bot_username_for_social_trigger(_WorkingGetMeBot())
        == "Zhvusha_bot"
    )


async def test_social_trigger_username_resolution_has_startup_timeout() -> None:
    from src.bot.main import _resolve_bot_username_for_social_trigger

    assert (
        await _resolve_bot_username_for_social_trigger(
            _HangingGetMeBot(),
            timeout_seconds=0.01,
        )
        == ""
    )
