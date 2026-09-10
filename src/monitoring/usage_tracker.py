"""LLM usage tracking with per-day stats and token-level cost.

Buckets are keyed ``"<provider>/<api_id>"``. Pricing comes from the
provider registry in :mod:`src.llm.providers` — adding a new provider /
model is one edit there, no changes here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.llm.providers import get_pricing, resolve_model

if TYPE_CHECKING:
    from pathlib import Path

    from src.monitoring.dashboard import UsageDashboard

logger = structlog.get_logger()


CACHE_WRITE_MULT = 1.25  # 5-min ephemeral cache write multiplier (Anthropic API)
CACHE_READ_MULT = 0.1  # read discount (90% off)


def _bucket(provider: str, model: str) -> str:
    """Canonical bucket key ``"<provider>/<api_id>"``.

    If ``model`` is an alias (``haiku``), the registry resolves it to the
    api_id so calls under either name aggregate into one bucket.
    Unknown models fall through unchanged (cost looks up to None → zero).
    """
    spec = resolve_model(provider, model)
    api_id = spec.api_id if spec is not None else model
    return f"{provider}/{api_id}"


def _migrate_legacy_bucket(key: str) -> str:
    """Map a pre-provider-aware bucket key to the new ``provider/api_id`` form.

    Older monthly JSON files stored ``"haiku"`` / ``"claude-sonnet-4-6"``
    without a provider. We assume Anthropic for any Claude-shaped name
    (the only adapter that historically reported per-token usage) and
    Gemini for ``gemini-...`` names. Anything else passes through.
    """
    if not key or "/" in key:
        return key
    if key in ("haiku", "sonnet", "opus"):
        spec = resolve_model("anthropic_api", key)
        api_id = spec.api_id if spec is not None else key
        return f"anthropic_api/{api_id}"
    if key.startswith("claude-"):
        return f"anthropic_api/{key}"
    if key.startswith("gemini-"):
        return f"gemini/{key}"
    return key


def _rebucket(data: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in data.items():
        new_key = _migrate_legacy_bucket(key)
        out[new_key] = out.get(new_key, 0) + int(value or 0)
    return out


@dataclass
class DayStats:
    """Usage stats for a single day."""

    api_calls: dict[str, int] = field(default_factory=dict)
    cli_sessions: int = 0
    cli_calls: int = 0
    gemini_calls: int = 0
    input_tokens: dict[str, int] = field(default_factory=dict)
    output_tokens: dict[str, int] = field(default_factory=dict)
    cache_read_tokens_by_model: dict[str, int] = field(default_factory=dict)
    cache_write_tokens_by_model: dict[str, int] = field(default_factory=dict)
    caller_counts: dict[str, int] = field(default_factory=dict)

    @property
    def cache_read_tokens(self) -> int:
        return sum(self.cache_read_tokens_by_model.values())

    @property
    def cache_write_tokens(self) -> int:
        return sum(self.cache_write_tokens_by_model.values())

    @property
    def cost_usd(self) -> float:
        cost = 0.0
        for bucket in {
            *self.input_tokens,
            *self.output_tokens,
            *self.cache_read_tokens_by_model,
            *self.cache_write_tokens_by_model,
        }:
            provider, _, model = bucket.partition("/")
            if not provider:
                continue
            spec = get_pricing(provider, model)
            if spec is None:
                continue
            cost += self.input_tokens.get(bucket, 0) * spec.input_per_mtok / 1_000_000
            cost += self.output_tokens.get(bucket, 0) * spec.output_per_mtok / 1_000_000
            cost += (
                self.cache_write_tokens_by_model.get(bucket, 0)
                * spec.input_per_mtok
                * spec.cache_write_mult
                / 1_000_000
            )
            cost += (
                self.cache_read_tokens_by_model.get(bucket, 0)
                * spec.input_per_mtok
                * spec.cache_read_mult
                / 1_000_000
            )
        return cost

    @property
    def total_api(self) -> int:
        return sum(self.api_calls.values())

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_read_tokens + self.cache_write_tokens
        if total == 0:
            return 0.0
        return self.cache_read_tokens / total

    @property
    def cache_saved_usd(self) -> float:
        """Estimated savings vs reading the same tokens at full input price."""
        if not self.cache_read_tokens_by_model:
            return 0.0
        saved = 0.0
        for bucket, tokens in self.cache_read_tokens_by_model.items():
            provider, _, model = bucket.partition("/")
            if not provider:
                continue
            spec = get_pricing(provider, model)
            if spec is None:
                continue
            full = tokens * spec.input_per_mtok / 1_000_000
            cached = full * spec.cache_read_mult
            saved += full - cached
        return saved

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "cli_sessions": self.cli_sessions,
            "cli_calls": self.cli_calls,
            "gemini_calls": self.gemini_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens_by_model": self.cache_read_tokens_by_model,
            "cache_write_tokens_by_model": self.cache_write_tokens_by_model,
            "caller_counts": self.caller_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DayStats:
        # Cache fields have two valid shapes on disk:
        # (a) new format: "cache_read_tokens_by_model" → {bucket: tokens}
        # (b) legacy scalar: "cache_read_tokens" → 1000 (pre per-model fix)
        # Legacy scalars are attributed to sonnet (the only model the bot
        # used at the time the per-model split was introduced).
        read_by_model = data.get("cache_read_tokens_by_model")
        if read_by_model is None:
            legacy_read = int(data.get("cache_read_tokens", 0) or 0)
            read_by_model = {"sonnet": legacy_read} if legacy_read else {}
        write_by_model = data.get("cache_write_tokens_by_model")
        if write_by_model is None:
            legacy_write = int(data.get("cache_write_tokens", 0) or 0)
            write_by_model = {"sonnet": legacy_write} if legacy_write else {}
        return cls(
            api_calls=data.get("api_calls", {}),
            cli_sessions=data.get("cli_sessions", 0),
            cli_calls=data.get("cli_calls", 0),
            gemini_calls=data.get("gemini_calls", 0),
            input_tokens=data.get("input_tokens", {}),
            output_tokens=data.get("output_tokens", {}),
            cache_read_tokens_by_model=dict(read_by_model),
            cache_write_tokens_by_model=dict(write_by_model),
            caller_counts=data.get("caller_counts", {}),
        )


class UsageTracker:
    """Tracks LLM usage per day, persists to monthly JSON files."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, DayStats] = {}
        self._current_month: str = ""
        self._dashboard: UsageDashboard | None = None
        self._call_timestamps: list[float] = []
        self._load_current_month()

    def set_dashboard(self, dashboard: UsageDashboard) -> None:
        self._dashboard = dashboard

    def record_api_call(
        self,
        model: str,
        *,
        provider: str = "anthropic_api",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        caller: str = "",
    ) -> None:
        """Record a per-token API call.

        ``provider`` defaults to ``anthropic_api`` so historical callers
        that only knew the model name keep working — but the router now
        always supplies ``provider`` explicitly, taken from the same
        ``providers_by_tier`` it built at startup.
        """
        bucket = _bucket(provider, model)
        today = self._get_today()
        today.api_calls[bucket] = today.api_calls.get(bucket, 0) + 1
        today.input_tokens[bucket] = today.input_tokens.get(bucket, 0) + input_tokens
        today.output_tokens[bucket] = today.output_tokens.get(bucket, 0) + output_tokens
        if cache_read_tokens:
            today.cache_read_tokens_by_model[bucket] = (
                today.cache_read_tokens_by_model.get(bucket, 0) + cache_read_tokens
            )
        if cache_write_tokens:
            today.cache_write_tokens_by_model[bucket] = (
                today.cache_write_tokens_by_model.get(bucket, 0) + cache_write_tokens
            )
        if caller:
            today.caller_counts[caller] = today.caller_counts.get(caller, 0) + 1
        self._record_timestamp()
        self._save()
        self._notify_dashboard()

    def record_cli_session(self, caller: str = "") -> None:
        """Record a full subscription-backed CLI session (Codex /morning)."""
        today = self._get_today()
        today.cli_sessions += 1
        if caller:
            today.caller_counts[caller] = today.caller_counts.get(caller, 0) + 1
        self._record_timestamp()
        self._save()
        self._notify_dashboard()

    def record_cli_call(
        self,
        *,
        provider: str = "claude_cli",
        model: str = "",
        caller: str = "",
    ) -> None:
        """Record a single subscription-billed CLI call.

        Cost is zero — billing is per-month subscription, not per token.
        We track call counts only so caller breakdown still works.
        """
        del (
            provider,
            model,
        )  # accepted for symmetry; subscription billing has no cost lookup
        today = self._get_today()
        today.cli_calls += 1
        if caller:
            today.caller_counts[caller] = today.caller_counts.get(caller, 0) + 1
        self._record_timestamp()
        self._save()
        self._notify_dashboard()

    def record_gemini_call(
        self,
        *,
        provider: str = "gemini",
        model: str = "",
        caller: str = "",
    ) -> None:
        """Record a Gemini call (free tier — no per-token cost)."""
        del provider, model
        today = self._get_today()
        today.gemini_calls += 1
        if caller:
            today.caller_counts[caller] = today.caller_counts.get(caller, 0) + 1
        self._record_timestamp()
        self._save()
        self._notify_dashboard()

    def get_today(self) -> DayStats:
        return self._get_today()

    def get_calls_in_last_hour(self) -> int:
        self._prune_timestamps()
        return len(self._call_timestamps)

    def get_month_total(self) -> float:
        self._ensure_current_month()
        return sum(stats.cost_usd for stats in self._cache.values())

    def _get_today(self) -> DayStats:
        self._ensure_current_month()
        today_key = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if today_key not in self._cache:
            self._cache[today_key] = DayStats()
        return self._cache[today_key]

    def _ensure_current_month(self) -> None:
        month = datetime.now(tz=UTC).strftime("%Y-%m")
        if month != self._current_month:
            self._current_month = month
            self._cache = {}
            self._load_current_month()

    def _month_file(self) -> Path:
        return self._data_dir / f"usage_{self._current_month}.json"

    def _load_current_month(self) -> None:
        self._current_month = datetime.now(tz=UTC).strftime("%Y-%m")
        path = self._month_file()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for day_key, day_data in raw.items():
                stats = DayStats.from_dict(day_data)
                # Migrate legacy bucket keys (no provider prefix).
                stats.api_calls = _rebucket(stats.api_calls)
                stats.input_tokens = _rebucket(stats.input_tokens)
                stats.output_tokens = _rebucket(stats.output_tokens)
                stats.cache_read_tokens_by_model = _rebucket(
                    stats.cache_read_tokens_by_model
                )
                stats.cache_write_tokens_by_model = _rebucket(
                    stats.cache_write_tokens_by_model
                )
                self._cache[day_key] = stats
        except (json.JSONDecodeError, OSError):
            logger.warning("usage_tracker_load_failed", path=str(path), exc_info=True)

    def _save(self) -> None:
        data = {k: v.to_dict() for k, v in self._cache.items()}
        try:
            self._month_file().write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            logger.warning("usage_tracker_save_failed", exc_info=True)

    def _record_timestamp(self) -> None:
        self._call_timestamps.append(time.monotonic())
        self._prune_timestamps()

    def _prune_timestamps(self) -> None:
        cutoff = time.monotonic() - 3600.0
        if self._call_timestamps and self._call_timestamps[0] >= cutoff:
            return
        self._call_timestamps = [ts for ts in self._call_timestamps if ts >= cutoff]

    def _notify_dashboard(self) -> None:
        if self._dashboard is not None:
            self._dashboard.schedule_update()
