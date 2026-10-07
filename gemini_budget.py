"""
A hard daily cap on Gemini API calls, shared across GitHub Actions runs.

The bot runs stateless every 20 minutes, and the Gemini key is shared with other projects, so the
count of calls made today lives in a small JSON file carried between runs by the Actions cache
(`actions/cache/restore` before the bot, `actions/cache/save` after it, key prefix `gemini-usage-`):
    {"day": "2026-10-07", "calls": 42}
`day` is the Pacific date, because Google's free-tier daily quotas reset at midnight Pacific.
(A repository variable was the first design; GITHUB_TOKEN gets 403 on the variables API, 2026-10-07.)
The env var GEMINI_USAGE, if set, overrides the file (manual reset or correction).

Per run: read, allow at most `cap - calls` more calls, and at exit write `calls + this run` back to
the file. Two overlapping runs can undercount each other by one run's calls; the cap is conservative.

Also here: a per-model sliding-window rate limiter (free tier has a low per-minute quota) and the
budget object the bot's Gemini wrapper consults before every call.

Pure helpers (parse/format/Budget/RateLimiter) have no I/O: see tests/test_gemini_budget.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

VARIABLE_NAME = "GEMINI_USAGE"
DEFAULT_USAGE_FILE = ".gemini-usage/usage.json"
DEFAULT_DAILY_CAP = 300  # 60% of the lowest credible free-tier RPD (500) we found for Flash-Lite; override via GEMINI_DAILY_CAP
DEFAULT_RPM = 8  # per model; free-tier Flash-Lite per-minute quota is ~10-15, and 429s cost retries
MAX_CONSECUTIVE_RATE_LIMITS = 8  # after this many 429s in a row, assume the daily quota is gone and stop the run


class GeminiBudgetExhausted(RuntimeError):
    """Raised before a Gemini call when today's cap is used up (or a daily-quota 429 was seen)."""


def pacific_today(now: datetime | None = None) -> str:
    now = now or datetime.now(tz=ZoneInfo("America/Los_Angeles"))
    return now.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def parse_usage(raw: str | None, today: str) -> int:
    """Calls already used today according to the stored JSON; 0 for another day, empty or garbage."""
    if not raw or not raw.strip():
        return 0
    try:
        data = json.loads(raw)
        if data.get("day") == today:
            return max(0, int(data.get("calls", 0)))
    except (ValueError, TypeError, AttributeError):
        logger.warning(f"Could not parse {VARIABLE_NAME}={raw!r}; treating as 0 calls today")
    return 0


def format_usage(today: str, calls: int) -> str:
    return json.dumps({"day": today, "calls": int(calls)})


def is_daily_quota_error(message: str) -> bool:
    """Does a 429 message point at the per-day quota (as opposed to per-minute)?"""
    m = message.lower()
    return "perday" in m or "per day" in m or "daily" in m or "rpd" in m


class Budget:
    def __init__(self, cap: int, used_at_start: int, today: str):
        self.cap = int(cap)
        self.used_at_start = int(used_at_start)
        self.today = today
        self.used_this_run = 0
        self.exhausted_reason: str | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.used_at_start - self.used_this_run)

    def take(self) -> None:
        """Reserve one call, or raise GeminiBudgetExhausted."""
        if self.exhausted_reason:
            raise GeminiBudgetExhausted(self.exhausted_reason)
        if self.remaining <= 0:
            self.exhausted_reason = (
                f"Gemini daily cap reached: {self.used_at_start + self.used_this_run}/{self.cap} calls on {self.today}"
            )
            raise GeminiBudgetExhausted(self.exhausted_reason)
        self.used_this_run += 1

    def exhaust(self, reason: str) -> None:
        self.exhausted_reason = reason

    def summary(self) -> str:
        return (
            f"Gemini calls: {self.used_this_run} this run, {self.used_at_start + self.used_this_run}/{self.cap} today "
            f"({self.today} Pacific)" + (f"; stopped: {self.exhausted_reason}" if self.exhausted_reason else "")
        )


def usage_file(env=os.environ) -> Path:
    return Path(env.get("GEMINI_USAGE_FILE") or DEFAULT_USAGE_FILE)


def read_stored_usage(env=os.environ) -> str | None:
    """The GEMINI_USAGE env var if set, else the cached file's contents, else None."""
    if (env.get(VARIABLE_NAME) or "").strip():
        return env[VARIABLE_NAME]
    f = usage_file(env)
    try:
        return f.read_text() if f.exists() else None
    except OSError as e:
        logger.warning(f"Could not read {f}: {e}")
        return None


def budget_from_env(env=os.environ) -> Budget:
    today = pacific_today()
    cap = int(env.get("GEMINI_DAILY_CAP") or DEFAULT_DAILY_CAP)
    return Budget(cap=cap, used_at_start=parse_usage(read_stored_usage(env), today), today=today)


def record_usage(delta: int, env=os.environ) -> str | None:
    """Add this run's calls to the cached file (re-read first). Returns the stored JSON, or None on failure."""
    f = usage_file(env)
    today = pacific_today()
    try:
        current = 0
        if f.exists():
            current = parse_usage(f.read_text(), today)
        value = format_usage(today, current + max(0, int(delta)))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(value)
        logger.info(f"Recorded Gemini usage in {f}: {value}")
        return value
    except Exception as e:  # never fail a run on bookkeeping; the cap is conservative
        logger.warning(f"Could not record Gemini usage ({type(e).__name__}: {str(e)[:200]})")
        return None


class RateLimiter:
    """Sliding window: at most `per_minute` acquisitions in any 60 s. Async-safe within one event loop."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleeper=asyncio.sleep):
        self.per_minute = max(1, int(per_minute))
        self._times: deque[float] = deque()
        self._clock = clock
        self._sleep = sleeper
        self._lock: asyncio.Lock | None = None

    def wait_needed(self, now: float) -> float:
        while self._times and now - self._times[0] >= 60.0:
            self._times.popleft()
        if len(self._times) < self.per_minute:
            return 0.0
        return 60.0 - (now - self._times[0])

    async def acquire(self) -> float:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            waited = 0.0
            while True:
                w = self.wait_needed(self._clock())
                if w <= 0:
                    break
                waited += w
                await self._sleep(w)
            self._times.append(self._clock())
            return waited


_limiters: dict[str, RateLimiter] = {}


def limiter_for(model: str, env=os.environ) -> RateLimiter:
    if model not in _limiters:
        _limiters[model] = RateLimiter(int(env.get("GEMINI_RPM") or DEFAULT_RPM))
    return _limiters[model]
