"""
A hard daily cap on Gemini API calls, shared across GitHub Actions runs.

The bot runs stateless every 20 minutes, and the Gemini key is shared with other projects, so the
count of calls made today lives in a repository variable `GEMINI_USAGE` as JSON:
    {"day": "2026-10-07", "calls": 42}
`day` is the Pacific date, because Google's free-tier daily quotas reset at midnight Pacific.

Per run: read the variable (passed in as the GEMINI_USAGE env var by the workflow), allow at most
`cap - calls` more calls, and at exit add this run's calls back to the variable through the GitHub
REST API using the workflow's GITHUB_TOKEN (needs `permissions: actions: write`). The write re-reads
the variable first so two overlapping runs don't lose each other's counts.

Pure helpers (parse/format/Budget) have no I/O: see tests/test_gemini_budget.py.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

VARIABLE_NAME = "GEMINI_USAGE"
DEFAULT_DAILY_CAP = 300  # 60% of the lowest credible free-tier RPD (500) we found for Flash-Lite; override via GEMINI_DAILY_CAP


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


def budget_from_env(env=os.environ) -> Budget:
    today = pacific_today()
    cap = int(env.get("GEMINI_DAILY_CAP") or DEFAULT_DAILY_CAP)
    return Budget(cap=cap, used_at_start=parse_usage(env.get(VARIABLE_NAME), today), today=today)


# ----------------------------------------------------------------------------- GitHub variable I/O


def _gh(env):
    token = env.get("GITHUB_TOKEN")
    repo = env.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return None, None
    return f"https://api.github.com/repos/{repo}/actions/variables", {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def record_usage(delta: int, env=os.environ) -> str | None:
    """Add this run's calls to the shared variable. Returns the stored JSON, or None if it couldn't."""
    if delta <= 0:
        return None
    base, headers = _gh(env)
    if base is None:
        logger.warning("GITHUB_TOKEN/GITHUB_REPOSITORY not set: Gemini usage not recorded (local run?)")
        return None
    today = pacific_today()
    try:
        r = requests.get(f"{base}/{VARIABLE_NAME}", headers=headers, timeout=20)
        if r.status_code == 404:
            current, exists = 0, False
        else:
            r.raise_for_status()
            current, exists = parse_usage(r.json().get("value"), today), True
        value = format_usage(today, current + delta)
        if exists:
            w = requests.patch(f"{base}/{VARIABLE_NAME}", headers=headers, json={"name": VARIABLE_NAME, "value": value}, timeout=20)
        else:
            w = requests.post(base, headers=headers, json={"name": VARIABLE_NAME, "value": value}, timeout=20)
        w.raise_for_status()
        logger.info(f"Recorded Gemini usage: {value}")
        return value
    except Exception as e:  # never fail a run on bookkeeping; the cap is conservative
        logger.warning(f"Could not record Gemini usage ({type(e).__name__}: {str(e)[:200]})")
        return None
