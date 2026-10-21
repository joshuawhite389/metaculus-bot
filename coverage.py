"""
Coverage report: of every question in a tournament (open, closed or resolved), how many has this bot account
forecast? Read-only: no LLM call, nothing posted. Needs METACULUS_TOKEN (the API answers 403 without it, and
"forecast by me" comes from the token's own my_forecasts).

Run:  python gigbot_bot.py --coverage   (or in Actions: Test Bot -> Run workflow -> coverage)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from forecasting_tools import MetaculusClient
from forecasting_tools.helpers.metaculus_client import ApiFilter

logger = logging.getLogger(__name__)


def all_questions(client: MetaculusClient, tournament: str | int) -> list:
    """Every question in the tournament, groups unpacked, paging through the API (100 per page)."""
    api_filter = ApiFilter(
        allowed_tournaments=[tournament],
        allowed_statuses=["open", "closed", "resolved"],
        group_question_mode="unpack_subquestions",
        order_by="-open_time",
    )
    out, offset = [], 0
    while True:
        params = client._create_url_params_for_search(api_filter, offset=offset)
        page = client._get_questions_from_api(params, "unpack_subquestions")
        out.extend(page)
        if len(page) == 0 or offset > 5000:
            break
        offset += client.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
    seen, unique = set(), []
    for q in out:
        if q.id_of_question not in seen:
            seen.add(q.id_of_question)
            unique.append(q)
    return unique


def _hours(a: datetime | None, b: datetime | None) -> str:
    if not a or not b:
        return "?"
    return f"{(b - a).total_seconds() / 3600:.1f}h"


def report(client: MetaculusClient, tournaments: list[str | int], since: datetime | None = None) -> dict:
    """Print per-tournament coverage; return {tournament: (forecast, eligible)}. Questions that opened before
    `since` (when the bot went live) are left out of the denominator."""
    now = datetime.now(timezone.utc)
    result = {}
    for t in tournaments:
        try:
            qs = all_questions(client, t)
        except Exception as e:
            print(f"❌  {t}: could not list questions: {type(e).__name__}: {str(e)[:200]}")
            continue
        eligible = [q for q in qs if q.open_time and q.open_time <= now and (since is None or q.open_time >= since)]
        done = [q for q in eligible if q.already_forecasted]
        missed = [q for q in eligible if not q.already_forecasted]
        pct = 100 * len(done) / len(eligible) if eligible else 0.0
        print(f"📊  {t}: forecast {len(done)}/{len(eligible)} ({pct:.0f}%) of questions opened since "
              f"{since.date() if since else 'ever'}")
        windows = sorted(
            (q.close_time - q.open_time).total_seconds() / 3600 for q in eligible if q.open_time and q.close_time
        )
        if windows:
            print(f"    open windows: min {windows[0]:.1f}h, median {windows[len(windows) // 2]:.1f}h, max {windows[-1]:.1f}h")
        for q in sorted(missed, key=lambda q: q.open_time):
            state = "open now" if q.close_time and q.close_time > now else "closed"
            print(f"    ✗ {q.page_url}  opened {q.open_time:%m-%d %H:%M}Z, window {_hours(q.open_time, q.close_time)}, "
                  f"{state}: {q.question_text[:70]}")
        result[t] = (len(done), len(eligible))
    return result


def leaderboard(client: MetaculusClient, tournament: str, user_id: int | None) -> None:
    """Best effort: print the tournament leaderboard size and this bot's rank (endpoints as the website uses them)."""
    import requests

    headers = client._get_auth_headers()["headers"]
    base = client.base_url
    try:
        proj = requests.get(f"{base}/projects/tournaments/{tournament}/", headers=headers, timeout=30)
        proj.raise_for_status()
        pid = proj.json()["id"]
        r = requests.get(f"{base}/leaderboards/project/{pid}/", headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"ℹ️   {tournament} leaderboard: not available ({type(e).__name__}: {str(e)[:120]})")
        return
    entries = data.get("entries") or []
    mine = [e for e in entries if (e.get("user") or {}).get("id") == user_id]
    me = data.get("userEntry") or (mine[0] if mine else None)
    name = data.get("name") or data.get("score_type") or ""
    if me:
        print(f"🏁  {tournament} leaderboard {name}: rank {me.get('rank')} of {len(entries)}, "
              f"score {me.get('score')}, coverage {me.get('coverage')}, questions {me.get('contribution_count')}")
    else:
        print(f"🏁  {tournament} leaderboard {name}: {len(entries)} entries, this bot not listed yet")
