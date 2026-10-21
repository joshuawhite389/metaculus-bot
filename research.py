"""
Free, keyless live research for the forecast prompt (2026-10-21), for when no paid search is available.

Per question:
1. One LLM "planner" call turns the question into a short news query, Wikipedia titles and, when the question is
   about a US labour/price statistic, BLS series ids (JSON). If the planner fails, a keyword query from the title is
   used instead, so research never costs more than one call and never blocks a forecast.
2. GDELT DOC 2.0 API (no key; asks for at most one request per 5 s): recent headlines with dates and outlets.
3. Wikipedia (no key): the lead section of the pages the planner named, or of the top search hit.
4. The resolution source: any http(s) link in the resolution criteria / fine print / background, fetched once if
   its robots.txt allows us, reduced to plain text. This is where a question's data series usually lives
   ("latest value").
5. BLS public API v1 (no key, 25 queries/day): the latest values of the series the planner named.

Every source is best-effort: a failure is logged and skipped. Nothing here needs a secret.
Rules kept: public pages only, robots.txt honoured, an identifying User-Agent, GDELT's rate limit honoured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "gigbot-metaculus-bot/1.0 (forecasting research; +https://github.com/joshuawhite389/metaculus-bot)"
TIMEOUT = 20
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_GAP = 6.0  # seconds between GDELT requests (they ask for >= 5)
WIKI_API = "https://en.wikipedia.org/w/api.php"
BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
MAX_SOURCE_CHARS = 3500
MAX_SOURCES = 2
SKIP_HOSTS = ("metaculus.com", "twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com", "youtube.com")

_gdelt_lock = asyncio.Lock()
_gdelt_last = 0.0
_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


################################ PLANNER ################################


def planner_prompt(question_text: str, resolution_criteria: str, today: str) -> str:
    return f"""
You prepare searches for a forecaster. Do not forecast. Today is {today}.

Question: {question_text}

Resolution criteria: {resolution_criteria[:1500]}

Reply with JSON only, no prose, in this shape:
{{"news_query": "...", "wiki_titles": ["...", "..."], "bls_series": []}}

- news_query: 2-5 distinctive keywords for a news search engine that matches all words (names, places,
  organisations; no dates, no question words, no quotes). Example: Tesla deliveries
- wiki_titles: up to 2 exact English Wikipedia article titles most useful as background (the event, the
  organisation, the election, the index). Empty list if none fits.
- bls_series: only if the question is about a US Bureau of Labor Statistics number (payrolls, unemployment rate,
  CPI), the BLS series ids (e.g. CES0000000001 for total nonfarm payrolls, LNS14000000 for the unemployment rate,
  CUUR0000SA0 for CPI-U). Otherwise an empty list.
""".strip()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_STOP = set(
    "will what when which who whom whose how many much does do did is are was were be been the a an of in on at to for "
    "by before after from with than more less least most or and not any its it this that as per between during "
    "reach report reported close closing exceed above below over under level value price number total new next "
    "official officially according 2025 2026 2027".split()
)


def parse_plan(text: str) -> dict:
    """Planner output -> {"news_query": str, "wiki_titles": [..], "bls_series": [..]}; {} if unparseable."""
    m = _JSON_RE.search(text or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    plan = {
        "news_query": str(data.get("news_query") or "").replace('"', "").strip()[:120],
        "wiki_titles": [str(t).strip() for t in (data.get("wiki_titles") or []) if str(t).strip()][:2],
        "bls_series": [
            str(s).strip().upper()
            for s in (data.get("bls_series") or [])
            if re.fullmatch(r"[A-Z0-9]{8,24}", str(s).strip().upper())
        ][:3],
    }
    return plan


def fallback_query(question_text: str) -> str:
    """Keyword query from the title when the planner gave nothing: capitalised words first, max 4 words."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'.-]+", question_text)
    keep = [w.strip(".'") for w in words if w.lower() not in _STOP and len(w) > 2]
    caps = [w for w in keep if w[0].isupper()]
    chosen = (caps + [w for w in keep if w not in caps])[:4]
    return " ".join(chosen)


################################ SOURCES ################################


def _get(url: str, **kw) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kw.pop("headers", {})}
    return requests.get(url, headers=headers, timeout=kw.pop("timeout", TIMEOUT), **kw)


def gdelt_query_string(query: str) -> str:
    """GDELT wants >= 3-char words and rejects some punctuation; keep it simple."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", query) if len(w) >= 3]
    return " ".join(words[:6])


def format_gdelt(articles: list[dict], limit: int = 10) -> str:
    lines, seen = [], set()
    for a in articles:
        title = (a.get("title") or "").strip()
        key = title.lower()[:80]
        if not title or key in seen:
            continue
        seen.add(key)
        d = a.get("seendate") or ""
        date = f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) >= 8 else "?"
        lines.append(f"- {date} ({a.get('domain') or '?'}): {title}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


async def gdelt_headlines(query: str, timespan: str = "3w") -> str:
    """Recent English headlines matching all words, newest first. Serialised with a >= 6 s gap per process."""
    global _gdelt_last
    q = gdelt_query_string(query)
    if not q:
        return ""
    params = {
        "query": f"{q} sourcelang:english",
        "mode": "artlist",
        "maxrecords": 25,
        "format": "json",
        "timespan": timespan,
        "sort": "hybridrel",
    }
    async with _gdelt_lock:
        wait = GDELT_MIN_GAP - (time.monotonic() - _gdelt_last)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            r = await asyncio.to_thread(_get, GDELT_URL, params=params, timeout=30)
            if r.status_code == 429:  # their limiter is per IP, and Actions runners share IPs: one slower retry
                await asyncio.sleep(GDELT_MIN_GAP * 2)
                r = await asyncio.to_thread(_get, GDELT_URL, params=params, timeout=30)
        finally:
            _gdelt_last = time.monotonic()
    if r.status_code != 200 or not r.text.lstrip().startswith("{"):
        raise RuntimeError(f"GDELT {r.status_code}: {r.text[:120]}")
    articles = r.json().get("articles") or []
    articles.sort(key=lambda a: a.get("seendate") or "", reverse=True)
    return format_gdelt(articles)


def wikipedia_extracts(titles: list[str], fallback_search: str, chars: int = 1500) -> str:
    """Lead sections of the given titles (redirects followed); if none exist, of the top search hit."""
    out = []
    names = list(titles)
    if not names and fallback_search:
        r = _get(WIKI_API, params={"action": "query", "list": "search", "srsearch": fallback_search,
                                   "srlimit": 1, "format": "json"})
        hits = r.json().get("query", {}).get("search", [])
        names = [hits[0]["title"]] if hits else []
    for title in names[:2]:
        r = _get(WIKI_API, params={"action": "query", "prop": "extracts|revisions", "exintro": 1, "explaintext": 1,
                                   "redirects": 1, "titles": title, "rvprop": "timestamp", "format": "json"})
        for page in r.json().get("query", {}).get("pages", {}).values():
            text = (page.get("extract") or "").strip()
            if not text:
                continue
            edited = (page.get("revisions") or [{}])[0].get("timestamp", "")[:10]
            out.append(f"### {page.get('title')} (Wikipedia, last edited {edited or '?'})\n{text[:chars]}")
    return "\n\n".join(out)


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "head"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in ("p", "br", "tr", "li", "h1", "h2", "h3", "h4", "div", "table"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    text = "".join(p.parts)
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip(" |") for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if len(ln) > 1)


_URL_RE = re.compile(r"https?://[^\s)\]>\"'<]+")


def source_urls(*texts: str) -> list[str]:
    seen, out = set(), []
    for t in texts:
        for u in _URL_RE.findall(t or ""):
            u = u.rstrip(".,;:")
            host = urlparse(u).netloc.lower()
            if not host or any(host == h or host.endswith("." + h) for h in SKIP_HOSTS):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def robots_allows(url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = _get(base + "/robots.txt", timeout=10)
            if r.status_code in (401, 403):
                rp = None  # robots.txt itself refused: treat as "keep out"
            elif r.status_code >= 400:
                rp.parse([])  # no robots.txt: allowed
            else:
                rp.parse(r.text.splitlines())
        except Exception:
            rp = None
        _robots[base] = rp
    rp = _robots[base]
    return bool(rp and rp.can_fetch(USER_AGENT, url))


def fetch_source(url: str) -> str:
    """Plain text of a public resolution-source page, if robots.txt lets us; '' otherwise."""
    if not robots_allows(url):
        logger.info(f"robots.txt disallows {url}; skipped")
        return ""
    r = _get(url, timeout=TIMEOUT, stream=True)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    raw = r.raw.read(1_500_000, decode_content=True)
    text = raw.decode(r.encoding or "utf-8", errors="replace")
    if "json" in ctype or "csv" in ctype or "text/plain" in ctype:
        body = text
    elif "html" in ctype or text.lstrip().startswith("<"):
        body = html_to_text(text)
    else:
        return ""  # pdf, images, binaries
    body = body.strip()
    if len(body) > MAX_SOURCE_CHARS:
        body = body[: MAX_SOURCE_CHARS // 2] + "\n[...]\n" + body[-MAX_SOURCE_CHARS // 2 :]
    return body


def bls_latest(series: list[str], n: int = 6) -> str:
    if not series:
        return ""
    year = datetime.now(timezone.utc).year
    r = requests.post(BLS_URL, json={"seriesid": series, "startyear": str(year - 1), "endyear": str(year)},
                      headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    data = r.json()
    out = []
    for s in data.get("Results", {}).get("series", []):
        pts = s.get("data") or []
        if not pts:
            continue
        vals = ", ".join(f"{p['year']}-{p['periodName'][:3]} {p['value']}" for p in pts[:n])
        out.append(f"- BLS {s.get('seriesID')}: {vals} (newest first)")
    return "\n".join(out)


################################ ASSEMBLY ################################


async def gather_research(question, planner_llm, today: str | None = None) -> str:
    """All free sources for one question, as a block of text for the forecast prompt. Never raises."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    criteria = question.resolution_criteria or ""
    plan: dict = {}
    if planner_llm is not None:
        try:
            plan = parse_plan(await planner_llm.invoke(planner_prompt(question.question_text, criteria, today)))
        except Exception as e:
            logger.warning(f"Research planner failed for {question.page_url}: {type(e).__name__}: {str(e)[:160]}")
    query = plan.get("news_query") or fallback_query(question.question_text)
    sections: list[str] = []

    try:
        news = await gdelt_headlines(query)
        if not news and len(query.split()) > 2:
            news = await gdelt_headlines(" ".join(query.split()[:2]))
        sections.append(f"## Recent news headlines (GDELT, query: {query})\n{news or 'No matching articles in the last 3 weeks.'}")
    except Exception as e:
        logger.warning(f"GDELT failed for {question.page_url}: {str(e)[:160]}")

    try:
        wiki = await asyncio.to_thread(wikipedia_extracts, plan.get("wiki_titles") or [], query)
        if wiki:
            sections.append("## Background (Wikipedia)\n" + wiki)
    except Exception as e:
        logger.warning(f"Wikipedia failed for {question.page_url}: {str(e)[:160]}")

    urls = source_urls(criteria, question.fine_print or "", question.background_info or "")
    fetched = 0
    for url in urls:
        if fetched >= MAX_SOURCES:
            break
        try:
            text = await asyncio.to_thread(fetch_source, url)
        except Exception as e:
            logger.warning(f"Source fetch failed for {url}: {str(e)[:160]}")
            continue
        if text:
            fetched += 1
            sections.append(f"## Resolution source as fetched today ({url})\n{text}")

    try:
        bls = await asyncio.to_thread(bls_latest, plan.get("bls_series") or [])
        if bls:
            sections.append("## Latest official data (BLS)\n" + bls)
    except Exception as e:
        logger.warning(f"BLS failed for {question.page_url}: {str(e)[:160]}")

    if not sections:
        return "No live research could be retrieved; rely on background knowledge and the question text."
    header = (
        f"Automated research gathered {today} from free public sources (headlines only, not full articles; "
        "page text may be partial). Use it for the current status; it is not a forecast."
    )
    return header + "\n\n" + "\n\n".join(sections)
