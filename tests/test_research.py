"""Free research helpers (research.py) and the polling loop (gigbot_bot.poll_tournaments). No network."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("METACULUS_TOKEN", "offline-test")
import research as r  # noqa: E402
import gigbot_bot as g  # noqa: E402


def test_parse_plan_reads_json_inside_prose_and_cleans_it():
    text = 'Sure:\n```json\n{"news_query": "\\"Tesla\\" deliveries", "wiki_titles": ["Tesla, Inc.", "", "a", "b"], "bls_series": ["ces0000000001", "bad id!"]}\n```'
    plan = r.parse_plan(text)
    assert plan["news_query"] == "Tesla deliveries"
    assert plan["wiki_titles"] == ["Tesla, Inc.", "a"]
    assert plan["bls_series"] == ["CES0000000001"]


def test_parse_plan_garbage_is_empty():
    assert r.parse_plan("no json here") == {}
    assert r.parse_plan("{not json}") == {}
    assert r.parse_plan("") == {}


def test_fallback_query_keeps_distinctive_words():
    q = r.fallback_query("Will the FAA type-certify the Boeing 737 MAX 10 before 2027?")
    assert "FAA" in q and "Boeing" in q
    assert "Will" not in q and "2027" not in q
    assert len(q.split()) <= 4


def test_gdelt_query_string_drops_short_words_and_punctuation():
    assert r.gdelt_query_string('Putin "Trump" meet, in?') == "Putin Trump meet"


def test_format_gdelt_dedupes_and_dates():
    arts = [
        {"title": "Big news", "seendate": "20261020T101500Z", "domain": "a.com"},
        {"title": "Big news", "seendate": "20261019T101500Z", "domain": "b.com"},
        {"title": "Other", "seendate": "", "domain": ""},
    ]
    out = r.format_gdelt(arts).splitlines()
    assert out == ["- 2026-10-20 (a.com): Big news", "- ? (?): Other"]


def test_source_urls_finds_links_and_skips_social_and_metaculus():
    text = (
        "Resolves per https://www.eia.gov/petroleum/gasdiesel/. See also (https://x.com/foo) and "
        "https://www.metaculus.com/questions/1/ and https://www.eia.gov/petroleum/gasdiesel/."
    )
    assert r.source_urls(text, "fine print https://data.bls.gov/x?y=1") == [
        "https://www.eia.gov/petroleum/gasdiesel/",
        "https://data.bls.gov/x?y=1",
    ]


def test_html_to_text_drops_scripts_and_keeps_table_cells():
    html = "<html><head><title>t</title><script>var x=1</script></head><body><nav>menu</nav>" \
           "<p>Gatun Lake level</p><table><tr><td>Oct 20</td><td>86.2 ft</td></tr></table></body></html>"
    text = r.html_to_text(html)
    assert "var x" not in text and "menu" not in text
    assert "Gatun Lake level" in text
    assert "Oct 20 | 86.2 ft" in text


def test_gather_research_never_raises_when_everything_fails(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("down")

    def boom_sync(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(r, "gdelt_headlines", boom)
    monkeypatch.setattr(r, "wikipedia_extracts", boom_sync)
    monkeypatch.setattr(r, "fetch_source", boom_sync)
    monkeypatch.setattr(r, "bls_latest", boom_sync)

    class Planner:
        async def invoke(self, prompt):
            raise RuntimeError("quota")

    q = SimpleNamespace(question_text="Will X happen?", resolution_criteria="see https://example.org/a",
                        fine_print="", background_info="", page_url="u")
    out = asyncio.run(r.gather_research(q, Planner()))
    assert "No live research" in out


def test_gather_research_assembles_sections(monkeypatch):
    calls = []

    async def news(query, timespan="3w"):
        calls.append(query)
        return "- 2026-10-20 (a.com): Headline"

    monkeypatch.setattr(r, "gdelt_headlines", news)
    monkeypatch.setattr(r, "wikipedia_extracts", lambda titles, q: f"### {titles[0]} (Wikipedia)\nlead")
    monkeypatch.setattr(r, "fetch_source", lambda url: "Latest: 42")
    monkeypatch.setattr(r, "bls_latest", lambda s: "- BLS X: 1" if s else "")

    class Planner:
        async def invoke(self, prompt):
            return '{"news_query": "Gatun Lake level", "wiki_titles": ["Gatun Lake"], "bls_series": []}'

    q = SimpleNamespace(question_text="Gatun?", resolution_criteria="https://pancanal.com/en/water", fine_print="",
                        background_info="", page_url="u")
    out = asyncio.run(r.gather_research(q, Planner(), today="2026-10-21"))
    assert calls == ["Gatun Lake level"]
    assert "Headline" in out and "### Gatun Lake" in out and "Latest: 42" in out and "BLS" not in out


############################## polling loop ##############################


class FakeClient:
    def __init__(self, open_by_pass):
        self.open_by_pass = open_by_pass  # list of {tournament: [questions]}
        self.calls = 0

    def get_all_open_questions_from_tournament(self, t):
        self.calls += 1
        idx = min((self.calls - 1) // 2, len(self.open_by_pass) - 1)
        return self.open_by_pass[idx].get(t, [])


class FakeBot:
    def __init__(self, client, fail_ids=()):
        self.metaculus_client = client
        self.fail_ids = set(fail_ids)
        self.forecast = []

    async def forecast_questions(self, qs, return_exceptions=True):
        out = []
        for q in qs:
            self.forecast.append(q.id_of_question)
            out.append(RuntimeError("503") if q.id_of_question in self.fail_ids else "report")
        return out


def Q(i, done=False):
    return SimpleNamespace(id_of_question=i, already_forecasted=done)


def test_poll_single_pass_forecasts_both_tournaments(monkeypatch):
    monkeypatch.setattr(g, "MODE", "openrouter")
    client = FakeClient([{g.SEASONAL_TOURNAMENT: [Q(1), Q(2, done=True)], g.MINIBENCH_TOURNAMENT: [Q(3)]}])
    bot = FakeBot(client)
    reports = asyncio.run(g.poll_tournaments(bot, minutes=0, every=12))
    assert bot.forecast == [1, 3] and reports == ["report", "report"]


def test_poll_retries_a_failed_question_at_most_twice(monkeypatch):
    monkeypatch.setattr(g, "MODE", "openrouter")
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(g.asyncio, "sleep", fake_sleep)
    clock = iter(range(0, 10_000, 60))
    monkeypatch.setattr(g.time, "monotonic", lambda: next(clock) * 1.0)
    client = FakeClient([{g.MINIBENCH_TOURNAMENT: [Q(7)]}])
    bot = FakeBot(client, fail_ids={7})
    asyncio.run(g.poll_tournaments(bot, minutes=6, every=1))
    assert bot.forecast == [7, 7]  # two failures, then left alone
    assert slept and all(s == 60 for s in slept)


def test_bot_run_research_uses_free_research(monkeypatch):
    """The real method (regression: a local variable named `research` once shadowed the module)."""
    async def fake_gather(question, planner):
        return "FREE RESEARCH BLOCK"

    monkeypatch.setattr(g.free_research, "gather_research", fake_gather)
    monkeypatch.setattr(g, "USE_FREE_RESEARCH", True)
    monkeypatch.setattr(g, "NO_RESEARCH", True)
    fake_self = SimpleNamespace(_concurrency_limiter=asyncio.Semaphore(2), get_llm=lambda *a: None)
    q = SimpleNamespace(page_url="u", question_text="q")
    out = asyncio.run(g.GigbotBot.run_research(fake_self, q))
    assert out == "FREE RESEARCH BLOCK"


def test_source_urls_keeps_balanced_parentheses_and_drops_markdown_closers():
    text = r"See [EC](https://en.wikipedia.org/wiki/Electoral_College_\(United_States\)) and (https://a.org/x)."
    assert r.source_urls(text) == ["https://en.wikipedia.org/wiki/Electoral_College_(United_States)", "https://a.org/x"]


def test_overloaded_error_detection():
    import gemini_budget as gb
    assert gb.is_overloaded_error("ServiceUnavailableError", "")
    assert gb.is_overloaded_error("X", '{"code": 503, "message": "This model is currently experiencing high demand"}')
    assert not gb.is_overloaded_error("RateLimitError", "429 RESOURCE_EXHAUSTED")


def test_gemini_llm_falls_back_once_on_503(monkeypatch):
    calls = []

    async def fake_super_invoke(self, prompt, system_prompt=None):
        calls.append(self.model)
        if self.model.endswith("3.1-flash-lite"):
            raise RuntimeError('{"code": 503, "status": "UNAVAILABLE"}')
        return "ok"

    monkeypatch.setattr(g.GeneralLlm, "invoke", fake_super_invoke)
    monkeypatch.setattr(g.GeminiLlm, "overloaded_until", {})
    monkeypatch.setattr(g, "GEMINI_BUDGET", g.gemini_budget.Budget(cap=10, used_at_start=0, today="d"))
    llm = g.make_llm("gemini/gemini-3.1-flash-lite", temperature=0.1)
    assert asyncio.run(llm.invoke("p")) == "ok"
    assert calls == ["gemini/gemini-3.1-flash-lite", "gemini/gemini-3.5-flash-lite"]
    assert g.GEMINI_BUDGET.used_this_run == 2


def test_overloaded_model_is_skipped_during_cooldown(monkeypatch):
    calls = []

    async def fake_super_invoke(self, prompt, system_prompt=None):
        calls.append(self.model)
        if self.model.endswith("3.1-flash-lite"):
            raise RuntimeError("503 UNAVAILABLE")
        return "ok"

    monkeypatch.setattr(g.GeneralLlm, "invoke", fake_super_invoke)
    monkeypatch.setattr(g.GeminiLlm, "overloaded_until", {})
    monkeypatch.setattr(g, "GEMINI_BUDGET", g.gemini_budget.Budget(cap=10, used_at_start=0, today="d"))
    llm = g.make_llm("gemini/gemini-3.1-flash-lite", temperature=0.1)
    asyncio.run(llm.invoke("p"))
    asyncio.run(llm.invoke("p"))
    assert calls == ["gemini/gemini-3.1-flash-lite", "gemini/gemini-3.5-flash-lite", "gemini/gemini-3.5-flash-lite"]
    assert g.GEMINI_BUDGET.used_this_run == 3
