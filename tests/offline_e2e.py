"""
Offline end-to-end test of gigbot_bot.py: the real bot pipeline (research -> 5 forecasts ->
parsing -> aggregation -> report) on hand-built binary, multiple-choice and numeric questions,
with no Metaculus or OpenRouter access. Every LLM call is routed to the local `claude` CLI
(Haiku), and nothing is posted.

    .venv/bin/python tests/offline_e2e.py
"""

import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("METACULUS_TOKEN", "offline-test")

from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericQuestion  # noqa: E402

import gigbot_bot  # noqa: E402

calls: Counter = Counter()


async def fake_invoke(self, prompt, system_prompt=None):
    if isinstance(prompt, list):
        prompt = "\n\n".join(f"{m.get('role')}: {m.get('content')}" for m in prompt)
    if system_prompt:
        prompt = system_prompt + "\n\n" + prompt
    calls[self.model] += 1
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", "--model", "claude-haiku-4-5-20251001", "--setting-sources", "",
        "--strict-mcp-config", "--tools", "", "--no-session-persistence",
        "--system-prompt", "You have no tools or web access. Answer in plain text.",
        "--output-format", "json",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/tmp",
    )
    out, err = await proc.communicate(str(prompt).encode())
    if proc.returncode:
        raise RuntimeError(err.decode()[-300:])
    return json.loads(out)["result"]


GeneralLlm.invoke = fake_invoke

close = datetime.now(timezone.utc) + timedelta(days=60)
questions = [
    BinaryQuestion(
        question_text="Will the US Federal Reserve cut the federal funds target range at its December 2026 meeting?",
        page_url="offline://binary",
        resolution_criteria="Resolves Yes if the FOMC statement after its December 2026 meeting lowers the target range.",
        fine_print="",
        background_info="The FOMC meets eight times a year.",
        close_time=close,
        scheduled_resolution_time=close,
    ),
    MultipleChoiceQuestion(
        question_text="Which party will win the most seats in the next UK general election?",
        page_url="offline://mc",
        options=["Labour", "Conservative", "Reform UK", "Other"],
        resolution_criteria="Resolves to the party winning the most House of Commons seats.",
        fine_print="",
        background_info="",
        close_time=close,
        scheduled_resolution_time=close,
    ),
    NumericQuestion(
        question_text="What will the US unemployment rate (U-3, seasonally adjusted) be for November 2026?",
        page_url="offline://numeric",
        resolution_criteria="Resolves to the BLS figure first published for November 2026.",
        fine_print="",
        background_info="",
        unit_of_measure="percent",
        lower_bound=2.0,
        upper_bound=10.0,
        open_lower_bound=True,
        open_upper_bound=True,
        close_time=close,
        scheduled_resolution_time=close,
    ),
]


async def main():
    bot = gigbot_bot.build_bot(publish=False, predictions=5)
    reports = await bot.forecast_questions(questions, return_exceptions=True)
    ok = True
    for q, r in zip(questions, reports):
        if isinstance(r, BaseException):
            ok = False
            print(f"FAIL {q.page_url}: {r!r}")
            continue
        pred = r.prediction
        if isinstance(q, BinaryQuestion):
            assert gigbot_bot.core.P_MIN <= pred <= gigbot_bot.core.P_MAX
            shown = f"{pred:.3f}"
        elif isinstance(q, MultipleChoiceQuestion):
            total = sum(o.probability for o in pred.predicted_options)
            assert abs(total - 1) < 1e-6, total
            shown = ", ".join(f"{o.option_name}={o.probability:.2f}" for o in pred.predicted_options)
        else:
            cdf = [p.percentile for p in pred.cdf]
            assert all(b >= a for a, b in zip(cdf, cdf[1:]))
            shown = "declared " + ", ".join(f"p{round(x.percentile * 100)}={x.value:.2f}" for x in pred.declared_percentiles)
        print(f"OK   {q.page_url}: {shown}  (explanation {len(r.explanation)} chars)")
    print("LLM calls by model:", dict(calls))
    return ok


sys.exit(0 if asyncio.run(main()) else 1)
