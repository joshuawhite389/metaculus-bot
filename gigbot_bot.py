"""
gigbot's FutureEval bot: the Metaculus template (main.py, FallTemplateBot2026) plus:

- Frontier models on the free Metaculus OpenRouter credits, with a two-provider ensemble
  (forecasts alternate between the models in FORECASTER_MODELS).
- A research step that asks for base rates, the current status and scheduled dates, using
  OpenRouter's native web search (the ":online" suffix, covered by the credits). AskNews is
  added on top if its keys are set.
- Binary questions: the outside-view prompt and mean-of-log-odds aggregation from
  forecaster_core.py (backtested; see ../BACKTEST.md).
- A credit guard: reads the OpenRouter key's remaining credit before each run, cuts the
  ensemble when credit runs low and stops cleanly before it runs out. No paid key is ever used
  unless someone sets one.
- A key gate (provider_keys.py): a run skips quietly, saying which key is missing, until every
  configured model's provider key is set; it starts forecasting automatically once it is. The
  Metaculus LLM proxy ("metaculus/..." models) is reported as unusable: its hostname no longer
  exists (checked 2026-10-07).
- `--check-only`: proves METACULUS_TOKEN authenticates and lists the open bot-testing-area
  questions, without calling any LLM or posting anything.

Everything else (question fetching, numeric/multiple-choice/date handling, posting the private
reasoning comment) is the template's, unchanged.

Run:  python gigbot_bot.py --mode tournament | test_questions | metaculus_cup  [--dry-run] [--check-only]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import os
from datetime import datetime

import requests

from bot_helpers import check_environment, print_run_summary_banner, print_startup_banner
from main import FallTemplateBot2026  # also runs the template's dependency silencing
from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    PredictionTypes,
    ReasonedPrediction,
)

import forecaster_core as core
import provider_keys

logger = logging.getLogger(__name__)

# Model names are litellm/OpenRouter ids. Override any of them with env vars (GitHub repo
# variables) without touching code. Only Anthropic, OpenAI and Google models work on the
# Metaculus-donated OpenRouter key.
# An unset GitHub repo variable arrives as an empty string, so treat "" as unset.
def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


FORECASTER_MODELS = [
    m.strip()
    for m in _env("FORECASTER_MODELS", "openrouter/anthropic/claude-sonnet-5.5,openrouter/openai/gpt-6.1-sol").split(",")
    if m.strip()
]
RESEARCHER_MODEL = _env("RESEARCHER_MODEL", "openrouter/openai/gpt-6.1-sol:online")
PARSER_MODEL = _env("PARSER_MODEL", "openrouter/openai/gpt-6-luna")
PREDICTIONS_PER_QUESTION = int(_env("PREDICTIONS_PER_QUESTION", "5"))

# Credit guard, in US dollars of OpenRouter credit.
STOP_BELOW = float(_env("CREDIT_STOP_BELOW", "3"))
REDUCE_BELOW = float(_env("CREDIT_REDUCE_BELOW", "15"))


def openrouter_credit_remaining() -> float | None:
    """Remaining credit on the OpenRouter key, or None if unknown/unlimited."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        remaining = data.get("limit_remaining")
        return float(remaining) if remaining is not None else None
    except Exception as e:  # never block a run on the check itself
        logger.warning(f"Could not read OpenRouter credit: {e}")
        return None


class GigbotBot(FallTemplateBot2026):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forecasters = itertools.cycle(
            [GeneralLlm(model=m, temperature=0.5, timeout=180, allowed_tries=3) for m in FORECASTER_MODELS]
        )

    # Every template forecast function calls get_llm("default", "llm"); alternating here gives
    # every question type a two-provider ensemble.
    def get_llm(self, purpose: str = "default", guarantee_type=None):  # type: ignore[override]
        if purpose == "default" and guarantee_type == "llm":
            return next(self._forecasters)
        return super().get_llm(purpose, guarantee_type)

    ################################ RESEARCH ################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            parts = []
            try:
                web = await self.get_llm("researcher", "llm").invoke(research_prompt(question))
                parts.append(web)
            except Exception as e:
                logger.warning(f"Web research failed for {question.page_url}: {e}")
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                try:
                    news = await AskNewsSearcher().call_preconfigured_version(
                        "asknews/news-summaries", question.question_text
                    )
                    parts.append("News summaries (AskNews):\n" + news)
                except Exception as e:
                    logger.warning(f"AskNews failed for {question.page_url}: {e}")
            research = "\n\n".join(parts) or "No research could be retrieved."
            logger.info(f"Research for {question.page_url}:\n{research}")
            return research

    ################################ BINARY ################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = core.improved_binary_prompt(
            question.question_text,
            question.background_info or "",
            question.resolution_criteria or "",
            (question.fine_print or "") + "\n" + self._get_conditional_disclaimer_if_necessary(question),
            research,
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _aggregate_predictions(
        self, predictions: list[PredictionTypes], question: MetaculusQuestion
    ) -> PredictionTypes:
        if isinstance(question, BinaryQuestion):
            probs = [float(p) for p in predictions]  # type: ignore[arg-type]
            final = core.aggregate_binary(probs)
            logger.info(f"{question.page_url}: samples {probs} -> {final:.3f}")
            return final  # type: ignore[return-value]
        return await super()._aggregate_predictions(predictions, question)


def research_prompt(question: MetaculusQuestion) -> str:
    return f"""
You are the research assistant to a superforecaster. Search the web and report facts, not a forecast.

Question: {question.question_text}

Resolution criteria: {question.resolution_criteria}

{question.fine_print or ""}

Today is {datetime.now().strftime("%Y-%m-%d")}. Report, with dates and sources:
1. The current status: the latest numbers, events or official statements that bear on the criteria.
2. Anything scheduled before the resolution date (votes, releases, deadlines, data publications).
3. Base rates: how often things like this have happened historically, and recent trends or
   volatility if the question is about a measured quantity.
4. Whether the question may already be effectively decided, and why.
5. Any public forecasts or prediction-market prices on the same or a closely related question.
Keep it under 500 words.
""".strip()


ALL_MODELS = FORECASTER_MODELS + [RESEARCHER_MODEL, PARSER_MODEL]


def check_only(client: MetaculusClient, tournament: str = "bot-testing-area") -> int:
    """Prove the Metaculus token works and show what the bot would see. Exit code: 0 ok, 1 not."""
    try:
        user_id = client.get_current_user_id()
    except Exception as e:
        print(f"❌  METACULUS_TOKEN does not authenticate: {type(e).__name__}: {str(e)[:200]}")
        return 1
    print(f"✅  METACULUS_TOKEN authenticates (bot user id {user_id}).")
    try:
        questions = client.get_all_open_questions_from_tournament(tournament)
    except Exception as e:
        print(f"❌  Could not list open questions in {tournament}: {type(e).__name__}: {str(e)[:200]}")
        return 1
    print(f"✅  {len(questions)} open question(s) in {tournament}:")
    for q in questions:
        print(f"    • {q.page_url}")
    blockers = provider_keys.run_blockers(ALL_MODELS, os.environ)
    if blockers:
        print("⏸️   LLM keys: not ready, the bot would skip this run because:")
        for b in blockers:
            print(f"    • {b}")
    else:
        print(f"✅  LLM keys: ready for {', '.join(ALL_MODELS)}")
    return 0


def build_bot(publish: bool, predictions: int) -> GigbotBot:
    return GigbotBot(
        research_reports_per_question=1,
        predictions_per_research_report=predictions,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(model=FORECASTER_MODELS[0], temperature=0.5, timeout=180, allowed_tries=3),
            "summarizer": GeneralLlm(model=PARSER_MODEL, temperature=0.3),
            "researcher": GeneralLlm(model=RESEARCHER_MODEL, temperature=0.1, timeout=240, allowed_tries=2),
            "parser": GeneralLlm(model=PARSER_MODEL, temperature=0.0),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run gigbot's forecasting bot")
    parser.add_argument("--mode", choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    parser.add_argument("--dry-run", action="store_true", help="forecast but don't post to Metaculus")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only check the Metaculus token, list open bot-testing-area questions and report key readiness",
    )
    args = parser.parse_args()

    check_environment(strict=True)
    if args.check_only:
        raise SystemExit(check_only(MetaculusClient()))
    blockers = provider_keys.run_blockers(ALL_MODELS, os.environ)
    if blockers:
        # Normal while waiting for the credits email: skip quietly instead of failing every 20 min.
        # The run starts forecasting by itself once the missing key is set as a repository secret.
        print("Not forecasting this run; a model's provider key is missing:")
        for b in blockers:
            print(f"  • {b}")
        raise SystemExit(0)
    predictions = PREDICTIONS_PER_QUESTION
    credit = openrouter_credit_remaining()
    if credit is not None:
        logger.info(f"OpenRouter credit remaining: ${credit:.2f}")
        if credit < STOP_BELOW:
            print(f"Credit ${credit:.2f} is below ${STOP_BELOW:.2f}: not forecasting this run.")
            raise SystemExit(0)
        if credit < REDUCE_BELOW:
            predictions = min(predictions, 3)
            logger.warning(f"Credit low (${credit:.2f}): using {predictions} predictions per question.")

    publish = not args.dry_run
    print_startup_banner(args.mode, will_publish=publish)
    bot = build_bot(publish, predictions)
    client = MetaculusClient()
    if args.mode == "tournament":
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports += asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
        url = "https://www.metaculus.com/tournament/fall-futureeval-2026/"
    elif args.mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
        url = "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/"
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        url = "https://www.metaculus.com/tournament/bot-testing-area/"
    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=url)
