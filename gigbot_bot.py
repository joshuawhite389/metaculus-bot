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
- Interim Gemini mode (2026-10-07): with no OPENROUTER_API_KEY but a GEMINI_API_KEY, the defaults
  switch to free-tier Gemini Flash-Lite models, 3 forecasts per question, no web research (Google
  Search grounding is not on the free tier), no research summary, the binary probability parsed by
  regex first (parser LLM only as fallback), a per-model rate limiter, and a hard daily cap on
  Gemini calls shared across runs through a cached usage file (gemini_budget.py). The OpenRouter
  ensemble takes over by itself the moment OPENROUTER_API_KEY exists. Repository variables still
  override every model name.

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
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    PredictionTypes,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

import forecaster_core as core
import gemini_budget
import provider_keys

logger = logging.getLogger(__name__)

# Model names are litellm/OpenRouter ids. Override any of them with env vars (GitHub repo
# variables) without touching code. Only Anthropic, OpenAI and Google models work on the
# Metaculus-donated OpenRouter key.
# An unset GitHub repo variable arrives as an empty string, so treat "" as unset.
def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


# Defaults per provider mode. RESEARCHER_MODEL "none" means: skip the web-research step.
DEFAULTS = {
    "openrouter": {
        "FORECASTER_MODELS": "openrouter/anthropic/claude-sonnet-5.5,openrouter/openai/gpt-6.1-sol",
        "RESEARCHER_MODEL": "openrouter/openai/gpt-6.1-sol:online",
        "PARSER_MODEL": "openrouter/openai/gpt-6-luna",
        "PREDICTIONS_PER_QUESTION": "5",
    },
    # Free tier: Flash-Lite only (checked 2026-10-07: gemini-3.1-flash-lite and gemini-3.5-flash-lite
    # answer on this key; Google Search grounding returns 429 RESOURCE_EXHAUSTED, so no research).
    "gemini": {
        "FORECASTER_MODELS": "gemini/gemini-3.1-flash-lite,gemini/gemini-3.5-flash-lite",
        "RESEARCHER_MODEL": "none",
        "PARSER_MODEL": "gemini/gemini-3.1-flash-lite",
        "PREDICTIONS_PER_QUESTION": "3",
    },
}


def provider_mode(env=os.environ) -> str:
    """OpenRouter whenever its key exists (the credits key wins); Gemini as the interim; else OpenRouter defaults (the key gate then skips)."""
    if (env.get("OPENROUTER_API_KEY") or "").strip():
        return "openrouter"
    if (env.get("GEMINI_API_KEY") or "").strip():
        return "gemini"
    return "openrouter"


MODE = provider_mode()
FORECASTER_MODELS = [m.strip() for m in _env("FORECASTER_MODELS", DEFAULTS[MODE]["FORECASTER_MODELS"]).split(",") if m.strip()]
RESEARCHER_MODEL = _env("RESEARCHER_MODEL", DEFAULTS[MODE]["RESEARCHER_MODEL"])
PARSER_MODEL = _env("PARSER_MODEL", DEFAULTS[MODE]["PARSER_MODEL"])
PREDICTIONS_PER_QUESTION = int(_env("PREDICTIONS_PER_QUESTION", DEFAULTS[MODE]["PREDICTIONS_PER_QUESTION"]))
NO_RESEARCH = RESEARCHER_MODEL.strip().lower() in ("none", "off", "skip")

# One shared Gemini budget per process (see gemini_budget.py). Only consulted by GeminiLlm.
GEMINI_BUDGET = gemini_budget.budget_from_env()


class GeminiLlm(GeneralLlm):
    """
    GeneralLlm that, per call: waits for the per-model rate limiter, counts the call against the shared
    daily cap, and on 429s either stops the run (daily quota, or too many 429s in a row) or lets the
    library's backoff retry (per-minute).
    """

    consecutive_rate_limits = 0  # class-wide, across models

    async def invoke(self, prompt, system_prompt=None):  # type: ignore[override]
        GEMINI_BUDGET.take()
        waited = await gemini_budget.limiter_for(self.model).acquire()
        if waited > 1:
            logger.info(f"Rate limiter held {self.model} for {waited:.0f}s")
        try:
            result = await super().invoke(prompt, system_prompt)
            GeminiLlm.consecutive_rate_limits = 0
            return result
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RateLimit" in type(e).__name__ or "RESOURCE_EXHAUSTED" in msg:
                GeminiLlm.consecutive_rate_limits += 1
                if gemini_budget.is_daily_quota_error(msg):
                    GEMINI_BUDGET.exhaust("Gemini daily quota 429 from Google: no more calls this run")
                    logger.error(GEMINI_BUDGET.exhausted_reason)
                elif GeminiLlm.consecutive_rate_limits >= gemini_budget.MAX_CONSECUTIVE_RATE_LIMITS:
                    GEMINI_BUDGET.exhaust(
                        f"{GeminiLlm.consecutive_rate_limits} Gemini 429s in a row: assuming the quota is gone, no more calls this run"
                    )
                    logger.error(GEMINI_BUDGET.exhausted_reason)
                else:
                    logger.warning(f"Gemini rate limited ({GeminiLlm.consecutive_rate_limits} in a row): {msg[:160]}")
            raise


def make_llm(model: str, **kwargs) -> GeneralLlm:
    cls = GeminiLlm if model.startswith("gemini/") else GeneralLlm
    return cls(model=model, **kwargs)

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
            [make_llm(m, temperature=0.5, timeout=180, allowed_tries=3) for m in FORECASTER_MODELS]
        )
        if MODE == "gemini":
            # Every parser validation sample is another free-tier call; one is enough.
            self._structure_output_validation_samples = 1

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
            if NO_RESEARCH:
                logger.info(f"Web research skipped (RESEARCHER_MODEL={RESEARCHER_MODEL}) for {question.page_url}")
            else:
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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prob = core.parse_probability(reasoning)
        if prob is None:
            # The prompt asks for "Probability: NN%" on the last line; only if that's missing do we spend a parser call.
            logger.warning(f"No 'Probability: NN%' line for {question.page_url}; falling back to the parser LLM")
            parsed: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=clean_indents(
                    f"""
                    The text given to you is trying to give a probability forecast for a binary question.
                    {self._create_resolved_question_parsing_message()}
                    """
                ),
            )
            prob = parsed.prediction_in_decimal
        prob = max(0.01, min(0.99, float(prob)))
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prob}.")
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

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


ALL_MODELS = FORECASTER_MODELS + ([] if NO_RESEARCH else [RESEARCHER_MODEL]) + [PARSER_MODEL]


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
    done = sum(1 for q in questions if q.already_forecasted)
    print(f"✅  {len(questions)} open question(s) in {tournament}, {done} already forecast by this bot account:")
    for q in questions:
        print(f"    • {q.page_url}  {'(forecast submitted)' if q.already_forecasted else '(no forecast yet)'}")
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
        enable_summarize_research=False,  # the summary is only decoration in the report; it cost one call per question
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": make_llm(FORECASTER_MODELS[0], temperature=0.5, timeout=180, allowed_tries=3),
            "summarizer": make_llm(PARSER_MODEL, temperature=0.3),
            # With no research, point "researcher" at the parser model: it's never invoked (see run_research),
            # but forecasting-tools would otherwise fill in a dead metaculus/ default.
            "researcher": make_llm(PARSER_MODEL if NO_RESEARCH else RESEARCHER_MODEL, temperature=0.1, timeout=240, allowed_tries=2),
            "parser": make_llm(PARSER_MODEL, temperature=0.0),
        },
    )


TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def _run(bot: GigbotBot, client: MetaculusClient, mode: str) -> list:
    if mode == "tournament":
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports += asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
        return reports
    if mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        return asyncio.run(bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
    bot.skip_previously_forecasted_questions = False
    return asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))


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
    print(f"Provider mode: {MODE}; forecasters {FORECASTER_MODELS}; researcher {RESEARCHER_MODEL}; parser {PARSER_MODEL}; {predictions} predictions/question")
    if MODE == "gemini":
        print(f"Gemini daily cap: {GEMINI_BUDGET.used_at_start}/{GEMINI_BUDGET.cap} used before this run ({GEMINI_BUDGET.today} Pacific)")
        if GEMINI_BUDGET.remaining <= 0:
            print("Gemini daily cap already reached: not forecasting this run.")
            raise SystemExit(0)
    bot = build_bot(publish, predictions)
    client = MetaculusClient()
    reports = []
    try:
        reports = _run(bot, client, args.mode)
    finally:
        if MODE == "gemini":
            print(GEMINI_BUDGET.summary())
            gemini_budget.record_usage(GEMINI_BUDGET.used_this_run)
    # Partial failures (a question that lost too many samples to 429s) are reported, not fatal: the other
    # forecasts were submitted. The run fails only when nothing succeeded although there were questions.
    bot.log_report_summary(reports, raise_errors=False)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=TOURNAMENT_URLS[args.mode])
    if reports and all(isinstance(r, BaseException) for r in reports):
        raise SystemExit(1)
