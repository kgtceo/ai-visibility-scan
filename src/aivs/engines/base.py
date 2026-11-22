"""An answer engine is anything that takes a shopper's question and returns an
answer with the sources it used. Every adapter normalises to `EngineAnswer`.

Two kinds:
- Browser engines (browser.py), no key: the consumer surfaces themselves, asked
  signed out through a real Chrome. perplexity-web, gemini-web, brave-web, duckai-web.
- Browser engine on your own login, opt-in: grok-web (Grok signed out shows a
  sign-up card). `aivs login grok` once; labelled 'signed in' in the report.
- API engines, keyed and optional: the providers' API products with web search on
  (claude, chatgpt, gemini, perplexity, grok). Closest stand-ins where a surface cannot
  be read in a browser (ChatGPT signed out sits behind a Cloudflare challenge), and
  what a client's own automation would call.
Google AI Overviews / AI Mode: CAPTCHA on first request and no API. Out of scope.
"""

from __future__ import annotations

import time
from typing import Protocol

from ..config import Settings
from ..models import Brief, EngineAnswer, Question


class Engine(Protocol):
    name: str
    model: str

    def ask(self, question: Question, brief: Brief) -> EngineAnswer: ...


def shopper_prompt(question: Question, brief: Brief) -> str:
    """The question exactly as a shopper would type it. No steering towards or away
    from any brand; the only addition is the market, because a UK shopper's
    assistant knows where they are."""
    return f"{question.text} (I'm in the {market_name(brief.market)}.)"


def market_name(code: str) -> str:
    return {"GB": "UK", "US": "US", "NL": "Netherlands", "DE": "Germany"}.get(code, code)


def timed(fn):
    """Run fn(), return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t0, 2)


def error_answer(
    engine: str, model: str, question: Question, run: int, exc: Exception, access: str = "api", signed_in: bool = False
) -> EngineAnswer:
    """A failed question is kept, labelled with how it was attempted, so the report
    can count it as attempted and never as answered."""
    return EngineAnswer(
        engine=engine,
        model=model,
        question_id=question.id,
        run=run,
        text="",
        error=f"{type(exc).__name__}: {exc}"[:500],
        access=access,
        signed_in=signed_in,
    )


def build_engines(names: list[str], settings: Settings) -> list[Engine]:
    """Instantiate the requested engines. Unknown names raise; missing keys raise
    with the variable name so the fix is obvious. Browser engines share one Chrome
    (started on first use). 'web' expands to the four keyless browser engines;
    'api' expands to every API engine that has a key in the environment, so a scan
    can run without opening Chrome at all."""
    from .browser import BROWSER_ENGINES, BrowserSession

    session: BrowserSession | None = None
    engines: list[Engine] = []
    expanded: list[str] = []
    for n in names:
        if n == "web":
            expanded.extend(Settings.keyless_engines())
        elif n == "api":
            keyed = settings.keyed_engines()
            if not keyed:
                raise ValueError(
                    "'api' expands to the API engines that have a key, and none is set. Put at least one of ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, PERPLEXITY_API_KEY, XAI_API_KEY in .env"
                )
            expanded.extend(keyed)
        else:
            expanded.append(n)
    for n in expanded:
        if n in BROWSER_ENGINES:
            session = session or BrowserSession(settings)
            engines.append(BROWSER_ENGINES[n](session, settings))
        elif n == "fake":
            from .fake_engine import FakeEngine

            engines.append(FakeEngine())
        elif n == "claude":
            from .anthropic_engine import ClaudeEngine

            engines.append(ClaudeEngine(settings))
        elif n == "chatgpt":
            from .openai_engine import ChatGPTEngine

            engines.append(ChatGPTEngine(settings))
        elif n == "gemini":
            from .gemini_engine import GeminiEngine

            engines.append(GeminiEngine(settings))
        elif n == "perplexity":
            from .perplexity_engine import PerplexityEngine

            engines.append(PerplexityEngine(settings))
        elif n == "grok":
            from .xai_engine import GrokEngine

            engines.append(GrokEngine(settings))
        else:
            raise ValueError(
                f"Unknown engine '{n}'. Known: {', '.join(BROWSER_ENGINES)} (no key), grok-web (your login), claude, chatgpt, gemini, perplexity, grok (keys), web (the four keyless), api (every keyed one), fake"
            )
    return engines
