"""Grok API stand-in: xAI Responses API with the server-side `web_search` tool.

Docs (read 21 Nov 2026, docs.x.ai): OpenAI SDK with base_url https://api.x.ai/v1;
`client.responses.create(model="grok-4.7", input=[...], tools=[{"type": "web_search"}])`.
Citations come back as `url_citation` annotations on the message's output_text
(url, start_index, end_index; `title` is the citation number, not a page title).
Optional tool filters: allowed_domains / excluded_domains (max 5, not both).
No user_location parameter is documented, so the market comes from the prompt
alone. Web search is billed at $5 per 1,000 calls on top of tokens.
"""

from __future__ import annotations

from ..config import Settings
from ..models import Brief, EngineAnswer, Question
from .base import error_answer, shopper_prompt, timed
from .openai_engine import parse_responses_output

XAI_BASE_URL = "https://api.x.ai/v1"


class GrokEngine:
    name = "grok"

    def __init__(self, settings: Settings) -> None:
        if not settings.xai_api_key:
            raise RuntimeError("XAI_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.xai_api_key, base_url=XAI_BASE_URL, timeout=settings.request_timeout_s)
        self._settings = settings
        self.model = settings.xai_model

    def ask(self, question: Question, brief: Brief, run: int = 1) -> EngineAnswer:
        try:
            resp, secs = timed(
                lambda: self._client.responses.create(
                    model=self.model,
                    tools=[{"type": "web_search"}],
                    input=[{"role": "user", "content": shopper_prompt(question, brief)}],
                )
            )
        except Exception as exc:  # noqa: BLE001
            return error_answer(self.name, self.model, question, run, exc)
        text, citations, queries = parse_responses_output(resp)
        return EngineAnswer(
            engine=self.name,
            model=self.model,
            question_id=question.id,
            run=run,
            text=text,
            citations=citations,
            search_queries=queries,
            latency_s=secs,
        )
