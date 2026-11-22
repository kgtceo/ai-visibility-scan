"""Perplexity through its Agent API (the Sonar chat-completions endpoint was retired;
on 21 Nov 2026 it answered every call with 403 "Sonar is now the Agent API").

Docs (read 21 Nov 2026, docs.perplexity.ai/docs/agent-api): OpenAI SDK with
base_url https://api.perplexity.ai/v1 and `client.responses.create(...)`.
`model="perplexity/sonar"` is Perplexity's own grounded model ($0.25/$2.50 per M
tokens); `tools=[{"type": "web_search", "user_location": {...}}]` turns search on
($2.50 per 1,000 invocations) and locates it. Sources arrive as an output item of
type `search_results` whose `results[]` carry id, url, title, snippet, date; the
answer may also carry `url_citation` annotations, which the shared parser reads.
Inline [n] markers in the text refer to the result ids.
"""

from __future__ import annotations

from ..config import Settings
from ..models import Brief, Citation, EngineAnswer, Question
from .base import error_answer, shopper_prompt, timed
from .openai_engine import parse_responses_output

PERPLEXITY_BASE_URL = "https://api.perplexity.ai/v1"


class PerplexityEngine:
    name = "perplexity"

    def __init__(self, settings: Settings) -> None:
        if not settings.perplexity_api_key:
            raise RuntimeError("PERPLEXITY_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.perplexity_api_key, base_url=PERPLEXITY_BASE_URL, timeout=settings.request_timeout_s
        )
        self._settings = settings
        self.model = settings.perplexity_model

    def ask(self, question: Question, brief: Brief, run: int = 1) -> EngineAnswer:
        tool = {"type": "web_search", "user_location": {"country": brief.market, "city": brief.city}}
        try:
            resp, secs = timed(
                lambda: self._client.responses.create(
                    model=self.model,
                    input=shopper_prompt(question, brief),
                    tools=[tool],
                    max_output_tokens=self._settings.max_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return error_answer(self.name, self.model, question, run, exc)
        text, citations, queries = parse_responses_output(resp)
        citations = merge_search_results(resp, citations)
        queries = queries or search_queries(resp)
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


def _field(obj, name: str, default=None):
    """Read a field from an SDK object or a plain dict (unknown output item types
    reach us as loosely typed objects, so both shapes are handled)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def merge_search_results(resp, citations: list[Citation]) -> list[Citation]:
    """Add the `search_results` output item's results, in id order, after any
    url_citation annotations already found, deduped by URL."""
    out: dict[str, Citation] = {c.url: c for c in citations}
    for item in _field(resp, "output", None) or []:
        if _field(item, "type") != "search_results":
            continue
        for r in _field(item, "results", None) or []:
            url = _field(r, "url")
            if url:
                out.setdefault(url, Citation(url=url, title=_field(r, "title", "") or ""))
    return list(out.values())


def search_queries(resp) -> list[str]:
    for item in _field(resp, "output", None) or []:
        if _field(item, "type") == "search_results":
            return [q for q in (_field(item, "queries", None) or []) if isinstance(q, str)]
    return []
