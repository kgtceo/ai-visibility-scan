"""Gemini with Grounding with Google Search (google-genai SDK, generate_content).

Docs (read 20 Nov 2026): tools=[Tool(google_search=GoogleSearch())]; default model
gemini-3.8-flash (on 21 Nov 2026 gemini-2.5-flash returned 404 "no longer available
to new users, use models/gemini-3.8-flash"). A 429 RESOURCE_EXHAUSTED on every model
with "check your plan and billing" means the key's Google project has no quota for
these models; that is fixed in AI Studio (billing on the project), not in code.
`grounding_metadata.grounding_chunks[].web.uri/title` are the sources,
`grounding_metadata.web_search_queries` the queries. Billing: per search query on
Gemini 3+ (5,000 free a month), per prompt on 2.5 and older.

Known quirk recorded for the report: grounding URIs are Google redirect links
(vertexaisearch.cloud.google.com/grounding-api-redirect/...). The chunk `title`
carries the publisher domain, so the source classifier falls back to it.
"""

from __future__ import annotations

from ..config import Settings
from ..models import Brief, Citation, EngineAnswer, Question
from .base import error_answer, shopper_prompt, timed


class GeminiEngine:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._settings = settings
        self.model = settings.gemini_model

    def ask(self, question: Question, brief: Brief, run: int = 1) -> EngineAnswer:
        from google.genai import types

        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=self._settings.max_tokens,
        )
        try:
            resp, secs = timed(
                lambda: self._client.models.generate_content(
                    model=self.model, contents=shopper_prompt(question, brief), config=config
                )
            )
        except Exception as exc:  # noqa: BLE001
            return error_answer(self.name, self.model, question, run, exc)

        citations: dict[str, Citation] = {}
        queries: list[str] = []
        cands = getattr(resp, "candidates", None) or []
        meta = getattr(cands[0], "grounding_metadata", None) if cands else None
        if meta is not None:
            for chunk in getattr(meta, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None) if web is not None else None
                if uri:
                    citations.setdefault(uri, Citation(url=uri, title=getattr(web, "title", "") or ""))
            queries = list(getattr(meta, "web_search_queries", None) or [])
        return EngineAnswer(
            engine=self.name,
            model=self.model,
            question_id=question.id,
            run=run,
            text=(getattr(resp, "text", "") or "").strip(),
            citations=list(citations.values()),
            search_queries=queries,
            latency_s=secs,
        )
