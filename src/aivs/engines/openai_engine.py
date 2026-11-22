"""ChatGPT stand-in: OpenAI Responses API with the `web_search` tool.

Docs (read 20 Nov 2026): tools=[{"type": "web_search", "user_location": {...},
"search_context_size": "medium"}]; citations are `url_citation` annotations on the
message content (url, title, start_index, end_index). The legacy
`web_search_preview` type still exists but lacks filters.
"""

from __future__ import annotations

from ..config import Settings
from ..models import Brief, Citation, EngineAnswer, Question
from .base import error_answer, shopper_prompt, timed


class ChatGPTEngine:
    name = "chatgpt"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_s)
        self._settings = settings
        self.model = settings.openai_model

    def ask(self, question: Question, brief: Brief, run: int = 1) -> EngineAnswer:
        tool = {
            "type": "web_search",
            "user_location": {"type": "approximate", "country": brief.market, "city": brief.city},
            "search_context_size": "medium",
        }
        try:
            resp, secs = timed(
                lambda: self._client.responses.create(
                    model=self.model,
                    tools=[tool],
                    input=shopper_prompt(question, brief),
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


def parse_responses_output(resp) -> tuple[str, list[Citation], list[str]]:
    """Answer text, deduped url_citation annotations and search queries from a
    Responses API result. Shared by the OpenAI and xAI adapters (same shape).
    Falls back to joining output_text parts when the SDK's `output_text` is empty."""
    citations: dict[str, Citation] = {}
    queries: list[str] = []
    parts_text: list[str] = []
    for item in getattr(resp, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "output_text" and getattr(part, "text", None):
                    parts_text.append(part.text)
                for ann in getattr(part, "annotations", None) or []:
                    if getattr(ann, "type", None) == "url_citation" and getattr(ann, "url", None):
                        title = getattr(ann, "title", "") or ""
                        if title.isdigit():  # xAI puts the citation number here, not a page title
                            title = ""
                        citations.setdefault(ann.url, Citation(url=ann.url, title=title))
        elif itype == "web_search_call":
            action = getattr(item, "action", None)
            q = getattr(action, "query", None) if action is not None else None
            if q:
                queries.append(q)
    text = (getattr(resp, "output_text", "") or "").strip() or "\n\n".join(parts_text).strip()
    return text, list(citations.values()), queries
