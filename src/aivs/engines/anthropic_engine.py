"""Claude with the server-side web search tool (Messages API).

Docs (read 20 Nov 2026): tool type `web_search_20250305` (basic), `max_uses`,
`user_location`; citations arrive as `web_search_result_location` blocks with
`url`, `title`, `cited_text`; the queries Claude ran are `server_tool_use` blocks.
$10 per 1,000 searches plus tokens.
"""

from __future__ import annotations

from ..config import Settings
from ..models import Brief, Citation, EngineAnswer, Question
from .base import error_answer, shopper_prompt, timed


class ClaudeEngine:
    name = "claude"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        from anthropic import Anthropic

        self._client = Anthropic(api_key=settings.anthropic_api_key, timeout=settings.request_timeout_s)
        self._settings = settings
        self.model = settings.claude_model

    def ask(self, question: Question, brief: Brief, run: int = 1) -> EngineAnswer:
        tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": self._settings.max_searches_per_answer,
            "user_location": {"type": "approximate", "city": brief.city, "country": brief.market},
        }
        try:
            resp, secs = timed(
                lambda: self._client.messages.create(
                    model=self.model,
                    max_tokens=self._settings.max_tokens,
                    messages=[{"role": "user", "content": shopper_prompt(question, brief)}],
                    tools=[tool],
                )
            )
        except Exception as exc:  # noqa: BLE001 - one failed answer must not stop the scan
            return error_answer(self.name, self.model, question, run, exc)

        text_parts: list[str] = []
        citations: dict[str, Citation] = {}
        queries: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                for c in getattr(block, "citations", None) or []:
                    url = getattr(c, "url", None)
                    if url:
                        citations.setdefault(url, Citation(url=url, title=getattr(c, "title", "") or ""))
            elif btype == "server_tool_use":
                q = (getattr(block, "input", None) or {}).get("query")
                if q:
                    queries.append(q)
        return EngineAnswer(
            engine=self.name,
            model=self.model,
            question_id=question.id,
            run=run,
            text="".join(text_parts).strip(),
            citations=list(citations.values()),
            search_queries=queries,
            latency_s=secs,
        )
