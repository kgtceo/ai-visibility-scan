"""Offline engine for tests and the no-keys demo.

Answers come from `fixtures/<brand>.json`, keyed by question id. They are written
by hand to look like answer-engine output and are flagged `synthetic=True` so the
report labels them. A live scan replaces them.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Brief, Citation, EngineAnswer, Question

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"


class FakeEngine:
    name = "fake"
    model = "fixture"
    access = "fixture"

    def __init__(self, fixture_path: Path | None = None, engine_name: str | None = None) -> None:
        self._path = fixture_path
        self._data: dict | None = None
        if engine_name:
            self.name = engine_name

    def _load(self, brief: Brief) -> dict:
        if self._data is None:
            path = self._path or FIXTURES / f"{brief.client.name.lower().replace(' ', '-')}.json"
            self._data = json.loads(path.read_text()) if path.exists() else {}
        return self._data

    def ask(self, question: Question, brief: Brief, run: int = 1) -> EngineAnswer:
        data = self._load(brief)
        entry = data.get(self.name, {}).get(question.id) or data.get("default", {}).get(question.id)
        if entry is None:
            return EngineAnswer(
                engine=self.name,
                model=self.model,
                question_id=question.id,
                run=run,
                text="",
                error="no fixture for this question",
                synthetic=True,
                access=self.access,
            )
        return EngineAnswer(
            engine=self.name,
            model=self.model,
            question_id=question.id,
            run=run,
            text=entry["text"],
            citations=[Citation(**c) for c in entry.get("citations", [])],
            search_queries=entry.get("search_queries", []),
            synthetic=True,
            access=self.access,
        )
