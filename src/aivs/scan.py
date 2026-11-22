"""Run a scan: questions x engines x runs, cache every raw answer, score, aggregate.

Every answer and every judgement is written to runs/<scan_id>/ as JSON the moment
it arrives, so a scan can be interrupted and resumed, re-scored with a different
judge, or re-reported without asking the engines again. The raw text is the
evidence; the report links back to it.

The report is built from EVERY answer cached under the scan id, not only the
engines of the current invocation: `--engines web` today and `--engines api`
tomorrow with the same --scan-id give one report with both groups. Re-scoring
cached answers never asks an engine anything.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .engines.base import Engine
from .metrics import aggregate, findings
from .models import AnswerScore, Brief, EngineAnswer, ScanResult
from .scoring import JUDGE_SYSTEM, Judge, score_answer

Progress = Callable[[str], None]


def new_scan_id(brief: Brief) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    return f"{brief.client.name.lower().replace(' ', '-')}-{stamp}"


class ScanRunner:
    def __init__(self, settings: Settings, progress: Progress | None = None) -> None:
        self._settings = settings
        self._say = progress or (lambda _msg: None)

    def run(
        self,
        brief: Brief,
        engines: list[Engine],
        runs: int = 1,
        judge: Judge | None = None,
        scan_id: str | None = None,
        limit: int | None = None,
        brief_path: str = "",
    ) -> ScanResult:
        scan_id = scan_id or new_scan_id(brief)
        root = self._settings.runs_dir / scan_id
        (root / "answers").mkdir(parents=True, exist_ok=True)
        (root / "scores").mkdir(parents=True, exist_ok=True)
        questions = brief.questions[:limit] if limit else brief.questions

        answers: list[EngineAnswer] = []
        try:
            for eng in engines:
                for q in questions:
                    for r in range(1, runs + 1):
                        path = root / "answers" / f"{eng.name}--{q.id}--r{r}.json"
                        if path.exists():
                            ans = EngineAnswer.model_validate_json(path.read_text())
                            if not ans.error:
                                answers.append(ans)
                                self._say(f"[cached] {eng.name} · {q.id} · run {r}")
                                continue
                        self._say(f"[ask] {eng.name} · {q.id} · run {r}: {q.text}")
                        ans = eng.ask(q, brief, r) if _accepts_run(eng) else eng.ask(q, brief)
                        ans.run = r
                        path.write_text(ans.model_dump_json(indent=1))
                        if ans.error:
                            self._say(f"   ! {ans.error}")
                        else:
                            self._say(f"   {len(ans.text)} chars, {len(ans.citations)} sources, {ans.latency_s}s")
                        answers.append(ans)
        finally:
            for eng in engines:  # browser engines hold a Chrome open; API engines have nothing to close
                close = getattr(eng, "close", None)
                if callable(close):
                    close()

        answers, engine_names, runs = _everything_cached(root, [e.name for e in engines], answers, runs)

        qmap = {q.id: q for q in brief.questions}
        # Judge scores depend on what the judge was told (verified facts, attributes, brands),
        # so the cache key carries a digest of that: edit a fact and the judge runs again.
        judge_tag = f"{judge.model}@{judge_inputs_digest(brief)}" if judge else "none"
        scores = self._score_all(answers, qmap, brief, judge, judge_tag, root)

        per_engine = {e: aggregate(brief, answers, scores, e) for e in engine_names}
        overall = aggregate(brief, answers, scores, None)
        result = ScanResult(
            scan_id=scan_id,
            brief=brief,
            engines=engine_names,
            runs=runs,
            answers=answers,
            scores=scores,
            per_engine=per_engine,
            overall=overall,
            findings=findings(brief, per_engine, overall),
            synthetic=any(a.synthetic for a in answers),
            judge_model=judge.model if judge else "none",
            brief_path=brief_path,
        )
        (root / "scan.json").write_text(result.model_dump_json(indent=1))
        return result


    def _score_all(self, answers, qmap, brief: Brief, judge: Judge | None, judge_tag: str, root: Path) -> list[AnswerScore]:
        """Score every answer, reusing cached scores. Uncached ones go to the judge in
        parallel (API calls, independent of each other); each result is written to its
        own file as it arrives, so an interrupted rescore resumes where it stopped."""
        from concurrent.futures import ThreadPoolExecutor

        def path_for(ans: EngineAnswer) -> Path:
            return root / "scores" / f"{ans.engine}--{ans.question_id}--r{ans.run}--{_safe(judge_tag)}.json"

        def score_one(ans: EngineAnswer) -> AnswerScore:
            self._say(f"[score] {ans.engine} · {ans.question_id} · run {ans.run}")
            sc = score_answer(ans, qmap[ans.question_id], brief, judge)
            if sc.judge_error:
                self._say(f"   ! judge: {sc.judge_error[:160]}")
            path_for(ans).write_text(sc.model_dump_json(indent=1))
            return sc

        def usable(ans: EngineAnswer) -> bool:
            """A cached score counts unless the judge failed on it; those are retried."""
            p = path_for(ans)
            if not p.exists():
                return False
            if judge is None:
                return True
            return AnswerScore.model_validate_json(p.read_text()).judge_error is None

        todo = [a for a in answers if not usable(a)]
        workers = max(1, self._settings.judge_workers) if judge is not None else 1
        fresh: dict[tuple[str, str, int], AnswerScore] = {}
        if todo:
            with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
                for sc in pool.map(score_one, todo):
                    fresh[(sc.engine, sc.question_id, sc.run)] = sc
        out: list[AnswerScore] = []
        for ans in answers:
            key = (ans.engine, ans.question_id, ans.run)
            out.append(fresh[key] if key in fresh else AnswerScore.model_validate_json(path_for(ans).read_text()))
        return out


def judge_inputs_digest(brief: Brief) -> str:
    """Short hash of everything the judge sees besides the answer: verified facts,
    attributes, tracked brand names and aliases. Part of the score-cache key."""
    payload = {
        "facts": sorted((f.id, f.statement) for f in brief.facts if f.status == "verified"),
        "attributes": list(brief.attributes),
        "brands": [(b.name, sorted(b.aliases)) for b in brief.tracked()],
        "client": brief.client.name,
        "judge_prompt": hashlib.sha1(JUDGE_SYSTEM.encode()).hexdigest()[:8],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def _everything_cached(
    root: Path, asked: list[str], fresh: list[EngineAnswer], runs: int
) -> tuple[list[EngineAnswer], list[str], int]:
    """All answers on disk under runs/<scan_id>/answers, with the ones from this
    invocation taking precedence (same file either way). Engines are ordered as
    asked, then any others found on disk; `runs` is the highest run seen."""
    by_key: dict[tuple[str, str, int], EngineAnswer] = {}
    for path in sorted((root / "answers").glob("*.json")):
        try:
            ans = EngineAnswer.model_validate_json(path.read_text())
        except ValueError:
            continue  # a half-written or foreign file never sinks the report
        by_key[(ans.engine, ans.question_id, ans.run)] = ans
    for ans in fresh:
        by_key[(ans.engine, ans.question_id, ans.run)] = ans
    answers = sorted(by_key.values(), key=lambda a: (a.engine, a.question_id, a.run))
    on_disk = sorted({a.engine for a in answers} - set(asked))
    engine_names = [e for e in asked if any(a.engine == e for a in answers)] + on_disk
    max_run = max((a.run for a in answers), default=runs)
    return answers, engine_names, max(runs, max_run)


def load_scan(path: str | Path) -> ScanResult:
    p = Path(path)
    if p.is_dir():
        p = p / "scan.json"
    return ScanResult.model_validate_json(p.read_text())


def _accepts_run(eng: Engine) -> bool:
    import inspect

    try:
        return "run" in inspect.signature(eng.ask).parameters
    except (TypeError, ValueError):
        return False


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def dump_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
