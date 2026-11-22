from pathlib import Path

from aivs.config import Settings, load_brief
from aivs.engines.fake_engine import FakeEngine
from aivs.engines.perplexity_engine import merge_search_results, search_queries
from aivs.metrics import aggregate, findings
from aivs.report import render
from aivs.scan import ScanRunner, load_scan
from aivs.scoring import HeuristicJudge

ROOT = Path(__file__).resolve().parents[1]
BRIEF = load_brief(ROOT / "brands" / "monzo.yaml")


def _run(tmp_path, runs=1):
    settings = Settings(runs_dir=tmp_path)
    engines = [FakeEngine(engine_name="chatgpt"), FakeEngine(engine_name="claude")]
    return ScanRunner(settings).run(BRIEF, engines, runs=runs, judge=HeuristicJudge(), scan_id="t")


def test_end_to_end_offline(tmp_path):
    r = _run(tmp_path)
    assert r.synthetic is True
    assert set(r.engines) == {"chatgpt", "claude"}
    assert len(r.answers) == 2 * len(BRIEF.questions)
    assert len(r.scores) == len(r.answers)
    fixtures_answered = [a for a in r.answers if not a.error]
    assert len(fixtures_answered) >= 10
    cm = r.overall.brands["Monzo"]
    assert cm.unbranded_answers > 0
    assert cm.presence_rate is not None
    assert (tmp_path / "t" / "scan.json").exists()
    assert len(list((tmp_path / "t" / "answers").glob("*.json"))) == len(r.answers)


def test_rerun_uses_cache(tmp_path):
    r1 = _run(tmp_path)
    settings = Settings(runs_dir=tmp_path)
    seen = []
    engines = [FakeEngine(engine_name="chatgpt"), FakeEngine(engine_name="claude")]
    ScanRunner(settings, progress=seen.append).run(BRIEF, engines, runs=1, judge=HeuristicJudge(), scan_id="t")
    cached = [m for m in seen if m.startswith("[cached]")]
    answered = [a for a in r1.answers if not a.error]
    assert len(cached) == len(answered)  # errored fixtures are retried, good ones are not


def test_metrics_shape_and_share_of_voice(tmp_path):
    r = _run(tmp_path)
    m = aggregate(BRIEF, r.answers, r.scores, None)
    assert m.answers == len(r.answers)
    total_rec = sum(b.recommended_in_unbranded for b in m.brands.values())
    if total_rec:
        assert m.share_of_voice == round(m.brands["Monzo"].recommended_in_unbranded / total_rec, 3)
    assert abs(sum(m.citation_share_by_type.values()) - 1.0) < 0.02
    assert m.top_domains and m.top_domains[0][1] >= 1


def test_consistency_with_two_runs(tmp_path):
    r = _run(tmp_path, runs=2)
    for e in r.engines:
        assert r.per_engine[e].stance_consistency == 1.0  # fixtures are deterministic


def test_findings_are_explainable(tmp_path):
    r = _run(tmp_path)
    fs = findings(BRIEF, r.per_engine, r.overall)
    for f in fs:
        assert f.severity in {"high", "medium", "low"}
        assert f.title and f.detail


def test_report_renders_and_labels_synthetic(tmp_path):
    r = _run(tmp_path)
    html = render(r)
    assert "Synthetic demo data" in html
    assert "Heuristic judge" in html
    assert "AI Visibility Scan · Monzo" in html
    assert '<mark class="client">Monzo</mark>' in html
    for q in BRIEF.questions:
        assert q.text in html
    assert load_scan(tmp_path / "t").scan_id == "t"


def test_perplexity_agent_api_sources_come_from_the_search_results_item():
    from types import SimpleNamespace as NS

    from aivs.models import Citation

    resp = NS(
        output_text="Monzo and Starling [1][2].",
        output=[
            NS(type="search_results", queries=["best uk bank account"], results=[
                NS(id=1, url="https://a.com/x", title="A", snippet="..."),
                {"id": 2, "url": "https://b.org/y", "title": "B"},  # dict shape, in case the SDK leaves it raw
                NS(id=3, url="https://a.com/x", title="dup"),
            ]),
            NS(type="message", content=[NS(type="output_text", text="Monzo and Starling [1][2].", annotations=[])]),
        ],
    )
    out = merge_search_results(resp, [Citation(url="https://c.net/z", title="C")])
    assert [c.url for c in out] == ["https://c.net/z", "https://a.com/x", "https://b.org/y"]
    assert out[1].title == "A"
    assert search_queries(resp) == ["best uk bank account"]
    assert merge_search_results(NS(output=[]), []) == [] and search_queries(NS(output=None)) == []


def test_report_covers_every_engine_cached_under_the_scan_id(tmp_path):
    """`--engines web` today and `--engines api` tomorrow with one scan id give ONE report."""
    settings = Settings(runs_dir=tmp_path)
    ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt")], runs=1, judge=HeuristicJudge(), scan_id="t")
    seen = []
    r2 = ScanRunner(settings, progress=seen.append).run(BRIEF, [FakeEngine(engine_name="claude")], runs=1, judge=HeuristicJudge(), scan_id="t")
    assert r2.engines == ["claude", "chatgpt"]  # asked first, then what was already on disk
    assert len(r2.answers) == 2 * len(BRIEF.questions)
    assert set(r2.per_engine) == {"claude", "chatgpt"}
    assert r2.per_engine["chatgpt"].answers == len(BRIEF.questions)
    assert not [m for m in seen if m.startswith("[ask] chatgpt")]  # cached engines are never re-queried
    assert load_scan(tmp_path / "t").engines == ["claude", "chatgpt"]


def test_runs_reported_is_the_highest_run_cached(tmp_path):
    settings = Settings(runs_dir=tmp_path)
    ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt")], runs=2, judge=HeuristicJudge(), scan_id="t")
    r = ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="claude")], runs=1, judge=HeuristicJudge(), scan_id="t")
    assert r.runs == 2


def test_report_shows_access_badges_and_answered_over_attempted(tmp_path):
    import re

    r = _run(tmp_path)
    html = render(r)
    assert 'class="acc fixture">fixture</span>' in html
    assert html.count('class="acc api">API</span>') == 1  # the legend only: failed fixture rows take the engine's badge
    assert r.overall.errors > 0  # fixtures leave some questions unanswered on purpose
    assert re.search(r'<td class="num">\d+ / \d+ <span class="small">\(\d+ failed\)</span></td>', html)
    assert 'class="ind">n=' in html  # a rate on fewer than ten answers is marked indicative
    assert "3. By question" in html and "attributes in the brief" in html


def test_stage_rollup_and_access_label(tmp_path):
    from aivs.metrics import access_label
    from aivs.models import EngineAnswer

    r = _run(tmp_path)
    m = r.overall
    assert set(m.by_stage) == {q.stage for q in BRIEF.questions}
    assert sum(s.questions for s in m.by_stage.values()) == len(BRIEF.questions)
    judged_client_rows = sum(1 for s in r.scores if s.judgement)
    assert sum(s.answers + s.branded_answers for s in m.by_stage.values()) == judged_client_rows
    assert m.by_stage["switch"].answers == 0  # both switch questions are branded: nothing in the unbranded column
    assert m.access_label == "fixture"
    web_out = EngineAnswer(engine="e", model="m", question_id="D1", run=1, text="x", access="web")
    web_in = EngineAnswer(engine="e", model="m", question_id="D1", run=1, text="x", access="web", signed_in=True)
    api = EngineAnswer(engine="e", model="m", question_id="D1", run=1, text="x")
    assert access_label([web_out]) == "web · signed out"
    assert access_label([web_in]) == "web · signed in"
    assert access_label([api]) == "API"
    assert access_label([api, web_out]) == "mixed"
    assert access_label([]) == ""
    failed_default = EngineAnswer(engine="e", model="m", question_id="D2", run=1, text="", error="LoginWall: x")
    assert access_label([web_out, failed_default]) == "web · signed out"  # a failure never flips the badge
    assert access_label([failed_default]) == "API"


def test_runner_with_no_engines_rescores_the_cache_and_asks_nothing(tmp_path):
    settings = Settings(runs_dir=tmp_path)
    ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt")], runs=1, judge=HeuristicJudge(), scan_id="t")
    seen = []
    r = ScanRunner(settings, progress=seen.append).run(BRIEF, [], runs=1, judge=None, scan_id="t")
    assert r.engines == ["chatgpt"]
    assert len(r.answers) == len(BRIEF.questions)
    assert not [m for m in seen if m.startswith("[ask]")]
    assert r.judge_model == "none" and all(s.judgement is None for s in r.scores)


def test_scoring_runs_in_parallel_and_caches_per_answer(tmp_path):
    import threading

    class _SlowJudge:
        model = "slow"

        def __init__(self):
            self.seen = set()
            self.lock = threading.Lock()

        def judge(self, answer, question, brief, present):
            from aivs.models import AnswerJudgement

            with self.lock:
                self.seen.add(threading.get_ident())
            return AnswerJudgement(brands=[])

    settings = Settings(runs_dir=tmp_path, judge_workers=4)
    j = _SlowJudge()
    r = ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt")], runs=1, judge=j, scan_id="t")
    assert len(r.scores) == len(r.answers)
    assert len(j.seen) > 1  # more than one worker thread judged
    assert len(list((tmp_path / "t" / "scores").glob("*--slow_*.json"))) == len(r.answers)  # judge model + brief digest; _safe() turns @ into _
    seen_before = set(j.seen)
    ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt")], runs=1, judge=j, scan_id="t")
    assert j.seen == seen_before  # second pass: every score came from the cache


def test_gemini_web_notes_an_answer_without_sources():
    from aivs.engines.browser import GeminiWeb
    from aivs.models import Citation

    class _Page:
        def __init__(self, n):
            self._n = n

        def locator(self, sel):
            n = self._n

            class L:
                def count(self):
                    return n

            return L()

    g = GeminiWeb.__new__(GeminiWeb)
    assert g._note(_Page(0), []) == GeminiWeb.NO_SOURCES_NOTE
    assert g._note(_Page(3), []) == ""  # grounding UI present, extraction just found nothing
    assert g._note(_Page(0), [Citation(url="https://a.com")]) == ""


def test_rescore_retries_answers_whose_judge_failed_last_time(tmp_path):
    from aivs.models import AnswerJudgement

    class _FlakyJudge:
        model = "flaky"

        def __init__(self):
            self.calls = 0
            self.fail = True

        def judge(self, *a, **k):
            self.calls += 1
            if self.fail:
                raise RuntimeError("400 tool_choice not supported")
            return AnswerJudgement(brands=[])

    settings = Settings(runs_dir=tmp_path, judge_workers=1)
    j = _FlakyJudge()
    r1 = ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt")], runs=1, judge=j, scan_id="t")
    answered = [a for a in r1.answers if not a.error]
    assert all(s.judge_error for s in r1.scores if s.judgement is None and not next(a for a in r1.answers if a.question_id == s.question_id).error)
    j.fail = False
    r2 = ScanRunner(settings).run(BRIEF, [], runs=1, judge=j, scan_id="t")
    assert j.calls == 2 * len(answered)  # every failed one was judged again, nothing else
    assert all(s.judge_error is None for s in r2.scores)


def test_editing_a_verified_fact_changes_the_score_cache_key_and_rejudges(tmp_path):
    from aivs.scan import judge_inputs_digest

    class _CountingJudge:
        model = "count"
        calls = 0

        def judge(self, *a, **k):
            from aivs.models import AnswerJudgement

            type(self).calls += 1
            return AnswerJudgement(brands=[])

    settings = Settings(runs_dir=tmp_path, judge_workers=1)
    j = _CountingJudge()
    r1 = ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt")], runs=1, judge=j, scan_id="t")
    n = _CountingJudge.calls
    assert n > 0 and r1.judge_model == "count"
    ScanRunner(settings).run(BRIEF, [], runs=1, judge=j, scan_id="t")
    assert _CountingJudge.calls == n  # same brief: all cached
    edited = BRIEF.model_copy(deep=True)
    edited.facts[0].statement = edited.facts[0].statement + " (corrected)"  # a verified fact changed
    assert judge_inputs_digest(edited) != judge_inputs_digest(BRIEF)
    ScanRunner(settings).run(edited, [], runs=1, judge=j, scan_id="t")
    assert _CountingJudge.calls == 2 * n  # a verified fact changed: every answer judged again


def test_label_template_is_blind_sampled_and_carries_the_answer_text(tmp_path):
    import csv

    from typer.testing import CliRunner

    from aivs.cli import app

    _run(tmp_path)
    out = tmp_path / "labels.csv"
    res = CliRunner().invoke(app, ["label-template", str(tmp_path / "t"), "--out", str(out), "--sample", "6", "--seed", "1"])
    assert res.exit_code == 0, res.output
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6
    assert set(rows[0].keys()) == {"engine", "question_id", "run", "question", "answer_text", "human_stance"}
    assert all(r["answer_text"] and r["human_stance"] == "" for r in rows)
    assert len({r["engine"] for r in rows}) == 2  # spread across both fixture engines
    assert "recommended =" in res.output and "mentioned   =" in res.output


def test_findings_group_contradictions_name_actions_and_watch_untracked_brands(tmp_path):
    from aivs.metrics import aggregate, findings
    from aivs.models import AnswerJudgement, BrandJudgement, Claim

    class _FactJudge:
        model = "facts"

        def judge(self, answer, question, brief, present):
            rows = [BrandJudgement(brand=b, stance="mentioned" if p else "absent") for b, p in present.items()]
            claims = [Claim(text="FSCS protects up to £85,000", verdict="contradicted", fact_id="F5", note="limit is £120,000")]
            return AnswerJudgement(brands=rows, client_claims=claims)

    settings = Settings(runs_dir=tmp_path, judge_workers=2)
    r = ScanRunner(settings).run(BRIEF, [FakeEngine(engine_name="chatgpt"), FakeEngine(engine_name="claude")], runs=1, judge=_FactJudge(), scan_id="t")
    m = aggregate(BRIEF, r.answers, r.scores, None)
    g = m.contradictions["F5"]
    assert g.count == m.claims_contradicted and set(g.engines) == {"chatgpt", "claude"} and g.examples
    fs = findings(BRIEF, r.per_engine, m)
    grouped = [f for f in fs if f.title.startswith(f"{g.count} claims contradict a verified fact:")]
    assert grouped and grouped[0].severity == "high" and "F5" not in grouped[0].title and not grouped[0].title.endswith("deposi")
    assert g.answers == g.answers_with_sources + (g.answers - g.answers_with_sources) and g.answers > 0
    assert "answers carrying it" in grouped[0].action or "showed no source" in grouped[0].action
    assert all(f.action for f in fs if f.severity != "low" or "further contradicted" in f.title)
    assert [f.severity for f in fs] == sorted((f.severity for f in fs), key={"high": 0, "medium": 1, "low": 2}.get)
    # watchlist: counted by name in unbranded answers only, never judged
    assert set(m.watch) == {w.name for w in BRIEF.watchlist}
    unbranded = sum(1 for a in r.answers if not a.error and not next(q for q in BRIEF.questions if q.id == a.question_id).branded)
    assert all(w.unbranded_answers == unbranded for w in m.watch.values())
    html = render(r)
    assert "Also named, not tracked" in html and "Do: " in html and "Contradicted claims, by the fact they contradict" in html
    assert "3. By question" in html and "D1 · " in html
    assert "citations counted" in html


def test_c1_names_competitors_so_it_is_branded():
    q = next(q for q in BRIEF.questions if q.id == "C1")
    assert q.branded is True
    assert sum(1 for q in BRIEF.questions if not q.branded) == 8


def test_question_level_findings_and_reasons(tmp_path):
    from aivs.metrics import aggregate, findings
    from aivs.models import AnswerJudgement, BrandJudgement

    class _PickStarling:
        """Names Monzo on every unbranded answer, recommends Starling for fees, never Monzo."""

        model = "pick"

        def judge(self, answer, question, brief, present):
            rows = []
            for b, p in present.items():
                if not p:
                    rows.append(BrandJudgement(brand=b, stance="absent"))
                elif b == "Starling":
                    rows.append(BrandJudgement(brand=b, stance="recommended", attribute_sentiment={"fees": "positive"}, evidence="Go with Starling."))
                else:
                    rows.append(BrandJudgement(brand=b, stance="mentioned", attribute_sentiment={"customer_service": "negative"} if b == "Monzo" else {}))
            return AnswerJudgement(brands=rows, client_claims=[])

    settings = Settings(runs_dir=tmp_path, judge_workers=2)
    engines = [FakeEngine(engine_name="chatgpt"), FakeEngine(engine_name="claude")]  # 2 engines x 4 runs = 8 answers a question
    r = ScanRunner(settings).run(BRIEF, engines, runs=4, judge=_PickStarling(), scan_id="t")
    m = aggregate(BRIEF, r.answers, r.scores, None)
    assert set(m.by_question) == {q.id for q in BRIEF.questions if not q.branded}
    assert max(q.answers for q in m.by_question.values()) >= 8
    fs = findings(BRIEF, r.per_engine, m)
    titles = [f.title for f in fs]
    assert any(t.startswith("Named but never put forward on") for t in titles)  # Monzo named, never picked
    lead = next(f for f in fs if f.title.startswith("Starling is recommended most often"))
    assert "fees" in lead.detail and "Go with Starling." in lead.detail and "fees" in lead.action
    weak = next(f for f in fs if "customer service" in f.title.lower())
    assert "cite" in weak.action or "showed no sources" in weak.action
    assert not any("without showing any source" in t for t in titles)  # moved to the method section
    html = render(r)
    assert "no source displayed" in html or "Some answers came with no source" in html or "answers with no sources" in html


def test_label_template_verify_mode_and_evals_accept_ok(tmp_path):
    import csv
    import subprocess
    import sys

    from typer.testing import CliRunner

    from aivs.cli import app

    _run(tmp_path)
    out = tmp_path / "labels.csv"
    res = CliRunner().invoke(app, ["label-template", str(tmp_path / "t"), "--out", str(out), "--sample", "4", "--show-judge", "--include-branded"])
    assert res.exit_code == 0, res.output
    rows = list(csv.DictReader(out.open()))
    assert rows and "judge_stance" in rows[0] and "judge_evidence" in rows[0] and list(rows[0].keys())[-1] == "human_stance"
    for i, r in enumerate(rows):
        r["human_stance"] = "ok" if i else ("mentioned" if r["judge_stance"] == "recommended" else "recommended")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    proc = subprocess.run([sys.executable, str(ROOT / "evals" / "run_evals.py"), str(tmp_path / "t"), str(out)], capture_output=True, text=True)
    assert "mode: verified" in proc.stdout and f"agreement: {len(rows) - 1}/{len(rows)}" in proc.stdout, proc.stdout + proc.stderr


def test_verify_command_walks_answers_and_saves_verified_labels(tmp_path):
    import csv

    from typer.testing import CliRunner

    from aivs.cli import app

    _run(tmp_path)
    out = tmp_path / "labels.csv"
    # 4 answers: agree, correct to mentioned, skip, agree  → 3 saved, 1 corrected
    res = CliRunner().invoke(app, ["verify", str(tmp_path / "t"), "--sample", "4", "--out", str(out)], input="\nm\ns\n\n")
    assert res.exit_code == 0, res.output
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 3
    assert sum(1 for r in rows if r["human_stance"] == "ok") == 2
    assert "3 answers verified, 1 corrected" in res.output and "verified, judge shown" in res.output


def test_report_shows_the_judge_check_when_present(tmp_path):
    import json

    from aivs.report import write_report

    r = _run(tmp_path)
    root = tmp_path / "t"
    html = write_report(r, root / "report.html").read_text()
    assert "Judge check: not yet run" in html
    (root / "judge_check.json").write_text(json.dumps({"rows": 30, "agree": 28, "corrected": 2, "mode": "verified", "judge_model": "x", "labels": "l"}))
    html = write_report(r, root / "report.html").read_text()
    assert "30 answers verified by hand, 2 corrected (93% agreement" in html
