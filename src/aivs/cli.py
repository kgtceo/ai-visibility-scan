"""Command line: scan, re-report, demo, label export."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import typer
from rich.console import Console

from .config import Settings, load_brief
from .engines.base import build_engines
from .engines.fake_engine import FakeEngine
from .report import write_report
from .scan import ScanRunner, load_scan
from .scoring import ClaudeJudge, HeuristicJudge

app = typer.Typer(add_completion=False, help="AI Visibility Scan: what the answer engines say about a brand.")
console = Console()


@app.command()
def engines() -> None:
    """Show which engines can run now: browser engines need no key, API engines need one."""
    s = Settings.from_env()
    console.print(f"No key needed (browser, signed out): {', '.join(s.keyless_engines())}")
    console.print(f"With your own login (browser):       {', '.join(s.login_engines())}  (run `aivs login grok` once; opt-in, not in 'web')")
    console.print(f"With keys in the environment:        {', '.join(s.keyed_engines()) or 'none'}")
    console.print("Keys: ANTHROPIC_API_KEY (claude engine + model judge), OPENAI_API_KEY (chatgpt), GEMINI_API_KEY (gemini), PERPLEXITY_API_KEY (perplexity), XAI_API_KEY (grok)")
    console.print("Shortcuts: --engines web (the four keyless, Chrome) · --engines api (every keyed one, no Chrome) · --engines web,api or no flag for both")
    console.print(f"Judge: {'claude (model judge)' if s.anthropic_api_key else 'heuristic (set ANTHROPIC_API_KEY for the model judge)'}")
    console.print(f"Chrome: channel {s.browser_channel}, profile {s.browser_profile_dir}, {'headless' if s.browser_headless else 'visible window'}")


@app.command()
def login(site: str = typer.Argument("grok", help="grok | perplexity")) -> None:
    """Sign in to a surface once, in the scan's own Chrome profile, for the opt-in signed-in engines."""
    from .engines.browser import LOGIN_SITES, BrowserSession

    if site not in LOGIN_SITES:
        raise typer.BadParameter(f"unknown site '{site}'. Known: {', '.join(LOGIN_SITES)}")
    settings = Settings.from_env()
    settings.browser_headless = False
    session = BrowserSession(settings)
    page = session.page()
    page.goto(LOGIN_SITES[site], wait_until="domcontentloaded", timeout=60_000)
    console.print(f"A Chrome window is open on {LOGIN_SITES[site]}. Sign in with your own account there.")
    typer.prompt("When you are signed in, press Enter here", default="", show_default=False)
    try:
        signed_out = page.get_by_role("button", name=re.compile(r"^Sign ?(up|In)$", re.I)).first.is_visible(timeout=1500)
    except Exception:  # noqa: BLE001
        signed_out = False
    session.close()
    if signed_out:
        console.print(f"[yellow]The page still shows a Sign up button, so the login may not have taken. Run `aivs login {site}` again.[/yellow]")
    else:
        console.print(f"Saved in {settings.browser_profile_dir}. {site}-web will now run on your account and the report labels it 'signed in'.")


@app.command()
def questions(brief: Path = typer.Option(..., exists=True)) -> None:
    """Print the question set in the brief."""
    b = load_brief(brief)
    for q in b.questions:
        console.print(f"{q.id:>5}  {q.stage:<8} {'branded ' if q.branded else '         '} {q.text}")
    console.print(f"{len(b.questions)} questions, {sum(1 for q in b.questions if not q.branded)} unbranded")


@app.command()
def scan(
    brief: Path = typer.Option(..., exists=True, help="brands/<brand>.yaml"),
    engines: str = typer.Option("", help="comma list. 'web' = the four keyless browser engines (Chrome, ~1.5 h for a full scan). 'api' = every API engine with a key (no Chrome, minutes). Or name them: perplexity-web,gemini-web,brave-web,duckai-web, grok-web (your login), claude,chatgpt,gemini,perplexity,grok (keys). Default: web + keyed"),
    runs: int = typer.Option(1, min=1, max=5),
    judge: str = typer.Option("auto", help="auto (claude if ANTHROPIC_API_KEY else heuristic) | claude | heuristic | none"),
    scan_id: str = typer.Option(None, help="resume or name a scan"),
    limit: int = typer.Option(None, help="only the first N questions"),
    headless: bool = typer.Option(False, help="no Chrome window (Brave and Duck.ai showed bot checks headless; Perplexity and Gemini are fine)"),
) -> None:
    """Ask every engine every question, score, write runs/<scan_id>/report.html."""
    settings = Settings.from_env()
    if headless:
        settings.browser_headless = True
    names = [e.strip() for e in engines.split(",") if e.strip()] or settings.available_engines()
    b = load_brief(brief)
    engs = build_engines(names, settings)
    j = _pick_judge(judge, settings)
    runner = ScanRunner(settings, progress=console.print)
    result = runner.run(b, engs, runs=runs, judge=j, scan_id=scan_id, limit=limit, brief_path=str(brief))
    out = write_report(result, settings.runs_dir / result.scan_id / "report.html")
    _summary(result)
    console.print(f"\nReport: {out.resolve()}")


def _pick_judge(choice: str, settings: Settings):
    choice = choice.lower()
    if choice == "none":
        return None
    if choice == "heuristic":
        return HeuristicJudge()
    if choice == "claude":
        return ClaudeJudge(settings)
    if choice != "auto":
        raise typer.BadParameter("judge must be auto, claude, heuristic or none")
    if settings.anthropic_api_key:
        return ClaudeJudge(settings)
    console.print("[yellow]No ANTHROPIC_API_KEY: using the heuristic judge. Presence and sources are exact; recommended/mentioned and sentiment are approximate and the report says so.[/yellow]")
    return HeuristicJudge()


@app.command()
def demo(
    brief: Path = typer.Option(Path("brands/monzo.yaml"), exists=True),
    runs: int = typer.Option(1, min=1, max=3),
    model_judge: bool = typer.Option(False, help="use the Claude judge on the fixture answers (needs key)"),
) -> None:
    """Offline end-to-end run on fixture answers (labelled synthetic in the report)."""
    settings = Settings.from_env()
    b = load_brief(brief)
    engs = [FakeEngine(engine_name="chatgpt"), FakeEngine(engine_name="claude")]
    j = ClaudeJudge(settings) if model_judge else HeuristicJudge()
    runner = ScanRunner(settings, progress=console.print)
    result = runner.run(b, engs, runs=runs, judge=j, scan_id="demo-" + b.client.name.lower())
    out = write_report(result, settings.runs_dir / result.scan_id / "report.html")
    _summary(result)
    console.print(f"\nReport: {out.resolve()}")


@app.command()
def rescore(
    scan: Path = typer.Argument(..., exists=True, help="runs/<scan_id>"),
    judge: str = typer.Option("auto", help="auto | claude | heuristic | none"),
    brief: Path = typer.Option(None, exists=True, help="brief YAML; default: the one the scan was run from"),
) -> None:
    """Re-judge every cached answer under a scan id and rebuild scan.json and the report.
    Asks no engine anything. Re-reads the brief YAML, so edited facts count; answers whose
    judge inputs changed are judged again, the rest come from the cache."""
    root = scan if scan.is_dir() else scan.parent
    previous = load_scan(root)
    brief_path = brief or (Path(previous.brief_path) if previous.brief_path and Path(previous.brief_path).exists() else None)
    if brief_path is None:
        candidates = sorted(Path("brands").glob("*.yaml")) if Path("brands").is_dir() else []
        match = [c for c in candidates if load_brief(c).client.name == previous.brief.client.name]
        brief_path = match[0] if len(match) == 1 else None
    if brief_path is not None:
        b = load_brief(brief_path)
        console.print(f"Brief: {brief_path} ({sum(1 for f in b.facts if f.status == 'verified')} of {len(b.facts)} facts verified)")
    else:
        b = previous.brief
        console.print("[yellow]Brief: the copy saved in scan.json (pass --brief to use an edited YAML).[/yellow]")
    settings = Settings.from_env()
    j = _pick_judge(judge, settings)
    runner = ScanRunner(settings, progress=console.print)
    result = runner.run(b, [], runs=previous.runs, judge=j, scan_id=root.name, brief_path=str(brief_path or previous.brief_path))
    out = write_report(result, root / "report.html")
    _summary(result)
    console.print(f"\nReport: {out.resolve()}")


@app.command()
def report(scan: Path = typer.Argument(..., exists=True, help="runs/<scan_id> or its scan.json")) -> None:
    """Re-render the HTML report from a saved scan."""
    result = load_scan(scan)
    root = scan if scan.is_dir() else scan.parent
    out = write_report(result, root / "report.html")
    console.print(f"Report: {out.resolve()}")


LABEL_COLUMNS = ["engine", "question_id", "run", "question", "answer_text", "human_stance"]


@app.command("label-template")
def label_template(
    scan: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("evals/labels.csv")),
    sample: int = typer.Option(30, help="rows to label, spread evenly across engines (0 = every answer)"),
    seed: int = typer.Option(7, help="random seed, so the sample is reproducible"),
    include_branded: bool = typer.Option(False, help="also sample branded questions (off: unbranded only, where the recommendation rate is measured)"),
    show_judge: bool = typer.Option(False, help="verify mode: include the judge's stance and evidence; write 'ok' to confirm a row or the corrected word. Faster, but reported as verified, not blind"),
) -> None:
    """Export a blind labelling sheet: unbranded answers where the client is named, with the
    full answer text and an empty `human_stance` column. Unbranded only by default, because
    the recommendation rate in the report is computed on unbranded questions; a branded
    question ("does Monzo charge fees abroad?") gets a description, not a recommendation.
    The judge's own call is NOT in the sheet, so the labeller is not led;
    `evals/run_evals.py` joins it back from scan.json."""
    import random

    result = load_scan(scan)
    c = result.brief.client.name
    qmap = {q.id: q.text for q in result.brief.questions}
    branded = {q.id for q in result.brief.questions if q.branded}
    amap = {(a.engine, a.question_id, a.run): a for a in result.answers}
    by_engine: dict[str, list[dict]] = {}
    for s in result.scores:
        if not s.judgement or (s.question_id in branded and not include_branded):
            continue
        bj = next((x for x in s.judgement.brands if x.brand == c), None)
        if bj is None or bj.stance == "absent":
            continue
        a = amap.get((s.engine, s.question_id, s.run))
        row = {
            "engine": s.engine,
            "question_id": s.question_id,
            "run": s.run,
            "question": qmap.get(s.question_id, ""),
            "answer_text": a.text if a else "",
        }
        if show_judge:
            row["judge_stance"] = bj.stance
            row["judge_evidence"] = bj.evidence
        row["human_stance"] = ""
        by_engine.setdefault(s.engine, []).append(row)
    rng = random.Random(seed)
    for rows_e in by_engine.values():
        rng.shuffle(rows_e)
    rows: list[dict] = []
    if sample and by_engine:
        while len(rows) < sample and any(by_engine.values()):  # round-robin across engines
            for e in sorted(by_engine):
                if by_engine[e] and len(rows) < sample:
                    rows.append(by_engine[e].pop())
    else:
        rows = [r for e in sorted(by_engine) for r in by_engine[e]]
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = LABEL_COLUMNS[:-1] + (["judge_stance", "judge_evidence"] if show_judge else []) + ["human_stance"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    mode = "verify mode: the judge's call is shown" if show_judge else "blind: the judge's call is not shown"
    console.print(f"{len(rows)} rows → {out} (seed {seed}, {len(by_engine)} engines, {'branded included' if include_branded else 'unbranded questions only'}, {mode}).")
    if show_judge:
        console.print("Read each answer_text against judge_stance. In human_stance write ok if you agree, or the corrected word (recommended | mentioned).")
    else:
        console.print("Read each answer_text and fill human_stance:")
    console.print("  recommended = the answer puts the brand forward as a choice for THIS question (named as the pick, the best, a top option, 'go with', or in an endorsed shortlist)")
    console.print("  mentioned   = named without endorsement for this question (an aside, a comparison point, a caveat, a negative)")
    console.print(f"Then: .venv/bin/python evals/run_evals.py {scan} {out}")


@app.command()
def verify(
    scan: Path = typer.Argument(..., exists=True, help="runs/<scan_id>"),
    sample: int = typer.Option(30, help="answers to check, spread across engines, unbranded questions only"),
    seed: int = typer.Option(7),
    out: Path = typer.Option(Path("evals/labels.csv"), help="where the labels are saved (same format as label-template --show-judge)"),
) -> None:
    """Check the judge in the terminal, one answer at a time: the question, the answer, the
    judge's call and its quote. Press Enter if you agree, r or m to correct it (recommended /
    mentioned), s to skip, q to stop. Saves the labels and prints the agreement.
    Verify mode: the judge's call is shown, so report the result as "verified", not "blind"."""
    import random

    result = load_scan(scan)
    c = result.brief.client.name
    qtext = {q.id: q.text for q in result.brief.questions}
    branded = {q.id for q in result.brief.questions if q.branded}
    amap = {(a.engine, a.question_id, a.run): a for a in result.answers}
    by_engine: dict[str, list[tuple]] = {}
    for s in result.scores:
        if not s.judgement or s.question_id in branded:
            continue
        bj = next((x for x in s.judgement.brands if x.brand == c), None)
        if bj is None or bj.stance == "absent":
            continue
        by_engine.setdefault(s.engine, []).append((s, bj))
    rng = random.Random(seed)
    for v in by_engine.values():
        rng.shuffle(v)
    picked: list[tuple] = []
    while len(picked) < sample and any(by_engine.values()):
        for e in sorted(by_engine):
            if by_engine[e] and len(picked) < sample:
                picked.append(by_engine[e].pop())
    console.print(f"[bold]{len(picked)} answers to check.[/bold] Enter = agree with the judge · r = recommended · m = mentioned · s = skip · q = stop\n")
    rows: list[dict] = []
    agree = corrected = 0
    for i, (s, bj) in enumerate(picked, 1):
        a = amap.get((s.engine, s.question_id, s.run))
        console.rule(f"{i}/{len(picked)} · {s.engine} · {s.question_id} · run {s.run}")
        console.print(f"[bold]{qtext.get(s.question_id, '')}[/bold]\n")
        console.print((a.text if a else "")[:2500] + ("…" if a and len(a.text) > 2500 else ""))
        console.print(f"\n[cyan]Judge says {c} is {bj.stance.upper()}[/cyan]" + (f' · "{bj.evidence}"' if bj.evidence else ""))
        while True:
            key = typer.prompt("Enter / r / m / s / q", default="", show_default=False).strip().lower()
            if key in ("", "ok", "o", "r", "m", "s", "q"):
                break
        if key == "q":
            break
        if key == "s":
            continue
        human = {"": bj.stance, "ok": bj.stance, "o": bj.stance, "r": "recommended", "m": "mentioned"}[key]
        rows.append({"engine": s.engine, "question_id": s.question_id, "run": s.run, "question": qtext.get(s.question_id, ""), "answer_text": a.text if a else "", "judge_stance": bj.stance, "judge_evidence": bj.evidence, "human_stance": "ok" if human == bj.stance else human})
        agree += human == bj.stance
        corrected += human != bj.stance
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "question_id", "run", "question", "answer_text", "judge_stance", "judge_evidence", "human_stance"])
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    if n:
        import json

        root = scan if scan.is_dir() else scan.parent
        (root / "judge_check.json").write_text(json.dumps({"rows": n, "agree": agree, "corrected": corrected, "mode": "verified", "judge_model": result.judge_model, "labels": str(out)}, indent=1))
        result_out = write_report(load_scan(root), root / "report.html")
        console.print(f"\n[bold]{n} answers verified, {corrected} corrected: {agree}/{n} = {agree / n:.0%} agreement (verified, judge shown).[/bold] Saved to {out} and {root / 'judge_check.json'}; report re-rendered: {result_out}")
        console.print(f'Deck wording: "{n} answers verified by hand, {corrected} corrected".')
    else:
        console.print("Nothing saved.")


def _summary(result) -> None:
    c = result.brief.client.name
    m = result.overall.brands[c]
    console.print(
        f"\n[bold]{c}[/bold] · presence {_p(m.presence_rate)} · recommended {_p(m.recommendation_rate)} · "
        f"share of voice {_p(result.overall.share_of_voice)} · own-site citations "
        f"{_p(result.overall.citation_share_by_type.get('brand_owned', 0))} · {len(result.findings)} findings"
    )
    for f in result.findings:
        console.print(f"  [{f.severity}] {f.title}")


def _p(x) -> str:
    return "n/a" if x is None else f"{round(x * 100)}%"


if __name__ == "__main__":
    app()
