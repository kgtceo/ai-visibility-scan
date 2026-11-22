"""Render a scan as one self-contained HTML page a brand team can open and share.

Order follows how a commercial reader thinks: the headline numbers, the engine
comparison, who wins the shortlist, what is said about the deciding attributes,
which sources shape the answers, the fix list, then every answer verbatim as
evidence, then method and limits.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from jinja2 import BaseLoader, Environment

from .metrics import pct
from .models import ScanResult

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Visibility Scan · {{ r.brief.client.name }}</title>
<style>
:root{--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#fff;--soft:#f6f7f9;--brand:#0f3d5e;--hi:#b91c1c;--mid:#b45309;--lo:#1d4ed8;--ok:#047857}
*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,Helvetica,sans-serif;color:var(--ink);background:var(--bg);line-height:1.45;font-size:15px}
.wrap{max-width:1100px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;color:var(--brand);border-bottom:1px solid var(--line);padding-bottom:6px}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--muted);margin:0 0 14px}
.banner{background:#fff7ed;border:1px solid #fdba74;color:#7c2d12;padding:10px 12px;border-radius:8px;margin:14px 0;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:12px}
.tile .n{font-size:28px;font-weight:700;line-height:1.1}.tile .l{font-size:12.5px;color:var(--muted);margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:var(--soft);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--line);min-width:160px}
.bar span{display:block;height:100%}.pos{background:#059669}.neu{background:#9ca3af}.neg{background:#dc2626}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--line)}
.b-recommended{background:#ecfdf5;color:var(--ok);border-color:#a7f3d0}.b-mentioned{background:#fffbeb;color:var(--mid);border-color:#fde68a}.b-absent{background:#f3f4f6;color:var(--muted)}
.find{border-left:4px solid var(--line);padding:8px 12px;margin:8px 0;background:var(--soft);border-radius:0 8px 8px 0}
.find.high{border-color:var(--hi)}.find.medium{border-color:var(--mid)}.find.low{border-color:var(--lo)}
.find b{display:block}.find small{color:var(--muted)}.find .act{margin-top:4px;font-size:13.5px}.find .act::first-letter{font-weight:700}
details{border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:0 12px;background:#fff}summary{cursor:pointer;padding:10px 0;font-weight:600}
.ans{border-top:1px dashed var(--line);padding:10px 0}.ans .meta{font-size:12.5px;color:var(--muted);margin-bottom:4px}
.ans .text{white-space:pre-wrap;font-size:13.5px;background:var(--soft);padding:10px;border-radius:8px}
.ans mark{background:#fef08a;padding:0 2px}.ans mark.client{background:#bfdbfe}
.cites{font-size:12.5px;margin:6px 0 0;padding-left:18px}.cites .t{color:var(--muted)}
.claims li{font-size:13px}.v-supported{color:var(--ok)}.v-contradicted{color:var(--hi);font-weight:600}.v-unverifiable{color:var(--muted)}
.small{font-size:13px;color:var(--muted)}
.acc{display:inline-block;padding:0 7px;border-radius:999px;font-size:11.5px;font-weight:600;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;white-space:nowrap;vertical-align:middle}
.acc.api{background:#f0fdf4;color:#166534;border-color:#bbf7d0}.acc.signed{background:#fff7ed;color:#9a3412;border-color:#fed7aa}.acc.fixture{background:#f3f4f6;color:#4b5563}
.ind{font-size:11px;color:var(--mid);font-weight:600;margin-left:3px;white-space:nowrap}
@media print{details{break-inside:avoid}summary{pointer-events:none}details[open]{display:block}}
</style></head><body><div class="wrap">
<h1>AI Visibility Scan · {{ r.brief.client.name }}</h1>
<p class="sub">{{ r.brief.category }} · market {{ r.brief.market }} · {{ r.brief.questions|length }} shopper questions × {{ r.runs }} run(s) · judge: {{ r.judge_model }} · {{ r.created_at[:16].replace('T',' ') }} UTC · scan {{ r.scan_id }}</p>
<p class="sub">Engines: {% for e in r.engines %}{{ e }} {{ acc(r.per_engine[e]) }}{% if not loop.last %} · {% endif %}{% endfor %}</p>
{% if judge_check %}<p class="sub">Judge check: {{ judge_check.rows }} answers {{ 'verified by hand' if judge_check.mode == 'verified' else 'labelled blind' }}, {{ judge_check.corrected }} corrected ({{ (100 * judge_check.agree / judge_check.rows)|round|int }}% agreement{% if judge_check.mode == 'verified' %}, judge's call shown to the checker{% endif %}).</p>{% else %}<p class="sub">Judge check: not yet run for this scan (<code>aivs verify runs/{{ r.scan_id }}</code>).</p>{% endif %}
{% if r.synthetic %}<div class="banner"><b>Synthetic demo data.</b> These answers are hand-written fixtures so the pipeline can be shown without API keys. Numbers here show the method, not {{ r.brief.client.name }}'s real visibility. Run a live scan for real figures.</div>{% endif %}
{% if r.judge_model == 'heuristic' %}<div class="banner"><b>Heuristic judge.</b> Stances and sentiment were set by keyword rules, not the model judge. Presence and sources are exact; recommended/mentioned and sentiment are approximate.</div>{% endif %}

<h2>1. The headline numbers for {{ c }}</h2>
{% set cm = r.overall.brands[c] %}
<div class="tiles">
<div class="tile"><div class="n">{{ rate(cm.presence_rate, cm.unbranded_answers) }}</div><div class="l">Presence · appears in unbranded shopper questions ({{ cm.unbranded_answers }} answers)</div></div>
<div class="tile"><div class="n">{{ rate(cm.recommendation_rate, cm.unbranded_answers) }}</div><div class="l">Recommended · put forward as the choice</div></div>
<div class="tile"><div class="n">{{ rate(r.overall.share_of_voice, r.overall.recommended_total) }}</div><div class="l">Share of voice · of {{ r.overall.recommended_total }} recommendations across tracked brands</div></div>
<div class="tile"><div class="n">{{ r.overall.claims_supported }} / {{ r.overall.claims_checked }}</div><div class="l">Accuracy · checkable claims about {{ c }} that match its facts ({{ rate(r.overall.accuracy_rate, r.overall.claims_checked) }}); {{ r.overall.claims_contradicted }} contradict, {{ r.overall.claims_unverifiable }} not checkable against the facts given</div></div>
<div class="tile"><div class="n">{{ rate(r.overall.citation_share_by_type.get('brand_owned', 0), r.overall.citations_total) }}</div><div class="l">Own pages · share of {{ r.overall.citations_total }} citations that are {{ c }}'s own site</div></div>
<div class="tile"><div class="n">{{ rate(r.overall.stance_consistency, r.overall.repeated_questions) }}</div><div class="l">Consistency · same stance on repeat runs{% set rep = r.per_engine.values()|selectattr('repeated_questions')|map(attribute='engine')|list %}{% if rep %} ({{ rep|length }} of {{ r.engines|length }} engines ran twice){% else %} (needs ≥2 runs){% endif %}</div></div>
</div>
<p class="small">{{ r.overall.answered }} of {{ r.overall.answers }} questions answered across engines{% if r.overall.errors %} ({{ r.overall.errors }} failed, listed in section 8){% endif %}. <span class="ind">n=…</span> marks a rate on fewer than ten answers: indicative only.</p>

<h2>2. By engine</h2>
<table><tr><th>Engine</th><th>Access</th><th class="num">Answered</th><th class="num">Presence</th><th class="num">Recommended</th><th class="num">Avg position</th><th class="num">Share of voice</th><th class="num">Accuracy</th><th class="num">Own-site citations</th><th class="num">Consistency</th></tr>
{% for e, m in r.per_engine.items() %}{% set b = m.brands.get(c) %}
<tr><td>{{ e }}</td><td>{{ acc(m) }}</td><td class="num">{% if m.errors %}{{ m.answered }} / {{ m.answers }} <span class="small">({{ m.errors }} failed)</span>{% else %}{{ m.answers }}{% endif %}</td><td class="num">{{ rate(b.presence_rate, b.unbranded_answers) if b else 'n/a' }}</td><td class="num">{{ rate(b.recommendation_rate, b.unbranded_answers) if b else 'n/a' }}</td><td class="num">{{ b.avg_position if b and b.avg_position else '–' }}</td><td class="num">{{ rate(m.share_of_voice, m.recommended_total) }}</td><td class="num">{{ rate(m.accuracy_rate, m.claims_checked) }}</td><td class="num">{{ rate(m.citation_share_by_type.get('brand_owned', 0), m.citations_total) }}</td><td class="num">{{ rate(m.stance_consistency, m.repeated_questions) }}</td></tr>
{% endfor %}</table>
<p class="small">Access: <span class="acc">web · signed out</span> the consumer surface read through Chrome with no account, what a shopper sees. <span class="acc signed">web · signed in</span> the same surface on the scan owner's own account (memory and personalisation can shape the answer). <span class="acc api">API</span> the provider's developer endpoint with web search on, not the app.</p>

<h2>3. By question</h2>
<p class="small">The unbranded questions, one row each: how many engines named {{ c }} and how many put it forward, out of the answers judged. The headline numbers rest on these questions. Branded questions name a tracked brand, so presence there is a given; their stage mix is shown underneath.</p>
<table><tr><th>Question</th><th>Stage</th><th class="num">Answers</th><th class="num">{{ c }} named</th><th class="num">{{ c }} put forward</th>{% for e in r.engines %}<th class="num">{{ e }}</th>{% endfor %}</tr>
{% for q, row in by_question %}<tr><td>{{ q.id }} · {{ q.text }}</td><td class="small">{{ q.stage }}</td><td class="num">{{ row.n }}</td><td class="num">{{ row.present }}</td><td class="num">{{ row.rec }}</td>{% for e in r.engines %}{% set cell = row.engines.get(e) %}<td class="num small">{{ cell if cell else '–' }}</td>{% endfor %}</tr>
{% endfor %}</table>
<p class="small">Per engine: named / put forward / answers. Branded questions: {% for st, sm in r.overall.by_stage.items() if sm.branded_questions %}{{ st }} {{ sm.branded_questions }}{% if not loop.last %}, {% endif %}{% endfor %}; {{ c }} is put forward in {{ pct(branded_rec_rate) }} of their {{ branded_answers }} answers, which mostly echoes the question and is not a visibility number.</p>

<h2>4. Who wins the shortlist (unbranded questions)</h2>
<table><tr><th>Brand</th><th class="num">Presence</th><th class="num">Recommended</th><th class="num">Avg position</th>{% for e in r.engines %}<th class="num">{{ e }} rec.</th>{% endfor %}</tr>
{% for name, b in brands_sorted %}
<tr><td>{% if name == c %}<b>{{ name }}</b>{% else %}{{ name }}{% endif %}</td><td class="num">{{ rate(b.presence_rate, b.unbranded_answers) }}</td><td class="num">{{ rate(b.recommendation_rate, b.unbranded_answers) }}</td><td class="num">{{ b.avg_position if b.avg_position else '–' }}</td>{% for e in r.engines %}{% set eb = r.per_engine[e].brands.get(name) %}<td class="num">{{ rate(eb.recommendation_rate, eb.unbranded_answers) if eb else 'n/a' }}</td>{% endfor %}</tr>
{% endfor %}</table>

{% if r.overall.watch %}<h3>Also named, not tracked</h3>
<p class="small">Brands on the brief's watchlist, counted by name in unbranded answers (rules only, no judge). The shortlist above ignores them; a brand that keeps appearing belongs in the competitor set next run.</p>
<table><tr><th>Brand</th><th class="num">Named in unbranded answers</th>{% for e in r.engines %}<th class="num">{{ e }}</th>{% endfor %}</tr>
{% for name, w in r.overall.watch.items()|sort(attribute='1.present_in_unbranded', reverse=True) %}<tr><td>{{ name }}</td><td class="num">{{ w.present_in_unbranded }} / {{ w.unbranded_answers }} ({{ pct(w.presence_rate) }})</td>{% for e in r.engines %}{% set ew = r.per_engine[e].watch.get(name) %}<td class="num">{{ ew.present_in_unbranded if ew else '–' }}</td>{% endfor %}</tr>
{% endfor %}</table>{% endif %}

<h2>5. What is said about {{ c }} on the attributes in the brief</h2>
<table><tr><th>Attribute</th><th>Positive / neutral / negative</th><th class="num">Discussed</th><th class="num">Negative share</th></tr>
{% for attr, s in r.overall.client_sentiment.items() %}
<tr><td>{{ attr.replace('_',' ') }}</td><td>{% if s.discussed %}<div class="bar"><span class="pos" style="width:{{ 100*s.positive/s.discussed }}%"></span><span class="neu" style="width:{{ 100*s.neutral/s.discussed }}%"></span><span class="neg" style="width:{{ 100*s.negative/s.discussed }}%"></span></div>{% else %}<span class="small">not discussed</span>{% endif %}</td><td class="num">{{ s.discussed }}</td><td class="num">{{ rate(s.negative/s.discussed, s.discussed) if s.discussed else '–' }}</td></tr>
{% endfor %}</table>

<h2>6. Where the answers come from</h2>
<table><tr><th>Source type</th>{% for e in r.engines %}<th class="num">{{ e }}</th>{% endfor %}<th class="num">All</th></tr>
<tr><td class="small">citations counted</td>{% for e in r.engines %}<td class="num small">{{ r.per_engine[e].citations_total }}</td>{% endfor %}<td class="num small">{{ r.overall.citations_total }}</td></tr>
<tr><td class="small">answers with no sources at all</td>{% for e in r.engines %}<td class="num small">{{ r.per_engine[e].answers_without_sources }} / {{ r.per_engine[e].answered }}</td>{% endfor %}<td class="num small">{{ r.overall.answers_without_sources }} / {{ r.overall.answered }}</td></tr>
{% for t in source_types %}<tr><td>{{ t.replace('_',' ') }}</td>{% for e in r.engines %}<td class="num">{{ pct(r.per_engine[e].citation_share_by_type.get(t, 0)) if r.per_engine[e].citations_total else '–' }}</td>{% endfor %}<td class="num">{{ pct(r.overall.citation_share_by_type.get(t, 0)) if r.overall.citations_total else '–' }}</td></tr>{% endfor %}
</table>
<p class="small">The "All" column is weighted by volume: an engine that cites many pages per answer dominates it. Read the per-engine columns for the fix list.</p>
<h3>Domains cited most</h3>
<table><tr><th>Domain</th><th class="num">Citations</th></tr>{% for d, n in r.overall.top_domains %}<tr><td>{{ d }}</td><td class="num">{{ n }}</td></tr>{% endfor %}</table>

<h2>7. Findings and the fix list</h2>
{% if not r.findings %}<p class="small">No rule fired. Either the brand is in good shape on this question set, or the sample is too small (see method).</p>{% endif %}
{% for f in r.findings %}<div class="find {{ f.severity }}"><b>{{ f.title }}</b>{{ f.detail }}{% if f.engines %} <small>({{ f.engines|join(', ') }})</small>{% endif %}{% if f.action %}<div class="act">Do: {{ f.action }}</div>{% endif %}</div>{% endfor %}
{% if r.overall.contradictions %}<h3>Contradicted claims, by the fact they contradict</h3>
<table><tr><th>Verified fact</th><th class="num">Claims</th><th>Engines · pages cited in those answers</th><th>Examples (the claim as the judge recorded it)</th></tr>
{% for g in r.overall.contradictions.values() %}<tr><td><span class="small">{{ clip(g.statement, 120) }}</span></td><td class="num">{{ g.count }}<div class="small">{{ g.answers_with_sources }} of {{ g.answers }} answers sourced</div></td><td class="small">{% for e, n in g.engines.items() %}{{ e }} {{ n }}{% if not loop.last %} · {% endif %}{% endfor %}{% if g.top_domains %}<br>cited: {% for d, n in g.top_domains %}{{ d }} {{ n }}{% if not loop.last %} · {% endif %}{% endfor %}{% endif %}</td><td class="small">{% for x in g.examples %}“{{ x }}”{% if not loop.last %}<br>{% endif %}{% endfor %}</td></tr>
{% endfor %}</table>{% endif %}

<h2>8. Every answer, verbatim</h2>
<p class="small">Client highlighted in blue, competitors in yellow. Each answer shows the stance the judge gave and the sources the engine cited.</p>
{% for q in r.brief.questions %}
<details><summary>{{ q.id }} · {{ q.text }} <span class="small">· {{ q.stage }}{% if q.branded %} · branded{% endif %}</span></summary>
{% for a in answers_for(q.id) %}{% set s = score_for(a) %}
<div class="ans"><div class="meta"><b>{{ a.engine }}</b> {{ acc_answer(a) }} ({{ a.model }}) · run {{ a.run }}{% if a.latency_s %} · {{ a.latency_s }}s{% endif %}
{% if s and s.judgement %}{% for bj in s.judgement.brands if bj.stance != 'absent' %} <span class="badge b-{{ bj.stance }}">{{ bj.brand }}: {{ bj.stance }}{% if bj.position %} #{{ bj.position }}{% endif %}</span>{% endfor %}{% endif %}
{% if s and s.judge_error %} <span class="badge">judge failed: {{ s.judge_error }}</span>{% endif %}
{% if a.search_queries %}<div>searched: {{ a.search_queries|join(' · ') }}</div>{% endif %}{% if a.note %}<div class="small">{{ a.note }}</div>{% endif %}</div>
{% if a.error %}<div class="text">Engine error: {{ a.error }}</div>{% else %}<div class="text">{{ highlight(a.text) }}</div>{% endif %}
{% if s and s.judgement and s.judgement.client_claims %}<ul class="claims">{% for cl in s.judgement.client_claims %}<li><span class="v-{{ cl.verdict }}">{{ cl.verdict }}</span>: “{{ cl.text }}”{% if cl.fact_id %} <span class="small">(fact {{ cl.fact_id }}{% if cl.note %}: {{ cl.note }}{% endif %})</span>{% endif %}</li>{% endfor %}</ul>{% endif %}
{% if s and s.sources %}<ol class="cites">{% for src in s.sources %}<li><a href="{{ src.url }}">{{ src.domain }}</a> <span class="t">· {{ src.source_type.replace('_',' ') }}{% if src.title %} · {{ src.title[:80] }}{% endif %}</span></li>{% endfor %}</ol>{% elif not a.error and not a.note %}<div class="small">No citations returned.</div>{% endif %}
</div>
{% endfor %}
</details>
{% endfor %}

<h2>9. Method and limits</h2>
<ul class="small">
<li><b>Questions.</b> {{ r.brief.questions|length }} questions a UK shopper would type, across discover, compare, choose, trust and switch. Unbranded questions name no tracked brand and measure whether the brand gets onto the shortlist; branded ones name one (the client or a competitor) and measure what is said once a shopper asks. Each is asked as typed, with only the market added. Brands on the watchlist are counted by name only.</li>
<li><b>Engines.</b> Two kinds. <i>Web</i> engines (names ending -web) are the consumer surfaces themselves, asked signed out through a real Chrome in {{ r.brief.city }}, {{ r.brief.market }}, one question at a time at a human pace; no API key. What they show is what a shopper sees, including the sources panel. <i>API</i> engines are the providers' API products with web search switched on, located to the same city: the closest stand-in where a surface cannot be read in a browser (ChatGPT signed out sits behind a bot challenge). Google AI Overviews / AI Mode show a CAPTCHA on the first automated request and have no API; out of scope. Where a surface hides a source URL (Gemini chips, redirects) we classify on the source name. Every failed question is listed, never dropped.{% set nosrc = r.per_engine.values()|selectattr('answers_without_sources')|list %}{% if nosrc %} Some answers came with no source displayed ({% for m in nosrc %}{{ m.engine }} {{ m.answers_without_sources }} of {{ m.answered }}{% if not loop.last %}, {% endif %}{% endfor %}); they are scored like any other answer but nothing in them can be traced to a page.{% endif %}</li>
<li><b>Scoring.</b> Presence and sources are rules (alias match, domain lists), never a model's opinion. Recommended vs mentioned, sentiment per attribute and claim checks come from a model judge that only sees the answer text, the tracked brands, the attributes and {{ c }}'s verified facts. The recommended/mentioned call is checked against a hand-labelled sample each scan{% if judge_check %}: this run, {{ judge_check.rows }} unbranded answers {{ 'verified by hand with the judge\'s call shown' if judge_check.mode == 'verified' else 'labelled blind' }}, {{ judge_check.corrected }} corrected{% else %} (not yet done for this run){% endif %}. The judge's fact checks over-flag, so they are spot-read.</li>
<li><b>Sample size.</b> One run is a snapshot. Engines vary run to run, so trends need several runs and a repeat on a fixed cadence. Rates on fewer than ten answers are marked <span class="ind">n=…</span> and are indicative only.</li>
<li><b>Several sittings.</b> Answers were collected in more than one sitting (browser engines and API engines on different days); the report covers all of them. Failed questions count as attempted, never as answered.</li>
<li><b>Facts.</b> Only facts marked verified are used for accuracy; {{ r.brief.facts|selectattr('status','equalto','verified')|list|length }} of {{ r.brief.facts|length }} in this brief.</li>
</ul>
</div></body></html>
"""


def render(result: ScanResult, judge_check: dict | None = None) -> str:
    env = Environment(loader=BaseLoader(), autoescape=False)
    tmpl = env.from_string(_TEMPLATE)
    c = result.brief.client.name
    brands_sorted = sorted(
        result.overall.brands.items(), key=lambda kv: (kv[1].recommendation_rate or 0, kv[1].presence_rate or 0), reverse=True
    )
    source_types = ["brand_owned", "competitor_owned", "review_comparison", "forum_social", "news", "regulator_official", "wikipedia", "other"]
    score_index = {(s.engine, s.question_id, s.run): s for s in result.scores}
    patterns = [(b.name, b.name == c, _pattern(b.all_names())) for b in result.brief.tracked()]

    from collections import defaultdict

    class _Row:
        def __init__(self):
            self.n = 0
            self.present = 0
            self.rec = 0
            self._e = defaultdict(lambda: [0, 0, 0])

        @property
        def engines(self):
            return {e: f"{v[0]}/{v[1]}/{v[2]}" for e, v in self._e.items()}

    rows: dict[str, _Row] = defaultdict(_Row)
    branded_answers = branded_rec = 0
    qbr = {q.id: q for q in result.brief.questions}
    for sc in result.scores:
        if not sc.judgement:
            continue
        bj = next((b for b in sc.judgement.brands if b.brand == c), None)
        if bj is None:
            continue
        q = qbr.get(sc.question_id)
        if q is None:
            continue
        if q.branded:
            branded_answers += 1
            branded_rec += bj.stance == "recommended"
            continue
        row = rows[q.id]
        row.n += 1
        row.present += bj.stance != "absent"
        row.rec += bj.stance == "recommended"
        cell = row._e[sc.engine]
        cell[0] += bj.stance != "absent"
        cell[1] += bj.stance == "recommended"
        cell[2] += 1
    by_question = [(q, rows[q.id]) for q in result.brief.questions if not q.branded]
    branded_rec_rate = (branded_rec / branded_answers) if branded_answers else None

    def answers_for(qid: str):
        return sorted((a for a in result.answers if a.question_id == qid), key=lambda a: (a.engine, a.run))

    def score_for(a):
        return score_index.get((a.engine, a.question_id, a.run))

    def clip(text: str, n: int = 90) -> str:
        if len(text) <= n:
            return text
        return text[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"

    def rate(x, den: int) -> str:
        """Percentage, with the denominator shown when it is under ten (indicative)."""
        if x is None:
            return "n/a"
        return pct(x) + (f'<span class="ind">n={den}</span>' if den < 10 else "")

    _ACC_CLASS = {"API": "api", "web · signed in": "signed", "fixture": "fixture"}

    def acc(m) -> str:
        label = m.access_label or "unknown"
        return f'<span class="acc {_ACC_CLASS.get(label, "")}">{html.escape(label)}</span>'

    def acc_answer(a) -> str:
        if a.error:  # a failed row carries its engine's access, not the default of an empty answer
            return acc(result.per_engine[a.engine]) if a.engine in result.per_engine else ""
        if a.access == "web":
            label = "web · signed in" if a.signed_in else "web · signed out"
        elif a.access == "fixture":
            label = "fixture"
        else:
            label = "API"
        return f'<span class="acc {_ACC_CLASS.get(label, "")}">{label}</span>'

    def highlight(text: str) -> str:
        out = html.escape(text)
        for _name, is_client, pat in patterns:
            cls = ' class="client"' if is_client else ""
            out = pat.sub(lambda m, cls=cls: f"<mark{cls}>{m.group(0)}</mark>", out)
        return out

    return tmpl.render(
        r=result, c=c, pct=pct, brands_sorted=brands_sorted, source_types=source_types,
        answers_for=answers_for, score_for=score_for, highlight=highlight, rate=rate, acc=acc, acc_answer=acc_answer,
        by_question=by_question, branded_answers=branded_answers, branded_rec_rate=branded_rec_rate, clip=clip,
        judge_check=judge_check,
    )


def _pattern(names: list[str]) -> re.Pattern:
    names = sorted({n for n in names if n}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])", re.I)


def write_report(result: ScanResult, path: Path) -> Path:
    """Render next to scan.json; picks up judge_check.json (from `aivs verify` or
    evals/run_evals.py) when it is there."""
    import json

    check_path = path.parent / "judge_check.json"
    check = json.loads(check_path.read_text()) if check_path.exists() else None
    path.write_text(render(result, check), encoding="utf-8")
    return path
