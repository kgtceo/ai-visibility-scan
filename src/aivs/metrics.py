"""Aggregate scores into the numbers a brand team can act on, per engine and overall.

The seven headline numbers for the client:
1. Presence rate: share of unbranded shopper questions where the brand appears at all.
2. Recommendation rate: share of unbranded questions where it is put forward as a choice.
   (Visible is not the same as recommended.)
3. Share of voice: the client's share of all "recommended" verdicts across tracked brands.
4. Sentiment per deciding attribute (fees, service, safety...), from the branded and unbranded answers.
5. Accuracy: claims about the client that match its facts vs claims that contradict them.
6. Sources: what share of citations are the brand's own pages vs reviews, forums, news;
   which domains shape the answers.
7. Consistency: does the engine give the same stance on repeat runs of the same question?
Every rate carries its denominator so the report can mark small samples as indicative.
A per-journey-stage rollup (discover, compare, choose, trust, switch) shows where in the
funnel the client drops out. Findings are derived from these numbers by fixed rules, so
the fix list is explainable.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .models import (
    AnswerScore,
    AttributeSentimentCounts,
    BrandMetrics,
    Brief,
    ContradictionGroup,
    EngineAnswer,
    EngineMetrics,
    Finding,
    QuestionMetrics,
    StageMetrics,
    TrackedBrand,
)


def aggregate(
    brief: Brief, answers: list[EngineAnswer], scores: list[AnswerScore], engine: str | None
) -> EngineMetrics:
    qmap = {q.id: q for q in brief.questions}
    client = brief.client.name
    sel_answers = [a for a in answers if engine is None or a.engine == engine]
    sel_scores = [s for s in scores if engine is None or s.engine == engine]
    m = EngineMetrics(engine=engine or "all", access_label=access_label(sel_answers))
    m.answers = len(sel_answers)
    m.errors = sum(1 for a in sel_answers if a.error)
    m.answers_without_sources = sum(1 for a in sel_answers if not a.error and not a.citations)
    for b in brief.tracked():
        m.brands[b.name] = BrandMetrics(brand=b.name)
    m.client_sentiment = {a: AttributeSentimentCounts() for a in brief.attributes}
    m.by_stage = {st: StageMetrics(stage=st) for st in dict.fromkeys(q.stage for q in brief.questions)}
    for q in brief.questions:
        m.by_stage[q.stage].questions += 1
        if q.branded:
            m.by_stage[q.stage].branded_questions += 1
    # watchlist: brands the brief does not track, counted by alias rule in unbranded answers
    for w in brief.watchlist:
        wm = m.watch[w.name] = BrandMetrics(brand=w.name)
        pat = _alias_regex(w)
        for a in sel_answers:
            q = qmap.get(a.question_id)
            if a.error or q is None or q.branded:
                continue
            wm.unbranded_answers += 1
            if pat.search(a.text):
                wm.present_in_unbranded += 1
    facts = {f.id: f.statement for f in brief.facts}

    recommended_total = 0
    recommended_client = 0
    group_domains: dict[str, Counter] = {}
    neg_domains: dict[str, Counter] = {}
    m.by_question = {
        q.id: QuestionMetrics(question_id=q.id, text=q.text, stage=q.stage) for q in brief.questions if not q.branded
    }
    type_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    # consistency compares repeat runs of the same question on the SAME engine
    stances_by_q: dict[tuple[str, str], list[str]] = defaultdict(list)

    for s in sel_scores:
        q = qmap.get(s.question_id)
        if q is None:
            continue
        # sources (deterministic, available even when the judge failed)
        for src in s.sources:
            type_counter[src.source_type] += 1
            domain_counter[src.domain] += 1
        m.citations_total += len(s.sources)
        if any(src.source_type == "brand_owned" for src in s.sources):
            m.answers_citing_client_domain += 1

        if s.judgement is None:
            continue
        qm = m.by_question.get(s.question_id) if not q.branded else None
        if qm is not None:
            qm.answers += 1
        for bj in s.judgement.brands:
            bm = m.brands.setdefault(bj.brand, BrandMetrics(brand=bj.brand))
            if not q.branded:
                bm.unbranded_answers += 1
                if bj.stance != "absent":
                    bm.present_in_unbranded += 1
                    if bj.position:
                        bm.positions.append(bj.position)
                if bj.stance == "recommended":
                    bm.recommended_in_unbranded += 1
                    recommended_total += 1
                    for attr, sent in bj.attribute_sentiment.items():
                        if sent == "positive":
                            bm.positive_attributes[attr] = bm.positive_attributes.get(attr, 0) + 1
                    if bj.evidence and len(bm.evidence) < 3 and bj.evidence not in bm.evidence:
                        bm.evidence.append(bj.evidence[:120])
                    if bj.brand == client:
                        recommended_client += 1
                if qm is not None:
                    if bj.brand == client:
                        qm.present += bj.stance != "absent"
                        qm.recommended += bj.stance == "recommended"
                    elif bj.stance == "recommended":
                        qm.others_recommended[bj.brand] = qm.others_recommended.get(bj.brand, 0) + 1
            if bj.brand == client:
                stances_by_q[(s.engine, s.question_id)].append(bj.stance)
                st = m.by_stage.setdefault(q.stage, StageMetrics(stage=q.stage))
                if q.branded:
                    st.branded_answers += 1
                    if bj.stance == "recommended":
                        st.branded_recommended += 1
                else:
                    st.answers += 1
                    if bj.stance != "absent":
                        st.present += 1
                    if bj.stance == "recommended":
                        st.recommended += 1
                for attr, sent in bj.attribute_sentiment.items():
                    counts = m.client_sentiment.setdefault(attr, AttributeSentimentCounts())
                    setattr(counts, sent, getattr(counts, sent) + 1)
                    if sent == "negative":
                        nd = neg_domains.setdefault(attr, Counter())
                        for src in s.sources:
                            nd[src.domain] += 1
        seen_groups: set[str] = set()
        for c in s.judgement.client_claims:
            if c.verdict == "supported":
                m.claims_supported += 1
            elif c.verdict == "contradicted":
                m.claims_contradicted += 1
                fid = c.fact_id or "?"
                g = m.contradictions.setdefault(fid, ContradictionGroup(fact_id=fid, statement=facts.get(fid, "")))
                g.count += 1
                g.engines[s.engine] = g.engines.get(s.engine, 0) + 1
                if len(g.examples) < 3 and c.text not in g.examples:
                    g.examples.append(c.text[:140])
                if fid not in seen_groups:
                    seen_groups.add(fid)
                    g.answers += 1
                    if s.sources:
                        g.answers_with_sources += 1
                        dc = group_domains.setdefault(fid, Counter())
                        for src in s.sources:
                            dc[src.domain] += 1
            else:
                m.claims_unverifiable += 1

    m.recommended_total = recommended_total
    m.share_of_voice = round(recommended_client / recommended_total, 3) if recommended_total else None
    if m.citations_total:
        m.citation_share_by_type = {
            k: round(v / m.citations_total, 3) for k, v in sorted(type_counter.items(), key=lambda kv: -kv[1])
        }
    m.top_domains = domain_counter.most_common(12)
    repeated = [v for v in stances_by_q.values() if len(v) >= 2]
    m.repeated_questions = len(repeated)
    if repeated:
        m.stance_consistency = round(sum(1 for v in repeated if len(set(v)) == 1) / len(repeated), 3)
    for fid, g in m.contradictions.items():
        g.top_domains = group_domains.get(fid, Counter()).most_common(4)
    m.contradictions = dict(sorted(m.contradictions.items(), key=lambda kv: -kv[1].count))
    m.negative_sources = {a: c.most_common(5) for a, c in neg_domains.items()}
    for bm in m.brands.values():
        bm.positive_attributes = dict(sorted(bm.positive_attributes.items(), key=lambda kv: -kv[1]))
    for qm in m.by_question.values():
        qm.others_recommended = dict(sorted(qm.others_recommended.items(), key=lambda kv: -kv[1]))
    return m


def _sources_action(overall: EngineMetrics, attr: str, client: str) -> str:
    srcs = overall.negative_sources.get(attr) or []
    if not srcs:
        return "The answers that say so showed no sources; nothing to fix on a page, track it next run."
    listed = ", ".join(f"{d} ({n})" for d, n in srcs[:4])
    own = any(d == od or d.endswith("." + od) for d, _ in srcs for od in overall_client_domains(overall, client))
    tail = f" {client}'s own pages are among the pages those answers cite, so read them alongside the complaints." if own else ""
    return f"The answers that say so cite {listed} most: those threads and pages are the ones to answer or fix.{tail}"


def overall_client_domains(overall: EngineMetrics, client: str) -> list[str]:
    # the brief is not in EngineMetrics; the client's own domains are the ones classified brand_owned,
    # which the aggregate already applied, so match on the client's name in the domain as a fallback
    return [client.lower() + ".com", "community." + client.lower() + ".com"]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _attr(a: str) -> str:
    return a.replace("_", " ")


def _clip(text: str, n: int = 90) -> str:
    """Cut at a word boundary, never mid-word."""
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def _alias_regex(brand: TrackedBrand) -> re.Pattern:
    names = sorted({n for n in brand.all_names() if n}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])", re.I)


def access_label(answers: list[EngineAnswer]) -> str:
    """How the answers were obtained, for the report badge: 'web · signed out',
    'web · signed in', 'API', 'fixture', or 'mixed' across engines."""
    # A failed question says less about access than an answered one (and answers cached
    # before the access field existed default to API), so answered rows decide when there are any.
    answered = [a for a in answers if not a.error]
    labels: set[str] = set()
    for a in answered or answers:
        if a.access == "web":
            labels.add("web · signed in" if a.signed_in else "web · signed out")
        elif a.access == "fixture":
            labels.add("fixture")
        else:
            labels.add("API")
    if not labels:
        return ""
    return labels.pop() if len(labels) == 1 else "mixed"


def findings(brief: Brief, per_engine: dict[str, EngineMetrics], overall: EngineMetrics) -> list[Finding]:
    """Fixed rules over the metrics. Each finding says what the brand team does about
    it and names the engines it comes from, so the fix list is explainable."""
    out: list[Finding] = []
    client = brief.client.name
    cm = overall.brands.get(client)

    # 1. Visible but not recommended / missing
    if cm and cm.presence_rate is not None and cm.recommendation_rate is not None and cm.unbranded_answers >= 4:
        if cm.presence_rate >= 0.5 and cm.recommendation_rate < cm.presence_rate * 0.6:
            out.append(
                Finding(
                    severity="high",
                    title=f"{client} is visible but not recommended",
                    detail=(
                        f"Named in {pct(cm.presence_rate)} of unbranded shopper questions, but put forward as the "
                        f"choice in only {pct(cm.recommendation_rate)}."
                    ),
                    action="Read the answers where it is named but not picked: the reason given is the attribute to work on.",
                )
            )
        if cm.presence_rate < 0.35:
            out.append(
                Finding(
                    severity="high",
                    title=f"{client} is missing from most unbranded questions",
                    detail=f"Appears in {pct(cm.presence_rate)} of unbranded questions. Competitors are filling the shortlist.",
                    action="Check the sources the engines cite for those questions; the brand needs to be on the pages they read.",
                )
            )

    # 2. Who wins the shortlist
    leaders = sorted(
        ((b.recommendation_rate or 0, b.brand) for b in overall.brands.values() if b.unbranded_answers),
        reverse=True,
    )
    if leaders and leaders[0][1] != client and leaders[0][0] > 0:
        lead_rate, lead = leaders[0]
        mine = cm.recommendation_rate if cm else None
        lb = overall.brands[lead]
        top3 = list(lb.positive_attributes.items())[:3]
        reasons = ", ".join(f"{_attr(a)} ({n})" for a, n in top3)
        names = " and ".join(", ".join(_attr(a) for a, _ in top3).rsplit(", ", 1)) if top3 else "those attributes"
        quote = f' Typical wording: "{lb.evidence[0]}".' if lb.evidence else ""
        out.append(
            Finding(
                severity="high" if mine is not None and lead_rate >= 1.5 * mine else "medium",
                title=f"{lead} is recommended most often: {pct(lead_rate)} of unbranded questions, {client} {pct(mine)}",
                detail=", ".join(f"{name} {pct(rate)}" for rate, name in leaders[:5])
                + f" (recommendation rate on unbranded questions). In the answers that pick {lead}, the attributes rated positive for it are {reasons}: those are the reasons given.{quote}",
                action=f"Answer {lead} on {names} on {client}'s own pages and on the comparison sites the engines cite (section 6), in the words the questions use.",
            )
        )

    # 3. Sentiment: the usual threshold, then the single worst attribute if nothing fired
    fired_attr = False
    for attr, counts in overall.client_sentiment.items():
        if counts.discussed >= 3 and counts.negative / counts.discussed >= 0.34:
            fired_attr = True
            out.append(
                Finding(
                    severity="high" if counts.negative / counts.discussed >= 0.5 else "medium",
                    title=f"Negative on {_attr(attr)}",
                    detail=f"{counts.negative} of {counts.discussed} answers that discuss {client}'s {_attr(attr)} are negative.",
                    action=_sources_action(overall, attr, client),
                )
            )
    if not fired_attr:
        shares = sorted(
            ((c.negative / c.discussed, a, c) for a, c in overall.client_sentiment.items() if c.discussed >= 10), reverse=True
        )
        if shares and shares[0][0] >= 0.2:
            share, attr, counts = shares[0]
            next_share = shares[1][0] if len(shares) > 1 else 0.0
            out.append(
                Finding(
                    severity="medium",
                    title=f"{_attr(attr).capitalize()} is the weakest attribute: {pct(share)} negative",
                    detail=f"{counts.negative} of {counts.discussed} answers that discuss {client}'s {_attr(attr)} are negative; the next attribute is at {pct(next_share)}.",
                    action=_sources_action(overall, attr, client),
                )
            )

    # 4. Contradicted claims, grouped by the fact they contradict; the action comes from
    #    what those answers cited, not from a template
    for g in overall.contradictions.values():
        if g.count < 3:
            continue
        engines = ", ".join(f"{e} ({n})" for e, n in sorted(g.engines.items(), key=lambda kv: -kv[1]))
        example = f' For example, as the judge recorded it: "{g.examples[0]}".' if g.examples else ""
        unsourced = g.answers - g.answers_with_sources
        own = next((n for d, n in g.top_domains if any(d == od or d.endswith("." + od) for od in brief.client.domains)), 0)
        parts = []
        if g.answers_with_sources:
            doms = ", ".join(d for d, _ in g.top_domains[:3])
            parts.append(f"{g.answers_with_sources} of the {g.answers} answers carrying it cited pages (most often {doms})")
            if own:
                parts.append(
                    f"{client}'s own site is cited {own} times alongside the wrong figure, so check that the page the engines read states the current one plainly"
                )
            else:
                parts.append("ask those sites to update, and publish the current figure where the engines look")
        if unsourced:
            parts.append(f"{_plural(unsourced, 'answer')} showed no source, so there is no page to fix there; re-check next month")
        out.append(
            Finding(
                severity="high" if g.count >= 10 else "medium",
                title=f"{g.count} claims contradict a verified fact: {_clip(g.statement)}",
                detail=f"From {engines}.{example}",
                action=". ".join(parts) + "." if parts else "",
                engines=list(g.engines),
            )
        )
    small = sum(g.count for g in overall.contradictions.values() if g.count < 3)
    if small:
        out.append(
            Finding(
                severity="low",
                title=f"{small} further contradicted claim(s) on other facts",
                detail="Listed under each answer in the drilldown with the fact they conflict with.",
                action="Spot-read them: a few are the judge being strict (a range against a point figure).",
            )
        )

    # 5. Own pages / forums
    own_share = overall.citation_share_by_type.get("brand_owned", 0.0)
    if overall.citations_total >= 10 and own_share < 0.2:
        out.append(
            Finding(
                severity="medium",
                title=f"{client}'s own pages are {pct(own_share)} of citations",
                detail="The engines are describing the brand from third parties.",
                action="Make the deciding facts readable where the engines look: plain HTML fee and rate tables (not PDFs or app-only screens), a comparison page in the brand's own words, answers to the exact questions in this set.",
            )
        )
    forum_share = overall.citation_share_by_type.get("forum_social", 0.0)
    if overall.citations_total >= 10 and forum_share >= 0.2:
        out.append(
            Finding(
                severity="medium",
                title=f"Forums and social are {pct(forum_share)} of citations",
                detail="Service complaints and community threads are being read as evidence.",
                action="CX and comms own this, not SEO: answer the threads the engines cite and fix what they complain about.",
            )
        )

    # 6. Engine gaps (only engines with at least ten unbranded answers; smaller samples are indicative)
    rates = {
        e: m.brands[client].presence_rate
        for e, m in per_engine.items()
        if m.brands.get(client) and m.brands[client].unbranded_answers >= 10 and m.brands[client].presence_rate is not None
    }
    if len(rates) >= 2:
        hi, lo = max(rates, key=rates.get), min(rates, key=rates.get)
        if rates[hi] - rates[lo] >= 0.25:
            lo_own = per_engine[lo].citation_share_by_type.get("brand_owned", 0.0)
            hi_own = per_engine[hi].citation_share_by_type.get("brand_owned", 0.0)
            missing_qs = [
                qid for qid, qm in per_engine[lo].by_question.items() if qm.answers and qm.present < qm.answers
            ]
            where = f" It leaves {client} out on {', '.join(missing_qs[:5])}." if missing_qs else ""
            if lo_own >= 0.8 * hi_own:
                action = (
                    f"{lo} cites {client}'s own site for {pct(lo_own)} of its sources ({hi}: {pct(hi_own)}), so this is not a missing-pages problem. "
                    f"Read what {lo} says on those questions (section 3) and which rival it picks there."
                )
            else:
                action = f"{lo} cites {client}'s own site for {pct(lo_own)} of its sources against {hi}'s {pct(hi_own)}; the pages {lo} reads instead are listed in section 6."
            out.append(
                Finding(
                    severity="medium",
                    title=f"Engines disagree: {hi} names {client} in {pct(rates[hi])} of unbranded questions, {lo} in {pct(rates[lo])}",
                    detail=f"Each engine reads different sources, so presence differs by engine (engines with fewer than ten unbranded answers are left out).{where}",
                    action=action,
                    engines=[hi, lo],
                )
            )

    # 6b. Question-level gaps: named but never put forward, or missing from most answers
    for qm in overall.by_question.values():
        if qm.answers < 8:
            continue
        others = ", ".join(f"{b} ({n})" for b, n in list(qm.others_recommended.items())[:2]) or "no tracked brand"
        if qm.present >= 0.5 * qm.answers and qm.recommended == 0:
            out.append(
                Finding(
                    severity="high",
                    title=f'Named but never put forward on "{qm.text}"',
                    detail=f"{client} is named in {qm.present} of {qm.answers} answers and recommended in none; the answers pick {others}.",
                    action=f"{others} are picked on it; the reasons the engines give the leader are in the recommendation finding in this list. If {client}'s verified facts answer this question, they are not reaching the engines in these words: put them on a page that answers it directly.",
                )
            )
        elif qm.present <= 0.25 * qm.answers:
            out.append(
                Finding(
                    severity="medium",
                    title=f'Absent from most answers to "{qm.text}"',
                    detail=f"{client} is named in {qm.present} of {qm.answers} answers; the answers pick {others}.",
                    action=f"{client} is not in the running on this question. Decide whether it should be; if so, the pages the engines cite for it (section 6) are where it has to appear.",
                )
            )

    # 8. Brands the brief does not track
    loud = sorted(
        ((w.presence_rate or 0, w.brand) for w in overall.watch.values() if w.unbranded_answers >= 8),
        reverse=True,
    )
    loud = [(r, b) for r, b in loud if r >= 0.2]
    if loud:
        out.append(
            Finding(
                severity="medium",
                title="Brands outside the tracked set keep appearing: " + ", ".join(f"{b} {pct(r)}" for r, b in loud[:4]),
                detail="Share of unbranded answers that name them. They are not scored, so the shortlist table understates the competition.",
                action="Add them to the brief as competitors for the next run.",
            )
        )

    # 9. Unstable answers
    for e, m in per_engine.items():
        if m.stance_consistency is not None and m.stance_consistency < 0.7:
            out.append(
                Finding(
                    severity="low",
                    title=f"{e} changes its mind on repeat runs",
                    detail=f"Same question, same day: the stance on {client} matched on {pct(m.stance_consistency)} of repeated questions.",
                    action="Report trends on several runs, never one.",
                    engines=[e],
                )
            )
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda f: order[f.severity])


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{round(x * 100)}%"
