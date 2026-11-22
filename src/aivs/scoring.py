"""Two-layer scoring: rules first, then a judge.

Layer 1 (deterministic): is each tracked brand named in the answer (alias regex),
and which sources were cited (see sources.py). The judge never gets to say a brand
is present when the text does not name it.

Layer 2 (model judge, structured output): for the brands that ARE present:
recommended or merely mentioned, order of mention, sentiment per attribute, and a
check of the answer's claims about the client against the brief's facts. The judge
is validated against a hand-labelled sample (evals/), the same way the planted-bug
set checks the code reviewer: we measure the judge before we trust it.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .models import (
    AnswerJudgement,
    AnswerScore,
    BrandJudgement,
    Brief,
    Claim,
    EngineAnswer,
    Question,
    Sentiment,
    Stance,
    TrackedBrand,
)
from .sources import classify_all

# ---------- layer 1 ----------


def _alias_pattern(brand: TrackedBrand) -> re.Pattern:
    names = sorted({n for n in brand.all_names() if n}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])", re.I)


def presence(text: str, brief: Brief) -> dict[str, bool]:
    return {b.name: bool(_alias_pattern(b).search(text)) for b in brief.tracked()}


def first_positions(text: str, brief: Brief) -> dict[str, int]:
    """1-based order of first mention among the tracked brands that appear."""
    firsts = []
    for b in brief.tracked():
        m = _alias_pattern(b).search(text)
        if m:
            firsts.append((m.start(), b.name))
    return {name: i + 1 for i, (_, name) in enumerate(sorted(firsts))}


# ---------- layer 2 ----------


class Judge(Protocol):
    model: str

    def judge(self, answer: EngineAnswer, question: Question, brief: Brief, present: dict[str, bool]) -> AnswerJudgement: ...


JUDGE_SYSTEM = """You score one AI assistant's answer to a shopper's question for a brand-visibility report.
You are given the tracked brands, the attributes that decide purchase in this category, and a list of
verified facts about the client brand. Judge ONLY what the answer text says. Never infer from outside knowledge.

Stance per tracked brand that is present in the text:
- "recommended": the answer puts the brand forward as a choice for THIS question (named as the pick, the best,
  a top option, "go with", or in an endorsed shortlist).
- "mentioned": named, but without endorsement for this question (an aside, a comparison point, a caveat, a negative).
Brands you are told are absent must be returned with stance "absent".

position: 1 for the tracked brand named first, 2 for the next, and so on; null when absent.
attribute_sentiment: one entry for EVERY attribute in the attributes list, for every brand that is present:
how the answer characterises that brand on that attribute: "positive", "neutral", "negative", or
"not_discussed". Never leave the list empty for a present brand; use "not_discussed" when the answer
says nothing about it. Absent brands get an empty list.
evidence: a short quote (max 20 words) from the answer that supports the stance.

client_claims: every factual statement the answer makes about the client brand (prices, fees, rates, limits,
protection, customer numbers, features, dates). For each, compare with the facts list:
"supported" (matches a fact; give fact_id), "contradicted" (give fact_id and say how), "unverifiable" (no fact
covers it). "contradicted" is reserved for a statement that cannot be true alongside the fact: a different
figure, limit, price, name or condition. It is NOT contradicted when the answer gives a range or bound that
contains the fact's figure ("up to £19" against "from £19", "2.75-3.25%" against 2.80%), rounds within a
few percent, describes a condition or plan the fact does not cover, or is vaguer than the fact; those are
"unverifiable". A claim that matches one figure or part of a fact is "supported" (a fact listing three plan
prices supports a claim that quotes one of them correctly), even if the rest of the sentence says something
the facts do not cover. Do not invent claims; quote them.
Return only the judgement object, nothing else."""


class _AttrSentiment(BaseModel):
    attribute: str
    sentiment: Sentiment


class _JudgeBrand(BaseModel):
    brand: str
    stance: Stance
    position: int | None = Field(default=None, description="1 = named first among the tracked brands; null when absent")
    attribute_sentiment: list[_AttrSentiment] = Field(
        description="One entry per attribute in the attributes list for a present brand; empty for an absent brand"
    )
    evidence: str = Field(description="Short quote from the answer backing the stance; empty when absent")


class JudgeOutput(BaseModel):
    """What the model returns. A list of (attribute, sentiment) pairs instead of a
    free-form dict, because structured output needs a closed schema: the dict form
    came back empty on every answer in the first live rescore."""

    brands: list[_JudgeBrand]
    client_claims: list[Claim]

    def to_judgement(self, attributes: list[str]) -> AnswerJudgement:
        rows = []
        for b in self.brands:
            sent = {a.attribute: a.sentiment for a in b.attribute_sentiment if a.attribute in attributes}
            if b.stance != "absent":
                for a in attributes:
                    sent.setdefault(a, "not_discussed")
            rows.append(
                BrandJudgement(brand=b.brand, stance=b.stance, position=b.position, attribute_sentiment=sent, evidence=b.evidence)
            )
        return AnswerJudgement(brands=rows, client_claims=self.client_claims)


class ClaudeJudge:
    """Structured output: `messages.parse(output_format=JudgeOutput)` makes the
    API return JSON that validates against the model. (Forcing a tool with
    tool_choice was the old way; Claude Sonnet 5.5 and Opus 5.5 answer that with
    400 "tool_choice: type tool and any are not supported for this model", which
    is exactly what sank the first live rescore on 21 Nov 2026.)"""

    def __init__(self, settings: Settings, client=None) -> None:
        if client is None:
            if not settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for the judge)")
            from anthropic import Anthropic

            client = Anthropic(api_key=settings.anthropic_api_key, timeout=settings.request_timeout_s)
        self._client = client
        self.model = settings.judge_model
        self._max_tokens = 8000  # thinking is on by default and counts against this

    def judge(self, answer: EngineAnswer, question: Question, brief: Brief, present: dict[str, bool]) -> AnswerJudgement:
        user = _judge_user_message(answer, question, brief, present)
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=self._max_tokens,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_format=JudgeOutput,
            output_config={"effort": "medium"},
        )
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            raise RuntimeError(f"judge returned no judgement (stop_reason={getattr(resp, 'stop_reason', None)})")
        try:
            out = parsed if isinstance(parsed, JudgeOutput) else JudgeOutput.model_validate(parsed)
        except ValidationError as exc:
            raise RuntimeError(f"judge output failed validation: {exc}") from exc
        return out.to_judgement(brief.attributes)


def _judge_user_message(answer: EngineAnswer, question: Question, brief: Brief, present: dict[str, bool]) -> str:
    brands = [
        {"name": b.name, "aliases": b.aliases, "present_in_text": present.get(b.name, False)}
        for b in brief.tracked()
    ]
    facts = [{"id": f.id, "statement": f.statement} for f in brief.facts if f.status == "verified"]
    payload = {
        "client_brand": brief.client.name,
        "category": brief.category,
        "attributes": brief.attributes,
        "tracked_brands": brands,
        "client_facts": facts,
        "question": question.text,
        "answer_text": answer.text,
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class HeuristicJudge:
    """Offline judge for tests and the no-keys demo. Keyword rules only, scoped to
    the sentence that names the brand; the report labels its output as heuristic.
    It exists so the pipeline runs end to end without a key, not as a substitute
    for the model judge."""

    model = "heuristic"
    _ENDORSE = re.compile(
        r"\b(best|top pick|top choice|recommend\w*|go with|great choice|better (?:pick|choice|option)|"
        r"strong (?:choice|option|pick)|stands? out|stand-out|standout|winner|my pick|first choice|ideal|"
        r"excellent if|safest picks?)\b",
        re.I,
    )
    # Negated or compound fee/charge phrases are positive and are counted first, then
    # removed, so "no fees", "fee-free", "doesn't charge" never read as negative.
    _NEG_FEE_IS_POS = re.compile(
        r"\b(?:no|zero|without|free of|free from|never charg\w*|(?:doesn't|does not|don't|do not|won't|will not) charge)"
        r"(?:\s+\w+){0,2}?\s+(?:fees?|charges?)\b|\b(?:fee|charge|commission)[- ]free\b|\bfree (?:of charge|to use)\b",
        re.I,
    )
    _NEG = re.compile(
        r"\b(?:fees?|charges?|charged|complain\w*|frozen|freez\w*|poor|slow|limited|downsides?|worse|lower)\b", re.I
    )
    _POS = re.compile(
        r"\b(?:free|excellent|great|easy|fast|strong|best|higher|generous|protected)\b", re.I
    )
    _STOPWORDS = frozenset({"and", "or", "of", "the", "a", "an", "to", "for", "in", "on", "with"})

    def judge(self, answer: EngineAnswer, question: Question, brief: Brief, present: dict[str, bool]) -> AnswerJudgement:
        text = answer.text
        positions = first_positions(text, brief)
        out: list[BrandJudgement] = []
        for b in brief.tracked():
            if not present.get(b.name):
                out.append(BrandJudgement(brand=b.name, stance="absent"))
                continue
            stance = "mentioned"
            evidence = ""
            pat = _alias_pattern(b)
            for sentence in _SENTENCE_SPLIT.split(text):
                if pat.search(sentence) and self._ENDORSE.search(sentence):
                    stance = "recommended"
                    evidence = sentence.strip().replace("\n", " ")[:120]
                    break
            sentiment = {}
            for attr in brief.attributes:
                sentiment[attr] = self._attr_sentiment(text, b, attr)
            out.append(
                BrandJudgement(
                    brand=b.name,
                    stance=stance,
                    position=positions.get(b.name),
                    attribute_sentiment=sentiment,
                    evidence=evidence,
                )
            )
        return AnswerJudgement(brands=out, client_claims=[])

    @classmethod
    def attribute_pattern(cls, attr: str) -> re.Pattern:
        """Whole words only, stopwords dropped, singular and plural: 'app_and_features'
        matches app/apps/feature/features and never the 'and' in every sentence."""
        words = [w for w in attr.lower().replace("_", " ").split() if w not in cls._STOPWORDS]
        stems = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
        return re.compile(r"\b(?:" + "|".join(re.escape(s) + r"(?:e?s)?" for s in stems) + r")\b", re.I)

    @classmethod
    def polarity(cls, window: str) -> tuple[int, int]:
        """(negative, positive) keyword counts with fee negation handled."""
        pos = len(cls._NEG_FEE_IS_POS.findall(window))
        rest = cls._NEG_FEE_IS_POS.sub(" ", window)
        neg = len(cls._NEG.findall(rest))
        pos += len(cls._POS.findall(rest))
        return neg, pos

    def _attr_sentiment(self, text: str, brand: TrackedBrand, attr: str) -> str:
        pat = self.attribute_pattern(attr)
        for m in _alias_pattern(brand).finditer(text):
            window = text[max(0, m.start() - 200) : m.end() + 200]
            if not pat.search(window):
                continue
            neg, pos = self.polarity(window)
            if neg > pos:
                return "negative"
            if pos > neg:
                return "positive"
            return "neutral"
        return "not_discussed"


# ---------- putting the layers together ----------


def enforce_presence(j: AnswerJudgement, present: dict[str, bool], brief: Brief) -> AnswerJudgement:
    """The rules win: a brand the text never names is absent whatever the judge said,
    and every tracked brand gets a row."""
    by_name = {b.brand: b for b in j.brands}
    rows: list[BrandJudgement] = []
    for b in brief.tracked():
        row = by_name.get(b.name) or BrandJudgement(brand=b.name, stance="absent")
        if not present.get(b.name):
            row = BrandJudgement(brand=b.name, stance="absent")
        elif row.stance == "absent":
            row = row.model_copy(update={"stance": "mentioned"})
        rows.append(row)
    return AnswerJudgement(brands=rows, client_claims=j.client_claims)


def score_answer(answer: EngineAnswer, question: Question, brief: Brief, judge: Judge | None) -> AnswerScore:
    present = presence(answer.text, brief) if answer.text else {b.name: False for b in brief.tracked()}
    sources = classify_all(answer.citations, brief)
    score = AnswerScore(
        engine=answer.engine, question_id=answer.question_id, run=answer.run, present=present, sources=sources
    )
    if judge is None or answer.error or not answer.text:
        return score
    try:
        j = judge.judge(answer, question, brief, present)
        score.judgement = enforce_presence(j, present, brief)
    except Exception as exc:  # noqa: BLE001 - one judge failure must not sink the scan
        score.judge_error = f"{type(exc).__name__}: {exc}"[:300]
    return score
