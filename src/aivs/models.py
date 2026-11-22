"""Data model for a scan.

A scan is: a brand brief (client, competitors, attributes, facts) x a question set x
a set of answer engines x N repeat runs. Every answer is kept verbatim with its
citations, then scored in two layers: deterministic rules (is the brand present,
which sources were cited) and a model judge (recommended or just mentioned,
sentiment per attribute, whether claims about the client match its facts).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Stance = Literal["recommended", "mentioned", "absent"]
Sentiment = Literal["positive", "neutral", "negative", "not_discussed"]
Verdict = Literal["supported", "contradicted", "unverifiable"]
Stage = Literal["discover", "compare", "choose", "trust", "switch"]
SourceType = Literal[
    "brand_owned",
    "competitor_owned",
    "review_comparison",
    "forum_social",
    "news",
    "regulator_official",
    "wikipedia",
    "other",
]


# ---------- brief ----------


class TrackedBrand(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)

    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]


class Fact(BaseModel):
    """A claim about the client the scan can check answers against."""

    id: str
    statement: str
    source: str = ""
    status: Literal["verified", "to_confirm"] = "to_confirm"


class Question(BaseModel):
    id: str
    text: str
    stage: Stage
    branded: bool = False  # True when ANY tracked brand is named in the question (client or competitor)
    attributes: list[str] = Field(default_factory=list)


class Brief(BaseModel):
    client: TrackedBrand
    competitors: list[TrackedBrand] = Field(default_factory=list)
    category: str
    market: str = "GB"
    city: str = "London"
    attributes: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    source_types: dict[str, list[str]] = Field(default_factory=dict)
    watchlist: list[TrackedBrand] = Field(default_factory=list)  # named-but-not-tracked brands, counted by rule only

    def tracked(self) -> list[TrackedBrand]:
        return [self.client, *self.competitors]


# ---------- raw answers ----------


class Citation(BaseModel):
    url: str
    title: str = ""


class EngineAnswer(BaseModel):
    engine: str
    model: str
    question_id: str
    run: int
    text: str
    citations: list[Citation] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    asked_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    error: str | None = None
    synthetic: bool = False  # True for fixture answers used in offline demos
    access: str = "api"  # api | web (consumer surface through a browser) | fixture
    signed_in: bool = False  # web only: True when the surface was read on the user's own account
    note: str = ""  # anything the engine observed that a reader should know, e.g. "no sources shown"


# ---------- scoring ----------


class BrandJudgement(BaseModel):
    brand: str
    stance: Stance
    position: int | None = Field(
        default=None, description="1 = named first among the tracked brands in the answer"
    )
    attribute_sentiment: dict[str, Sentiment] = Field(default_factory=dict)
    evidence: str = Field(default="", description="Short quote from the answer backing the stance")


class Claim(BaseModel):
    text: str
    verdict: Verdict
    fact_id: str | None = None
    note: str = ""


class AnswerJudgement(BaseModel):
    brands: list[BrandJudgement]
    client_claims: list[Claim] = Field(default_factory=list)


class ClassifiedSource(BaseModel):
    url: str
    domain: str
    source_type: SourceType
    title: str = ""


class AnswerScore(BaseModel):
    engine: str
    question_id: str
    run: int
    present: dict[str, bool]  # deterministic: alias regex per tracked brand
    sources: list[ClassifiedSource]
    judgement: AnswerJudgement | None = None
    judge_error: str | None = None


# ---------- aggregate ----------


class AttributeSentimentCounts(BaseModel):
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    not_discussed: int = 0

    @property
    def discussed(self) -> int:
        return self.positive + self.neutral + self.negative


class BrandMetrics(BaseModel):
    brand: str
    unbranded_answers: int = 0
    present_in_unbranded: int = 0
    recommended_in_unbranded: int = 0
    positions: list[int] = Field(default_factory=list)
    positive_attributes: dict[str, int] = Field(default_factory=dict)  # in the answers that recommend it
    evidence: list[str] = Field(default_factory=list)  # a few judge quotes from those answers

    @property
    def presence_rate(self) -> float | None:
        return _rate(self.present_in_unbranded, self.unbranded_answers)

    @property
    def recommendation_rate(self) -> float | None:
        return _rate(self.recommended_in_unbranded, self.unbranded_answers)

    @property
    def avg_position(self) -> float | None:
        return round(sum(self.positions) / len(self.positions), 2) if self.positions else None


class QuestionMetrics(BaseModel):
    """The client on one unbranded question, across engines and runs."""

    question_id: str
    text: str
    stage: str
    answers: int = 0
    present: int = 0
    recommended: int = 0
    others_recommended: dict[str, int] = Field(default_factory=dict)


class StageMetrics(BaseModel):
    """Client visibility per journey stage. Unbranded and branded answers are kept
    apart: a branded question names the brand, so presence there is a given and
    "recommended" mostly echoes the question ("how do I switch to X")."""

    stage: str
    questions: int = 0
    branded_questions: int = 0
    answers: int = 0  # judged UNBRANDED answers in this stage
    present: int = 0
    recommended: int = 0
    branded_answers: int = 0  # judged branded answers
    branded_recommended: int = 0

    @property
    def presence_rate(self) -> float | None:
        return _rate(self.present, self.answers)

    @property
    def recommendation_rate(self) -> float | None:
        return _rate(self.recommended, self.answers)

    @property
    def branded_recommendation_rate(self) -> float | None:
        return _rate(self.branded_recommended, self.branded_answers)


class ContradictionGroup(BaseModel):
    """Contradicted claims about the client, grouped by the fact they contradict."""

    fact_id: str
    statement: str = ""
    count: int = 0
    engines: dict[str, int] = Field(default_factory=dict)
    examples: list[str] = Field(default_factory=list)  # a few claim texts, verbatim
    answers: int = 0  # distinct answers carrying at least one such claim
    answers_with_sources: int = 0
    top_domains: list[tuple[str, int]] = Field(default_factory=list)  # cited in those answers


class EngineMetrics(BaseModel):
    engine: str
    access_label: str = ""  # "web · signed out" | "web · signed in" | "API" | "fixture" | "mixed"
    answers: int = 0  # attempted (answered + failed)
    errors: int = 0
    brands: dict[str, BrandMetrics] = Field(default_factory=dict)
    share_of_voice: float | None = None  # client's share of all "recommended" verdicts
    client_sentiment: dict[str, AttributeSentimentCounts] = Field(default_factory=dict)
    claims_supported: int = 0
    claims_contradicted: int = 0
    claims_unverifiable: int = 0
    citations_total: int = 0
    citation_share_by_type: dict[str, float] = Field(default_factory=dict)
    top_domains: list[tuple[str, int]] = Field(default_factory=list)
    answers_citing_client_domain: int = 0
    answers_without_sources: int = 0  # answered, but the engine returned no citations at all
    stance_consistency: float | None = None  # agreement across repeat runs, same question
    recommended_total: int = 0  # denominator of share_of_voice
    repeated_questions: int = 0  # denominator of stance_consistency
    by_stage: dict[str, StageMetrics] = Field(default_factory=dict)
    watch: dict[str, BrandMetrics] = Field(default_factory=dict)  # watchlist brands, presence by rule only
    contradictions: dict[str, ContradictionGroup] = Field(default_factory=dict)
    by_question: dict[str, QuestionMetrics] = Field(default_factory=dict)  # unbranded questions only
    negative_sources: dict[str, list[tuple[str, int]]] = Field(default_factory=dict)  # attr -> domains cited in negative answers

    @property
    def answered(self) -> int:
        return self.answers - self.errors

    @property
    def accuracy_rate(self) -> float | None:
        return _rate(self.claims_supported, self.claims_supported + self.claims_contradicted)

    @property
    def claims_checked(self) -> int:
        return self.claims_supported + self.claims_contradicted


class Finding(BaseModel):
    severity: Literal["high", "medium", "low"]
    title: str
    detail: str
    action: str = ""  # what the brand team does about it
    engines: list[str] = Field(default_factory=list)


class ScanResult(BaseModel):
    scan_id: str
    brief: Brief
    engines: list[str]
    runs: int
    answers: list[EngineAnswer]
    scores: list[AnswerScore]
    per_engine: dict[str, EngineMetrics]
    overall: EngineMetrics
    findings: list[Finding]
    synthetic: bool = False
    judge_model: str = ""
    brief_path: str = ""  # the YAML the scan was run from, so a rescore picks up edited facts
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None
