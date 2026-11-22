from pathlib import Path

from aivs.config import load_brief
from aivs.models import AnswerJudgement, BrandJudgement, EngineAnswer, Question
from aivs.scoring import HeuristicJudge, enforce_presence, first_positions, presence, score_answer
from aivs.sources import classify, domain_of

BRIEF = load_brief(Path(__file__).resolve().parents[1] / "brands" / "monzo.yaml")


def test_presence_uses_aliases_and_word_boundaries():
    text = "Chase UK is fine. Monzo's Pots are great. Nationwidex is not a brand."
    p = presence(text, BRIEF)
    assert p["Chase"] is True
    assert p["Monzo"] is True
    assert p["Nationwide"] is False
    assert p["Starling"] is False


def test_first_positions_order_of_mention():
    text = "Starling first, then Monzo, and Revolut last."
    assert first_positions(text, BRIEF) == {"Starling": 1, "Monzo": 2, "Revolut": 3}


def test_enforce_presence_overrides_judge():
    present = {"Monzo": False, "Starling": True, "Revolut": False, "Chase": False, "Nationwide": False}
    j = AnswerJudgement(brands=[BrandJudgement(brand="Monzo", stance="recommended"), BrandJudgement(brand="Starling", stance="absent")])
    fixed = enforce_presence(j, present, BRIEF)
    by = {b.brand: b.stance for b in fixed.brands}
    assert by["Monzo"] == "absent"  # judge said present, text did not name it
    assert by["Starling"] == "mentioned"  # text names it, judge said absent
    assert set(by) == {b.name for b in BRIEF.tracked()}


def test_heuristic_judge_recommends_on_endorsement_words():
    ans = EngineAnswer(engine="x", model="m", question_id="D1", run=1, text="If you want one main account, go with Starling. Monzo is also mentioned by some.")
    q = Question(id="D1", stage="discover", text="best?")
    j = HeuristicJudge().judge(ans, q, BRIEF, presence(ans.text, BRIEF))
    by = {b.brand: b for b in j.brands}
    assert by["Starling"].stance == "recommended"
    assert by["Monzo"].stance == "mentioned"
    assert by["Revolut"].stance == "absent"


def test_source_classification():
    assert classify("https://www.monzo.com/plans", "", BRIEF).source_type == "brand_owned"
    assert classify("https://www.starlingbank.com/x", "", BRIEF).source_type == "competitor_owned"
    assert classify("https://www.reddit.com/r/monzo/", "", BRIEF).source_type == "forum_social"
    assert classify("https://www.which.co.uk/money", "", BRIEF).source_type == "review_comparison"
    assert classify("https://en.wikipedia.org/wiki/Monzo", "", BRIEF).source_type == "wikipedia"
    assert classify("https://example.org/", "", BRIEF).source_type == "other"


def test_gemini_redirect_classified_on_title():
    url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbCdEf"
    assert domain_of(url, "moneysavingexpert.com") == "moneysavingexpert.com"
    assert classify(url, "which.co.uk", BRIEF).source_type == "review_comparison"


def test_score_answer_without_judge_keeps_rules_only():
    ans = EngineAnswer(engine="x", model="m", question_id="D1", run=1, text="Monzo and Starling.", citations=[])
    q = Question(id="D1", stage="discover", text="best?")
    s = score_answer(ans, q, BRIEF, judge=None)
    assert s.present["Monzo"] and s.present["Starling"]
    assert s.judgement is None and s.judge_error is None


class _BoomJudge:
    model = "boom"

    def judge(self, *a, **k):
        raise RuntimeError("provider down")


def test_score_answer_judge_failure_is_recorded_not_raised():
    ans = EngineAnswer(engine="x", model="m", question_id="D1", run=1, text="Monzo.", citations=[])
    q = Question(id="D1", stage="discover", text="best?")
    s = score_answer(ans, q, BRIEF, judge=_BoomJudge())
    assert s.judgement is None
    assert "provider down" in (s.judge_error or "")


def test_heuristic_attribute_pattern_ignores_stopwords_and_matches_whole_words():
    pat = HeuristicJudge.attribute_pattern("app_and_features")
    assert not pat.search("Monzo and Starling are banks")  # "and" must never count as the attribute
    assert pat.search("the app is quick") and pat.search("its apps") and pat.search("a feature") and pat.search("features")
    fees = HeuristicJudge.attribute_pattern("fees")
    assert fees.search("a £1 fee") and fees.search("no fees") and not fees.search("coffee")
    assert HeuristicJudge.attribute_pattern("savings_rates").search("the rate on savings")


def test_heuristic_polarity_treats_negated_fees_as_positive():
    assert HeuristicJudge.polarity("Monzo has no fees abroad") == (0, 1)
    assert HeuristicJudge.polarity("fee-free spending with the Mastercard rate") == (0, 1)
    assert HeuristicJudge.polarity("Monzo doesn't charge any fees on card payments") == (0, 1)
    assert HeuristicJudge.polarity("Monzo covers any fees") == (1, 0)  # a fee that exists is still a fee
    neg, pos = HeuristicJudge.polarity("Monzo charges a 3% fee at foreign ATMs over the limit, which is poor")
    assert neg > pos


def test_heuristic_sentiment_is_per_attribute_not_identical_across_all():
    text = "Monzo charges no fees abroad and the app is great. Starling is also fee-free."
    ans = EngineAnswer(engine="x", model="m", question_id="D4", run=1, text=text)
    q = Question(id="D4", stage="discover", text="abroad?")
    j = HeuristicJudge().judge(ans, q, BRIEF, presence(text, BRIEF))
    monzo = next(b for b in j.brands if b.brand == "Monzo")
    assert monzo.attribute_sentiment["fees"] == "positive"
    assert monzo.attribute_sentiment["spending_abroad"] == "positive"
    assert monzo.attribute_sentiment["app_and_features"] == "positive"
    assert monzo.attribute_sentiment["safety_and_trust"] == "not_discussed"
    assert monzo.attribute_sentiment["customer_service"] == "not_discussed"


def test_source_path_rules_beat_bare_host_rules():
    assert classify("https://www.moneysavingexpert.com/forums/x/y", "", BRIEF).source_type == "forum_social"
    assert classify("https://forums.moneysavingexpert.com/discussion/1", "", BRIEF).source_type == "review_comparison"  # host rule only
    assert classify("https://www.moneysavingexpert.com/banking/best-bank-accounts/", "", BRIEF).source_type == "review_comparison"
    assert classify("https://www.moneysavingexpert.com/forums", "", BRIEF).source_type == "forum_social"


def test_redirect_without_a_domain_in_the_title_falls_back_to_the_title():
    url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbCdEf"
    assert domain_of(url, "") == "unknown"
    assert domain_of(url, "Which?") == "which?"
    assert classify(url, "", BRIEF).domain == "unknown"


class _ParseClient:
    """Stand-in for anthropic.Anthropic: records the parse call, returns a parsed judgement."""

    class _Messages:
        def __init__(self):
            self.calls = []

        def parse(self, **kw):
            self.calls.append(kw)
            from types import SimpleNamespace

            from aivs.scoring import JudgeOutput

            out = JudgeOutput.model_validate({
                "brands": [
                    {"brand": "Monzo", "stance": "recommended", "position": 1, "evidence": "Go with Monzo.",
                     "attribute_sentiment": [{"attribute": "fees", "sentiment": "positive"}, {"attribute": "made_up", "sentiment": "negative"}]},
                    {"brand": "Starling", "stance": "absent", "position": None, "evidence": "", "attribute_sentiment": []},
                ],
                "client_claims": [],
            })
            return SimpleNamespace(parsed_output=out, stop_reason="end_turn")

    def __init__(self):
        self.messages = self._Messages()


def test_claude_judge_uses_structured_output_not_forced_tool_choice():
    from aivs.config import Settings
    from aivs.scoring import ClaudeJudge

    client = _ParseClient()
    judge = ClaudeJudge(Settings(anthropic_api_key="k"), client=client)
    ans = EngineAnswer(engine="x", model="m", question_id="D1", run=1, text="Go with Monzo.")
    q = Question(id="D1", stage="discover", text="best?")
    out = judge.judge(ans, q, BRIEF, presence(ans.text, BRIEF))
    from aivs.scoring import JudgeOutput

    monzo = out.brands[0]
    assert monzo.stance == "recommended" and monzo.position == 1
    assert monzo.attribute_sentiment["fees"] == "positive"
    assert "made_up" not in monzo.attribute_sentiment  # attributes outside the brief are dropped
    assert all(monzo.attribute_sentiment[a] == "not_discussed" for a in BRIEF.attributes if a != "fees")  # the rest filled in
    assert out.brands[1].attribute_sentiment == {}  # absent brand carries no sentiment
    call = client.messages.calls[0]
    assert call["output_format"] is JudgeOutput
    assert "tool_choice" not in call and "tools" not in call  # forced tool use is a 400 on Sonnet 5.5 / Opus 5.5
    assert "Monzo" in call["messages"][0]["content"] and call["model"] == "claude-sonnet-5-5"
