"""xAI (Grok) API adapter, no network: the Responses-shaped parsing and the key gate."""

from types import SimpleNamespace as NS

import pytest

from aivs.config import Settings
from aivs.engines.base import build_engines
from aivs.engines.openai_engine import parse_responses_output
from aivs.engines.xai_engine import XAI_BASE_URL, GrokEngine


def test_parse_responses_output_handles_xai_numeric_titles_and_missing_output_text():
    resp = NS(
        output_text="",
        output=[
            NS(type="web_search_call", action=NS(query="best UK bank account travel")),
            NS(
                type="message",
                content=[
                    NS(
                        type="output_text",
                        text="Starling is the pick [[1]](https://www.which.co.uk/a). Monzo caps ATM use [[2]](https://monzo.com/fees).",
                        annotations=[
                            NS(type="url_citation", url="https://www.which.co.uk/a", title="1", start_index=20, end_index=50),
                            NS(type="url_citation", url="https://monzo.com/fees", title="2", start_index=70, end_index=95),
                            NS(type="url_citation", url="https://monzo.com/fees", title="2", start_index=70, end_index=95),
                        ],
                    )
                ],
            ),
        ],
    )
    text, cites, queries = parse_responses_output(resp)
    assert text.startswith("Starling is the pick")
    assert [c.url for c in cites] == ["https://www.which.co.uk/a", "https://monzo.com/fees"]
    assert all(c.title == "" for c in cites)  # a citation number is not a title
    assert queries == ["best UK bank account travel"]


def test_grok_engine_requires_key_and_uses_xai_base_url():
    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        GrokEngine(Settings())
    eng = build_engines(["grok"], Settings(xai_api_key="k"))[0]
    assert isinstance(eng, GrokEngine)
    assert eng.model == "grok-4.7"
    assert str(eng._client.base_url).rstrip("/") == XAI_BASE_URL
    assert Settings(xai_api_key="k").keyed_engines() == ["grok"]
