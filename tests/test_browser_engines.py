"""Browser engines without a browser: the pure helpers, the registry and the
settings. Live behaviour is checked by `aivs scan --engines web --limit 1`."""

import pytest

from aivs.config import Settings
from aivs.engines.base import build_engines
from aivs.engines.browser import (
    BROWSER_ENGINES,
    BraveWeb,
    BrowserSession,
    DuckWeb,
    GeminiWeb,
    PerplexityWeb,
    clean_links,
    is_bot_check,
    is_login_wall,
    labels_from_aria,
    strip_question,
    strip_volatile,
)
from aivs.sources import domain_of


def test_strip_volatile_removes_spinners_and_clock_times():
    assert strip_volatile("Sources ⠼ 12:47 Answer") == "Sources   Answer"
    assert strip_volatile("no change") == "no change"


def test_bot_check_detection_matches_what_the_surfaces_showed():
    assert is_bot_check("Verifying you're not a bot\nQuick check before you continue searching.")  # Brave
    assert is_bot_check("Just a moment...")  # Cloudflare
    assert is_bot_check("Our systems have detected unusual traffic from your computer network")  # Google
    assert is_bot_check("Please complete the following challenge to confirm this prompt was made by a human")  # Duck.ai
    assert not is_bot_check("Starling Bank is the most consistently recommended option for UK travellers.")


def test_clean_links_dedupes_and_drops_the_surface_itself():
    pairs = [
        {"h": "https://www.which.co.uk/a", "t": "which.co"},
        {"h": "https://www.which.co.uk/a", "t": "Which? best debit cards"},
        {"h": "https://www.perplexity.ai/hub/legal/terms-of-service", "t": "Terms"},
        {"h": "javascript:void(0)", "t": "x"},
        {"h": "https://wise.com/gb/blog/travel-bank-account", "t": ""},
        {"h": "https://wise.com/gb/blog/travel-bank-account#:~:text=All%20inclusive", "t": "Wise"},
    ]
    out = clean_links(pairs, ("perplexity.ai",))
    assert [c.url for c in out] == ["https://www.which.co.uk/a", "https://wise.com/gb/blog/travel-bank-account"]
    assert out[0].title == "which.co"


def test_strip_question_removes_echo_and_ui_chrome():
    prompt = "What is the best bank account? (I'm in the UK.)"
    thread = f"12:47\n{prompt}\n\n1s\nFinished\n\nStarling is the pick.\nwhich.co\n+2\n\nSources\n"
    assert strip_question(thread, prompt, (r"Finished", r"Sources")) == "Starling is the pick.\nwhich.co"


def test_gemini_chip_labels_are_parsed_from_aria():
    assert labels_from_aria("View source details for citations from MoneySuperMarket and Wise. Press Enter to open sources dialog.") == ["MoneySuperMarket", "Wise"]
    assert labels_from_aria("View source details for citation from Which.co.uk. Press Enter to open sources dialog.") == ["Which.co.uk"]
    assert labels_from_aria("Copy") == []


def test_domain_of_falls_back_to_the_source_name_when_url_is_hidden():
    assert domain_of("", "Which.co.uk") == "which.co.uk"
    assert domain_of("", "MoneySuperMarket") == "moneysupermarket"
    assert domain_of("", "") == "unknown"


def test_registry_builds_all_four_browser_engines_sharing_one_chrome_without_launching():
    s = Settings()  # no keys
    engines = build_engines(["web"], s)
    assert [type(e) for e in engines] == [PerplexityWeb, GeminiWeb, BraveWeb, DuckWeb]
    sessions = {id(e._s) for e in engines}
    assert len(sessions) == 1  # one BrowserSession
    assert engines[0]._s._ctx is None  # nothing launched until the first question
    assert set(BROWSER_ENGINES) == set(Settings.keyless_engines()) | set(Settings.login_engines())


def test_keyless_engines_are_available_with_no_keys_and_keyed_ones_need_keys():
    s = Settings()
    assert s.available_engines() == ["perplexity-web", "gemini-web", "brave-web", "duckai-web"]
    s2 = Settings(anthropic_api_key="k")
    assert s2.available_engines()[-1] == "claude"


def test_headless_flags_record_what_was_observed():
    assert PerplexityWeb.headless_ok and GeminiWeb.headless_ok
    assert not BraveWeb.headless_ok and not DuckWeb.headless_ok
    assert BrowserSession(Settings()).headless is False


def test_grok_web_is_opt_in_and_needs_login():
    from aivs.engines.browser import LOGIN_SITES, GrokWeb

    s = Settings()
    assert "grok-web" not in s.keyless_engines()
    assert s.login_engines() == ["grok-web"]
    assert GrokWeb.needs_login and not GrokWeb.headless_ok
    assert LOGIN_SITES["grok"] == "https://grok.com/"
    engines = build_engines(["web", "grok-web"], s)
    assert [e.name for e in engines][-1] == "grok-web"
    assert len({id(e._s) for e in engines}) == 1  # shares the same Chrome as the others


def test_login_wall_detection_matches_perplexity_and_grok_walls_and_is_not_a_bot_check():
    pplx = "Finished\n3s\nSign up and repeat your request.\nSources\nSign in to continue using Perplexity"
    grok = "Continue your conversation\nSign up to continue seamlessly with Grok's full power"
    assert is_login_wall(pplx) and is_login_wall(grok)
    assert not is_bot_check(pplx) and not is_bot_check(grok)
    assert not is_login_wall("Sign In\nWhat do you want to know?")  # the ordinary header button is not a wall


def test_perplexity_offers_own_login_as_the_quota_fallback():
    from aivs.engines.browser import LOGIN_SITES

    assert PerplexityWeb.login_site == "perplexity" and "perplexity" in LOGIN_SITES
    assert "perplexity-web" in Settings.keyless_engines()  # still keyless by default


def test_api_alias_expands_to_every_keyed_engine_and_opens_no_chrome():
    s = Settings(anthropic_api_key="k", xai_api_key="k")
    engines = build_engines(["api"], s)
    assert [e.name for e in engines] == ["claude", "grok"]
    assert all(getattr(e, "access", "api") == "api" for e in engines)


def test_api_alias_without_keys_says_which_keys_to_set():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_engines(["api"], Settings())


def test_signed_in_comes_from_the_engine_not_from_a_string():
    s = Settings()
    engines = {e.name: e for e in build_engines(["web", "grok-web"], s)}
    assert engines["grok-web"]._signed_in(None) is True  # needs a login, so signed in by construction
    assert engines["gemini-web"]._signed_in(None) is False
    assert engines["brave-web"]._signed_in(None) is False
