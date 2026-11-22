"""Browser engines: the consumer answer surfaces themselves, read through a real
Chrome. No API key.

Why: the brief is about how a brand shows up where shoppers actually ask. The
API products (see anthropic_engine.py and friends) are the keyed stand-ins; these
engines ask the public web apps the way a shopper does, signed out, from a UK
browser, one question at a time, at a human pace.

Verified 21 Nov 2026 from London (see tests/ and the README for the method):
- Perplexity (perplexity.ai): answers signed out, cites sources. Works headless.
- Gemini (gemini.google.com): answers signed out since the no-login rollout, shows
  source chips. Works headless.
- Ask Brave (search.brave.com/ask): answers signed out, cites sources. Shows a
  bot check in headless Chrome, fine in a visible window.
- Duck.ai (duck.ai): answers signed out with web search and sources. Shows a
  "find the duck" check in headless Chrome, fine in a visible window.
- Google AI Mode: CAPTCHA on the first request. Out of scope.
- ChatGPT signed out: Cloudflare challenge in headless Chrome. Not included.
- Copilot: no guest mode from the UK. Not included.
- Grok signed out: accepts the question, then a sign-up card instead of an answer. Not included.

Rules of engagement, deliberately: one Chrome, one question at a time, a pause
between questions, no CAPTCHA solving. If a surface shows a bot check we wait for
a human to clear it in the visible window or record the question as failed. The
report shows every failure. UI selectors will drift; each engine keeps its
selectors in one place at the top of its class.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models import Brief, Citation, EngineAnswer, Question
from .base import error_answer, shopper_prompt

# ---------- pure helpers (unit-tested, no browser) ----------

BOT_CHECK_PATTERNS = (
    "verifying you're not a bot",
    "verifying you are not a bot",
    "just a moment...",
    "unusual traffic from your computer network",
    "confirm this prompt was made by a human",
    "select all squares containing",
    "drag the slider",
    "checking your browser",
    "g-recaptcha",
)

_VOLATILE = re.compile(r"[⠀-⣿]|\b\d{1,2}:\d{2}\b")
_BLANKS = re.compile(r"\n{3,}")
_ARIA_SOURCES = re.compile(r"citations? from (.+?)\. Press Enter", re.I)


LOGIN_WALL_PATTERNS = (
    "sign in to continue using perplexity",
    "sign up and repeat your request",
    "sign up to continue seamlessly",
    "anon-paywall",
)


def is_login_wall(text: str) -> bool:
    """A sign-in demand that blocks the answer (signed-out quota reached, or the
    surface never answers signed out). Different from a bot check: nothing to clear."""
    low = text.lower()
    return any(p in low for p in LOGIN_WALL_PATTERNS)


def strip_volatile(text: str) -> str:
    """Remove spinner glyphs and clock times so 'has the page stopped changing' is answerable."""
    return _VOLATILE.sub("", text)


def is_bot_check(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in BOT_CHECK_PATTERNS)


def clean_links(pairs: list[dict[str, Any]], exclude_hosts: tuple[str, ...]) -> list[Citation]:
    """Dedupe by URL, drop the surface's own links, keep the first non-empty title."""
    seen: set[str] = set()
    out: list[Citation] = []
    for p in pairs:
        url = (p.get("h") or "").strip().split("#", 1)[0]  # drop #:~:text= highlight fragments
        if not url.startswith("http"):
            continue
        if any(h in url for h in exclude_hosts):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(Citation(url=url, title=(p.get("t") or "").strip()[:160]))
    return out


_CHROME_LINES = (r"\d{1,2}:\d{2}", r"\d+s", r"\+\d+")  # timestamps, "1s" timers, "+2" source badges


def strip_question(text: str, prompt: str, ui_lines: tuple[str, ...] = ()) -> str:
    """Remove the echoed question and known interface chrome from a thread's text."""
    out = text.replace(prompt, "")
    pat = re.compile(r"^(?:" + "|".join(_CHROME_LINES + tuple(ui_lines)) + r")\s*$", re.I | re.M)
    out = pat.sub("", out)
    return _BLANKS.sub("\n\n", out).strip()


def labels_from_aria(aria: str) -> list[str]:
    """'View source details for citations from MoneySuperMarket and Wise. Press Enter...'
    -> ['MoneySuperMarket', 'Wise']."""
    m = _ARIA_SOURCES.search(aria or "")
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r",\s*|\s+and\s+", raw)
    return [p.strip() for p in parts if p.strip()]


class BotCheck(RuntimeError):
    pass


class LoginWall(RuntimeError):
    pass


# ---------- one Chrome per scan ----------


class BrowserSession:
    """Lazily launches one persistent Chrome profile shared by every browser engine
    in a scan, so cookies and consent choices survive between questions and runs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pw = None
        self._ctx = None
        self._last_nav = 0.0

    @property
    def headless(self) -> bool:
        return self._settings.browser_headless

    def page(self):
        if self._ctx is None:
            self._launch()
        return self._ctx.new_page()

    def _launch(self) -> None:
        from playwright.sync_api import sync_playwright

        s = self._settings
        Path(s.browser_profile_dir).mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        kwargs = dict(
            user_data_dir=str(s.browser_profile_dir),
            headless=s.browser_headless,
            args=(["--headless=new"] if s.browser_headless else []) + ["--disable-blink-features=AutomationControlled"],
            locale=s.browser_locale,
            timezone_id=s.browser_timezone,
            viewport={"width": 1280, "height": 900},
        )
        try:
            self._ctx = self._pw.chromium.launch_persistent_context(channel=s.browser_channel, **kwargs)
        except Exception as exc:  # installed Chrome missing: fall back to Playwright's Chromium
            msg = str(exc)
            try:
                self._ctx = self._pw.chromium.launch_persistent_context(**kwargs)
            except Exception as exc2:
                raise RuntimeError(
                    f"could not start Chrome ({s.browser_channel}: {msg[:120]}) or Playwright Chromium "
                    f"({exc2!s:.120}). Install Google Chrome or run: playwright install chromium"
                ) from exc2

    def pace(self) -> None:
        """Human pace between navigations."""
        gap = self._settings.browser_pace_s - (time.monotonic() - self._last_nav)
        if self._last_nav and gap > 0:
            time.sleep(gap)
        self._last_nav = time.monotonic()

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.close()
            finally:
                self._ctx = None
        if self._pw is not None:
            try:
                self._pw.stop()
            finally:
                self._pw = None


# ---------- the shared flow ----------

_CONSENT_BUTTONS = ("Accept all", "Accept All", "Accept all cookies", "Accept", "Reject all", "Got it", "Continue", "Stay logged out", "Not now", "No thanks", "I agree")
_LINKS_JS = """() => Array.from(document.querySelectorAll('a[href^="http"]')).map(a => ({h: a.href, t: (a.innerText || a.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' ').slice(0, 160)}))"""
_DEEPEST_JS = """([marker, minLen]) => {
  let best = null;
  for (const e of document.querySelectorAll('body *')) {
    const t = e.innerText || '';
    if (t.length < minLen || !t.includes(marker)) continue;
    if (!best || t.length <= (best.innerText || '').length) best = e;
  }
  return best ? best.innerText : '';
}"""


class BrowserEngine:
    """Common ask() flow. Subclasses set the surface's URL and selectors and may
    override the hooks for answer text and sources."""

    name = "browser"
    model = "web · signed out"
    access = "web"
    url = ""
    input_selector = ""
    answer_selector: str | None = None  # CSS for the assistant message; None -> thread text minus question
    exclude_hosts: tuple[str, ...] = ()
    ui_lines: tuple[str, ...] = ()
    headless_ok = True  # False: showed a bot check headless when tested
    needs_login = False  # True: the surface answers only when signed in (opt-in, your own account)
    login_site = ""  # key in LOGIN_SITES when `aivs login <site>` is an option
    min_wait_s = 8.0

    def __init__(self, session: BrowserSession, settings: Settings) -> None:
        self._s = session
        self._settings = settings

    # -- public --

    def ask(self, question: Question, brief: Brief) -> EngineAnswer:
        prompt = shopper_prompt(question, brief)
        t0 = time.perf_counter()
        page = None
        try:
            if self._s.headless and not self.headless_ok:
                self._say(f"{self.name}: showed a bot check in headless Chrome when tested; trying anyway")
            page = self._s.page()
            self._s.pace()
            page.goto(self.url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(int(self._settings.browser_settle_ms))
            self._dismiss(page)
            self._guard_bot_check(page)
            box = page.locator(self.input_selector).first
            box.wait_for(state="visible", timeout=30_000)
            box.click()
            box.type(prompt, delay=12)
            box.press("Enter")
            self._after_submit(page, box, prompt)
            self._wait_for_answer(page, prompt)
            text = self._answer_text(page, prompt)
            if not text.strip():
                raise RuntimeError("answer text came back empty (selector drift?)")
            citations = self._citations(page)
            return EngineAnswer(
                engine=self.name,
                model=self._model_label(page),
                question_id=question.id,
                run=1,
                text=text,
                citations=citations,
                latency_s=round(time.perf_counter() - t0, 2),
                access=self.access,
                signed_in=self._signed_in(page),
                note=self._note(page, citations),
            )
        except Exception as exc:  # noqa: BLE001 - one failed question must not sink the scan
            return error_answer(self.name, self.model, question, 1, exc, access=self.access, signed_in=self.needs_login)
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        self._s.close()

    # -- hooks --

    def _after_submit(self, page, box, prompt: str) -> None:
        """Some surfaces show a consent dialog on the first message. Clear it; resend
        only if the question did not go through."""
        page.wait_for_timeout(2000)
        body = page.inner_text("body")
        self._guard_login_wall(page, body)
        if any(k in body for k in ("By clicking 'Continue'", "Stay logged out")):
            self._dismiss(page)
            page.wait_for_timeout(1500)
            body = page.inner_text("body")
            if body.count(prompt) < 2:  # only the echo in the box: not sent
                try:
                    box.click(timeout=3000)
                    box.press("Enter")
                except Exception:  # noqa: BLE001 - box went read-only because the message did go through
                    pass

    def _done(self, page, body: str) -> bool:
        """Surface-specific 'the answer is finished' signal. Default: no extra signal."""
        return True

    def _answer_text(self, page, prompt: str) -> str:
        if self.answer_selector:
            loc = page.locator(self.answer_selector)
            if loc.count():
                return loc.last.inner_text().strip()
        thread = page.evaluate(_DEEPEST_JS, [prompt, 300])
        return strip_question(thread, prompt, self.ui_lines)

    def _citations(self, page) -> list[Citation]:
        return clean_links(page.evaluate(_LINKS_JS), self.exclude_hosts)

    def _model_label(self, page) -> str:
        return self.model

    def _signed_in(self, page) -> bool:
        """True when the answer came from the user's own account. Engines that need a
        login are signed in by construction; others may detect it on the page."""
        return self.needs_login

    def _note(self, page, citations: list[Citation]) -> str:
        """A short observation for the report, e.g. that the surface showed no sources."""
        return ""

    # -- mechanics --

    def _say(self, msg: str) -> None:
        print(f"   {msg}", file=sys.stderr)

    def _dismiss(self, page) -> None:
        for label in _CONSENT_BUTTONS:
            try:
                b = page.get_by_role("button", name=re.compile(rf"^{re.escape(label)}$", re.I)).first
                if b.is_visible(timeout=600):
                    b.click(timeout=2000)
                    page.wait_for_timeout(500)
            except Exception:  # noqa: BLE001
                continue

    def _guard_login_wall(self, page, body: str | None = None) -> None:
        body = body if body is not None else page.inner_text("body")
        if not is_login_wall(body):
            return
        hint = f" Or `aivs login {self.login_site}` to use your own account." if self.login_site else ""
        raise LoginWall(
            f"{self.name} asked for a sign-in instead of answering (signed-out quota reached, or sign-in required). "
            f"Resume later with the same --scan-id; failed questions are retried.{hint}"
        )

    def _guard_bot_check(self, page) -> None:
        body = page.inner_text("body")
        if not is_bot_check(body):
            return
        if self._s.headless or self._settings.browser_human_wait_s <= 0:
            raise BotCheck(f"{self.name} showed a bot check. Run without --headless and clear it in the window.")
        self._say(f"{self.name}: bot check shown in the Chrome window. Clear it to continue "
                  f"(waiting up to {int(self._settings.browser_human_wait_s)}s).")
        t0 = time.monotonic()
        while time.monotonic() - t0 < self._settings.browser_human_wait_s:
            page.wait_for_timeout(2000)
            if not is_bot_check(page.inner_text("body")):
                return
        raise BotCheck(f"{self.name} bot check was not cleared in time")

    def _wait_for_answer(self, page, prompt: str) -> None:
        """Wait until the page stops changing for three ticks (spinners ignored), the
        surface says it is done, and a minimum time has passed. Bot checks mid-flow
        are handled the same way as on load."""
        t0 = time.monotonic()
        base = len(strip_volatile(page.inner_text("body")))
        last = ""
        stable = 0
        while time.monotonic() - t0 < self._settings.browser_answer_timeout_s:
            page.wait_for_timeout(2500)
            body = page.inner_text("body")
            if is_bot_check(body):
                self._guard_bot_check(page)
                continue
            self._guard_login_wall(page, body)
            cur = strip_volatile(body)
            if cur == last:
                stable += 1
            else:
                stable = 0
                last = cur
            elapsed = time.monotonic() - t0
            if stable >= 3 and elapsed >= self.min_wait_s and len(cur) > base + 150 and self._done(page, body):
                return
        raise TimeoutError(f"{self.name} did not finish answering within {int(self._settings.browser_answer_timeout_s)}s")


# ---------- the four surfaces ----------


class PerplexityWeb(BrowserEngine):
    name = "perplexity-web"
    model = "perplexity.ai · signed out · default model"
    url = "https://www.perplexity.ai/"
    input_selector = "[contenteditable=true][aria-label]"
    exclude_hosts = ("perplexity.ai",)
    login_site = "perplexity"  # signed out has an undocumented quota (13 of 24 in one sitting, 21 Nov 2026)
    ui_lines = (r"Answer", r"Links", r"Images", r"Share", r"Finished", r"Sources", r"\d+ sources", r"View all", r"View plans",
                r"Sign In", r"Sign in.*", r"Ask a follow-up.*", r"New", r"Automations", r"Artefacts", r"Plugins", r"Teammates",
                r"Hire your first teammate", r"Persistent agents that work for you\.", r"Create new teammate", r"Projects",
                r"No projects", r"Sessions", r"Search", r"Computer", r"Model", r"Search anything", r"Copy", r"Try again")
    _SOURCES_BTN = re.compile(r"^\s*\d+ sources?\s*$|^\s*Sources\s*\d+\s*$", re.I)

    def _done(self, page, body: str) -> bool:
        return "Finished" in body or bool(re.search(r"\d+ sources", body))

    def _signed_in(self, page) -> bool:
        try:
            return not page.get_by_role("button", name=re.compile(r"^Sign In$", re.I)).first.is_visible(timeout=800)
        except Exception:  # noqa: BLE001
            return False

    def _model_label(self, page) -> str:
        return "perplexity.ai · signed in (your account) · default model" if self._signed_in(page) else self.model

    def _answer_text(self, page, prompt: str) -> str:
        thread = page.evaluate(_DEEPEST_JS, [prompt, 300])
        text = strip_question(thread, prompt, self.ui_lines)
        # the thread ends with follow-up suggestions after the sources count; cut there
        m = re.search(r"\n\d+ sources\n", thread)
        if m:
            text = strip_question(thread[: m.start()], prompt, self.ui_lines)
        return text

    def _citations(self, page) -> list[Citation]:
        try:
            btn = page.get_by_role("button", name=self._SOURCES_BTN).first
            if btn.is_visible(timeout=1500):
                btn.click(timeout=3000)
                page.wait_for_timeout(2500)
        except Exception:  # noqa: BLE001 - inline citations still give us URLs
            pass
        return clean_links(page.evaluate(_LINKS_JS), self.exclude_hosts)


class GeminiWeb(BrowserEngine):
    """Gemini signed out sometimes answers from the model alone: no inline source
    chips, no footnotes, an empty <sources-list>, nothing to click (observed on 9 of
    24 Monzo questions on 21 Nov 2026 and reproduced in a probe). That is what the
    shopper sees, so it is recorded as an answer with no sources and a note, never
    retried or filled in from elsewhere."""

    name = "gemini-web"
    model = "gemini.google.com · signed out · Auto"
    url = "https://gemini.google.com/app"
    input_selector = "div.ql-editor[contenteditable=true]"
    answer_selector = "message-content .markdown"
    exclude_hosts = ("google.", "gstatic.com", "g.co/", "youtube.com/t/terms")
    _CHIPS = 'button[aria-label^="View source details"]'
    _GROUNDING = "source-inline-chip, source-footnote, sources-carousel-inline, sources-list a[href]"
    NO_SOURCES_NOTE = "Gemini showed no sources for this answer (no inline citations or footnotes); it answered without a visible web search."

    def _note(self, page, citations: list[Citation]) -> str:
        if citations:
            return ""
        try:
            grounded = page.locator(self._GROUNDING).count() > 0
        except Exception:  # noqa: BLE001
            grounded = False
        return "" if grounded else self.NO_SOURCES_NOTE

    def _done(self, page, body: str) -> bool:
        try:
            return page.locator('button[aria-label="Stop response"]').count() == 0
        except Exception:  # noqa: BLE001
            return True

    def _citations(self, page) -> list[Citation]:
        """Gemini keeps source URLs behind the inline chips. Click each chip and read
        the dialog; fall back to the chip labels (source names) when no URL appears."""
        chips = page.locator(self._CHIPS)
        n = min(chips.count(), 12)
        before = {p["h"] for p in page.evaluate(_LINKS_JS)}
        found: list[dict[str, Any]] = []
        labels: list[str] = []
        for i in range(n):
            try:
                aria = chips.nth(i).get_attribute("aria-label") or ""
                labels.extend(labels_from_aria(aria))
                chips.nth(i).click(timeout=3000)
                page.wait_for_timeout(1500)
                found.extend(p for p in page.evaluate(_LINKS_JS) if p["h"] not in before)
                page.keyboard.press("Escape")
                page.wait_for_timeout(400)
            except Exception:  # noqa: BLE001
                continue
        cites = clean_links(found, self.exclude_hosts)
        if cites:
            return cites
        seen: set[str] = set()
        out: list[Citation] = []
        for lab in labels:
            if lab.lower() in seen:
                continue
            seen.add(lab.lower())
            out.append(Citation(url="", title=lab))
        return out


class BraveWeb(BrowserEngine):
    name = "brave-web"
    model = "search.brave.com/ask · signed out"
    url = "https://search.brave.com/ask"
    input_selector = "#tap-input-field"
    answer_selector = "div.message.assistant.llm-output"
    exclude_hosts = ("brave.com",)
    headless_ok = False

    def _done(self, page, body: str) -> bool:
        try:
            return page.locator("div.tap-round.is-complete").count() > 0
        except Exception:  # noqa: BLE001
            return True

    def _citations(self, page) -> list[Citation]:
        try:
            btn = page.get_by_role("button", name=re.compile(r"^View all", re.I)).first
            if btn.is_visible(timeout=1500):
                btn.click(timeout=3000)
                page.wait_for_timeout(2500)
        except Exception:  # noqa: BLE001
            pass
        return clean_links(page.evaluate(_LINKS_JS), self.exclude_hosts)


class DuckWeb(BrowserEngine):
    name = "duckai-web"
    model = "duck.ai · signed out · default model"
    url = "https://duck.ai/"
    input_selector = "textarea[name=user-prompt]"
    exclude_hosts = ("duckduckgo.com", "duck.ai")
    ui_lines = (r"Duck\.ai said", r"You said", r"Copy to clipboard", r"Read More", r"Related Searches", r"2nd Opinion",
                r"Duck\.ai works best in the private and free DuckDuckGo browser!", r"Download App", r"Learn More",
                r"Jump to latest response", r"Tools", r"Fast", r"All chats are private\. AI can make mistakes\.")
    headless_ok = False
    _MODEL_JS = """() => { const t = document.body.innerText; const m = t.match(/Duck\\.ai said\\s*\\n\\s*([^\\n]{2,40})\\n/); return m ? m[1].trim() : ''; }"""

    def _done(self, page, body: str) -> bool:
        return "Generating response" not in body and "Stop generating" not in body

    def _answer_text(self, page, prompt: str) -> str:
        thread = page.evaluate(_DEEPEST_JS, ["Duck.ai said", 200])
        text = strip_question(thread, prompt, self.ui_lines)
        model = page.evaluate(self._MODEL_JS)
        if model:
            text = re.sub(rf"^\s*{re.escape(model)}\s*\n", "", text)
        return text.strip()

    def _model_label(self, page) -> str:
        model = page.evaluate(self._MODEL_JS)
        return f"duck.ai · signed out · {model}" if model else self.model


class GrokWeb(BrowserEngine):
    """Grok signed out shows a sign-up card instead of an answer, so this engine
    runs on YOUR login: `aivs login grok` once, then the persistent profile keeps
    the session. Labelled 'signed in' in the report because memory and
    personalisation can shape the answer. Opt-in: not part of `--engines web`."""

    name = "grok-web"
    model = "grok.com · signed in (your account)"
    url = "https://grok.com/"
    input_selector = "div.tiptap[contenteditable=true]"
    exclude_hosts = ("grok.com", "x.ai", "cookiepedia.co.uk", "onetrust.com")
    ui_lines = (r"Skip to main content", r"You", r"Grok", r"Copy", r"Regenerate", r"Share", r"Sign in", r"Sign up.*")
    headless_ok = False
    needs_login = True
    login_site = "grok"

    def _after_submit(self, page, box, prompt: str) -> None:
        page.wait_for_timeout(2500)
        if page.locator('[data-testid="anon-paywall-sign-up-card"]').count():
            raise LoginWall("grok-web is not signed in: Grok shows a sign-up card instead of an answer. Run `aivs login grok` once, then retry.")
        self._guard_login_wall(page)

    def _done(self, page, body: str) -> bool:
        try:
            return page.locator('button[aria-label="Stop model response"], button[aria-label="Stop"]').count() == 0
        except Exception:  # noqa: BLE001
            return True


BROWSER_ENGINES: dict[str, type[BrowserEngine]] = {
    PerplexityWeb.name: PerplexityWeb,
    GeminiWeb.name: GeminiWeb,
    BraveWeb.name: BraveWeb,
    DuckWeb.name: DuckWeb,
    GrokWeb.name: GrokWeb,
}

LOGIN_SITES: dict[str, str] = {"grok": GrokWeb.url, "perplexity": PerplexityWeb.url}
