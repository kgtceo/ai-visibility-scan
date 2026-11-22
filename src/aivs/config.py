"""Settings from the environment and the brand brief from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .models import Brief

load_dotenv()


@dataclass
class Settings:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    perplexity_api_key: str | None = None
    xai_api_key: str | None = None

    # The judge reads every answer; keep it one strong, cheap-enough model.
    judge_model: str = "claude-sonnet-5-5"
    judge_workers: int = 6  # parallel judge calls during scoring (AIVS_JUDGE_WORKERS)
    # Engine models. Each is the API-backed stand-in for a consumer answer engine.
    claude_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-6.2-astra"
    gemini_model: str = "gemini-3.8-flash"  # 2.5-flash is closed to new accounts (404 on 21 Nov 2026); Google names 3.8-flash
    perplexity_model: str = "perplexity/sonar"
    xai_model: str = "grok-4.7"

    # Browser engines (no key): one persistent Chrome profile, human pace, visible window by default
    browser_headless: bool = False
    browser_channel: str = "chrome"
    browser_profile_dir: Path = field(default_factory=lambda: Path.home() / ".aivs" / "chrome-profile")
    browser_locale: str = "en-GB"
    browser_timezone: str = "Europe/London"
    browser_pace_s: float = 4.0
    browser_settle_ms: int = 3500
    browser_answer_timeout_s: float = 120.0
    browser_human_wait_s: float = 120.0

    max_tokens: int = 1800
    max_searches_per_answer: int = 5
    request_timeout_s: float = 120.0
    runs_dir: Path = field(default_factory=lambda: Path("runs"))

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
            xai_api_key=os.getenv("XAI_API_KEY"),
            judge_model=os.getenv("AIVS_JUDGE_MODEL", cls.judge_model),
            judge_workers=int(os.getenv("AIVS_JUDGE_WORKERS", cls.judge_workers)),
            claude_model=os.getenv("AIVS_CLAUDE_MODEL", cls.claude_model),
            openai_model=os.getenv("AIVS_OPENAI_MODEL", cls.openai_model),
            gemini_model=os.getenv("AIVS_GEMINI_MODEL", cls.gemini_model),
            perplexity_model=os.getenv("AIVS_PERPLEXITY_MODEL", cls.perplexity_model),
            xai_model=os.getenv("AIVS_XAI_MODEL", cls.xai_model),
            runs_dir=Path(os.getenv("AIVS_RUNS_DIR", "runs")),
            browser_headless=os.getenv("AIVS_BROWSER_HEADLESS", "").lower() in ("1", "true", "yes"),
            browser_channel=os.getenv("AIVS_BROWSER_CHANNEL", cls.browser_channel),
            browser_profile_dir=Path(os.getenv("AIVS_BROWSER_PROFILE", str(Path.home() / ".aivs" / "chrome-profile"))),
            browser_pace_s=float(os.getenv("AIVS_BROWSER_PACE_S", cls.browser_pace_s)),
        )

    @staticmethod
    def keyless_engines() -> list[str]:
        """Consumer surfaces read through Chrome signed out. No key, no login."""
        return ["perplexity-web", "gemini-web", "brave-web", "duckai-web"]

    @staticmethod
    def login_engines() -> list[str]:
        """Surfaces that answer only when signed in with your own account. Opt-in: `aivs login <site>` first."""
        return ["grok-web"]

    def keyed_engines(self) -> list[str]:
        out = []
        if self.anthropic_api_key:
            out.append("claude")
        if self.openai_api_key:
            out.append("chatgpt")
        if self.gemini_api_key:
            out.append("gemini")
        if self.perplexity_api_key:
            out.append("perplexity")
        if self.xai_api_key:
            out.append("grok")
        return out

    def available_engines(self) -> list[str]:
        """Everything runnable now: the keyless browser engines plus any API engine with a key."""
        return self.keyless_engines() + self.keyed_engines()


def load_brief(path: str | Path) -> Brief:
    data = yaml.safe_load(Path(path).read_text())
    return Brief.model_validate(data)
