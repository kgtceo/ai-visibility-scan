"""Deterministic source classification. No model involved.

Every cited URL is mapped to a domain and a source type. The brief supplies the
brand-owned and competitor-owned domains; the rest come from a category list in the
brief (review and comparison sites, forums, news, regulators) with a generic
fallback. A rule may be a host ("reddit.com", matching subdomains too) or a host
plus path prefix ("moneysavingexpert.com/forums"); path rules are checked first so
a forum section of a review site counts as a forum. Gemini hides sources behind
redirect URLs, so when the host is a known redirector we classify on the citation
title instead (it carries the domain), the same way an empty URL falls back.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import Brief, Citation, ClassifiedSource, SourceType

_REDIRECT_HOSTS = ("vertexaisearch.cloud.google.com", "google.com/url")
_DOMAIN_IN_TITLE = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,})\b", re.I)


def domain_of(url: str, title: str = "") -> str:
    """Host of the URL. When a surface hides the URL (Gemini chips, redirects) we fall
    back to a domain in the title, then to the source's display name."""
    host = (urlparse(url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not url:
        m = _DOMAIN_IN_TITLE.search(title or "")
        return (m.group(1).lower().removeprefix("www.") if m else (title or "").strip().lower()) or "unknown"
    if any(r in url for r in _REDIRECT_HOSTS):
        # The redirect host says nothing about the source; the title does, or nothing does.
        m = _DOMAIN_IN_TITLE.search(title or "")
        return (m.group(1).lower().removeprefix("www.") if m else (title or "").strip().lower()) or "unknown"
    return host


def _host_matches(host: str, rule_host: str) -> bool:
    return host == rule_host or host.endswith("." + rule_host)


def _matches(host: str, path: str, rules: list[str], with_path: bool) -> bool:
    """with_path=True checks only 'host/path-prefix' rules, False only bare host rules."""
    for rule in (r.lower().strip() for r in rules):
        rule_host, _, rule_path = rule.partition("/")
        if bool(rule_path) != with_path:
            continue
        if not _host_matches(host, rule_host):
            continue
        if not rule_path or path.startswith("/" + rule_path.rstrip("/")):
            return True
    return False


def classify(url: str, title: str, brief: Brief) -> ClassifiedSource:
    host = domain_of(url, title)
    path = (urlparse(url).path or "/").lower().rstrip("/") + "/"
    stype: SourceType = "other"
    if _matches(host, path, brief.client.domains, False):
        stype = "brand_owned"
    elif any(_matches(host, path, c.domains, False) for c in brief.competitors):
        stype = "competitor_owned"
    elif host.endswith("wikipedia.org"):
        stype = "wikipedia"
    else:
        for with_path in (True, False):  # a path rule beats a bare-host rule of another type
            hit = next((name for name, rules in brief.source_types.items() if _matches(host, path, rules, with_path)), None)
            if hit:
                stype = hit  # type: ignore[assignment]
                break
    return ClassifiedSource(url=url, domain=host, source_type=stype, title=title)


def classify_all(citations: list[Citation], brief: Brief) -> list[ClassifiedSource]:
    return [classify(c.url, c.title, brief) for c in citations]
