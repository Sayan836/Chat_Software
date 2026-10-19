"""Web search providers and a guarded page fetcher for research and tools.

Search uses an explicitly configured provider (no scraping of search engines):
``searxng`` (self-hosted, open source; ``SEARXNG_URL``), ``brave`` (``BRAVE_SEARCH_API_KEY``) or
``tavily`` (``TAVILY_API_KEY``). ``RESEARCH_SEARCH_PROVIDER`` picks one; otherwise the first
configured is used.

``fetch_url`` only fetches public http(s) addresses: every hop (including redirects) is resolved and
refused if it points at loopback, private, link-local, multicast or reserved space. Responses are
size-capped and reduced to readable text. All returned text is untrusted page content.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable, Optional
from urllib.parse import urljoin, urlsplit

import httpx

from chat_software import secrets_store


MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 60_000
USER_AGENT = "ChatSoftware-Research/0.1 (+local assistant)"
PROVIDERS = ("searxng", "brave", "tavily")


class WebError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "source": self.source}


def configured_provider() -> Optional[str]:
    chosen = (secrets_store.get("RESEARCH_SEARCH_PROVIDER") or "").strip().lower()
    available = {
        "searxng": bool(secrets_store.get("SEARXNG_URL")),
        "brave": bool(secrets_store.get("BRAVE_SEARCH_API_KEY")),
        "tavily": bool(secrets_store.get("TAVILY_API_KEY")),
    }
    if chosen in available:
        return chosen if available[chosen] else None
    return next((name for name in PROVIDERS if available[name]), None)


def web_search(query: str, max_results: int = 6, *, transport: Any = None) -> list[SearchHit]:
    provider = configured_provider()
    if provider is None:
        raise WebError("No web search provider is configured (SearXNG URL, Brave or Tavily key).")
    query = " ".join(query.split())[:400]
    max_results = max(1, min(int(max_results), 10))
    with httpx.Client(timeout=20, transport=transport, headers={"User-Agent": USER_AGENT}) as client:
        try:
            if provider == "searxng":
                base = (secrets_store.get("SEARXNG_URL") or "").rstrip("/")
                response = client.get(f"{base}/search", params={"q": query, "format": "json"})
                response.raise_for_status()
                rows = response.json().get("results", [])
                hits = [(r.get("title"), r.get("url"), r.get("content")) for r in rows]
            elif provider == "brave":
                response = client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={"X-Subscription-Token": secrets_store.get("BRAVE_SEARCH_API_KEY") or "",
                             "Accept": "application/json"},
                )
                response.raise_for_status()
                rows = (response.json().get("web") or {}).get("results", [])
                hits = [(r.get("title"), r.get("url"), r.get("description")) for r in rows]
            else:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "max_results": max_results},
                    headers={"Authorization": f"Bearer {secrets_store.get('TAVILY_API_KEY') or ''}"},
                )
                response.raise_for_status()
                rows = response.json().get("results", [])
                hits = [(r.get("title"), r.get("url"), r.get("content")) for r in rows]
        except httpx.HTTPStatusError as exc:
            raise WebError(f"{provider} search failed (HTTP {exc.response.status_code})") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise WebError(f"{provider} search failed ({exc.__class__.__name__})") from exc
    out = []
    for title, url, snippet in hits:
        if url and str(url).startswith(("http://", "https://")):
            out.append(SearchHit(_clean(title)[:200] or url, str(url), _clean(snippet)[:500], provider))
        if len(out) >= max_results:
            break
    return out


Resolver = Callable[[str], list[str]]


def _default_resolver(host: str) -> list[str]:
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except socket.gaierror as exc:
        raise WebError(f"Could not resolve {host}") from exc


def check_public_url(url: str, resolver: Resolver = _default_resolver) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise WebError("Only http(s) URLs can be fetched.")
    if parts.username or parts.password:
        raise WebError("URLs with embedded credentials are refused.")
    host = parts.hostname
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        if host.lower() in {"localhost"} or host.lower().endswith((".local", ".internal", ".localhost")):
            raise WebError("Local network addresses are refused.")
        addresses = resolver(host)
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified or getattr(ip, "is_site_local", False)):
            raise WebError("Private, loopback and link-local addresses are refused.")


def fetch_url(url: str, *, transport: Any = None, resolver: Resolver = _default_resolver) -> dict[str, Any]:
    current = url.strip()
    with httpx.Client(timeout=20, transport=transport, follow_redirects=False,
                      headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/json"}) as client:
        for _hop in range(4):
            check_public_url(current, resolver)
            try:
                with client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308} and response.headers.get("location"):
                        current = urljoin(current, response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        raise WebError(f"Fetch failed (HTTP {response.status_code})")
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if content_type and not (content_type.startswith("text/") or content_type in {
                            "application/json", "application/xhtml+xml", "application/xml"}):
                        raise WebError(f"Unsupported content type {content_type}")
                    body = b""
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) > MAX_FETCH_BYTES:
                            break
                    text = body[:MAX_FETCH_BYTES].decode(response.encoding or "utf-8", errors="replace")
            except httpx.HTTPError as exc:
                raise WebError(f"Fetch failed ({exc.__class__.__name__})") from exc
            title, readable = html_to_text(text) if "html" in content_type or text.lstrip().startswith("<") else ("", text)
            return {
                "url": current,
                "title": title or current,
                "text": readable[:MAX_TEXT_CHARS],
                "truncated": len(readable) > MAX_TEXT_CHARS or len(body) > MAX_FETCH_BYTES,
                "content_trust": "untrusted_web_content",
            }
    raise WebError("Too many redirects.")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"}
    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> tuple[str, str]:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
    except Exception:
        return "", _clean(re.sub(r"<[^>]+>", " ", markup))
    text = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return _clean(parser.title), "\n".join(line for line in lines if line)


def _clean(value: Any) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", str(value or "")))
    return " ".join(text.split())
