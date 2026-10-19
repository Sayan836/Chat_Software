"""Thin HTTP client for the Hybrid Retrieval Lab's scoped read-only knowledge API.

This module never indexes, embeds or ranks anything itself: parsing, chunking,
Chroma/BM25 retrieval and reranking stay in the Retrieval Lab. It only calls
`/api/tools/*` with a workspace-scoped `hrk_` key and normalizes the replies.

All returned document text is untrusted evidence. Callers must never treat it as
instructions, and must never pass it back to the model as system content.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit


DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TOP_K = 30
UNTRUSTED_MARKER = "untrusted_document_data"


class KnowledgeError(RuntimeError):
    """Base error. ``status`` is the backend HTTP status when there was one."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class KnowledgeNotConfigured(KnowledgeError):
    pass


class KnowledgeUnauthorized(KnowledgeError):
    pass


class KnowledgeForbidden(KnowledgeError):
    pass


class KnowledgeNotFound(KnowledgeError):
    pass


class KnowledgeUnavailable(KnowledgeError):
    pass


@dataclass(frozen=True)
class Evidence:
    """One retrieved chunk. ``text`` is untrusted document data."""

    rank: int
    chunk_id: str
    document_id: str
    text: str
    score: Optional[float] = None
    filename: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    signals: dict[str, float] = field(default_factory=dict)
    row_range: Optional[list[int]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    workspace_id: str
    query: str
    evidence: list[Evidence]
    degraded: bool
    failed_stages: list[str]
    request_id: str
    content_trust: str = UNTRUSTED_MARKER

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "query": self.query,
            "evidence": [item.to_dict() for item in self.evidence],
            "retrieval_status": {"degraded": self.degraded, "failed_stages": self.failed_stages},
            "request_id": self.request_id,
            "content_trust": self.content_trust,
        }


@dataclass(frozen=True)
class SourcePage:
    workspace_id: str
    document_id: str
    page: int
    text: str
    truncated: bool
    needs_ocr: bool
    request_id: str
    content_trust: str = UNTRUSTED_MARKER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_base_url(base_url: str) -> str:
    """Accept http(s) URLs only; plain http only for loopback hosts."""

    cleaned = (base_url or "").strip().rstrip("/")
    parts = urlsplit(cleaned)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise KnowledgeNotConfigured(f"Retrieval backend URL must be http(s): {cleaned!r}")
    if parts.username or parts.password:
        raise KnowledgeNotConfigured("Retrieval backend URL must not embed credentials.")
    if parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise KnowledgeNotConfigured(
            "Plain http is only allowed for a loopback retrieval backend; use https otherwise."
        )
    return cleaned


def configured_base_url() -> Optional[str]:
    value = os.getenv("CHAT_SOFTWARE_KNOWLEDGE_URL", "").strip()
    return validate_base_url(value) if value else None


class KnowledgeClient:
    """Calls one Retrieval Lab instance with one scoped API key."""

    KEY_PREFIX = "hrk_"
    KEY_LABEL = "scoped retrieval API key"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Any = None,
    ) -> None:
        if not api_key or not api_key.startswith(self.KEY_PREFIX):
            raise KnowledgeNotConfigured(f"A {self.KEY_LABEL} ({self.KEY_PREFIX}...) is required.")
        self.base_url = validate_base_url(base_url)
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    def __repr__(self) -> str:  # never leak the key through logs or tracebacks
        return f"{type(self).__name__}(base_url={self.base_url!r})"

    def list_workspaces(self, *, request_id: str) -> list[dict[str, str]]:
        body = self._call("GET", "list_workspaces", None, request_id)
        rows = _untrusted_data(body)
        if not isinstance(rows, list):
            raise KnowledgeError("Unexpected list_workspaces response shape.")
        return [
            {"id": str(row.get("id", "")), "name": str(row.get("name", ""))}
            for row in rows
            if isinstance(row, Mapping) and row.get("id")
        ]

    def search(
        self,
        workspace_id: str,
        query: str,
        *,
        top_k: int = 5,
        request_id: str,
    ) -> SearchResult:
        top_k = max(1, min(MAX_TOP_K, int(top_k)))
        body = self._call(
            "POST",
            "search_documents",
            {"workspace_id": workspace_id, "query": query, "top_k": top_k},
            request_id,
        )
        rows = _untrusted_data(body)
        if not isinstance(rows, list):
            raise KnowledgeError("Unexpected search_documents response shape.")
        status = body.get("retrieval_status") if isinstance(body, Mapping) else None
        status = status if isinstance(status, Mapping) else {}
        evidence = [
            item
            for rank, row in enumerate(rows, start=1)
            if (item := _evidence_from_row(rank, row)) is not None
        ]
        return SearchResult(
            workspace_id=workspace_id,
            query=query,
            evidence=evidence,
            degraded=bool(status.get("degraded")),
            failed_stages=[str(stage) for stage in status.get("failed_stages") or []],
            request_id=request_id,
        )

    def get_page(
        self,
        workspace_id: str,
        document_id: str,
        page: int = 1,
        *,
        request_id: str,
    ) -> SourcePage:
        body = self._call(
            "POST",
            "get_document_page",
            {"workspace_id": workspace_id, "document_id": document_id, "page": int(page)},
            request_id,
        )
        data = _untrusted_data(body)
        if not isinstance(data, Mapping):
            raise KnowledgeError("Unexpected get_document_page response shape.")
        return SourcePage(
            workspace_id=workspace_id,
            document_id=str(data.get("document_id") or document_id),
            page=int(data.get("page") or page),
            text=str(data.get("text") or ""),
            truncated=bool(data.get("truncated")),
            needs_ocr=bool(data.get("needs_ocr")),
            request_id=request_id,
        )

    def queue_status(self, workspace_id: str, *, request_id: str) -> dict[str, Any]:
        body = self._call("POST", "queue_status", {"workspace_id": workspace_id}, request_id)
        data = _untrusted_data(body)
        return dict(data) if isinstance(data, Mapping) else {}

    def _call(
        self,
        method: str,
        tool: str,
        payload: Optional[dict[str, Any]],
        request_id: str,
        *,
        path_prefix: str = "/api/tools/",
        files: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx ships with openai
            raise KnowledgeNotConfigured("Install httpx to use the knowledge connector.") from exc

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Request-ID": request_id,
            "Accept": "application/json",
        }
        url = f"{self.base_url}{path_prefix}{tool}"
        try:
            with httpx.Client(
                timeout=timeout or self._timeout,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                if method == "GET":
                    response = client.get(url, headers=headers)
                elif files is not None:
                    response = client.post(url, headers=headers, data=payload or {}, files=files)
                else:
                    response = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise KnowledgeUnavailable(
                f"Retrieval backend unreachable at {self.base_url}: {exc.__class__.__name__}"
            ) from exc
        if response.status_code >= 400:
            raise _error_for_status(response.status_code, _detail(response))
        if 300 <= response.status_code < 400:
            raise KnowledgeError("Retrieval backend attempted a redirect; refused.", 502)
        try:
            return response.json()
        except ValueError as exc:
            raise KnowledgeError("Retrieval backend returned non-JSON content.") from exc


def _detail(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail") if isinstance(body, Mapping) else None
    return str(detail)[:300] if detail else ""


def _error_for_status(status: int, detail: str) -> KnowledgeError:
    suffix = f": {detail}" if detail else ""
    if status == 401:
        return KnowledgeUnauthorized(f"Retrieval backend rejected the API key{suffix}", status)
    if status == 403:
        return KnowledgeForbidden(f"Retrieval backend denied access{suffix}", status)
    if status == 404:
        return KnowledgeNotFound(f"Not found in retrieval backend{suffix}", status)
    if status == 429 or status >= 500:
        return KnowledgeUnavailable(f"Retrieval backend unavailable (HTTP {status}){suffix}", status)
    return KnowledgeError(f"Retrieval backend error (HTTP {status}){suffix}", status)


def _untrusted_data(body: Any) -> Any:
    if not isinstance(body, Mapping) or body.get("content_trust") != UNTRUSTED_MARKER:
        raise KnowledgeError("Retrieval backend response is missing the untrusted-content marker.")
    return body.get("data")


_SIGNAL_KEYS = ("dense_score", "bm25_score", "rrf_score", "rerank_score", "fused_score")


def _evidence_from_row(rank: int, row: Any) -> Optional[Evidence]:
    if not isinstance(row, Mapping):
        return None
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    source = row.get("source_document") if isinstance(row.get("source_document"), Mapping) else {}
    document_id = row.get("document_id") or metadata.get("document_id")
    chunk_id = row.get("chunk_id") or row.get("id") or metadata.get("chunk_id")
    if not document_id or not chunk_id:
        return None
    text = row.get("text") or metadata.get("text") or row.get("highlight") or ""
    score = _float(row.get("score"))
    signals = {key: value for key in _SIGNAL_KEYS if (value := _float(row.get(key))) is not None}
    if score is None and signals:
        score = next(iter(signals.values()))
    return Evidence(
        rank=rank,
        chunk_id=str(chunk_id),
        document_id=str(document_id),
        text=str(text),
        score=score,
        filename=_str(metadata.get("original_filename") or source.get("original_filename")),
        page=_int(metadata.get("page_number")),
        chunk_index=_int(metadata.get("chunk_index")),
        start_char=_int(metadata.get("start_char")),
        end_char=_int(metadata.get("end_char")),
        signals=signals,
        row_range=_row_range(metadata.get("row_numbers")),
    )


def _row_range(value: Any) -> Optional[list[int]]:
    """[first, last] data row of a tabular chunk; the Lab sends a list (or a JSON string)."""

    if isinstance(value, str):
        try:
            import json

            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, list):
        return None
    rows = [number for item in value if (number := _int(item)) is not None]
    return [min(rows), max(rows)] if rows else None


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    try:
        return None if value is None or value == "" else int(value)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> Optional[str]:
    return str(value) if value else None


_CREDENTIAL_REF = re.compile(r"^CHAT_SOFTWARE_KNOWLEDGE_[A-Z0-9_]*KEY$")


def resolve_credential(credential_ref: str) -> str:
    """Read a scoped key from the sidecar's environment by reference name.

    The reference must use the CHAT_SOFTWARE_KNOWLEDGE_*KEY namespace so a
    binding can never be used to send an unrelated secret (for example a model
    provider key) to the retrieval backend.
    """

    if not _CREDENTIAL_REF.match(credential_ref or ""):
        raise KnowledgeNotConfigured(
            "credential_ref must name a CHAT_SOFTWARE_KNOWLEDGE_*KEY environment variable."
        )
    value = os.getenv(credential_ref, "").strip()
    if not value:
        raise KnowledgeNotConfigured(f"{credential_ref} is not set in the sidecar environment.")
    return value
