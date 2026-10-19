"""Connector to the Hybrid Retrieval Lab's scoped knowledge API (no local index)."""

from .client import (
    Evidence,
    KnowledgeClient,
    KnowledgeError,
    KnowledgeForbidden,
    KnowledgeNotConfigured,
    KnowledgeNotFound,
    KnowledgeUnauthorized,
    KnowledgeUnavailable,
    SearchResult,
    SourcePage,
)
from .connector import DEFAULT_CREDENTIAL_REF, KnowledgeConnector
from .ingestion import (
    DEFAULT_INGEST_CREDENTIAL_REF,
    IngestClient,
    IngestionService,
    SubmissionStateError,
)

__all__ = [
    "DEFAULT_CREDENTIAL_REF",
    "DEFAULT_INGEST_CREDENTIAL_REF",
    "Evidence",
    "IngestClient",
    "IngestionService",
    "KnowledgeClient",
    "KnowledgeConnector",
    "KnowledgeError",
    "KnowledgeForbidden",
    "KnowledgeNotConfigured",
    "KnowledgeNotFound",
    "KnowledgeUnauthorized",
    "KnowledgeUnavailable",
    "SearchResult",
    "SourcePage",
    "SubmissionStateError",
]
