"""Root-scoped knowledge connector.

Joins Chat_Software's permission system (approved roots in ``SafetyStore``) to the
Retrieval Lab's workspace-scoped API. The contract, from the migration map:

* A root is bound to exactly one backend workspace; callers address roots, never
  raw workspace ids, so a request cannot reach a corpus the root is not bound to.
* Root approval and binding are rechecked before every search and source open.
  Revoking the root (or the binding) stops access immediately.
* The API key stays in the sidecar environment; responses never include it.
* Each call gets a correlation id that is sent to the backend and audited here.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from chat_software.safety_store import SafetyStore

from .client import (
    KnowledgeClient,
    KnowledgeForbidden,
    KnowledgeNotConfigured,
    SearchResult,
    SourcePage,
    configured_base_url,
    resolve_credential,
)


DEFAULT_CREDENTIAL_REF = "CHAT_SOFTWARE_KNOWLEDGE_API_KEY"

ClientFactory = Callable[[str, str], KnowledgeClient]


class KnowledgeConnector:
    def __init__(
        self,
        safety: SafetyStore,
        *,
        base_url: Optional[str] = None,
        client_factory: Optional[ClientFactory] = None,
    ) -> None:
        self.safety = safety
        self._base_url = base_url
        self._client_factory = client_factory or (lambda url, key: KnowledgeClient(url, key))

    @property
    def base_url(self) -> Optional[str]:
        return self._base_url if self._base_url is not None else configured_base_url()

    def status(self) -> dict[str, Any]:
        try:
            url = self.base_url
            error = None
        except KnowledgeNotConfigured as exc:
            url, error = None, str(exc)
        return {
            "configured": bool(url),
            "base_url": url,
            "error": error,
            "bindings": len(self.safety.list_knowledge_bindings()),
        }

    def bind(
        self,
        root_id: str,
        workspace_id: str,
        credential_ref: str = DEFAULT_CREDENTIAL_REF,
    ) -> dict[str, Any]:
        """Bind an approved root to a workspace the scoped key can actually see."""

        self.safety.get_approved_root(root_id)
        request_id = _request_id()
        client = self._client(credential_ref)
        visible = {row["id"]: row for row in client.list_workspaces(request_id=request_id)}
        if workspace_id not in visible:
            self._audit("knowledge.bind_denied", root_id, request_id, workspace_id=workspace_id)
            raise KnowledgeForbidden(
                "Workspace is not visible to this retrieval API key; it cannot be bound.", 403
            )
        binding = self.safety.bind_knowledge(root_id, workspace_id, credential_ref)
        return public_binding(binding, workspace_name=visible[workspace_id].get("name"))

    def unbind(self, root_id: str) -> None:
        self.safety.unbind_knowledge(root_id)

    def bindings(self) -> list[dict[str, Any]]:
        return [public_binding(binding) for binding in self.safety.list_knowledge_bindings()]

    def search(
        self,
        root_id: str,
        query: str,
        *,
        top_k: int = 5,
        request_id: Optional[str] = None,
    ) -> SearchResult:
        binding = self.safety.get_knowledge_binding(root_id)
        request_id = request_id or _request_id()
        client = self._client(binding["credential_ref"])
        self._audit(
            "knowledge.search",
            root_id,
            request_id,
            workspace_id=binding["workspace_id"],
            top_k=top_k,
            query_chars=len(query),
        )
        return client.search(binding["workspace_id"], query, top_k=top_k, request_id=request_id)

    def open_source(
        self,
        root_id: str,
        document_id: str,
        page: int = 1,
        *,
        request_id: Optional[str] = None,
    ) -> SourcePage:
        binding = self.safety.get_knowledge_binding(root_id)
        request_id = request_id or _request_id()
        client = self._client(binding["credential_ref"])
        self._audit(
            "knowledge.open_source",
            root_id,
            request_id,
            workspace_id=binding["workspace_id"],
            document_id=document_id,
            page=page,
        )
        return client.get_page(binding["workspace_id"], document_id, page, request_id=request_id)

    def _client(self, credential_ref: str) -> KnowledgeClient:
        url = self.base_url
        if not url:
            raise KnowledgeNotConfigured("Set CHAT_SOFTWARE_KNOWLEDGE_URL to enable document search.")
        return self._client_factory(url, resolve_credential(credential_ref))

    def _audit(self, event_type: str, root_id: str, request_id: str, **metadata: Any) -> None:
        self.safety.record_audit(
            event_type,
            root_id=root_id,
            metadata={"request_id": request_id, **metadata},
        )


def public_binding(binding: dict[str, Any], workspace_name: Optional[str] = None) -> dict[str, Any]:
    """UI-safe view: the credential is identified by reference name only."""

    payload = {
        "root_id": binding["root_id"],
        "root_name": binding.get("root_name"),
        "root_path": binding.get("root_path"),
        "workspace_id": binding["workspace_id"],
        "credential_ref": binding["credential_ref"],
        "ingest_enabled": bool(binding.get("ingest_credential_ref")),
        "ingest_credential_ref": binding.get("ingest_credential_ref"),
        "updated_at": binding.get("updated_at"),
    }
    if workspace_name is not None:
        payload["workspace_name"] = workspace_name
    return payload


def _request_id() -> str:
    return str(uuid.uuid4())
