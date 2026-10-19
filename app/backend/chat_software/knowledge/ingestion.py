"""Approved selected-file ingestion into a root's bound retrieval workspace (M3).

Two explicit steps, like organize preview/apply:

1. ``preview`` resolves each selected path inside the approved root, checks type and size
   against the backend's limits, fingerprints the file (SHA-256) and records a submission that
   names exactly what will be sent and where. Nothing leaves the machine.
2. ``approve`` rechecks the root, the binding, the ingest key's workspace and every fingerprint,
   then uploads only unchanged files with a write-scoped ``hik_`` key bound to that one workspace.

Local-root approval is not a blanket grant to upload: only files listed in an approved
submission are sent. Status polling also rechecks the root, so revocation stops it. Extraction,
OCR, chunking, embedding and indexing all run in the Retrieval Lab; nothing is indexed here.
The backend's job queue cannot cancel, so neither can this module.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from chat_software.safety_store import SafetyStore
from chat_software.tools.scope import ScopedPathResolver

from .client import (
    KnowledgeClient,
    KnowledgeError,
    KnowledgeForbidden,
    KnowledgeNotConfigured,
    configured_base_url,
    resolve_credential,
)


DEFAULT_INGEST_CREDENTIAL_REF = "CHAT_SOFTWARE_KNOWLEDGE_INGEST_KEY"
MAX_FILES_PER_SUBMISSION = 50
UPLOAD_TIMEOUT_SECONDS = 600.0
TERMINAL_FILE_STATES = {"completed", "failed", "changed", "rejected", "error"}


class SubmissionStateError(ValueError):
    """The submission is not in a state that allows this action (HTTP 409)."""


class IngestClient(KnowledgeClient):
    """Write-scoped client: may only submit files and read their status (``/api/ingest/*``)."""

    KEY_PREFIX = "hik_"
    KEY_LABEL = "workspace ingest key"

    def whoami(self, *, request_id: str) -> dict[str, Any]:
        body = self._call("GET", "whoami", None, request_id, path_prefix="/api/ingest/")
        if not isinstance(body, dict) or not body.get("workspace_id"):
            raise KnowledgeError("Unexpected ingest whoami response shape.")
        return body

    def submit_file(
        self,
        path: Path,
        *,
        source_ref: str,
        request_id: str,
    ) -> dict[str, Any]:
        with path.open("rb") as handle:
            body = self._call(
                "POST",
                "files",
                {"source_ref": source_ref},
                request_id,
                path_prefix="/api/ingest/",
                files={"file": (path.name, handle)},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        if not isinstance(body, dict) or not body.get("asset_id"):
            raise KnowledgeError("Unexpected ingest submit response shape.")
        return body

    def job_status(self, asset_id: str, *, request_id: str) -> dict[str, Any]:
        body = self._call("GET", f"jobs/{asset_id}", None, request_id, path_prefix="/api/ingest/")
        if not isinstance(body, dict):
            raise KnowledgeError("Unexpected ingest status response shape.")
        return body


IngestClientFactory = Callable[[str, str], IngestClient]


class IngestionService:
    def __init__(
        self,
        safety: SafetyStore,
        *,
        base_url: Optional[str] = None,
        client_factory: Optional[IngestClientFactory] = None,
    ) -> None:
        self.safety = safety
        self._base_url = base_url
        self._client_factory = client_factory or (lambda url, key: IngestClient(url, key))

    @property
    def base_url(self) -> Optional[str]:
        return self._base_url if self._base_url is not None else configured_base_url()

    def enable(
        self,
        root_id: str,
        credential_ref: str = DEFAULT_INGEST_CREDENTIAL_REF,
    ) -> dict[str, Any]:
        """Attach an ingest key to a bound root after proving it writes to that same workspace."""

        binding = self.safety.get_knowledge_binding(root_id)
        identity = self._identity(binding["workspace_id"], credential_ref, root_id)
        self.safety.set_ingest_credential(root_id, credential_ref)
        return {
            "root_id": root_id,
            "workspace_id": binding["workspace_id"],
            "ingest_credential_ref": credential_ref,
            "limits": _limits(identity),
        }

    def disable(self, root_id: str) -> None:
        self.safety.set_ingest_credential(root_id, None)

    def preview(self, root_id: str, paths: list[str]) -> dict[str, Any]:
        binding, credential_ref = self._ingest_binding(root_id)
        identity = self._identity(binding["workspace_id"], credential_ref, root_id)
        limits = _limits(identity)
        selected = list(dict.fromkeys(str(path).strip() for path in paths if str(path).strip()))
        if not selected:
            raise ValueError("Select at least one file to submit.")
        if len(selected) > MAX_FILES_PER_SUBMISSION:
            raise ValueError(f"Select at most {MAX_FILES_PER_SUBMISSION} files per submission.")

        root = Path(binding["root_path"])
        resolver = ScopedPathResolver.from_roots([root])
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for requested in selected:
            try:
                path = resolver.resolve(requested)
            except PermissionError:
                rejected.append({"path": requested, "reason": "outside the approved root"})
                continue
            rel_path = path.relative_to(root.resolve()).as_posix()
            reason = _reject_reason(path, limits)
            if reason:
                rejected.append({"path": rel_path, "reason": reason})
                continue
            accepted.append(
                {"rel_path": rel_path, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
        if not accepted:
            return {
                "submission_id": None,
                "status": "nothing_to_submit",
                "files": [],
                "rejected": rejected,
                "destination": self._destination(identity),
            }
        submission_id = self.safety.create_ingest_submission(
            root_id, binding["workspace_id"], accepted
        )
        return {
            **self.safety.get_ingest_submission(submission_id),
            "rejected": rejected,
            "destination": self._destination(identity),
        }

    def approve(self, submission_id: str) -> dict[str, Any]:
        submission = self.safety.get_ingest_submission(submission_id)
        if submission["status"] != "awaiting_approval":
            raise SubmissionStateError(
                f"Submission is {submission['status']}; only a pending preview can be approved."
            )
        root_id = submission["root_id"]
        binding, credential_ref = self._ingest_binding(root_id)
        if binding["workspace_id"] != submission["workspace_id"]:
            raise KnowledgeForbidden(
                "The root was re-bound to another workspace after this preview.", 403
            )
        identity = self._identity(binding["workspace_id"], credential_ref, root_id)
        limits = _limits(identity)
        client = self._client(credential_ref)
        root = Path(binding["root_path"])
        resolver = ScopedPathResolver.from_roots([root])

        self.safety.update_ingest_submission_status(submission_id, "submitting")
        for item in submission["files"]:
            request_id = str(uuid.uuid4())
            try:
                path = resolver.resolve(item["rel_path"])
            except PermissionError:
                self.safety.update_ingest_file(
                    item["id"], status="rejected", error="outside the approved root"
                )
                continue
            reason = _reject_reason(path, limits)
            if reason:
                self.safety.update_ingest_file(item["id"], status="rejected", error=reason)
                continue
            if _sha256(path) != item["sha256"]:
                self.safety.update_ingest_file(
                    item["id"], status="changed", error="file changed after preview; preview it again"
                )
                continue
            try:
                result = client.submit_file(path, source_ref=item["rel_path"], request_id=request_id)
            except KnowledgeError as exc:
                self.safety.update_ingest_file(item["id"], status="error", error=str(exc)[:500])
                continue
            self.safety.update_ingest_file(
                item["id"],
                status=str(result.get("status") or "queued"),
                asset_id=str(result["asset_id"]),
                document_id=str(result.get("document_id") or ""),
            )
            self.safety.record_audit(
                "knowledge.ingest_file_submitted",
                root_id=root_id,
                path=item["rel_path"],
                metadata={
                    "submission_id": submission_id,
                    "asset_id": result["asset_id"],
                    "sha256": item["sha256"],
                    "request_id": request_id,
                },
            )
        self.safety.update_ingest_submission_status(submission_id, "submitted")
        return self.refresh(submission_id)

    def refresh(self, submission_id: str) -> dict[str, Any]:
        submission = self.safety.get_ingest_submission(submission_id)
        if submission["status"] in {"awaiting_approval", "discarded"}:
            return submission
        binding, credential_ref = self._ingest_binding(submission["root_id"])
        if binding["workspace_id"] != submission["workspace_id"]:
            raise KnowledgeForbidden("The root is bound to another workspace now.", 403)
        client = self._client(credential_ref)
        for item in submission["files"]:
            if not item["asset_id"] or item["status"] in TERMINAL_FILE_STATES:
                continue
            try:
                status = client.job_status(item["asset_id"], request_id=str(uuid.uuid4()))
            except KnowledgeError as exc:
                self.safety.update_ingest_file(
                    item["id"], error=f"status unavailable: {str(exc)[:300]}"
                )
                continue
            self.safety.update_ingest_file(
                item["id"],
                status=str(status.get("status") or item["status"]),
                chunks_indexed=status.get("chunks_indexed"),
                error=status.get("error"),
            )
        updated = self.safety.get_ingest_submission(submission_id)
        if updated["status"] == "submitted" and all(
            item["status"] in TERMINAL_FILE_STATES for item in updated["files"]
        ):
            self.safety.update_ingest_submission_status(submission_id, "finished")
            updated = self.safety.get_ingest_submission(submission_id)
        return updated

    def discard(self, submission_id: str) -> None:
        submission = self.safety.get_ingest_submission(submission_id)
        if submission["status"] != "awaiting_approval":
            raise SubmissionStateError("Only a pending preview can be discarded.")
        self.safety.update_ingest_submission_status(submission_id, "discarded")

    def _ingest_binding(self, root_id: str) -> tuple[dict[str, Any], str]:
        binding = self.safety.get_knowledge_binding(root_id)  # PermissionError when revoked
        credential_ref = binding.get("ingest_credential_ref")
        if not credential_ref:
            raise PermissionError("File submission is not enabled for this root.")
        return binding, credential_ref

    def _identity(self, workspace_id: str, credential_ref: str, root_id: str) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        identity = self._client(credential_ref).whoami(request_id=request_id)
        if identity.get("workspace_id") != workspace_id:
            self.safety.record_audit(
                "knowledge.ingest_key_mismatch",
                root_id=root_id,
                metadata={"bound_workspace_id": workspace_id, "request_id": request_id},
            )
            raise KnowledgeForbidden(
                "This ingest key writes to a different workspace than the root is bound to.", 403
            )
        return identity

    def _client(self, credential_ref: str) -> IngestClient:
        url = self.base_url
        if not url:
            raise KnowledgeNotConfigured("Set CHAT_SOFTWARE_KNOWLEDGE_URL to enable file submission.")
        return self._client_factory(url, resolve_credential(credential_ref))

    def _destination(self, identity: dict[str, Any]) -> dict[str, Any]:
        host = urlsplit(self.base_url or "").hostname or ""
        local = host in {"127.0.0.1", "localhost", "::1"}
        return {
            "base_url": self.base_url,
            "workspace_id": identity.get("workspace_id"),
            "workspace_name": identity.get("workspace_name"),
            "location": "this computer" if local else f"remote server {host}",
            "note": (
                "Selected files are copied to the Retrieval Lab, which extracts text (including OCR), "
                "embeds and indexes them with the workspace's configured embedding model."
            ),
            "cancellation_supported": bool(identity.get("cancellation_supported")),
        }


def _limits(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_upload_bytes": int(identity.get("max_upload_bytes") or 0),
        "extensions": [str(ext).lower() for ext in identity.get("extensions") or []],
    }


def _reject_reason(path: Path, limits: dict[str, Any]) -> Optional[str]:
    if not path.exists():
        return "file not found"
    if not path.is_file():
        return "not a regular file"
    extension = path.suffix.lower()
    if extension not in limits["extensions"]:
        return f"unsupported type {extension or '(none)'}"
    size = path.stat().st_size
    if size == 0:
        return "empty file"
    if limits["max_upload_bytes"] and size > limits["max_upload_bytes"]:
        return f"larger than the backend limit ({limits['max_upload_bytes'] // (1024 * 1024)} MB)"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
