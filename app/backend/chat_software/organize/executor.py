"""Deterministic apply/undo executor for stored organize plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chat_software.safety_store import SafetyStore
from chat_software.tools.file_ops_write import make_dir, remove_empty_dir, safe_move
from chat_software.tools.scope import ScopedPathResolver


def apply_organize_plan(store: SafetyStore, plan_id: str) -> dict[str, object]:
    plan = store.get_organize_plan(plan_id)
    status = str(plan["status"])
    if status == "applied":
        return {"plan_id": plan_id, "status": "applied", "applied": [], "already_applied": True}
    if status not in {"planned", "failed"}:
        raise RuntimeError(f"Plan {plan_id} cannot be applied from status {status!r}.")

    root = Path(str(plan["root"])).expanduser().resolve()
    resolver = ScopedPathResolver.from_roots([root])
    actions = store.get_organize_actions(plan_id)
    applied: list[dict[str, object]] = []
    failed: dict[str, object] | None = None

    store.update_plan_status(plan_id, "applying")
    store.record_audit(
        "organize.apply_started",
        root_id=str(plan["root_id"]),
        plan_id=plan_id,
        path=str(root),
        metadata={"actions": len(actions)},
    )

    for action_row in actions:
        if action_row["status"] == "applied":
            continue
        action = dict(action_row["action"])
        action_id = int(action_row["id"])
        op = str(action.get("op") or "")
        store.update_action_status(action_id, "running")
        reverse = _reverse_for_action(action)
        journal_id = store.start_journal_entry(
            batch_id=plan_id,
            plan_id=plan_id,
            action_id=action_id,
            op=op,
            forward=action,
            reverse=reverse,
        )

        try:
            result = _apply_action(action, resolver, plan_id)
            if op == "move":
                reverse = {
                    "op": "move",
                    "from": result["to"],
                    "to": result["from"],
                    "source_quarantine": result.get("source_quarantine"),
                }
            elif op == "mkdir" and not result.get("created"):
                reverse = {"op": "noop", "reason": "directory already existed"}
            store.complete_journal_entry(journal_id, reverse=reverse)
            store.update_action_status(action_id, "applied")
            applied.append({"action_id": action_id, "action": action, "result": result})
        except Exception as exc:
            store.complete_journal_entry(journal_id, status="failed", error=str(exc))
            store.update_action_status(action_id, "failed")
            failed = {"action_id": action_id, "action": action, "error": str(exc)}
            break

    if failed:
        final_status = "partial" if applied else "failed"
        store.update_plan_status(plan_id, final_status)
        store.record_audit(
            "organize.apply_failed",
            root_id=str(plan["root_id"]),
            plan_id=plan_id,
            path=str(root),
            metadata={"applied": len(applied), "failed": failed},
        )
        return {"plan_id": plan_id, "status": final_status, "applied": applied, "failed": failed}

    store.update_plan_status(plan_id, "applied")
    store.record_audit(
        "organize.apply_completed",
        root_id=str(plan["root_id"]),
        plan_id=plan_id,
        path=str(root),
        metadata={"applied": len(applied)},
    )
    return {"plan_id": plan_id, "status": "applied", "applied": applied}


def undo_organize_plan(store: SafetyStore, plan_id: str) -> dict[str, object]:
    plan = store.get_organize_plan(plan_id)
    status = str(plan["status"])
    if status == "undone":
        return {"plan_id": plan_id, "status": "undone", "undone": [], "already_undone": True}
    if status not in {"applied", "partial"}:
        raise RuntimeError(f"Plan {plan_id} cannot be undone from status {status!r}.")

    root = Path(str(plan["root"])).expanduser().resolve()
    resolver = ScopedPathResolver.from_roots([root])
    entries = [
        entry
        for entry in store.get_journal_entries(plan_id, reverse_order=True)
        if entry["status"] == "completed"
    ]
    undone: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    failed: dict[str, object] | None = None

    store.record_audit(
        "organize.undo_started",
        root_id=str(plan["root_id"]),
        plan_id=plan_id,
        path=str(root),
        metadata={"journal_entries": len(entries)},
    )

    for entry in entries:
        reverse = dict(entry["reverse"])
        try:
            result = _apply_reverse(reverse, resolver, plan_id)
            store.complete_journal_entry(int(entry["id"]), status="undone")
            if entry["action_id"] is not None:
                store.update_action_status(int(entry["action_id"]), "undone")
            if result.get("skipped"):
                skipped.append({"journal_id": entry["id"], "reverse": reverse, "result": result})
            else:
                undone.append({"journal_id": entry["id"], "reverse": reverse, "result": result})
        except Exception as exc:
            failed = {"journal_id": entry["id"], "reverse": reverse, "error": str(exc)}
            break

    if failed:
        store.update_plan_status(plan_id, "partial")
        store.record_audit(
            "organize.undo_failed",
            root_id=str(plan["root_id"]),
            plan_id=plan_id,
            path=str(root),
            metadata={"undone": len(undone), "skipped": len(skipped), "failed": failed},
        )
        return {
            "plan_id": plan_id,
            "status": "partial",
            "undone": undone,
            "skipped": skipped,
            "failed": failed,
        }

    store.update_plan_status(plan_id, "undone")
    store.record_audit(
        "organize.undo_completed",
        root_id=str(plan["root_id"]),
        plan_id=plan_id,
        path=str(root),
        metadata={"undone": len(undone), "skipped": len(skipped)},
    )
    return {"plan_id": plan_id, "status": "undone", "undone": undone, "skipped": skipped}


def _apply_action(
    action: dict[str, Any],
    resolver: ScopedPathResolver,
    batch_id: str,
) -> dict[str, object]:
    op = str(action.get("op") or "")
    if op == "mkdir":
        path = _required(action, "path")
        return make_dir(path, resolver)
    if op == "move":
        source = _required(action, "from")
        destination = _required(action, "to")
        return safe_move(source, destination, resolver, batch_id=batch_id)
    raise ValueError(f"Unsupported organize action: {op}")


def _apply_reverse(
    reverse: dict[str, Any],
    resolver: ScopedPathResolver,
    batch_id: str,
) -> dict[str, object]:
    op = str(reverse.get("op") or "")
    if op == "move":
        source = _required(reverse, "from")
        destination = _required(reverse, "to")
        source_path = resolver.resolve(source)
        if not source_path.exists():
            return {"op": "move", "from": str(source_path), "skipped": True, "reason": "missing"}
        return safe_move(source, destination, resolver, batch_id=f"{batch_id}-undo")
    if op == "rmdir":
        path = _required(reverse, "path")
        target = resolver.resolve(path)
        if not target.exists():
            return {"op": "rmdir", "path": str(target), "skipped": True, "reason": "missing"}
        try:
            return remove_empty_dir(path, resolver)
        except OSError as exc:
            return {
                "op": "rmdir",
                "path": str(target),
                "skipped": True,
                "reason": str(exc),
            }
    if op == "noop":
        return {"op": "noop", "skipped": True}
    raise ValueError(f"Unsupported reverse action: {op}")


def _reverse_for_action(action: dict[str, Any]) -> dict[str, Any]:
    op = str(action.get("op") or "")
    if op == "mkdir":
        return {"op": "rmdir", "path": _required(action, "path")}
    if op == "move":
        return {
            "op": "move",
            "from": _required(action, "to"),
            "to": _required(action, "from"),
        }
    return {"op": "noop"}


def _required(action: dict[str, Any], key: str) -> str:
    value = action.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Action is missing required field {key!r}.")
    return value
