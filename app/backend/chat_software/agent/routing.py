"""Pure routing predicates for the Phase 3 root graph."""

from __future__ import annotations

import re
from dataclasses import dataclass


FILE_COUNT_TYPE_WORDS = {
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "pictures",
    "video",
    "videos",
    "audio",
    "archive",
    "archives",
    "document",
    "documents",
    "spreadsheet",
    "spreadsheets",
    "presentation",
    "presentations",
    "pdf",
    "pdfs",
    "png",
    "pngs",
    "jpg",
    "jpgs",
    "jpeg",
    "jpegs",
    "gif",
    "gifs",
    "svg",
    "svgs",
    "webp",
    "heic",
}


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reason: str


def route_message(
    message: str,
    *,
    roots_present: bool,
    enable_llm_planner: bool,
) -> RoutingDecision:
    lowered = message.lower()
    if requires_scoped_tool(lowered):
        if not roots_present:
            return RoutingDecision("file_ops", "file request needs an approved root")
        return RoutingDecision("file_ops", "deterministic file intent")
    if enable_llm_planner and roots_present and likely_filesystem_question(lowered):
        return RoutingDecision("file_ops", "LLM planner candidate")
    return RoutingDecision("chat", "ordinary chat")


def requires_scoped_tool(lowered: str) -> bool:
    return (
        looks_like_organize_request(lowered)
        or looks_like_direct_write_request(lowered)
        or looks_like_file_count_request(lowered)
        or looks_like_directory_check_request(lowered)
        or "git diff" in lowered
        or ("diff" in lowered and "repo" in lowered)
        or "git status" in lowered
        or ("status" in lowered and "repo" in lowered)
        or looks_like_read_request(lowered)
        or lowered.startswith("stat")
        or lowered.startswith("info for")
        or "glob" in lowered
        or "match files" in lowered
        or looks_like_list_request(lowered)
        or looks_like_project_overview_request(lowered)
        or looks_like_file_search_request(lowered)
    )


def likely_filesystem_question(lowered: str) -> bool:
    return any(
        word in lowered
        for word in [
            "file",
            "folder",
            "directory",
            "repo",
            "repository",
            "project",
            "path",
            "workspace",
        ]
    )


def looks_like_file_search_request(lowered: str) -> bool:
    if not re.search(r"\b(?:find|search|summari[sz]e|about)\b", lowered, flags=re.IGNORECASE):
        return False
    has_search_action = bool(
        re.search(r"\b(?:find|search|summari[sz]e)\b", lowered, flags=re.IGNORECASE)
    )
    has_file_context = bool(
        re.search(
            r"\b(?:file|files|folder|folders|directory|directories|repo|repository|project|"
            r"workspace|path|root|selected|current)\b",
            lowered,
            flags=re.IGNORECASE,
        )
    )
    has_path_or_glob = bool(
        re.search(r"(?:\*\.|/|\\|\.[a-z0-9]{1,8}\b)", lowered, flags=re.IGNORECASE)
    )
    has_location_phrase = bool(
        re.search(r"\b(?:in|inside|under|from)\s+(?:this|the|selected|current|\.|/|~)", lowered)
    )
    return has_search_action and (has_file_context or has_path_or_glob or has_location_phrase)


def looks_like_organize_request(lowered: str) -> bool:
    return any(
        word in lowered
        for word in ["organize", "organise", "arrange", "distribute", "group", "separate"]
    ) or (
        "sort" in lowered and any(word in lowered for word in ["file", "folder", "directory"])
    )


def looks_like_direct_write_request(lowered: str) -> bool:
    cleaned = strip_request_prefix(lowered)
    return (
        cleaned.startswith(("mkdir ", "copy ", "move ", "rename ", "edit ", "replace "))
        or bool(
            re.search(
                r"\b(?:create|make|add)\s+(?:a\s+|new\s+)?(?:file|folder|directory|dir)\b",
                cleaned,
                flags=re.IGNORECASE,
            )
        )
        or cleaned.startswith("write file ")
        or bool(re.search(r"^write\s+.+\s+to\s+(?:file\s+)?", cleaned, flags=re.IGNORECASE))
    )


def looks_like_file_count_request(lowered: str) -> bool:
    has_count_word = bool(
        re.search(r"\b(?:count|number\s+of|how\s+many|total)\b", lowered, flags=re.IGNORECASE)
    )
    if not has_count_word:
        return False
    has_file_word = bool(re.search(r"\bfiles?\b", lowered, flags=re.IGNORECASE))
    has_explicit_pattern = bool(
        re.search(r"(\*\.[a-z0-9][a-z0-9_-]*)", lowered, flags=re.IGNORECASE)
    )
    words = set(re.findall(r"\b[a-z0-9][a-z0-9_-]*\b", lowered, flags=re.IGNORECASE))
    return bool(
        has_file_word
        or has_explicit_pattern
        or any(word in FILE_COUNT_TYPE_WORDS for word in words)
    )


def looks_like_directory_check_request(lowered: str) -> bool:
    cleaned = lowered.strip()
    return bool(
        re.search(
            r"^(?:please\s+)?(?:check|inspect|explore|scan|open|look\s+at)\s+"
            r"(?:the\s+)?(?:(?:this|current|approved|selected)\s+)?"
            r"(?:folder|directory|dir|root|path)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"^(?:please\s+)?(?:check|inspect|explore|scan|open|look\s+at)\s+.+\s+"
            r"(?:folder|directory|dir)\s*$",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def looks_like_read_request(lowered: str) -> bool:
    return (
        lowered.startswith("read ")
        or (lowered.startswith("show ") and not looks_like_list_request(lowered))
        or (lowered.startswith("open ") and not looks_like_directory_check_request(lowered))
        or lowered.startswith("cat ")
        or "read file" in lowered
        or bool(re.search(r"\bshow\s+file\b", lowered, flags=re.IGNORECASE))
    )


def looks_like_list_request(lowered: str) -> bool:
    return (
        lowered.startswith("list")
        or "list files" in lowered
        or "show files" in lowered
        or "show the files" in lowered
        or "show folders" in lowered
        or "show the folders" in lowered
        or "what files" in lowered
        or "which files" in lowered
        or "files are in" in lowered
        or "files in selected" in lowered
        or "files from selected" in lowered
        or "what is in this folder" in lowered
        or "what is in this directory" in lowered
        or "what is in selected" in lowered
        or "what is inside" in lowered
    )


def looks_like_project_overview_request(lowered: str) -> bool:
    if not any(word in lowered for word in ["project", "repo", "repository", "folder", "directory", "path"]):
        return False
    return any(
        phrase in lowered
        for phrase in [
            "what is this",
            "what does this",
            "check this project",
            "check this repo",
            "check this repository",
            "inspect this project",
            "inspect this repo",
            "inspect this repository",
            "overview",
            "summarize this",
            "summarise this",
            "summarize selected",
            "summarise selected",
            "summary of this",
            "data of the files",
            "data in the files",
            "get the data",
            "explain this",
        ]
    )


def strip_request_prefix(value: str) -> str:
    return re.sub(
        r"^(?:(?:please|kindly)\s+|(?:could|can|would|will)\s+you\s+)",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    ).strip()
