"""Path-safe semantic plan compiler helpers.

The LLM-facing contract may name categories or assign file IDs, but this module
is the only semantic layer that turns those labels into folder names.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

TOPIC_RULES: tuple[tuple[str, set[str]], ...] = (
    ("Finance", {"bank", "bill", "budget", "invoice", "payment", "payroll", "receipt", "salary", "statement", "tax"}),
    ("Identity", {"aadhaar", "card", "id", "identity", "license", "pan", "passport", "visa"}),
    ("Resumes", {"bio", "cover", "cv", "job", "letter", "portfolio", "resume"}),
    ("Travel", {"boarding", "flight", "hotel", "itinerary", "ticket", "travel", "trip", "vacation"}),
    ("Medical", {"claim", "doctor", "health", "hospital", "insurance", "lab", "medical", "prescription", "report"}),
    (
        "Education",
        {"certificate", "class", "course", "degree", "education", "exam", "marksheet", "school", "transcript"},
    ),
    ("Contracts", {"agreement", "contract", "legal", "nda", "policy", "signed"}),
    ("Projects", {"brief", "plan", "proposal", "project", "roadmap", "spec", "strategy"}),
    ("Screenshots", {"screen", "screenshot", "shot"}),
    ("Photos", {"camera", "dsc", "dscf", "img", "image", "photo", "pic", "picture"}),
)

EXTENSION_TOPICS = {
    ".aac": "Audio",
    ".avi": "Videos",
    ".bmp": "Photos",
    ".c": "Code",
    ".cpp": "Code",
    ".csv": "Spreadsheets",
    ".doc": "Documents",
    ".docx": "Documents",
    ".flac": "Audio",
    ".gif": "Photos",
    ".go": "Code",
    ".heic": "Photos",
    ".jpeg": "Photos",
    ".jpg": "Photos",
    ".js": "Code",
    ".json": "Code",
    ".m4a": "Audio",
    ".md": "Documents",
    ".mkv": "Videos",
    ".mov": "Videos",
    ".mp3": "Audio",
    ".mp4": "Videos",
    ".pdf": "Documents",
    ".png": "Photos",
    ".ppt": "Presentations",
    ".pptx": "Presentations",
    ".py": "Code",
    ".rs": "Code",
    ".svg": "Photos",
    ".ts": "Code",
    ".txt": "Documents",
    ".wav": "Audio",
    ".webm": "Videos",
    ".webp": "Photos",
    ".xls": "Spreadsheets",
    ".xlsx": "Spreadsheets",
    ".zip": "Archives",
}


def sanitize_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "-", value).strip(" .-_")
    cleaned = re.sub(r"\s+", " ", cleaned).rstrip(" .")
    root_name = cleaned.split(".", 1)[0].upper()
    if root_name in WINDOWS_RESERVED_NAMES:
        return "Unsorted"
    return cleaned[:80] or "Unsorted"


def topic_folder_for_source(
    source: Path,
    *,
    root: str | Path,
    prompt: str | None = None,
    file_filter: str = "all",
) -> str:
    label = topic_label_for_source(source, root=root, prompt=prompt)
    if file_filter != "all":
        return str(PurePosixPath(_filter_base_folder(file_filter)) / sanitize_folder_name(label))
    return sanitize_folder_name(label)


def topic_label_for_source(source: Path, *, root: str | Path, prompt: str | None = None) -> str:
    root_path = Path(root).expanduser().resolve()
    tokens = _source_tokens(source, root_path)
    prompt_tokens = set(_tokenize(prompt or ""))
    for label, keywords in TOPIC_RULES:
        if tokens & keywords:
            return label
    for label, keywords in TOPIC_RULES:
        if prompt_tokens & keywords and tokens & _broad_type_tokens(label):
            return label
    return EXTENSION_TOPICS.get(source.suffix.lower(), "Unsorted")


def _source_tokens(source: Path, root: Path) -> set[str]:
    words = set(_tokenize(source.stem))
    try:
        parent = source.parent.resolve().relative_to(root)
    except ValueError:
        parent = Path()
    for part in parent.parts[:3]:
        words.update(_tokenize(part))
    return words


def _tokenize(value: str) -> list[str]:
    return [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", value)
        if token and not token.isdigit()
    ]


def _filter_base_folder(file_filter: str) -> str:
    return {
        "image": "Images",
        "video": "Videos",
        "audio": "Audio",
        "document": "Documents",
    }.get(file_filter, "Topics")


def _broad_type_tokens(label: str) -> set[str]:
    if label in {"Photos", "Screenshots"}:
        return {"jpg", "jpeg", "png", "heic", "webp", "photo", "image", "img"}
    if label == "Videos":
        return {"mp4", "mov", "video"}
    if label == "Audio":
        return {"mp3", "wav", "audio"}
    return set()
