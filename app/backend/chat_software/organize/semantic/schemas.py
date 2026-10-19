"""Flat structured-output contracts for semantic file planning."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileLabel(BaseModel):
    id: int
    label: str = Field(max_length=32)


class LabelBatch(BaseModel):
    labels: list[FileLabel]


class Folder(BaseModel):
    index: int
    name: str = Field(max_length=48)
    parent: int | None = None
    rationale: str = Field(max_length=120)


class Taxonomy(BaseModel):
    folders: list[Folder] = Field(max_length=24)


class FileAssignment(BaseModel):
    file_id: int
    folder_index: int
