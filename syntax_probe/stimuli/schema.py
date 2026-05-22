"""Schema for stimulus records.

A `StimulusRecord` carries the rendered sentence, its tokenization, and per-role
token indices. The role-indices design is critical: experimental word pairs are
specified by *roles* (e.g., "wh", "embedded_verb"), and each stimulus records
the integer index of each role's first token. This way the same word-pair spec
applies across all stimuli regardless of lexical variation (e.g., multi-word
subjects like "the teacher" shifting downstream indices).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StimulusRecord(BaseModel):
    """A single stimulus with all metadata needed for downstream alignment."""

    model_config = ConfigDict(extra="forbid")

    stimulus_id: str
    """Unique within a stimulus set."""

    item_id: str
    """Identifier of the underlying lexical item (shared across conditions)."""

    condition: str
    """E.g., 'bare', 'infinitival', 'finite' for wh-extraction."""

    text: str
    """Rendered sentence as a single string. Joined with spaces between tokens."""

    tokens: list[str]
    """Word-level tokenization of `text`. These are the canonical word units; the
    LLM tokenizer is aligned to these via offset mapping. Using the same
    `tokenize_words` function as everywhere else in the codebase keeps tokenization
    consistent."""

    role_indices: dict[str, int]
    """Map from role name (e.g., 'wh', 'embedded_verb') to the token index of the
    *first* token of that role's surface expression. Every word pair the
    experiment cares about is specified by a (role_left, role_right) pair."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Free-form metadata: lexical items used, condition rank, etc. Available for
    statistical analysis as random effects."""


class StimulusSet(BaseModel):
    """A collection of stimuli for one experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    experiment: str
    records: list[StimulusRecord]
    description: str | None = None

    def by_condition(self) -> dict[str, list[StimulusRecord]]:
        out: dict[str, list[StimulusRecord]] = {}
        for record in self.records:
            out.setdefault(record.condition, []).append(record)
        return out

    def by_item(self) -> dict[str, list[StimulusRecord]]:
        out: dict[str, list[StimulusRecord]] = {}
        for record in self.records:
            out.setdefault(record.item_id, []).append(record)
        return out


__all__ = ["StimulusRecord", "StimulusSet"]
