"""Schema for parsed sentences and dependency arcs.

A `ParsedSentence` carries the surface tokens, the dependency arcs, and
optionally a split label (train/dev/test). The dependency arc convention
follows UD: each arc points from a head to a dependent, with relation labels
from UD's tag set. Indices are 0-based into the `tokens` list. The root of
the sentence is represented by an arc with `head_index = -1`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Split = Literal["train", "dev", "test"]


class DependencyArc(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    head_index: int
    """Index of the head token; -1 for the root arc."""

    dependent_index: int
    """Index of the dependent token (always >= 0)."""

    relation: str
    """UD dependency relation (e.g., 'nsubj', 'obj', 'xcomp')."""


class ParsedSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_id: str
    text: str
    """Reconstructed sentence string. May not exactly equal the original raw text
    because UD tokens are space-joined."""

    tokens: list[str]
    """The UD tokens. These are the canonical word units for distance computation
    and probe alignment."""

    dependency_arcs: list[DependencyArc]
    split: Split | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ParsedCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sentences: list[ParsedSentence]
    source: str | None = None
    """Where the parses came from (e.g., URL, file path, parser name + version)."""

    def by_split(self, split: Split) -> list[ParsedSentence]:
        return [s for s in self.sentences if s.split == split]


__all__ = ["DependencyArc", "ParsedCorpus", "ParsedSentence", "Split"]
