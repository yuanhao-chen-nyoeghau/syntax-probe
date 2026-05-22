"""Minimal-pair stimulus generators.

Each experiment has its own generator that produces a `StimulusSet` of records
with a controlled minimal-pair / minimal-triple design and rich metadata
(condition labels, role indices, lexical items used). Stimuli are deterministic:
running the generator with the same arguments twice produces identical output.

The public API consists of the per-experiment ``generate_*`` factory functions
plus the ``StimulusRecord``/``StimulusSet`` types. The internal lexical-item
dataclasses (``ReflexiveItem``, ``BoundVarItem``, ``PrincipleCItem``,
``WhExtractionItem``) live in their respective submodules and are imported
directly from there when needed (e.g., by tests).
"""

from __future__ import annotations

from .c_command import generate_c_command_stimuli
from .schema import StimulusRecord, StimulusSet
from .wh_extraction import generate_wh_extraction_stimuli

__all__ = [
    "StimulusRecord",
    "StimulusSet",
    "generate_c_command_stimuli",
    "generate_wh_extraction_stimuli",
]
