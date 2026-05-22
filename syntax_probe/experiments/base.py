"""Common experiment-runner protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..core.config import AppConfig
from ..core.context import RunContext


@dataclass(slots=True)
class ExperimentResult:
    """Returned from each experiment runner. The runner has already written all
    artifacts to disk; this is just a summary for logging and chaining."""

    experiment_kind: str
    run_id: str
    summary: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)


class ExperimentRunner(Protocol):
    """Each experiment kind implements this protocol."""

    kind: str

    def run(self, *, app_config: AppConfig, context: RunContext) -> ExperimentResult: ...


__all__ = ["ExperimentResult", "ExperimentRunner"]
