"""Registry of experiment runners by kind."""

from __future__ import annotations

from collections.abc import Callable

from .base import ExperimentRunner

_REGISTRY: dict[str, Callable[[], ExperimentRunner]] = {}


def register_experiment_runner(kind: str, factory: Callable[[], ExperimentRunner]) -> None:
    if kind in _REGISTRY:
        raise ValueError(f"Experiment kind {kind!r} is already registered")
    _REGISTRY[kind] = factory


def get_experiment_runner(kind: str) -> ExperimentRunner:
    if kind not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"No experiment runner registered for kind={kind!r}. Available: {available}"
        )
    return _REGISTRY[kind]()


__all__ = ["get_experiment_runner", "register_experiment_runner"]
