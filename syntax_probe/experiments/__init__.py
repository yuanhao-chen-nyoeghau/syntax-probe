"""Experiment runners.

Each runner takes a validated `AppConfig` plus a `RunContext` and produces
artifacts (cached activations, trained probes, JSON output) inside
`context.run_dir`. Runners are registered in the registry and selected by
config kind.
"""

from __future__ import annotations

from .activation_patching import ActivationPatchingRunner
from .apply_probes import ApplyProbesRunner
from .base import ExperimentResult, ExperimentRunner
from .c_command_stimuli import CCommandStimuliRunner
from .probe_training import ProbeTrainingRunner
from .registry import get_experiment_runner, register_experiment_runner
from .verify_stimuli import StimulusVerificationRunner
from .wh_extraction_stimuli import WhExtractionStimuliRunner

__all__ = [
    "ActivationPatchingRunner",
    "ApplyProbesRunner",
    "CCommandStimuliRunner",
    "ExperimentResult",
    "ExperimentRunner",
    "ProbeTrainingRunner",
    "StimulusVerificationRunner",
    "WhExtractionStimuliRunner",
    "get_experiment_runner",
    "register_experiment_runner",
]
