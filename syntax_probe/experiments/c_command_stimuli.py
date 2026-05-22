"""Generate c-command stimuli and write them to disk.

A thin wrapper around `stimuli.c_command.generate_c_command_stimuli` that
produces a config + manifest trail, mirroring `wh_extraction_stimuli.py`.
"""

from __future__ import annotations

import logging

from ..core.config import AppConfig, CCommandStimuliConfig
from ..core.context import RunContext, write_manifest
from ..core.io import write_jsonl
from ..core.seed import seed_everything
from ..stimuli.c_command import generate_c_command_stimuli
from .base import ExperimentResult, ExperimentRunner
from .registry import register_experiment_runner

logger = logging.getLogger(__name__)


class CCommandStimuliRunner(ExperimentRunner):
    kind = "c_command_stimuli"

    def run(self, *, app_config: AppConfig, context: RunContext) -> ExperimentResult:
        if not isinstance(app_config.experiment, CCommandStimuliConfig):
            raise TypeError("Expected c_command_stimuli config")

        seed_everything(app_config.runtime.seed)
        stimulus_set = generate_c_command_stimuli(num_items=app_config.experiment.num_items)

        stimuli_path = context.artifact_dir / "c_command_stimuli.jsonl"
        write_jsonl(
            stimuli_path,
            (record.model_dump(mode="json") for record in stimulus_set.records),
        )

        condition_counts = {
            condition: len(records)
            for condition, records in stimulus_set.by_condition().items()
        }
        logger.info(
            "Wrote %d stimuli (%s) to %s",
            len(stimulus_set.records),
            condition_counts,
            stimuli_path,
        )

        manifest_path = write_manifest(
            context,
            {
                "stimulus_count": len(stimulus_set.records),
                "condition_counts": condition_counts,
                "artifacts": {"stimuli": str(stimuli_path)},
            },
        )

        return ExperimentResult(
            experiment_kind=self.kind,
            run_id=context.run_id,
            summary={
                "stimulus_count": len(stimulus_set.records),
                "condition_counts": condition_counts,
            },
            artifacts={"stimuli": stimuli_path, "manifest": manifest_path},
        )


register_experiment_runner("c_command_stimuli", CCommandStimuliRunner)


__all__ = ["CCommandStimuliRunner"]
