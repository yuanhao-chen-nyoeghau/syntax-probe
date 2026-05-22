"""Generate wh-extraction stimuli and write them to disk.

This is a thin wrapper around `stimuli.wh_extraction.generate_wh_extraction_stimuli`.
It exists as a runner mostly so that stimulus generation has a config + manifest
trail like every other experiment.
"""

from __future__ import annotations

import logging

from ..core.config import AppConfig, WhExtractionStimuliConfig
from ..core.context import RunContext, write_manifest
from ..core.io import write_jsonl
from ..core.seed import seed_everything
from ..stimuli.wh_extraction import generate_wh_extraction_stimuli
from .base import ExperimentResult, ExperimentRunner
from .registry import register_experiment_runner

logger = logging.getLogger(__name__)


class WhExtractionStimuliRunner(ExperimentRunner):
    kind = "wh_extraction_stimuli"

    def run(self, *, app_config: AppConfig, context: RunContext) -> ExperimentResult:
        if not isinstance(app_config.experiment, WhExtractionStimuliConfig):
            raise TypeError("Expected wh_extraction_stimuli config")

        seed_everything(app_config.runtime.seed)
        stimulus_set = generate_wh_extraction_stimuli(num_items=app_config.experiment.num_items)

        stimuli_path = context.artifact_dir / "wh_extraction_stimuli.jsonl"
        write_jsonl(stimuli_path, (record.model_dump(mode="json") for record in stimulus_set.records))

        condition_counts = {
            condition: len(records) for condition, records in stimulus_set.by_condition().items()
        }
        logger.info("Wrote %d stimuli (%s) to %s",
                    len(stimulus_set.records), condition_counts, stimuli_path)

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


register_experiment_runner("wh_extraction_stimuli", WhExtractionStimuliRunner)


__all__ = ["WhExtractionStimuliRunner"]
