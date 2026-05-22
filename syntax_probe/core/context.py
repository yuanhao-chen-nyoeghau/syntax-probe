"""Per-run execution context.

A `RunContext` ties together the config, output directories, and a stable run id
that survives across re-runs of the same config. Every experiment writes a
`manifest.json` to its run dir documenting all artifacts and provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunContext:
    """All paths and identifiers needed during an experiment run."""

    config: AppConfig
    config_path: Path
    config_hash: str
    run_id: str
    run_dir: Path
    artifact_dir: Path
    cache_dir: Path

    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"


def build_run_context(*, app_config: AppConfig, config_path: Path) -> RunContext:
    """Create a RunContext, materializing output directories on disk."""
    config_hash = _compute_config_hash(app_config)
    run_id = _generate_run_id(app_config, config_hash)
    run_dir = app_config.runtime.output_dir / app_config.experiment.kind / run_id
    artifact_dir = run_dir / "artifacts"
    cache_dir = app_config.runtime.cache_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run id: %s", run_id)
    logger.info("Run dir: %s", run_dir)

    return RunContext(
        config=app_config,
        config_path=config_path,
        config_hash=config_hash,
        run_id=run_id,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        cache_dir=cache_dir,
    )


def _compute_config_hash(app_config: AppConfig) -> str:
    """Stable hash of the config, ignoring runtime knobs that don't affect outputs."""
    data = app_config.model_dump(mode="json")
    # Exclude log_level / output_dir from the hash; they don't affect results.
    runtime = data.get("runtime", {})
    for key in ("log_level", "output_dir"):
        runtime.pop(key, None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _generate_run_id(app_config: AppConfig, config_hash: str) -> str:
    """Run id: <experiment-name>-<config-hash>-<timestamp>."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    name = app_config.experiment.name
    return f"{name}-{config_hash[:8]}-{timestamp}"


def write_manifest(context: RunContext, payload: dict[str, object]) -> Path:
    """Write a manifest with run provenance and artifact paths."""
    manifest_path = context.manifest_path()
    full_payload = {
        "run_id": context.run_id,
        "config_hash": context.config_hash,
        "config_path": str(context.config_path),
        "experiment_kind": context.config.experiment.kind,
        "config": context.config.model_dump(mode="json"),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        **payload,
    }
    manifest_path.write_text(json.dumps(full_payload, indent=2, default=str), encoding="utf-8")
    return manifest_path


__all__ = ["RunContext", "build_run_context", "write_manifest"]
