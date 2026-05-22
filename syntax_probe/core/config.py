"""Typed configuration schemas.

Configs are loaded from YAML and validated with pydantic. The top-level `AppConfig`
is a discriminated union over experiment kinds; each experiment kind has its own
config schema and runner. This makes it easy to add new experiments without
touching shared code.

CLI overrides ``--set key.path=value`` are applied via :func:`apply_config_overrides`
before validation; pydantic then coerces the override strings to the appropriate
types (int / float / Path / Literal / bool / etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases for the new pluggable knobs
# ---------------------------------------------------------------------------

SubwordPooling = Literal["mean", "first"]
"""How to combine multiple subwords into one word vector during extraction.

* ``"mean"`` (default): average the hidden states of all subwords belonging to
  the word. Matches H&M (2019) footnote 4.
* ``"first"``: use only the first subword's hidden state. Matches some
  decoder-only probing papers.
"""

ProbeInputNormalization = Literal[
    "none",
    "l2_norm",
    "per_corpus_standardize",
    "per_token_layernorm",
]
"""How to normalize activations before they enter the probe.

* ``"none"``: feed activations directly to the probe (current/legacy behavior).
* ``"l2_norm"``: per-token L2-normalize each activation to unit length. Cheap;
  fully suppresses outlier-feature magnitude blowup in modern decoder-only LLMs;
  destroys magnitude information that may carry depth-like signal.
* ``"per_corpus_standardize"`` (default): subtract per-dimension mean and
  divide by per-dimension std, both computed once over the training corpus.
  Best balance between robustness and information preservation; recommended
  for modern LLMs.
* ``"per_token_layernorm"``: standardize each activation per-token (zero mean,
  unit variance over the hidden dimension). Equivalent to non-affine
  ``nn.LayerNorm``. Robust but compresses signal under outlier features.
"""


# ---------------------------------------------------------------------------
# Shared sub-configs
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Runtime knobs shared across all experiments."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 13
    output_dir: Path = Path("results")
    cache_dir: Path = Path("data/cache")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ModelConfig(BaseModel):
    """LLM specification for activation extraction."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "Qwen/Qwen2.5-1.5B"
    tokenizer_name: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    device_map: str | None = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "bfloat16"
    max_length: int = 2048
    batch_size: int = 8
    subword_pooling: SubwordPooling = "mean"
    """How to pool subwords into per-word vectors. See ``SubwordPooling``."""


class ProbeConfig(BaseModel):
    """Hewitt & Manning structural probe hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    rank: int = 64
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 20
    """Number of sentences per training batch (H&M default)."""

    input_normalization: ProbeInputNormalization = "per_corpus_standardize"
    """How to normalize activations before the probe linear projection.
    See ``ProbeInputNormalization``."""

    # H&M LR-reset-on-plateau schedule.
    lr_decay_factor: float = 0.1
    """Multiplicative LR decay factor applied at each plateau reset."""

    lr_decay_patience: int = 1
    """Reset the optimizer if dev loss does not improve for this many epochs.
    H&M's recipe is 1 (reset every plateau)."""

    lr_decay_max_resets: int = 4
    """Stop training entirely after this many plateau resets without improvement."""


# ---------------------------------------------------------------------------
# Experiment-specific configs (one per experiment kind)
# ---------------------------------------------------------------------------


class StimulusVerificationConfig(BaseModel):
    """Configuration for Task 0.5: verify dependency-distance equality across conditions.

    For each experiment, parse a sample of stimuli with a dependency parser and check
    that minimal pairs / triples produce equal gold tree distances for the experimental
    word pairs. This is a precondition for the experiment to be interpretable.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["verify_stimuli"] = "verify_stimuli"
    name: str = "verify_stimuli"
    target_experiment: Literal["wh_extraction", "c_command", "sr_sc"] = "wh_extraction"
    parser: Literal["spacy", "stanza"] = "spacy"
    spacy_model: str = "en_core_web_lg"
    max_distance_difference: int = 1
    """Acceptable difference in gold tree distance across conditions (in edges).

    For wh_extraction this is 1 (UD distances should be ~equal across
    conditions for the experimental pairs). For c_command this is more
    permissive (e.g. 4): UD path-length differences across (subject vs.
    modifier-internal) NPs are expected and are not the experimental signal.
    The verification still checks within-condition replicability of UD
    distances across lexical items.
    """

    num_items: int = 1000
    """Number of base lexical items to generate for verification.
    Total stimuli = 3 * num_items for wh_extraction, 4 * num_items for
    c_command. The combinatorial pool sizes are ~103K (wh) and ~12K (cc),
    so 1000 provides broad coverage in both."""

    gpu_id: int | None = None
    """Which GPU to use for parsing (relevant for ``en_core_web_trf`` and
    other transformer-based parser pipelines). ``None`` means CPU. For lg/md/sm
    spaCy models this has no effect — they run on CPU regardless. Override
    from the CLI via ``--set experiment.gpu_id=0``."""

    parser_batch_size: int = 64
    """Batch size for parsing. spaCy's ``nlp.pipe(batch_size=N)`` is dramatically
    faster than per-sentence parsing for transformer models (~100x speedup on
    GPU). Safe even on CPU; default 64 should fit in any environment."""


class WhExtractionStimuliConfig(BaseModel):
    """Stimulus generation for the wh-extraction experiment."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["wh_extraction_stimuli"] = "wh_extraction_stimuli"
    name: str = "wh_extraction_stimuli"
    num_items: int = 1000
    """Number of base lexical items; total stimuli = 3 * num_items.
    Default 1000 gives 3,000 stimuli from the ~103K-item combinatorial pool."""


class CCommandStimuliConfig(BaseModel):
    """Stimulus generation for the c-command (reflexive binding) experiment."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["c_command_stimuli"] = "c_command_stimuli"
    name: str = "c_command_stimuli"
    num_items: int = 1000
    """Number of base lexical items; total stimuli = 4 * num_items.
    Default 1000 gives 4,000 stimuli from the ~12K-item combinatorial pool."""


class ProbeTrainingConfig(BaseModel):
    """Train Hewitt & Manning structural probes on a parsed corpus."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["probe_training"] = "probe_training"
    name: str = "probe_training"

    corpus_path: Path
    """Path to parsed corpus JSONL (output of `download-corpus` script)."""

    probe: ProbeConfig = Field(default_factory=ProbeConfig)

    layers: list[int] | None = None
    """Layer indices to train probes on. `None` = all available layers."""

    train_split: Literal["train"] = "train"
    dev_split: Literal["dev"] = "dev"
    test_split: Literal["test"] = "test"

    activation_cache_name: str = "ud_ewt"
    """Name for the activation cache subdirectory."""


class ApplyProbesConfig(BaseModel):
    """Apply trained probes to experimental stimuli to produce per-condition layer profiles.

    Provide either ``probe_run_dir`` (an explicit run directory) or
    ``probe_run_name`` (a run-name prefix that is auto-resolved at runtime to
    the latest non-empty matching run under ``runtime.output_dir/probe_training``).
    Exactly one must be set.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["apply_probes"] = "apply_probes"
    name: str = "apply_probes"

    probe_run_dir: Path | None = None
    """Explicit probe-training run directory. If unset, ``probe_run_name`` is
    used to auto-resolve the latest matching run."""

    probe_run_name: str | None = None
    """Run-name prefix to auto-resolve. The runner picks the most-recently
    modified non-empty directory under ``runtime.output_dir/probe_training``
    whose name starts with this prefix followed by ``-``. If unset, an
    explicit ``probe_run_dir`` must be provided."""

    stimuli_kind: Literal["wh_extraction", "c_command"] = "wh_extraction"
    stimuli_config: WhExtractionStimuliConfig | CCommandStimuliConfig = Field(
        default_factory=WhExtractionStimuliConfig,
        discriminator="kind",
    )

    activation_cache_name: str = "wh_extraction_stimuli"


# ---------------------------------------------------------------------------
# Activation patching
# ---------------------------------------------------------------------------


class W2VariantConfig(BaseModel):
    """W2 — phase-edge specificity. Per-position patching at one fixed
    layer, two source-target pairs, lexical-DP-only stimulus filter.
    See ``docs/activation_patching_plan.md`` §3.1.
    """

    model_config = ConfigDict(extra="forbid")

    variant: Literal["w2"] = "w2"

    intervention_layer: int
    """The single ``L_patch = L_measure`` layer for this model. Set to
    the model's β-peak observational layer for the wh-esubj
    finite-bare contrast (from ``stats_summary.json``'s
    ``prediction_diagnostics.phase_count_gradient.finite_peak_layer``)."""

    intervention_roles: list[str] = Field(
        default_factory=lambda: ["wh", "embedded_subject", "embedded_verb"]
    )
    """Lexically-clean roles only by default. ``embedded_verb`` is
    auto-skipped for source-target pairs that include the finite
    condition (where 'ate' vs 'eat' creates a tense confound)."""

    source_target_pairs: list[tuple[str, str]] = Field(
        default_factory=lambda: [("finite", "bare"), ("infinitival", "bare")]
    )

    filter_to_lexical_dp: bool = True
    """Filter stimuli to items whose embedded subject is a lexical DP
    (the student / the assistant / the suspect), eliminating the
    case-marking confound at the embedded-subject position."""


class W4VariantConfig(BaseModel):
    """W4 — layer-localization. Single role, single source-target pair,
    a sweep of intervention layers, one fixed measurement layer.
    See ``docs/activation_patching_plan.md`` §3.2.
    """

    model_config = ConfigDict(extra="forbid")

    variant: Literal["w4"] = "w4"

    intervention_layers: list[int]
    """The L_patch sweep. Stage 1 typically uses all layers
    [0, 1, ..., n_layers-1]."""

    measurement_layer: int
    """The fixed L_measure (typically the model's β-peak observational
    layer for finite-bare). Probe distances at the patched target are
    computed at this layer for the headline contrast."""

    intervention_role: str = "embedded_subject"
    """Single intervention position. ``embedded_subject`` is the default
    (lexically clean for filtered items, structurally embedded). Could
    also be ``wh`` or ``matrix_verb``."""

    source_target_pair: tuple[str, str] = ("finite", "bare")

    filter_to_lexical_dp: bool = True


class N1VariantConfig(BaseModel):
    """N1 — per-layer robustness causality. Patching at the anaphor
    token, bidirectional by default.
    See ``docs/activation_patching_plan.md`` §3.3.
    """

    model_config = ConfigDict(extra="forbid")

    variant: Literal["n1"] = "n1"

    intervention_layer: int
    """Per-model β-peak layer for the cc reflexive 3-way (averaged
    across modifier types)."""

    bidirectional: bool = True
    """If True, run both match→swap and swap→match patches. Each gives
    independent causal evidence."""


_PatchingVariantConfig = Annotated[
    W2VariantConfig | W4VariantConfig | N1VariantConfig,
    Field(discriminator="variant"),
]


class ActivationPatchingConfig(BaseModel):
    """Run activation-patching trials for one model.

    The ``variant_config`` field selects the Tier-1 experiment (W2/W4/N1)
    and carries its variant-specific options. The runner generates trial
    specs from the existing observational stimuli (no new stimulus
    generation), caches source residuals at the relevant cells, runs
    patched and unpatched target forward passes, applies the trained
    probe to both, and writes a per-trial JSONL for downstream stats.

    Output schema (``per_trial_predictions.jsonl``): one row per trial
    with both patched and unpatched probe distances at the trial's
    measurement layers and pairs. Δβ analysis happens in the stats
    pipeline, not here, to keep the runner concerned only with raw
    measurements.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["activation_patching"] = "activation_patching"
    name: str = "activation_patching"

    probe_run_dir: Path | None = None
    probe_run_name: str | None = None

    stimuli_kind: Literal["wh_extraction", "c_command"] = "wh_extraction"
    stimuli_config: WhExtractionStimuliConfig | CCommandStimuliConfig = Field(
        default_factory=WhExtractionStimuliConfig,
        discriminator="kind",
    )

    variant_config: _PatchingVariantConfig

    @model_validator(mode="after")
    def _check_variant_stimuli_consistency(self) -> ActivationPatchingConfig:
        wh_variants = {"w2", "w4"}
        cc_variants = {"n1"}
        v = self.variant_config.variant
        if v in wh_variants and self.stimuli_kind != "wh_extraction":
            raise ValueError(
                f"Variant {v} requires stimuli_kind='wh_extraction', got "
                f"{self.stimuli_kind!r}."
            )
        if v in cc_variants and self.stimuli_kind != "c_command":
            raise ValueError(
                f"Variant {v} requires stimuli_kind='c_command', got "
                f"{self.stimuli_kind!r}."
            )
        return self


# ---------------------------------------------------------------------------
# Discriminated union over experiment kinds
# ---------------------------------------------------------------------------


ExperimentConfig = Annotated[
    StimulusVerificationConfig
    | WhExtractionStimuliConfig
    | CCommandStimuliConfig
    | ProbeTrainingConfig
    | ApplyProbesConfig
    | ActivationPatchingConfig,
    Field(discriminator="kind"),
]


class AppConfig(BaseModel):
    """Top-level config. `experiment.kind` selects the runner."""

    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    experiment: ExperimentConfig


# ---------------------------------------------------------------------------
# Loading + overrides
# ---------------------------------------------------------------------------


def load_app_config(
    config_path: Path,
    *,
    overrides: list[str] | None = None,
) -> AppConfig:
    """Load and validate a YAML config file, optionally applying overrides.

    Args:
        config_path: path to the YAML config file.
        overrides: list of ``"key.path=value"`` strings (from CLI ``--set``).
            See :func:`apply_config_overrides` for the supported value syntax.

    Raises:
        TypeError: if the config is not a YAML mapping.
        ValueError: if an override is malformed or refers to a non-existent path.
    """
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}, got {type(raw).__name__}")

    if overrides:
        raw = apply_config_overrides(raw, overrides)

    return AppConfig.model_validate(raw)


def apply_config_overrides(
    raw: dict[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """Apply ``--set key.path=value`` overrides to a raw config dict in place.

    Each override is a string of the form ``"<dotted.key.path>=<value>"``.

    Value syntax:
        * Strings starting with ``"["``, ``"{"``, ``"true"``, ``"false"``, or
          ``"null"``, or that parse as numbers, are decoded as JSON.
        * Otherwise the value is treated as a plain string. Pydantic will then
          coerce it to the schema-declared type (int, float, Path, Literal, ...).

    Raises:
        ValueError: if an override is malformed or its key path does not exist
            in the raw config.

    Returns:
        The mutated ``raw`` dict (also returned for convenience).
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}: expected 'key.path=value' form."
            )
        key_path, raw_value = override.split("=", 1)
        key_path = key_path.strip()
        if not key_path:
            raise ValueError(f"Invalid override {override!r}: empty key path.")

        value: Any = _decode_override_value(raw_value)
        _set_nested_key(raw, key_path.split("."), value, original=override)

    return raw


def _decode_override_value(text: str) -> Any:
    """Decode an override value, falling back to a plain string."""
    stripped = text.strip()
    if not stripped:
        return text  # preserve empty string
    # Try JSON first to capture numbers, bools, null, lists, dicts.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def _set_nested_key(
    target: dict[str, Any],
    parts: list[str],
    value: Any,
    *,
    original: str,
) -> None:
    """Walk ``parts`` into ``target`` and set the leaf.

    If an intermediate key is absent, an empty dict is created for it. This
    allows overrides to set fields on nested pydantic models whose keys are not
    present in the YAML (because the model uses pydantic defaults). The
    downstream ``AppConfig.model_validate`` call will reject truly invalid keys
    thanks to ``extra="forbid"`` on all config models.
    """
    cursor: Any = target
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError(
                f"Override {original!r} cannot descend into non-mapping value at {part!r}."
            )
        if part not in cursor:
            cursor[part] = {}
        cursor = cursor[part]
    leaf = parts[-1]
    if not isinstance(cursor, dict):
        raise ValueError(
            f"Override {original!r} cannot set {leaf!r} on non-mapping value."
        )
    cursor[leaf] = value


__all__ = [
    "AppConfig",
    "ApplyProbesConfig",
    "CCommandStimuliConfig",
    "ModelConfig",
    "ProbeConfig",
    "ProbeInputNormalization",
    "ProbeTrainingConfig",
    "RuntimeConfig",
    "StimulusVerificationConfig",
    "SubwordPooling",
    "WhExtractionStimuliConfig",
    "apply_config_overrides",
    "load_app_config",
]
