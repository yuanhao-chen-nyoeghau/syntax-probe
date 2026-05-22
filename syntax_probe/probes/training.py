"""Train Hewitt & Manning structural probes per layer.

Training procedure (mirroring H&M's `run_experiment.py`):
1. For each layer, instantiate a fresh probe with the configured input normalizer.
2. Train on `train` split with Adam optimizer, L1 loss, batches of 20 sentences.
3. After each epoch, evaluate on `dev` split.
4. **LR-reset-on-plateau** (H&M Appendix A.2): if dev loss does not improve for
   ``lr_decay_patience`` epochs, reset the optimizer (no momentum kept) and
   multiply the learning rate by ``lr_decay_factor``. After ``lr_decay_max_resets``
   such resets without further improvement, stop training.
5. Save probe parameters (including normalizer state), per-epoch loss history,
   and final dev metrics.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from tqdm.auto import tqdm

from ..core.config import ProbeConfig, ProbeInputNormalization
from ..core.io import write_json
from .data import ProbeBatch, ProbeTrainingData, collate_probe_batch
from .metrics import spearman_correlation, undirected_unlabeled_attachment_score
from .normalization import ProbeInputNormalizer
from .structural import StructuralProbe, l1_distance_loss

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProbeEvaluation:
    """Per-layer evaluation metrics."""

    dev_loss: float
    dev_spearman: float
    dev_uuas: float


@dataclass(slots=True)
class ProbeTrainingHistory:
    """Per-epoch training/dev losses, plus reset-schedule events.

    Saved to disk per layer (``history.json``) so divergence can be diagnosed
    after the fact without re-running.
    """

    train_losses_per_epoch: list[float] = field(default_factory=list)
    dev_losses_per_epoch: list[float] = field(default_factory=list)
    learning_rate_per_epoch: list[float] = field(default_factory=list)
    reset_epochs: list[int] = field(default_factory=list)
    """1-indexed epochs at which the LR-decay reset was applied."""

    epochs_completed: int = 0


@dataclass(slots=True)
class ProbeTrainResult:
    """Output of training one probe at one layer."""

    layer_index: int
    probe_state: dict[str, torch.Tensor]
    """State dict of the trained probe (CPU tensors), including normalizer buffers."""

    evaluation: ProbeEvaluation
    history: ProbeTrainingHistory
    rank: int
    hidden_size: int
    input_normalization: ProbeInputNormalization


@dataclass(slots=True)
class LayerProbeBank:
    """A collection of trained probes, one per layer.

    Used at probe-application time to predict distances on stimuli. The bank
    records the normalizer kind so that fresh ``StructuralProbe`` modules can
    be constructed with the right module shape before loading state.
    """

    probes: dict[int, StructuralProbe]
    evaluations: dict[int, ProbeEvaluation]
    rank: int
    hidden_size: int
    input_normalization: ProbeInputNormalization

    def layer_indices(self) -> list[int]:
        return sorted(self.probes)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "rank": self.rank,
                "hidden_size": self.hidden_size,
                "input_normalization": self.input_normalization,
                "probes": {idx: probe.state_dict() for idx, probe in self.probes.items()},
                "evaluations": {
                    idx: {
                        "dev_loss": ev.dev_loss,
                        "dev_spearman": ev.dev_spearman,
                        "dev_uuas": ev.dev_uuas,
                    }
                    for idx, ev in self.evaluations.items()
                },
            },
            directory / "probe_bank.pt",
        )
        write_json(
            directory / "probe_bank_summary.json",
            {
                "rank": self.rank,
                "hidden_size": self.hidden_size,
                "input_normalization": self.input_normalization,
                "layers": [
                    {
                        "layer_index": idx,
                        "dev_loss": self.evaluations[idx].dev_loss,
                        "dev_spearman": self.evaluations[idx].dev_spearman,
                        "dev_uuas": self.evaluations[idx].dev_uuas,
                    }
                    for idx in self.layer_indices()
                ],
            },
        )

    @classmethod
    def load(cls, directory: Path) -> LayerProbeBank:
        payload = torch.load(directory / "probe_bank.pt", map_location="cpu", weights_only=True)
        rank = int(payload["rank"])
        hidden_size = int(payload["hidden_size"])
        # Older banks (pre-v3) don't carry the normalization kind. Default to
        # "none" for backward compatibility; a saved probe-bank summary will
        # tell us if that's wrong.
        input_normalization: ProbeInputNormalization = payload.get(
            "input_normalization", "none"
        )

        probes: dict[int, StructuralProbe] = {}
        for idx, state in payload["probes"].items():
            idx_int = int(idx)
            normalizer = ProbeInputNormalizer(
                kind=input_normalization, hidden_size=hidden_size
            )
            probe = StructuralProbe(
                hidden_size=hidden_size, rank=rank, normalizer=normalizer
            )
            probe.load_state_dict(state)
            probe.eval()
            probes[idx_int] = probe
        evaluations: dict[int, ProbeEvaluation] = {}
        for idx, ev in payload["evaluations"].items():
            evaluations[int(idx)] = ProbeEvaluation(
                dev_loss=ev["dev_loss"],
                dev_spearman=ev["dev_spearman"],
                dev_uuas=ev["dev_uuas"],
            )
        return cls(
            probes=probes,
            evaluations=evaluations,
            rank=rank,
            hidden_size=hidden_size,
            input_normalization=input_normalization,
        )


def train_probe(
    *,
    layer_index: int,
    train_data: ProbeTrainingData,
    dev_data: ProbeTrainingData,
    config: ProbeConfig,
    seed: int,
    device: torch.device | str | None = None,
    normalizer: ProbeInputNormalizer | None = None,
) -> ProbeTrainResult:
    """Train a single structural probe on one layer's activations.

    Args:
        layer_index: layer being trained (used only for logging).
        train_data, dev_data: per-layer training and dev data.
        config: probe hyperparameters, including ``input_normalization`` and the
            LR-reset-on-plateau schedule.
        seed: per-layer random seed.
        device: optional torch device.
        normalizer: optional pre-built normalizer. If ``None``, one is built
            from ``config.input_normalization`` with no corpus stats; the caller
            is responsible for setting stats if the kind requires them. Pass an
            already-configured normalizer (e.g., with corpus stats from cache)
            to ensure reproducibility.

    Returns the trained state dict, dev evaluation, and training history.
    """
    torch_device = torch.device(device) if device else _default_device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    if normalizer is None:
        normalizer = ProbeInputNormalizer(
            kind=config.input_normalization, hidden_size=train_data.hidden_size
        )
    if normalizer.hidden_size != train_data.hidden_size:
        raise ValueError(
            f"Normalizer hidden_size={normalizer.hidden_size} does not match "
            f"train_data.hidden_size={train_data.hidden_size}."
        )
    probe = StructuralProbe(
        hidden_size=train_data.hidden_size, rank=config.rank, normalizer=normalizer
    ).to(torch_device)

    current_lr = config.learning_rate
    optimizer = torch.optim.Adam(probe.parameters(), lr=current_lr)

    history = ProbeTrainingHistory()
    best_dev_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {
        k: v.detach().clone().cpu() for k, v in probe.state_dict().items()
    }

    epochs_without_improvement = 0
    resets_done = 0

    train_indices = list(range(len(train_data)))

    for epoch in range(1, config.epochs + 1):
        # Train one epoch.
        probe.train()
        rng.shuffle(train_indices)
        train_losses: list[float] = []
        for batch_indices in _batched_indices(train_indices, config.batch_size):
            batch = collate_probe_batch(batch_indices, train_data).to(torch_device)
            predicted = probe(batch.activations)
            loss = l1_distance_loss(predicted, batch.gold_distances, batch.lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.train_losses_per_epoch.append(train_loss)

        # Evaluate dev loss.
        dev_loss = _compute_dev_loss(probe, dev_data, config.batch_size, torch_device)
        history.dev_losses_per_epoch.append(dev_loss)
        history.learning_rate_per_epoch.append(current_lr)
        history.epochs_completed = epoch

        logger.info(
            "Layer %d epoch %d/%d  lr=%.1e  train_loss=%.4f  dev_loss=%.4f",
            layer_index,
            epoch,
            config.epochs,
            current_lr,
            train_loss,
            dev_loss,
        )

        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss = dev_loss
            best_state = {
                k: v.detach().clone().cpu() for k, v in probe.state_dict().items()
            }
            epochs_without_improvement = 0
            continue

        # No improvement.
        epochs_without_improvement += 1
        if epochs_without_improvement < config.lr_decay_patience:
            continue

        # Time to (try to) reset.
        if resets_done >= config.lr_decay_max_resets:
            logger.info(
                "Layer %d: stopping after %d resets without improvement (max=%d).",
                layer_index,
                resets_done,
                config.lr_decay_max_resets,
            )
            break

        resets_done += 1
        current_lr = current_lr * config.lr_decay_factor
        # Re-create the optimizer to drop momentum (H&M's "no momentum terms are kept").
        optimizer = torch.optim.Adam(probe.parameters(), lr=current_lr)
        epochs_without_improvement = 0
        history.reset_epochs.append(epoch)
        logger.info(
            "Layer %d: LR reset %d/%d at epoch %d, new lr=%.1e",
            layer_index,
            resets_done,
            config.lr_decay_max_resets,
            epoch,
            current_lr,
        )

    # Restore best state and compute final dev metrics.
    probe.load_state_dict(best_state)
    dev_predictions = _predict_distance_matrices(
        probe, dev_data, config.batch_size, torch_device
    )
    evaluation = ProbeEvaluation(
        dev_loss=best_dev_loss,
        dev_spearman=spearman_correlation(dev_predictions, dev_data.gold_distances),
        dev_uuas=undirected_unlabeled_attachment_score(
            dev_predictions, dev_data.gold_distances
        ),
    )

    return ProbeTrainResult(
        layer_index=layer_index,
        probe_state=best_state,
        evaluation=evaluation,
        history=history,
        rank=config.rank,
        hidden_size=train_data.hidden_size,
        input_normalization=config.input_normalization,
    )


def train_layer_probe_bank(
    *,
    layer_indices: list[int],
    train_data_per_layer: dict[int, ProbeTrainingData],
    dev_data_per_layer: dict[int, ProbeTrainingData],
    config: ProbeConfig,
    seed: int,
    device: torch.device | str | None = None,
    progress: bool = True,
    normalizers_per_layer: dict[int, ProbeInputNormalizer] | None = None,
    history_dir: Path | None = None,
) -> LayerProbeBank:
    """Train one probe per layer, returning a `LayerProbeBank`.

    Each layer gets its own probe with a per-layer seed (`seed + layer_index`)
    so that retraining individual layers is reproducible.

    Args:
        layer_indices: which layers to train.
        train_data_per_layer, dev_data_per_layer: per-layer data.
        config: probe hyperparameters.
        seed: base seed; per-layer seed = ``seed + layer_index``.
        device: optional torch device.
        progress: show a tqdm progress bar over layers.
        normalizers_per_layer: optional per-layer pre-built normalizers (e.g.,
            with corpus stats already loaded). If unset, normalizers are built
            from ``config.input_normalization`` per layer; for normalization
            kinds that require corpus stats this will fail at training time.
        history_dir: if provided, write per-layer training history to
            ``history_dir/layer_{idx:03d}.history.json``.
    """
    probes: dict[int, StructuralProbe] = {}
    evaluations: dict[int, ProbeEvaluation] = {}
    iterator = (
        tqdm(layer_indices, desc="training probes", unit="layer")
        if progress
        else iter(layer_indices)
    )

    rank = config.rank
    hidden_size: int | None = None

    if history_dir is not None:
        history_dir.mkdir(parents=True, exist_ok=True)

    for layer_index in iterator:
        train_data = train_data_per_layer[layer_index]
        dev_data = dev_data_per_layer[layer_index]
        normalizer = (
            normalizers_per_layer.get(layer_index)
            if normalizers_per_layer is not None
            else None
        )
        result = train_probe(
            layer_index=layer_index,
            train_data=train_data,
            dev_data=dev_data,
            config=config,
            seed=seed + layer_index,
            device=device,
            normalizer=normalizer,
        )

        # Persist per-layer training history.
        if history_dir is not None:
            write_json(
                history_dir / f"layer_{layer_index:03d}.history.json",
                {
                    "layer_index": layer_index,
                    "epochs_completed": result.history.epochs_completed,
                    "train_losses_per_epoch": result.history.train_losses_per_epoch,
                    "dev_losses_per_epoch": result.history.dev_losses_per_epoch,
                    "learning_rate_per_epoch": result.history.learning_rate_per_epoch,
                    "reset_epochs": result.history.reset_epochs,
                },
            )

        # Re-instantiate a probe with the right normalizer kind so we can load
        # the trained state dict. The normalizer's corpus stats are inside the
        # state dict so this round-trips correctly.
        load_normalizer = ProbeInputNormalizer(
            kind=result.input_normalization, hidden_size=result.hidden_size
        )
        probe = StructuralProbe(
            hidden_size=result.hidden_size,
            rank=result.rank,
            normalizer=load_normalizer,
        )
        probe.load_state_dict(result.probe_state)
        probe.eval()
        probes[layer_index] = probe
        evaluations[layer_index] = result.evaluation
        hidden_size = result.hidden_size

    if hidden_size is None:
        raise RuntimeError("No layers were trained")

    return LayerProbeBank(
        probes=probes,
        evaluations=evaluations,
        rank=rank,
        hidden_size=hidden_size,
        input_normalization=config.input_normalization,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _batched_indices(indices: list[int], batch_size: int) -> list[list[int]]:
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def _compute_dev_loss(
    probe: nn.Module,
    dev_data: ProbeTrainingData,
    batch_size: int,
    device: torch.device,
) -> float:
    probe.eval()
    indices = list(range(len(dev_data)))
    losses: list[float] = []
    for batch_indices in _batched_indices(indices, batch_size):
        batch: ProbeBatch = collate_probe_batch(batch_indices, dev_data).to(device)
        predicted = probe(batch.activations)
        loss = l1_distance_loss(predicted, batch.gold_distances, batch.lengths)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.inference_mode()
def _predict_distance_matrices(
    probe: nn.Module,
    data: ProbeTrainingData,
    batch_size: int,
    device: torch.device,
) -> list[NDArray[np.float32]]:
    """Predict pairwise distances for every sentence; returns one matrix per sentence."""
    probe.eval()
    predictions: list[NDArray[np.float32]] = []
    indices = list(range(len(data)))
    for batch_indices in _batched_indices(indices, batch_size):
        batch = collate_probe_batch(batch_indices, data).to(device)
        predicted = probe(batch.activations).detach().cpu().numpy()  # (B, max_len, max_len)
        for batch_position, sent_index in enumerate(batch_indices):
            n = data.activations[sent_index].shape[0]
            predictions.append(predicted[batch_position, :n, :n].astype(np.float32))
    return predictions


__all__ = [
    "LayerProbeBank",
    "ProbeEvaluation",
    "ProbeTrainResult",
    "ProbeTrainingHistory",
    "train_layer_probe_bank",
    "train_probe",
]
