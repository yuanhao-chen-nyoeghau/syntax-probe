"""Shared utilities for the wh-extraction and c-command analysis scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from figures._runs import Model

logger = logging.getLogger("analyses._common")


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _run_id_timestamp(p: Path, prefix: str) -> str:
    """Sortable timestamp from a run-directory name.

    Run dirs are named ``{prefix}{config_hash}-{date}-{time}``. The trailing
    ``{date}-{time}`` is lexicographically sortable. Falls back to mtime for
    directories that don't match the expected layout.
    """
    suffix = p.name[len(prefix):] if p.name.startswith(prefix) else p.name
    parts = suffix.split("-")
    if len(parts) >= 3:
        return f"{parts[-2]}-{parts[-1]}"
    return str(p.stat().st_mtime)


def newest_nonempty_run(
    parent: Path, prefix: str, required_artifact: str,
) -> Path | None:
    """Return the newest run dir under ``parent`` that starts with ``prefix``
    and that contains ``required_artifact`` (relative path).

    Returns ``None`` if no qualifying directory exists.
    """
    if not parent.is_dir():
        return None
    candidates = sorted(
        (p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: _run_id_timestamp(p, prefix),
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / required_artifact).exists():
            return candidate
    return None


def discover_runs(
    results_dir: Path,
    *,
    apply_prefix: str,
    verify_prefix: str,
    apply_run: Path | None = None,
    verify_run: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve apply_probes and verify_stimuli run directories.

    Explicit overrides win; otherwise the newest matching non-empty run is
    used. Raises ``FileNotFoundError`` if a run can't be located.
    """
    if apply_run is None:
        apply_run = newest_nonempty_run(
            results_dir / "apply_probes",
            prefix=apply_prefix,
            required_artifact="artifacts/per_stimulus_predictions.jsonl",
        )
        if apply_run is None:
            raise FileNotFoundError(
                f"No apply_probes run with prefix {apply_prefix!r} found under "
                f"{results_dir / 'apply_probes'}. Pass --apply-run explicitly."
            )
        logger.info("Auto-discovered apply_probes run: %s", apply_run.name)
    elif not (apply_run / "artifacts/per_stimulus_predictions.jsonl").exists():
        raise FileNotFoundError(
            f"--apply-run {apply_run} does not contain "
            f"artifacts/per_stimulus_predictions.jsonl."
        )

    if verify_run is None:
        verify_run = newest_nonempty_run(
            results_dir / "verify_stimuli",
            prefix=verify_prefix,
            required_artifact="artifacts/verification_report.json",
        )
        if verify_run is None:
            raise FileNotFoundError(
                f"No verify_stimuli run with prefix {verify_prefix!r} found "
                f"under {results_dir / 'verify_stimuli'}. Pass --verify-run "
                f"explicitly."
            )
        logger.info("Auto-discovered verify_stimuli run: %s", verify_run.name)
    elif not (verify_run / "artifacts/verification_report.json").exists():
        raise FileNotFoundError(
            f"--verify-run {verify_run} does not contain "
            f"artifacts/verification_report.json."
        )

    return apply_run, verify_run


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions(apply_run: Path, *, expected_columns: set[str]) -> pd.DataFrame:
    """Load ``per_stimulus_predictions.jsonl`` and verify its schema."""
    path = apply_run / "artifacts" / "per_stimulus_predictions.jsonl"
    df = pd.read_json(path, lines=True)
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"per_stimulus_predictions.jsonl missing columns: {missing}"
        )
    return df


def load_excluded_items(verify_run: Path) -> set[str]:
    """Item ids whose verifier ``ok=False`` (within-template UD spread above
    the configured tolerance). These items are excluded from model fits to
    avoid conflating probe signal with parser-instability signal.
    """
    path = verify_run / "artifacts" / "verification_report.json"
    report = json.loads(path.read_text())
    return {row["item_id"] for row in report["rows"] if not row["ok"]}


# ---------------------------------------------------------------------------
# Fractional layer depth
# ---------------------------------------------------------------------------


def add_fractional_depth(
    df: pd.DataFrame, *, layer_col: str = "layer_index",
) -> pd.DataFrame:
    """Add a ``fractional_depth`` column scaled to [0, 1] over the observed
    layers. ``layer_index = max`` becomes 1.0; ``layer_index = 0`` becomes 0.0.

    Cross-model plots can then put fractional_depth on the x-axis to compare
    models with different layer counts on the same horizontal scale (the
    convention from Tenney et al. 2019 and most subsequent probing work).
    """
    if df.empty:
        df = df.copy()
        df["fractional_depth"] = pd.Series(dtype=float)
        return df
    out = df.copy()
    max_layer = int(out[layer_col].max())
    if max_layer == 0:
        out["fractional_depth"] = 0.0
    else:
        out["fractional_depth"] = out[layer_col].astype(float) / max_layer
    return out


# ---------------------------------------------------------------------------
# Multiple-comparisons correction
# ---------------------------------------------------------------------------


def add_corrections(
    df: pd.DataFrame,
    *,
    p_col: str = "p_value",
    bonf_col: str = "p_bonferroni",
    fdr_col: str = "p_fdr_bh",
    test_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply Bonferroni and BH-FDR correction to ``df[p_col]``, jointly over
    all rows where ``test_mask`` is True AND ``p_col`` is non-NaN.

    Returns a copy of ``df`` with two added columns. Rows get NaN for the
    corrected columns in two cases:

    * ``test_mask`` is False (e.g., the row is a model intercept that we
      report but don't subject to multiple-comparisons correction).
    * ``p_col`` is NaN. This happens at e.g. layer 0 cells where the
      contrast has identically-zero variance — the SE is undefined, so
      the test statistic and p-value are NaN. We mask those rows out of
      ``multipletests`` rather than passing NaN through, because
      ``multipletests(method="fdr_bh")`` propagates NaN to every output
      whenever any input is NaN — silently turning the entire FDR column
      into NaN. Bonferroni handles NaN element-wise but is also slightly
      anti-conservative if we count NaN rows in the multiplier.

    The correction count (Bonferroni multiplier; BH ranking depth) is the
    number of NON-NaN p-values being corrected, not the total row count.
    """
    if df.empty:
        out = df.copy()
        out[bonf_col] = pd.Series(dtype=float)
        out[fdr_col] = pd.Series(dtype=float)
        return out
    out = df.copy()
    if test_mask is None:
        test_mask = pd.Series(True, index=out.index)
    # Only correct rows that are both (a) in the test set and (b) have a
    # finite p-value to correct. NaN p-values stay NaN in the output.
    valid_mask = test_mask & out[p_col].notna()
    test_p = out.loc[valid_mask, p_col].to_numpy()
    bonf = np.full(len(out), np.nan, dtype=float)
    fdr  = np.full(len(out), np.nan, dtype=float)
    if test_p.size > 0:
        _, p_bonf, _, _ = multipletests(test_p, method="bonferroni")
        _, p_fdr, _, _  = multipletests(test_p, method="fdr_bh")
        bonf[valid_mask.to_numpy()] = p_bonf
        fdr[valid_mask.to_numpy()]  = p_fdr
    out[bonf_col] = bonf
    out[fdr_col]  = fdr
    return out


# ---------------------------------------------------------------------------
# Cluster bootstrap for descriptive effect sizes
# ---------------------------------------------------------------------------


def cluster_bootstrap_diff(
    df: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str,
    condition_col: str,
    treatment_value: str,
    reference_value: str,
    n_resamples: int = 1000,
    rng_seed: int = 13,
) -> dict[str, float]:
    """Bootstrap the ``treatment − reference`` mean difference, resampling
    clusters (typically ``item_id``) with replacement.

    For each bootstrap iteration:
      1. Resample cluster ids (``item_id``) with replacement.
      2. Take all rows belonging to those clusters.
      3. Compute mean(value | treatment) − mean(value | reference).

    Returns ``{mean_diff, ci_low, ci_high, n_clusters, n_obs}`` where the
    point estimate is the mean of the bootstrap distribution and the CI is
    the empirical 2.5th–97.5th percentile (basic percentile method, the most
    commonly reported form of bootstrap CI).
    """
    if df.empty or df[condition_col].nunique() < 2:
        return {
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_clusters": 0,
            "n_obs": 0,
        }

    # Cache cluster-indexed data once; resampling then becomes a numpy
    # operation rather than repeated groupby calls.
    cluster_ids = df[cluster_col].unique()
    n_clusters = len(cluster_ids)
    cluster_to_rows: dict[str, np.ndarray] = {
        cid: np.where(df[cluster_col].to_numpy() == cid)[0]
        for cid in cluster_ids
    }
    values = df[value_col].to_numpy()
    conditions = df[condition_col].to_numpy()

    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = rng.choice(cluster_ids, size=n_clusters, replace=True)
        # Concatenate row indices for sampled clusters.
        idx = np.concatenate([cluster_to_rows[c] for c in sampled])
        v = values[idx]
        c = conditions[idx]
        treat = v[c == treatment_value]
        ref   = v[c == reference_value]
        if treat.size == 0 or ref.size == 0:
            diffs[i] = np.nan
        else:
            diffs[i] = treat.mean() - ref.mean()

    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        mean_diff = ci_low = ci_high = float("nan")
    else:
        mean_diff = float(diffs.mean())
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        ci_low = float(ci_low)
        ci_high = float(ci_high)

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_clusters": int(n_clusters),
        "n_obs": int(len(df)),
    }


def bootstrap_per_layer_pair(
    df: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str,
    condition_col: str,
    treatment_value: str,
    reference_value: str,
    layer_col: str = "layer_index",
    pair_col: str | None = None,
    extra_group_cols: Iterable[str] = (),
    n_resamples: int = 1000,
    rng_seed: int = 13,
) -> pd.DataFrame:
    """Run ``cluster_bootstrap_diff`` per (layer × pair × extras) and return
    a long-form DataFrame.

    ``pair_col`` is optional; if omitted, layers are not split by pair.
    Extra groupby columns can be passed via ``extra_group_cols`` (e.g.,
    ``modifier_type`` for the reflexive sub-experiment).
    """
    group_cols = [layer_col]
    if pair_col is not None:
        group_cols.append(pair_col)
    group_cols.extend(extra_group_cols)

    rows: list[dict] = []
    # Use a fixed seed sequence so each (layer, pair) combination has its
    # own deterministic, reproducible bootstrap. We derive a per-group seed
    # by hashing the group keys via SHA-256 (Python's built-in `hash` is
    # randomized per process, so it can't be used here).
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        digest = hashlib.sha256(
            f"{rng_seed}|{'|'.join(str(k) for k in keys)}".encode()
        ).digest()
        seed_int = int.from_bytes(digest[:4], "big")
        out = cluster_bootstrap_diff(
            sub,
            value_col=value_col,
            cluster_col=cluster_col,
            condition_col=condition_col,
            treatment_value=treatment_value,
            reference_value=reference_value,
            n_resamples=n_resamples,
            rng_seed=seed_int,
        )
        row: dict[str, object] = dict(zip(group_cols, keys, strict=True))
        row["treatment"] = treatment_value
        row["reference"] = reference_value
        row.update(out)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inferential fits — OLS+CR1 (default) or MixedLM with fallback
# ---------------------------------------------------------------------------

# Valid arguments for the ``method`` parameter of ``fit_clustered_regression``
# and the ``--fit-method`` CLI flag of the two stats scripts.
FIT_METHODS = ("ols_cr1", "mixedlm")


def fit_clustered_regression(
    formula: str,
    data: pd.DataFrame,
    *,
    group_col: str,
    method: str = "ols_cr1",
) -> tuple[object | None, str]:
    """Fit ``formula`` to ``data`` using one of two methods that handle
    within-cluster correlation, and return ``(result, fit_method_used)``.

    Methods
    -------
    ``"ols_cr1"`` (default)
        OLS with cluster-robust standard errors (CR1) clustered on
        ``group_col``. Distribution-free, asymptotically valid for any
        pattern of within-cluster correlation, numerically stable (no
        boundary singularities possible), and faster than MixedLM. Valid
        when the number of clusters is ≥ 30-50 (Cameron & Miller 2015);
        our designs use ≥ 250 items per cell, comfortably above threshold.

    ``"mixedlm"``
        Mixed-effects model with random intercept on ``group_col``, fit by
        REML. Smaller standard errors than CR1 *when the random-intercept
        variance is genuinely positive and the normality assumptions hold*.
        statsmodels raises ``LinAlgError("Singular matrix")`` when REML
        pushes the variance to the boundary at zero, which happens
        frequently in low-noise probe data. On that error we fall back to
        ``"ols_cr1"`` automatically — the limit of MixedLM as variance → 0
        is OLS, so the fallback is internally consistent. ``UserWarning``
        from statsmodels (the verbose "Random effects covariance is
        singular" message) is suppressed inside the helper since we handle
        the singularity via the exception path.

    Returns
    -------
    ``(result, fit_method_used)`` where ``result`` exposes ``.params``,
    ``.bse``, ``.tvalues``, and ``.pvalues``. ``fit_method_used`` is one of
    ``"mixedlm"``, ``"ols_cr1"``, or ``"failed"``; in the last case
    ``result`` is ``None``.

    statsmodels' OLS+CR1 result names them ``.tvalues`` despite using
    cluster-robust SEs and t-distribution-based p-values; with our cluster
    counts (≥ 250) the t and z distributions are essentially identical, so
    downstream code can treat them uniformly as z-statistics.

    Reference: Cameron, A. C., & Miller, D. L. (2015). A Practitioner's
    Guide to Cluster-Robust Inference. Journal of Human Resources 50(2),
    317-372.
    """
    if method not in FIT_METHODS:
        raise ValueError(
            f"method must be one of {FIT_METHODS}; got {method!r}"
        )

    if method == "mixedlm":
        # Try MixedLM first. The "Random effects covariance is singular"
        # UserWarning is statsmodels' way of saying REML pushed the random-
        # intercept variance to the boundary; we handle this by catching the
        # subsequent LinAlgError and falling back to OLS+CR1, so the warning
        # itself is just noise in the log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                model = smf.mixedlm(
                    formula, data=data, groups=data[group_col],
                )
                result = model.fit(method="lbfgs", reml=True, disp=False)
                return result, "mixedlm"
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.debug(
                    "MixedLM failed (%s); falling back to OLS+CR1", exc,
                )

    # OLS + cluster-robust SEs path.
    # Used either as the explicit default or as the MixedLM fallback above.
    try:
        model = smf.ols(formula, data=data)
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": data[group_col]},
        )
        return result, "ols_cr1"
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("OLS+CR1 fit failed: %s", exc)
        return None, "failed"


# ---------------------------------------------------------------------------
# Headline helpers
# ---------------------------------------------------------------------------


def find_first_significant_layer(
    df: pd.DataFrame,
    *,
    p_col: str,
    threshold: float = 0.05,
    layer_col: str = "layer_index",
) -> int | None:
    """Smallest ``layer_index`` at which ``df[p_col] < threshold``.

    Returns ``None`` if no such layer exists. Useful for "the effect emerges
    at layer K" annotations on plots.
    """
    if df.empty:
        return None
    sig = df[df[p_col] < threshold]
    if sig.empty:
        return None
    return int(sig[layer_col].min())


def peak_row(
    df: pd.DataFrame,
    *,
    by_col: str = "z_value",
    descending: bool = True,
) -> dict | None:
    """Return the row maximizing ``by_col`` as a plain dict, or ``None`` if
    the table is empty. Convenience helper for summary JSON construction.
    """
    if df.empty:
        return None
    sorted_df = df.sort_values(by_col, ascending=not descending)
    row = sorted_df.iloc[0]
    return {k: _to_python(v) for k, v in row.items()}


def _to_python(v: object) -> object:
    """Convert numpy scalars to Python scalars for JSON serialization."""
    # Check bool BEFORE integer: numpy bool is a numpy scalar but NOT a Python
    # bool, and json.dumps rejects it. (In some numpy versions np.bool_ also
    # passes isinstance(_, np.integer); we order the check defensively.)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ---------------------------------------------------------------------------
# Multi-model discovery helpers
# ---------------------------------------------------------------------------
# The model registry lives in figures/_runs.py (single source of truth shared
# with the figure-generation scripts). The stats scripts import it directly;
# Model is referenced via TYPE_CHECKING for static type info without forcing
# a runtime import here in _common.


def discover_apply_run_for_model(
    model: Model,
    experiment: str,
    results_dir: Path,
) -> Path | None:
    """Return the newest qualifying apply_probes run for *(model, experiment)*.

    ``experiment`` is ``"wh_extraction"`` or ``"c_command"``. Returns ``None``
    if no matching run exists yet (caller should skip, not error).
    """
    prefix = f"{experiment}_{model.slug}-"
    return newest_nonempty_run(
        results_dir / "apply_probes",
        prefix=prefix,
        required_artifact="artifacts/per_stimulus_predictions.jsonl",
    )


def add_model_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--all-models`` / ``--include`` / ``--fail-fast`` to a stats script
    parser. These flags activate multi-model batch mode in both stats scripts.

    When *neither* flag is given, the script runs in single-model mode
    (existing behaviour, unchanged).

    ``--all-models`` includes every entry in the figures registry (including
    those with ``default=False``). ``--include`` restricts to the listed slugs.
    The two flags are mutually exclusive.

    ``--fail-fast`` aborts the batch on the first model that fails. The default
    is to continue and report all failures at the end (useful since models are
    independent — one bad run shouldn't prevent the others from being analysed).
    """
    g = parser.add_argument_group("multi-model batch")
    exc = g.add_mutually_exclusive_group()
    exc.add_argument(
        "--all-models", action="store_true", dest="all_models", default=False,
        help="Analyse every model in the registry that has an apply_probes run.",
    )
    exc.add_argument(
        "--include", nargs="+", metavar="SLUG", default=None,
        help="Analyse only these model slugs (space-separated).",
    )
    g.add_argument(
        "--fail-fast", action="store_true", default=False,
        help="Abort batch on first failure (default: continue and summarise).",
    )


__all__ = [
    "FIT_METHODS",
    "add_corrections",
    "add_fractional_depth",
    "add_model_selection_args",
    "bootstrap_per_layer_pair",
    "cluster_bootstrap_diff",
    "discover_apply_run_for_model",
    "discover_runs",
    "find_first_significant_layer",
    "fit_clustered_regression",
    "load_excluded_items",
    "load_predictions",
    "newest_nonempty_run",
    "peak_row",
]
