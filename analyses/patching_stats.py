"""Statistical analysis of activation-patching outputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from figures._runs import select_models

from ._common import (
    FIT_METHODS,
    add_corrections,
    add_fractional_depth,
    add_model_selection_args,
    fit_clustered_regression,
    newest_nonempty_run,
    peak_row,
)

logger = logging.getLogger("patching_stats")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APPLY_RUN_PARENT = "activation_patching"

# Variant → stimuli-kind prefix used in run-dir names. Patching run dirs are
# named ``{stimuli_prefix}_{variant}_{slug}-{config_hash}-{date}-{time}``.
_VARIANT_TO_STIMULI_PREFIX = {
    "w2": "wh",
    "w4": "wh",
    "n1": "cc",
}

# Mapping from role-name (in ``intervention_role``) to its short token in
# pair labels (``wh-esubj``, ``esubj-evb``, ``anaphor-subject``). Used to
# decide whether a cell is role-relevant at ``L_measure = L_patch``.
_ROLE_SHORT: dict[str, str] = {
    "wh": "wh",
    "embedded_subject": "esubj",
    "embedded_verb": "evb",
    "anaphor": "anaphor",
    "subject": "subject",
    "modifier": "modifier",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_per_trial(apply_run: Path) -> pd.DataFrame:
    """Load ``per_trial_predictions.jsonl`` and lift metadata fields out
    of the nested ``metadata`` dict into top-level columns."""
    path = apply_run / "artifacts" / "per_trial_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty.")
    df = pd.DataFrame(rows)
    for key in ("source_condition", "target_condition",
                "source_gender_config", "target_gender_config",
                "modifier_type"):
        df[key] = df["metadata"].apply(lambda meta, k=key: meta.get(k))
    # Unify: cc trials use *_gender_config; wh trials use *_condition. Reduce
    # to a single pair of columns so the downstream code is variant-agnostic.
    df["source_condition"] = df["source_condition"].fillna(df["source_gender_config"])
    df["target_condition"] = df["target_condition"].fillna(df["target_gender_config"])
    return df


def explode_to_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Expand per-trial rows to one row per (trial × pair × layer) with a
    ``delta_beta = patched_distance − unpatched_distance`` column.

    Only layers that appear in both ``patched_distances`` and
    ``unpatched_distances`` produce rows.
    """
    rows: list[dict] = []
    for r in df.itertuples():
        for pair, by_layer in r.patched_distances.items():
            for layer_str, patched in by_layer.items():
                try:
                    unpatched = r.unpatched_distances[pair][layer_str]
                except KeyError:
                    continue
                rows.append({
                    "trial_id": r.trial_id,
                    "experiment_kind": r.experiment_kind,
                    "item_id": r.item_id,
                    "intervention_role": r.intervention_role,
                    "intervention_layer": int(r.intervention_layer),
                    "measurement_layer": int(layer_str),
                    "pair_label": pair,
                    "source_condition": r.source_condition,
                    "target_condition": r.target_condition,
                    # ``modifier_type`` is None for wh-extraction trials
                    # (W2/W4) and populated for c-command trials (N1).
                    # Preserved so a downstream filter can subset N1 by
                    # modifier configuration (see --modifier-type flag in
                    # the CLI).
                    "modifier_type": getattr(r, "modifier_type", None),
                    "patched_distance": float(patched),
                    "unpatched_distance": float(unpatched),
                    "delta_beta": float(patched) - float(unpatched),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One-sample bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_one_sample_mean(
    values: np.ndarray, *, n_resamples: int, rng_seed: int,
) -> dict[str, float]:
    """Single-sample non-parametric bootstrap on the mean of ``values``.

    Returns ``{mean, ci_low, ci_high}`` with a basic percentile 95% CI.
    No within-cell clustering is needed because each item contributes
    one observation per cell by construction.
    """
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    boot_means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
    }


# ---------------------------------------------------------------------------
# Per-cell fit
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    experiment_kind: str
    intervention_role: str
    source_condition: str | None
    target_condition: str | None
    intervention_layer: int
    measurement_layer: int
    pair_label: str
    n_obs: int
    n_items: int
    estimate: float          # mean Δβ (regression intercept)
    std_err: float
    z_value: float
    p_value: float
    converged: bool
    fit_method: str
    boot_mean: float
    boot_ci_low: float
    boot_ci_high: float
    is_role_relevant: bool   # patched role is an endpoint of pair_label


def _is_role_relevant(intervention_role: str, pair_label: str) -> bool:
    """Whether the patched role appears as an endpoint of ``pair_label``."""
    role_short = _ROLE_SHORT.get(intervention_role)
    if role_short is None:
        return True  # unknown role: don't pre-filter; let the data speak
    return role_short in pair_label.split("-")


def fit_one_cell(
    cell_df: pd.DataFrame,
    *,
    intervention_role: str,
    source_condition: str | None,
    target_condition: str | None,
    intervention_layer: int,
    measurement_layer: int,
    pair_label: str,
    fit_method: str,
    n_bootstrap: int,
    rng_seed: int,
) -> CellResult | None:
    """Fit ``delta_beta ~ 1`` on a single cell and bootstrap its mean.

    ``cell_df`` must already be filtered to the rows of this cell. Caller
    handles selection so this helper stays slice-agnostic.
    """
    if cell_df.empty:
        return None
    result, used_method = fit_clustered_regression(
        "delta_beta ~ 1", cell_df, group_col="item_id", method=fit_method,
    )
    if result is None or "Intercept" not in result.params.index:
        logger.warning(
            "Fit failed for cell role=%s src=%s tgt=%s L=%d pair=%s",
            intervention_role, source_condition, target_condition,
            intervention_layer, pair_label,
        )
        return None
    boot = _bootstrap_one_sample_mean(
        cell_df["delta_beta"].to_numpy(),
        n_resamples=n_bootstrap, rng_seed=rng_seed,
    )
    return CellResult(
        experiment_kind=str(cell_df["experiment_kind"].iloc[0]),
        intervention_role=intervention_role,
        source_condition=source_condition,
        target_condition=target_condition,
        intervention_layer=intervention_layer,
        measurement_layer=measurement_layer,
        pair_label=pair_label,
        n_obs=int(len(cell_df)),
        n_items=int(cell_df["item_id"].nunique()),
        estimate=float(result.params["Intercept"]),
        std_err=float(result.bse["Intercept"]),
        z_value=float(result.tvalues["Intercept"]),
        p_value=float(result.pvalues["Intercept"]),
        converged=bool(getattr(result, "converged", True)),
        fit_method=used_method,
        boot_mean=boot["mean"],
        boot_ci_low=boot["ci_low"],
        boot_ci_high=boot["ci_high"],
        is_role_relevant=_is_role_relevant(intervention_role, pair_label),
    )


def fit_all_cells(
    df: pd.DataFrame, *, fit_method: str, n_bootstrap: int, rng_seed: int,
) -> pd.DataFrame:
    """Iterate cells; return tidy DataFrame with BH-FDR/Bonferroni applied
    over role-relevant cells only (see module docstring)."""
    cell_cols = [
        "intervention_role", "source_condition", "target_condition",
        "intervention_layer", "measurement_layer", "pair_label",
    ]
    results: list[CellResult] = []
    grouped = df.groupby(cell_cols, dropna=False, sort=True)
    bar = tqdm(grouped, total=grouped.ngroups, desc="patching", unit="cell")
    for keys, sub in bar:
        (role, src, tgt, int_layer, meas_layer, pair) = keys
        r = fit_one_cell(
            sub,
            intervention_role=str(role),
            source_condition=None if pd.isna(src) else str(src),
            target_condition=None if pd.isna(tgt) else str(tgt),
            intervention_layer=int(int_layer),
            measurement_layer=int(meas_layer),
            pair_label=str(pair),
            fit_method=fit_method,
            n_bootstrap=n_bootstrap,
            rng_seed=rng_seed,
        )
        if r is not None:
            results.append(r)

    out = pd.DataFrame([asdict(r) for r in results])
    if out.empty:
        return out
    out = add_corrections(out, test_mask=out["is_role_relevant"])
    return add_fractional_depth(out, layer_col="intervention_layer")


# ---------------------------------------------------------------------------
# Sanity-pattern checks
# ---------------------------------------------------------------------------


def _check_predicted_pattern(stats: pd.DataFrame) -> tuple[list[str], dict]:
    """Advisory checks against design-level expectations.

    Returns ``(warnings, diagnostics)``. Warnings flag departures from
    expected patterns but never block the run.

    Patterns checked:

    1. **Negative control on wh cells**. For W2, patches at the ``wh``
       position with ``L_patch = L_measure`` should produce Δβ ≈ 0
       because the wh-token's residual depends only on the BOS+wh
       prefix, identical across source and target. A large effect here
       suggests an alignment/infrastructure issue rather than a phase
       finding.

    2. **W2 phase-count ordering on embedded_subject**. The finite-bare
       contrast should produce a strictly larger |Δβ| than the
       infinitival-bare contrast on the ``wh-esubj`` pair if the model
       encodes the predicted phase-count gradient.
    """
    warnings: list[str] = []
    diagnostics: dict[str, object] = {}

    role_col = "intervention_role"
    if role_col not in stats.columns or stats.empty:
        return warnings, diagnostics

    # (1) Negative control.
    wh_mask = (stats[role_col] == "wh") & stats["is_role_relevant"]
    wh_cells = stats[wh_mask]
    if not wh_cells.empty:
        worst = wh_cells.assign(abs_beta=wh_cells["estimate"].abs()) \
            .sort_values("abs_beta", ascending=False).iloc[0]
        diagnostics["wh_negative_control_max_abs_beta"] = float(worst["abs_beta"])
        # Heuristic threshold: probe distances in this codebase are on a
        # scale where ~0.05 is well outside the bf16/batch-composition
        # noise floor (~0.02-0.03) but small enough not to flag a real
        # effect on its own.
        if worst["abs_beta"] >= 0.05:
            cell = (worst["source_condition"], worst["target_condition"],
                    worst["pair_label"])
            warnings.append(
                f"Negative-control wh-cell {cell} shows "
                f"|Δβ|={worst['abs_beta']:.3f} (expected near 0 by "
                f"causal attention on BOS+wh prefix). Check tokenizer "
                f"alignment if this is structurally surprising."
            )

    # (2) Phase-count ordering on (embedded_subject, *, bare) wh-esubj.
    esubj_wh = stats[
        (stats[role_col] == "embedded_subject")
        & (stats["pair_label"] == "wh-esubj")
        & (stats["target_condition"] == "bare")
    ]
    fin = esubj_wh[esubj_wh["source_condition"] == "finite"]
    inf = esubj_wh[esubj_wh["source_condition"] == "infinitival"]
    if not fin.empty and not inf.empty:
        # If the same intervention_layer is present in both, compare; else
        # compare the best (max |β|) row in each.
        fin_beta = float(fin["estimate"].iloc[0])
        inf_beta = float(inf["estimate"].iloc[0])
        diagnostics["w2_finite_minus_bare_beta"] = fin_beta
        diagnostics["w2_infinitival_minus_bare_beta"] = inf_beta
        if not (abs(fin_beta) > abs(inf_beta) > 0):
            warnings.append(
                f"W2 phase-count gradient on (embedded_subject, *, bare) "
                f"wh-esubj does not show |finite|={abs(fin_beta):.3f} > "
                f"|infinitival|={abs(inf_beta):.3f} > 0. "
                f"Predicted ordering not observed."
            )

    return warnings, diagnostics


# ---------------------------------------------------------------------------
# Run discovery for patching
# ---------------------------------------------------------------------------


def _discover_patching_run(
    *, model_slug: str, variant: str, results_dir: Path,
) -> Path | None:
    """Find the newest activation_patching run for (model, variant).

    Patching run dirs are named
    ``{stimuli_prefix}_{variant}_{slug}-{config_hash}-{date}-{time}``;
    ``stimuli_prefix`` is ``wh`` for w2/w4 and ``cc`` for n1.
    """
    stimuli_prefix = _VARIANT_TO_STIMULI_PREFIX.get(variant)
    if stimuli_prefix is None:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of "
            f"{sorted(_VARIANT_TO_STIMULI_PREFIX)}."
        )
    return newest_nonempty_run(
        results_dir / _APPLY_RUN_PARENT,
        prefix=f"{stimuli_prefix}_{variant}_{model_slug}-",
        required_artifact="artifacts/per_trial_predictions.jsonl",
    )


def _resolve_apply_run(
    results_dir: Path,
    *,
    variant: str,
    apply_run: Path | None = None,
    apply_prefix: str | None = None,
) -> Path:
    """Resolve the activation_patching run dir from explicit override,
    explicit prefix, or auto-discovery from the variant."""
    if apply_run is not None:
        if not (apply_run / "artifacts/per_trial_predictions.jsonl").exists():
            raise FileNotFoundError(
                f"--apply-run {apply_run} does not contain "
                f"artifacts/per_trial_predictions.jsonl."
            )
        return apply_run
    if apply_prefix is not None:
        found = newest_nonempty_run(
            results_dir / _APPLY_RUN_PARENT,
            prefix=apply_prefix,
            required_artifact="artifacts/per_trial_predictions.jsonl",
        )
    else:
        stimuli_prefix = _VARIANT_TO_STIMULI_PREFIX[variant]
        found = newest_nonempty_run(
            results_dir / _APPLY_RUN_PARENT,
            prefix=f"{stimuli_prefix}_{variant}_",
            required_artifact="artifacts/per_trial_predictions.jsonl",
        )
    if found is None:
        raise FileNotFoundError(
            f"No activation_patching run found under "
            f"{results_dir / _APPLY_RUN_PARENT}. Pass --apply-run explicitly."
        )
    logger.info("Auto-discovered activation_patching run: %s", found.name)
    return found


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    apply_run: Path,
    output_dir: Path,
    *,
    n_bootstrap: int,
    fit_method: str,
    rng_seed: int,
    modifier_type: str | None = None,
) -> dict[str, object]:
    """End-to-end: load → explode → fit-and-bootstrap → write artifacts.

    If ``modifier_type`` is set (e.g. ``'poss'``), the per-trial rows
    are filtered to that modifier configuration before aggregation.
    This only affects N1 (c-command) trials, since W2/W4 trials have no
    modifier_type tagging. The output CSV is named
    ``patching_per_cell.csv`` if no filter is applied, or
    ``patching_per_cell__modifier_{modifier_type}.csv`` if a filter is
    applied (so a filtered re-analysis does not overwrite the pooled
    main result).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load.
    raw = load_per_trial(apply_run)
    logger.info("Loaded %d trials", len(raw))

    # 2. Explode to per-(trial × pair × layer) rows with delta_beta.
    long = explode_to_deltas(raw)
    if long.empty:
        raise RuntimeError("No (trial × pair × layer) rows produced.")
    logger.info("Exploded to %d (trial × pair × layer) rows", len(long))

    # 2b. Optional modifier-type subset (N1 only).
    if modifier_type is not None:
        before = len(long)
        long = long[long["modifier_type"] == modifier_type]
        after = len(long)
        logger.info(
            "Filtered to modifier_type=%r: %d → %d rows",
            modifier_type, before, after,
        )
        if long.empty:
            raise RuntimeError(
                f"No rows remain after filtering to modifier_type="
                f"{modifier_type!r}. (W2/W4 trials have no modifier_type "
                f"tagging; this filter only makes sense for N1.)"
            )

    # 3. Per-cell regression + bootstrap.
    logger.info("Fitting (method=%s, n_bootstrap=%d)…", fit_method, n_bootstrap)
    stats = fit_all_cells(
        long, fit_method=fit_method, n_bootstrap=n_bootstrap, rng_seed=rng_seed,
    )
    if stats.empty:
        raise RuntimeError("No per-cell fits succeeded.")
    csv_name = (
        "patching_per_cell.csv" if modifier_type is None
        else f"patching_per_cell__modifier_{modifier_type}.csv"
    )
    csv_path = output_dir / csv_name
    stats.to_csv(csv_path, index=False)
    logger.info("Wrote per-cell stats: %s", csv_path)

    # 4. Sanity checks (advisory).
    pred_warnings, pred_diagnostics = _check_predicted_pattern(stats)
    for w in pred_warnings:
        logger.warning(w)

    # 5. Headlines per role on role-relevant rows.
    rr = stats[stats["is_role_relevant"]]
    role_peaks: dict[str, dict | None] = {}
    for role in sorted(rr["intervention_role"].unique()):
        role_peaks[role] = peak_row(
            rr[rr["intervention_role"] == role]
              .assign(abs_beta=rr.loc[rr["intervention_role"] == role, "estimate"].abs()),
            by_col="abs_beta",
        )

    summary: dict[str, object] = {
        "apply_run": str(apply_run),
        "n_trials": int(len(raw)),
        "n_cells_tested": int(len(stats)),
        "n_cells_role_relevant": int(stats["is_role_relevant"].sum()),
        "alpha_bonferroni": (
            float(0.05 / stats["is_role_relevant"].sum())
            if stats["is_role_relevant"].any() else None
        ),
        "n_significant_bonferroni": int((stats["p_bonferroni"] < 0.05).sum()),
        "n_significant_fdr": int((stats["p_fdr_bh"] < 0.05).sum()),
        "fit_method_requested": fit_method,
        "fit_method_counts": {
            str(k): int(v) for k, v in stats["fit_method"].value_counts().items()
        },
        "role_peaks": role_peaks,
        "prediction_warnings": pred_warnings,
        "prediction_diagnostics": pred_diagnostics,
        "n_bootstrap_resamples": int(n_bootstrap),
    }
    summary_path = output_dir / "patching_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary: %s", summary_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Project results directory (default: ./results)",
    )
    p.add_argument(
        "--apply-run", type=Path, default=None,
        help="Explicit activation_patching run directory. If omitted, the "
             "newest non-empty matching run is auto-discovered.",
    )
    p.add_argument(
        "--apply-prefix", type=str, default=None,
        help="Prefix used to discover activation_patching runs. Use e.g. "
             "'wh_w2_qwen25_7b-' to pin a specific model. Mutually "
             "exclusive with --apply-run; if both are omitted, --variant "
             "is used to derive the prefix.",
    )
    p.add_argument(
        "--variant", choices=sorted(_VARIANT_TO_STIMULI_PREFIX), default="w2",
        help="Which Tier-1 experiment to analyse. Determines the default "
             "auto-discovery prefix (w2/w4→wh_, n1→cc_) and is forwarded "
             "to multi-model batch mode.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write stats CSV and summary. Defaults to the "
             "activation_patching run directory itself.",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples for mean-Δβ CIs.",
    )
    p.add_argument(
        "--fit-method", choices=list(FIT_METHODS), default="ols_cr1",
        help="Inferential model. 'ols_cr1' (default) = OLS with cluster-"
             "robust SEs clustered on item_id; 'mixedlm' = mixed-effects "
             "with random intercept on item_id (falls back to ols_cr1 on "
             "boundary singularity).",
    )
    p.add_argument(
        "--rng-seed", type=int, default=13,
        help="Seed for the bootstrap resampler.",
    )
    p.add_argument(
        "--modifier-type", type=str, default=None,
        choices=["pp", "poss", "rcobj", "rcsubj"],
        help="If set, restrict the per-cell aggregation to trials with "
             "this modifier_type. Only affects N1 (c-command) trials; "
             "W2/W4 trials have no modifier_type tagging and the filter "
             "will reject them. The filtered output is written to "
             "patching_per_cell__modifier_{type}.csv so the pooled "
             "result is preserved alongside.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    add_model_selection_args(p)
    return p


def _print_headline(summary: dict, output_dir: Path) -> None:
    """Print a human-readable headline for one patching analysis run."""
    print("\n" + "=" * 64)
    print(f"  apply_run:              {Path(str(summary['apply_run'])).name}")
    print(f"  trials:                 {summary['n_trials']}")
    print(f"  cells tested:           {summary['n_cells_tested']} "
          f"({summary['n_cells_role_relevant']} role-relevant)")
    if summary['alpha_bonferroni'] is not None:
        print(f"  Bonferroni α:           {summary['alpha_bonferroni']:.6f}")
    print(f"  significant tests:      Bonf={summary['n_significant_bonferroni']}, "
          f"FDR={summary['n_significant_fdr']}")
    print(f"  fit method:             {summary['fit_method_requested']} "
          f"(actual: {summary['fit_method_counts']})")
    for role, peak in (summary.get("role_peaks") or {}).items():
        if peak is None:
            continue
        print(
            f"  peak |Δβ| {role:18s}  "
            f"src={peak['source_condition']} tgt={peak['target_condition']}  "
            f"pair={peak['pair_label']}  L={peak['intervention_layer']}  "
            f"β={peak['estimate']:+.3f}  "
            f"[{peak['boot_ci_low']:+.3f}, {peak['boot_ci_high']:+.3f}]  "
            f"p={peak['p_value']:.2e}  q={peak['p_fdr_bh']:.2e}"
        )
    if summary["prediction_warnings"]:
        print("  prediction warnings:")
        for w in summary["prediction_warnings"]:
            print(f"    ⚠ {w}")
    print("=" * 64)
    print(f"  outputs written to:     {output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.apply_run is not None and args.apply_prefix is not None:
        print("error: --apply-run and --apply-prefix are mutually exclusive.",
              file=sys.stderr)
        return 2

    # ── Multi-model batch mode ──────────────────────────────────────────────
    if args.all_models or args.include:
        if args.apply_run is not None or args.output_dir is not None:
            print(
                "error: --apply-run and --output-dir are not compatible with "
                "--all-models / --include (each model writes to its own run dir).",
                file=sys.stderr,
            )
            return 2

        models = select_models(args.include, args.all_models)
        successes: list[str] = []
        failures: list[tuple[str, str]] = []
        skipped: list[str] = []

        for model in models:
            apply_run = _discover_patching_run(
                model_slug=model.slug, variant=args.variant,
                results_dir=args.results_dir,
            )
            if apply_run is None:
                logger.info(
                    "No %s activation_patching run for %s — skipping.",
                    args.variant, model.slug,
                )
                skipped.append(model.slug)
                continue

            print(f"\n{'─' * 64}")
            print(f"  model: {model.display}  ({apply_run.name})")
            print(f"{'─' * 64}")
            try:
                summary = run_analysis(
                    apply_run, apply_run,
                    n_bootstrap=args.n_bootstrap,
                    fit_method=args.fit_method,
                    rng_seed=args.rng_seed,
                    modifier_type=args.modifier_type,
                )
                _print_headline(summary, apply_run)
                successes.append(model.slug)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("FAILED %s: %s", model.slug, msg)
                failures.append((model.slug, msg))
                if args.fail_fast:
                    logger.error("Aborting due to --fail-fast.")
                    break

        print(f"\n{'=' * 64}")
        print(f"  BATCH SUMMARY  ({len(models)} models considered, "
              f"variant={args.variant})")
        print(f"  succeeded: {len(successes)}   failed: {len(failures)}"
              f"   skipped (no run): {len(skipped)}")
        print(f"{'=' * 64}")
        if successes:
            print("  ✓ " + ", ".join(successes))
        if skipped:
            print("  – " + ", ".join(skipped) + "  (no activation_patching run yet)")
        if failures:
            print("  ✗ failures:")
            for slug, msg in failures:
                print(f"      {slug}: {msg}")
        return 1 if failures else 0

    # ── Single-run mode ────────────────────────────────────────────────────
    apply_run = _resolve_apply_run(
        args.results_dir,
        variant=args.variant,
        apply_run=args.apply_run,
        apply_prefix=args.apply_prefix,
    )
    output_dir = args.output_dir if args.output_dir is not None else apply_run

    summary = run_analysis(
        apply_run, output_dir,
        n_bootstrap=args.n_bootstrap,
        fit_method=args.fit_method,
        rng_seed=args.rng_seed,
        modifier_type=args.modifier_type,
    )
    _print_headline(summary, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
