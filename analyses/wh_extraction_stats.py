"""Statistical analysis of wh-extraction probe outputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.auto import tqdm

from ._common import (
    FIT_METHODS,
    add_corrections,
    add_fractional_depth,
    add_model_selection_args,
    bootstrap_per_layer_pair,
    discover_apply_run_for_model,
    discover_runs,
    find_first_significant_layer,
    fit_clustered_regression,
    load_excluded_items,
    load_predictions,
    peak_row,
)
from figures._runs import select_models

logger = logging.getLogger("wh_stats")


# ---------------------------------------------------------------------------
# Constants — must match what apply_probes wrote.
# ---------------------------------------------------------------------------

_PAIR_LABELS = ["wh-evb", "wh-esubj", "esubj-evb"]
_CONTRASTS = ["finite", "infinitival"]
_REFERENCE_CONDITION = "bare"

_APPLY_RUN_PREFIX = "wh_extraction_"
_VERIFY_RUN_PREFIX = "verify_wh_extraction_spacy-"

_PRED_COLS = {
    "stimulus_id", "item_id", "condition", "layer_index",
    "pair_label", "predicted_distance",
}


# ---------------------------------------------------------------------------
# Mixed-effects fits
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    layer_index: int
    pair_label: str
    contrast: str          # 'finite' or 'infinitival' (vs reference 'bare')
    estimate: float        # coefficient
    std_err: float
    z_value: float
    p_value: float
    n_obs: int             # total observations used
    n_items: int           # unique item_ids
    converged: bool
    fit_method: str        # 'mixedlm' or 'ols_cr1'


def fit_one_layer_pair(
    df: pd.DataFrame,
    *,
    layer_index: int,
    pair_label: str,
    fit_method: str = "ols_cr1",
) -> list[FitResult]:
    """Fit a single (layer, pair) slice via OLS+CR1 or MixedLM.

    Model formula: ``predicted_distance ~ condition`` with treatment coding,
    'bare' as reference. Returns one FitResult per non-reference contrast
    (finite, infinitival).

    See ``_common.fit_clustered_regression`` for fit-method semantics.
    """
    sub = df[(df["layer_index"] == layer_index) & (df["pair_label"] == pair_label)]
    if len(sub) == 0:
        return []

    formula = f"predicted_distance ~ C(condition, Treatment('{_REFERENCE_CONDITION}'))"
    result, used_method = fit_clustered_regression(
        formula, sub, group_col="item_id", method=fit_method,
    )
    if result is None:
        logger.warning(
            "Fit failed for layer=%d pair=%s", layer_index, pair_label,
        )
        return []

    converged = bool(getattr(result, "converged", True))
    out: list[FitResult] = []
    for contrast in _CONTRASTS:
        coef_name = (
            f"C(condition, Treatment('{_REFERENCE_CONDITION}'))[T.{contrast}]"
        )
        if coef_name not in result.params.index:
            logger.warning(
                "Coefficient %r not found at layer=%d pair=%s",
                coef_name, layer_index, pair_label,
            )
            continue
        out.append(FitResult(
            layer_index=layer_index,
            pair_label=pair_label,
            contrast=contrast,
            estimate=float(result.params[coef_name]),
            std_err=float(result.bse[coef_name]),
            z_value=float(result.tvalues[coef_name]),
            p_value=float(result.pvalues[coef_name]),
            n_obs=int(len(sub)),
            n_items=int(sub["item_id"].nunique()),
            converged=converged,
            fit_method=used_method,
        ))
    return out


def fit_all_layers(
    df: pd.DataFrame, *, fit_method: str = "ols_cr1",
) -> pd.DataFrame:
    """Fit at every (layer, pair); return tidy DataFrame."""
    layers = sorted(df["layer_index"].unique())
    results: list[FitResult] = []
    for layer_idx in tqdm(layers, desc="wh-extraction", unit="layer"):
        for pair in _PAIR_LABELS:
            results.extend(fit_one_layer_pair(
                df, layer_index=layer_idx, pair_label=pair, fit_method=fit_method,
            ))

    out = pd.DataFrame([asdict(r) for r in results])
    if out.empty:
        return out
    out = add_corrections(out)
    return add_fractional_depth(out)


# ---------------------------------------------------------------------------
# Bootstrap effect sizes
# ---------------------------------------------------------------------------


def bootstrap_effect_sizes(df: pd.DataFrame, *, n_resamples: int = 1000) -> pd.DataFrame:
    """Cluster-bootstrap mean-diff with 95% CIs for every
    (layer, pair, contrast) cell. Returns long-form DataFrame.

    Resampling is at the ``item_id`` level so within-item correlation
    (random intercept structure) is preserved in the bootstrap distribution.
    """
    rows: list[pd.DataFrame] = []
    for contrast in _CONTRASTS:
        sub = df[df["condition"].isin([contrast, _REFERENCE_CONDITION])]
        boot = bootstrap_per_layer_pair(
            sub,
            value_col="predicted_distance",
            cluster_col="item_id",
            condition_col="condition",
            treatment_value=contrast,
            reference_value=_REFERENCE_CONDITION,
            pair_col="pair_label",
            n_resamples=n_resamples,
        )
        boot["contrast"] = contrast
        rows.append(boot)
    out = pd.concat(rows, ignore_index=True)
    return add_fractional_depth(out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


_CONTRAST_STYLE = {
    "finite":      {"color": "tab:red",    "linestyle": "-",  "marker": "o"},
    "infinitival": {"color": "tab:orange", "linestyle": "--", "marker": "s"},
}

# Colors per pair for the heatmap and cross-pair plots.
_PAIR_COLORS = {
    "wh-evb":   "tab:blue",
    "wh-esubj": "tab:purple",
    "esubj-evb": "tab:gray",
}


def plot_z_profile(
    stats: pd.DataFrame, output_path: Path, *, fdr_threshold: float = 0.05,
) -> None:
    """Three-panel z-statistic profile across layers (one panel per pair).

    Adds a vertical-line annotation at the first layer where the contrast
    becomes BH-FDR-significant, and a horizontal Bonferroni line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    layers = sorted(stats["layer_index"].unique())
    if not layers:
        logger.warning("No layers in stats; skipping z-profile plot.")
        plt.close(fig)
        return

    n_tests = len(stats)
    alpha_bonf = 0.05 / n_tests if n_tests > 0 else 0.05
    z_thresh = norm.ppf(1 - alpha_bonf / 2)

    for ax, pair in zip(axes, _PAIR_LABELS, strict=True):
        sub = stats[stats["pair_label"] == pair]
        for contrast in _CONTRASTS:
            cdata = sub[sub["contrast"] == contrast].sort_values("layer_index")
            if len(cdata) == 0:
                continue
            style = _CONTRAST_STYLE[contrast]
            ax.plot(
                cdata["layer_index"], cdata["z_value"],
                label=f"{contrast} − bare",
                **style,
                markersize=5, linewidth=1.5,
            )
            # Annotate the first FDR-significant layer for this contrast.
            first_sig = find_first_significant_layer(
                cdata, p_col="p_fdr_bh", threshold=fdr_threshold,
            )
            if first_sig is not None:
                ax.axvline(
                    first_sig, color=style["color"],
                    linestyle=":", linewidth=0.8, alpha=0.5,
                )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(z_thresh, color="gray", linestyle=":", linewidth=0.7,
                   label=f"Bonferroni (z={z_thresh:.2f})")
        ax.axhline(-z_thresh, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xlabel("Layer")
        ax.set_title(pair)
        ax.set_xticks(layers[::2])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("z-statistic")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Mixed-effects contrast z-profile (vertical lines: first FDR-sig layer)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote z-profile plot: %s", output_path)


def plot_effect_size_profile(boot: pd.DataFrame, output_path: Path) -> None:
    """Three-panel effect-size (mean-diff) profile with bootstrap CI ribbons."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    layers = sorted(boot["layer_index"].unique())
    if not layers:
        logger.warning("No layers in bootstrap; skipping effect-size plot.")
        plt.close(fig)
        return

    for ax, pair in zip(axes, _PAIR_LABELS, strict=True):
        sub = boot[boot["pair_label"] == pair]
        for contrast in _CONTRASTS:
            cdata = sub[sub["contrast"] == contrast].sort_values("layer_index")
            if len(cdata) == 0:
                continue
            style = _CONTRAST_STYLE[contrast]
            ax.plot(
                cdata["layer_index"], cdata["mean_diff"],
                label=f"{contrast} − bare",
                color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markersize=5, linewidth=1.5,
            )
            ax.fill_between(
                cdata["layer_index"], cdata["ci_low"], cdata["ci_high"],
                color=style["color"], alpha=0.15,
            )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_title(pair)
        ax.set_xticks(layers[::2])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Δ predicted distance (treatment − bare)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Effect sizes across layers (cluster-bootstrap 95% CIs, n=1000)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote effect-size plot: %s", output_path)


def plot_z_heatmap(stats: pd.DataFrame, output_path: Path) -> None:
    """Heatmap: rows = layer, columns = (pair, contrast), cell = z-value.

    Easier to scan for which (layer, pair, contrast) cells reach
    significance than three line plots side by side.
    """
    if stats.empty:
        logger.warning("No stats; skipping heatmap.")
        return

    pivot = stats.pivot_table(
        index="layer_index",
        columns=["pair_label", "contrast"],
        values="z_value",
        aggfunc="first",
    )
    # Order columns: pair major (wh-evb, wh-esubj, esubj-evb), contrast minor.
    desired_cols = [
        (pair, contrast) for pair in _PAIR_LABELS for contrast in _CONTRASTS
    ]
    desired_cols = [c for c in desired_cols if c in pivot.columns]
    pivot = pivot.reindex(columns=desired_cols)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(pivot))))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{p}\n{c}" for p, c in pivot.columns], rotation=0, fontsize=8,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Pair × Contrast")
    ax.set_ylabel("Layer")
    cbar = fig.colorbar(im, ax=ax, label="z (mixed-effects)")
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("z-statistic across layers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote z-heatmap: %s", output_path)


# ---------------------------------------------------------------------------
# Sanity-check assertions
# ---------------------------------------------------------------------------


def _check_predicted_pattern(
    stats: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:
    """Headline wh-extraction prediction checks.

    Returns ``(warnings, diagnostics)``:

    * ``warnings`` — list of strings describing predictions that failed or
      look suspect. Empty if everything checks out.
    * ``diagnostics`` — JSON-serializable dict with the underlying metric
      numbers, suitable for embedding in the summary. Always populated
      regardless of whether warnings fired.

    Predictions:

    (1) **wh-esubj finite contrast is positive and FDR-significant
        somewhere.** The LM should treat the finite condition as more
        separated than the bare condition for the wh ↔ embedded-subject
        pair.

    (2) **Phase-count gradient — finite > infinitival in MAGNITUDE.**
        Under MP (Chomsky 2001), finite embedding crosses both vP and CP
        phase boundaries, while infinitival crosses only vP. So the
        wh-esubj finite-bare contrast should produce a *larger* peak β
        than the infinitival-bare contrast. We compare PEAK β (not peak z),
        because z mixes effect size with residual noise that varies layer-
        by-layer for reasons unrelated to phase count. Threshold: we flag
        only if finite peak β is meaningfully smaller than infinitival
        peak β (ratio < 0.9). A near-equal ratio is *informational* not a
        failure — the literature on whether infinitival vPs are equally
        phasal is itself contested (Bošković 2007; Wurmbrand 2014).

    (3) **Baseline (esubj-evb) effects should be smaller than wh-esubj.**
        The esubj-evb pair has both anchor points inside the embedded
        clause, so cross-clause structure shouldn't matter much. If this
        baseline pair shows effects comparable to or exceeding wh-esubj,
        something other than cross-clause structure is driving the signal.
    """
    warnings: list[str] = []
    diagnostics: dict[str, object] = {}

    wh_esubj_fin = stats[
        (stats["pair_label"] == "wh-esubj") & (stats["contrast"] == "finite")
    ]
    if wh_esubj_fin.empty:
        warnings.append("No wh-esubj finite results found; cannot check predictions.")
        return warnings, diagnostics

    # ----- Prediction (1): finite is positive and FDR-sig somewhere -----
    sig_pos = wh_esubj_fin[
        (wh_esubj_fin["p_fdr_bh"] < 0.05) & (wh_esubj_fin["estimate"] > 0)
    ]
    diagnostics["finite_significance"] = {
        "n_layers_fdr_pos_sig": int(len(sig_pos)),
        "n_layers_total": int(len(wh_esubj_fin)),
    }
    if sig_pos.empty:
        warnings.append(
            "Prediction (1) NOT met: no layer shows a positive, "
            "FDR-significant wh-esubj finite − bare contrast."
        )

    # ----- Prediction (2): phase-count gradient (compare peak β) -----
    wh_esubj_inf = stats[
        (stats["pair_label"] == "wh-esubj") & (stats["contrast"] == "infinitival")
    ]
    if not wh_esubj_inf.empty:
        # Peak β: argmax over the absolute estimate, but we only care about
        # POSITIVE peaks (the prediction is about positive separation, so a
        # large negative peak would be a different story handled below).
        fin_peak_idx = wh_esubj_fin["estimate"].idxmax()
        inf_peak_idx = wh_esubj_inf["estimate"].idxmax()
        fin_peak_beta = float(wh_esubj_fin.loc[fin_peak_idx, "estimate"])
        inf_peak_beta = float(wh_esubj_inf.loc[inf_peak_idx, "estimate"])
        fin_peak_layer = int(wh_esubj_fin.loc[fin_peak_idx, "layer_index"])
        inf_peak_layer = int(wh_esubj_inf.loc[inf_peak_idx, "layer_index"])
        ratio = (
            fin_peak_beta / inf_peak_beta
            if inf_peak_beta > 0 else float("nan")
        )
        diagnostics["phase_count_gradient"] = {
            "finite_peak_beta": fin_peak_beta,
            "finite_peak_layer": fin_peak_layer,
            "infinitival_peak_beta": inf_peak_beta,
            "infinitival_peak_layer": inf_peak_layer,
            "ratio_finite_over_infinitival": ratio,
            "threshold": 0.9,
        }
        # Only flag if finite peak is meaningfully smaller than infinitival
        # peak. Near-equal ratios are informational, not failures.
        if np.isfinite(ratio) and ratio < 0.9:
            warnings.append(
                f"Prediction (2) NOT met: wh-esubj finite peak β="
                f"{fin_peak_beta:+.3f} (layer {fin_peak_layer}) is "
                f"meaningfully smaller than infinitival peak β="
                f"{inf_peak_beta:+.3f} (layer {inf_peak_layer}); "
                f"ratio={ratio:.2f} < 0.9. Phase-count gradient inverted."
            )

    # ----- Prediction (3): baseline shouldn't dominate -----
    base_fin = stats[
        (stats["pair_label"] == "esubj-evb") & (stats["contrast"] == "finite")
    ]
    if not base_fin.empty and not wh_esubj_fin.empty:
        # Baseline magnitude: signed peak β with largest absolute value.
        base_peak_beta = float(
            base_fin.loc[base_fin["estimate"].abs().idxmax(), "estimate"]
        )
        wh_peak_beta_abs = float(wh_esubj_fin["estimate"].abs().max())
        diagnostics["baseline_dominance"] = {
            "esubj_evb_finite_peak_beta_signed": base_peak_beta,
            "wh_esubj_finite_peak_beta_abs": wh_peak_beta_abs,
            "baseline_dominates": bool(abs(base_peak_beta) >= wh_peak_beta_abs),
        }
        if abs(base_peak_beta) >= wh_peak_beta_abs:
            warnings.append(
                f"Prediction (3) WEAK: esubj-evb finite peak |β|="
                f"{abs(base_peak_beta):.3f} ≥ wh-esubj finite peak |β|="
                f"{wh_peak_beta_abs:.3f}. The baseline pair shouldn't "
                f"dominate the structural pair."
            )
    return warnings, diagnostics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    apply_run: Path,
    verify_run: Path,
    output_dir: Path,
    *,
    n_bootstrap: int = 1000,
    fit_method: str = "ols_cr1",
) -> dict[str, object]:
    """End-to-end: load → exclude → fit → bootstrap → plot → write artifacts.

    Returns the JSON-serializable summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load.
    preds = load_predictions(apply_run, expected_columns=_PRED_COLS)
    excluded_items = load_excluded_items(verify_run)
    n_total_items = preds["item_id"].nunique()

    # 2. Filter.
    clean = preds[~preds["item_id"].isin(excluded_items)].copy()
    n_used_items = clean["item_id"].nunique()
    n_excluded = len(excluded_items & set(preds["item_id"]))
    logger.info(
        "Items: %d total, %d excluded, %d used",
        n_total_items, n_excluded, n_used_items,
    )

    if n_used_items < 10:
        raise ValueError(
            f"Only {n_used_items} items remain after exclusion — too few to fit "
            f"a meaningful model. Check your verification report."
        )

    # 3. Inferential fits.
    logger.info("Fitting (method=%s)…", fit_method)
    stats = fit_all_layers(clean, fit_method=fit_method)
    if stats.empty:
        raise RuntimeError("No model fits succeeded.")
    csv_path = output_dir / "stats_per_layer.csv"
    stats.to_csv(csv_path, index=False)
    logger.info("Wrote per-layer stats: %s", csv_path)

    # 4. Bootstrap effect sizes.
    logger.info("Running cluster bootstrap (n=%d resamples)…", n_bootstrap)
    boot = bootstrap_effect_sizes(clean, n_resamples=n_bootstrap)
    boot_csv = output_dir / "effect_sizes_per_layer.csv"
    boot.to_csv(boot_csv, index=False)
    logger.info("Wrote effect-size bootstrap: %s", boot_csv)

    # 5. Plots.
    plot_z_profile(stats, output_dir / "z_profile.png")
    plot_effect_size_profile(boot, output_dir / "effect_size_profile.png")
    plot_z_heatmap(stats, output_dir / "z_heatmap.png")

    # 6. Sanity checks (advisory).
    pred_warnings, pred_diagnostics = _check_predicted_pattern(stats)
    for w in pred_warnings:
        logger.warning(w)

    # 7. Summary.
    wh_esubj_fin = stats[
        (stats["pair_label"] == "wh-esubj") & (stats["contrast"] == "finite")
    ]
    peak = peak_row(wh_esubj_fin, by_col="z_value")
    first_sig_layer = find_first_significant_layer(
        wh_esubj_fin, p_col="p_fdr_bh", threshold=0.05,
    )

    summary: dict[str, object] = {
        "apply_run": str(apply_run),
        "verify_run": str(verify_run),
        "n_items_total": int(n_total_items),
        "n_items_excluded": int(n_excluded),
        "n_items_used": int(n_used_items),
        "n_layers": int(stats["layer_index"].nunique()),
        "n_tests": int(len(stats)),
        "alpha_bonferroni": float(0.05 / len(stats)) if len(stats) > 0 else None,
        "n_significant_bonferroni": int((stats["p_bonferroni"] < 0.05).sum()),
        "n_significant_fdr": int((stats["p_fdr_bh"] < 0.05).sum()),
        "fit_method_requested": fit_method,
        "fit_method_counts": {
            str(k): int(v) for k, v in stats["fit_method"].value_counts().items()
        } if "fit_method" in stats.columns else {},
        "wh_esubj_first_fdr_layer": first_sig_layer,
        "peak_wh_esubj_finite": peak,
        "prediction_warnings": pred_warnings,
        "prediction_diagnostics": pred_diagnostics,
        "n_bootstrap_resamples": int(n_bootstrap),
    }
    summary_path = output_dir / "stats_summary.json"
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
        help="Explicit apply_probes run directory. If omitted, the newest "
             "non-empty matching run is used.",
    )
    p.add_argument(
        "--verify-run", type=Path, default=None,
        help="Explicit verify_stimuli run directory. If omitted, the newest "
             "non-empty matching run is used.",
    )
    p.add_argument(
        "--apply-prefix", type=str, default=_APPLY_RUN_PREFIX,
        help="Prefix used to discover apply-probes runs. Use e.g. "
             "'wh_extraction_qwen25_7b-' to pin a specific model.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write stats CSV / summary / plots. Defaults to the "
             "apply_probes run directory itself.",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of cluster-bootstrap resamples for effect-size CIs.",
    )
    p.add_argument(
        "--fit-method", choices=list(FIT_METHODS), default="ols_cr1",
        help="Inferential model. 'ols_cr1' (default) = OLS with cluster-"
             "robust SEs clustered on item_id; distribution-free, never "
             "fails. 'mixedlm' = mixed-effects with random intercept on "
             "item_id; falls back to ols_cr1 on boundary singularity.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    add_model_selection_args(p)
    return p


def _print_wh_headline(summary: dict, output_dir: Path) -> None:
    """Print human-readable headline for one wh-extraction analysis run."""
    peak = summary.get("peak_wh_esubj_finite")
    print("\n" + "=" * 64)
    print(f"  apply_run:           {Path(str(summary['apply_run'])).name}")
    print(f"  verify_run:          {Path(str(summary['verify_run'])).name}")
    print(f"  items used / total:  {summary['n_items_used']} / {summary['n_items_total']}  "
          f"(excluded: {summary['n_items_excluded']})")
    print(f"  total tests:         {summary['n_tests']}  "
          f"(layers={summary['n_layers']}, pairs=3, contrasts=2)")
    if summary['alpha_bonferroni'] is not None:
        print(f"  Bonferroni α:        {summary['alpha_bonferroni']:.6f}")
    print(f"  significant tests:   Bonf={summary['n_significant_bonferroni']}, "
          f"FDR={summary['n_significant_fdr']}")
    print(f"  fit method:          {summary['fit_method_requested']}  "
          f"(actual: {summary['fit_method_counts']})")
    if summary["wh_esubj_first_fdr_layer"] is not None:
        print(f"  wh-esubj finite first FDR-sig layer: "
              f"{summary['wh_esubj_first_fdr_layer']}")
    if peak is not None:
        print(f"  wh-esubj fin−bare peak: layer {peak['layer_index']}  "
              f"β={peak['estimate']:+.4f}  z={peak['z_value']:+.2f}  "
              f"p={peak['p_value']:.2e}")
    if summary["prediction_warnings"]:
        print("  prediction warnings:")
        for w in summary["prediction_warnings"]:
            print(f"    ⚠ {w}")
    print("=" * 64)
    print(f"  outputs written to:  {output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )

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
            apply_run = discover_apply_run_for_model(
                model, "wh_extraction", args.results_dir,
            )
            if apply_run is None:
                logger.info("No wh_extraction run for %s — skipping.", model.slug)
                skipped.append(model.slug)
                continue

            print(f"\n{'─' * 64}")
            print(f"  model: {model.display}  ({apply_run.name})")
            print(f"{'─' * 64}")
            try:
                _, verify_run = discover_runs(
                    args.results_dir,
                    apply_prefix=f"wh_extraction_{model.slug}-",
                    verify_prefix=_VERIFY_RUN_PREFIX,
                    apply_run=apply_run,
                    verify_run=args.verify_run,
                )
                summary = run_analysis(
                    apply_run, verify_run, apply_run,
                    n_bootstrap=args.n_bootstrap,
                    fit_method=args.fit_method,
                )
                _print_wh_headline(summary, apply_run)
                successes.append(model.slug)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("FAILED %s: %s", model.slug, msg)
                failures.append((model.slug, msg))
                if args.fail_fast:
                    logger.error("Aborting due to --fail-fast.")
                    break

        print(f"\n{'=' * 64}")
        print(f"  BATCH SUMMARY  ({len(models)} models considered)")
        print(f"  succeeded: {len(successes)}   failed: {len(failures)}"
              f"   skipped (no run): {len(skipped)}")
        print(f"{'=' * 64}")
        if successes:
            print("  ✓ " + ", ".join(successes))
        if skipped:
            print("  – " + ", ".join(skipped) + "  (no apply_probes run yet)")
        if failures:
            print("  ✗ failures:")
            for slug, msg in failures:
                print(f"      {slug}: {msg}")
        return 1 if failures else 0

    # ── Single-model mode (original behaviour, unchanged) ──────────────────
    apply_run, verify_run = discover_runs(
        args.results_dir,
        apply_prefix=args.apply_prefix,
        verify_prefix=_VERIFY_RUN_PREFIX,
        apply_run=args.apply_run,
        verify_run=args.verify_run,
    )
    output_dir = args.output_dir if args.output_dir is not None else apply_run

    summary = run_analysis(
        apply_run, verify_run, output_dir,
        n_bootstrap=args.n_bootstrap,
        fit_method=args.fit_method,
    )
    _print_wh_headline(summary, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
