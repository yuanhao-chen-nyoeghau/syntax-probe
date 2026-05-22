"""Cross-model correlation between observational C2 sign-consistency and
the causal N1 Δβ summary across the 13 registered models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from figures import _load, _runs

logger = logging.getLogger("patching_cross_model")


# ---------------------------------------------------------------------------
# Per-model row assembly
# ---------------------------------------------------------------------------


def _per_model_row(
    slug: str,
    *,
    cc_summary: dict | None,
    per_cell: pd.DataFrame | None,
) -> dict[str, object] | None:
    """Compute one model's (C2, aligned_mean_Y, per-cell β) row.

    Returns None if either piece of data is missing or inadequate.
    """
    if cc_summary is None:
        logger.warning("No cc_summary for %s; skipping.", slug)
        return None
    c2 = _load.get_c2_sign_consistency(cc_summary)
    if c2 is None:
        logger.warning("No C2 sign-consistency diagnostic for %s; skipping.", slug)
        return None
    if per_cell is None or per_cell[per_cell["slug"] == slug].empty:
        logger.warning("No patching per_cell rows for %s; skipping.", slug)
        return None

    aligned = _load.n1_aligned_mean_beta(per_cell, slug)
    if aligned["n_cells"] == 0:
        logger.warning("No recognized N1 cells for %s; skipping.", slug)
        return None

    # Expose per-cell βs as columns for transparency.
    sub = per_cell[per_cell["slug"] == slug]
    cell_betas: dict[str, float] = {}
    for _, r in sub.iterrows():
        sign = _load.n1_predicted_sign(
            str(r["source_condition"]),
            str(r["target_condition"]),
            str(r["pair_label"]),
        )
        if sign is None:
            continue
        key = f"beta_{r['source_condition']}_to_{r['target_condition']}_{r['pair_label']}"
        cell_betas[key] = float(r["estimate"])

    row: dict[str, object] = {
        "slug": slug,
        "c2_sign_consistency": float(c2),
        "aligned_mean_beta": float(aligned["aligned_mean"]),
        "n_cells": int(aligned["n_cells"]),
        "n_cells_correct_sign": int(aligned["n_cells_correct_sign"]),
    }
    row.update(cell_betas)
    return row


# ---------------------------------------------------------------------------
# Spearman with model-level bootstrap CI
# ---------------------------------------------------------------------------


def _spearman_with_bootstrap_ci(
    x: np.ndarray, y: np.ndarray,
    *, n_bootstrap: int, rng_seed: int,
) -> dict[str, float]:
    """Point estimate + percentile bootstrap CI for Spearman ρ.

    Resamples model indices (not within-model observations) with
    replacement. Returns ``rho``, ``ci_low``, ``ci_high`` (95%) and
    the two-sided ``p_value`` from the asymptotic test at the point
    estimate.
    """
    if x.size < 3:
        return {"rho": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan")}
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(rng_seed)
    rhos: list[float] = []
    n = x.size
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        # A degenerate resample (all identical x or y values) yields a
        # NaN Spearman; skip those samples rather than letting them
        # contaminate the CI.
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            continue
        rho_b, _ = spearmanr(xb, yb)
        if not np.isnan(rho_b):
            rhos.append(float(rho_b))
    if not rhos:
        return {"rho": float(rho), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float(p)}
    return {
        "rho": float(rho),
        "ci_low": float(np.percentile(rhos, 2.5)),
        "ci_high": float(np.percentile(rhos, 97.5)),
        "p_value": float(p),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    *,
    results_dir: Path,
    output_dir: Path,
    selected_models: list,
    cc_overrides: dict[str, Path],
    n1_overrides: dict[str, Path],
    n_bootstrap: int,
    rng_seed: int,
) -> dict[str, object]:
    """End-to-end: discover runs → assemble per-model rows → Spearman →
    write CSV + summary JSON."""
    cc_runs = _runs.resolve_runs(
        "c_command",
        results_dir=results_dir,
        models=selected_models,
        overrides=cc_overrides,
        required_artifact="cc_summary.json",
        skip_missing=True,
    )
    n1_runs = _runs.resolve_patching_runs(
        "n1",
        results_dir=results_dir,
        models=selected_models,
        overrides=n1_overrides,
        required_artifact=_load.PATCHING_REQUIRED_ARTIFACT,
        skip_missing=True,
    )
    if not cc_runs:
        raise FileNotFoundError(
            f"No c_command runs found under {results_dir}/apply_probes/."
        )
    if not n1_runs:
        raise FileNotFoundError(
            f"No activation_patching/cc_n1_* runs found under "
            f"{results_dir}/activation_patching/."
        )

    cc_summaries = _load.load_cc_summary(cc_runs)
    # Load per_cell across the n1 runs we have. load_patching_per_cell
    # concatenates with a 'slug' column.
    per_cell = _load.load_patching_per_cell(n1_runs)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for m in selected_models:
        row = _per_model_row(
            m.slug,
            cc_summary=cc_summaries.get(m.slug),
            per_cell=per_cell,
        )
        if row is None:
            missing.append(m.slug)
            continue
        rows.append(row)
    if len(rows) < 3:
        raise RuntimeError(
            f"Only {len(rows)} models have complete data; need ≥3 for "
            f"Spearman. Missing: {missing}"
        )

    df = pd.DataFrame(rows).sort_values("slug").reset_index(drop=True)

    # Spearman on the matched X, Y arrays.
    x = df["c2_sign_consistency"].to_numpy()
    y = df["aligned_mean_beta"].to_numpy()
    spear = _spearman_with_bootstrap_ci(
        x, y, n_bootstrap=n_bootstrap, rng_seed=rng_seed,
    )

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cross_model_n1_correlation.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote per-model rows: %s", csv_path)

    summary: dict[str, object] = {
        "n_models_total": len(selected_models),
        "n_models_used": len(rows),
        "missing_models": missing,
        "spearman_rho": spear["rho"],
        "spearman_ci_low": spear["ci_low"],
        "spearman_ci_high": spear["ci_high"],
        "spearman_p_value": spear["p_value"],
        "n_bootstrap": n_bootstrap,
        "interpretation_threshold": {
            "rho_min_for_real_but_localized": 0.5,
            "description": (
                "Per docs/activation_patching_plan.md §3.3, ρ > 0.5 "
                "supports the 'real-but-localized' interpretation: low-C2 "
                "models really do encode c-command less robustly at the "
                "causal level too. ρ ≈ 0 supports 'C2 is a metric quirk, "
                "decoupled from causality'."
            ),
        },
        "per_model": df.to_dict(orient="records"),
    }
    summary_path = output_dir / "cross_model_n1_correlation_summary.json"
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
        "--output-dir", type=Path, default=Path("results/cross_model"),
        help="Where to write cross-model artifacts.",
    )
    p.add_argument(
        "--cc-runs", nargs="*", default=[], metavar="slug=path",
        help="Override c_command run dirs.",
    )
    p.add_argument(
        "--n1-runs", nargs="*", default=[], metavar="slug=path",
        help="Override activation_patching n1 run dirs.",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples for Spearman ρ CI.",
    )
    p.add_argument(
        "--rng-seed", type=int, default=13,
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    _runs.add_model_selection_args(p)
    return p


def _print_headline(summary: dict) -> None:
    print("\n" + "=" * 64)
    print(f"  N1 cross-model correlation (n = {summary['n_models_used']}"
          f" of {summary['n_models_total']})")
    print("=" * 64)
    rho = summary["spearman_rho"]
    lo = summary["spearman_ci_low"]
    hi = summary["spearman_ci_high"]
    p = summary["spearman_p_value"]
    print(f"  Spearman ρ:    {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"asymptotic p={p:.3g}")
    if summary["missing_models"]:
        print(f"  missing:       {', '.join(summary['missing_models'])}")
    print()
    print(f"  {'slug':18s}  {'C2':>5s}  {'Y_m':>7s}  cells correct")
    for r in summary["per_model"]:
        print(f"  {r['slug']:18s}  {r['c2_sign_consistency']:>5.2f}  "
              f"{r['aligned_mean_beta']:>+7.4f}  "
              f"{r['n_cells_correct_sign']}/{r['n_cells']}")
    rho_thresh = summary["interpretation_threshold"]["rho_min_for_real_but_localized"]
    # Use the CI, not the point estimate, to decide what the data supports.
    # Three honest cases: CI below threshold (real-but-localized not
    # supported), CI above threshold (supported), or CI spans threshold
    # (inconclusive at this n).
    if hi < rho_thresh:
        verdict = ("CI lies below threshold; real-but-localized "
                   "interpretation not supported at this n.")
    elif lo >= rho_thresh:
        verdict = "CI lies above threshold; real-but-localized supported."
    else:
        verdict = ("CI spans threshold; inconclusive at n="
                   f"{summary['n_models_used']}.")
    print(f"\n  Threshold (ρ ≥ {rho_thresh}, plan §3.3): {verdict}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    selected = _runs.select_models(args.include, args.all_models)
    cc_overrides = _runs.parse_run_overrides(args.cc_runs)
    n1_overrides = _runs.parse_run_overrides(args.n1_runs)

    summary = run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        selected_models=selected,
        cc_overrides=cc_overrides,
        n1_overrides=n1_overrides,
        n_bootstrap=args.n_bootstrap,
        rng_seed=args.rng_seed,
    )
    _print_headline(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
