"""Statistical analysis of c-command probe outputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

logger = logging.getLogger("cc_stats")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APPLY_RUN_PREFIX = "c_command_"
_VERIFY_RUN_PREFIX = "verify_c_command_spacy-"

_PRED_COLS = {
    "stimulus_id", "item_id", "condition", "subexperiment",
    "layer_index", "pair_label", "predicted_distance",
}

_REFL_MODIFIER_TYPES = ("pp", "poss", "rcsubj", "rcobj")

# The headline interaction term (statsmodels names it via Treatment coding).
_REFL_HEADLINE_TERM = (
    "C(role, Treatment('subj'))[T.mod]:"
    "C(anaphor, Treatment('reflexive'))[T.pronoun]:"
    "C(gender_config, Treatment('match'))[T.swap]"
)


# ---------------------------------------------------------------------------
# Reflexive sub-experiment — fit + bootstrap
# ---------------------------------------------------------------------------


def _decompose_reflexive(condition: str) -> dict[str, str]:
    """``refl_<anaphor>_<gender>_<modifier>`` → factor levels.

    Examples:
        refl_refl_match_pp   → {anaphor='reflexive', gender='match', modifier='pp'}
        refl_pron_swap_rcobj → {anaphor='pronoun',   gender='swap',  modifier='rcobj'}
    """
    parts = condition.split("_")
    if len(parts) != 4 or parts[0] != "refl":
        raise ValueError(f"bad reflexive condition: {condition!r}")
    return {
        "anaphor":       {"refl": "reflexive", "pron": "pronoun"}[parts[1]],
        "gender_config": parts[2],
        "modifier_type": parts[3],
    }


def _annotate_reflexive(df: pd.DataFrame) -> pd.DataFrame:
    """Add (anaphor, gender_config, modifier_type, role) columns to a
    reflexive-sub-experiment DataFrame."""
    out = df.copy()
    decomp = out["condition"].apply(_decompose_reflexive).apply(pd.Series)
    out = pd.concat([out, decomp], axis=1)
    role_map = {"subj-anaphor": "subj", "mod-anaphor": "mod"}
    if "pair_label" in out.columns:
        out["role"] = out["pair_label"].map(role_map)
    return out


def fit_reflexive(
    df: pd.DataFrame, *, fit_method: str = "ols_cr1",
) -> pd.DataFrame:
    """Fit per (layer, modifier_type) the 3-way interaction model.

    Returns a long-form DataFrame with one row per (layer, modifier_type, term).
    Includes ``n_obs``, ``n_items``, ``converged``, ``fit_method``.
    Multiple-comparisons correction is applied jointly across all terms.

    See ``_common.fit_clustered_regression`` for fit-method semantics.
    """
    if df.empty:
        return pd.DataFrame()
    df = _annotate_reflexive(df)

    rows: list[dict] = []
    layers = sorted(df["layer_index"].unique())
    # The groupby below iterates (layer, modifier) in sorted order. We want
    # one tqdm tick per layer, so we update the bar when the loop finishes
    # the last modifier_type of each layer.
    last_modifier = sorted(df["modifier_type"].unique())[-1]
    pbar = tqdm(total=len(layers), desc="reflexive", unit="layer")
    try:
        for (layer, modifier), sub in df.groupby(
            ["layer_index", "modifier_type"], sort=True,
        ):
            formula = (
                "predicted_distance ~ "
                "C(role, Treatment('subj'))"
                " * C(anaphor, Treatment('reflexive'))"
                " * C(gender_config, Treatment('match'))"
            )
            result, used_method = fit_clustered_regression(
                formula, sub, group_col="item_id", method=fit_method,
            )
            if result is None:
                logger.warning(
                    "Fit failed: layer=%d modifier=%s", layer, modifier,
                )
            else:
                converged = bool(getattr(result, "converged", True))
                for term, est, se, z, p in zip(
                    result.params.index, result.params.values, result.bse.values,
                    result.tvalues.values, result.pvalues.values,
                    strict=True,
                ):
                    if term == "Group Var" or str(term).lower().startswith("group"):
                        continue
                    rows.append({
                        "layer_index": int(layer),
                        "modifier_type": modifier,
                        "term": term,
                        "estimate": float(est),
                        "std_err": float(se),
                        "z_value": float(z),
                        "p_value": float(p),
                        "n_obs": int(len(sub)),
                        "n_items": int(sub["item_id"].nunique()),
                        "converged": converged,
                        "fit_method": used_method,
                    })
            if modifier == last_modifier:
                pbar.update(1)
    finally:
        pbar.close()

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Don't correct intercepts.
    is_test = ~out["term"].str.startswith("Intercept")
    out = add_corrections(out, test_mask=is_test)
    return add_fractional_depth(out)


def bootstrap_reflexive(
    df: pd.DataFrame, *, n_resamples: int = 1000,
) -> pd.DataFrame:
    """Bootstrap descriptive (mod − subj) mean-diff per (layer × modifier_type
    × anaphor × gender_config). The thing the headline interaction is
    summarising at the LME level — useful as a complementary descriptive view.
    """
    if df.empty:
        return pd.DataFrame()
    df = _annotate_reflexive(df)

    # We need (mod − subj) per (anaphor, gender_config) cell, per layer per
    # modifier_type. This means the "treatment" is role=mod and the
    # "reference" is role=subj. We groupby (layer, modifier_type, anaphor,
    # gender_config) and bootstrap inside each cell.
    rows: list[pd.DataFrame] = []
    for (anaphor, gender_config), sub in df.groupby(
        ["anaphor", "gender_config"], sort=True,
    ):
        boot = bootstrap_per_layer_pair(
            sub,
            value_col="predicted_distance",
            cluster_col="item_id",
            condition_col="role",
            treatment_value="mod",
            reference_value="subj",
            extra_group_cols=["modifier_type"],
            n_resamples=n_resamples,
        )
        boot["anaphor"] = anaphor
        boot["gender_config"] = gender_config
        boot["subexperiment"] = "reflexive"
        rows.append(boot)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    return add_fractional_depth(out)


# ---------------------------------------------------------------------------
# Principle C
# ---------------------------------------------------------------------------


def fit_principle_c(
    df: pd.DataFrame, *, fit_method: str = "ols_cr1",
) -> pd.DataFrame:
    """Fit per layer: ``predicted_distance ~ condition``.

    Treatment-coded with ``prinC_violation`` as reference. See
    ``_common.fit_clustered_regression`` for fit-method semantics.
    """
    if df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    layers = sorted(df["layer_index"].unique())
    pbar = tqdm(total=len(layers), desc="principle_c", unit="layer")
    try:
        for layer, sub in df.groupby("layer_index", sort=True):
            formula = (
                "predicted_distance ~ C(condition, Treatment('prinC_violation'))"
            )
            result, used_method = fit_clustered_regression(
                formula, sub, group_col="item_id", method=fit_method,
            )
            if result is None:
                logger.warning("Fit failed: principle_c layer=%d", layer)
            else:
                converged = bool(getattr(result, "converged", True))
                for term, est, se, z, p in zip(
                    result.params.index, result.params.values, result.bse.values,
                    result.tvalues.values, result.pvalues.values,
                    strict=True,
                ):
                    if term == "Group Var" or str(term).lower().startswith("group"):
                        continue
                    rows.append({
                        "layer_index": int(layer),
                        "term": term,
                        "estimate": float(est),
                        "std_err": float(se),
                        "z_value": float(z),
                        "p_value": float(p),
                        "n_obs": int(len(sub)),
                        "n_items": int(sub["item_id"].nunique()),
                        "converged": converged,
                        "fit_method": used_method,
                    })
            pbar.update(1)
    finally:
        pbar.close()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    is_test = ~out["term"].str.startswith("Intercept")
    out = add_corrections(out, test_mask=is_test)
    return add_fractional_depth(out)


def bootstrap_principle_c(
    df: pd.DataFrame, *, n_resamples: int = 1000,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    boot = bootstrap_per_layer_pair(
        df,
        value_col="predicted_distance",
        cluster_col="item_id",
        condition_col="condition",
        treatment_value="prinC_obviated",
        reference_value="prinC_violation",
        n_resamples=n_resamples,
    )
    boot["subexperiment"] = "principle_c"
    if not boot.empty:
        boot = add_fractional_depth(boot)
    return boot


# ---------------------------------------------------------------------------
# Bound variable
# ---------------------------------------------------------------------------


def fit_bound_var(
    df: pd.DataFrame, *, fit_method: str = "ols_cr1",
) -> pd.DataFrame:
    """Fit per (layer, pair_label): ``predicted_distance ~ condition``.

    See ``_common.fit_clustered_regression`` for fit-method semantics.
    """
    if df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    layers = sorted(df["layer_index"].unique())
    last_pair = sorted(df["pair_label"].unique())[-1]
    pbar = tqdm(total=len(layers), desc="bound_var", unit="layer")
    try:
        for (layer, pair), sub in df.groupby(
            ["layer_index", "pair_label"], sort=True,
        ):
            formula = (
                "predicted_distance ~ C(condition, Treatment('bv_subj_quant'))"
            )
            result, used_method = fit_clustered_regression(
                formula, sub, group_col="item_id", method=fit_method,
            )
            if result is None:
                logger.warning(
                    "Fit failed: bound_var layer=%d pair=%s", layer, pair,
                )
            else:
                converged = bool(getattr(result, "converged", True))
                for term, est, se, z, p in zip(
                    result.params.index, result.params.values, result.bse.values,
                    result.tvalues.values, result.pvalues.values,
                    strict=True,
                ):
                    if term == "Group Var" or str(term).lower().startswith("group"):
                        continue
                    rows.append({
                        "layer_index": int(layer),
                        "pair_label": pair,
                        "term": term,
                        "estimate": float(est),
                        "std_err": float(se),
                        "z_value": float(z),
                        "p_value": float(p),
                        "n_obs": int(len(sub)),
                        "n_items": int(sub["item_id"].nunique()),
                        "converged": converged,
                        "fit_method": used_method,
                    })
            if pair == last_pair:
                pbar.update(1)
    finally:
        pbar.close()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    is_test = ~out["term"].str.startswith("Intercept")
    out = add_corrections(out, test_mask=is_test)
    return add_fractional_depth(out)


def bootstrap_bound_var(
    df: pd.DataFrame, *, n_resamples: int = 1000,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    boot = bootstrap_per_layer_pair(
        df,
        value_col="predicted_distance",
        cluster_col="item_id",
        condition_col="condition",
        treatment_value="bv_mod_quant",
        reference_value="bv_subj_quant",
        pair_col="pair_label",
        n_resamples=n_resamples,
    )
    boot["subexperiment"] = "bound_var"
    if not boot.empty:
        boot = add_fractional_depth(boot)
    return boot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


# Distinguishable colors for the four modifier types.
_MOD_COLORS = {
    "pp":     "tab:blue",
    "poss":   "tab:red",     # Highlight: poss has the linear-order inversion.
    "rcsubj": "tab:green",
    "rcobj":  "tab:orange",
}


def _bonferroni_z_threshold(n_tests: int) -> float:
    if n_tests == 0:
        return float("nan")
    alpha_bonf = 0.05 / n_tests
    return float(norm.ppf(1 - alpha_bonf / 2))


def _filter_term(df: pd.DataFrame, term: str) -> pd.DataFrame:
    """Return rows where ``df['term'] == term``. Empty DataFrames pass through
    so plotting code can short-circuit without column-error noise."""
    if df.empty or "term" not in df.columns:
        return df.iloc[0:0].copy() if not df.empty else df.copy()
    return df[df["term"] == term].copy()


def plot_reflexive_z_profile(
    refl_stats: pd.DataFrame, output_path: Path,
) -> None:
    """4-panel z-profile of the headline 3-way interaction per modifier type."""
    headline = _filter_term(refl_stats, _REFL_HEADLINE_TERM)
    if headline.empty:
        logger.warning("No headline 3-way term found; skipping reflexive z-profile.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    n_tests = int(refl_stats[~refl_stats["term"].str.startswith("Intercept")].shape[0])
    z_thresh = _bonferroni_z_threshold(n_tests)
    layers = sorted(headline["layer_index"].unique())

    for ax, modifier in zip(axes, _REFL_MODIFIER_TYPES, strict=True):
        sub = headline[headline["modifier_type"] == modifier].sort_values("layer_index")
        if sub.empty:
            ax.set_title(f"{modifier} (no data)")
            continue
        ax.plot(
            sub["layer_index"], sub["z_value"],
            color=_MOD_COLORS[modifier],
            marker="o", linewidth=1.6, markersize=5,
        )
        first_sig = find_first_significant_layer(
            sub, p_col="p_fdr_bh", threshold=0.05,
        )
        if first_sig is not None:
            ax.axvline(
                first_sig, color=_MOD_COLORS[modifier],
                linestyle=":", linewidth=0.8, alpha=0.6,
                label=f"first FDR-sig: layer {first_sig}",
            )
            ax.legend(loc="best", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
        if np.isfinite(z_thresh):
            ax.axhline(z_thresh, color="gray", linestyle=":", linewidth=0.7)
            ax.axhline(-z_thresh, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xlabel("Layer")
        ax.set_title(f"modifier_type = {modifier}")
        ax.set_xticks(layers[::2])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("z (3-way: role × anaphor × gender)")
    fig.suptitle(
        "Reflexive: 3-way interaction z-profile by modifier type "
        "(consistent direction across panels = c-command, not linear distance)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote reflexive z-profile: %s", output_path)


def plot_reflexive_effect_size(
    boot: pd.DataFrame, output_path: Path,
) -> None:
    """4-panel descriptive effect size (mod − subj) per modifier type, with
    refl/pron × match/swap as 4 lines per panel. Color encodes anaphor type
    (reflexive=dark, pronoun=light), linestyle encodes gender (match=solid,
    swap=dashed)."""
    if boot.empty:
        logger.warning("No reflexive bootstrap; skipping effect-size plot.")
        return

    # Distinct colors so the 4 (anaphor × gender) lines are easy to tell apart
    # *within* each panel. Panel = modifier_type (we use the panel title for that).
    line_styles = {
        ("reflexive", "match"): {"color": "tab:red",    "linestyle": "-",  "marker": "o"},
        ("reflexive", "swap"):  {"color": "tab:red",    "linestyle": "--", "marker": "o"},
        ("pronoun",   "match"): {"color": "tab:blue",   "linestyle": "-",  "marker": "s"},
        ("pronoun",   "swap"):  {"color": "tab:blue",   "linestyle": "--", "marker": "s"},
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    layers = sorted(boot["layer_index"].unique())

    for ax, modifier in zip(axes, _REFL_MODIFIER_TYPES, strict=True):
        sub = boot[boot["modifier_type"] == modifier]
        if sub.empty:
            ax.set_title(f"{modifier} (no data)")
            continue
        for (anaphor, gender), style in line_styles.items():
            cell = sub[
                (sub["anaphor"] == anaphor)
                & (sub["gender_config"] == gender)
            ].sort_values("layer_index")
            if cell.empty:
                continue
            ax.plot(
                cell["layer_index"], cell["mean_diff"],
                color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markersize=4, linewidth=1.4,
                label=f"{anaphor}/{gender}",
            )
            ax.fill_between(
                cell["layer_index"], cell["ci_low"], cell["ci_high"],
                color=style["color"], alpha=0.10,
            )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_title(f"modifier_type = {modifier}")
        ax.set_xticks(layers[::2])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Δ predicted distance (mod − subj)")
    axes[-1].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(
        "Reflexive: effect sizes (mod − subj) by anaphor type and gender match"
        " (cluster bootstrap 95% CI)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote reflexive effect-size plot: %s", output_path)


def plot_reflexive_consistency(
    refl_stats: pd.DataFrame, output_path: Path,
) -> None:
    """Cross-modifier-type consistency check.

    For each layer, plot the headline 3-way interaction estimate as 4 points
    (one per modifier type). If all four points have the same sign, that's
    consistent with c-command. If poss diverges (sign-flips), that's
    consistent with linear distance.
    """
    headline = _filter_term(refl_stats, _REFL_HEADLINE_TERM)
    if headline.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    layers = sorted(headline["layer_index"].unique())

    for modifier in _REFL_MODIFIER_TYPES:
        sub = headline[headline["modifier_type"] == modifier].sort_values("layer_index")
        if sub.empty:
            continue
        ax.errorbar(
            sub["layer_index"], sub["estimate"],
            yerr=1.96 * sub["std_err"],
            fmt="o", capsize=3, color=_MOD_COLORS[modifier],
            label=modifier,
            markersize=5, linewidth=1.0, alpha=0.85,
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("β (3-way interaction estimate)")
    ax.set_title(
        "Reflexive: cross-modifier-type consistency of the 3-way interaction.\n"
        "Same sign across all modifier types ⇒ c-command. "
        "If 'poss' (red) flips relative to others ⇒ linear distance."
    )
    ax.set_xticks(layers[::2])
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote reflexive consistency plot: %s", output_path)


def plot_principle_c(
    pc_stats: pd.DataFrame, pc_boot: pd.DataFrame, output_path: Path,
) -> None:
    """Two-panel: z-profile (left) + effect size with CIs (right)."""
    contrast_term = "C(condition, Treatment('prinC_violation'))[T.prinC_obviated]"
    z_data = _filter_term(pc_stats, contrast_term)
    if not z_data.empty:
        z_data = z_data.sort_values("layer_index")

    if z_data.empty and pc_boot.empty:
        logger.warning("No principle_c data; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    layers = (
        sorted(pc_stats["layer_index"].unique())
        if not pc_stats.empty
        else (sorted(pc_boot["layer_index"].unique()) if not pc_boot.empty else [])
    )

    # Left: z-profile
    if not z_data.empty:
        axes[0].plot(
            z_data["layer_index"], z_data["z_value"],
            color="tab:purple", marker="o", linewidth=1.5, markersize=5,
        )
        n_tests = int(pc_stats[~pc_stats["term"].str.startswith("Intercept")].shape[0])
        z_thresh = _bonferroni_z_threshold(n_tests)
        if np.isfinite(z_thresh):
            axes[0].axhline(z_thresh, color="gray", linestyle=":", linewidth=0.7)
            axes[0].axhline(-z_thresh, color="gray", linestyle=":", linewidth=0.7)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("z (obviated − violation)")
    axes[0].set_title("Principle C: z-profile")
    if layers:
        axes[0].set_xticks(layers[::2])
    axes[0].grid(alpha=0.3)

    # Right: effect size
    if not pc_boot.empty:
        sub = pc_boot.sort_values("layer_index")
        axes[1].plot(
            sub["layer_index"], sub["mean_diff"],
            color="tab:purple", marker="o", linewidth=1.5, markersize=5,
        )
        axes[1].fill_between(
            sub["layer_index"], sub["ci_low"], sub["ci_high"],
            color="tab:purple", alpha=0.18,
        )
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Δ d (obviated − violation)")
    axes[1].set_title("Principle C: effect size with bootstrap CI")
    if layers:
        axes[1].set_xticks(layers[::2])
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Principle C — pronoun in c-command position vs. adjunct",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote principle_c plot: %s", output_path)


def plot_bound_var(
    bv_stats: pd.DataFrame, bv_boot: pd.DataFrame, output_path: Path,
) -> None:
    """2x2 grid: top row z-profiles per pair, bottom row effect sizes per pair."""
    contrast_term = "C(condition, Treatment('bv_subj_quant'))[T.bv_mod_quant]"
    pairs = sorted(bv_stats["pair_label"].unique()) if not bv_stats.empty else []

    fig, axes = plt.subplots(2, len(pairs) or 1, figsize=(7 * max(len(pairs), 1), 8))
    if len(pairs) == 0:
        plt.close(fig)
        return
    if len(pairs) == 1:
        axes = axes.reshape(2, 1)

    layers = sorted(bv_stats["layer_index"].unique())
    n_tests = int(bv_stats[~bv_stats["term"].str.startswith("Intercept")].shape[0])
    z_thresh = _bonferroni_z_threshold(n_tests)

    for col, pair in enumerate(pairs):
        # Top row: z-profile
        z_data = bv_stats[
            (bv_stats["pair_label"] == pair) & (bv_stats["term"] == contrast_term)
        ].sort_values("layer_index")
        if not z_data.empty:
            axes[0, col].plot(
                z_data["layer_index"], z_data["z_value"],
                color="tab:green", marker="o", linewidth=1.5, markersize=5,
            )
            if np.isfinite(z_thresh):
                axes[0, col].axhline(z_thresh, color="gray", linestyle=":", linewidth=0.7)
                axes[0, col].axhline(-z_thresh, color="gray", linestyle=":", linewidth=0.7)
        axes[0, col].axhline(0, color="gray", linewidth=0.5)
        axes[0, col].set_title(f"{pair} — z-profile")
        axes[0, col].set_xlabel("Layer")
        axes[0, col].set_ylabel("z (mod_quant − subj_quant)")
        axes[0, col].set_xticks(layers[::2])
        axes[0, col].grid(alpha=0.3)

        # Bottom row: effect size
        sub_b = bv_boot[bv_boot["pair_label"] == pair].sort_values("layer_index")
        if not sub_b.empty:
            axes[1, col].plot(
                sub_b["layer_index"], sub_b["mean_diff"],
                color="tab:green", marker="o", linewidth=1.5, markersize=5,
            )
            axes[1, col].fill_between(
                sub_b["layer_index"], sub_b["ci_low"], sub_b["ci_high"],
                color="tab:green", alpha=0.18,
            )
        axes[1, col].axhline(0, color="gray", linewidth=0.5)
        axes[1, col].set_title(f"{pair} — effect size")
        axes[1, col].set_xlabel("Layer")
        axes[1, col].set_ylabel("Δ d (mod_quant − subj_quant)")
        axes[1, col].set_xticks(layers[::2])
        axes[1, col].grid(alpha=0.3)

    fig.suptitle(
        "Bound variable — quantifier in subject vs. modifier position",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote bound_var plot: %s", output_path)


def plot_overview(
    refl_stats: pd.DataFrame,
    pc_stats: pd.DataFrame,
    bv_stats: pd.DataFrame,
    output_path: Path,
) -> None:
    """Paper-ready 3-row summary figure.

      Row 1 (reflexive): 3-way interaction z, lines per modifier_type.
      Row 2 (principle_c): single z curve.
      Row 3 (bound_var): one curve per pair_label.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # Row 1
    headline = _filter_term(refl_stats, _REFL_HEADLINE_TERM)
    layers: list[int] = []
    if not headline.empty:
        layers = sorted(headline["layer_index"].unique())
        for modifier in _REFL_MODIFIER_TYPES:
            sub = headline[headline["modifier_type"] == modifier].sort_values("layer_index")
            if sub.empty:
                continue
            axes[0].plot(
                sub["layer_index"], sub["z_value"],
                color=_MOD_COLORS[modifier], marker="o",
                markersize=4, linewidth=1.4, label=modifier,
            )
        axes[0].legend(loc="best", fontsize=8, title="modifier_type")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_ylabel("z (3-way interaction)")
    axes[0].set_title("Reflexive — role × anaphor × gender, per modifier type")
    axes[0].grid(alpha=0.3)

    # Row 2
    pc_term = "C(condition, Treatment('prinC_violation'))[T.prinC_obviated]"
    pc_data = _filter_term(pc_stats, pc_term).sort_values("layer_index")
    if not pc_data.empty:
        if not layers:
            layers = sorted(pc_data["layer_index"].unique())
        axes[1].plot(
            pc_data["layer_index"], pc_data["z_value"],
            color="tab:purple", marker="o", markersize=4, linewidth=1.4,
        )
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_ylabel("z (obviated − violation)")
    axes[1].set_title("Principle C — pronoun-Rexp distance, c-command vs. adjunct")
    axes[1].grid(alpha=0.3)

    # Row 3
    bv_term = "C(condition, Treatment('bv_subj_quant'))[T.bv_mod_quant]"
    bv_pairs = sorted(bv_stats["pair_label"].unique()) if not bv_stats.empty else []
    bv_palette = {"qnoun-pron": "tab:green", "quant-pron": "tab:olive"}
    for pair in bv_pairs:
        sub = bv_stats[
            (bv_stats["pair_label"] == pair) & (bv_stats["term"] == bv_term)
        ].sort_values("layer_index")
        if sub.empty:
            continue
        if not layers:
            layers = sorted(sub["layer_index"].unique())
        axes[2].plot(
            sub["layer_index"], sub["z_value"],
            color=bv_palette.get(pair, "tab:gray"),
            marker="o", markersize=4, linewidth=1.4, label=pair,
        )
    if bv_pairs:
        axes[2].legend(loc="best", fontsize=8, title="pair")
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].set_ylabel("z (mod_quant − subj_quant)")
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Bound variable — quantifier in subject vs. modifier")
    axes[2].grid(alpha=0.3)

    if layers:
        axes[2].set_xticks(layers[::2])

    fig.suptitle("c-command experiment — overview", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote overview plot: %s", output_path)


# ---------------------------------------------------------------------------
# Sanity-check assertions
# ---------------------------------------------------------------------------


def _check_predictions(
    refl_stats: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:
    """Headline reflexive prediction checks.

    Returns ``(warnings, diagnostics)``:

    * ``warnings`` — list of strings describing predictions that failed or
      look suspect. Empty if everything checks out.
    * ``diagnostics`` — JSON-serializable dict with the underlying metric
      numbers, suitable for embedding in the summary. Always populated
      regardless of whether warnings fired, so consumers of the summary
      see the actual numbers (not just a binary warning bit).

    Predictions:

    (a) **Role main effect — subj closer than mod.** With treatment coding
        ``subj`` as reference, the coefficient on ``[T.mod]`` is
        ``E[d(mod-anaphor)] − E[d(subj-anaphor)]``. **Positive β** means
        mod-anaphor distance > subj-anaphor distance, i.e. subj is closer
        to the anaphor than the modifier is — the c-command-consistent
        direction. We expect this to hold in most (layer, modifier) cells.

    (b) **3-way interaction is FDR-significant somewhere.** The
        ``role × anaphor × gender_config`` interaction should reach
        FDR<.05 at some mid layer for at least one modifier type.

    (c) **Cross-modifier-type sign consistency on the 3-way interaction.**
        The 3-way β should have the same sign across all four modifier
        types in a meaningful fraction of layers. Single-layer agreement
        is too brittle (transition layers can flip), so we aggregate:
        fraction of layers where all four modifier types share a sign.
        By chance with 4 random signs, P(all-agree) = 12.5%; we threshold
        at 60% — well above chance, demanding majority agreement, but
        leaving headroom for layers where the LM hasn't yet developed a
        coherent representation.
    """
    warnings: list[str] = []
    diagnostics: dict[str, object] = {}
    if refl_stats.empty:
        warnings.append("Reflexive stats empty; cannot check predictions.")
        return warnings, diagnostics

    # ----- Prediction (a): role main effect, β > 0 means subj closer -----
    role_term = "C(role, Treatment('subj'))[T.mod]"
    role_data = _filter_term(refl_stats, role_term)
    if role_data.empty:
        warnings.append("Prediction (a): role main-effect term not found.")
    else:
        # β > 0 means mod-anaphor > subj-anaphor distance, i.e. subj closer.
        n_pos = int((role_data["estimate"] > 0).sum())
        n_total = int(len(role_data))
        frac_pos = n_pos / n_total if n_total else 0.0
        # Per-modifier breakdown — useful diagnostic for the paper, surfaces
        # the by-modifier-type pattern (e.g. rcsubj defects to linear).
        per_mod: dict[str, dict[str, int]] = {}
        for mod, sub in role_data.groupby("modifier_type"):
            per_mod[str(mod)] = {
                "n_pos": int((sub["estimate"] > 0).sum()),
                "n_total": int(len(sub)),
            }
        diagnostics["role_main_effect"] = {
            "n_subj_closer": n_pos,
            "n_total_cells": n_total,
            "fraction_subj_closer": float(frac_pos),
            "per_modifier_type": per_mod,
        }
        # Threshold: 60% of cells should show subj-closer for the prediction
        # to count as met. Below that we warn. Above 75% counts as strong.
        if frac_pos < 0.60:
            warnings.append(
                f"Prediction (a) NOT met: subj-closer-than-mod (β > 0) in "
                f"only {n_pos}/{n_total} ({frac_pos:.0%}) cells; expected "
                f"≥ 60%."
            )

    # ----- Prediction (b): 3-way is FDR-sig somewhere -----
    headline = _filter_term(refl_stats, _REFL_HEADLINE_TERM)
    if headline.empty:
        warnings.append("Prediction (b): no 3-way term found.")
        return warnings, diagnostics
    sig = headline[headline["p_fdr_bh"] < 0.05]
    diagnostics["headline_3way"] = {
        "n_fdr_sig": int(len(sig)),
        "n_total_cells": int(len(headline)),
    }
    if sig.empty:
        warnings.append(
            "Prediction (b) NOT met: no layer×modifier_type cell shows an "
            "FDR-significant 3-way interaction."
        )

    # ----- Prediction (c): layer-aggregate cross-modifier-type sign agreement -----
    # Pivot to (layer × modifier_type) of the 3-way β, then per layer ask:
    # do all four modifier types agree on sign? Aggregate over layers.
    pivot = headline.pivot_table(
        index="layer_index",
        columns="modifier_type",
        values="estimate",
        aggfunc="first",
    )
    if pivot.empty or pivot.shape[1] < 2:
        warnings.append(
            "Prediction (c): insufficient modifier-type coverage for "
            "cross-modifier consistency check."
        )
    else:
        signs = np.sign(pivot.to_numpy())
        # A layer counts as "agreeing" if all modifier types have the same
        # non-zero sign. Layers with any NaN or zero are excluded from the
        # numerator (safer than calling them agreement).
        all_pos = (signs > 0).all(axis=1)
        all_neg = (signs < 0).all(axis=1)
        n_agree = int((all_pos | all_neg).sum())
        n_layers = int(len(pivot))
        frac_agree = n_agree / n_layers if n_layers else 0.0
        diagnostics["sign_consistency"] = {
            "n_layers_all_agree": n_agree,
            "n_layers_total": n_layers,
            "fraction_layers_agree": float(frac_agree),
            "threshold": 0.60,
            "modifier_types": list(pivot.columns),
        }
        if frac_agree < 0.60:
            warnings.append(
                f"Prediction (c) NOT met: only {n_agree}/{n_layers} "
                f"({frac_agree:.0%}) layers have all modifier types agreeing "
                f"on sign of the 3-way interaction (threshold: 60%). "
                f"Below-threshold consistency suggests linear-distance rather "
                f"than c-command sensitivity."
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
    """End-to-end: load → exclude → fit (×3 sub-experiments) → bootstrap →
    plot → write artifacts. Returns the JSON-serializable summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load.
    preds = load_predictions(apply_run, expected_columns=_PRED_COLS)
    excluded_items = load_excluded_items(verify_run)
    n_total_items = preds["item_id"].nunique()

    clean = preds[~preds["item_id"].isin(excluded_items)].copy()
    n_used_items = clean["item_id"].nunique()
    n_excluded = len(excluded_items & set(preds["item_id"]))
    logger.info(
        "Items: %d total, %d excluded, %d used",
        n_total_items, n_excluded, n_used_items,
    )

    if n_used_items < 10:
        raise ValueError(
            f"Only {n_used_items} items remain after exclusion — too few "
            f"to fit meaningful models."
        )

    # Sub-experiment counts
    sub_counts = clean["subexperiment"].value_counts().to_dict()
    logger.info("Sub-experiment counts: %s", sub_counts)

    df_refl = clean[clean["subexperiment"] == "reflexive"]
    df_pc   = clean[clean["subexperiment"] == "principle_c"]
    df_bv   = clean[clean["subexperiment"] == "bound_var"]

    # 2. Inferential fits.
    logger.info("Fitting reflexive sub-experiment (method=%s)…", fit_method)
    refl_stats = fit_reflexive(df_refl, fit_method=fit_method)
    logger.info("Fitting principle_c sub-experiment (method=%s)…", fit_method)
    pc_stats = fit_principle_c(df_pc, fit_method=fit_method)
    logger.info("Fitting bound_var sub-experiment (method=%s)…", fit_method)
    bv_stats = fit_bound_var(df_bv, fit_method=fit_method)

    refl_stats.to_csv(output_dir / "cc_reflexive_per_layer.csv", index=False)
    pc_stats.to_csv(output_dir / "cc_principle_c_per_layer.csv", index=False)
    bv_stats.to_csv(output_dir / "cc_bound_var_per_layer.csv", index=False)
    logger.info("Wrote per-layer LME CSVs.")

    # 3. Bootstrap effect sizes.
    logger.info("Cluster bootstrap (n=%d resamples)…", n_bootstrap)
    refl_boot = bootstrap_reflexive(df_refl, n_resamples=n_bootstrap)
    pc_boot   = bootstrap_principle_c(df_pc, n_resamples=n_bootstrap)
    bv_boot   = bootstrap_bound_var(df_bv, n_resamples=n_bootstrap)
    all_boot = pd.concat(
        [b for b in [refl_boot, pc_boot, bv_boot] if not b.empty],
        ignore_index=True,
    )
    all_boot.to_csv(output_dir / "cc_effect_sizes_per_layer.csv", index=False)
    logger.info("Wrote effect-size bootstrap CSV.")

    # 4. Plots.
    plot_reflexive_z_profile(refl_stats, output_dir / "cc_reflexive_z_profile.png")
    plot_reflexive_effect_size(refl_boot, output_dir / "cc_reflexive_effect_size.png")
    plot_reflexive_consistency(refl_stats, output_dir / "cc_reflexive_consistency.png")
    plot_principle_c(pc_stats, pc_boot, output_dir / "cc_principle_c.png")
    plot_bound_var(bv_stats, bv_boot, output_dir / "cc_bound_var.png")
    plot_overview(refl_stats, pc_stats, bv_stats, output_dir / "cc_overview.png")

    # 5. Predictions.
    pred_warnings, pred_diagnostics = _check_predictions(refl_stats)
    for w in pred_warnings:
        logger.warning(w)

    # 6. Summary.
    refl_headline = _filter_term(refl_stats, _REFL_HEADLINE_TERM)
    refl_peak = peak_row(refl_headline, by_col="z_value")
    refl_first_sig = find_first_significant_layer(
        refl_headline, p_col="p_fdr_bh", threshold=0.05,
    )

    pc_data = _filter_term(
        pc_stats, "C(condition, Treatment('prinC_violation'))[T.prinC_obviated]"
    )
    pc_peak = peak_row(pc_data, by_col="z_value")
    pc_first_sig = find_first_significant_layer(
        pc_data, p_col="p_fdr_bh", threshold=0.05,
    )

    bv_data = _filter_term(
        bv_stats, "C(condition, Treatment('bv_subj_quant'))[T.bv_mod_quant]"
    )
    bv_peaks_by_pair: dict[str, dict] = {}
    bv_first_sig_by_pair: dict[str, int | None] = {}
    if not bv_data.empty:
        for pair, sub in bv_data.groupby("pair_label"):
            bv_peaks_by_pair[str(pair)] = peak_row(sub, by_col="z_value") or {}
            bv_first_sig_by_pair[str(pair)] = find_first_significant_layer(
                sub, p_col="p_fdr_bh", threshold=0.05,
            )

    def _fit_method_counts(stats_df: pd.DataFrame) -> dict[str, int]:
        if stats_df.empty or "fit_method" not in stats_df.columns:
            return {}
        return {
            str(k): int(v) for k, v in stats_df["fit_method"].value_counts().items()
        }

    summary: dict[str, object] = {
        "apply_run": str(apply_run),
        "verify_run": str(verify_run),
        "n_items_total": int(n_total_items),
        "n_items_excluded": int(n_excluded),
        "n_items_used": int(n_used_items),
        "subexperiment_counts": {k: int(v) for k, v in sub_counts.items()},
        "n_bootstrap_resamples": int(n_bootstrap),
        "n_layers": int(clean["layer_index"].nunique()),
        "fit_method_requested": fit_method,
        "reflexive": {
            "n_tests": int(len(refl_stats)),
            "n_significant_fdr": int((refl_stats["p_fdr_bh"] < 0.05).sum()) if not refl_stats.empty else 0,
            "fit_method_counts": _fit_method_counts(refl_stats),
            "headline_first_fdr_layer": refl_first_sig,
            "headline_peak": refl_peak,
            "prediction_diagnostics": pred_diagnostics,
        },
        "principle_c": {
            "n_tests": int(len(pc_stats)),
            "n_significant_fdr": int((pc_stats["p_fdr_bh"] < 0.05).sum()) if not pc_stats.empty else 0,
            "fit_method_counts": _fit_method_counts(pc_stats),
            "first_fdr_layer": pc_first_sig,
            "peak": pc_peak,
        },
        "bound_var": {
            "n_tests": int(len(bv_stats)),
            "n_significant_fdr": int((bv_stats["p_fdr_bh"] < 0.05).sum()) if not bv_stats.empty else 0,
            "fit_method_counts": _fit_method_counts(bv_stats),
            "first_fdr_layer_by_pair": bv_first_sig_by_pair,
            "peak_by_pair": bv_peaks_by_pair,
        },
        "prediction_warnings": pred_warnings,
    }
    summary_path = output_dir / "cc_summary.json"
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
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--apply-run", type=Path, default=None)
    p.add_argument("--verify-run", type=Path, default=None)
    p.add_argument(
        "--apply-prefix", type=str, default=_APPLY_RUN_PREFIX,
        help="Prefix used to discover apply-probes runs. Use e.g. "
             "'c_command_qwen25_7b-' to pin a specific model.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
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


def _print_cc_headline(summary: dict, output_dir: Path) -> None:
    """Print human-readable headline for one c-command analysis run."""
    print("\n" + "=" * 64)
    print(f"  apply_run:           {Path(str(summary['apply_run'])).name}")
    print(f"  verify_run:          {Path(str(summary['verify_run'])).name}")
    print(f"  items used / total:  {summary['n_items_used']} / {summary['n_items_total']}")
    print(f"  sub-experiments:     {summary['subexperiment_counts']}")
    print(f"  fit method:          {summary['fit_method_requested']}")
    print()
    refl = summary["reflexive"]
    print(f"  reflexive: {refl['n_significant_fdr']}/{refl['n_tests']} FDR-sig  "
          f"fit_methods={refl['fit_method_counts']}")
    if refl["headline_peak"]:
        peak = refl["headline_peak"]
        print(f"    headline 3-way peak: layer {peak['layer_index']} ({peak['modifier_type']})  "
              f"β={peak['estimate']:+.4f}  z={peak['z_value']:+.2f}")
    if refl["headline_first_fdr_layer"] is not None:
        print(f"    first FDR-sig layer: {refl['headline_first_fdr_layer']}")
    pc = summary["principle_c"]
    print(f"  principle_c: {pc['n_significant_fdr']}/{pc['n_tests']} FDR-sig  "
          f"fit_methods={pc['fit_method_counts']}")
    if pc["peak"]:
        peak = pc["peak"]
        print(f"    peak: layer {peak['layer_index']}  β={peak['estimate']:+.4f}  "
              f"z={peak['z_value']:+.2f}")
    bv = summary["bound_var"]
    print(f"  bound_var: {bv['n_significant_fdr']}/{bv['n_tests']} FDR-sig  "
          f"fit_methods={bv['fit_method_counts']}")
    for pair, peak in bv["peak_by_pair"].items():
        if peak:
            print(f"    peak ({pair}): layer {peak['layer_index']}  "
                  f"β={peak['estimate']:+.4f}  z={peak['z_value']:+.2f}")
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
                model, "c_command", args.results_dir,
            )
            if apply_run is None:
                logger.info("No c_command run for %s — skipping.", model.slug)
                skipped.append(model.slug)
                continue

            print(f"\n{'─' * 64}")
            print(f"  model: {model.display}  ({apply_run.name})")
            print(f"{'─' * 64}")
            try:
                _, verify_run = discover_runs(
                    args.results_dir,
                    apply_prefix=f"c_command_{model.slug}-",
                    verify_prefix=_VERIFY_RUN_PREFIX,
                    apply_run=apply_run,
                    verify_run=args.verify_run,
                )
                summary = run_analysis(
                    apply_run, verify_run, apply_run,
                    n_bootstrap=args.n_bootstrap,
                    fit_method=args.fit_method,
                )
                _print_cc_headline(summary, apply_run)
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
    _print_cc_headline(summary, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
