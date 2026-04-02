#!/usr/bin/env python3
"""Generate alternate degradation-curve figures with ±1 SE confidence bands.

The canonical manuscript degradation PDFs are produced by
`scripts/make_composite_figures.py`. This script keeps a confidence-band variant
available under distinct filenames so there is only one manuscript-output build
path.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "paper", "figures")

DATASETS = [
    ("Adult", "lineage_d9_06_adult_missingness_robustness"),
    ("German Credit", "lineage_d9_07_credit_g_missingness_robustness"),
    ("Bank Marketing", "lineage_d9_08_bank_marketing_missingness_robustness"),
    ("Give Me Some Credit", "lineage_d9_09_give_me_some_credit_missingness_robustness"),
    ("Covertype", "lineage_d9_10_covertype_missingness_robustness"),
]

MODELS = [
    ("mask_augmented_imputation_training", "MAIT", "#1f77b4", "o", "-"),
    ("lightgbm", "LightGBM", "#ff7f0e", "s", "--"),
    ("xgboost", "XGBoost", "#2ca02c", "^", "--"),
    ("catboost", "CatBoost", "#d62728", "D", "--"),
    ("random_forest", "Random Forest", "#9467bd", "v", "--"),
    ("logistic_regression", "LR (floor)", "#8c564b", "x", ":"),
]

N_SEEDS = 20
SEVERITIES = ["missingness_10", "missingness_20", "missingness_30"]
OVERLAY_RATES = [0, 10, 20, 30]


def load_data():
    """Load nominal and robustness data for all datasets."""
    all_data = {}
    for dname, ddir in DATASETS:
        fpath = os.path.join(ROOT, "results", ddir, "aggregated", "performance_summary.json")
        with open(fpath) as f:
            data = json.load(f)
        all_data[dname] = data
    return all_data


def extract_trajectories(data, mode="absolute"):
    """Extract mean and SE trajectories for each model.

    mode="absolute": returns raw AUROC values
    mode="delta": returns ΔAUROC from nominal
    """
    results = {}
    bs = data.get("baseline_summary", {})
    ms = data.get("method_summary", {})
    rs = data.get("robustness_summary", {})

    all_models = {}
    all_models.update(bs)
    all_models.update(ms)

    for model_key, label, color, marker, ls in MODELS:
        if model_key not in all_models:
            continue

        nominal_mean = all_models[model_key]["mean_auroc"]
        nominal_std = all_models[model_key]["std_auroc"]
        nominal_se = nominal_std / math.sqrt(N_SEEDS)

        means = [nominal_mean]
        ses = [nominal_se]

        rob = rs.get(model_key, {})
        for sev in SEVERITIES:
            if sev in rob:
                if mode == "absolute":
                    means.append(rob[sev]["mean_auroc"])
                    ses.append(rob[sev]["std_auroc"] / math.sqrt(N_SEEDS))
                else:  # delta
                    means.append(rob[sev]["mean_auroc_delta"])
                    ses.append(rob[sev]["std_auroc_delta"] / math.sqrt(N_SEEDS))
            else:
                means.append(float("nan"))
                ses.append(float("nan"))

        if mode == "delta":
            means[0] = 0.0
            ses[0] = 0.0

        results[model_key] = {
            "label": label,
            "color": color,
            "marker": marker,
            "ls": ls,
            "means": np.array(means),
            "ses": np.array(ses),
        }
    return results


def plot_figure(all_data, mode, output_path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=False)
    fig.subplots_adjust(wspace=0.35, bottom=0.22, top=0.88)

    for idx, (dname, _) in enumerate(DATASETS):
        ax = axes[idx]
        data = all_data[dname]
        trajectories = extract_trajectories(data, mode=mode)

        for model_key, label, color, marker, ls in MODELS:
            if model_key not in trajectories:
                continue
            t = trajectories[model_key]
            ax.plot(OVERLAY_RATES, t["means"], color=t["color"], marker=t["marker"],
                    markersize=5, linewidth=1.5, linestyle=t["ls"], label=t["label"],
                    zorder=3 if model_key == "mask_augmented_imputation_training" else 2)
            ax.fill_between(OVERLAY_RATES,
                            t["means"] - t["ses"],
                            t["means"] + t["ses"],
                            color=t["color"], alpha=0.15, zorder=1)

        ax.set_title(dname, fontsize=10, fontweight="bold")
        ax.set_xlabel("MCAR Overlay (%)", fontsize=9)
        ax.set_xticks(OVERLAY_RATES)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if idx == 0:
            if mode == "absolute":
                ax.set_ylabel("AUROC", fontsize=10)
            else:
                ax.set_ylabel("ΔAUROC", fontsize=10)

    # Single shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    all_data = load_data()
    plot_figure(all_data, "absolute",
                os.path.join(FIGURES_DIR, "degradation_curves_confidence_band_absolute.pdf"))
    plot_figure(all_data, "delta",
                os.path.join(FIGURES_DIR, "degradation_curves_confidence_band_delta.pdf"))


if __name__ == "__main__":
    main()
