"""Generate composite multi-panel degradation-curve figures for the manuscript."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 20

STUDIES = [
    ("lineage_d9_06_adult_missingness_robustness", "Adult"),
    ("lineage_d9_07_credit_g_missingness_robustness", "German Credit"),
    ("lineage_d9_08_bank_marketing_missingness_robustness", "Bank Marketing"),
    ("lineage_d9_09_give_me_some_credit_missingness_robustness", "Give Me Some Credit"),
    ("lineage_d9_10_covertype_missingness_robustness", "Covertype"),
]

ABSOLUTE_MODELS = {
    "lightgbm": ("LightGBM", "#2ca02c", "s", "-"),
    "xgboost": ("XGBoost", "#d62728", "D", "-"),
    "catboost": ("CatBoost", "#9467bd", "^", "-"),
    "random_forest": ("Random Forest", "#8c564b", "v", "-"),
    "logistic_regression": ("Logistic Reg.", "#7f7f7f", "x", "-"),
    "mask_augmented_imputation_training": ("MAIT", "#1f77b4", "o", "-"),
}

DELTA_MODELS = {
    **ABSOLUTE_MODELS,
    "mlp_only": ("MLP-only", "#111111", "o", "--"),
}

OVERLAY_KEYS = ["missingness_10", "missingness_20", "missingness_30"]
OVERLAY_RATES = [0, 10, 20, 30]


def load_summary(study_id: str) -> dict | None:
    path = ROOT / "results" / study_id / "aggregated" / "performance_summary.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _nominal_record(summary: dict, model_key: str) -> dict | None:
    for group_key in ("baseline_summary", "method_summary", "ablation_summary"):
        group = summary.get(group_key, {})
        if model_key in group:
            return group[model_key]
    return None


def _stderr(std_value: float) -> float:
    return std_value / math.sqrt(N_SEEDS)


def extract_auroc_curve(summary: dict, model_key: str) -> tuple[list[float], list[float]] | None:
    """Return mean and stderr curves for [nominal, 10%, 20%, 30%] AUROC."""
    nominal = _nominal_record(summary, model_key)
    if nominal is None:
        return None

    robustness = summary.get("robustness_summary", {}).get(model_key, {})
    means = [nominal["mean_auroc"]]
    stderrs = [_stderr(nominal["std_auroc"])]
    for key in OVERLAY_KEYS:
        if key not in robustness:
            return None
        means.append(robustness[key]["mean_auroc"])
        stderrs.append(_stderr(robustness[key]["std_auroc"]))
    return means, stderrs


def extract_delta_curve(summary: dict, model_key: str) -> tuple[list[float], list[float]] | None:
    """Return mean and stderr curves for [0, delta_10, delta_20, delta_30]."""
    robustness = summary.get("robustness_summary", {}).get(model_key, {})
    means = [0.0]
    stderrs = [0.0]
    for key in OVERLAY_KEYS:
        if key not in robustness:
            return None
        means.append(robustness[key]["mean_auroc_delta"])
        stderrs.append(_stderr(robustness[key]["std_auroc_delta"]))
    return means, stderrs


def make_degradation_panel(output_path: Path, mode: str = "absolute") -> None:
    """Create a multi-panel figure. mode='absolute' for AUROC, mode='delta' for delta AUROC."""
    available = []
    for study_id, label in STUDIES:
        summary = load_summary(study_id)
        if summary is not None:
            available.append((study_id, label, summary))

    if not available:
        raise SystemExit("No aggregated summaries found.")

    models = ABSOLUTE_MODELS if mode == "absolute" else DELTA_MODELS
    n = len(available)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)

    for idx, (_, label, summary) in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for model_key, (model_label, color, marker, linestyle) in models.items():
            if mode == "absolute":
                curve = extract_auroc_curve(summary, model_key)
            else:
                curve = extract_delta_curve(summary, model_key)
            if curve is None:
                continue

            means, stderrs = curve
            lower = [m - s for m, s in zip(means, stderrs)]
            upper = [m + s for m, s in zip(means, stderrs)]

            ax.plot(
                OVERLAY_RATES,
                means,
                marker=marker,
                label=model_label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5.5,
                markeredgewidth=0.5,
                markeredgecolor="white",
            )
            ax.fill_between(OVERLAY_RATES, lower, upper, color=color, alpha=0.12, linewidth=0)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Additional MCAR Rate (%)", fontsize=10)
        ax.set_ylabel("AUROC" if mode == "absolute" else r"$\Delta$ AUROC", fontsize=10)
        ax.set_xticks(OVERLAY_RATES)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def output_dirs() -> list[Path]:
    return [ROOT / "paper" / "figures", ROOT / "figures"]


if __name__ == "__main__":
    for output_dir in output_dirs():
        output_dir.mkdir(parents=True, exist_ok=True)
        make_degradation_panel(output_dir / "degradation_curves_absolute.pdf", mode="absolute")
        make_degradation_panel(output_dir / "degradation_curves_delta.pdf", mode="delta")
        make_degradation_panel(output_dir / "degradation_curves_absolute.png", mode="absolute")
        make_degradation_panel(output_dir / "degradation_curves_delta.png", mode="delta")
