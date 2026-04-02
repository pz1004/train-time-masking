#!/usr/bin/env python3
"""Generate the practical-significance forest plot for the manuscript."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT, "paper", "submission_summary", "robustness_advantages.json")
OUTPUT_DIRS = [
    os.path.join(ROOT, "figures"),
    os.path.join(ROOT, "paper", "figures"),
]
OUTPUT_BASENAME = "robustness_advantage_forest"
DELTA_THRESHOLD = 0.005

DATASET_ORDER = [
    "Adult",
    "German Credit",
    "Bank Marketing",
    "Give Me Some Credit",
    "Covertype",
]

COMPARATOR_ORDER = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "Random Forest",
]

COMPARATOR_COLORS = {
    "LightGBM": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "CatBoost": "#d62728",
    "Random Forest": "#9467bd",
}


def _load_rows() -> list[dict[str, object]]:
    with open(INPUT_PATH, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list payload in {INPUT_PATH}")
    return rows


def _ordered_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    indexed = {
        (str(row["dataset_label"]), str(row["comparator_label"])): row
        for row in rows
    }
    ordered: list[dict[str, object]] = []
    dataset_centers: dict[str, float] = {}
    y_position = 0.0
    for dataset_label in DATASET_ORDER:
        dataset_rows: list[dict[str, object]] = []
        dataset_positions: list[float] = []
        for comparator_label in COMPARATOR_ORDER:
            key = (dataset_label, comparator_label)
            if key not in indexed:
                raise KeyError(f"Missing robustness-advantage row for {key}")
            row = dict(indexed[key])
            row["y_position"] = y_position
            dataset_rows.append(row)
            dataset_positions.append(y_position)
            y_position += 1.0
        ordered.extend(dataset_rows)
        dataset_centers[dataset_label] = sum(dataset_positions) / len(dataset_positions)
        y_position += 0.8
    return ordered, dataset_centers


def _plot(rows: list[dict[str, object]], dataset_centers: dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8.5))
    plt.subplots_adjust(left=0.33, right=0.98, top=0.97, bottom=0.10)

    group_bounds: list[tuple[float, float]] = []
    for dataset_label in DATASET_ORDER:
        dataset_rows = [row for row in rows if row["dataset_label"] == dataset_label]
        y_values = [float(row["y_position"]) for row in dataset_rows]
        group_bounds.append((min(y_values) - 0.5, max(y_values) + 0.5))

    for group_index, (y_min, y_max) in enumerate(group_bounds):
        if group_index % 2 == 0:
            ax.axhspan(y_min, y_max, color="#f6f6f6", zorder=0)
        ax.axhline(y_max, color="#dddddd", linewidth=0.8, zorder=1)

    for row in rows:
        mean_advantage = float(row["mean_advantage"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        comparator_label = str(row["comparator_label"])
        y_position = float(row["y_position"])
        color = COMPARATOR_COLORS[comparator_label]
        ax.errorbar(
            mean_advantage,
            y_position,
            xerr=[[mean_advantage - ci_low], [ci_high - mean_advantage]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.5,
            markersize=6.5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            zorder=3,
        )

    ax.axvline(0.0, color="black", linewidth=1.0, zorder=2)
    ax.axvline(DELTA_THRESHOLD, color="crimson", linestyle="--", linewidth=1.6, zorder=2)

    all_ci_lows = [float(row["ci_low"]) for row in rows]
    all_ci_highs = [float(row["ci_high"]) for row in rows]
    x_min = min(-0.01, min(all_ci_lows) - 0.004)
    x_max = max(all_ci_highs) + 0.01
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Practical robustness advantage at 30% MCAR overlay", fontsize=11)
    ax.set_yticks([float(row["y_position"]) for row in rows])
    ax.set_yticklabels([str(row["comparator_label"]) for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for dataset_label, center in dataset_centers.items():
        ax.text(
            -0.27,
            center,
            dataset_label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7, markeredgecolor="black",
               markeredgewidth=0.4, color=COMPARATOR_COLORS[label], label=label)
        for label in COMPARATOR_ORDER
    ]
    legend_handles.append(
        Line2D([0], [0], color="crimson", linestyle="--", linewidth=1.6, label=r"$\delta = 0.005$")
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)

    return fig


def main() -> None:
    rows = _load_rows()
    ordered_rows, dataset_centers = _ordered_rows(rows)
    figure = _plot(ordered_rows, dataset_centers)
    for output_dir in OUTPUT_DIRS:
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"{OUTPUT_BASENAME}.pdf")
        png_path = os.path.join(output_dir, f"{OUTPUT_BASENAME}.png")
        figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
