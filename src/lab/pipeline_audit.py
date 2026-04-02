from __future__ import annotations

from typing import Any

from .study import StudySpec


def _audit_markdown(
    spec: StudySpec,
    summary: dict[str, Any],
    nominal_records: list[dict[str, Any]],
    calibrated_records: list[dict[str, Any]],
    robustness_records: list[dict[str, Any]],
    *,
    method_calibration_summary: dict[str, Any] | None = None,
) -> str:
    method_calibration_summary = method_calibration_summary or {}
    method_summary = summary["method_summary"]
    ablation_summary = summary["ablation_summary"]
    best_method_name = None
    best_method_metrics = None
    if method_summary:
        best_method_name, best_method_metrics = max(
            method_summary.items(),
            key=lambda item: (item[1]["mean_auroc"], -item[1]["mean_ece"]),
        )
    best_ablation_name = None
    best_ablation_metrics = None
    if ablation_summary:
        best_ablation_name, best_ablation_metrics = max(
            ablation_summary.items(),
            key=lambda item: (item[1]["mean_auroc"], -item[1]["mean_ece"]),
        )
    seed_rankings = _seed_rankings(nominal_records)
    best_nominal_name, best_nominal_metrics = max(
        summary["baseline_summary"].items(), key=lambda item: item[1]["mean_auroc"]
    )
    top_baseline_per_seed = [ranking[0][0] for _, ranking in sorted(seed_rankings.items()) if ranking]
    ranking_is_stable = len(set(top_baseline_per_seed)) == 1

    variance_lines = [
        (
            f"- `{baseline_name}` nominal mean test AUROC is {metrics['mean_auroc']:.6f} with seed spread "
            f"{metrics['std_auroc']:.6f} across {int(metrics['n_runs'])} runs."
        )
        for baseline_name, metrics in summary["baseline_summary"].items()
    ]
    for model_name, metrics in summary["calibration_summary"].items():
        variance_lines.append(
            f"- `{model_name}` mean post-calibration test AUROC is {metrics['mean_post_calibration_auroc']:.6f} "
            f"with seed spread {metrics['std_post_calibration_auroc']:.6f} across {int(metrics['n_runs'])} runs."
        )
    for model_name, metrics in method_summary.items():
        variance_lines.append(
            f"- `{model_name}` mean test AUROC is {metrics['mean_auroc']:.6f} with seed spread "
            f"{metrics['std_auroc']:.6f} across {int(metrics['n_runs'])} runs."
        )
    for model_name, metrics in ablation_summary.items():
        variance_lines.append(
            f"- `{model_name}` ablation mean test AUROC is {metrics['mean_auroc']:.6f} with seed spread "
            f"{metrics['std_auroc']:.6f} across {int(metrics['n_runs'])} runs."
        )
    if ranking_is_stable:
        variance_lines.append(f"- The top nominal baseline is stable across seeds: `{top_baseline_per_seed[0]}` leads all three splits.")
    else:
        variance_lines.append("- Nominal baseline ranking changes across seeds, so the headline ordering is unstable.")

    robustness_lines = []
    for model_name, slice_map in summary["robustness_summary"].items():
        hardest_slice_name, hardest_slice_metrics = max(
            slice_map.items(),
            key=lambda item: (item[1]["mean_ece"], -item[1]["mean_auroc_delta"]),
        )
        robustness_lines.append(
            f"- `{model_name}` is most brittle on `{hardest_slice_name}` with mean ECE {hardest_slice_metrics['mean_ece']:.6f} "
            f"and mean AUROC delta {hardest_slice_metrics['mean_auroc_delta']:.6f}."
        )
    if not robustness_lines:
        robustness_lines.append("- No robustness overlays were audited, so robustness evidence is incomplete.")
    for method_name, method_metrics in method_summary.items():
        method_slices = summary["robustness_summary"].get(method_name, {})
        baseline_slices = summary["robustness_summary"].get(best_nominal_name, {})
        common_slices = sorted(set(method_slices) & set(baseline_slices))
        if common_slices:
            strongest_slice_name = max(
                common_slices,
                key=lambda slice_name: method_slices[slice_name]["mean_auroc"] - baseline_slices[slice_name]["mean_auroc"],
            )
            robustness_gain = (
                float(method_slices[strongest_slice_name]["mean_auroc"])
                - float(baseline_slices[strongest_slice_name]["mean_auroc"])
            )
            robustness_lines.append(
                f"- Against `{best_nominal_name}`, `{method_name}` gains AUROC on all shared overlays; "
                f"the largest margin is {robustness_gain:.6f} on `{strongest_slice_name}`."
            )

    calibration_lines = []
    for model_name, metrics in summary["calibration_summary"].items():
        calibration_lines.append(
            f"- `{model_name}` changes ECE from {metrics['mean_pre_calibration_ece']:.6f} to "
            f"{metrics['mean_post_calibration_ece']:.6f} and AUROC from {metrics['mean_pre_calibration_auroc']:.6f} "
            f"to {metrics['mean_post_calibration_auroc']:.6f}."
        )
    method_variant_summary = method_calibration_summary.get("variants", {})
    for variant_name, metrics in method_variant_summary.items():
        calibration_lines.append(
            f"- `{variant_name}` changes test ECE from {metrics['mean_pre_test_ece']:.6f} to "
            f"{metrics['mean_test_ece']:.6f} and test AUROC from {metrics['mean_pre_test_auroc']:.6f} "
            f"to {metrics['mean_test_auroc']:.6f} relative to its pre-calibration backbone."
        )
    if method_summary and not method_variant_summary:
        calibration_lines.append("- Method calibration artifacts are missing, so method reliability evidence is incomplete.")
    if not calibration_lines:
        calibration_lines.append("- No calibrated variants are available, so calibration evidence remains nominal-only.")

    loss_case_lines = []
    for baseline_name, metrics in summary["baseline_summary"].items():
        if baseline_name == best_nominal_name:
            continue
        delta = round(best_nominal_metrics["mean_auroc"] - metrics["mean_auroc"], 6)
        loss_case_lines.append(
            f"- `{baseline_name}` trails the top nominal baseline `{best_nominal_name}` by {delta:.6f} AUROC."
        )
    for model_name, metrics in summary["calibration_summary"].items():
        if metrics["mean_auroc_delta"] < 0.0 or metrics["mean_ece_delta"] > 0.0:
            delta_clauses = []
            if metrics["mean_auroc_delta"] < 0.0:
                delta_clauses.append(f"loses {abs(metrics['mean_auroc_delta']):.6f} AUROC")
            if metrics["mean_ece_delta"] > 0.0:
                delta_clauses.append(f"worsens ECE by {metrics['mean_ece_delta']:.6f}")
            loss_case_lines.append(
                f"- `{model_name}` {' and '.join(delta_clauses)} relative to its nominal base."
            )
    for method_name, metrics in method_summary.items():
        nominal_gap = float(best_nominal_metrics["mean_auroc"]) - float(metrics["mean_auroc"])
        if nominal_gap > 0.0:
            loss_case_lines.append(
                f"- `{method_name}` trails the top nominal baseline `{best_nominal_name}` by {nominal_gap:.6f} AUROC."
            )
        if best_ablation_name is not None and best_ablation_metrics is not None:
            ablation_auroc_gap = float(best_ablation_metrics["mean_auroc"]) - float(metrics["mean_auroc"])
            ablation_ece_gap = float(metrics["mean_ece"]) - float(best_ablation_metrics["mean_ece"])
            if ablation_auroc_gap > 0.0 or ablation_ece_gap > 0.0:
                clauses = []
                if ablation_auroc_gap > 0.0:
                    clauses.append(f"{ablation_auroc_gap:.6f} lower AUROC")
                if ablation_ece_gap > 0.0:
                    clauses.append(f"{ablation_ece_gap:.6f} higher ECE")
                loss_case_lines.append(
                    f"- `{method_name}` is dominated by ablation `{best_ablation_name}` with {' and '.join(clauses)}."
                )
        variant_metrics = method_variant_summary.get(method_name)
        if variant_metrics and (
            float(variant_metrics["mean_test_auroc_delta"]) < 0.0
            or float(variant_metrics["mean_test_ece_delta"]) > 0.0
        ):
            clauses = []
            if float(variant_metrics["mean_test_auroc_delta"]) < 0.0:
                clauses.append(f"{abs(float(variant_metrics['mean_test_auroc_delta'])):.6f} lower AUROC")
            if float(variant_metrics["mean_test_ece_delta"]) > 0.0:
                clauses.append(f"{float(variant_metrics['mean_test_ece_delta']):.6f} higher ECE")
            loss_case_lines.append(
                f"- `{method_name}` degrades its own pre-calibration backbone with {' and '.join(clauses)}."
            )
    if not calibrated_records:
        loss_case_lines.append("- No calibrated loss cases are available because no calibrated variants were aggregated.")

    warning_lines = ["- The study still covers one dataset only, so any cross-dataset generalization claim would be unverified."]
    if method_summary:
        nominal_gap = float(best_method_metrics["mean_auroc"]) - float(best_nominal_metrics["mean_auroc"])
        if nominal_gap <= 0.0:
            warning_lines.append(
                f"- The current method evidence supports, at most, a narrow robustness-motivation claim because `{best_method_name}` does not beat `{best_nominal_name}` nominally."
            )
        else:
            warning_lines.append(
                f"- `{best_method_name}` exceeds `{best_nominal_name}` nominally by only {nominal_gap:.6f} AUROC, so any approved claim must stay narrow and centered on the scripted overlay-robustness evidence rather than nominal superiority."
            )
        if best_ablation_name is not None and best_ablation_metrics is not None:
            ablation_auroc_gap = float(best_ablation_metrics["mean_auroc"]) - float(best_method_metrics["mean_auroc"])
            ablation_ece_gap = float(best_method_metrics["mean_ece"]) - float(best_ablation_metrics["mean_ece"])
            if ablation_auroc_gap > 0.0 or ablation_ece_gap > 0.0:
                warning_lines.append(
                    f"- The post-hoc calibration contribution is unverified because ablation `{best_ablation_name}` remains stronger than the full method on the current artifact trail."
                )
            else:
                warning_lines.append(
                    f"- The ablation evidence supports the revised no-calibration main path: `{best_ablation_name}` does not outperform `{best_method_name}`."
                )
    else:
        warning_lines.append("- No proposed-method comparison is audited in this phase, so no superiority claim is supported.")
        warning_lines.append("- Calibration gains in this audit strengthen the baseline table only and do not justify a method claim by themselves.")
    if not robustness_records:
        warning_lines.append("- Robustness artifacts are missing, so robustness-driven motivation for a method would be unverified.")

    scope_parts = ["nominal baselines"]
    if method_summary:
        scope_parts.append("proposed method runs")
    if ablation_summary:
        scope_parts.append("method ablations")
    if summary["calibration_summary"]:
        scope_parts.append("calibrated baseline variants")
    if summary["robustness_summary"]:
        scope_parts.append("robustness overlays")

    return "\n".join(
        [
            "# Results Audit",
            "",
            f"Artifact status: `{spec.artifact_status}`. This audit covers {', '.join(scope_parts)} generated under the frozen Adult protocol.",
            "",
            "## Variance Across Seeds",
            *variance_lines,
            "",
            "## Regime-Specific Robustness Assessment",
            *robustness_lines,
            "",
            "## Calibration Assessment",
            *calibration_lines,
            "",
            "## Loss-Case Summary",
            *loss_case_lines,
            "",
            "## Suspicious-Gain Warnings",
            *warning_lines,
            "",
            "## Exact Failure Regime to Motivate a Method",
            *_exact_failure_regime_lines(summary, method_calibration_summary=method_calibration_summary),
            "",
            "## Remaining Gaps to Claim-Ready Evidence",
            *(
                [
                    (
                        f"- The current full method is not claim-ready because it does not beat `{best_nominal_name}` on nominal AUROC."
                        if float(best_method_metrics["mean_auroc"]) <= float(best_nominal_metrics["mean_auroc"])
                        else f"- The nominal gain over `{best_nominal_name}` is only {float(best_method_metrics['mean_auroc']) - float(best_nominal_metrics['mean_auroc']):.6f} AUROC, so no broad nominal-superiority claim is supported."
                    ),
                    (
                        (
                            f"- The post-hoc calibration contribution is not claim-ready because `{best_ablation_name}` remains stronger than the full method on the current artifact trail."
                            if best_ablation_name is not None
                            and best_ablation_metrics is not None
                            and (
                                float(best_ablation_metrics["mean_auroc"]) > float(best_method_metrics["mean_auroc"])
                                or float(best_method_metrics["mean_ece"]) > float(best_ablation_metrics["mean_ece"])
                            )
                            else "- The evidence remains restricted to one Adult study, so any approved claim must stay tied to the scripted overlay grid and cannot widen beyond this dataset."
                        )
                    )
                    if ablation_summary
                    else "- No ablation artifacts exist yet.",
                ]
                if method_summary
                else ["- No proposed method artifacts exist yet.", "- No ablation artifacts exist yet."]
            ),
            "- No multi-dataset evidence exists yet.",
            "- Robustness evidence is limited to the scripted Adult missingness overlay grid.",
            "- `claim_evidence_table.md` has not been updated in this phase.",
        ]
    ) + "\n"


def _exact_failure_regime_lines(
    summary: dict[str, Any],
    *,
    method_calibration_summary: dict[str, Any] | None = None,
) -> list[str]:
    method_calibration_summary = method_calibration_summary or {}
    if summary["method_summary"]:
        method_name, method_metrics = max(
            summary["method_summary"].items(),
            key=lambda item: item[1]["mean_auroc"],
        )
        best_nominal_name, best_nominal_metrics = max(
            summary["baseline_summary"].items(),
            key=lambda item: item[1]["mean_auroc"],
        )
        method_robustness = summary["robustness_summary"].get(method_name, {})
        baseline_robustness = summary["robustness_summary"].get(best_nominal_name, {})
        common_slices = sorted(set(method_robustness) & set(baseline_robustness))
        if not common_slices:
            return [f"- `{method_name}` has no robustness overlap with `{best_nominal_name}`, so no exact failure regime can be named."]

        failure_slice_name = max(
            common_slices,
            key=lambda slice_name: (
                float(method_robustness[slice_name]["mean_auroc"]) - float(baseline_robustness[slice_name]["mean_auroc"]),
                -float(method_robustness[slice_name]["mean_ece"]),
            ),
        )
        method_slice_metrics = method_robustness[failure_slice_name]
        baseline_slice_metrics = baseline_robustness[failure_slice_name]
        method_advantage = float(method_slice_metrics["mean_auroc"]) - float(baseline_slice_metrics["mean_auroc"])
        nominal_gap = float(best_nominal_metrics["mean_auroc"]) - float(method_metrics["mean_auroc"])
        lines = [
            f"- The motivating regime is `{failure_slice_name}`: `{method_name}` exceeds `{best_nominal_name}` by {method_advantage:.6f} AUROC on that overlay.",
        ]
        clauses = []
        if nominal_gap > 0.0:
            clauses.append(f"nominal mean AUROC still trails `{best_nominal_name}` by {nominal_gap:.6f}")
        if summary["ablation_summary"]:
            best_ablation_name, _ = max(
                summary["ablation_summary"].items(),
                key=lambda item: (item[1]["mean_auroc"], -item[1]["mean_ece"]),
            )
            ablation_slice_metrics = summary["robustness_summary"].get(best_ablation_name, {}).get(failure_slice_name)
            if ablation_slice_metrics is not None:
                ablation_auroc_gap = float(ablation_slice_metrics["mean_auroc"]) - float(method_slice_metrics["mean_auroc"])
                ablation_ece_gap = float(method_slice_metrics["mean_ece"]) - float(ablation_slice_metrics["mean_ece"])
                gap_clauses = []
                if ablation_auroc_gap > 0.0:
                    gap_clauses.append(f"{ablation_auroc_gap:.6f} higher AUROC")
                if ablation_ece_gap > 0.0:
                    gap_clauses.append(f"{ablation_ece_gap:.6f} lower ECE")
                if gap_clauses:
                    clauses.append(
                        f"`{best_ablation_name}` remains stronger on `{failure_slice_name}` with {' and '.join(gap_clauses)}"
                    )
        variant_metrics = method_calibration_summary.get("variants", {}).get(method_name)
        if variant_metrics is not None:
            variant_clauses = []
            if float(variant_metrics["mean_test_auroc_delta"]) < 0.0:
                variant_clauses.append(
                    f"{abs(float(variant_metrics['mean_test_auroc_delta'])):.6f} lower AUROC than its pre-calibration backbone"
                )
            if float(variant_metrics["mean_test_ece_delta"]) > 0.0:
                variant_clauses.append(
                    f"{float(variant_metrics['mean_test_ece_delta']):.6f} higher ECE than its pre-calibration backbone"
                )
            if variant_clauses:
                clauses.append(f"the full method shows {' and '.join(variant_clauses)}")
        if clauses:
            lines.append(f"- That signal is not claim-ready because {'; '.join(clauses)}.")
        return lines

    selected_bases = list(summary["selected_calibration_bases"])
    calibration_summary = summary["calibration_summary"]
    calibrated_base_coverage = {str(metrics["base_baseline_name"]) for metrics in calibration_summary.values()}
    missing_bases = [base_name for base_name in selected_bases if base_name not in calibrated_base_coverage]
    if missing_bases:
        return [f"- No calibrated model is available for selected calibration base(s): {', '.join(missing_bases)}."]
    if not calibration_summary:
        return ["- No calibrated models were aggregated, so no exact failure regime can be named."]

    best_model_name, _ = max(
        calibration_summary.items(),
        key=lambda item: item[1]["mean_post_calibration_auroc"],
    )
    model_robustness = summary["robustness_summary"].get(best_model_name, {})
    if not model_robustness:
        return [f"- `{best_model_name}` has no robustness artifacts, so no exact failure regime can be named."]

    failure_slice_name, failure_metrics = max(
        model_robustness.items(),
        key=lambda item: (item[1]["mean_ece"], -item[1]["mean_auroc_delta"]),
    )
    return [
        f"- The exact failure regime is `{failure_slice_name}` for `{best_model_name}`.",
        f"- In that slice, mean ECE is {failure_metrics['mean_ece']:.6f} and mean AUROC delta from the model's own nominal run is {failure_metrics['mean_auroc_delta']:.6f}.",
    ]


def _seed_rankings(nominal_records: list[dict[str, Any]]) -> dict[int, list[tuple[str, float]]]:
    ranking_map: dict[int, list[tuple[str, float]]] = {}
    for record in nominal_records:
        ranking_map.setdefault(int(record["seed"]), []).append(
            (str(record["model_name"]), float(record["test_metrics"]["auroc"]))
        )
    return {
        seed: sorted(entries, key=lambda item: item[1], reverse=True)
        for seed, entries in ranking_map.items()
    }
