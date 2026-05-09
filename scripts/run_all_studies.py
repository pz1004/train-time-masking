from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lab.study import RESULTS_DIR_PREFIX_ENV


ROOT = Path(__file__).resolve().parents[1]
LAMBDA_SWEEP_STUDY_CONFIGS = {
    "lineage_d9_06_adult_missingness_robustness.toml",
    "lineage_d9_10_covertype_missingness_robustness.toml",
}

CORE_SCRIPTS = (
    "run_baselines.py",
    "run_method.py",
    "run_ablations.py",
    "evaluate_robustness.py",
    "evaluate_calibration.py",
    "aggregate_results.py",
    "make_tables.py",
    "make_figures.py",
    "audit_results.py",
)

EXTRA_SCRIPTS = (
    "run_significance.py",
    "run_mask_sweep.py",
    "evaluate_mar.py",
)

REVISION_SCRIPTS = (
    "evaluate_structured_missingness.py",
    "run_leakage_ablation.py",
    "run_feature_stability.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full missingness-robustness study suite across multiple datasets.")
    parser.add_argument(
        "--study-glob",
        default="configs/studies/*missingness_robustness*.toml",
        help="Glob for the study config files to execute.",
    )
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help=(
            "Also run significance, mask sweep, MAR evaluation, the Adult/Covertype lambda sweeps, "
            "and the manuscript-facing cross-study summaries and canonical degradation figures."
        ),
    )
    parser.add_argument(
        "--revision-full",
        action="store_true",
        help=(
            "Run reviewer-response expansion stages: structured MAR/MNAR overlays, leakage ablation, "
            "feature-stability analysis, all-model significance/ranks, and revision table aggregation."
        ),
    )
    parser.add_argument(
        "--results-prefix",
        default="",
        help=(
            "Optional folder-name prefix for results directories. With --revision-full, defaults to "
            "YYYYmmdd_HHMMSS_revision_full so existing results/<study_id> folders are left untouched."
        ),
    )
    parser.add_argument(
        "--skip-submission-summary",
        action="store_true",
        help="Skip the final cross-study submission summary aggregation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    args = parser.parse_args(argv)

    study_configs = sorted(glob(args.study_glob))
    if not study_configs:
        raise SystemExit(f"No study configs matched: {args.study_glob}")

    script_names = list(CORE_SCRIPTS)
    if args.include_extras:
        script_names.extend(EXTRA_SCRIPTS)
    if args.revision_full:
        script_names.extend(REVISION_SCRIPTS)

    child_env = os.environ.copy()
    if args.results_prefix.strip():
        results_prefix = args.results_prefix.strip()
    elif args.revision_full:
        results_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_revision_full"
    else:
        results_prefix = ""
    if results_prefix:
        child_env[RESULTS_DIR_PREFIX_ENV] = results_prefix
        print(f"# Results directory prefix: {results_prefix}")

    for study_config in study_configs:
        _print_header(study_config)
        for script_name in script_names:
            _run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts" / script_name),
                    "--study-config",
                    study_config,
                ],
                dry_run=args.dry_run,
                env=child_env,
            )
        if args.include_extras and Path(study_config).name in LAMBDA_SWEEP_STUDY_CONFIGS:
            _run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_lambda_sweep.py"),
                    "--study-config",
                    study_config,
                ],
                dry_run=args.dry_run,
                env=child_env,
            )

    if args.include_extras:
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_mar_results.py"),
                "--study-glob",
                args.study_glob,
            ],
            dry_run=args.dry_run,
            env=child_env,
        )
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_robustness_advantages.py"),
                "--study-glob",
                args.study_glob,
            ],
            dry_run=args.dry_run,
            env=child_env,
        )
    if args.revision_full:
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_all_model_significance.py"),
                "--study-glob",
                args.study_glob,
            ],
            dry_run=args.dry_run,
            env=child_env,
        )
    if not args.skip_submission_summary:
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_submission_results.py"),
                "--study-glob",
                args.study_glob,
            ],
            dry_run=args.dry_run,
            env=child_env,
        )
    if args.revision_full:
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_revision_results.py"),
                "--study-glob",
                args.study_glob,
            ],
            dry_run=args.dry_run,
            env=child_env,
        )
    if args.include_extras:
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "make_composite_figures.py"),
            ],
            dry_run=args.dry_run,
            env=child_env,
        )
    return 0


def _run_command(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    pretty = " ".join(_quote(part) for part in command)
    print(pretty)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _print_header(study_config: str) -> None:
    print()
    print(f"# {study_config}")


def _quote(token: str) -> str:
    if all(character.isalnum() or character in "/._-:=*" for character in token):
        return token
    return repr(token)


if __name__ == "__main__":
    raise SystemExit(main())
