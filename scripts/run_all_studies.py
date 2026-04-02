from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
import subprocess
import sys


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
        )
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_robustness_advantages.py"),
                "--study-glob",
                args.study_glob,
            ],
            dry_run=args.dry_run,
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
        )
    if args.include_extras:
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "make_composite_figures.py"),
            ],
            dry_run=args.dry_run,
        )
    return 0


def _run_command(command: list[str], *, dry_run: bool) -> None:
    pretty = " ".join(_quote(part) for part in command)
    print(pretty)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def _print_header(study_config: str) -> None:
    print()
    print(f"# {study_config}")


def _quote(token: str) -> str:
    if all(character.isalnum() or character in "/._-:=*" for character in token):
        return token
    return repr(token)


if __name__ == "__main__":
    raise SystemExit(main())
