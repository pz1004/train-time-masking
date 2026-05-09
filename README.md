# When Does Train-Time Masking Help?

Official repository for the paper being submitted. This repository contains the implementation, study configurations, reproducibility scripts, rendered figures, and tests for the missingness-robustness experiments.

## Repository Contents

- `src/lab/`: MAIT, baseline wrappers, evaluation logic, reporting, and study orchestration
- `configs/`: dataset, protocol, method, robustness, reporting, ablation, and study TOML files
- `scripts/`: experiment, audit, aggregation, and figure-generation entrypoints
- `figures/`: rendered degradation and robustness-advantage figures
- `tests/`: regression and consistency checks for the released code path

Generated experiment outputs are written under `results/` and generated paper artifacts under `paper/`; both are intentionally ignored so the public repository remains compact.

## Setup

Run commands from the repository root:

```bash
python -m pip install -r requirements-research.txt
export PYTHONPATH=src
```

The experiments fetch benchmark data from OpenML at runtime. Optional modern baselines use the optional dependencies listed in `requirements-research.txt`.

## Reproduce the Study

Run the full five-dataset workflow:

```bash
python scripts/run_all_studies.py --include-extras
```

To rerun one study at a time, replace `$STUDY` with a file under `configs/studies/`:

```bash
python scripts/run_baselines.py --study-config "$STUDY"
python scripts/run_method.py --study-config "$STUDY"
python scripts/run_ablations.py --study-config "$STUDY"
python scripts/evaluate_robustness.py --study-config "$STUDY"
python scripts/evaluate_calibration.py --study-config "$STUDY"
python scripts/aggregate_results.py --study-config "$STUDY"
python scripts/make_tables.py --study-config "$STUDY"
python scripts/make_figures.py --study-config "$STUDY"
python scripts/audit_results.py --study-config "$STUDY"
python scripts/run_significance.py --study-config "$STUDY"
python scripts/run_mask_sweep.py --study-config "$STUDY"
python scripts/evaluate_mar.py --study-config "$STUDY"
python scripts/evaluate_structured_missingness.py --study-config "$STUDY"
python scripts/run_leakage_ablation.py --study-config "$STUDY"
python scripts/run_feature_stability.py --study-config "$STUDY"
```

Cross-study summaries and figures can be regenerated after the per-study artifacts exist:

```bash
python scripts/aggregate_mar_results.py --study-glob 'configs/studies/*missingness_robustness*.toml'
python scripts/aggregate_robustness_advantages.py --study-glob 'configs/studies/*missingness_robustness*.toml'
python scripts/aggregate_submission_results.py --study-glob 'configs/studies/*missingness_robustness*.toml'
python scripts/run_all_model_significance.py --study-glob 'configs/studies/*missingness_robustness*.toml'
python scripts/make_composite_figures.py
```

## Checks

```bash
export PYTHONPATH=src
python -m py_compile scripts/*.py src/lab/**/*.py
python -m pytest tests/test_pipeline_refactor.py tests/test_draft_code_consistency.py
python scripts/run_all_studies.py --include-extras --dry-run
```
