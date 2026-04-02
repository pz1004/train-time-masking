from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import shutil

from .study import StudySpec
from .toml_tools import dumps_toml


REQUIRED_STUDY_DOCS = (
    "problem_statement.md",
    "benchmark_protocol.md",
    "leakage_checklist.md",
    "experiment_plan.md",
    "hypotheses.md",
    "novelty_boundary.md",
)


def ensure_layout(spec: StudySpec) -> None:
    for path in (
        spec.research_dir,
        spec.results_dir,
        spec.manifests_dir,
        spec.raw_dir,
        spec.aggregated_dir,
        spec.audits_dir,
        spec.tables_dir,
        spec.figures_dir,
        spec.logs_dir,
        spec.paper_dir,
        spec.paper_tables_dir,
        spec.paper_figures_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_required_docs(spec: StudySpec) -> None:
    missing = [name for name in REQUIRED_STUDY_DOCS if not _is_nonempty(spec.doc_path(name))]
    if missing:
        raise RuntimeError(f"Missing required study docs for {spec.study_id}: {', '.join(missing)}")


def stage_completion_path(spec: StudySpec, stage_name: str) -> Path:
    return spec.logs_dir / f"{stage_name}_completion.json"


def stage_log_path(spec: StudySpec, stage_name: str) -> Path:
    return spec.logs_dir / f"{stage_name}.log"


def stage_seed_manifest_path(spec: StudySpec, stage_name: str) -> Path:
    return spec.manifests_dir / f"{stage_name}_seed_manifest.json"


def stage_manifest_path(spec: StudySpec, stage_name: str) -> Path:
    return spec.manifests_dir / f"{stage_name}_resolved_config.toml"


def stage_complete(spec: StudySpec, stage_name: str) -> bool:
    return stage_completion_path(spec, stage_name).exists()


def reset_stage_outputs(spec: StudySpec, stage_name: str) -> None:
    for path in (
        stage_completion_path(spec, stage_name),
        stage_log_path(spec, stage_name),
        stage_seed_manifest_path(spec, stage_name),
        stage_manifest_path(spec, stage_name),
    ):
        if path.exists():
            path.unlink()


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_stage_manifest(spec: StudySpec, stage_name: str) -> None:
    ensure_layout(spec)
    write_text(stage_manifest_path(spec, stage_name), dumps_toml(spec.resolved_config()))


def write_seed_manifest(spec: StudySpec, stage_name: str) -> None:
    payload = {
        "study_id": spec.study_id,
        "stage": stage_name,
        "artifact_status": spec.artifact_status,
        "seed_list": spec.seed_list,
    }
    write_json(stage_seed_manifest_path(spec, stage_name), payload)


def write_completion(spec: StudySpec, stage_name: str, artifact_paths: Iterable[Path], summary: str) -> None:
    payload = {
        "study_id": spec.study_id,
        "stage": stage_name,
        "status": "completed",
        "artifact_status": spec.artifact_status,
        "generated_at": _timestamp(),
        "artifacts": [str(path.relative_to(spec.root)) for path in artifact_paths],
        "summary": summary,
    }
    write_json(stage_completion_path(spec, stage_name), payload)
    write_text(
        stage_log_path(spec, stage_name),
        f"{_timestamp()} {stage_name} completed for {spec.study_id}\n{summary}\n",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_nonempty(path: Path) -> bool:
    return path.exists() and path.read_text(encoding="utf-8").strip() != ""
