from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


class StudyConfigError(RuntimeError):
    """Raised when a study manifest is incomplete or inconsistent."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_repo_path(raw_path: str, *, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    local_candidate = (base_dir / candidate).resolve()
    if local_candidate.exists():
        return local_candidate

    return (repo_root() / candidate).resolve()


@dataclass(frozen=True)
class StudySpec:
    root: Path
    study_config_path: Path
    study: dict[str, Any]
    execution: dict[str, Any]
    config_refs: dict[str, str]
    configs: dict[str, dict[str, Any]]
    research_dir: Path
    results_dir: Path
    paper_dir: Path
    seed_list: list[int]

    @property
    def study_id(self) -> str:
        return str(self.study["id"])

    @property
    def task_family(self) -> str:
        return str(self.study["task_family"])

    @property
    def task_type(self) -> str:
        return str(self.study["task_type"])

    @property
    def artifact_status(self) -> str:
        return str(self.study.get("status", "planned"))

    @property
    def active_stages(self) -> list[str]:
        return [str(stage_name) for stage_name in self.execution.get("active_stages", [])]

    @property
    def manifests_dir(self) -> Path:
        return self.results_dir / "manifests"

    @property
    def raw_dir(self) -> Path:
        return self.results_dir / "raw"

    @property
    def aggregated_dir(self) -> Path:
        return self.results_dir / "aggregated"

    @property
    def audits_dir(self) -> Path:
        return self.results_dir / "audits"

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.results_dir / "logs"

    @property
    def paper_tables_dir(self) -> Path:
        return self.paper_dir / "tables"

    @property
    def paper_figures_dir(self) -> Path:
        return self.paper_dir / "figures"

    def doc_path(self, name: str) -> Path:
        return self.research_dir / name

    def resolved_config(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "execution": self.execution,
            "config_refs": self.config_refs,
            "paths": {
                "study_config": str(self.study_config_path),
                "research_dir": str(self.research_dir),
                "results_dir": str(self.results_dir),
                "paper_dir": str(self.paper_dir),
            },
            "configs": self.configs,
        }


def load_study_spec(study_config_path: str | Path) -> StudySpec:
    config_path = resolve_repo_path(str(study_config_path), base_dir=repo_root()).resolve()
    if not config_path.exists():
        raise StudyConfigError(f"Study config does not exist: {config_path}")

    raw_config = read_toml(config_path)

    study = _require_mapping(raw_config, "study", config_path)
    execution = _require_mapping(raw_config, "execution", config_path)
    config_refs = _require_mapping(raw_config, "config_refs", config_path)
    paths = _require_mapping(raw_config, "paths", config_path)

    required_study_fields = ("id", "task_family", "task_type")
    for field_name in required_study_fields:
        if field_name not in study:
            raise StudyConfigError(f"Missing study field '{field_name}' in {config_path}")

    configs = {
        name: read_toml(resolve_repo_path(path, base_dir=config_path.parent)) for name, path in config_refs.items()
    }

    seed_list = list(execution.get("seed_list") or configs.get("protocol", {}).get("split", {}).get("seeds", []))
    if not seed_list:
        raise StudyConfigError(f"No seeds defined in {config_path}")

    research_dir = resolve_repo_path(str(paths["research_dir"]), base_dir=config_path.parent)
    results_dir = resolve_repo_path(str(paths["results_dir"]), base_dir=config_path.parent)
    paper_dir = resolve_repo_path(str(paths["paper_dir"]), base_dir=config_path.parent)

    return StudySpec(
        root=repo_root(),
        study_config_path=config_path,
        study=study,
        execution=execution,
        config_refs={name: str(path) for name, path in config_refs.items()},
        configs=configs,
        research_dir=research_dir,
        results_dir=results_dir,
        paper_dir=paper_dir,
        seed_list=[int(seed) for seed in seed_list],
    )


def _require_mapping(raw_config: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise StudyConfigError(f"Expected [{key}] table in {path}")
    return value
