from pathlib import Path

from masw.models.computing import ComputingConfig


def save_project(config: ComputingConfig, project_dir: Path):

    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "config.json").write_text(config.model_dump_json(indent=4))
