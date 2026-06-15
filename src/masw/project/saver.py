from pathlib import Path

from masw.models.computing import ComputingConfig


def load_project(project_dir: Path) -> ComputingConfig:

    return ComputingConfig.model_validate_json(
        (project_dir / "config.json").read_text()
    )
