from pathlib import Path

from pydantic import TypeAdapter

from masw.models.computing import AnyComputingConfig

_ADAPTER: TypeAdapter[AnyComputingConfig] = TypeAdapter(AnyComputingConfig)


def load_project(project_dir: Path) -> AnyComputingConfig:
    json_text = (project_dir / "config.json").read_text(encoding="utf-8")
    config: AnyComputingConfig = _ADAPTER.validate_json(json_text)
    return config
