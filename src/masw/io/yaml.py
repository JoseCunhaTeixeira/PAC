from pathlib import Path

import yaml


def read_receiver_positions(
    file_path: Path,
) -> list[float]:

    with file_path.open(
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def write_receiver_positions(
    file_path: Path,
    positions: list[float],
) -> None:

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            positions,
            f,
            sort_keys=False,
        )


def read_source_positions(
    file_path: Path,
) -> dict[str, float]:

    with file_path.open(
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def write_source_positions(
    file_path: Path,
    source_positions: dict[str, float],
) -> None:

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            source_positions,
            f,
            sort_keys=False,
        )
