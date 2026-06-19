from pathlib import Path

import yaml

from masw.models.acquisition import PositionXZ


def _position_to_mapping(position: PositionXZ) -> dict[str, float]:
    x, z = position
    return {"x": x, "z": z}


def _mapping_to_position(mapping: dict[str, float]) -> PositionXZ:
    return float(mapping["x"]), float(mapping["z"])


def read_receiver_positions(
    file_path: Path,
) -> list[PositionXZ]:

    with file_path.open(
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f)

    return [_mapping_to_position(mapping) for mapping in raw]


def write_receiver_positions(
    file_path: Path,
    positions: list[PositionXZ],
) -> None:

    raw = [_position_to_mapping(position) for position in positions]

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            raw,
            f,
            sort_keys=False,
        )


def read_source_positions(
    file_path: Path,
) -> dict[str, PositionXZ]:

    with file_path.open(
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f)

    return {filename: _mapping_to_position(mapping) for filename, mapping in raw.items()}


def write_source_positions(
    file_path: Path,
    source_positions: dict[str, PositionXZ],
) -> None:

    raw = {
        filename: _position_to_mapping(position)
        for filename, position in source_positions.items()
    }

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            raw,
            f,
            sort_keys=False,
        )
