from obspy import read

from masw.io.paths import INPUT_DIR
from masw.models.acquisition import AcquisitionInfo

from .yaml import read_sensor_positions, read_source_positions


def load_acquisition(
    folder_name: str,
) -> AcquisitionInfo:

    folder_path = INPUT_DIR / folder_name

    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    files = sorted(
        [
            file.name
            for file in folder_path.iterdir()
            if file.is_file()
            and not file.name.startswith(".")
            and file.suffix not in [".json", ".yaml"]
        ]
    )

    if not files:
        raise ValueError(f"No seismic files found in {folder_path}")

    durations = []

    stream = None

    for file in files:
        file_path = folder_path / file

        print(f"Reading {file_path}")

        stream = read(
            str(file_path),
        )

        print(f"Successfully read {file_path}")

        durations.append(float(stream[0].stats.endtime - stream[0].stats.starttime))

    if stream is None:
        raise ValueError("Unable to read seismic files")

    n_receivers = len(stream)

    source_positions_file = folder_path / "source_positions.yaml"
    if not source_positions_file.exists():
        raise ValueError(f"Missing file: {source_positions_file}")
    source_positions = read_source_positions(source_positions_file)
    if list(source_positions.keys()) != files:
        raise ValueError("source_positions.yaml does not match seismic files")

    sensor_positions_file = folder_path / "sensor_positions.yaml"
    if not sensor_positions_file.exists():
        raise ValueError(f"Missing file: {sensor_positions_file}")
    sensor_positions = read_sensor_positions(sensor_positions_file)
    if len(sensor_positions) != n_receivers:
        raise ValueError("sensor_positions.yaml does not match seismic files")

    return AcquisitionInfo(
        folder_path=folder_path,
        files=files,
        durations=durations,
        source_positions=list(source_positions.values()),
        sensor_positions=list(sensor_positions),
        n_receivers=n_receivers,
    )
