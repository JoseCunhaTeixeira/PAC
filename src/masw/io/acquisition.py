from obspy import read

from masw.io.paths import INPUT_DIR
from masw.models.acquisition import AcquisitionParameters

from .yaml import read_receiver_positions, read_source_positions


def load_acquisition(
    folder_name: str,
) -> AcquisitionParameters:

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
    sampling_frequencies = []
    stream = None

    for file in files:
        file_path = folder_path / file
        stream = read(
            str(file_path),
        )
        durations.append(float(stream[0].stats.endtime - stream[0].stats.starttime))
        sampling_frequencies.append(float(stream[0].stats.sampling_rate))

    if stream is None:
        raise ValueError("Unable to read seismic files")

    n_receivers = len(stream)

    # source_positions.yaml is optional: passive acquisitions have no real
    # source, so the file may simply not exist for them
    source_positions_file = folder_path / "source_positions.yaml"
    if source_positions_file.exists():
        source_positions = read_source_positions(source_positions_file)
        if list(source_positions.keys()) != files:
            raise ValueError("source_positions.yaml does not match seismic files")
        source_positions_list = list(source_positions.values())
    else:
        source_positions_list = []

    receiver_positions_file = folder_path / "receiver_positions.yaml"
    if not receiver_positions_file.exists():
        raise ValueError(f"Missing receiver_positions.yaml file in {folder_path}")
    receiver_positions = read_receiver_positions(receiver_positions_file)
    if len(receiver_positions) != n_receivers:
        raise ValueError("receiver_positions.yaml does not match seismic files")

    return AcquisitionParameters(
        folder_path=folder_path,
        files=files,
        durations=durations,
        sampling_frequencies=list(sampling_frequencies),
        source_positions=source_positions_list,
        receiver_positions=list(receiver_positions),
    )
