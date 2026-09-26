"""A profile as PAC's pages show it: its records (file, duration, sampling rate, source) and its
receivers, read by sigpipe's MASW layer."""

from pathlib import Path

from pydantic import BaseModel

from masw.io.paths import workspace
from sigpipe.masw.profiles import MODES, load_profile

type PositionXZ = tuple[float, float]


class Acquisition(BaseModel):
    """What the computing pages read of a profile. Source positions are empty for a passive
    profile."""

    folder_path: Path
    files: list[str]
    durations: list[float]
    sampling_frequencies: list[float]
    source_positions: list[PositionXZ]
    receiver_positions: list[PositionXZ]
    kind: str
    modes: list[str]


def load_acquisition(folder_name: str) -> Acquisition:
    """Profile `folder_name`; raises sigpipe's ProfileError (a ValueError) when it is unknown or
    malformed."""
    profile = load_profile(folder_name, workspace())
    return Acquisition(
        folder_path=profile.folder,
        files=[record.path.name for record in profile.records],
        durations=[record.duration_s for record in profile.records],
        sampling_frequencies=[record.sampling_rate_hz for record in profile.records],
        source_positions=[
            (record.source.x, record.source.z)
            for record in profile.records
            if record.source is not None
        ],
        receiver_positions=[(receiver.x, receiver.z) for receiver in profile.receivers],
        kind=profile.kind.value,
        modes=[mode.value for mode in MODES[profile.kind]],
    )
