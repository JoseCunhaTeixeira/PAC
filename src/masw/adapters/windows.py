import logging
from dataclasses import dataclass
from pathlib import Path

from sigproc.base.acquisition import Acquisition
from sigproc.base.coordinate import Coordinate

from masw.models.acquisition import AcquisitionParameters
from masw.models.masw import MASWParameters

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MASWWindow:
    xmid: float
    selected_files: list[Path]
    receiver_indices: list[int]
    acquisitions: list[Acquisition]


def build_windows(
    acquisition_params: AcquisitionParameters,
    masw_params: MASWParameters,
) -> list[MASWWindow]:

    positions = acquisition_params.receiver_positions

    windows: list[MASWWindow] = []

    for start in range(
        0,
        len(positions) - masw_params.length + 1,
        masw_params.step,
    ):
        stop = start + masw_params.length

        receiver_indices = list(range(start, stop))

        receiver_positions = positions[start:stop]

        xmin = receiver_positions[0]
        xmax = receiver_positions[-1]

        xmid = 0.5 * (xmin + xmax)

        selected_files = []
        acquisitions = []

        receiver_coords = tuple(
            Coordinate(x=x, y=0.0, z=0.0) for x in receiver_positions
        )

        for i, (file, source_x) in enumerate(
            zip(
                acquisition_params.files,
                acquisition_params.source_positions,
                strict=True,
            )
        ):
            # source must be outside the MASW window
            if xmin < source_x < xmax:
                continue

            distance = abs(source_x - xmid)

            if distance <= masw_params.distance_min:
                continue

            if distance >= masw_params.distance_max:
                continue

            selected_files.append(acquisition_params.folder_path / file)

            acquisitions.append(
                Acquisition(
                    source=Coordinate(x=source_x, y=0.0, z=0.0),
                    receivers=receiver_coords,
                )
            )

        if not selected_files:
            logger.warning(f"No valid shots for xmid={xmid:.2f}")
            continue

        windows.append(
            MASWWindow(
                xmid=xmid,
                receiver_indices=receiver_indices,
                selected_files=selected_files,
                acquisitions=acquisitions,
            )
        )

    logger.info(f"Built {len(windows)} valid MASW windows")

    return windows
