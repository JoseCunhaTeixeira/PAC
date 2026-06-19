import logging
import math
from pathlib import Path

from pydantic import BaseModel
from sigproc.base.acquisition import UNKNOWN_ACQUISITION, Acquisition
from sigproc.base.coordinate import Coordinate

from masw.models.acquisition import AcquisitionParameters, PositionXZ
from masw.models.masw import MASWParameters

logger = logging.getLogger(__name__)


class MASWWindow(BaseModel):
    xmid: float
    selected_files: list[Path]
    receiver_indices: list[int]
    acquisitions: list[Acquisition]


def _arc_midpoint_x(positions: list[PositionXZ]) -> float:
    """x-coordinate of the midpoint by arc length along the (x, z) profile.

    A plain average of the first and last x would ignore z entirely, so on
    sloped or irregular topography the geometric middle of the window (the
    point that splits the ground-surface path in half) can sit at a
    different x than the midpoint of the x-only span.
    """

    cumulative = [0.0]
    for (x0, z0), (x1, z1) in zip(positions[:-1], positions[1:], strict=True):
        cumulative.append(cumulative[-1] + math.hypot(x1 - x0, z1 - z0))

    half = cumulative[-1] / 2

    for i in range(1, len(cumulative)):
        if cumulative[i] >= half:
            segment = cumulative[i] - cumulative[i - 1]
            t = (half - cumulative[i - 1]) / segment if segment > 0 else 0.0
            x0 = positions[i - 1][0]
            x1 = positions[i][0]
            return x0 + t * (x1 - x0)

    return positions[-1][0]


def build_windows(
    acquisition_params: AcquisitionParameters,
    masw_params: MASWParameters,
) -> list[MASWWindow]:

    positions = acquisition_params.receiver_positions
    has_sources = len(acquisition_params.source_positions) > 0

    windows: list[MASWWindow] = []

    for start in range(
        0,
        len(positions) - masw_params.length + 1,
        masw_params.step,
    ):
        stop = start + masw_params.length

        receiver_indices = list(range(start, stop))

        receiver_positions = positions[start:stop]

        xmin = receiver_positions[0][0]
        xmax = receiver_positions[-1][0]

        xmid = _arc_midpoint_x(receiver_positions)

        selected_files = []
        acquisitions = []

        # y is not tracked in MASW's own position data (always 0 for sigproc's
        # 3D Coordinate), so it is hardcoded here rather than read from disk
        receiver_coords = tuple(Coordinate(x=x, y=0.0, z=z) for x, z in receiver_positions)

        if has_sources:
            for file, source_position in zip(
                acquisition_params.files,
                acquisition_params.source_positions,
                strict=True,
            ):
                source_x, source_z = source_position

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
                        source=Coordinate(x=source_x, y=0.0, z=source_z),
                        receivers=receiver_coords,
                    )
                )
        else:
            # passive data has no real source: every shot is kept for every
            # window (there is no source-distance geometry to filter by),
            # using sigproc's sentinel to mark the source as unknown while
            # keeping the real receiver geometry
            for file in acquisition_params.files:
                selected_files.append(acquisition_params.folder_path / file)
                acquisitions.append(
                    Acquisition(source=UNKNOWN_ACQUISITION.source, receivers=receiver_coords)
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
