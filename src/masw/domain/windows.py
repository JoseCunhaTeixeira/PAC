from dataclasses import dataclass


@dataclass(slots=True)
class MASWWindow:
    id: int

    start_idx: int
    end_idx: int

    xmid: float

    @property
    def n_sensors(self) -> int:
        return self.end_idx - self.start_idx + 1


def create_windows(
    positions: list[float],
    masw_length: int,
    masw_step: int,
) -> list[MASWWindow]:

    positions = positions

    dx = positions[1] - positions[0]

    window_length = (masw_length - 1) * dx

    step = masw_step * dx

    xmid = positions[0] + window_length / 2

    windows: list[MASWWindow] = []

    window_id = 0

    while xmid + window_length / 2 <= positions[-1]:
        start_position = xmid - window_length / 2
        end_position = xmid + window_length / 2

        start_idx = positions.index(round(start_position, 3))
        end_idx = positions.index(round(end_position, 3))

        windows.append(
            MASWWindow(
                id=window_id,
                start_idx=start_idx,
                end_idx=end_idx,
                xmid=round(xmid, 3),
            )
        )

        window_id += 1
        xmid += step

    return windows
