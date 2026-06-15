from plotly import graph_objects as go

from masw.adapters.windows import build_windows
from masw.models.acquisition import AcquisitionParameters
from masw.models.masw import MASWParameters


def build_geometry_figure(
    acquisition_params: AcquisitionParameters,
    masw_params: MASWParameters,
):

    windows = build_windows(
        acquisition_params,
        masw_params,
    )

    fig = go.Figure()

    # Receivers
    fig.add_scatter(
        x=acquisition_params.positions,
        y=[0] * len(acquisition_params.positions),
        mode="markers",
        name="Receivers",
    )

    # Sources
    fig.add_scatter(
        x=acquisition_params.source_positions,
        y=[1] * len(acquisition_params.source_positions),
        mode="markers",
        name="Sources",
    )

    # MASW windows
    for window in windows:
        xmin = acquisition_params.positions[window.receiver_indices[0]]

        xmax = acquisition_params.positions[window.receiver_indices[-1]]

        fig.add_vrect(
            x0=xmin,
            x1=xmax,
            opacity=0.15,
        )

        fig.add_vline(
            x=window.xmid,
            line_dash="dash",
        )

    fig.update_layout(
        title="MASW Geometry",
        xaxis_title="Position (m)",
        showlegend=True,
    )

    return fig
