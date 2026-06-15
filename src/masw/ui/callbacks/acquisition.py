from dash import (
    Input,
    Output,
    State,
    dash_table,
    html,
)
from dash.exceptions import PreventUpdate

from masw.io.acquisition import load_acquisition
from masw.io.folders import get_input_folders


def register_acquisition_callbacks(app):

    @app.callback(
        Output("folder-dropdown", "options"),
        Input("folder-dropdown", "id"),
    )
    def update_folders(_):

        folders = get_input_folders()

        return [
            {
                "label": folder,
                "value": folder,
            }
            for folder in folders
        ]

    @app.callback(
        Output("acquisition-store", "data"),
        Output("folder-status", "children"),
        Output("folder-status", "color"),
        Output("folder-status", "is_open"),
        Output("acquisition-summary", "children"),
        Input("folder-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_folder(folder):

        if folder is None:
            raise PreventUpdate

        try:
            acquisition = load_acquisition(folder)

        except Exception as exc:
            return (
                None,
                str(exc),
                "danger",
                True,
                None,
            )

        summary_table = dash_table.DataTable(
            data=[
                {
                    "Folder": acquisition.folder_path.name,
                    "Number of Files": len(acquisition.files),
                    "Number of Receivers": acquisition.n_receivers,
                }
            ]
        )

        files_table = dash_table.DataTable(
            data=[
                {
                    "File": file,
                    "Duration [s]": duration,
                    "Source Position [m]": source,
                }
                for file, duration, source in zip(
                    acquisition.files,
                    acquisition.durations,
                    acquisition.source_positions,
                    strict=True,
                )
            ],
            page_size=10,
        )

        return (
            acquisition.model_dump(mode="json"),
            "Seismic files loaded.",
            "success",
            True,
            [
                summary_table,
                html.Br(),
                files_table,
            ],
        )

    @app.callback(
        Output(
            "geometry-summary",
            "children",
        ),
        Input("x-start", "value"),
        Input("x-step", "value"),
        State(
            "acquisition-store",
            "data",
        ),
    )
    def update_geometry(
        x_start,
        x_step,
        acquisition,
    ):

        if acquisition is None:
            raise PreventUpdate

        if x_start is None:
            raise PreventUpdate

        if x_step is None:
            raise PreventUpdate

        positions = [
            round(
                x_start + i * x_step,
                3,
            )
            for i in range(acquisition["n_receivers"])
        ]

        return dash_table.DataTable(
            data=[
                {
                    "Number of Sensors": len(positions),
                    "Sensor Positions [m]": str(positions),
                }
            ]
        )
