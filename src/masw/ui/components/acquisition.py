import dash_bootstrap_components as dbc
from dash import dcc, html


def acquisition_card():

    return dbc.Card(
        [
            dbc.CardHeader("Acquisition"),
            dbc.CardBody(
                [
                    dcc.Store(id="acquisition-store"),
                    dbc.Label("Data Folder"),
                    dcc.Dropdown(
                        id="folder-dropdown",
                        placeholder="Select a folder",
                    ),
                    html.Br(),
                    dbc.Alert(
                        id="folder-status",
                        is_open=False,
                    ),
                    html.Hr(),
                    html.H5("Acquisition Summary"),
                    html.Div(
                        id="acquisition-summary",
                    ),
                    html.Hr(),
                    html.H5("Sensor Geometry"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("First Sensor Position [m]"),
                                    dbc.Input(
                                        id="x-start",
                                        type="number",
                                        value=0.0,
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Sensor Spacing [m]"),
                                    dbc.Input(
                                        id="x-step",
                                        type="number",
                                        value=1.0,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Br(),
                    html.Div(
                        id="geometry-summary",
                    ),
                ]
            ),
        ]
    )
