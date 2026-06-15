import dash_bootstrap_components as dbc
from dash import html

from masw.ui.components.acquisition import acquisition_card

# layout = dbc.Container(
#     [
#         html.H2("👨‍💻 Active computing"),
#         dcc.Store(id="config-store"),
#         acquisition_card(),
#         html.Br(),
#         masw_card(),
#         html.Br(),
#         dispersion_card(),
#         html.Br(),
#         stacking_card(),
#         html.Br(),
#         execution_card(),
#         html.Br(),
#         dbc.Button(
#             "Preview Geometry",
#             id="preview-btn",
#         ),
#         dbc.Button(
#             "Run",
#             id="run-btn",
#             color="success",
#         ),
#         html.Br(),
#         html.Br(),
#         dcc.Graph(id="geometry-graph"),
#         html.Div(id="summary"),
#     ],
#     fluid=True,
# )


layout = dbc.Container(
    [
        html.H2("Active MASW"),
        acquisition_card(),
    ],
    fluid=True,
)
