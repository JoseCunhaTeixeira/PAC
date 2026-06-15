import dash_bootstrap_components as dbc
from dash import Dash

from masw.ui.callbacks.acquisition import (
    register_acquisition_callbacks,
)
from masw.ui.pages.active import layout

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = layout

register_acquisition_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
