from dash import Input, Output

from masw.ui.plots.geometry import build_geometry_figure


def register_callbacks(app):

    @app.callback(
        Output("geometry-graph", "figure"),
        Input("preview-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def preview(n_clicks):

        config = ...

        fig = build_geometry_figure(
            config.acquisition_params,
            config.masw_params,
        )

        return fig
