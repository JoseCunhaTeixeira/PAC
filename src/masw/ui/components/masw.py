import dash_bootstrap_components as dbc


def masw_card():

    return dbc.Card(
        [
            dbc.CardHeader("MASW Parameters"),
            dbc.CardBody(
                [
                    dbc.Label("Length"),
                    dbc.Input(id="length", type="number"),
                    dbc.Label("Step"),
                    dbc.Input(id="window-step", type="number"),
                    dbc.Label("Distance Min"),
                    dbc.Input(id="distance-min", type="number"),
                    dbc.Label("Distance Max"),
                    dbc.Input(id="distance-max", type="number"),
                ]
            ),
        ]
    )
