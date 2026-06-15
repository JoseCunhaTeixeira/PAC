import dash_bootstrap_components as dbc


def dispersion_card():

    return dbc.Card(
        [
            dbc.CardHeader("Dispersion"),
            dbc.CardBody(
                [
                    dbc.Input(id="fmin", type="number"),
                    dbc.Input(id="fmax", type="number"),
                    dbc.Input(id="vmin", type="number"),
                    dbc.Input(id="vmax", type="number"),
                ]
            ),
        ]
    )
