import dash_bootstrap_components as dbc


def stacking_card():

    return dbc.Card(
        [
            dbc.CardHeader("Stacking"),
            dbc.CardBody(
                [
                    dbc.Select(
                        id="stack-method",
                        options=[
                            {"label": "Linear", "value": "linear"},
                            {"label": "PWS", "value": "pws"},
                            {"label": "Root", "value": "root"},
                        ],
                    )
                ]
            ),
        ]
    )
