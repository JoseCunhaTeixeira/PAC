import dash_bootstrap_components as dbc


def execution_card():

    return dbc.Card(
        [
            dbc.CardHeader("Execution"),
            dbc.CardBody(
                [
                    dbc.Label("Output Folder"),
                    dbc.Input(id="output-folder"),
                    dbc.Label("Workers"),
                    dbc.Input(
                        id="n-workers",
                        type="number",
                        value=1,
                    ),
                ]
            ),
        ]
    )
