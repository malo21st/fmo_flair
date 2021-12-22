import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app, UUID

layout = html.Div([
    html.H3('App 1'),
    dcc.Dropdown(
        id='app-1-dropdown',
        options=[
            {'label': 'App 1 - {}'.format(i), 'value': i} for i in [
                'NYC', 'MTL', 'LA'
            ]
        ]
    ),
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to App 2', href='/apps/app2'),
    html.Label(str(UUID)),
])


@app.callback(
    Output('app-1-display-value', 'children'),
    Input('app-1-dropdown', 'value'),
    prevent_initial_call = True,)
def display_value(value):
    return 'You have selected "{}"'.format(value)