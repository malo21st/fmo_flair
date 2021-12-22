import dash 
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Scheme, Trim
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly import subplots
import pandas as pd

df = pd.read_csv('ifie_data/P2LRP.csv')

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP], # SUPERHERO BOOTSTRAP YETI 
                meta_tags=[{
                    'name': 'viewport',
                    'content': 'width=device-width, initial-scale=1.0'
                    }]
                )

def set_data(df):
    # RES
    def res_num(row):
        return row.iloc[1] + str(row.iloc[0])

    df_res = df.iloc[:,3:5]
    df_res['RES'] = df_res.apply(res_num, axis=1)
    # DIST+FMO
    df_ifie = df.iloc[:,8:14]
    # RES + (DIST + FMO)
    df_fmo = pd.concat([df_res[['RES']], df_ifie], axis=1)
    # rename columns
    df_fmo.columns = ['RES','DIST','TOTAL IFIE', 'ES', 'EX', 'CT+mix', 'DI']
    return df_fmo

df_fmo = set_data(df)

# View
title_row = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Alert(html.H3("FMOデモ"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])

headder =  [{"name": 'RES', "id": "RES"}] + [{"name": i, "id": i, 'type':'numeric', 'format':Format(precision=3, scheme=Scheme.fixed)} for i in df_fmo.columns[1:]]
# {"name": i, "id": i, "deletable": False, "selectable": False} for i in df_fmo.columns

table_row = html.Div([
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='id-table',
                columns=headder,
                data=df_fmo.to_dict('records'),
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="single",
                style_data={
                    'width': '{}%'.format(100. / len(df.columns)),
                    'textOverflow': 'hidden'
                },
                style_filter={'backgroundColor':'lightgray'},
                # filter_options={'case':'sensitive'},
                # style_filter_conditional=[
                #     {
                #         'if': {'column_id': 'RES',},
                #         'column_type': 'text',
                #     },
                #     {
                #         'if': {'column_id': 'DIST'},
                #         'column_type': 'numeric',
                #     },
                # ],
                page_action="native",
                page_current= 0,
                page_size= 15,
            ),
            width = 10
        ),
    ], justify="center")
])

set_row = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.InputGroup(
                [dbc.InputGroupAddon("DISTの閾値", addon_type="prepend"), 
                 dbc.Input(id='id-threshold', value='5'),
                 dbc.Button('グラフ', id='id-show-graph'),
                 ],
            ),
            width = 4,
        )
    ], justify="center")
])

graph_row = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='id-graph'),
            width = 12,
        ),
    ], justify="center")
])

app.layout = dbc.Container([
                            title_row, 
                            table_row, 
                            html.Br(),
                            set_row,
                            graph_row,
                            # debug_row,
                            ])

PIEDA = {
    "TOTAL IFIE":{"+":"#0000ff", "-":"#ff0000"},
    "ES":{"+":"#0000ff", "-":"#ff0000"},
    "EX":{"+":"#ff00ff", "-":"#ffffff"},
    "CT+mix":{"+":"#ffffff", "-":"#00ffff"},
    "DI":{"+":"#ffffff", "-":"#00ff00"},
    "DIST":{"+":"#6f6f6f", "-":"#ffffff"},
} 

@app.callback(
    Output('id-graph', 'figure'),
    Input('id-show-graph', 'n_clicks'),
    State('id-threshold', 'value'),
    PreventUpdate = True,
)
def componet_position(clicks, value):
    threshold = float(value)
    df_fmo_th = df_fmo[df_fmo['DIST'] <= threshold]
    df_fmo_th_sort = df_fmo_th.sort_values(by='DIST')

    items = ['TOTAL IFIE', 'DIST', 'ES', 'EX', 'CT+mix', 'DI']
    units = ['kcal/mol', 'Å'] + ['kcal/mol'] * 4
    titles = [f"{item} [{unit}]" for item, unit in zip(items, units)]

    fig = subplots.make_subplots(rows=3, cols=2, subplot_titles=titles)

    def get_trace(fig, item, row, col):
        def value_color(value, item):
            if value >= 0:
                return PIEDA[item]["+"] #f'rgba(0, 0, 255, 1.0)' # {value/max_value})'
            else:
                return PIEDA[item]["-"] #f'rgba(255, 0, 0, 1.0)' # {-value/max_value})'

        x = df_fmo_th_sort['RES']
        y = df_fmo_th_sort[item]
        max_value = max([abs(max(y)), abs(min(y))])
        my_colors = [value_color(v, item) for v in y]
        fig.append_trace(go.Bar(x=x, y=y, marker_color=my_colors), row, col)
        return fig

    for i, item in enumerate(items):
        r = i // 2 + 1
        c = i % 2 + 1
        fig = get_trace(fig, item, r, c)
    fig['layout'].update(height=900, width=1200, showlegend=False)
    return fig

app.run_server(host='0.0.0.0', port=8001, debug=True)